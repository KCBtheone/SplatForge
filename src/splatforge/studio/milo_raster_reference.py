"""Independent differentiable MILo mesh rasterization reference.

Camera-space clipping, perspective depth, exact pixel/triangle coverage and
visible polygon subtraction are written from geometry. No renderer source or
noncommercial binary is imported. Topology/visibility decisions are discrete;
all retained intersections, areas, depths and face normals keep autograd.

This is deliberately the correctness reference. Pixel polygon operations are
eager and must be accelerated before full-scene training.
"""

from collections import defaultdict
import math

import torch
import torch.nn.functional as F

from .milo_raster import MeshRender, camera_vertices


def _polygon_clip(polygon, plane):
    """Intersect a 2D convex polygon with a*x+b*y+c >= 0."""
    if len(polygon) < 3:
        return polygon[:0]
    values = polygon @ plane[:2] + plane[2]
    inside = (values >= 0).tolist()
    vertices = []
    for j, present in enumerate(inside):
        previous = (j-1) % len(polygon)
        if inside[previous] != present:
            fraction = values[previous]/(values[previous]-values[j])
            vertices.append(polygon[previous] + fraction*(polygon[j]-polygon[previous]))
        if present:
            vertices.append(polygon[j])
    return torch.stack(vertices) if len(vertices) >= 3 else polygon[:0]


def _area_centroid(polygon):
    following = polygon.roll(-1, 0)
    cross = polygon[:, 0]*following[:, 1]-polygon[:, 1]*following[:, 0]
    twice_area = cross.sum()
    if not bool(twice_area > 0):
        return None
    return twice_area/2, ((polygon+following)*cross[:, None]).sum(0)/(3*twice_area)


def _subtract_occluder(polygon, planes):
    """Disjoint convex pieces outside the intersection of occluder planes."""
    exterior = []
    remainder = polygon
    for plane in planes:
        outside = _polygon_clip(remainder, -plane)
        if len(outside) >= 3 and _area_centroid(outside) is not None:
            exterior.append(outside)
        remainder = _polygon_clip(remainder, plane)
        if len(remainder) < 3 or _area_centroid(remainder) is None:
            break
    return exterior


def _clip_triangle_depth(triangle, near, far):
    polygon = list(triangle.unbind(0))
    for boundary, positive in ((near, True), (far, False)):
        if len(polygon) < 3:
            return []
        output = []
        values = [(point[2]-boundary) if positive else (boundary-point[2]) for point in polygon]
        inside = [bool(value >= 0) for value in values]
        for j, present in enumerate(inside):
            previous = (j-1) % len(polygon)
            if inside[previous] != present:
                fraction = values[previous]/(values[previous]-values[j])
                output.append(polygon[previous]+fraction*(polygon[j]-polygon[previous]))
            if present:
                output.append(polygon[j])
        polygon = output
    return [torch.stack((polygon[0], polygon[j], polygon[j+1])) for j in range(1, len(polygon)-1)]


def _projected_faces(mesh, camera, near, far):
    view = camera_vertices(mesh.vertices, camera)
    triangles = view[mesh.faces]
    normals = F.normalize(torch.cross(triangles[:, 1]-triangles[:, 0],
        triangles[:, 2]-triangles[:, 0], dim=-1), dim=-1)
    prepared = []
    for face_id, triangle in enumerate(triangles):
        z = triangle[:, 2]
        if bool((z < near).all()) or bool((z > far).all()):
            continue
        pieces = [triangle] if bool(((z >= near) & (z <= far)).all()) else _clip_triangle_depth(triangle, near, far)
        for piece in pieces:
            projection = piece @ camera.K.to(piece).T
            xy = projection[:, :2]/piece[:, 2:3]
            sides = xy.roll(-1, 0)-xy
            area = (sides[0, 0]*(-sides[2, 1])-sides[0, 1]*(-sides[2, 0]))
            if not bool(torch.isfinite(xy).all()) or not bool(area != 0):
                continue
            planes = torch.stack((-sides[:, 1], sides[:, 0],
                sides[:, 1]*xy[:, 0]-sides[:, 0]*xy[:, 1]), -1) * area.sign()
            inverse_depth = torch.linalg.solve(torch.cat((xy, xy.new_ones(3, 1)), -1), piece[:, 2].reciprocal())
            prepared.append((face_id, xy, planes, inverse_depth))
    return prepared, normals


class IndependentMeshRenderer:
    """Pixel-box depth antialiasing with continuous coverage/occlusion gradients.

    Depth integrates visible pixel areas with centroid quadrature of each
    perspective plane. Face normals/validity use the nearest center sample,
    as required by MILo Eq. (7). Coplanar shared edges have no false dark seam.
    CPU/CUDA are explicit Torch devices, with no backend or dtype fallback.
    """
    backend = 'torch-polygon-reference'

    def __init__(self, near=.01, far=10000., tile_size=16):
        if not 0 < near < far or type(tile_size) is not int or tile_size < 1:
            raise ValueError('mesh depth bounds and tile size must be positive and ordered')
        self.near, self.far, self.tile_size = near, far, tile_size
        self.counts = defaultdict(int)

    def render(self, mesh, camera):
        vertices = mesh.vertices
        if vertices.dtype != torch.float32 or vertices.device.type not in ('cpu', 'cuda'):
            raise ValueError('independent mesh rasterization requires explicit CPU/CUDA float32')
        if mesh.faces.dtype != torch.long or mesh.faces.device != vertices.device:
            raise ValueError('mesh faces must be int64 on the vertex device')
        shape = camera.height, camera.width
        zero = vertices.sum()*0
        if not len(mesh.faces):
            return MeshRender(vertices.new_zeros(*shape, 1)+zero, vertices.new_zeros(*shape, 3)+zero,
                torch.zeros(shape, dtype=torch.bool, device=vertices.device),
                torch.full(shape, -1, dtype=torch.long, device=vertices.device))
        prepared, normals = _projected_faces(mesh, camera, self.near, self.far)
        columns = math.ceil(camera.width/self.tile_size)
        bins = [[] for _ in range(columns*math.ceil(camera.height/self.tile_size))]
        for index, (_, xy, _, _) in enumerate(prepared):
            lower, upper = xy.detach().amin(0).tolist(), xy.detach().amax(0).tolist()
            left = max(0, math.floor(lower[0]/self.tile_size))
            right = min(columns-1, math.floor(upper[0]/self.tile_size))
            top = max(0, math.floor(lower[1]/self.tile_size))
            bottom = min(math.ceil(camera.height/self.tile_size)-1, math.floor(upper[1]/self.tile_size))
            for row in range(top, bottom+1):
                for column in range(left, right+1):
                    bins[row*columns+column].append(index)
        depths, normal_rows, id_rows = [], [], []
        for y in range(camera.height):
            depth_row, normal_row, id_row = [], [], []
            for x in range(camera.width):
                candidates = bins[(y//self.tile_size)*columns+x//self.tile_size]
                square = vertices.new_tensor([[x, y], [x+1, y], [x+1, y+1], [x, y+1]])
                center = vertices.new_tensor([x+.5, y+.5, 1.])
                regions = []
                center_id, center_z = -1, math.inf
                for index in candidates:
                    face_id, _, planes, inverse_depth = prepared[index]
                    polygon = square
                    for plane in planes:
                        polygon = _polygon_clip(polygon, plane)
                        if len(polygon) < 3:
                            break
                    if len(polygon) < 3 or _area_centroid(polygon) is None:
                        continue
                    regions.append((index, polygon))
                    if bool((planes @ center >= 0).all()):
                        z = (inverse_depth @ center).reciprocal()
                        if float(z.detach()) < center_z:
                            center_id, center_z = face_id, float(z.detach())
                integrated = zero
                for index, polygon in regions:
                    face_id, _, _, inverse_depth = prepared[index]
                    visible = [polygon]
                    for other, _ in regions:
                        if other == index:
                            continue
                        other_face, _, other_planes, other_depth = prepared[other]
                        depth_plane = other_depth-inverse_depth
                        if not bool((depth_plane != 0).any()):
                            if (other_face, other) > (face_id, index):
                                continue
                            planes = other_planes
                        else:
                            planes = torch.cat((other_planes, depth_plane[None]), 0)
                        visible = [piece for region in visible for piece in _subtract_occluder(region, planes)]
                        if not visible:
                            break
                    for region in visible:
                        measure = _area_centroid(region)
                        if measure is not None:
                            area, centroid = measure
                            reciprocal = inverse_depth[:2] @ centroid + inverse_depth[2]
                            integrated = integrated + area/reciprocal
                depth_row.append(integrated)
                normal_row.append(normals[center_id] if center_id >= 0 else vertices.new_zeros(3)+zero)
                id_row.append(center_id)
            depths.append(torch.stack(depth_row))
            normal_rows.append(torch.stack(normal_row))
            id_rows.append(id_row)
        ids = torch.tensor(id_rows, dtype=torch.long, device=vertices.device)
        self.counts['rasterize'] += 1
        self.counts['pixel_box_antialias'] += 1
        self.counts['projected_triangles'] += len(prepared)
        return MeshRender(torch.stack(depths)[..., None], torch.stack(normal_rows), ids >= 0, ids)
