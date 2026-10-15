"""Small PLY I/O with no third-party reconstruction code."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_mesh(path, vertices, faces):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.asarray(vertices, dtype="<f4")
    faces = np.asarray(faces, dtype="<i4")
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("mesh must contain finite Nx3 vertices and Mx3 faces")
    if len(faces) and (faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("mesh indices are out of bounds")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\n"
        + f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(vertices.tobytes())
        records = np.empty(
            len(faces), dtype=[("count", "u1"), ("indices", "<i4", (3,))]
        )
        records["count"] = 3
        records["indices"] = faces
        handle.write(records.tobytes())


def write_gaussians(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = model.means.detach().cpu().numpy()
    dc = model.sh[:, 0].detach().cpu().numpy()
    rest = (
        model.sh[:, 1:]
        .detach()
        .cpu()
        .numpy()
        .transpose(0, 2, 1)
        .reshape(model.count, -1)
    )
    opacity = model.params["opacity_logits"].detach().cpu().numpy()[:, None]
    scale = model.params["log_scales"].detach().cpu().numpy()
    quats = model.quats.detach().cpu().numpy()
    names = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + [f"f_rest_{i}" for i in range(rest.shape[1])]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    values = np.concatenate(
        (positions, np.zeros_like(positions), dc, rest, opacity, scale, quats), 1
    ).astype("<f4")
    if not np.isfinite(values).all():
        raise ValueError("cannot export non-finite Gaussians")
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        + f"element vertex {model.count}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(values.tobytes())


def read_vertices(path):
    path = Path(path)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return np.asarray(data["points"], np.float64)
    types = {
        "char": "i1",
        "uchar": "u1",
        "int8": "i1",
        "uint8": "u1",
        "short": "<i2",
        "ushort": "<u2",
        "int": "<i4",
        "uint": "<u4",
        "float": "<f4",
        "double": "<f8",
        "float32": "<f4",
        "float64": "<f8",
    }
    with path.open("rb") as handle:
        if handle.readline().strip() != b"ply":
            raise ValueError("not a PLY file")
        encoding = None
        count = None
        properties = []
        element = None
        for _ in range(10000):
            line = handle.readline()
            if not line:
                raise ValueError("truncated PLY header")
            text = line.decode("ascii").strip()
            tokens = text.split()
            if text == "end_header":
                break
            if not tokens:
                continue
            if tokens[0] == "format":
                encoding = tokens[1]
            if tokens[0] == "element":
                element = tokens[1]
                if element == "vertex":
                    count = int(tokens[2])
            if tokens[0] == "property" and element == "vertex":
                if tokens[1] == "list":
                    raise ValueError("vertex list properties are unsupported")
                properties.append((tokens[2], types[tokens[1]]))
        else:
            raise ValueError("PLY header is too long")
        if count is None or any(
            axis not in [p[0] for p in properties] for axis in ("x", "y", "z")
        ):
            raise ValueError("PLY needs xyz vertex properties")
        if encoding == "binary_little_endian":
            array = np.fromfile(handle, dtype=np.dtype(properties), count=count)
            if len(array) != count:
                raise ValueError("truncated PLY vertices")
            return np.column_stack([array[axis] for axis in ("x", "y", "z")]).astype(
                np.float64
            )
        if encoding == "ascii":
            rows = [list(map(float, handle.readline().split())) for _ in range(count)]
            names = [p[0] for p in properties]
            return np.asarray(rows, np.float64)[
                :, [names.index(axis) for axis in ("x", "y", "z")]
            ]
        raise ValueError(f"unsupported PLY encoding: {encoding}")


def read_studio_mesh(path):
    """Read the binary triangle layout written by write_mesh for the viewer."""
    with Path(path).open("rb") as handle:
        vertices = None
        faces = None
        properties = []
        element = None
        encoding = None
        for _ in range(10000):
            raw = handle.readline()
            if not raw:
                raise ValueError("truncated mesh header")
            line = raw.decode("ascii").strip()
            tokens = line.split()
            if line == "end_header":
                break
            if not tokens:
                continue
            if tokens[0] == "format":
                encoding = tokens[1]
            if tokens[0] == "element":
                element = tokens[1]
                if element == "vertex":
                    vertices = int(tokens[2])
                if element == "face":
                    faces = int(tokens[2])
            if tokens[0] == "property" and element == "vertex":
                properties.append(tokens[1:])
        if (
            encoding != "binary_little_endian"
            or properties != [["float", "x"], ["float", "y"], ["float", "z"]]
            or vertices is None
            or faces is None
        ):
            raise ValueError("viewer mesh must use the Studio triangle PLY layout")
        xyz = np.fromfile(handle, dtype="<f4", count=vertices * 3).reshape(vertices, 3)
        triangles = np.fromfile(
            handle, dtype=[("count", "u1"), ("indices", "<i4", (3,))], count=faces
        )
        if len(triangles) != faces or not (triangles["count"] == 3).all():
            raise ValueError("invalid triangle records")
        return xyz, triangles["indices"]
