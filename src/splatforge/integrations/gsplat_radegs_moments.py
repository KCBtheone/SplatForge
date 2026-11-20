"""Explicit native-weighted RaDe-GS L2 moment backend.

Unlike the Torch reference, this backend uses the pinned port's rasterization
support, alpha threshold and early termination for ALL outputs. L2 uses the
same per-pixel native ray-depth planes and detached native weights. L1 is not
provided: the upstream center-depth L1 helper is a different quantity.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..studio.renderers import RenderOutput


def proposed_depth_xy_bridge(ray_t,xy,plane):
    """Add the direct depth derivative with respect to the projected center.

    Forward is unchanged. Native ray_t cotangents additionally reach xy with
    the direct plane-position derivative. Plane retains its native path only.
    """
    if ray_t.dtype!=torch.float32 or xy.dtype!=torch.float32 or plane.dtype!=torch.float32:
        raise ValueError('proposed depth bridge requires explicit float32 inputs')
    if xy.shape!=plane.shape or xy.shape[:-1]!=ray_t.shape or xy.shape[-1]!=2:
        raise ValueError('proposed depth bridge input shapes differ')
    return ray_t+((xy-xy.detach())*plane.detach()).sum(-1)


def homogeneous_quaternion_matrix32(quats):
    """Original clamped normalized-quaternion rotation, no sqrt/square roundtrip."""
    if quats.dtype!=torch.float32:raise ValueError('homogeneous quaternion requires float32')
    w,x,y,z=quats.unbind(-1);den=quats.square().sum(-1).clamp_min(1e-24)
    return (torch.stack((den-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
        2*(x*y+w*z),den-2*(x*x+z*z),2*(y*z-w*x),
        2*(x*z-w*y),2*(y*z+w*x),den-2*(x*x+y*y)),dim=-1)/den[:,None]).reshape(-1,3,3)


def exact_basis_ray_geometry(means,quats,scales,view,K,*,width,height,homogeneous_quaternion=False):
    """Explicit float32 basis solve of the original regularized geometry equations."""
    if means.dtype != torch.float32:raise ValueError('exact-basis actual geometry requires float32')
    if homogeneous_quaternion:rot=homogeneous_quaternion_matrix32(quats)
    else:
        w,x,y,z=F.normalize(quats,dim=-1).unbind(-1)
        rot=torch.stack((1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),
            2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
            2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)),dim=-1).reshape(-1,3,3)
    R=view[:3,:3];U=R@rot;A=R@(rot*scales[:,None,:]);gram=A@A.transpose(-1,-2)
    lam=(1e-6*gram.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-8)).clamp_min(1e-8)
    m=means@R.T+view[:3,3];fx,fy,cx,cy=K[0,0],K[1,1],K[0,2],K[1,2]
    u=(m[:,0]/m[:,2]).clamp(-(cx+.15*width)/fx,(width-cx+.15*width)/fx)
    v=(m[:,1]/m[:,2]).clamp(-(cy+.15*height)/fy,(height-cy+.15*height)/fy)
    h=torch.stack((u,v,torch.ones_like(u)),dim=-1)
    eye=torch.eye(3,dtype=U.dtype,device=U.device).expand(len(U),3,3)
    Ui=torch.linalg.solve(U,eye)
    P=torch.diag_embed(scales.square())+lam[:,None,None]*(Ui@Ui.transpose(-1,-2))
    rhs=torch.linalg.solve(U,h[...,None]);t=torch.linalg.solve(P,rhs)
    q=torch.linalg.solve(U.transpose(-1,-2),t)[...,0];n=F.normalize(q,dim=-1)
    den=(n*h).sum(-1).clamp_min(1e-7)
    nj=torch.stack((1+v*v,-u*v,-u,-u*v,1+u*u,-v),dim=-1).reshape(-1,2,3)
    ray_t=torch.linalg.vector_norm(m[:,2,None]*h,dim=-1);factor=ray_t/(1+u*u+v*v)
    plane=(nj@(n/den[:,None])[...,None])[...,0]*factor[:,None]/torch.stack((fx,fy))
    return torch.cat((ray_t[:,None],plane,-n),dim=-1)


def centered_l2_indexed32(xy,conic,opacity,ray_t,plane,gids,pids,pixels,K,width):
    """Connected float32 ordered event rows; native selection is supplied by caller."""
    if any(v.dtype!=torch.float32 for v in (xy,conic,opacity,ray_t,plane,K)):
        raise ValueError('centered indexed L2 requires float32')
    groups=torch.searchsorted(pixels,pids)
    counts=torch.bincount(groups,minlength=len(pixels))
    columns=max(int(counts.max()),1)
    starts=counts.cumsum(0)-counts
    ordinal=torch.arange(len(gids),device=gids.device)-torch.repeat_interleave(starts,counts)
    slots=groups*columns+ordinal
    px=(pids%width).to(xy.dtype)+.5;py=(pids//width).to(xy.dtype)+.5
    delta=xy[gids]-torch.stack((px,py),-1)
    depth=ray_t[gids]+(delta*plane[gids]).sum(-1)
    dx,dy=delta.detach().unbind(-1);a,b,c=conic[gids].detach().unbind(-1)
    alpha=(opacity[gids].detach()*torch.exp(-(.5*(a*dx*dx+c*dy*dy)+b*dx*dy))).clamp_max(.999)
    shape=(len(pixels),columns)
    depths=depth.new_zeros(len(pixels)*columns).index_copy(0,slots,depth).reshape(shape)
    alphas=alpha.new_zeros(len(pixels)*columns).index_copy(0,slots,alpha).reshape(shape)
    trans=torch.cat((torch.ones_like(alphas[:,:1]),(1-alphas).cumprod(-1)[:,:-1]),-1)
    weight=(alphas*trans).detach();offset=depths-depths[:,:1].detach()
    total=weight.sum(-1);first=(weight*offset).sum(-1);second=(weight*offset.square()).sum(-1)
    x=(pixels%width).to(xy.dtype)+.5;y=(pixels//width).to(xy.dtype)+.5
    norm2=((x-K[0,2])/K[0,0]).square()+((y-K[1,2])/K[1,1]).square()+1
    return (2*(total*second-first.square())/norm2).clamp_min(0),weight,counts


class GsplatRadeMomentRenderer:
    provides_rade_normal_numerator = True

    def __init__(self, config, *, ops, stable_geometry=False, cuda_geometry=False, exact_basis_geometry=False, centered_l2=False, depth_xy_bridge=False, homogeneous_quaternion=False, depth_readout='expected', ordered_adjoint_build_directory='', replay_transmittance=False):
        if depth_readout not in ('expected','median'):
            raise ValueError('RaDe depth readout must be expected or median')
        if depth_readout=='median' and not centered_l2:
            raise ValueError('Median method depth requires the explicit centered L2 entry')
        if exact_basis_geometry and (stable_geometry or cuda_geometry):
            raise ValueError('exact-basis geometry cannot mix with stable/CUDA geometry modes')
        if centered_l2 and not exact_basis_geometry:
            raise ValueError('centered L2 requires explicit owned exact-basis entry')
        from gsplat import rasterization
        from gsplat.cuda._wrapper import (
            fully_fused_projection,
            isect_offset_encode,
            isect_tiles,
            rasterize_to_pixels,
        )

        if "radegs" not in inspect.signature(fully_fused_projection).parameters:
            raise RuntimeError("the pinned gsplat RaDe-GS projection API is required")
        self.project = fully_fused_projection
        self.intersections = isect_tiles
        self.offsets = isect_offset_encode
        self.pixels = rasterize_to_pixels
        self.rasterization = rasterization
        self.config, self.ops = config, ops
        self.stable_geometry = stable_geometry
        self.cuda_geometry = cuda_geometry
        self.exact_basis_geometry = exact_basis_geometry
        self.centered_l2 = centered_l2
        self.depth_xy_bridge = depth_xy_bridge
        self.depth_readout=depth_readout;self.median_depth_readout_calls=0
        if homogeneous_quaternion and not exact_basis_geometry:raise ValueError('homogeneous quaternion requires owned exact-basis geometry')
        self.homogeneous_quaternion=homogeneous_quaternion;self.homogeneous_geometry_calls=0
        if cuda_geometry and not stable_geometry:
            raise ValueError("fused CUDA geometry requires the stable contract")
        self.ordered_adjoint_build_directory = ordered_adjoint_build_directory
        self.replay_transmittance = replay_transmittance
        self.ordered_adjoint_module = None
        self.ordered_adjoint_counts = None

    def bind_ordered_adjoint(self, extension_loader=None):
        """Explicitly load and install the ordered raster adjoint.

        Default behavior is unchanged when this is never called. The caller
        must provide a build directory and (when replay is requested) the
        prebuilt replay binary. No silent fallback is introduced.
        """
        from . import radegs_ordered_adjoint
        if not self.ordered_adjoint_build_directory:
            raise ValueError("ordered_adjoint_build_directory is required to bind the ordered adjoint")
        if self.ordered_adjoint_module is not None:
            raise ValueError("ordered adjoint is already bound")
        if extension_loader is None:
            import gsplat
            extension_loader = lambda: radegs_ordered_adjoint.extension(
                Path(gsplat.__file__).parent,
                Path(self.ordered_adjoint_build_directory),
                replay_transmittance=self.replay_transmittance,
            )
        module = extension_loader()

        from gsplat.cuda import _wrapper
        # Preserve the factory that was in place before we patch it so we can restore it later.
        original_factory = _wrapper._make_lazy_cuda_func

        # Create the private ordered backward callable.
        self.ordered_adjoint_counts = dict(ordered_backward=0, ordered_reduce=0)
        if self.replay_transmittance:
            self.ordered_adjoint_counts["transmittance_replay_backward"] = 0
        private = radegs_ordered_adjoint.OrderedRasterBackward(
            module, self.ordered_adjoint_counts, replay_transmittance=self.replay_transmittance
        )

        # The selected factory must distinguish the native ``radegs`` flag (the final argument).
        # When the flag is ``True`` we forward to the ordered backward; otherwise we delegate
        # to the original implementation. This mirrors the logic in ``radegs_native_trainer_check``.
        def selected(name):
            """Return a callable for the requested cuda function.

            For ``rasterize_to_pixels_3dgs_bwd`` we return a small wrapper that inspects the
            ``radegs`` flag (the last positional argument).  All other function names are
            passed straight through to the original factory.
            """
            if name != "rasterize_to_pixels_3dgs_bwd":
                return original_factory(name)

            native_callable = original_factory(name)

            def wrapper(*args, **kwargs):
                # The native contract expects exactly 27 positional arguments.
                if len(args) != 27:
                    raise ValueError("Original native backward27 contract required")
                # ``radegs`` flag is the last argument.
                if args[-1]:
                    return private(*args, **kwargs)
                # Preserve native behaviour – forward to the original callable.
                return native_callable(*args, **kwargs)

            return wrapper

        # Patch the global factory.
        _wrapper._make_lazy_cuda_func = selected
        # Store state for later unbinding.
        self.ordered_adjoint_module = module
        self.ordered_adjoint_original_factory = original_factory
        return self.ordered_adjoint_counts

    def unbind_ordered_adjoint(self):
        """Restore the original _make_lazy_cuda_func factory, resetting per-render scope.

        Must be called after bind_ordered_adjoint to restore global state.
        No-op if the adjoint was never bound.
        """
        if self.ordered_adjoint_module is None:
            return
        from gsplat.cuda import _wrapper
        original_factory = getattr(self, "ordered_adjoint_original_factory", None)
        if original_factory is not None:
            _wrapper._make_lazy_cuda_func = original_factory
            self.ordered_adjoint_original_factory = None
        self.ordered_adjoint_module = None
        self.ordered_adjoint_counts = None

    def geometry(self, scene, camera):
        result = self.project(
            scene.means,
            None,
            scene.quats,
            scene.scales,
            camera.world_to_camera[None],
            camera.K[None],
            camera.width,
            camera.height,
            eps2d=self.config.eps2d,
            near_plane=self.config.near,
            far_plane=self.config.far,
            packed=False,
            calc_compensations=self.config.antialiased,
            opacities=scene.opacities,
            radegs=not self.stable_geometry,
        )
        if len(result) != (5 if self.stable_geometry else 8):
            raise RuntimeError("unexpected pinned RaDe-GS projection tuple")
        if getattr(self,'exact_basis_geometry',False):
            radii=result[0];indices=torch.where((radii[0]>0).all(-1))[0]
            value=exact_basis_ray_geometry(scene.means[indices],scene.quats[indices],scene.scales[indices],
                camera.world_to_camera,camera.K,width=camera.width,height=camera.height,homogeneous_quaternion=getattr(self,'homogeneous_quaternion',False))
            if getattr(self,'homogeneous_quaternion',False):self.homogeneous_geometry_calls+=1
            dense=value.new_zeros((len(scene.means),6)).index_copy(0,indices,value)
            return (*result[:5],dense[:,0][None].contiguous(),dense[:,1:3][None].contiguous(),dense[:,3:6][None].contiguous())
        if self.stable_geometry:
            radii, xy, z, conics, compensation, *_ = result
            if getattr(self, "cuda_geometry", False):
                from ..providers.torch.gaussian.ray_geometry_cuda import (
                    regularized_ray_geometry_cuda,
                )

                geometry = regularized_ray_geometry_cuda(
                    scene.means,
                    scene.quats,
                    scene.scales,
                    camera.world_to_camera,
                    camera.K,
                    visible=(radii[0] > 0).all(-1),
                    width=camera.width,
                    height=camera.height,
                )
                return radii, xy, z, conics, compensation, *(x[None] for x in geometry)
            indices = (radii[0] > 0).all(-1).nonzero().flatten()
            rows = []

            def geometry_chunk(means, quats, scales, transform, intrinsics):
                return self.ops(
                    "gaussian.regularized_ray_geometry@1",
                    means,
                    quats,
                    scales,
                    transform,
                    intrinsics,
                    width=camera.width,
                    height=camera.height,
                )

            for batch in indices.split(16384):
                operands = (
                    scene.means[batch],
                    scene.quats[batch],
                    scene.scales[batch],
                    camera.world_to_camera,
                    camera.K,
                )
                rows.append(
                    checkpoint(
                        geometry_chunk,
                        *operands,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                    if torch.is_grad_enabled()
                    else geometry_chunk(*operands)
                )
            values = []
            for column, shape in enumerate(
                ((len(scene.means),), (len(scene.means), 2), (len(scene.means), 3))
            ):
                base = scene.means.new_zeros(shape)
                if rows:
                    base = base.index_copy(
                        0, indices, torch.cat([row[column] for row in rows])
                    )
                values.append(base[None])
            return radii, xy, z, conics, compensation, *values
        return result

    def centered_l2_tile(self,scene,camera,*,tile_xy):
        """Explicit bounded owned entry; does not replace render/default losses."""
        if not getattr(self,'exact_basis_geometry',False):raise ValueError('explicit owned exact-basis entry required')
        if scene.means.device.type!='cuda' or scene.means.dtype!=torch.float32:
            raise ValueError('centered tile requires CUDA float32')
        from gsplat.cuda._wrapper import rasterize_to_indices_in_range
        radii,xy,z,conic,comp,ray_t,plane,normal=self.geometry(scene,camera)
        tile=self.config.tile_size;tx,ty=tile_xy
        if tile!=16 or tx<0 or ty<0 or (tx+1)*tile>camera.width or (ty+1)*tile>camera.height:
            raise ValueError('one complete original16x16tile required')
        x=torch.arange(tx*tile,(tx+1)*tile,device=xy.device)
        y=torch.arange(ty*tile,(ty+1)*tile,device=xy.device)
        pixels=(y[:,None]*camera.width+x[None,:]).reshape(-1)
        _,intersections,flatten=self.intersections(xy,radii,z,tile,math.ceil(camera.width/tile),math.ceil(camera.height/tile),packed=False)
        offsets=self.offsets(intersections,1,math.ceil(camera.width/tile),math.ceil(camera.height/tile))
        opacity=scene.opacities[None]
        if comp is not None:opacity=opacity*comp
        initial=xy.new_zeros((1,camera.height,camera.width));initial.reshape(-1)[pixels]=1
        gids,pids,cids=rasterize_to_indices_in_range(0,2147483647,initial,xy.detach(),conic.detach(),opacity.detach(),camera.width,camera.height,tile,offsets,flatten)
        value,weights,counts=centered_l2_indexed32(xy[0],conic[0],opacity[0],ray_t[0],plane[0],gids,pids,pixels,camera.K,camera.width)
        return value,dict(gids=gids,pids=pids,pixels=pixels,weights=weights,counts=counts,
            replay_args=(xy,conic,None,opacity,ray_t,plane,normal,camera.width,camera.height,tile,offsets,flatten))

    def centered_l2_image(self,xy,conic,opacity,ray_t,plane,camera,offsets,flatten):
        """Full image, bounded padded rows; backward recomputes each chunk."""
        from gsplat.cuda._wrapper import rasterize_to_indices_in_range
        initial=xy.new_ones((1,camera.height,camera.width))
        gids,pids,_=rasterize_to_indices_in_range(0,2147483647,initial,xy.detach(),conic.detach(),opacity.detach(),
            camera.width,camera.height,self.config.tile_size,offsets,flatten)
        values=[];size=camera.width*camera.height
        def reduction(*args):return centered_l2_indexed32(*args)[0]
        for begin in range(0,size,4096):
            end=min(size,begin+4096);keep=(pids>=begin)&(pids<end)
            pixels=torch.arange(begin,end,device=xy.device)
            args=(xy[0],conic[0],opacity[0],ray_t[0],plane[0],gids[keep],pids[keep],pixels,camera.K,camera.width)
            value=checkpoint(reduction,*args,use_reentrant=False,preserve_rng_state=False) if torch.is_grad_enabled() and ray_t.requires_grad else reduction(*args)
            values.append(value)
        self.centered_event_count=len(gids);self.centered_chunk_count=len(values)
        return torch.cat(values).reshape(camera.height,camera.width,1)

    def render(self, scene, camera, *, selected=None):
        if selected is not None:
            raise ValueError(
                "native moment RaDe-GS does not implement an ambiguity selection mask"
            )
        if scene.means.device.type != "cuda" or scene.means.dtype != torch.float32:
            raise ValueError("native moment RaDe-GS requires explicit CUDA float32")
        radii, xy, z, conics, compensation, ray_t, planes, normals = self.geometry(
            scene, camera
        )
        if xy.requires_grad:
            xy.retain_grad()
        opacity = scene.opacities[None]
        if compensation is not None:
            opacity = opacity * compensation
        colors = self.ops(
            "gaussian.sh_color@1",
            scene.sh,
            scene.means - camera.origin,
            degree=scene.sh_degree,
        )[None]
        tile = self.config.tile_size
        tw, th = math.ceil(camera.width / tile), math.ceil(camera.height / tile)
        _, intersections, ids = self.intersections(
            xy, radii, z, tile, tw, th, packed=False
        )
        offsets = self.offsets(intersections, 1, tw, th)
        rgb, alpha, znum, median, nnum, _ = self.pixels(
            xy,
            conics,
            colors,
            opacity,
            proposed_depth_xy_bridge(ray_t,xy,planes) if self.depth_xy_bridge else ray_t,
            planes,
            normals,
            camera.width,
            camera.height,
            tile,
            offsets,
            ids,
            backgrounds=colors.new_tensor([self.config.background]),
            Ks=camera.K[None],
            packed=False,
            absgrad=True,
            radegs=True,
            distloss=False,
        )
        if getattr(self,'centered_l2',False):
            variance=self.centered_l2_image(xy,conics,opacity,ray_t,planes,camera,offsets,ids)
            if self.depth_readout=='median':
                depth=median[0];self.median_depth_readout_calls+=1
            else:depth=znum[0]/alpha[0].clamp_min(1e-8)
            return RenderOutput(rgb[0],alpha[0],depth,F.normalize(nnum[0],dim=-1),None,
                variance,median[0],torch.zeros_like(alpha[0]),torch.zeros_like(alpha[0]),xy,radii[0].amax(-1),
                normal_numerator=nnum[0],distortion_profile='l2')
        valid = (radii > 0).all(-1)
        safe_xy = torch.where(valid[..., None], xy, 0)
        safe_t = torch.where(valid, ray_t, 0)
        safe_planes = torch.where(valid[..., None], planes, 0)
        center = (
            safe_t[valid].detach().median() if valid.any() else safe_t.new_zeros(())
        )
        a = safe_t - center + ((safe_xy - camera.K[:2, 2]) * safe_planes).sum(-1)
        bx, by = (-safe_planes).unbind(-1)
        features = torch.stack(
            (
                a,
                bx,
                by,
                a.square(),
                2 * a * bx,
                2 * a * by,
                bx.square(),
                2 * bx * by,
                by.square(),
            ),
            -1,
        )
        features = torch.where(valid[..., None], features, 0)
        moments, frozen_alpha, *_ = self.pixels(
            xy.detach(),
            conics.detach(),
            features,
            opacity.detach(),
            None,
            None,
            None,
            camera.width,
            camera.height,
            tile,
            offsets,
            ids,
            packed=False,
            absgrad=False,
            radegs=False,
            distloss=False,
        )
        rays = self.ops(
            "camera.pixel_rays@1", camera.K, width=camera.width, height=camera.height
        )
        pixels = (rays @ camera.K.T)[..., :2] - camera.K[:2, 2]
        x, y = pixels.unbind(-1)
        m = moments[0]
        first = m[..., 0] + m[..., 1] * x + m[..., 2] * y
        second = (
            m[..., 3]
            + m[..., 4] * x
            + m[..., 5] * y
            + m[..., 6] * x.square()
            + m[..., 7] * x * y
            + m[..., 8] * y.square()
        )
        variance = (
            2
            * (frozen_alpha[0, ..., 0].detach() * second - first.square())
            / rays.square().sum(-1)
        ).clamp_min(0)[..., None]
        return RenderOutput(
            rgb[0],
            alpha[0],
            znum[0] / alpha[0].clamp_min(1e-8),
            F.normalize(nnum[0], dim=-1),
            None,
            variance,
            median[0],
            torch.zeros_like(alpha[0]),
            torch.zeros_like(alpha[0]),
            xy,
            radii[0].amax(-1),
            normal_numerator=nnum[0],
            distortion_profile="l2",
        )

    def render_rgb(self, scene, camera, *, selected=None):
        """Same native RGB support/VJP, without unused geometry or L2 passes."""
        if selected is not None:
            raise ValueError("native moment RGB path does not implement selection")
        if scene.means.device.type != "cuda" or scene.means.dtype != torch.float32:
            raise ValueError("native moment RGB path requires explicit CUDA float32")
        colors = self.ops(
            "gaussian.sh_color@1",
            scene.sh,
            scene.means - camera.origin,
            degree=scene.sh_degree,
        )
        rgb, alpha, meta = self.rasterization(
            scene.means,
            scene.quats,
            scene.scales,
            scene.opacities,
            colors,
            camera.world_to_camera[None],
            camera.K[None],
            camera.width,
            camera.height,
            packed=False,
            render_mode="RGB",
            absgrad=True,
            RadeGS=False,
            distloss=False,
            eps2d=self.config.eps2d,
            near_plane=self.config.near,
            far_plane=self.config.far,
            tile_size=self.config.tile_size,
            backgrounds=colors.new_tensor([self.config.background]),
            rasterize_mode="antialiased" if self.config.antialiased else "classic",
        )
        xy = meta["means2d"]
        if xy.requires_grad:
            xy.retain_grad()
        zero = torch.zeros_like(alpha[0])
        return RenderOutput(
            rgb[0],
            alpha[0],
            zero,
            zero.expand(-1, -1, 3),
            None,
            zero,
            zero,
            zero,
            zero,
            xy,
            meta["radii"][0].amax(-1),
            distortion_profile="none",
        )
