"""Optional private FP32 raster adjoint with ordered warp-partial reduction.

The current permitted gsplat reverse equations are read in place. Only their
global accumulation destinations change; the shared installation is read-only.
"""
from pathlib import Path
import re

import torch

from .._paths import output_path


def replay_transmittance_source(source):
    """Recover T by its forward product, not by subtracting rounded alpha."""
    initial='float T_final = 1.0f - render_alphas[pix_id];\n    float T = T_final;'
    anchor='const int32_t bin_final = inside ? last_ids[pix_id] : 0;'
    if source.count(initial)!=1 or source.count(anchor)!=1:
        raise ValueError('current native transmittance/last-id boundary required')
    source=source.replace(initial,'float T_final = 1.f;\n    float T;')
    replay=r'''
    // Original accepted-event product, in the original front-to-back order.
    // Alpha=1-T is rounded in the forward output and cannot recover small T.
    if (inside) {
        for (int32_t index=range_start; index<range_end && index<=bin_final; ++index) {
            const int32_t g=flatten_ids[index];
            const vec2 xy=means2d[g];
            const vec3 conic=conics[g];
            const float dx=xy.x-px,dy=xy.y-py;
            const float sigma=0.5f*(conic.x*dx*dx+conic.z*dy*dy)+conic.y*dx*dy;
            const float alpha=min(0.999f,opacities[g]*__expf(-sigma));
            if (sigma<0.f || alpha<ALPHA_THRESHOLD) continue;
            T_final *= (1.f-alpha);
        }
    }
    T=T_final;
'''
    return source.replace(anchor,anchor+replay)


def partial_source(source,*,replay_transmittance=False):
    marker='// Explicit Instantiation:'
    if marker not in source:
        raise ValueError('current native backward launch definition required')
    source=source.split(marker)[0]
    if replay_transmittance:source=replay_transmittance_source(source)
    destination='int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]'
    if source.count(destination)!=1:
        raise ValueError('unique native warp accumulation block required')
    source=source.replace(destination,
        'int32_t g = (batch_end - int32_t(t)) * int32_t(block_size / 32) + int32_t(warp.meta_group_rank());')
    source,count=re.subn(r'gpuAtomicAdd\(([^,\n]+), ([^;\n]+)\);',r'*(\1) = \2;',source)
    if count!=15 or 'gpuAtomicAdd(' in source:
        raise ValueError('current native gradient destinations differ; no fallback')
    source=source.replace('rasterize_to_pixels_3dgs_bwd_kernel','sf_ordered_pixels_bwd_kernel')
    return source+'\n} // namespace gsplat\n'


_ARGS='''
    const at::Tensor means2d, const at::Tensor conics,
    const at::Tensor colors, const at::Tensor opacities,
    const c10::optional<at::Tensor> ray_ts,
    const c10::optional<at::Tensor> ray_planes,
    const c10::optional<at::Tensor> normals,
    const c10::optional<at::Tensor> backgrounds,
    const c10::optional<at::Tensor> masks,
    const uint32_t image_width, const uint32_t image_height,
    const uint32_t tile_size, const c10::optional<at::Tensor> Ks,
    const at::Tensor tile_offsets, const at::Tensor flatten_ids,
    const at::Tensor render_colors, const at::Tensor render_alphas,
    const at::Tensor last_ids, const c10::optional<at::Tensor> median_ids,
    const at::Tensor v_render_colors, const at::Tensor v_render_alphas,
    const c10::optional<at::Tensor> v_expected_depths,
    const c10::optional<at::Tensor> v_median_depths,
    const c10::optional<at::Tensor> v_expected_normals,
    const at::Tensor v_render_distort, bool absgrad, const bool radegs
'''
_REDUCE_ARGS='const at::Tensor partial, const at::Tensor order, const at::Tensor offsets, at::Tensor output, const int64_t warps'


def native_include_paths(root):
    root=Path(root)
    return [str(root/'cuda/include'),str(root/'cuda/csrc'),str(root/'cuda/csrc/third_party/glm')]


def extension(native_root,build_directory,*,compile=False,replay_transmittance=False):
    root=Path(native_root);build=output_path(build_directory)
    name='sf_rade_ordered_adjoint_replay' if replay_transmittance else 'sf_rade_ordered_adjoint'
    if not compile:
        import importlib.util
        library=build/(name+'.so')
        if not library.is_file():raise FileNotFoundError('precompile the isolated adjoint before scientific dispatch')
        spec=importlib.util.spec_from_file_location(name,library)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module
    import os
    from torch.utils.cpp_extension import load_inline
    build.mkdir(parents=True,exist_ok=True)
    (build/'tmp').mkdir(exist_ok=True)
    os.environ['TMPDIR']=str(build/'tmp');os.environ['MAX_JOBS']='2'
    code=partial_source((root/'cuda/csrc/RasterizeToPixels3DGSBwd.cu').read_text(),replay_transmittance=replay_transmittance)
    code+='\n#include <c10/cuda/CUDAGuard.h>\n'
    code+='std::vector<at::Tensor> ordered_partials('+_ARGS+') {\n'+r'''
        const c10::cuda::CUDAGuard device_guard(means2d.device());
        TORCH_CHECK(means2d.is_cuda() && means2d.scalar_type()==at::kFloat,
                    "CUDA float32 native inputs required");
        TORCH_CHECK(means2d.dim()==3 && means2d.size(0)==1 && colors.size(-1)==3,
                    "original unpacked single-view RGB scope required");
        TORCH_CHECK(radegs && tile_size*tile_size%32==0 && tile_size<=32,
                    "original RaDe tile/geometry scope required");
        const int64_t P=flatten_ids.numel()*(tile_size*tile_size/32);
        TORCH_CHECK(P < 2147483647LL,"warp-partial index exceeds int32 scope");
        auto opts=means2d.options();
        auto va=at::zeros({P,2},opts), vx=at::zeros({P,2},opts);
        auto vc=at::zeros({P,3},opts), vf=at::zeros({P,3},opts);
        auto vo=at::zeros({P},opts), vt=at::zeros({P},opts);
        auto vp=at::zeros({P,2},opts), vn=at::zeros({P,3},opts);
        gsplat::launch_sf_ordered_pixels_bwd_kernel<3>(
            means2d,conics,colors,opacities,ray_ts,ray_planes,normals,
            backgrounds,masks,image_width,image_height,tile_size,Ks,
            tile_offsets,flatten_ids,render_colors,render_alphas,last_ids,median_ids,
            v_render_colors,v_render_alphas,v_expected_depths,v_median_depths,v_expected_normals,
            absgrad ? c10::optional<at::Tensor>(va) : c10::nullopt,
            vx,vc,vf,vo,vt,vp,vn,v_render_distort,radegs);
        return {va,vx,vc,vf,vo,vt,vp,vn};
    }

    __global__ void sf_fixed_reduce_kernel(const float* partial,const int64_t* order,
        const int64_t* offsets,float* output,int64_t N,int64_t D,int64_t W) {
        int64_t k=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
        if(k>=N*D)return;
        int64_t g=k/D,c=k%D;
        float sum=0.f;
        for(int64_t j=offsets[g];j<offsets[g+1];++j) {
            int64_t base=order[j]*W;
            for(int64_t w=0;w<W;++w)sum+=partial[(base+w)*D+c];
        }
        output[k]=sum;
    }
    '''
    code+='void ordered_reduce('+_REDUCE_ARGS+') {\n'+r'''
        const c10::cuda::CUDAGuard device_guard(output.device());
        int64_t N=offsets.numel()-1,D=output.numel()/N;
        sf_fixed_reduce_kernel<<<(N*D+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
            partial.data_ptr<float>(),order.data_ptr<int64_t>(),offsets.data_ptr<int64_t>(),
            output.data_ptr<float>(),N,D,warps);
    }
    '''
    declaration='std::vector<at::Tensor> ordered_partials('+_ARGS+');\nvoid ordered_reduce('+_REDUCE_ARGS+');'
    declaration+='\nbool transmittance_replay_enabled(){return '+('true' if replay_transmittance else 'false')+';}'
    return load_inline(name,cpp_sources=declaration,cuda_sources=code,
        functions=['ordered_partials','ordered_reduce','transmittance_replay_enabled'],
        extra_include_paths=native_include_paths(root),
        extra_cflags=['-O3'],extra_cuda_cflags=['-O3','-use_fast_math'],
        build_directory=str(build),verbose=False)


class OrderedRasterBackward:
    def __init__(self,module,counts,*,replay_transmittance=False):
        self.module=module;self.counts=counts;self.replay_transmittance=replay_transmittance
        if replay_transmittance and (not hasattr(module,'transmittance_replay_enabled') or not module.transmittance_replay_enabled()):
            raise ValueError('explicit transmittance replay binary required; no fallback')

    def __call__(self,*args):
        if len(args)!=27:
            raise ValueError('current native backward27argument contract required')
        means,conics,colors,opacity,ray_t,plane,normal=args[:7]
        if means.dtype!=torch.float32 or means.device.type!='cuda':
            raise ValueError('ordered adjoint has no precision/device fallback')
        ids=args[14];N=means.numel()//2;W=int(args[11])**2//32
        parts=self.module.ordered_partials(*args)
        self.counts['ordered_backward']+=1
        if self.replay_transmittance:self.counts['transmittance_replay_backward']+=1
        # Integer membership/order only; no floating atomic accumulation.
        keys=ids.long();order=torch.argsort(keys,stable=True)
        sizes=torch.bincount(keys,minlength=N)
        offsets=torch.cat((sizes.new_zeros(1),sizes.cumsum(0)))
        templates=(means,means,conics,colors,opacity,ray_t,plane,normal)
        outputs=[]
        for partial,template in zip(parts,templates):
            out=torch.empty_like(template)
            self.module.ordered_reduce(partial,order,offsets,out,W)
            outputs.append(out)
        self.counts['ordered_reduce']+=1
        return tuple(outputs)
