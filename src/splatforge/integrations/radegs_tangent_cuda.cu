#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "radegs_tangent_math.h"
#ifdef SF_GEOMETRY64
using GeometryScalar=double;
#else
using GeometryScalar=float;
#endif

__global__ void fields_f(const float*m,const float*q,const float*s,const float*v,const float*k,
        int64_t n,int width,int height,float*out) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    RadeValue<0,GeometryScalar> y[6];rade_fields_math<0,GeometryScalar>(m+3*i,q+4*i,s+3*i,v,k,width,height,y);
    for(int j=0;j<6;++j)out[6*i+j]=y[j].v;
}
__global__ void fields_b(const float*m,const float*q,const float*s,const float*v,const float*k,
        const float*g,int64_t n,int width,int height,float*gm,float*gq,float*gs) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;
    RadeValue<10,GeometryScalar> y[6];rade_fields_math<10,GeometryScalar>(m+3*i,q+4*i,s+3*i,v,k,width,height,y);
    GeometryScalar result[10]={};for(int d=0;d<10;++d)for(int j=0;j<6;++j)result[d]=rade_fma_scalar(GeometryScalar(g[6*i+j]),y[j].d[d],result[d]);
    for(int d=0;d<3;++d){gm[3*i+d]=result[d];gs[3*i+d]=result[d+7];}
    for(int d=0;d<4;++d)gq[4*i+d]=result[d+3];
}
static void check(at::Tensor m,at::Tensor q,at::Tensor s,at::Tensor v,at::Tensor k,bool cuda) {
    TORCH_CHECK(m.dim()==2&&m.size(1)==3&&q.sizes()==at::IntArrayRef({m.size(0),4})&&s.sizes()==m.sizes(),"parameter shapes");
    TORCH_CHECK(v.numel()==16&&k.numel()==9,"camera shapes");
    for(auto t:{m,q,s,v,k})TORCH_CHECK(t.scalar_type()==at::kFloat&&t.is_contiguous()&&t.device()==m.device()&&t.is_cuda()==cuda,"explicit contiguous FP32 device inputs required");
}
at::Tensor forward(at::Tensor m,at::Tensor q,at::Tensor s,at::Tensor v,at::Tensor k,int64_t w,int64_t h) {
    check(m,q,s,v,k,true);c10::cuda::CUDAGuard guard(m.device());auto out=at::empty({m.size(0),6},m.options());
    if(m.size(0))fields_f<<<(m.size(0)+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(m.data_ptr<float>(),q.data_ptr<float>(),s.data_ptr<float>(),v.data_ptr<float>(),k.data_ptr<float>(),m.size(0),w,h,out.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
std::vector<at::Tensor> backward(at::Tensor m,at::Tensor q,at::Tensor s,at::Tensor v,at::Tensor k,at::Tensor g,int64_t w,int64_t h) {
    check(m,q,s,v,k,true);TORCH_CHECK(g.is_cuda()&&g.scalar_type()==at::kFloat&&g.is_contiguous()&&g.device()==m.device()&&g.sizes()==at::IntArrayRef({m.size(0),6}),"cotangent shape/device");
    c10::cuda::CUDAGuard guard(m.device());auto gm=at::empty_like(m),gq=at::empty_like(q),gs=at::empty_like(s);
    if(m.size(0))fields_b<<<(m.size(0)+63)/64,64,0,at::cuda::getCurrentCUDAStream()>>>(m.data_ptr<float>(),q.data_ptr<float>(),s.data_ptr<float>(),v.data_ptr<float>(),k.data_ptr<float>(),g.data_ptr<float>(),m.size(0),w,h,gm.data_ptr<float>(),gq.data_ptr<float>(),gs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {gm,gq,gs};
}
// Explicit host diagnostic of the same scalar implementation; never fallback.
std::vector<at::Tensor> cpu_diagnostic(at::Tensor m,at::Tensor q,at::Tensor s,at::Tensor v,at::Tensor k,at::Tensor g,int64_t w,int64_t h) {
    check(m,q,s,v,k,false);TORCH_CHECK(!g.is_cuda()&&g.scalar_type()==at::kFloat&&g.is_contiguous()&&g.numel()==m.size(0)*6,"CPU diagnostic cotangent");
    auto out=at::empty({m.size(0),6},m.options()),gm=at::empty_like(m),gq=at::empty_like(q),gs=at::empty_like(s);
    for(int64_t i=0;i<m.size(0);++i) {
        RadeValue<10,GeometryScalar> y[6];rade_fields_math<10,GeometryScalar>(m.data_ptr<float>()+3*i,q.data_ptr<float>()+4*i,s.data_ptr<float>()+3*i,v.data_ptr<float>(),k.data_ptr<float>(),w,h,y);
        GeometryScalar result[10]={};for(int j=0;j<6;++j){out.data_ptr<float>()[6*i+j]=y[j].v;for(int d=0;d<10;++d)result[d]=rade_fma_scalar(GeometryScalar(g.data_ptr<float>()[6*i+j]),y[j].d[d],result[d]);}
        for(int d=0;d<3;++d){gm.data_ptr<float>()[3*i+d]=result[d];gs.data_ptr<float>()[3*i+d]=result[d+7];}
        for(int d=0;d<4;++d)gq.data_ptr<float>()[4*i+d]=result[d+3];
    }
    return {out,gm,gq,gs};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);m.def("cpu_diagnostic",&cpu_diagnostic);}
