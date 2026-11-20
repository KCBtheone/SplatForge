// GOF ray maxima and AmbiSuR planar events: fused forward + analytic VJP.
// Independent implementation of SplatForge's own tensor equations.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>
template<int D> using A=torch::PackedTensorAccessor32<float,D,torch::RestrictPtrTraits>;
using B2=torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits>;

struct Event {
 float dx,dy,squared,r[3],p[3],n[3],qd[3],offset[3],qoffset[3],rawden,den,num,depth,exponent,response,norm;
 bool gate,plane_valid;
};
__device__ Event evaluate(int b,int l,int p,A<4> delta,A<4> proxy,A<4> q2,A<4> q3,A<3> normal,A<2> distance,A<3> rays,A<3> squared,A<4> qray,A<3> ray_depth,A<3> exponent,B2 valid,int mode,float near,float far,float extent){
 Event e{};e.dx=delta[b][l][p][0];e.dy=delta[b][l][p][1];
 // Discrete support must preserve the reference einsum contraction order.
 // A one-ULP reassociation at extent^2 can add/remove an entire event.
 e.squared=squared[b][l][p];
 for(int j=0;j<3;++j){e.r[j]=rays[b][p][j];e.p[j]=proxy[b][l][p][j];}
 if(mode==0){
  // Shape-dependent cuBLAS contractions and pointwise reductions retain
  // their original CUDA nodes. Their exact VJPs are routed back below.
  for(int j=0;j<3;++j)e.qd[j]=qray[b][l][p][j];
  e.depth=ray_depth[b][l][p];e.exponent=exponent[b][l][p];
  e.response=expf(-.5f*fmaxf(e.exponent,0.f));
  for(int j=0;j<3;++j)e.norm+=e.qd[j]*e.qd[j];
  e.norm=sqrtf(e.norm);for(int j=0;j<3;++j)e.n[j]=-e.qd[j]/fmaxf(e.norm,1e-12f);
 }else{
  for(int j=0;j<3;++j){e.n[j]=normal[b][l][j];e.rawden+=e.n[j]*e.r[j];}
  e.plane_valid=fabsf(e.rawden)>1e-8f;e.den=e.plane_valid?e.rawden:1.f;
  e.depth=distance[b][l]/e.den;e.response=e.plane_valid?expf(-.5f*e.squared):0.f;
 }
 e.gate=valid[b][l]&&e.squared<=extent*extent&&near<e.depth&&e.depth<far;
 if(!e.gate)e.response=0.f;
 return e;
}

__global__ void fwd(int L,int P,int total,A<4> delta,A<4> proxy,A<4> q2,A<4> q3,A<3> normal,A<2> distance,A<2> opacity,A<3> rays,A<3> squared,A<4> qray,A<3> ray_depth,A<3> exponent,B2 valid,int mode,float near,float far,float extent,A<3> alpha,A<3> depth,A<4> normals){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
 int p=i%P,l=(i/P)%L,b=i/(P*L);
 Event e=evaluate(b,l,p,delta,proxy,q2,q3,normal,distance,rays,squared,qray,ray_depth,exponent,valid,mode,near,far,extent);
 alpha[b][l][p]=opacity[b][l]*e.response;depth[b][l][p]=e.depth;
 for(int j=0;j<3;++j)normals[b][l][p][j]=e.n[j];
}

__global__ void bwd(int L,int P,int total,A<4> delta,A<4> proxy,A<4> q2,A<4> q3,A<3> normal,A<2> distance,A<2> opacity,A<3> rays,A<3> squared,A<4> qray,A<3> ray_depth,A<3> exponent,B2 valid,int mode,float near,float far,float extent,A<3> ga,A<3> gd,A<4> gn,A<4> gdelta,A<4> gproxy,A<4> gq2,A<4> gq3,A<3> gnormal,A<2> gdistance,A<2> gopacity,A<3> grays,A<3> gsquared,A<4> gqray,A<3> gray_depth,A<3> gexponent){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=total)return;
 int p=i%P,l=(i/P)%L,b=i/(P*L);
 Event e=evaluate(b,l,p,delta,proxy,q2,q3,normal,distance,rays,squared,qray,ray_depth,exponent,valid,mode,near,far,extent);
 atomicAdd(&gopacity[b][l],ga[b][l][p]*e.response);
 if(mode==0){
  float ge=(e.gate&&e.exponent>=0.f)?-.5f*ga[b][l][p]*opacity[b][l]*e.response:0.f;
  float ndot=0.f;
  for(int j=0;j<3;++j)ndot+=gn[b][l][p][j]*e.n[j];
  for(int j=0;j<3;++j){
   gqray[b][l][p][j]=-(gn[b][l][p][j]-(e.norm>=1e-12f?e.n[j]*ndot:0.f))/fmaxf(e.norm,1e-12f);
  }
  gray_depth[b][l][p]=gd[b][l][p];gexponent[b][l][p]=ge;
 }else{
  float gs=(e.gate&&e.plane_valid)?-.5f*ga[b][l][p]*opacity[b][l]*e.response:0.f;
  // VJP through the original quadratic-form node; do not double count it
  // through delta/q2 as well. The other event adjoints remain fused.
  gsquared[b][l][p]=gs;
  atomicAdd(&gdistance[b][l],gd[b][l][p]/e.den);
  float dd=e.plane_valid?-gd[b][l][p]*distance[b][l]/(e.den*e.den):0.f;
  for(int j=0;j<3;++j){atomicAdd(&gnormal[b][l][j],gn[b][l][p][j]+dd*e.r[j]);atomicAdd(&grays[b][p][j],dd*e.n[j]);}
 }
}

void validate(std::vector<torch::Tensor> xs,torch::Tensor valid){
 auto device=xs[0].device();for(auto x:xs){TORCH_CHECK(x.is_cuda()&&x.device()==device&&x.scalar_type()==torch::kFloat32,"explicit same-device CUDA float32 event operands required");TORCH_CHECK(x.numel()<INT_MAX,"split execution batch before 32-bit overflow");}
 TORCH_CHECK(valid.device()==device&&valid.scalar_type()==torch::kBool,"same-device boolean slot mask required");
 auto d=xs[0],p=xs[1],q=xs[2],Q=xs[3],n=xs[4],z=xs[5],o=xs[6],r=xs[7];
 TORCH_CHECK(d.dim()==4&&d.size(3)==2&&p.dim()==4&&p.size(0)==d.size(0)&&p.size(1)==d.size(1)&&p.size(2)==d.size(2)&&p.size(3)==3,"invalid per-event shapes");
 int B=d.size(0),L=d.size(1),P=d.size(2);
 TORCH_CHECK(q.sizes().vec()==std::vector<int64_t>({B,L,2,2})&&Q.sizes().vec()==std::vector<int64_t>({B,L,3,3})&&n.sizes().vec()==std::vector<int64_t>({B,L,3})&&z.sizes().vec()==std::vector<int64_t>({B,L})&&o.sizes()==z.sizes()&&r.sizes().vec()==std::vector<int64_t>({B,P,3})&&valid.sizes()==z.sizes(),"invalid event operand shapes");
}
#define ACC(x,d) x.packed_accessor32<float,d,torch::RestrictPtrTraits>()
#define ARGS ACC(delta,4),ACC(proxy,4),ACC(q2,4),ACC(q3,4),ACC(normal,3),ACC(distance,2),ACC(opacity,2),ACC(rays,3),ACC(squared,3),ACC(qray,4),ACC(ray_depth,3),ACC(exponent,3),valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),mode,near,far,extent

std::vector<torch::Tensor> forward(torch::Tensor delta,torch::Tensor proxy,torch::Tensor q2,torch::Tensor q3,torch::Tensor normal,torch::Tensor distance,torch::Tensor opacity,torch::Tensor rays,torch::Tensor squared,torch::Tensor qray,torch::Tensor ray_depth,torch::Tensor exponent,torch::Tensor valid,int mode,double near,double far,double extent){
 validate({delta,proxy,q2,q3,normal,distance,opacity,rays},valid);c10::cuda::CUDAGuard guard(delta.device());
 int B=delta.size(0),L=delta.size(1),P=delta.size(2);TORCH_CHECK(mode==0||mode==1,"GOF or AmbiSuR mode required");
 TORCH_CHECK(squared.device()==delta.device()&&squared.scalar_type()==torch::kFloat32&&squared.sizes().vec()==std::vector<int64_t>({B,L,P}),"invalid reference quadratic form");
 for(auto x:{qray,ray_depth,exponent})TORCH_CHECK(x.device()==delta.device()&&x.scalar_type()==torch::kFloat32,"explicit same-device float32 prepared GOF nodes required");
 TORCH_CHECK(qray.sizes().vec()==std::vector<int64_t>({B,L,P,3})&&ray_depth.sizes().vec()==std::vector<int64_t>({B,L,P})&&exponent.sizes()==ray_depth.sizes(),"invalid prepared GOF nodes");
 auto alpha=torch::empty({B,L,P},delta.options()),depth=torch::empty_like(alpha),normals=torch::empty({B,L,P,3},delta.options());
 if(B*L*P)fwd<<<(B*L*P+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(L,P,B*L*P,ARGS,ACC(alpha,3),ACC(depth,3),ACC(normals,4));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {alpha,depth,normals};
}
std::vector<torch::Tensor> backward(torch::Tensor delta,torch::Tensor proxy,torch::Tensor q2,torch::Tensor q3,torch::Tensor normal,torch::Tensor distance,torch::Tensor opacity,torch::Tensor rays,torch::Tensor squared,torch::Tensor qray,torch::Tensor ray_depth,torch::Tensor exponent,torch::Tensor valid,int mode,double near,double far,double extent,torch::Tensor ga,torch::Tensor gd,torch::Tensor gn){
 validate({delta,proxy,q2,q3,normal,distance,opacity,rays},valid);c10::cuda::CUDAGuard guard(delta.device());
 int B=delta.size(0),L=delta.size(1),P=delta.size(2);
 TORCH_CHECK(ga.device()==delta.device()&&gd.device()==delta.device()&&gn.device()==delta.device()&&ga.scalar_type()==torch::kFloat32&&gd.scalar_type()==torch::kFloat32&&gn.scalar_type()==torch::kFloat32,"invalid event adjoint dtype/device");
 TORCH_CHECK(ga.sizes().vec()==std::vector<int64_t>({B,L,P})&&gd.sizes()==ga.sizes()&&gn.sizes().vec()==std::vector<int64_t>({B,L,P,3}),"invalid adjoint shapes");
 auto gdelta=torch::zeros(delta.sizes(),delta.options()),gproxy=torch::zeros(proxy.sizes(),proxy.options()),gq2=torch::zeros(q2.sizes(),q2.options()),gq3=torch::zeros(q3.sizes(),q3.options()),gnormal=torch::zeros(normal.sizes(),normal.options()),gdistance=torch::zeros(distance.sizes(),distance.options()),gopacity=torch::zeros(opacity.sizes(),opacity.options()),grays=torch::zeros(rays.sizes(),rays.options());
 auto gsquared=torch::zeros_like(squared);
 auto gqray=torch::zeros_like(qray),gray_depth=torch::zeros_like(ray_depth),gexponent=torch::zeros_like(exponent);
 if(B*L*P)bwd<<<(B*L*P+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(L,P,B*L*P,ARGS,ACC(ga,3),ACC(gd,3),ACC(gn,4),ACC(gdelta,4),ACC(gproxy,4),ACC(gq2,4),ACC(gq3,4),ACC(gnormal,3),ACC(gdistance,2),ACC(gopacity,2),ACC(grays,3),ACC(gsquared,3),ACC(gqray,4),ACC(gray_depth,3),ACC(gexponent,3));
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {gdelta,gproxy,gq2,gq3,gnormal,gdistance,gopacity,grays,gsquared,gqray,gray_depth,gexponent};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);}
