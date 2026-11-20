#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

template<int D> using A = torch::PackedTensorAccessor32<float,D,torch::RestrictPtrTraits>;
using B2 = torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits>;

struct Event {float r[3], p[3], v[3], o[3], n[3]; float den, rawden, num, d, e, len, response; bool gate;};

__device__ Event evaluate(int b,int l,int p,A<4> proxy,A<4> qr,A<4> qe,A<3> rays,A<3> sq,B2 valid,
                         float near,float far,float extent) {
    Event x;
    for(int i=0;i<3;i++){x.r[i]=rays[b][p][i];x.p[i]=proxy[b][l][p][i];}
    for(int i=0;i<3;i++) x.v[i]=(qr[b][l][i][0]*x.r[0]+qr[b][l][i][1]*x.r[1])+qr[b][l][i][2]*x.r[2];
    x.rawden=(x.v[0]*x.r[0]+x.v[1]*x.r[1])+x.v[2]*x.r[2];
    x.den=fmaxf(x.rawden,1e-12f);
    x.num=(x.p[0]*x.v[0]+x.p[1]*x.v[1])+x.p[2]*x.v[2];
    x.d=x.num/x.den;
    for(int i=0;i<3;i++)x.o[i]=x.d*x.r[i]-x.p[i];
    // Torch's left contraction o^T Q followed by dot with o.
    float left[3];
    for(int j=0;j<3;j++)left[j]=(x.o[0]*qe[b][l][0][j]+x.o[1]*qe[b][l][1][j])+x.o[2]*qe[b][l][2][j];
    x.e=(left[0]*x.o[0]+left[1]*x.o[1])+left[2]*x.o[2];
    x.len=sqrtf((x.v[0]*x.v[0]+x.v[1]*x.v[1])+x.v[2]*x.v[2]);
    for(int i=0;i<3;i++)x.n[i]=-x.v[i]/fmaxf(x.len,1e-12f);
    x.response=expf(-.5f*fmaxf(x.e,0.f));
    x.gate=valid[b][l] && sq[b][l][p]<=extent*extent && x.d>near && x.d<far;
    return x;
}

__global__ void forward_kernel(A<4> proxy,A<4> qr,A<4> qe,A<2> opacity,A<3> rays,A<3> sq,B2 valid,
 float near,float far,float extent,A<3> alpha,A<3> depth,A<4> normal,int L,int P,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
 int p=k%P,l=(k/P)%L,b=k/(P*L);
 Event x=evaluate(b,l,p,proxy,qr,qe,rays,sq,valid,near,far,extent);
 alpha[b][l][p]=opacity[b][l]*(x.gate?x.response:0.f);depth[b][l][p]=x.d;
 for(int i=0;i<3;i++)normal[b][l][p][i]=x.n[i];
}

__global__ void backward_kernel(A<4> proxy,A<4> qr,A<4> qe,A<2> opacity,A<3> rays,A<3> sq,B2 valid,
 float near,float far,float extent,A<3> ga,A<3> gd,A<4> gn,
 A<4> gp,A<4> gqr,A<4> gqe,A<2> gop,A<3> gr,int L,int P,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
 int p=k%P,l=(k/P)%L,b=k/(P*L);
 Event x=evaluate(b,l,p,proxy,qr,qe,rays,sq,valid,near,far,extent);
 float ge=(x.gate && x.e>=0.f)?-.5f*ga[b][l][p]*opacity[b][l]*x.response:0.f;
 atomicAdd(&gop[b][l],ga[b][l][p]*(x.gate?x.response:0.f));
 float go[3],gv[3];float gdepth=gd[b][l][p],ndot=0.f;
 for(int i=0;i<3;i++)ndot+=x.n[i]*gn[b][l][p][i];
 for(int i=0;i<3;i++){
   go[i]=0.f;
   for(int j=0;j<3;j++){
     go[i]+=ge*(qe[b][l][i][j]+qe[b][l][j][i])*x.o[j];
     atomicAdd(&gqe[b][l][i][j],ge*x.o[i]*x.o[j]);
   }
   gdepth+=go[i]*x.r[i];
   gv[i]=x.len>=1e-12f?-(gn[b][l][p][i]-x.n[i]*ndot)/x.len:-gn[b][l][p][i]/1e-12f;
 }
 float gnum=gdepth/x.den;
 float gden=x.rawden>=1e-12f?-gdepth*x.num/(x.den*x.den):0.f;
 for(int i=0;i<3;i++){
   gp[b][l][p][i]=-go[i]+gnum*x.v[i];
   gv[i]+=gnum*x.p[i]+gden*x.r[i];
 }
 for(int i=0;i<3;i++){
   float raygrad=go[i]*x.d+gden*x.v[i];
   for(int j=0;j<3;j++){
     raygrad+=qr[b][l][j][i]*gv[j];
     atomicAdd(&gqr[b][l][i][j],gv[i]*x.r[j]);
   }
   atomicAdd(&gr[b][p][i],raygrad);
 }
}

std::vector<torch::Tensor> forward(torch::Tensor proxy,torch::Tensor qr,torch::Tensor qe,
 torch::Tensor op,torch::Tensor rays,torch::Tensor sq,torch::Tensor valid,double near,double far,double extent){
 c10::cuda::CUDAGuard guard(proxy.device());
 int B=proxy.size(0),L=proxy.size(1),P=proxy.size(2),count=B*L*P;
 auto a=torch::empty({B,L,P},proxy.options()),d=torch::empty_like(a),n=torch::empty({B,L,P,3},proxy.options());
 forward_kernel<<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(proxy.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 qr.packed_accessor32<float,4,torch::RestrictPtrTraits>(),qe.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 op.packed_accessor32<float,2,torch::RestrictPtrTraits>(),rays.packed_accessor32<float,3,torch::RestrictPtrTraits>(),sq.packed_accessor32<float,3,torch::RestrictPtrTraits>(),valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),
 near,far,extent,a.packed_accessor32<float,3,torch::RestrictPtrTraits>(),d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),L,P,count);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {a,d,n};
}

std::vector<torch::Tensor> backward(torch::Tensor proxy,torch::Tensor qr,torch::Tensor qe,
 torch::Tensor op,torch::Tensor rays,torch::Tensor sq,torch::Tensor valid,double near,double far,double extent,
 torch::Tensor ga,torch::Tensor gd,torch::Tensor gn){
 c10::cuda::CUDAGuard guard(proxy.device());
 int B=proxy.size(0),L=proxy.size(1),P=proxy.size(2),count=B*L*P;
 auto gp=torch::empty({B,L,P,3},proxy.options()),gqr=torch::zeros_like(qr),gqe=torch::zeros_like(qe),gop=torch::zeros_like(op),gr=torch::zeros_like(rays);
 backward_kernel<<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(proxy.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 qr.packed_accessor32<float,4,torch::RestrictPtrTraits>(),qe.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 op.packed_accessor32<float,2,torch::RestrictPtrTraits>(),rays.packed_accessor32<float,3,torch::RestrictPtrTraits>(),sq.packed_accessor32<float,3,torch::RestrictPtrTraits>(),valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),near,far,extent,
 ga.packed_accessor32<float,3,torch::RestrictPtrTraits>(),gd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),gn.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 gp.packed_accessor32<float,4,torch::RestrictPtrTraits>(),gqr.packed_accessor32<float,4,torch::RestrictPtrTraits>(),gqe.packed_accessor32<float,4,torch::RestrictPtrTraits>(),gop.packed_accessor32<float,2,torch::RestrictPtrTraits>(),gr.packed_accessor32<float,3,torch::RestrictPtrTraits>(),L,P,count);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {gp,gqr,gqe,gop,gr};
}
// H024 changes only the pointwise stage; no Qray/quadratic contraction here.
__global__ void depth_offset_f(A<4> proxy,A<4> qray,A<3> rays,A<3> depth,A<4> offset,int L,int P,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
 int p=k%P,l=(k/P)%L,b=k/(L*P);
 float v[3],r[3],x[3];
 for(int i=0;i<3;i++){v[i]=qray[b][l][p][i];r[i]=rays[b][p][i];x[i]=proxy[b][l][p][i];}
 float den=fmaxf((v[0]*r[0]+v[1]*r[1])+v[2]*r[2],1e-12f);
 float num=(x[0]*v[0]+x[1]*v[1])+x[2]*v[2];
 float d=num/den;depth[b][l][p]=d;
 for(int i=0;i<3;i++)offset[b][l][p][i]=d*r[i]-x[i];
}

__global__ void depth_offset_b(A<4> proxy,A<4> qray,A<3> rays,A<3> depth,A<3> gd,A<4> go,
 A<4> gp,A<4> gq,A<4> gr,int L,int P,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
 int p=k%P,l=(k/P)%L,b=k/(L*P);
 float v[3],r[3],x[3],o[3];
 for(int i=0;i<3;i++){v[i]=qray[b][l][p][i];r[i]=rays[b][p][i];x[i]=proxy[b][l][p][i];o[i]=go[b][l][p][i];}
 float rawden=(v[0]*r[0]+v[1]*r[1])+v[2]*r[2],den=fmaxf(rawden,1e-12f);
 float d=depth[b][l][p];
 float gdepth=gd[b][l][p]+((o[0]*r[0]+o[1]*r[1])+o[2]*r[2]);
 float gnum=gdepth/den;
 // Preserve quotient node ordering rather than -g*num/(den*den).
 float gden=rawden>=1e-12f?(-gdepth)*(d/den):0.f;
 for(int i=0;i<3;i++){
   gp[b][l][p][i]=-o[i]+gnum*v[i];
   gq[b][l][p][i]=gnum*x[i]+gden*r[i];
   gr[b][l][p][i]=o[i]*d+gden*v[i];
 }
}

std::vector<torch::Tensor> depth_offset_forward(torch::Tensor proxy,torch::Tensor qray,torch::Tensor rays){
 c10::cuda::CUDAGuard guard(proxy.device());
 int B=proxy.size(0),L=proxy.size(1),P=proxy.size(2),count=B*L*P;
 auto d=torch::empty({B,L,P},proxy.options()),o=torch::empty({B,L,P,3},proxy.options());
 depth_offset_f<<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
 proxy.packed_accessor32<float,4,torch::RestrictPtrTraits>(),qray.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 rays.packed_accessor32<float,3,torch::RestrictPtrTraits>(),d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
 o.packed_accessor32<float,4,torch::RestrictPtrTraits>(),L,P,count);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {d,o};
}

std::vector<torch::Tensor> depth_offset_backward(torch::Tensor proxy,torch::Tensor qray,torch::Tensor rays,
 torch::Tensor depth,torch::Tensor gd,torch::Tensor go){
 c10::cuda::CUDAGuard guard(proxy.device());
 int B=proxy.size(0),L=proxy.size(1),P=proxy.size(2),count=B*L*P;
 auto gp=torch::empty({B,L,P,3},proxy.options()),gq=torch::empty_like(gp),gr=torch::empty_like(gp);
 depth_offset_b<<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
 proxy.packed_accessor32<float,4,torch::RestrictPtrTraits>(),qray.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 rays.packed_accessor32<float,3,torch::RestrictPtrTraits>(),depth.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
 gd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),go.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 gp.packed_accessor32<float,4,torch::RestrictPtrTraits>(),gq.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
 gr.packed_accessor32<float,4,torch::RestrictPtrTraits>(),L,P,count);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {gp,gq,gr.sum(1)};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);
 m.def("depth_offset_forward",&depth_offset_forward);m.def("depth_offset_backward",&depth_offset_backward);}
