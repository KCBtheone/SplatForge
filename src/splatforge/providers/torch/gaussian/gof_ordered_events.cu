#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
template<int D> using A=torch::PackedTensorAccessor32<float,D,torch::RestrictPtrTraits>;
using Bool3=torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits>;

__global__ void ordered(A<4> proxy,A<4> qr,A<4> qe,A<2> op,A<3> rays,
 A<4> v,A<3> rawden,A<3> den,A<3> num,A<3> depth,A<4> offset,A<3> exponent,
 A<3> response,Bool3 mask,A<4> normal,A<3> length,A<4> left,
 A<3> ga,A<3> gd,A<4> gn,A<4> gp,A<4> gqr,A<4> gqe,A<2> gop,A<4> rayparts,
 int L,int P,int count){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=count)return;
 int p=k%P,l=(k/P)%L,b=k/(L*P);
 float r[3],o[3],qv[3],go[3],gv[3],gl[3];
 for(int i=0;i<3;i++){r[i]=rays[b][p][i];o[i]=offset[b][l][p][i];qv[i]=v[b][l][p][i];}
 // Reverse actual alpha -> where -> exp -> multiply(-.5) -> clamp nodes.
 float grsp=mask[b][l][p] ? ga[b][l][p]*op[b][l] : 0.f;
 float ge=(grsp*response[b][l][p])*(-.5f);
 if(exponent[b][l][p]<0.f)ge=0.f;
 for(int i=0;i<3;i++)gl[i]=ge*o[i];
 float ndot=0.f;
 for(int i=0;i<3;i++)ndot+=normal[b][l][p][i]*gn[b][l][p][i];
 for(int i=0;i<3;i++){
   float branch=(gl[0]*qe[b][l][i][0]+gl[1]*qe[b][l][i][1])+gl[2]*qe[b][l][i][2];
   go[i]=branch+ge*left[b][l][p][i];
   float len=length[b][l][p];
   gv[i]=len>=1e-12f?-(gn[b][l][p][i]-normal[b][l][p][i]*ndot)/len:-gn[b][l][p][i]/1e-12f;
   for(int j=0;j<3;j++)atomicAdd(&gqe[b][l][i][j],o[i]*gl[j]);
 }
 float dgrad=gd[b][l][p]+((go[0]*r[0]+go[1]*r[1])+go[2]*r[2]);
 float dn=den[b][l][p],gnum=dgrad/dn;
 float gden=rawden[b][l][p]>=1e-12f?-dgrad*num[b][l][p]/(dn*dn):0.f;
 for(int i=0;i<3;i++){
   gp[b][l][p][i]=-go[i]+gnum*qv[i];
   gv[i]=(gv[i]+gnum*proxy[b][l][p][i])+gden*r[i];
 }
 for(int i=0;i<3;i++){
   float back=(qr[b][l][0][i]*gv[0]+qr[b][l][1][i]*gv[1])+qr[b][l][2][i]*gv[2];
   float gr=(go[i]*depth[b][l][p]+gden*qv[i])+back;
   for(int j=0;j<3;j++){
     atomicAdd(&gqr[b][l][i][j],gv[i]*r[j]);
   }
   rayparts[b][l][p][i]=gr;
 }
 atomicAdd(&gop[b][l],ga[b][l][p]*(mask[b][l][p]?response[b][l][p]:0.f));
}

std::vector<torch::Tensor> backward(std::vector<torch::Tensor> s,torch::Tensor ga,torch::Tensor gd,torch::Tensor gn){
 c10::cuda::CUDAGuard guard(s[0].device());
 int B=s[0].size(0),L=s[0].size(1),P=s[0].size(2),count=B*L*P;
 auto gp=torch::empty({B,L,P,3},s[0].options()),gqr=torch::zeros_like(s[1]),gqe=torch::zeros_like(s[2]);
 auto gop=torch::zeros_like(s[3]),rayparts=torch::empty_like(gp);
 #define T(I,D) s[I].packed_accessor32<float,D,torch::RestrictPtrTraits>()
 #define O(X,D) X.packed_accessor32<float,D,torch::RestrictPtrTraits>()
 ordered<<<(count+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
 T(0,4),T(1,4),T(2,4),T(3,2),T(4,3),T(5,4),T(6,3),T(7,3),T(8,3),T(9,3),
 T(10,4),T(11,3),T(12,3),s[13].packed_accessor32<bool,3,torch::RestrictPtrTraits>(),T(14,4),T(15,3),T(16,4),
 O(ga,3),O(gd,3),O(gn,4),O(gp,4),O(gqr,4),O(gqe,4),O(gop,2),O(rayparts,4),L,P,count);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
 return {gp,gqr,gqe,gop,rayparts.sum(1)};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("backward",&backward);}
