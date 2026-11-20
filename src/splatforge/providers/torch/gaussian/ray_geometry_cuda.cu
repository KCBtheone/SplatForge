// SplatForge-owned implementation of regularized_ray_geometry@1 equations.
// Double-precision Cholesky is intentional. No fast-math or adjugate inverse.
// Backward uses generated analytic reverse mode for fixed cameras. Camera
// VJPs use packed exact directional derivatives and shared atomic additions.
// No finite differences, precision switch or formula/regularization change.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>
#include "ray_geometry_reverse.cuh"

struct D {
  double v, g;
  __device__ D(double x=0., double dx=0.):v(x),g(dx){}
};
__device__ D operator+(D a,D b){return D(a.v+b.v,a.g+b.g);}
__device__ D operator-(D a,D b){return D(a.v-b.v,a.g-b.g);}
__device__ D operator-(D a){return D(-a.v,-a.g);}
__device__ D operator*(D a,D b){return D(a.v*b.v,a.g*b.v+a.v*b.g);}
__device__ D operator/(D a,D b){return D(a.v/b.v,(a.g*b.v-a.v*b.g)/(b.v*b.v));}
__device__ D root(D a){double x=sqrt(a.v);return D(x,a.g/(2*x));}
__device__ D lower(D a,D b){return a.v>=b.v?a:b;}
__device__ D upper(D a,D b){return a.v<=b.v?a:b;}
__device__ D square(D a){return a*a;}

// Parameter coordinates: mean3, quaternion4, scale3, transform16, intrinsics9.
__device__ D input(float x,int coordinate,int seed){return D(double(x),coordinate==seed?1.:0.);}
__device__ void evaluate(const float* means,const float* quats,const float* scales,
 const float* transform,const float* K,int seed,int width,int height,
 double relative,double absolute,D* out){
  D m[3],q[4],s[3],R[9],t[3],kk[9];
  for(int i=0;i<3;++i){m[i]=input(means[i],i,seed);s[i]=input(scales[i],7+i,seed);}
  D qnorm;
  for(int i=0;i<4;++i){q[i]=input(quats[i],3+i,seed);qnorm=qnorm+square(q[i]);}
  qnorm=lower(root(qnorm),D(1e-12));
  for(int i=0;i<4;++i)q[i]=q[i]/qnorm;
  for(int i=0;i<3;++i){
    t[i]=input(transform[i*4+3],10+i*4+3,seed);
    for(int j=0;j<3;++j)R[i*3+j]=input(transform[i*4+j],10+i*4+j,seed);
  }
  for(int i=0;i<9;++i)kk[i]=input(K[i],26+i,seed);
  D local[3];
  for(int i=0;i<3;++i){local[i]=t[i];for(int j=0;j<3;++j)local[i]=local[i]+R[i*3+j]*m[j];}
  D w=q[0],x=q[1],y=q[2],zq=q[3];
  D A[9]={D(1)-D(2)*(y*y+zq*zq),D(2)*(x*y-w*zq),D(2)*(x*zq+w*y),
    D(2)*(x*y+w*zq),D(1)-D(2)*(x*x+zq*zq),D(2)*(y*zq-w*x),
    D(2)*(x*zq-w*y),D(2)*(y*zq+w*x),D(1)-D(2)*(x*x+y*y)};
  // Match reference evaluation order: C_world=A diag(s^2) A^T,
  // then C_camera=R C_world R^T, then explicit symmetrization.
  D Cw[9],tmp[9],C[9];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    D a;for(int k=0;k<3;++k)a=a+(A[i*3+k]*square(s[k]))*A[j*3+k];Cw[i*3+j]=a;
  }
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    D a;for(int k=0;k<3;++k)a=a+R[i*3+k]*Cw[k*3+j];tmp[i*3+j]=a;
  }
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    D a;for(int k=0;k<3;++k)a=a+tmp[i*3+k]*R[j*3+k];C[i*3+j]=a;
  }
  D jitter=lower(D(relative)*lower(C[0]+C[4]+C[8],D(absolute)),D(absolute));
  D S[9];for(int i=0;i<3;++i)for(int j=0;j<3;++j)
    S[i*3+j]=(C[i*3+j]+C[j*3+i])*D(.5)+(i==j?jitter:D(0));
  D z=local[2],fx=kk[0],fy=kk[4],cx=kk[2],cy=kk[5];
  D u=upper(lower(local[0]/z,-cx/fx-D(.15*width)/fx),(D(width)-cx)/fx+D(.15*width)/fx);
  D v=upper(lower(local[1]/z,-cy/fy-D(.15*height)/fy),(D(height)-cy)/fy+D(.15*height)/fy);
  D ray[3]={u,v,D(1)};
  D L[9];
  for(int i=0;i<3;++i)for(int j=0;j<=i;++j){
    D a=S[i*3+j];for(int k=0;k<j;++k)a=a-L[i*3+k]*L[j*3+k];
    L[i*3+j]=i==j?root(a):a/L[j*3+j];
  }
  D b[3],h[3];
  for(int i=0;i<3;++i){D a=ray[i];for(int k=0;k<i;++k)a=a-L[i*3+k]*b[k];b[i]=a/L[i*3+i];}
  for(int i=2;i>=0;--i){D a=b[i];for(int k=i+1;k<3;++k)a=a-L[k*3+i]*h[k];h[i]=a/L[i*3+i];}
  D den=h[0]*u+h[1]*v+h[2];for(int i=0;i<3;++i)h[i]=h[i]/den;
  D p0=(v*v+D(1))*h[0]-u*v*h[1]-u*h[2];
  D p1=-u*v*h[0]+(u*u+D(1))*h[1]-v*h[2];
  D tx=z*u,ty=z*v,dist=root(tx*tx+ty*ty+z*z);
  D factor=dist/(u*u+v*v+D(1));
  D rn0=-p0*factor,rn1=-p1*factor;
  D n[3]={rn0/z-tx/dist,rn1/z-ty/dist,-tx/(z*z)*rn0-ty/(z*z)*rn1-z/dist};
  D nn=lower(root(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]),D(1e-12));
  out[0]=root(local[0]*local[0]+local[1]*local[1]+local[2]*local[2]);
  out[1]=p0*factor/fx;out[2]=p1*factor/fy;
  for(int i=0;i<3;++i)out[3+i]=n[i]/nn;
}

__global__ void forward_kernel(int N,const float* m,const float* q,const float* s,
 const float* T,const float* K,const bool* visible,int W,int H,double r,double a,
 float* d,float* p,float* n){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
  D out[6];
  if(visible[i])evaluate(m+3*i,q+4*i,s+3*i,T,K,-1,W,H,r,a,out);
  d[i]=float(out[0].v);for(int j=0;j<2;++j)p[2*i+j]=float(out[1+j].v);
  for(int j=0;j<3;++j)n[3*i+j]=float(out[3+j].v);
}

__global__ void backward_kernel(int N,int variables,const float* m,const float* q,const float* s,
 const float* T,const float* K,const bool* visible,int W,int H,double r,double a,
 const float* gd,const float* gp,const float* gn,float* gm,float* gq,float* gs,float* gT,float* gK){
  int64_t coordinate=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
  int i=coordinate/variables,seed=coordinate%variables;
  if(i>=N||seed>=variables||!visible[i])return;
  // Fixed cameras have ten coordinates, not a whole32-lane warp per Gaussian.
  // This preserves each directional derivative while packing3.2 Gaussians per
  // warp (35 coordinates in camera-VJP mode), instead of wasting22 lanes.
  D out[6];evaluate(m+3*i,q+4*i,s+3*i,T,K,seed,W,H,r,a,out);
  double grad=(gd?gd[i]:0.)*out[0].g;
  for(int j=0;j<2;++j)grad+=(gp?gp[2*i+j]:0.)*out[1+j].g;
  for(int j=0;j<3;++j)grad+=(gn?gn[3*i+j]:0.)*out[3+j].g;
  if(seed<3)gm[3*i+seed]=float(grad);
  else if(seed<7)gq[4*i+seed-3]=float(grad);
  else if(seed<10)gs[3*i+seed-7]=float(grad);
  else if(seed<26)atomicAdd(gT+seed-10,float(grad));
  else atomicAdd(gK+seed-26,float(grad));
}

__global__ void reverse_fixed_kernel(int N,const float* m,const float* q,const float* s,
 const float* T,const float* K,const bool* visible,int W,int H,double r,double a,
 const float* gd,const float* gp,const float* gn,float* gm,float* gq,float* gs){
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i>=N || !visible[i])return;
  double incoming[6]={gd?double(gd[i]):0.,gp?double(gp[2*i]):0.,
    gp?double(gp[2*i+1]):0.,gn?double(gn[3*i]):0.,
    gn?double(gn[3*i+1]):0.,gn?double(gn[3*i+2]):0.};
  reverse_fixed(m+3*i,q+4*i,s+3*i,T,K,W,H,r,a,incoming,gm+3*i,gq+4*i,gs+3*i);
}

void validate(torch::Tensor m,torch::Tensor q,torch::Tensor s,torch::Tensor T,torch::Tensor K,torch::Tensor visible){
  for(auto x:{m,q,s,T,K}){
    TORCH_CHECK(x.is_cuda()&&x.scalar_type()==torch::kFloat32&&x.is_contiguous(),"explicit contiguous CUDA float32 required");
    TORCH_CHECK(x.device()==m.device(),"all operands must use the same CUDA device");
  }
  TORCH_CHECK(m.dim()==2&&m.size(1)==3&&q.dim()==2&&q.size(0)==m.size(0)&&q.size(1)==4&&s.sizes()==m.sizes(),"invalid scene shapes");
  TORCH_CHECK(T.dim()==2&&T.size(0)==4&&T.size(1)==4&&K.dim()==2&&K.size(0)==3&&K.size(1)==3,"invalid camera shapes");
  TORCH_CHECK(visible.is_cuda()&&visible.device()==m.device()&&visible.scalar_type()==torch::kBool&&visible.is_contiguous()&&visible.dim()==1&&visible.size(0)==m.size(0),"invalid visibility mask");
  TORCH_CHECK(m.size(0)<INT_MAX,"split scene before 32-bit index overflow");
}

std::vector<torch::Tensor> forward(torch::Tensor m,torch::Tensor q,torch::Tensor s,torch::Tensor T,torch::Tensor K,torch::Tensor visible,int W,int H,double r,double a){
  validate(m,q,s,T,K,visible);c10::cuda::CUDAGuard guard(m.device());
  auto d=torch::empty({m.size(0)},m.options()),p=torch::empty({m.size(0),2},m.options()),n=torch::empty_like(m);
  if(m.size(0))forward_kernel<<<(m.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(m.size(0),m.data_ptr<float>(),q.data_ptr<float>(),s.data_ptr<float>(),T.data_ptr<float>(),K.data_ptr<float>(),visible.data_ptr<bool>(),W,H,r,a,d.data_ptr<float>(),p.data_ptr<float>(),n.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {d,p,n};
}

std::vector<torch::Tensor> backward(torch::Tensor m,torch::Tensor q,torch::Tensor s,torch::Tensor T,torch::Tensor K,torch::Tensor visible,int W,int H,double r,double a,c10::optional<torch::Tensor> gd,c10::optional<torch::Tensor> gp,c10::optional<torch::Tensor> gn,bool cameras){
  validate(m,q,s,T,K,visible);c10::cuda::CUDAGuard guard(m.device());
  for(auto pair: {std::make_pair(gd,std::vector<int64_t>{m.size(0)}),std::make_pair(gp,std::vector<int64_t>{m.size(0),2}),std::make_pair(gn,std::vector<int64_t>{m.size(0),3})})
    if(pair.first.has_value()){auto x=pair.first.value();TORCH_CHECK(x.is_contiguous()&&x.device()==m.device()&&x.scalar_type()==torch::kFloat32&&x.sizes().vec()==pair.second,"invalid output adjoint");}
  auto gm=torch::zeros_like(m),gq=torch::zeros_like(q),gs=torch::zeros_like(s),gT=torch::zeros_like(T),gK=torch::zeros_like(K);
  int variables=cameras?35:10;
  if(m.size(0) && cameras)backward_kernel<<<(m.size(0)*variables+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(m.size(0),variables,m.data_ptr<float>(),q.data_ptr<float>(),s.data_ptr<float>(),T.data_ptr<float>(),K.data_ptr<float>(),visible.data_ptr<bool>(),W,H,r,a,gd.has_value()?gd.value().data_ptr<float>():nullptr,gp.has_value()?gp.value().data_ptr<float>():nullptr,gn.has_value()?gn.value().data_ptr<float>():nullptr,gm.data_ptr<float>(),gq.data_ptr<float>(),gs.data_ptr<float>(),gT.data_ptr<float>(),gK.data_ptr<float>());
  if(m.size(0) && !cameras)reverse_fixed_kernel<<<(m.size(0)+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(m.size(0),m.data_ptr<float>(),q.data_ptr<float>(),s.data_ptr<float>(),T.data_ptr<float>(),K.data_ptr<float>(),visible.data_ptr<bool>(),W,H,r,a,gd.has_value()?gd.value().data_ptr<float>():nullptr,gp.has_value()?gp.value().data_ptr<float>():nullptr,gn.has_value()?gn.value().data_ptr<float>():nullptr,gm.data_ptr<float>(),gq.data_ptr<float>(),gs.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {gm,gq,gs,gT,gK};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);}
