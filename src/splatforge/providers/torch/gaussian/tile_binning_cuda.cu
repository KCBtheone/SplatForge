// Exact inclusive tile-edge footprint membership. Discrete, no VJP.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

__device__ bool rectangle(const float* a,int W,int H,int tile,double near,double far,int& l,int& r,int& t,int& b){
  float x=a[0],y=a[1],radius=a[2],z=a[3];
  if(!(near<z&&z<far)||x+radius<0||y+radius<0||x-radius>W||y-radius>H)return false;
  // NumPy reference rounds x +/- radius in float32, then divides in double.
  int columns=(W+tile-1)/tile,rows=(H+tile-1)/tile;
  // Clamp in double BEFORE conversion, including enormous finite footprints.
  l=int(fmax(0.,fmin(double(columns),ceil(double(x-radius)/tile)-1.)));
  r=int(fmin(double(columns-1),fmax(-1.,floor(double(x+radius)/tile))));
  t=int(fmax(0.,fmin(double(rows),ceil(double(y-radius)/tile)-1.)));
  b=int(fmin(double(rows-1),fmax(-1.,floor(double(y+radius)/tile))));
  return l<=r&&t<=b;
}
__global__ void count_kernel(int N,const float* bounds,int W,int H,int tile,double near,double far,int64_t* counts,int* error){
  int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N)return;
  const float* a=bounds+4*i;
  for(int j=0;j<4;++j)if(!isfinite(a[j])){atomicExch(error,1);counts[i]=0;return;}
  int l,r,t,b;counts[i]=rectangle(a,W,H,tile,near,far,l,r,t,b)?int64_t(r-l+1)*(b-t+1):0;
}
__global__ void emit_kernel(int N,const float* bounds,const int64_t* order,const int64_t* offsets,int W,int H,int tile,double near,double far,int64_t* keys,int64_t* ids){
  int rank=blockIdx.x*blockDim.x+threadIdx.x;if(rank>=N)return;
  int index=order[rank];int l,r,t,b;
  if(!rectangle(bounds+4*index,W,H,tile,near,far,l,r,t,b))return;
  int64_t cursor=offsets[index];
  for(int row=t;row<=b;++row)for(int col=l;col<=r;++col){
    keys[cursor]=int64_t(row*((W+tile-1)/tile)+col)*N+rank;
    ids[cursor]=index;++cursor;
  }
}
void validate(torch::Tensor bounds){
  TORCH_CHECK(bounds.is_cuda()&&bounds.scalar_type()==torch::kFloat32&&bounds.is_contiguous()&&bounds.dim()==2&&bounds.size(1)==4,"contiguous CUDA float32 bounds[N,4] required");
  TORCH_CHECK(bounds.size(0)<INT_MAX,"32-bit Gaussian index overflow");
}
std::vector<torch::Tensor> count(torch::Tensor bounds,int W,int H,int tile,double near,double far){
  validate(bounds);TORCH_CHECK(W>0&&H>0&&tile>0&&near<far,"invalid tile settings");
  c10::cuda::CUDAGuard guard(bounds.device());
  auto counts=torch::zeros({bounds.size(0)},bounds.options().dtype(torch::kInt64));
  auto error=torch::zeros({},bounds.options().dtype(torch::kInt32));
  if(bounds.size(0))count_kernel<<<(bounds.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(bounds.size(0),bounds.data_ptr<float>(),W,H,tile,near,far,counts.data_ptr<int64_t>(),error.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {counts,error};
}
std::vector<torch::Tensor> emit(torch::Tensor bounds,torch::Tensor order,torch::Tensor offsets,int64_t total,int W,int H,int tile,double near,double far){
  validate(bounds);c10::cuda::CUDAGuard guard(bounds.device());
  for(auto x:{order,offsets})TORCH_CHECK(x.device()==bounds.device()&&x.scalar_type()==torch::kInt64&&x.is_contiguous()&&x.dim()==1&&x.size(0)==bounds.size(0),"invalid binning index tensor");
  TORCH_CHECK(total>=0&&total<INT_MAX,"split binning before pair-index overflow");
  auto keys=torch::empty({total},order.options()),ids=torch::empty_like(keys);
  if(bounds.size(0))emit_kernel<<<(bounds.size(0)+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(bounds.size(0),bounds.data_ptr<float>(),order.data_ptr<int64_t>(),offsets.data_ptr<int64_t>(),W,H,tile,near,far,keys.data_ptr<int64_t>(),ids.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {keys,ids};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("count",&count);m.def("emit",&emit);}
