// Exact finite predicate for training diagnostics; no gradient arithmetic.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

struct View {
    const float* data;
    int64_t shape[8], stride[8], count, blocks;
    int rank;
};
struct Views { View values[16]; int count; };

__global__ void finite_kernel(Views views, int* result) {
    int64_t block=blockIdx.x;
    int tensor=0;
    while(tensor<views.count && block>=views.values[tensor].blocks) {
        block-=views.values[tensor].blocks;
        ++tensor;
    }
    if(tensor==views.count) return;
    const auto& view=views.values[tensor];
    int64_t index=block*256+threadIdx.x;
    if(index>=view.count) return;
    int64_t address=0;
    for(int axis=view.rank-1;axis>=0;--axis) {
        address+=(index%view.shape[axis])*view.stride[axis];
        index/=view.shape[axis];
    }
    if(!isfinite(view.data[address])) atomicExch(result,0);
}

torch::Tensor all_finite(std::vector<torch::Tensor> tensors) {
    TORCH_CHECK(!tensors.empty() && tensors.size()<=16,"one to16 explicit operands required");
    auto device=tensors[0].device();
    TORCH_CHECK(device.is_cuda(),"explicit CUDA operands required");
    c10::cuda::CUDAGuard guard(device);
    Views views{};
    views.count=tensors.size();
    int64_t blocks=0;
    for(size_t i=0;i<tensors.size();++i) {
        const auto& t=tensors[i];
        TORCH_CHECK(t.device()==device && t.scalar_type()==torch::kFloat32 && t.layout()==torch::kStrided,
            "same-device strided float32 operands required; no implicit cast/copy");
        TORCH_CHECK(t.dim()<=8,"at most8 dimensions required");
        auto& view=views.values[i];
        view.data=t.data_ptr<float>();
        view.rank=t.dim();view.count=t.numel();view.blocks=(t.numel()+255)/256;
        for(int axis=0;axis<t.dim();++axis) {
            view.shape[axis]=t.size(axis);view.stride[axis]=t.stride(axis);
        }
        blocks+=view.blocks;
    }
    TORCH_CHECK(blocks<=2147483647,"finite-check launch exceeds32-bit grid");
    auto result=torch::ones({},tensors[0].options().dtype(torch::kInt32));
    if(blocks) finite_kernel<<<blocks,256,0,at::cuda::getCurrentCUDAStream()>>>(views,result.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("all_finite",&all_finite); }
