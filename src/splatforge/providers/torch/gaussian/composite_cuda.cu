// Independent implementation of the versioned alpha-compositing equations.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

using A3 = torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>;
using A4 = torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>;
using I3 = torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits>;

struct Grad {
    const float* p;
    int64_t s0, s1, s2;
    __device__ float at(int b, int x, int y=0) const {
        return p ? p[b*s0 + x*s1 + y*s2] : 0.f;
    }
};

Grad grad_view(const c10::optional<torch::Tensor>& t) {
    if (!t.has_value()) return {nullptr, 0, 0, 0};
    const auto& v = t.value();
    return {v.data_ptr<float>(), v.stride(0), v.stride(1), v.dim()==3 ? v.stride(2) : 0};
}

__global__ void composite_fwd(A3 a, A3 c, A3 d, A4 n, I3 order,
    A3 rgb, A3 nnum, float* alpha, float* znum, float* l1, float* l2,
    float* median, A3 weights, A3 prefix, A3 auxw, A3 auxd, int* median_id,
    float* raw_l1, float* raw_l2, int N, int P, int BP, int mode) {
    int bp = blockIdx.x * blockDim.x + threadIdx.x;
    if (bp >= BP) return;
    int b = bp/P, p = bp%P;
    float T=1.f, W=0.f, D=0.f, D2=0.f, C[3]={0.f,0.f,0.f}, M[3]={0.f,0.f,0.f};
    int mid=-1;
    for (int i=0; i<N; ++i) {
        float opacity=fminf(0.999f, fmaxf(0.f, a[b][i][p]));
        float w=opacity*T;
        prefix[b][i][p]=T;
        weights[b][i][p]=w;
        T *= (1.f-opacity);
        W += w;
        D += w*d[b][i][p];
        D2 += w*d[b][i][p]*d[b][i][p];
        for (int k=0; k<3; ++k) { C[k]+=w*c[b][i][k]; M[k]+=w*n[b][i][p][k]; }
        if (mid<0 && W>=0.5f) mid=i;
    }
    alpha[bp]=W; znum[bp]=D;
    for (int k=0; k<3; ++k) { rgb[b][p][k]=C[k]; nnum[b][p][k]=M[k]; }
    median_id[bp]=mid;
    median[bp]=mid<0 ? 0.f : d[b][mid][p];
    raw_l2[bp]=(mode&2) ? 2.f*(W*D2-D*D) : 0.f;
    l2[bp]=fmaxf(0.f,raw_l2[bp]);
    float distortion=0.f;
    if (mode&1) {
        // Use PyTorch's actual sort indices, including its chosen tie ordering.
        float sw=0.f, sd=0.f;
        for (int rank=0; rank<N; ++rank) {
            int i=order[b][rank][p];
            float w=weights[b][i][p], depth=d[b][i][p];
            distortion += 2.f*w*(depth*sw-sd);
            auxw[b][i][p]=2.f*(2.f*depth*sw-depth*W+D-2.f*sd);
            auxd[b][i][p]=2.f*w*(2.f*sw+w-W);
            sw+=w; sd+=w*depth;
        }
    }
    raw_l1[bp]=distortion;
    l1[bp]=fmaxf(0.f, distortion);
}

__global__ void composite_bwd(A3 a, A3 c, A3 d, A4 n, A3 w, A3 prefix,
    A3 auxw, A3 auxd, const int* mid, const float* l1, const float* l2,
    Grad grgb, Grad ga, Grad gz, Grad gn, Grad gl1, Grad gl2, Grad gm, Grad gw,
    A3 da, A3 dc, A3 dd, A4 dn, int N, int P, int BP, int mode) {
    int bp=blockIdx.x*blockDim.x+threadIdx.x;
    if (bp>=BP) return;
    int b=bp/P, p=bp%P;
    float W=0.f, D=0.f;
    if (mode&2) for (int i=0; i<N; ++i) { W+=w[b][i][p]; D+=w[b][i][p]*d[b][i][p]; }
    float suffix=0.f;
    for (int i=N-1; i>=0; --i) {
        float wi=w[b][i][p], depth=d[b][i][p];
        float adj=ga.at(b,p)+gz.at(b,p)*depth+gw.at(b,i,p);
        for (int k=0; k<3; ++k) {
            adj+=grgb.at(b,p,k)*c[b][i][k]+gn.at(b,p,k)*n[b][i][p][k];
            atomicAdd(&dc[b][i][k], wi*grgb.at(b,p,k));
            dn[b][i][p][k]=wi*gn.at(b,p,k);
        }
        if ((mode&1) && l1[bp]>=0.f) adj+=gl1.at(b,p)*auxw[b][i][p];
        float raw=a[b][i][p], opacity=fminf(0.999f,fmaxf(0.f,raw));
        da[b][i][p]=(raw>=0.f && raw<=0.999f) ? prefix[b][i][p]*adj-suffix/(1.f-opacity) : 0.f;
        suffix+=wi*adj;
        float gd=wi*gz.at(b,p)+(mid[bp]==i ? gm.at(b,p) : 0.f);
        if ((mode&1) && l1[bp]>=0.f) gd+=gl1.at(b,p)*auxd[b][i][p];
        // Versioned L2 weights are detached: no L2 contribution to da/dc/dn.
        if ((mode&2) && l2[bp]>=0.f) gd+=gl2.at(b,p)*4.f*wi*(W*depth-D);
        dd[b][i][p]=gd;
    }
}

std::vector<torch::Tensor> selected_fwd(torch::Tensor a, torch::Tensor c, torch::Tensor d,
    torch::Tensor n, torch::Tensor order, int mode) {
    TORCH_CHECK(a.is_cuda() && a.scalar_type()==torch::kFloat32, "CUDA float32 required");
    c10::cuda::CUDAGuard guard(a.device());
    int B=a.size(0), N=a.size(1), P=a.size(2);
    auto opts=a.options();
    auto rgb=torch::zeros({B,P,3},opts), alpha=torch::zeros({B,P},opts), znum=torch::zeros({B,P},opts);
    auto nnum=torch::zeros({B,P,3},opts), l1=torch::zeros({B,P},opts), l2=torch::zeros({B,P},opts);
    auto median=torch::zeros({B,P},opts), weights=torch::zeros({B,N,P},opts), prefix=torch::zeros({B,N,P},opts);
    auto auxw=torch::zeros({B,(mode&1)?N:0,P},opts), auxd=torch::zeros_like(auxw);
    auto mid=torch::full({B,P},-1,opts.dtype(torch::kInt32));
    auto raw_l1=torch::zeros({B,P},opts),raw_l2=torch::zeros({B,P},opts);
    if (B*P) composite_fwd<<<(B*P+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        a.packed_accessor32<float,3,torch::RestrictPtrTraits>(), c.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        d.packed_accessor32<float,3,torch::RestrictPtrTraits>(), n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        order.packed_accessor32<int64_t,3,torch::RestrictPtrTraits>(), rgb.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        nnum.packed_accessor32<float,3,torch::RestrictPtrTraits>(), alpha.data_ptr<float>(), znum.data_ptr<float>(),
        l1.data_ptr<float>(), l2.data_ptr<float>(), median.data_ptr<float>(), weights.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        prefix.packed_accessor32<float,3,torch::RestrictPtrTraits>(), auxw.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        auxd.packed_accessor32<float,3,torch::RestrictPtrTraits>(), mid.data_ptr<int>(),raw_l1.data_ptr<float>(),raw_l2.data_ptr<float>(),N,P,B*P,mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rgb,alpha,znum,nnum,l1,l2,median,weights,prefix,auxw,auxd,mid,raw_l1,raw_l2};
}

std::vector<torch::Tensor> selected_bwd(torch::Tensor a, torch::Tensor c, torch::Tensor d, torch::Tensor n,
    torch::Tensor w, torch::Tensor prefix, torch::Tensor auxw, torch::Tensor auxd, torch::Tensor mid,
    torch::Tensor l1, torch::Tensor l2, c10::optional<torch::Tensor> grgb, c10::optional<torch::Tensor> ga,
    c10::optional<torch::Tensor> gz, c10::optional<torch::Tensor> gn, c10::optional<torch::Tensor> gl1,
    c10::optional<torch::Tensor> gl2, c10::optional<torch::Tensor> gm, c10::optional<torch::Tensor> gw, int mode) {
    c10::cuda::CUDAGuard guard(a.device());
    int B=a.size(0),N=a.size(1),P=a.size(2);
    auto da=torch::zeros(a.sizes(),a.options()),dc=torch::zeros(c.sizes(),c.options());
    auto dd=torch::zeros(d.sizes(),d.options()),dn=torch::zeros(n.sizes(),n.options());
    if (B*P) composite_bwd<<<(B*P+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(
        a.packed_accessor32<float,3,torch::RestrictPtrTraits>(),c.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),prefix.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        auxw.packed_accessor32<float,3,torch::RestrictPtrTraits>(),auxd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        mid.data_ptr<int>(),l1.data_ptr<float>(),l2.data_ptr<float>(),grad_view(grgb),grad_view(ga),grad_view(gz),grad_view(gn),
        grad_view(gl1),grad_view(gl2),grad_view(gm),grad_view(gw),da.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dc.packed_accessor32<float,3,torch::RestrictPtrTraits>(),dd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dn.packed_accessor32<float,4,torch::RestrictPtrTraits>(),N,P,B*P,mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {da,dc,dd,dn};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("forward",&selected_fwd); m.def("backward",&selected_bwd); }
