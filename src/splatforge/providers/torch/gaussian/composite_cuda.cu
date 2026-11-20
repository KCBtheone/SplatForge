// Independent implementation of the versioned alpha-compositing equations.
// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <cstring>
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
            // Preserve the accepted kernel's actual PTX arithmetic tree:
            // normal product rounds first, then RGB FMA, then the adjoint add.
            // Removing the color atomic let nvcc reverse which product fused,
            // changing da by ~1e-6 and amplified inverse-covariance VJPs.
            adj=__fadd_rn(adj,__fmaf_rn(grgb.at(b,p,k),c[b][i][k],
                __fmul_rn(gn.at(b,p,k),n[b][i][p][k])));
            dn[b][i][p][k]=wi*gn.at(b,p,k);
        }
        if ((mode&1) && !(mode&4) && l1[bp]>=0.f) adj+=gl1.at(b,p)*auxw[b][i][p];
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

// One warp owns one (batch, Gaussian, color-channel). The old pixel kernel
// issued three contended global atomics per event; all pixels in the tile
// targeted the same three addresses. Reduce locally and write once instead.
// No input/layout/precision change or hidden event pruning. This is an
// explicitly fingerprinted candidate until independent full-renderer VJPs pass.
__global__ void color_bwd(A3 w, Grad grgb, A3 dc, int N, int P, int rows) {
    int lane=threadIdx.x & 31;
    int row=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if (row>=rows) return;
    int k=row%3, i=(row/3)%N, b=row/(3*N);
    float sum=0.f;
    for (int p=lane; p<P; p+=32) sum+=w[b][i][p]*grgb.at(b,p,k);
    for (int offset=16; offset>0; offset/=2)
        sum+=__shfl_down_sync(0xffffffff, sum, offset);
    if (lane==0) dc[b][i][k]=sum;
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
    // Small tile batches often contain only 256 pixels. A 128-thread block
    // confines their long ordered Gaussian loops to two SMs. Warp-sized
    // blocks distribute those independent pixels over eight SMs instead,
    // without changing a pixel's arithmetic, ordering or termination rules.
    constexpr int pixel_threads=32;
    if (B*P) composite_fwd<<<(B*P+pixel_threads-1)/pixel_threads,pixel_threads,0,at::cuda::getCurrentCUDAStream()>>>(
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
    constexpr int pixel_threads=32;
    if (B*P) composite_bwd<<<(B*P+pixel_threads-1)/pixel_threads,pixel_threads,0,at::cuda::getCurrentCUDAStream()>>>(
        a.packed_accessor32<float,3,torch::RestrictPtrTraits>(),c.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),prefix.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        auxw.packed_accessor32<float,3,torch::RestrictPtrTraits>(),auxd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        mid.data_ptr<int>(),l1.data_ptr<float>(),l2.data_ptr<float>(),grad_view(grgb),grad_view(ga),grad_view(gz),grad_view(gn),
        grad_view(gl1),grad_view(gl2),grad_view(gm),grad_view(gw),da.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dc.packed_accessor32<float,3,torch::RestrictPtrTraits>(),dd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dn.packed_accessor32<float,4,torch::RestrictPtrTraits>(),N,P,B*P,mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (B*N && P) color_bwd<<<(B*N*3+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),grad_view(grgb),
        dc.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N*3);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {da,dc,dd,dn};
}

// H014 explicit candidate: original primals, warp-cooperative reverse suffix.
// This changes float32 reduction order and requires the original VJP gate.
__global__ void cooperative_bwd(A3 a, A3 c, A3 d, A4 n, A3 w, A3 prefix,
    A3 auxw, A3 auxd, const int* mid, const float* l1,
    Grad grgb, Grad ga, Grad gz, Grad gn, Grad gl1, Grad gm, Grad gw,
    A3 da, A3 dd, A4 dn, int N, int P, int BP) {
    int lane=threadIdx.x&31, bp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if (bp>=BP) return;
    int b=bp/P,p=bp%P;
    float carry=0.f;
    for (int top=N-1;top>=0;top-=32) {
        int i=top-lane;
        float wi=0.f,adj=0.f;
        if (i>=0) {
            wi=w[b][i][p];
            float depth=d[b][i][p];
            adj=ga.at(b,p)+gz.at(b,p)*depth+gw.at(b,i,p);
            for(int k=0;k<3;++k) {
                adj=__fadd_rn(adj,__fmaf_rn(grgb.at(b,p,k),c[b][i][k],
                    __fmul_rn(gn.at(b,p,k),n[b][i][p][k])));
                dn[b][i][p][k]=wi*gn.at(b,p,k);
            }
            if(l1[bp]>=0.f) adj+=gl1.at(b,p)*auxw[b][i][p];
            float gd=wi*gz.at(b,p)+(mid[bp]==i?gm.at(b,p):0.f);
            if(l1[bp]>=0.f) gd+=gl1.at(b,p)*auxd[b][i][p];
            dd[b][i][p]=gd;
        }
        float inclusive=__fmul_rn(wi,adj);
        for(int delta=1;delta<32;delta*=2) {
            float other=__shfl_up_sync(0xffffffff,inclusive,delta);
            if(lane>=delta) inclusive=__fadd_rn(inclusive,other);
        }
        float previous=__shfl_up_sync(0xffffffff,inclusive,1);
        float suffix=__fadd_rn(carry,lane?previous:0.f);
        if(i>=0) {
            float raw=a[b][i][p],opacity=fminf(0.999f,fmaxf(0.f,raw));
            da[b][i][p]=(raw>=0.f&&raw<=0.999f)?prefix[b][i][p]*adj-suffix/(1.f-opacity):0.f;
        }
        carry=__fadd_rn(carry,__shfl_sync(0xffffffff,inclusive,31));
    }
}

std::vector<torch::Tensor> cooperative_selected_bwd(torch::Tensor a,torch::Tensor c,torch::Tensor d,torch::Tensor n,
    torch::Tensor w,torch::Tensor prefix,torch::Tensor auxw,torch::Tensor auxd,torch::Tensor mid,
    torch::Tensor l1,torch::Tensor l2,c10::optional<torch::Tensor> grgb,c10::optional<torch::Tensor> ga,
    c10::optional<torch::Tensor> gz,c10::optional<torch::Tensor> gn,c10::optional<torch::Tensor> gl1,
    c10::optional<torch::Tensor> gl2,c10::optional<torch::Tensor> gm,c10::optional<torch::Tensor> gw,int mode) {
    TORCH_CHECK(mode==1,"H014 cooperative backward explicitly supports L1 only");
    c10::cuda::CUDAGuard guard(a.device());
    int B=a.size(0),N=a.size(1),P=a.size(2);
    auto da=torch::zeros_like(a),dc=torch::zeros_like(c),dd=torch::zeros_like(d),dn=torch::zeros_like(n);
    if(B*P) cooperative_bwd<<<(B*P+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(
        a.packed_accessor32<float,3,torch::RestrictPtrTraits>(),c.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),prefix.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        auxw.packed_accessor32<float,3,torch::RestrictPtrTraits>(),auxd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        mid.data_ptr<int>(),l1.data_ptr<float>(),grad_view(grgb),grad_view(ga),grad_view(gz),grad_view(gn),
        grad_view(gl1),grad_view(gm),grad_view(gw),da.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),dn.packed_accessor32<float,4,torch::RestrictPtrTraits>(),N,P,B*P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if(B*N&&P) color_bwd<<<(B*N*3+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),grad_view(grgb),
        dc.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N*3);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {da,dc,dd,dn};
}

__device__ float warp_total(float value) {
    for(int delta=16;delta>0;delta/=2)
        value=__fadd_rn(value,__shfl_down_sync(0xffffffff,value,delta));
    return __shfl_sync(0xffffffff,value,0);
}

// H015: explicit float32 scan reassociation, original Torch depth order.
__global__ void cooperative_fwd(A3 a,A3 c,A3 d,A4 n,I3 order,
    A3 rgb,A3 nnum,float* alpha,float* znum,float* l1,float* l2,
    float* median,A3 weights,A3 prefix,A3 auxw,A3 auxd,int* median_id,
    float* raw_l1,float* raw_l2,int N,int P,int BP) {
    int lane=threadIdx.x&31,bp=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(bp>=BP)return;
    int b=bp/P,p=bp%P,mid=-1;
    float T=1.f,W=0.f,D=0.f,C[3]={0.f,0.f,0.f},M[3]={0.f,0.f,0.f};
    for(int top=0;top<N;top+=32) {
        int i=top+lane;
        float op=i<N?fminf(.999f,fmaxf(0.f,a[b][i][p])):0.f;
        float product=1.f-op;
        for(int delta=1;delta<32;delta*=2) {
            float other=__shfl_up_sync(0xffffffff,product,delta);
            if(lane>=delta)product=__fmul_rn(product,other);
        }
        float previous=__shfl_up_sync(0xffffffff,product,1);
        float transmission=__fmul_rn(T,lane?previous:1.f);
        float wi=__fmul_rn(op,transmission),inclusive=wi;
        for(int delta=1;delta<32;delta*=2) {
            float other=__shfl_up_sync(0xffffffff,inclusive,delta);
            if(lane>=delta)inclusive=__fadd_rn(inclusive,other);
        }
        unsigned crossed=__ballot_sync(0xffffffff,i<N&&__fadd_rn(W,inclusive)>=.5f);
        if(mid<0&&crossed)mid=top+__ffs(crossed)-1;
        float depth=i<N?d[b][i][p]:0.f;
        if(i<N){prefix[b][i][p]=transmission;weights[b][i][p]=wi;}
        for(int k=0;k<3;++k) {
            C[k]+=warp_total(i<N?wi*c[b][i][k]:0.f);
            M[k]+=warp_total(i<N?wi*n[b][i][p][k]:0.f);
        }
        W+=__shfl_sync(0xffffffff,inclusive,31);
        D+=warp_total(wi*depth);
        T*=__shfl_sync(0xffffffff,product,31);
    }
    if(lane==0) {
        alpha[bp]=W;znum[bp]=D;median_id[bp]=mid;
        median[bp]=mid<0?0.f:d[b][mid][p];
        raw_l2[bp]=0.f;l2[bp]=0.f;
        for(int k=0;k<3;++k){rgb[b][p][k]=C[k];nnum[b][p][k]=M[k];}
    }
    __syncwarp(); // Sorted pass reads weights written by other lanes.
    float carryw=0.f,carryd=0.f,distortion=0.f;
    for(int top=0;top<N;top+=32) {
        int rank=top+lane,i=rank<N?order[b][rank][p]:0;
        float wi=rank<N?weights[b][i][p]:0.f,depth=rank<N?d[b][i][p]:0.f;
        float pw=wi,pd=wi*depth;
        for(int delta=1;delta<32;delta*=2) {
            float ow=__shfl_up_sync(0xffffffff,pw,delta),od=__shfl_up_sync(0xffffffff,pd,delta);
            if(lane>=delta){pw=__fadd_rn(pw,ow);pd=__fadd_rn(pd,od);}
        }
        float beforew=__shfl_up_sync(0xffffffff,pw,1),befored=__shfl_up_sync(0xffffffff,pd,1);
        float sw=carryw+(lane?beforew:0.f),sd=carryd+(lane?befored:0.f);
        float term=rank<N?2.f*wi*(depth*sw-sd):0.f;
        if(rank<N) {
            auxw[b][i][p]=2.f*(2.f*depth*sw-depth*W+D-2.f*sd);
            auxd[b][i][p]=2.f*wi*(2.f*sw+wi-W);
        }
        distortion+=warp_total(term);
        carryw+=__shfl_sync(0xffffffff,pw,31);carryd+=__shfl_sync(0xffffffff,pd,31);
    }
    if(lane==0){raw_l1[bp]=distortion;l1[bp]=fmaxf(0.f,distortion);}
}

std::vector<torch::Tensor> cooperative_selected_fwd(torch::Tensor a,torch::Tensor c,torch::Tensor d,
    torch::Tensor n,torch::Tensor order,int mode) {
    TORCH_CHECK(mode==1,"H015 cooperative forward explicitly supports L1 only");
    TORCH_CHECK(a.is_cuda()&&a.scalar_type()==torch::kFloat32,"CUDA float32 required");
    c10::cuda::CUDAGuard guard(a.device());
    int B=a.size(0),N=a.size(1),P=a.size(2);auto opts=a.options();
    auto rgb=torch::zeros({B,P,3},opts),alpha=torch::zeros({B,P},opts),znum=torch::zeros({B,P},opts);
    auto nnum=torch::zeros({B,P,3},opts),l1=torch::zeros({B,P},opts),l2=torch::zeros({B,P},opts);
    auto median=torch::zeros({B,P},opts),weights=torch::zeros({B,N,P},opts),prefix=torch::zeros({B,N,P},opts);
    auto auxw=torch::zeros({B,N,P},opts),auxd=torch::zeros_like(auxw);
    auto mid=torch::full({B,P},-1,opts.dtype(torch::kInt32));
    auto raw_l1=torch::zeros({B,P},opts),raw_l2=torch::zeros({B,P},opts);
    if(B*P)cooperative_fwd<<<(B*P+7)/8,256,0,at::cuda::getCurrentCUDAStream()>>>(
        a.packed_accessor32<float,3,torch::RestrictPtrTraits>(),c.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        order.packed_accessor32<int64_t,3,torch::RestrictPtrTraits>(),rgb.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        nnum.packed_accessor32<float,3,torch::RestrictPtrTraits>(),alpha.data_ptr<float>(),znum.data_ptr<float>(),
        l1.data_ptr<float>(),l2.data_ptr<float>(),median.data_ptr<float>(),weights.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        prefix.packed_accessor32<float,3,torch::RestrictPtrTraits>(),auxw.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        auxd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),mid.data_ptr<int>(),raw_l1.data_ptr<float>(),raw_l2.data_ptr<float>(),N,P,B*P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {rgb,alpha,znum,nnum,l1,l2,median,weights,prefix,auxw,auxd,mid,raw_l1,raw_l2};
}

// Explicit joint GOF handoff candidate. Original forward and matrix VJPs stay
// outside this boundary. No scan, support pruning, precision change or fallback.
using Bool3 = torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits>;
__global__ void joint_gof_bwd(A3 a,A3 c,A3 d,A4 n,A3 w,A3 prefix,A3 auxw,A3 auxd,
    const int* mid,const float* l1,A3 length,torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> opacity,
    A3 response,A3 exponent,Bool3 mask,Grad grgb,Grad ga,Grad gz,Grad gn,Grad gl1,Grad gm,Grad gw,
    A4 gq,A3 gd,A3 ge,torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> go,int N,int P,int BP) {
    int bp=blockIdx.x*blockDim.x+threadIdx.x;if(bp>=BP)return;
    int b=bp/P,p=bp%P;float suffix=0.f;
    for(int i=N-1;i>=0;--i){
        float wi=w[b][i][p],depth=d[b][i][p];
        float adj=ga.at(b,p)+gz.at(b,p)*depth+gw.at(b,i,p);
        float ng[3],ndot=0.f;
        for(int k=0;k<3;++k){
            adj=__fadd_rn(adj,__fmaf_rn(grgb.at(b,p,k),c[b][i][k],
                __fmul_rn(gn.at(b,p,k),n[b][i][p][k])));
            ng[k]=wi*gn.at(b,p,k);
            ndot=__fadd_rn(ndot,__fmul_rn(ng[k],n[b][i][p][k]));
        }
        if(l1[bp]>=0.f)adj+=gl1.at(b,p)*auxw[b][i][p];
        float raw=a[b][i][p],op=fminf(.999f,fmaxf(0.f,raw));
        float da=(raw>=0.f&&raw<=.999f)?prefix[b][i][p]*adj-suffix/(1.f-op):0.f;
        suffix+=wi*adj;
        float dd=wi*gz.at(b,p)+(mid[bp]==i?gm.at(b,p):0.f);
        if(l1[bp]>=0.f)dd+=gl1.at(b,p)*auxd[b][i][p];
        gd[b][i][p]=dd;
        float norm=length[b][i][p],den=fmaxf(norm,1e-12f);
        for(int k=0;k<3;++k){
            float correction=norm>=1e-12f?__fmul_rn(n[b][i][p][k],ndot):0.f;
            gq[b][i][p][k]=-__fsub_rn(ng[k],correction)/den;
        }
        float gr=mask[b][i][p]?__fmul_rn(da,opacity[b][i]):0.f;
        ge[b][i][p]=exponent[b][i][p]>=0.f?
            __fmul_rn(__fmul_rn(gr,response[b][i][p]),-.5f):0.f;
        float opgrad=__fmul_rn(da,mask[b][i][p]?response[b][i][p]:0.f);
        // All32 lanes are real pixels of one batch (P is a multiple of32).
        // One atomic per warp/member, not one contended atomic per event.
        for(int offset=16;offset>0;offset/=2)opgrad+=__shfl_down_sync(0xffffffff,opgrad,offset);
        if((threadIdx.x&31)==0)atomicAdd(&go[b][i],opgrad);
    }
}

std::vector<torch::Tensor> joint_gof_backward(torch::Tensor a,torch::Tensor c,torch::Tensor d,torch::Tensor n,
    torch::Tensor w,torch::Tensor prefix,torch::Tensor auxw,torch::Tensor auxd,torch::Tensor mid,
    torch::Tensor l1,torch::Tensor l2,torch::Tensor length,torch::Tensor opacity,torch::Tensor response,
    torch::Tensor exponent,torch::Tensor mask,c10::optional<torch::Tensor> grgb,c10::optional<torch::Tensor> ga,
    c10::optional<torch::Tensor> gz,c10::optional<torch::Tensor> gn,c10::optional<torch::Tensor> gl1,
    c10::optional<torch::Tensor> gl2,c10::optional<torch::Tensor> gm,c10::optional<torch::Tensor> gw) {
    c10::cuda::CUDAGuard guard(a.device());int B=a.size(0),N=a.size(1),P=a.size(2);
    TORCH_CHECK(a.is_cuda()&&a.scalar_type()==torch::kFloat32&&P%32==0,
        "joint GOF candidate requires CUDA float32 and complete32pixel warps; no fallback");
    TORCH_CHECK(length.sizes()==a.sizes()&&response.sizes()==a.sizes()&&exponent.sizes()==a.sizes()
        &&mask.sizes()==a.sizes()&&mask.scalar_type()==torch::kBool,"joint GOF handoff shapes");
    auto gq=torch::empty(n.sizes(),n.options()),gd=torch::empty_like(d),ge=torch::empty_like(a);
    auto go=torch::zeros_like(opacity),gc=torch::empty(c.sizes(),c.options());
    if(B*P)joint_gof_bwd<<<(B*P+31)/32,32,0,at::cuda::getCurrentCUDAStream()>>>(
        a.packed_accessor32<float,3,torch::RestrictPtrTraits>(),c.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),prefix.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        auxw.packed_accessor32<float,3,torch::RestrictPtrTraits>(),auxd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        mid.data_ptr<int>(),l1.data_ptr<float>(),length.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        opacity.packed_accessor32<float,2,torch::RestrictPtrTraits>(),response.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        exponent.packed_accessor32<float,3,torch::RestrictPtrTraits>(),mask.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
        grad_view(grgb),grad_view(ga),grad_view(gz),grad_view(gn),grad_view(gl1),grad_view(gm),grad_view(gw),
        gq.packed_accessor32<float,4,torch::RestrictPtrTraits>(),gd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        ge.packed_accessor32<float,3,torch::RestrictPtrTraits>(),go.packed_accessor32<float,2,torch::RestrictPtrTraits>(),N,P,B*P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if(B*N&&P)color_bwd<<<(B*N*3+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),grad_view(grgb),
        gc.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N*3);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {gq,gd,ge,go,gc};
}


struct GroupedForwardTask {
    A3 a,c,d; A4 n; I3 order;
    A3 rgb,nnum; float *alpha,*znum,*l1,*l2,*median;
    A3 weights,prefix,auxw,auxd; int* median_id;
    float *raw_l1,*raw_l2; int N,P,BP,offset;
};

// Explicit experiment only: retain original single-pixel arithmetic verbatim.
// Different N/P/strides remain separate; no tensor B concatenation or scan change.
__global__ void grouped_composite_fwd(const GroupedForwardTask* tasks,int count,int total,int mode) {
    int global_bp=blockIdx.x*blockDim.x+threadIdx.x;
    if(global_bp>=total)return;
    int task=0;
    while(task+1<count && global_bp>=tasks[task+1].offset)++task;
    const auto& t=tasks[task]; int bp=global_bp-t.offset,N=t.N,P=t.P;
    auto a=t.a,c=t.c,d=t.d; auto n=t.n; auto order=t.order;
    auto rgb=t.rgb,nnum=t.nnum,weights=t.weights,prefix=t.prefix,auxw=t.auxw,auxd=t.auxd;
    auto alpha=t.alpha,znum=t.znum,l1=t.l1,l2=t.l2,median=t.median;
    auto median_id=t.median_id;auto raw_l1=t.raw_l1,raw_l2=t.raw_l2;
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

std::vector<std::vector<torch::Tensor>> grouped_selected_fwd(
    std::vector<std::vector<torch::Tensor>> inputs,int mode) {
    TORCH_CHECK(!inputs.empty() && mode==1,"fixed ragged GOF L1 forward required");
    c10::cuda::CUDAGuard guard(inputs[0][0].device());
    std::vector<GroupedForwardTask> tasks;
    std::vector<std::vector<torch::Tensor>> results;
    int total=0;
    for(auto& values:inputs) {
        TORCH_CHECK(values.size()==5,"alpha/color/depth/normal/original order required");
        auto a=values[0],c=values[1],d=values[2],n=values[3],order=values[4];
        TORCH_CHECK(a.is_cuda() && a.scalar_type()==torch::kFloat32 && a.device()==inputs[0][0].device(),"same CUDA float32 device required");
        int B=a.size(0),N=a.size(1),P=a.size(2);auto opts=a.options();
        auto rgb=torch::zeros({B,P,3},opts),alpha=torch::zeros({B,P},opts),znum=torch::zeros({B,P},opts);
        auto nnum=torch::zeros({B,P,3},opts),l1=torch::zeros({B,P},opts),l2=torch::zeros({B,P},opts);
        auto median=torch::zeros({B,P},opts),weights=torch::zeros({B,N,P},opts),prefix=torch::zeros({B,N,P},opts);
        auto auxw=torch::zeros({B,N,P},opts),auxd=torch::zeros_like(auxw);
        auto mid=torch::full({B,P},-1,opts.dtype(torch::kInt32)),raw_l1=torch::zeros({B,P},opts),raw_l2=torch::zeros({B,P},opts);
        tasks.push_back(GroupedForwardTask{
            a.packed_accessor32<float,3,torch::RestrictPtrTraits>(),c.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
            order.packed_accessor32<int64_t,3,torch::RestrictPtrTraits>(),
            rgb.packed_accessor32<float,3,torch::RestrictPtrTraits>(),nnum.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            alpha.data_ptr<float>(),znum.data_ptr<float>(),l1.data_ptr<float>(),l2.data_ptr<float>(),median.data_ptr<float>(),
            weights.packed_accessor32<float,3,torch::RestrictPtrTraits>(),prefix.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            auxw.packed_accessor32<float,3,torch::RestrictPtrTraits>(),auxd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
            mid.data_ptr<int>(),raw_l1.data_ptr<float>(),raw_l2.data_ptr<float>(),N,P,B*P,total});
        total+=B*P;
        results.push_back({rgb,alpha,znum,nnum,l1,l2,median,weights,prefix,auxw,auxd,mid,raw_l1,raw_l2});
    }
    auto host=torch::empty({int64_t(tasks.size()*sizeof(GroupedForwardTask))},torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU));
    std::memcpy(host.data_ptr(),tasks.data(),tasks.size()*sizeof(GroupedForwardTask));
    auto device=host.to(inputs[0][0].device());
    if(total)grouped_composite_fwd<<<(total+31)/32,32,0,at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const GroupedForwardTask*>(device.data_ptr()),tasks.size(),total,mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return results;
}

// Diagnostic alternative: keep the dense event graph and only omit reverse
// arithmetic at slots disconnected by the original projected-support mask.
// The original kernels and their entrypoints above remain unchanged.
__global__ void support_masked_bwd(A3 a,A3 c,A3 d,A4 n,A3 w,A3 prefix,
    A3 auxw,A3 auxd,const int* mid,const float* l1,const float* l2,
    torch::PackedTensorAccessor32<bool,3,torch::RestrictPtrTraits> support,
    Grad grgb,Grad ga,Grad gz,Grad gn,Grad gl1,Grad gl2,Grad gm,Grad gw,
    A3 da,A3 dd,A4 dn,int N,int P,int BP,int mode) {
    int bp=blockIdx.x*blockDim.x+threadIdx.x;
    if(bp>=BP)return;
    int b=bp/P,p=bp%P;
    float W=0.f,D=0.f;
    if(mode&2)for(int i=0;i<N;++i){W+=w[b][i][p];D+=w[b][i][p]*d[b][i][p];}
    float suffix=0.f;
    for(int i=N-1;i>=0;--i){
        if(!support[b][i][p])continue;
        float wi=w[b][i][p],depth=d[b][i][p];
        float adj=ga.at(b,p)+gz.at(b,p)*depth+gw.at(b,i,p);
        for(int k=0;k<3;++k){
            adj=__fadd_rn(adj,__fmaf_rn(grgb.at(b,p,k),c[b][i][k],
                __fmul_rn(gn.at(b,p,k),n[b][i][p][k])));
            dn[b][i][p][k]=wi*gn.at(b,p,k);
        }
        if((mode&1)&&l1[bp]>=0.f)adj+=gl1.at(b,p)*auxw[b][i][p];
        float raw=a[b][i][p],opacity=fminf(0.999f,fmaxf(0.f,raw));
        da[b][i][p]=(raw>=0.f&&raw<=0.999f)?prefix[b][i][p]*adj-suffix/(1.f-opacity):0.f;
        suffix+=wi*adj;
        float gd=wi*gz.at(b,p)+(mid[bp]==i?gm.at(b,p):0.f);
        if((mode&1)&&l1[bp]>=0.f)gd+=gl1.at(b,p)*auxd[b][i][p];
        if((mode&2)&&l2[bp]>=0.f)gd+=gl2.at(b,p)*4.f*wi*(W*depth-D);
        dd[b][i][p]=gd;
    }
}

std::vector<torch::Tensor> support_selected_bwd(std::vector<torch::Tensor> v,
    std::vector<c10::optional<torch::Tensor>> g,torch::Tensor support,int mode){
    TORCH_CHECK(v.size()==11&&g.size()==8,"original11 saved tensors/8 output adjoints required");
    auto a=v[0],c=v[1],d=v[2],n=v[3],w=v[4];
    TORCH_CHECK(support.is_cuda()&&support.scalar_type()==torch::kBool&&support.sizes()==a.sizes(),"original projected support shape required");
    c10::cuda::CUDAGuard guard(a.device());
    int B=a.size(0),N=a.size(1),P=a.size(2);
    auto da=torch::zeros(a.sizes(),a.options()),dc=torch::zeros(c.sizes(),c.options());
    auto dd=torch::zeros(d.sizes(),d.options()),dn=torch::zeros(n.sizes(),n.options());
    if(B*P)support_masked_bwd<<<(B*P+31)/32,32,0,at::cuda::getCurrentCUDAStream()>>>(
        a.packed_accessor32<float,3,torch::RestrictPtrTraits>(),c.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        d.packed_accessor32<float,3,torch::RestrictPtrTraits>(),n.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v[5].packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        v[6].packed_accessor32<float,3,torch::RestrictPtrTraits>(),v[7].packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        v[8].data_ptr<int>(),v[9].data_ptr<float>(),v[10].data_ptr<float>(),
        support.packed_accessor32<bool,3,torch::RestrictPtrTraits>(),
        grad_view(g[0]),grad_view(g[1]),grad_view(g[2]),grad_view(g[3]),
        grad_view(g[4]),grad_view(g[5]),grad_view(g[6]),grad_view(g[7]),
        da.packed_accessor32<float,3,torch::RestrictPtrTraits>(),dd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dn.packed_accessor32<float,4,torch::RestrictPtrTraits>(),N,P,B*P,mode);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if(B*N&&P)color_bwd<<<(B*N*3+3)/4,128,0,at::cuda::getCurrentCUDAStream()>>>(
        w.packed_accessor32<float,3,torch::RestrictPtrTraits>(),grad_view(g[0]),
        dc.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N*3);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {da,dc,dd,dn};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
    m.def("backward_support_masked",&support_selected_bwd);
    m.def("forward_grouped",&grouped_selected_fwd);
    m.def("forward",&selected_fwd); m.def("backward",&selected_bwd);
    m.def("backward_cooperative",&cooperative_selected_bwd);
    m.def("forward_cooperative",&cooperative_selected_fwd);
    m.def("joint_gof_backward",&joint_gof_backward);
}
