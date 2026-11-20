// Independent first-order event equations; original eager projection is external.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
using A2 = torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits>;
using A3 = torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits>;
using A4 = torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits>;
using B2 = torch::PackedTensorAccessor32<bool,2,torch::RestrictPtrTraits>;

__device__ float quadratic(float x, float y, A4 q, int b, int n) {
    float u=x*q[b][n][0][0]+y*q[b][n][1][0];
    float v=x*q[b][n][0][1]+y*q[b][n][1][1];
    return u*x+v*y;
}

__global__ void forward_kernel(A4 d,A4 q,A2 o,A2 len,A3 g,A2 rn,B2 valid,
    A3 alpha,A3 depth,int N,int P,int total,float near,float far,float extent2,const bool* support) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total)return;
    int p=i%P,n=(i/P)%N,b=i/(P*N);
    float x=d[b][n][p][0],y=d[b][n][p][1];
    float sq=quadratic(x,y,q,b,n);
    float z=(len[b][n]+(g[b][n][0]*x+g[b][n][1]*y))/rn[b][p];
    depth[b][n][p]=z;
    bool keep=support ? support[i] : (valid[b][n] && sq<=extent2 && z>near && z<far);
    alpha[b][n][p]=keep ? o[b][n]*expf(-0.5f*sq) : 0.f;
}

__global__ void backward_kernel(A4 d,A4 q,A2 o,A2 len,A3 g,A2 rn,B2 valid,A3 ga,A3 gd,
    A4 dd,A4 dq,A2 dop,A2 dl,A3 dg,int N,int P,int rows,float near,float far,float extent2,const bool* support,float* gs_out) {
    int row=blockIdx.x;
    if(row>=rows)return;
    int b=row/N,n=row%N,t=threadIdx.x;
    float acc[8]={0.f,0.f,0.f,0.f,0.f,0.f,0.f,0.f};
    for(int p=t;p<P;p+=blockDim.x){
        float x=d[b][n][p][0],y=d[b][n][p][1];
        float sq=quadratic(x,y,q,b,n);
        float z=(len[b][n]+(g[b][n][0]*x+g[b][n][1]*y))/rn[b][p];
        bool keep=support ? support[(b*N+n)*P+p] : (valid[b][n] && sq<=extent2 && z>near && z<far);
        float r=keep ? expf(-0.5f*sq) : 0.f;
        float gs=-0.5f*(ga[b][n][p]*o[b][n]*r),gz=gd[b][n][p]/rn[b][p];
        if(gs_out)gs_out[(b*N+n)*P+p]=gs;
        dd[b][n][p][0]=gs*((q[b][n][0][0]+q[b][n][0][0])*x+(q[b][n][0][1]+q[b][n][1][0])*y)+gz*g[b][n][0];
        dd[b][n][p][1]=gs*((q[b][n][1][0]+q[b][n][0][1])*x+(q[b][n][1][1]+q[b][n][1][1])*y)+gz*g[b][n][1];
        acc[0]+=gs*x*x; acc[1]+=gs*x*y;acc[2]+=gs*y*x;acc[3]+=gs*y*y;
        acc[4]+=ga[b][n][p]*r;acc[5]+=gz;acc[6]+=gz*x;acc[7]+=gz*y;
    }
    __shared__ float sums[8][256];
    for(int k=0;k<8;k++)sums[k][t]=acc[k];
    __syncthreads();
    for(int stride=blockDim.x/2;stride>0;stride/=2){
        if(t<stride)for(int k=0;k<8;k++)sums[k][t]+=sums[k][t+stride];
        __syncthreads();
    }
    if(t==0){
        for(int k=0;k<4;k++)dq[b][n][k/2][k%2]=sums[k][0];
        dop[b][n]=sums[4][0];dl[b][n]=sums[5][0];
        dg[b][n][0]=sums[6][0];dg[b][n][1]=sums[7][0];
    }
}

std::vector<torch::Tensor> forward(torch::Tensor d,torch::Tensor q,torch::Tensor o,
    torch::Tensor len,torch::Tensor g,torch::Tensor rn,torch::Tensor valid,double near,double far,double extent){
    c10::cuda::CUDAGuard guard(d.device());
    int B=d.size(0),N=d.size(1),P=d.size(2);auto alpha=torch::empty({B,N,P},d.options()),depth=torch::empty_like(alpha);
    if(B*N*P)forward_kernel<<<(B*N*P+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        d.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        o.packed_accessor32<float,2,torch::RestrictPtrTraits>(),len.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        g.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rn.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),alpha.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        depth.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N*P,near,far,extent*extent,nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {alpha,depth};
}

std::vector<torch::Tensor> backward(torch::Tensor d,torch::Tensor q,torch::Tensor o,
    torch::Tensor len,torch::Tensor g,torch::Tensor rn,torch::Tensor valid,torch::Tensor ga,torch::Tensor gd,double near,double far,double extent){
    c10::cuda::CUDAGuard guard(d.device());
    int B=d.size(0),N=d.size(1),P=d.size(2);auto dd=torch::empty_like(d),dq=torch::empty_like(q),dop=torch::empty_like(o),dl=torch::empty_like(len),dg=torch::empty_like(g);
    if(B*N)backward_kernel<<<B*N,256,0,at::cuda::getCurrentCUDAStream()>>>(
        d.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        o.packed_accessor32<float,2,torch::RestrictPtrTraits>(),len.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        g.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rn.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),ga.packed_accessor32<float,3,torch::RestrictPtrTraits>(),gd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dd.packed_accessor32<float,4,torch::RestrictPtrTraits>(),dq.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        dop.packed_accessor32<float,2,torch::RestrictPtrTraits>(),dl.packed_accessor32<float,2,torch::RestrictPtrTraits>(),dg.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N,near,far,extent*extent,nullptr,nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {dd,dq,dop,dl,dg};
}

void check_support(torch::Tensor d,torch::Tensor support){
    TORCH_CHECK(support.scalar_type()==torch::kBool && support.device()==d.device() && support.is_contiguous(),"live support must be same-device contiguous bool");
    TORCH_CHECK(support.dim()==3 && support.size(0)==d.size(0) && support.size(1)==d.size(1) && support.size(2)==d.size(2),"live support shape mismatch");
}

std::vector<torch::Tensor> forward_support(torch::Tensor d,torch::Tensor q,torch::Tensor o,
    torch::Tensor len,torch::Tensor g,torch::Tensor rn,torch::Tensor valid,torch::Tensor support){
    c10::cuda::CUDAGuard guard(d.device());check_support(d,support);
    int B=d.size(0),N=d.size(1),P=d.size(2);auto alpha=torch::empty({B,N,P},d.options()),depth=torch::empty_like(alpha);
    if(B*N*P)forward_kernel<<<(B*N*P+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        d.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        o.packed_accessor32<float,2,torch::RestrictPtrTraits>(),len.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        g.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rn.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),alpha.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        depth.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N*P,0.f,0.f,0.f,support.data_ptr<bool>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {alpha,depth};
}

std::vector<torch::Tensor> backward_support(torch::Tensor d,torch::Tensor q,torch::Tensor o,
    torch::Tensor len,torch::Tensor g,torch::Tensor rn,torch::Tensor valid,torch::Tensor ga,torch::Tensor gd,torch::Tensor support){
    c10::cuda::CUDAGuard guard(d.device());check_support(d,support);
    int B=d.size(0),N=d.size(1),P=d.size(2);auto dd=torch::empty_like(d),dq=torch::empty_like(q),dop=torch::empty_like(o),dl=torch::empty_like(len),dg=torch::empty_like(g);
    if(B*N)backward_kernel<<<B*N,256,0,at::cuda::getCurrentCUDAStream()>>>(
        d.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        o.packed_accessor32<float,2,torch::RestrictPtrTraits>(),len.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        g.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rn.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),ga.packed_accessor32<float,3,torch::RestrictPtrTraits>(),gd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dd.packed_accessor32<float,4,torch::RestrictPtrTraits>(),dq.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        dop.packed_accessor32<float,2,torch::RestrictPtrTraits>(),dl.packed_accessor32<float,2,torch::RestrictPtrTraits>(),dg.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N,0.f,0.f,0.f,support.data_ptr<bool>(),nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {dd,dq,dop,dl,dg};
}
// Readonly observation of the actual production gs; arithmetic and reduction
// are shared with normal calls above, which pass a null observation pointer.
std::vector<torch::Tensor> backward_support_gs(torch::Tensor d,torch::Tensor q,torch::Tensor o,
    torch::Tensor len,torch::Tensor g,torch::Tensor rn,torch::Tensor valid,torch::Tensor ga,torch::Tensor gd,torch::Tensor support){
    c10::cuda::CUDAGuard guard(d.device());check_support(d,support);
    int B=d.size(0),N=d.size(1),P=d.size(2);auto dd=torch::empty_like(d),dq=torch::empty_like(q),dop=torch::empty_like(o),dl=torch::empty_like(len),dg=torch::empty_like(g),gs=torch::empty_like(ga);
    if(B*N)backward_kernel<<<B*N,256,0,at::cuda::getCurrentCUDAStream()>>>(
        d.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        o.packed_accessor32<float,2,torch::RestrictPtrTraits>(),len.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        g.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rn.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),ga.packed_accessor32<float,3,torch::RestrictPtrTraits>(),gd.packed_accessor32<float,3,torch::RestrictPtrTraits>(),
        dd.packed_accessor32<float,4,torch::RestrictPtrTraits>(),dq.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        dop.packed_accessor32<float,2,torch::RestrictPtrTraits>(),dl.packed_accessor32<float,2,torch::RestrictPtrTraits>(),dg.packed_accessor32<float,3,torch::RestrictPtrTraits>(),N,P,B*N,0.f,0.f,0.f,support.data_ptr<bool>(),gs.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();return {dd,dq,dop,dl,dg,gs};
}
// Diagnostic output only: production forward/backward and quadratic are intact.
__global__ void inspect_kernel(A4 d,A4 q,A2 o,A2 len,A3 g,A2 rn,B2 valid,A2 out,
    int b,int p,int N,float near,float far,float extent2){
    int n=blockIdx.x*blockDim.x+threadIdx.x;if(n>=N)return;
    float x=d[b][n][p][0],y=d[b][n][p][1];
    float u=x*q[b][n][0][0]+y*q[b][n][1][0];
    float v=x*q[b][n][0][1]+y*q[b][n][1][1];
    float sq=quadratic(x,y,q,b,n);
    float z=(len[b][n]+(g[b][n][0]*x+g[b][n][1]*y))/rn[b][p];
    float response=expf(-0.5f*sq);
    bool keep=valid[b][n] && sq<=extent2 && z>near && z<far;
    out[n][0]=u;out[n][1]=v;out[n][2]=sq;out[n][3]=z;
    out[n][4]=valid[b][n];out[n][5]=sq<=extent2;out[n][6]=z>near;out[n][7]=z<far;
    out[n][8]=response;out[n][9]=keep?response:0.f;
    out[n][10]=keep?o[b][n]*response:0.f;
}

torch::Tensor inspect_pixel(torch::Tensor d,torch::Tensor q,torch::Tensor o,
    torch::Tensor len,torch::Tensor g,torch::Tensor rn,torch::Tensor valid,
    int b,int p,double near,double far,double extent){
    c10::cuda::CUDAGuard guard(d.device());
    TORCH_CHECK(b>=0 && b<d.size(0) && p>=0 && p<d.size(2),"inspection pixel outside original batch");
    int N=d.size(1);auto out=torch::empty({N,11},d.options());
    if(N)inspect_kernel<<<(N+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(
        d.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
        o.packed_accessor32<float,2,torch::RestrictPtrTraits>(),len.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        g.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rn.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        valid.packed_accessor32<bool,2,torch::RestrictPtrTraits>(),out.packed_accessor32<float,2,torch::RestrictPtrTraits>(),
        b,p,N,near,far,extent*extent);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);m.def("backward",&backward);m.def("inspect_pixel",&inspect_pixel);m.def("forward_support",&forward_support);m.def("backward_support",&backward_support);m.def("backward_support_gs",&backward_support_gs);}
