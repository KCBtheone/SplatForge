// SPDX-License-Identifier: Apache-2.0
// Conservative binary64 projection separation. Uncertain signs are retained
// for the unchanged exact CPU predicate. Compile with --fmad=false, no fast math.
#include <math.h>
__device__ int certified(const double* a,const double* b,const double* c,int x,int y){
    double ax=a[x]-c[x],ay=a[y]-c[y],bx=b[x]-c[x],by=b[y]-c[y];
    double l=ax*by,r=ay*bx,d=l-r,p=fabs(l)+fabs(r);
    if(isfinite(p)&&p>1e-280&&fabs(d)>p*2e-15)return d>0?1:-1;
    return 0; // unknown, including exact zero
}
__device__ bool separate(const double* v,const long long* a,const long long* b){
    for(int drop=0;drop<3;++drop){
        int x=(drop+1)%3,y=(drop+2)%3;
        for(int edge=0;edge<3;++edge){
            long long i=a[edge],j=a[(edge+1)%3];
            int side=certified(v+3*i,v+3*j,v+3*a[(edge+2)%3],x,y);
            if(!side)continue;
            bool disjoint=true;
            for(int k=0;k<3;++k){
                // Only the SAME indexed endpoint is a permitted contact.
                if(b[k]==i||b[k]==j)continue;
                if(certified(v+3*i,v+3*j,v+3*b[k],x,y)!=-side){disjoint=false;break;}
            }
            if(disjoint)return true;
        }
    }
    return false;
}
extern "C" __global__ void filter_pairs(const double* v,const long long* faces,
    const long long* pairs,unsigned char* keep,unsigned long long count){
    unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=count)return;
    const long long* a=faces+3*pairs[2*i];const long long* b=faces+3*pairs[2*i+1];
    keep[i]=!(separate(v,a,b)||separate(v,b,a));
}
