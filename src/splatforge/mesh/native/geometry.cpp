// SPDX-License-Identifier: Apache-2.0
// Owned binary64-exact predicates and triangle BVH. No external geometry code.
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>
namespace py = pybind11;
using V = std::array<double,3>;
using F = std::array<int64_t,3>;
#include "expansion.h"

// Exact signed dyadic integers: sign * little-endian magnitude * 2^exponent.
// Used only when the floating-point filter cannot certify the determinant.
struct D {
    std::vector<uint32_t> w; int sign=0, exponent=0;
    D() = default;
    explicit D(double x) {
        uint64_t u; std::memcpy(&u,&x,8);
        uint64_t m=u&((uint64_t(1)<<52)-1); int e=int((u>>52)&2047);
        if(e) m|=uint64_t(1)<<52;
        if(!m) return;
        sign=(u>>63)?-1:1; exponent=e?e-1023-52:-1074;
        w={uint32_t(m),uint32_t(m>>32)}; trim();
    }
    void trim(){while(!w.empty() && !w.back())w.pop_back(); if(w.empty())sign=0;}
};
static std::vector<uint32_t> shift(const std::vector<uint32_t>& a,int bits){
    if(a.empty())return {};
    std::vector<uint32_t> r(a.size()+bits/32+1,0); uint64_t carry=0;
    for(size_t i=0;i<a.size();++i){uint64_t x=(uint64_t(a[i])<<(bits%32))|carry;
        r[i+bits/32]=uint32_t(x);carry=x>>32;}
    r[a.size()+bits/32]=uint32_t(carry);
    while(!r.empty()&&!r.back())r.pop_back();return r;
}
static int cmp(const std::vector<uint32_t>& a,const std::vector<uint32_t>& b){
    if(a.size()!=b.size())return a.size()>b.size()?1:-1;
    for(size_t i=a.size();i-->0;)if(a[i]!=b[i])return a[i]>b[i]?1:-1;return 0;
}
static D add(const D& a,const D& b){
    if(!a.sign)return b;if(!b.sign)return a;D r;
    r.exponent=std::min(a.exponent,b.exponent);
    auto aw=shift(a.w,a.exponent-r.exponent),bw=shift(b.w,b.exponent-r.exponent);
    if(a.sign==b.sign){r.sign=a.sign;size_t n=std::max(aw.size(),bw.size());r.w.resize(n+1);uint64_t carry=0;
        for(size_t i=0;i<n;++i){uint64_t s=carry+(i<aw.size()?aw[i]:0)+(uint64_t)(i<bw.size()?bw[i]:0);r.w[i]=uint32_t(s);carry=s>>32;}r.w[n]=uint32_t(carry);
    }else{
        int c=cmp(aw,bw);if(!c)return D();r.sign=c>0?a.sign:b.sign;
        if(c<0)aw.swap(bw);r.w.resize(aw.size());uint64_t borrow=0;
        for(size_t i=0;i<aw.size();++i){uint64_t s=(i<bw.size()?uint64_t(bw[i]):0)+borrow;
            r.w[i]=uint32_t(uint64_t(aw[i])-s);borrow=uint64_t(aw[i])<s;}
    }r.trim();return r;
}
static D neg(D x){x.sign=-x.sign;return x;}
static D sub(const D&a,const D&b){return add(a,neg(b));}
static D mul(const D&a,const D&b){
    if(!a.sign||!b.sign)return D();D r;r.sign=a.sign*b.sign;r.exponent=a.exponent+b.exponent;r.w.resize(a.w.size()+b.w.size(),0);
    for(size_t i=0;i<a.w.size();++i){uint64_t carry=0;
        for(size_t j=0;j<b.w.size();++j){uint64_t s=uint64_t(a.w[i])*b.w[j]+r.w[i+j]+carry;r.w[i+j]=uint32_t(s);carry=s>>32;}
        r.w[i+b.w.size()]=uint32_t(carry);
    }r.trim();return r;
}
static int o2(const V&a,const V&b,const V&c,int x,int y){
    if((a[x]==c[x]&&a[y]==c[y])||(b[x]==c[x]&&b[y]==c[y])||(a[x]==b[x]&&a[y]==b[y]))return 0;
    if((a[x]==b[x]&&a[x]==c[x])||(a[y]==b[y]&&a[y]==c[y]))return 0;
    double ax=a[x]-c[x],ay=a[y]-c[y],bx=b[x]-c[x],by=b[y]-c[y];
    double l=ax*by,r=ay*bx,d=l-r,p=std::abs(l)+std::abs(r);
    if(std::isfinite(p)&&p>1e-280&&std::abs(d)>p*2e-15)return d>0?1:-1;
    if(ordinary(a)&&ordinary(b)&&ordinary(c))return exact2_expansion(a,b,c,x,y);
    return sub(mul(sub(D(a[x]),D(c[x])),sub(D(b[y]),D(c[y]))),mul(sub(D(a[y]),D(c[y])),sub(D(b[x]),D(c[x])))).sign;
}
static int o3(const V&a,const V&b,const V&c,const V&d){
    if(a==b||a==c||a==d||b==c||b==d||c==d)return 0;
    for(int k=0;k<3;++k)if(a[k]==b[k]&&a[k]==c[k]&&a[k]==d[k])return 0;
    double ax=a[0]-d[0],ay=a[1]-d[1],az=a[2]-d[2];
    double bx=b[0]-d[0],by=b[1]-d[1],bz=b[2]-d[2];
    double cx=c[0]-d[0],cy=c[1]-d[1],cz=c[2]-d[2];
    double v=ax*(by*cz-bz*cy)+ay*(bz*cx-bx*cz)+az*(bx*cy-by*cx);
    double p=std::abs(ax)*(std::abs(by*cz)+std::abs(bz*cy))+std::abs(ay)*(std::abs(bz*cx)+std::abs(bx*cz))+std::abs(az)*(std::abs(bx*cy)+std::abs(by*cx));
    if(std::isfinite(p)&&p>1e-260&&std::abs(v)>p*8e-15)return v>0?1:-1;
    if(ordinary(a)&&ordinary(b)&&ordinary(c)&&ordinary(d))return exact3_expansion(a,b,c,d);
    D aa[3],bb[3],cc[3];for(int k=0;k<3;++k){aa[k]=sub(D(a[k]),D(d[k]));bb[k]=sub(D(b[k]),D(d[k]));cc[k]=sub(D(c[k]),D(d[k]));}
    D result;
    for(int k=0;k<3;++k){int j=(k+1)%3,l=(k+2)%3;result=add(result,mul(aa[k],sub(mul(bb[j],cc[l]),mul(bb[l],cc[j]))));}
    return result.sign;
}
static int projection(const V&a,const V&b,const V&c){
    // Choose a provably nonzero projected area, preferring the largest.
    double n[3];for(int k=0;k<3;++k){int j=(k+1)%3,l=(k+2)%3;n[k]=std::abs((b[j]-a[j])*(c[l]-a[l])-(b[l]-a[l])*(c[j]-a[j]));}
    int best=int(std::max_element(n,n+3)-n);
    for(int i=0;i<3;++i){int k=(best+i)%3;if(o2(a,b,c,(k+1)%3,(k+2)%3))return k;}return -1;
}
static bool inside2(const V&p,const V&a,const V&b,const V&c,int k){
    int x=(k+1)%3,y=(k+2)%3;
    int s=o2(a,b,p,x,y),t=o2(b,c,p,x,y),u=o2(c,a,p,x,y);
    return (s>=0&&t>=0&&u>=0)||(s<=0&&t<=0&&u<=0);
}
static bool between(const V&p,const V&a,const V&b,int k){
    int x=(k+1)%3,y=(k+2)%3;return p[x]>=std::min(a[x],b[x])&&p[x]<=std::max(a[x],b[x])&&p[y]>=std::min(a[y],b[y])&&p[y]<=std::max(a[y],b[y]);
}
static bool edge2(const V&a,const V&b,const V&c,const V&d,int64_t ai,int64_t bi,int64_t ci,int64_t di,int k){
    int x=(k+1)%3,y=(k+2)%3;
    int p=o2(a,b,c,x,y),q=o2(a,b,d,x,y),r=o2(c,d,a,x,y),s=o2(c,d,b,x,y);
    if(p*q<0&&r*s<0)return true;
    // Contact at the same indexed vertex is permitted; any other contact is not.
    if(!p&&between(c,a,b,k)&&ci!=ai&&ci!=bi)return true;
    if(!q&&between(d,a,b,k)&&di!=ai&&di!=bi)return true;
    if(!r&&between(a,c,d,k)&&ai!=ci&&ai!=di)return true;
    if(!s&&between(b,c,d,k)&&bi!=ci&&bi!=di)return true;
    return false;
}
static bool segment_triangle(const V&p,const V&q,int64_t pi,int64_t qi,const V*a,const F&f,int k){
    int sp=o3(a[0],a[1],a[2],p),sq=o3(a[0],a[1],a[2],q);
    if(sp*sq>0)return false;
    auto shared=[&](int64_t i){return i==f[0]||i==f[1]||i==f[2];};
    if(!sp&&!sq){
        if(!shared(pi)&&inside2(p,a[0],a[1],a[2],k))return true;
        if(!shared(qi)&&inside2(q,a[0],a[1],a[2],k))return true;
        for(int j=0;j<3;++j)if(edge2(p,q,a[j],a[(j+1)%3],pi,qi,f[j],f[(j+1)%3],k))return true;
        return false;
    }
    if(!sp)return !shared(pi)&&inside2(p,a[0],a[1],a[2],k);
    if(!sq)return !shared(qi)&&inside2(q,a[0],a[1],a[2],k);
    int s=o3(p,q,a[0],a[1]),t=o3(p,q,a[1],a[2]),u=o3(p,q,a[2],a[0]);
    return (s>=0&&t>=0&&u>=0)||(s<=0&&t<=0&&u<=0);
}
static bool intersect(const V*a,const V*b,const F&fa,const F&fb){
    int shared=0;for(auto i:fa)for(auto j:fb)shared+=i==j;
    if(shared==3)return true;
    if(shared==2){
        int p=-1,q=-1,ua=-1,ub=-1;
        for(int i=0;i<3;++i){bool found=false;for(int j=0;j<3;++j)if(fa[i]==fb[j]){found=true;if(p<0)p=i;else q=i;}if(!found)ua=i;}
        for(int j=0;j<3;++j)if(fb[j]!=fa[p]&&fb[j]!=fa[q])ub=j;
        if(o3(a[0],a[1],a[2],b[ub]))return false;
        int k=projection(a[0],a[1],a[2]);if(k<0)return false;
        return o2(a[p],a[q],a[ua],(k+1)%3,(k+2)%3)*o2(a[p],a[q],b[ub],(k+1)%3,(k+2)%3)>=0;
    }
    int ka=projection(a[0],a[1],a[2]),kb=projection(b[0],b[1],b[2]);
    if(ka<0||kb<0)return false; // Degeneracy is a separate failing audit.
    int da[3],db[3];for(int i=0;i<3;++i){da[i]=o3(b[0],b[1],b[2],a[i]);db[i]=o3(a[0],a[1],a[2],b[i]);}
    if((da[0]>0&&da[1]>0&&da[2]>0)||(da[0]<0&&da[1]<0&&da[2]<0)||(db[0]>0&&db[1]>0&&db[2]>0)||(db[0]<0&&db[1]<0&&db[2]<0))return false;
    for(int i=0;i<3;++i){int j=(i+1)%3;
        if(segment_triangle(a[i],a[j],fa[i],fa[j],b,fb,kb)||segment_triangle(b[i],b[j],fb[i],fb[j],a,fa,ka))return true;}
    return false;
}
struct Box{V lo,hi;};
struct Node{Box box;int begin,end,left=-1,right=-1;};
static bool overlap(const Box&a,const Box&b){for(int k=0;k<3;++k)if(a.lo[k]>b.hi[k]||b.lo[k]>a.hi[k])return false;return true;}
struct BVH{
    std::vector<F> faces;std::vector<V> vertices;std::vector<Box> boxes;std::vector<int> order;std::vector<Node> nodes;
    int build(int begin,int end){
        Box box=boxes[order[begin]];for(int i=begin+1;i<end;++i)for(int k=0;k<3;++k){box.lo[k]=std::min(box.lo[k],boxes[order[i]].lo[k]);box.hi[k]=std::max(box.hi[k],boxes[order[i]].hi[k]);}
        int index=int(nodes.size());nodes.push_back({box,begin,end});
        if(end-begin>8){int axis=0;for(int k=1;k<3;++k)if(box.hi[k]-box.lo[k]>box.hi[axis]-box.lo[axis])axis=k;int mid=(begin+end)/2;
            std::nth_element(order.begin()+begin,order.begin()+mid,order.begin()+end,[&](int a,int b){return boxes[a].lo[axis]*.5+boxes[a].hi[axis]*.5<boxes[b].lo[axis]*.5+boxes[b].hi[axis]*.5;});
            int l=build(begin,mid),r=build(mid,end);nodes[index].left=l;nodes[index].right=r;}
        return index;
    }
};
using VA=py::array_t<double,py::array::c_style>;
using FA=py::array_t<int64_t,py::array::c_style>;
static py::dict intersections(VA v,FA f,int64_t max_pairs){
    if(v.ndim()!=2||v.shape(1)!=3||f.ndim()!=2||f.shape(1)!=3||max_pairs<0)throw std::invalid_argument("invalid triangle arrays or pair limit");
    BVH tree;auto vv=v.unchecked<2>();auto ff=f.unchecked<2>();
    for(py::ssize_t i=0;i<v.shape(0);++i){V p={vv(i,0),vv(i,1),vv(i,2)};for(auto x:p)if(!std::isfinite(x))throw std::invalid_argument("nonfinite vertex");tree.vertices.push_back(p);}
    for(py::ssize_t i=0;i<f.shape(0);++i){F t={ff(i,0),ff(i,1),ff(i,2)};for(auto x:t)if(x<0||x>=v.shape(0))throw std::invalid_argument("invalid vertex index");tree.faces.push_back(t);}
    std::vector<std::array<int64_t,2>> result;uint64_t tested=0;bool complete=true;
    {
        py::gil_scoped_release release;
        int n=int(tree.faces.size());tree.boxes.resize(n);tree.order.resize(n);std::iota(tree.order.begin(),tree.order.end(),0);tree.nodes.reserve(2*n);
        for(int i=0;i<n;++i){auto t=tree.faces[i];Box b={tree.vertices[t[0]],tree.vertices[t[0]]};for(int j=1;j<3;++j)for(int k=0;k<3;++k){b.lo[k]=std::min(b.lo[k],tree.vertices[t[j]][k]);b.hi[k]=std::max(b.hi[k],tree.vertices[t[j]][k]);}tree.boxes[i]=b;}
        if(n)tree.build(0,n);
        std::vector<std::pair<int,int>> stack;if(n)stack.push_back({0,0});
        while(!stack.empty()){
            auto [ai,bi]=stack.back();stack.pop_back();auto a=tree.nodes[ai],b=tree.nodes[bi];if(!overlap(a.box,b.box))continue;
            if(a.left<0&&b.left<0){
                for(int p=a.begin;p<a.end&&complete;++p)for(int q=b.begin;q<b.end;++q){int i=tree.order[p],j=tree.order[q];if(ai==bi&&p>=q)continue;
                    if(!overlap(tree.boxes[i],tree.boxes[j]))continue;++tested;V x[3],y[3];for(int k=0;k<3;++k){x[k]=tree.vertices[tree.faces[i][k]];y[k]=tree.vertices[tree.faces[j][k]];}
                    if(intersect(x,y,tree.faces[i],tree.faces[j])){result.push_back({std::min(i,j),std::max(i,j)});if(max_pairs&&int64_t(result.size())>=max_pairs){complete=false;break;}}
                }if(!complete)break;
            }else if(ai==bi){stack.push_back({a.left,a.left});stack.push_back({a.left,a.right});stack.push_back({a.right,a.right});}
            else if(a.left>=0&&(b.left<0||a.end-a.begin>=b.end-b.begin)){stack.push_back({a.left,bi});stack.push_back({a.right,bi});}
            else{stack.push_back({ai,b.left});stack.push_back({ai,b.right});}
        }
        std::sort(result.begin(),result.end());
    }
    py::array_t<int64_t> pairs({py::ssize_t(result.size()),py::ssize_t(2)});if(!result.empty())std::memcpy(pairs.mutable_data(),result.data(),result.size()*sizeof(result[0]));
    py::dict out;out["pairs"]=pairs;out["complete"]=complete;out["narrow_phase_pairs"]=tested;out["bvh_nodes"]=tree.nodes.size();return out;
}
// Acceleration interface: enumerate every inclusive AABB candidate, then apply
// exactly the same predicate to an explicitly supplied conservative subset.
// The original exhaustive query above remains the independent CPU oracle.
static py::array_t<int64_t> broad_pairs(VA v,FA f,int64_t pair_budget){
    if(v.ndim()!=2||v.shape(1)!=3||f.ndim()!=2||f.shape(1)!=3||pair_budget<1||f.shape(0)>std::numeric_limits<int>::max())throw std::invalid_argument("invalid broad-phase input");
    BVH tree;auto vv=v.unchecked<2>();auto ff=f.unchecked<2>();
    for(py::ssize_t i=0;i<v.shape(0);++i){V p={vv(i,0),vv(i,1),vv(i,2)};for(auto x:p)if(!std::isfinite(x))throw std::invalid_argument("nonfinite vertex");tree.vertices.push_back(p);}
    int n=int(f.shape(0));tree.boxes.resize(n);tree.order.resize(n);std::iota(tree.order.begin(),tree.order.end(),0);tree.nodes.reserve(size_t(n)*2);
    for(int i=0;i<n;++i){F t={ff(i,0),ff(i,1),ff(i,2)};for(auto x:t)if(x<0||x>=v.shape(0))throw std::invalid_argument("bad vertex index");tree.faces.push_back(t);}
    std::vector<std::array<int64_t,2>> result;
    {
        py::gil_scoped_release release;
        for(int i=0;i<n;++i){auto t=tree.faces[i];Box b={tree.vertices[t[0]],tree.vertices[t[0]]};for(int j=1;j<3;++j)for(int k=0;k<3;++k){b.lo[k]=std::min(b.lo[k],tree.vertices[t[j]][k]);b.hi[k]=std::max(b.hi[k],tree.vertices[t[j]][k]);}tree.boxes[i]=b;}
        if(n)tree.build(0,n);
        std::vector<std::pair<int,int>> stack;if(n)stack.push_back({0,0});
        while(!stack.empty()){
            auto [ai,bi]=stack.back();stack.pop_back();auto a=tree.nodes[ai],b=tree.nodes[bi];if(!overlap(a.box,b.box))continue;
            if(a.left<0&&b.left<0){
                for(int p=a.begin;p<a.end;++p)for(int q=b.begin;q<b.end;++q){int i=tree.order[p],j=tree.order[q];if(ai==bi&&p>=q)continue;
                    if(!overlap(tree.boxes[i],tree.boxes[j]))continue;
                    if(int64_t(result.size())>=pair_budget)throw std::runtime_error("AABB pair budget exceeded; no truncated acceptance");
                    result.push_back({std::min(i,j),std::max(i,j)});
                }
            }else if(ai==bi){stack.push_back({a.left,a.left});stack.push_back({a.left,a.right});stack.push_back({a.right,a.right});}
            else if(a.left>=0&&(b.left<0||a.end-a.begin>=b.end-b.begin)){stack.push_back({a.left,bi});stack.push_back({a.right,bi});}
            else{stack.push_back({ai,b.left});stack.push_back({ai,b.right});}
        }
    }
    py::array_t<int64_t> out({py::ssize_t(result.size()),py::ssize_t(2)});if(!result.empty())std::memcpy(out.mutable_data(),result.data(),result.size()*sizeof(result[0]));return out;
}
static py::dict exact_pairs(VA v,FA f,FA pairs,int64_t max_pairs){
    if(v.ndim()!=2||v.shape(1)!=3||f.ndim()!=2||f.shape(1)!=3||pairs.ndim()!=2||pairs.shape(1)!=2||max_pairs<0)throw std::invalid_argument("invalid exact refinement input");
    auto vv=v.unchecked<2>();auto ff=f.unchecked<2>();auto pp=pairs.unchecked<2>();
    for(py::ssize_t i=0;i<v.shape(0);++i)for(int k=0;k<3;++k)if(!std::isfinite(vv(i,k)))throw std::invalid_argument("nonfinite vertex");
    for(py::ssize_t i=0;i<f.shape(0);++i)for(int k=0;k<3;++k)if(ff(i,k)<0||ff(i,k)>=v.shape(0))throw std::invalid_argument("bad vertex index");
    for(py::ssize_t i=0;i<pairs.shape(0);++i)if(pp(i,0)<0||pp(i,1)<0||pp(i,0)>=f.shape(0)||pp(i,1)>=f.shape(0)||pp(i,0)==pp(i,1))throw std::invalid_argument("bad face pair");
    std::vector<std::array<int64_t,2>> result;bool complete=true;
    {
        py::gil_scoped_release release;
        for(py::ssize_t i=0;i<pairs.shape(0);++i){auto a=pp(i,0),b=pp(i,1);F fa,fb;V x[3],y[3];for(int j=0;j<3;++j){fa[j]=ff(a,j);fb[j]=ff(b,j);for(int k=0;k<3;++k){x[j][k]=vv(fa[j],k);y[j][k]=vv(fb[j],k);}}
            if(intersect(x,y,fa,fb)){result.push_back({std::min(a,b),std::max(a,b)});if(max_pairs&&int64_t(result.size())>=max_pairs){complete=false;break;}}
        }
        std::sort(result.begin(),result.end());result.erase(std::unique(result.begin(),result.end()),result.end());
    }
    py::array_t<int64_t> out({py::ssize_t(result.size()),py::ssize_t(2)});if(!result.empty())std::memcpy(out.mutable_data(),result.data(),result.size()*sizeof(result[0]));
    py::dict report;report["pairs"]=out;report["complete"]=complete;report["narrow_phase_pairs"]=pairs.shape(0);return report;
}
#include "simplify.h"
#include "triangulate.h"
#include "patch.h"
PYBIND11_MODULE(_geometry_native,m){
    m.def("broad_pairs",&broad_pairs,py::arg("vertices").noconvert(),py::arg("faces").noconvert(),py::arg("pair_budget")=20000000);
    m.def("exact_pairs",&exact_pairs,py::arg("vertices").noconvert(),py::arg("faces").noconvert(),py::arg("pairs").noconvert(),py::arg("max_pairs")=0);
    m.def("patch_3d",&patch_native,py::arg("vertices").noconvert(),py::arg("loop").noconvert(),py::arg("forbidden_edges").noconvert(),py::arg("limit"));
    m.def("triangulate",&triangulate_native,py::arg("vertices").noconvert(),py::arg("rings"),py::arg("drop_axis"));
    m.def("projected_signs",[](VA v,FA f,int k){
        if(v.ndim()!=2||v.shape(1)!=3||f.ndim()!=2||f.shape(1)!=3||k<0||k>2)throw std::invalid_argument("invalid projection arrays");
        auto a=v.unchecked<2>();auto b=f.unchecked<2>();py::array_t<int> result(f.shape(0));auto r=result.mutable_unchecked<1>();
        for(py::ssize_t i=0;i<f.shape(0);++i){V p[3];for(int j=0;j<3;++j){auto id=b(i,j);if(id<0||id>=v.shape(0))throw std::invalid_argument("bad vertex index");for(int l=0;l<3;++l)p[j][l]=a(id,l);}r(i)=o2(p[0],p[1],p[2],(k+1)%3,(k+2)%3);}return result;
    },py::arg("vertices").noconvert(),py::arg("faces").noconvert(),py::arg("drop_axis"));
    m.def("degenerate_faces",[](VA v,FA f){
        if(v.ndim()!=2||v.shape(1)!=3||f.ndim()!=2||f.shape(1)!=3)throw std::invalid_argument("invalid mesh arrays");
        auto a=v.unchecked<2>();auto b=f.unchecked<2>();std::vector<int64_t> bad;
        for(py::ssize_t i=0;i<f.shape(0);++i){V p[3];for(int j=0;j<3;++j){auto index=b(i,j);if(index<0||index>=v.shape(0))throw std::invalid_argument("bad index");for(int k=0;k<3;++k){p[j][k]=a(index,k);if(!std::isfinite(p[j][k]))throw std::invalid_argument("nonfinite vertex");}}if(projection(p[0],p[1],p[2])<0)bad.push_back(i);}
        py::array_t<int64_t> out(bad.size());if(!bad.empty())std::memcpy(out.mutable_data(),bad.data(),bad.size()*8);return out;
    },py::arg("vertices").noconvert(),py::arg("faces").noconvert());
    m.def("simplify",&simplify_native,py::arg("vertices").noconvert(),py::arg("faces").noconvert(),py::arg("target"),py::arg("locked").noconvert(),py::arg("max_error"));
    m.def("intersections",&intersections,py::arg("vertices").noconvert(),py::arg("faces").noconvert(),py::arg("max_pairs")=0);
    m.def("orient2d",[](std::array<double,3>a,std::array<double,3>b,std::array<double,3>c){return o2(a,b,c,0,1);});
    m.def("orient3d",&o3);
    m.def("vertex_links",[](FA f,FA pairs,int64_t n){
        if(f.ndim()!=2||f.shape(1)!=3||pairs.ndim()!=2||pairs.shape(1)!=2||n<0)throw std::invalid_argument("invalid link arrays");
        auto ff=f.unchecked<2>();auto pp=pairs.unchecked<2>();
        std::vector<int64_t> parent(3*f.shape(0));std::iota(parent.begin(),parent.end(),0);
        auto root=[&](int64_t a){while(parent[a]!=a){parent[a]=parent[parent[a]];a=parent[a];}return a;};
        for(py::ssize_t i=0;i<pairs.shape(0);++i){auto a=pp(i,0),b=pp(i,1);if(a<0||b<0||a>=f.shape(0)||b>=f.shape(0))throw std::invalid_argument("invalid face pair");
            for(int j=0;j<3;++j)for(int k=0;k<3;++k)if(ff(a,j)==ff(b,k)){auto x=root(3*a+j),y=root(3*b+k);parent[x]=y;}}
        std::vector<int64_t> first(n,-1);std::vector<uint8_t> bad(n,0);
        for(py::ssize_t i=0;i<f.shape(0);++i)for(int j=0;j<3;++j){auto v=ff(i,j);if(v<0||v>=n)throw std::invalid_argument("invalid vertex index");auto r=root(3*i+j);if(first[v]<0)first[v]=r;else if(first[v]!=r)bad[v]=1;}
        std::vector<int64_t> ids;for(int64_t i=0;i<n;++i)if(bad[i])ids.push_back(i);
        py::array_t<int64_t> out(ids.size());if(!ids.empty())std::memcpy(out.mutable_data(),ids.data(),ids.size()*8);return out;
    },py::arg("faces").noconvert(),py::arg("pairs").noconvert(),py::arg("vertex_count"));
    m.def("corner_components",[](FA f,FA pairs){
        if(f.ndim()!=2||f.shape(1)!=3||pairs.ndim()!=2||pairs.shape(1)!=2)throw std::invalid_argument("invalid corner arrays");
        auto ff=f.unchecked<2>();auto pp=pairs.unchecked<2>();std::vector<int64_t> parent(3*f.shape(0));std::iota(parent.begin(),parent.end(),0);
        auto root=[&](int64_t a){while(parent[a]!=a){parent[a]=parent[parent[a]];a=parent[a];}return a;};
        for(py::ssize_t i=0;i<pairs.shape(0);++i){auto a=pp(i,0),b=pp(i,1);if(a<0||b<0||a>=f.shape(0)||b>=f.shape(0))throw std::invalid_argument("invalid face pair");for(int j=0;j<3;++j)for(int k=0;k<3;++k)if(ff(a,j)==ff(b,k))parent[root(3*a+j)]=root(3*b+k);}
        py::array_t<int64_t> result({f.shape(0),py::ssize_t(3)});auto r=result.mutable_unchecked<2>();for(py::ssize_t i=0;i<f.shape(0);++i)for(int j=0;j<3;++j)r(i,j)=root(3*i+j);return result;
    },py::arg("faces").noconvert(),py::arg("pairs").noconvert());
}
