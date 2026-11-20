// SPDX-License-Identifier: Apache-2.0
// Owned quadric edge collapse. Link, duplicate and normal-change constraints.
#include <queue>
#include <set>
#include <unordered_set>
struct Quadric {std::array<double,16> q{};};
static V vsub(const V&a,const V&b){return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}
static V vcross(const V&a,const V&b){return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
static double vdot(const V&a,const V&b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
static Quadric qadd(Quadric a,const Quadric&b){for(int i=0;i<16;++i)a.q[i]+=b.q[i];return a;}
static double qcost(const Quadric&q,const V&p){double x[4]={p[0],p[1],p[2],1},value=0;for(int i=0;i<4;++i)for(int j=0;j<4;++j)value+=x[i]*q.q[4*i+j]*x[j];return std::max(0.,value);}
struct Collapse{double cost;int a,b;uint64_t va,vb;V p;bool operator<(const Collapse&c)const{if(cost!=c.cost)return cost>c.cost;if(a!=c.a)return a>c.a;return b>c.b;}};
static bool qsolve(const Quadric&q,V&p){
    double m[3][4],largest=0;for(int i=0;i<3;++i){for(int j=0;j<3;++j){m[i][j]=q.q[4*i+j];largest=std::max(largest,std::abs(m[i][j]));}m[i][3]=-q.q[4*i+3];}
    if(largest==0)return false;
    for(int i=0;i<3;++i){int pivot=i;for(int j=i+1;j<3;++j)if(std::abs(m[j][i])>std::abs(m[pivot][i]))pivot=j;
        if(std::abs(m[pivot][i])<largest*1e-12)return false;
        for(int k=i;k<4;++k)std::swap(m[i][k],m[pivot][k]);double d=m[i][i];for(int k=i;k<4;++k)m[i][k]/=d;
        for(int j=0;j<3;++j)if(j!=i){double f=m[j][i];for(int k=i;k<4;++k)m[j][k]-=f*m[i][k];}}
    for(int i=0;i<3;++i){p[i]=m[i][3];if(!std::isfinite(p[i]))return false;}return true;
}
static py::dict simplify_native(VA vv,FA ff,int64_t target,py::array_t<bool,py::array::c_style> locks,double max_error){
    if(vv.ndim()!=2||vv.shape(1)!=3||ff.ndim()!=2||ff.shape(1)!=3||locks.ndim()!=1||locks.shape(0)!=vv.shape(0)||target<0||!(max_error>=0))throw std::invalid_argument("invalid simplification arguments");
    std::vector<V> v(vv.shape(0));std::vector<F> f(ff.shape(0));auto vin=vv.unchecked<2>();auto fin=ff.unchecked<2>();auto lin=locks.unchecked<1>();
    for(size_t i=0;i<v.size();++i)for(int j=0;j<3;++j){v[i][j]=vin(i,j);if(!std::isfinite(v[i][j]))throw std::invalid_argument("nonfinite vertex");}
    for(size_t i=0;i<f.size();++i)for(int j=0;j<3;++j){f[i][j]=fin(i,j);if(f[i][j]<0||f[i][j]>=int64_t(v.size()))throw std::invalid_argument("bad index");}
    std::vector<uint8_t> fixed(v.size());for(size_t i=0;i<v.size();++i)fixed[i]=lin(i);
    int64_t remaining=f.size(),collapsed=0;double max_used=0;
    {
    py::gil_scoped_release release;
    std::vector<Quadric> quadrics(v.size());std::vector<std::unordered_set<int>> incident(v.size());std::vector<uint64_t> version(v.size(),0);std::vector<uint8_t> alive(v.size(),1),valid(f.size(),1);
    std::unordered_map<uint64_t,int> edge_count;
    auto key=[](int a,int b){return (uint64_t(std::min(a,b))<<32)|uint32_t(std::max(a,b));};
    for(size_t i=0;i<f.size();++i){auto t=f[i];V n=vcross(vsub(v[t[1]],v[t[0]]),vsub(v[t[2]],v[t[0]]));double norm=std::sqrt(vdot(n,n));if(norm==0)throw std::invalid_argument("degenerate input face");for(auto&x:n)x/=norm;double p[4]={n[0],n[1],n[2],-vdot(n,v[t[0]])};
        for(int j=0;j<3;++j){incident[t[j]].insert(int(i));++edge_count[key(int(t[j]),int(t[(j+1)%3]))];for(int a=0;a<4;++a)for(int b=0;b<4;++b)quadrics[t[j]].q[4*a+b]+=p[a]*p[b];}}
    // Boundary vertices are locked, rather than moving a silhouette implicitly.
    for(auto e:edge_count)if(e.second==1){fixed[e.first>>32]=1;fixed[uint32_t(e.first)]=1;}
    std::priority_queue<Collapse> heap;
    auto enqueue=[&](int a,int b){
        if(a==b||!alive[a]||!alive[b]||(fixed[a]&&fixed[b]))return;
        if(fixed[b]||(!fixed[a]&&a>b))std::swap(a,b);
        Quadric q=qadd(quadrics[a],quadrics[b]);V p=v[a];double cost=qcost(q,p);
        if(!fixed[a]){V candidate=v[b];double c=qcost(q,candidate);if(c<cost){cost=c;p=candidate;}
            candidate={(v[a][0]+v[b][0])*.5,(v[a][1]+v[b][1])*.5,(v[a][2]+v[b][2])*.5};c=qcost(q,candidate);if(c<cost){cost=c;p=candidate;}
            if(qsolve(q,candidate)){c=qcost(q,candidate);if(c<cost){cost=c;p=candidate;}}}
        if(std::isfinite(cost)&&cost<=max_error)heap.push({cost,a,b,version[a],version[b],p});
    };
    for(auto e:edge_count)enqueue(int(e.first>>32),int(uint32_t(e.first)));
    while(remaining>target&&!heap.empty()){
        auto c=heap.top();heap.pop();int a=c.a,b=c.b;
        if(!alive[a]||!alive[b]||version[a]!=c.va||version[b]!=c.vb)continue;
        std::unordered_set<int> na,nb,local=incident[a];local.insert(incident[b].begin(),incident[b].end());std::set<int> opposite;
        int edge_faces=0;
        for(int fi:incident[a]){bool hasb=false;for(auto x:f[fi]){if(x!=a)na.insert(int(x));hasb|=x==b;}if(hasb){++edge_faces;for(auto x:f[fi])if(x!=a&&x!=b)opposite.insert(int(x));}}
        for(int fi:incident[b])for(auto x:f[fi])if(x!=b)nb.insert(int(x));
        if(!edge_faces||edge_faces>2||remaining-edge_faces<target)continue;
        std::set<int> common;for(auto x:na)if(nb.count(x))common.insert(x);
        if(common!=opposite)continue; // Edge-link condition, including boundaries.
        bool okay=true;std::set<F> unique;
        for(int fi:local){auto t=f[fi];bool hasa=false,hasb=false;for(auto x:t){hasa|=x==a;hasb|=x==b;}if(hasa&&hasb)continue;
            V old[3]={v[t[0]],v[t[1]],v[t[2]]},next[3];F newf=t;
            for(int j=0;j<3;++j){next[j]=(t[j]==a||t[j]==b)?c.p:old[j];if(newf[j]==b)newf[j]=a;}
            std::sort(newf.begin(),newf.end());if(!unique.insert(newf).second){okay=false;break;}
            V on=vcross(vsub(old[1],old[0]),vsub(old[2],old[0])),nn=vcross(vsub(next[1],next[0]),vsub(next[2],next[0]));
            double oo=vdot(on,on),n2=vdot(nn,nn),dot=vdot(on,nn);
            if(n2<=oo*1e-24||dot<=0||dot*dot<.0625*oo*n2){okay=false;break;}}
        if(!okay)continue;
        for(int fi:local){auto t=f[fi];for(auto x:t)incident[x].erase(fi);
            bool hasa=false,hasb=false;for(auto x:t){hasa|=x==a;hasb|=x==b;}
            if(hasa&&hasb){valid[fi]=0;--remaining;continue;}
            for(auto&x:f[fi])if(x==b)x=a;for(auto x:f[fi])incident[x].insert(fi);}
        v[a]=c.p;quadrics[a]=qadd(quadrics[a],quadrics[b]);alive[b]=0;incident[b].clear();++version[a];++version[b];++collapsed;max_used=std::max(max_used,c.cost);
        std::unordered_set<uint64_t> nearby;
        for(int fi:incident[a])for(int j=0;j<3;++j)nearby.insert(key(int(f[fi][j]),int(f[fi][(j+1)%3])));
        for(auto k:nearby)enqueue(int(k>>32),int(uint32_t(k)));
    }
    std::vector<F> kept;kept.reserve(size_t(remaining));for(size_t i=0;i<f.size();++i)if(valid[i])kept.push_back(f[i]);f.swap(kept);
    }
    py::array_t<double> outv({py::ssize_t(v.size()),py::ssize_t(3)});py::array_t<int64_t> outf({py::ssize_t(f.size()),py::ssize_t(3)});
    if(!v.empty())std::memcpy(outv.mutable_data(),v.data(),v.size()*sizeof(V));if(!f.empty())std::memcpy(outf.mutable_data(),f.data(),f.size()*sizeof(F));
    py::dict out;out["vertices"]=outv;out["faces"]=outf;out["collapses"]=collapsed;out["max_quadric_error"]=max_used;out["target_reached"]=remaining==target;return out;
}
