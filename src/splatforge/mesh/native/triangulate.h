// SPDX-License-Identifier: Apache-2.0
// Exact-sign ear clipping and visible bridges for planar polygons with holes.
static int ring_sign(const std::vector<int64_t>&r,const std::vector<V>&v,int k){
    int x=(k+1)%3,y=(k+2)%3;D sum;V origin=v[r[0]];
    for(size_t i=0;i<r.size();++i){auto a=v[r[i]],b=v[r[(i+1)%r.size()]];
        sum=add(sum,sub(mul(sub(D(a[x]),D(origin[x])),sub(D(b[y]),D(origin[y]))),mul(sub(D(a[y]),D(origin[y])),sub(D(b[x]),D(origin[x])))));}
    return sum.sign;
}
static bool point_ring(const V&p,const std::vector<int64_t>&ring,const std::vector<V>&v,int k){
    int x=(k+1)%3,y=(k+2)%3;bool inside=false;
    for(size_t i=0;i<ring.size();++i){auto a=v[ring[i]],b=v[ring[(i+1)%ring.size()]];
        if((a[y]>p[y])!=(b[y]>p[y])){int s=o2(a,b,p,x,y);if((b[y]>a[y]&&s>0)||(b[y]<a[y]&&s<0))inside=!inside;}}
    return inside;
}
static py::array_t<int64_t> triangulate_native(VA vv,py::list input_rings,int k){
    if(vv.ndim()!=2||vv.shape(1)!=3||k<0||k>2)throw std::invalid_argument("invalid cap vertices/projection");
    auto a=vv.unchecked<2>();std::vector<V> v(vv.shape(0));for(size_t i=0;i<v.size();++i)for(int j=0;j<3;++j){v[i][j]=a(i,j);if(!std::isfinite(v[i][j]))throw std::invalid_argument("nonfinite cap vertex");}
    std::vector<std::vector<int64_t>> rings;
    for(auto item:input_rings){auto r=py::cast<std::vector<int64_t>>(item);if(r.size()<3)throw std::invalid_argument("cap ring too short");for(auto i:r)if(i<0||i>=int64_t(v.size()))throw std::invalid_argument("cap vertex out of bounds");rings.push_back(std::move(r));}
    if(rings.empty())throw std::invalid_argument("no cap rings");std::vector<F> triangles;
    {
    py::gil_scoped_release release;
    for(size_t i=0;i<rings.size();++i){int s=ring_sign(rings[i],v,k);if(!s)throw std::runtime_error("zero-area cap contour");if((i==0&&s<0)||(i>0&&s>0))std::reverse(rings[i].begin(),rings[i].end());}
    std::vector<int64_t> path=rings[0];int x=(k+1)%3,y=(k+2)%3;
    // Rightmost holes first avoid bridges through unmerged holes.
    std::vector<size_t> holes;for(size_t i=1;i<rings.size();++i)holes.push_back(i);
    auto rightmost=[&](size_t h){return size_t(std::max_element(rings[h].begin(),rings[h].end(),[&](int64_t i,int64_t j){return v[i][x]<v[j][x];})-rings[h].begin());};
    std::sort(holes.begin(),holes.end(),[&](size_t i,size_t j){return v[rings[i][rightmost(i)]][x]>v[rings[j][rightmost(j)]][x];});
    for(auto h:holes){auto ring=rings[h];size_t hi=rightmost(h);int64_t hid=ring[hi];V hp=v[hid];std::vector<size_t> order(path.size());std::iota(order.begin(),order.end(),0);
        std::sort(order.begin(),order.end(),[&](size_t i,size_t j){auto p=vsub(v[path[i]],hp),q=vsub(v[path[j]],hp);double dp=p[x]*p[x]+p[y]*p[y],dq=q[x]*q[x]+q[y]*q[y];return dp!=dq?dp<dq:i<j;});
        size_t chosen=path.size();
        for(size_t oi:order){int64_t oid=path[oi];V op=v[oid];if(op==hp)continue;bool clear=true;
            for(size_t j=0;j<path.size()&&clear;++j)if(edge2(hp,op,v[path[j]],v[path[(j+1)%path.size()]],hid,oid,path[j],path[(j+1)%path.size()],k))clear=false;
            for(size_t j=1;j<rings.size()&&clear;++j)for(size_t l=0;l<rings[j].size()&&clear;++l){auto c=rings[j][l],d=rings[j][(l+1)%rings[j].size()];if(edge2(hp,op,v[c],v[d],hid,oid,c,d,k))clear=false;}
            if(!clear)continue;V mid={hp[0]*.5+op[0]*.5,hp[1]*.5+op[1]*.5,hp[2]*.5+op[2]*.5};
            if(!point_ring(mid,rings[0],v,k))continue;
            for(size_t j=1;j<rings.size()&&clear;++j)if(point_ring(mid,rings[j],v,k))clear=false;
            if(clear){chosen=oi;break;}}
        if(chosen==path.size())throw std::runtime_error("no visible cap hole bridge");
        std::vector<int64_t> merged;merged.insert(merged.end(),path.begin(),path.begin()+chosen+1);
        for(size_t j=0;j<=ring.size();++j)merged.push_back(ring[(hi+j)%ring.size()]);
        merged.insert(merged.end(),path.begin()+chosen,path.end());path.swap(merged);
    }
    int n=int(path.size());std::vector<int> prev(n),next(n);std::vector<uint8_t> alive(n,1);for(int i=0;i<n;++i){prev[i]=(i+n-1)%n;next[i]=(i+1)%n;}
    int remaining=n,current=0,misses=0;
    while(remaining>3){int pi=prev[current],ni=next[current];auto ia=path[pi],ib=path[current],ic=path[ni];V a=v[ia],b=v[ib],c=v[ic];int turn=o2(a,b,c,x,y);bool ear=turn>0;
        if(ear){double lx=std::min({a[x],b[x],c[x]}),hx=std::max({a[x],b[x],c[x]}),ly=std::min({a[y],b[y],c[y]}),hy=std::max({a[y],b[y],c[y]});
            for(int j=next[ni];j!=pi;j=next[j]){auto id=path[j];V p=v[id];if(id==ia||id==ib||id==ic)continue;
                if(p[x]<lx||p[x]>hx||p[y]<ly||p[y]>hy)continue;
                if(inside2(p,a,b,c,k)){ear=false;break;}}}
        bool collinear=turn==0&&ia!=ic&&between(b,a,c,k);
        if(ear||collinear){if(ear)triangles.push_back({ia,ib,ic});alive[current]=0;next[pi]=ni;prev[ni]=pi;--remaining;current=ni;misses=0;}
        else{current=next[current];if(++misses>remaining)throw std::runtime_error("cap contour cannot be triangulated without crossing or degeneracy");}
    }
    int a=current,b=next[a],c=next[b];if(o2(v[path[a]],v[path[b]],v[path[c]],x,y)>0)triangles.push_back({path[a],path[b],path[c]});
    }
    py::array_t<int64_t> out({py::ssize_t(triangles.size()),py::ssize_t(3)});if(!triangles.empty())std::memcpy(out.mutable_data(),triangles.data(),triangles.size()*sizeof(F));return out;
}
