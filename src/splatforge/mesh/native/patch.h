// SPDX-License-Identifier: Apache-2.0
// Minimum-area combinatorial triangulation of a 3D boundary, O(n^3)/O(n^2).
// Explicit bounded provider for folded nonplanar loops; never claims embedding.
static py::array_t<int64_t> patch_native(VA vv,py::array_t<int64_t,py::array::c_style> loop,FA forbidden,int limit){
    if(vv.ndim()!=2||vv.shape(1)!=3||loop.ndim()!=1||loop.size()<3||loop.size()>limit)throw std::invalid_argument("3D patch boundary exceeds its explicit size bound");
    if(forbidden.ndim()!=2||forbidden.shape(1)!=2)throw std::invalid_argument("invalid patch constraints");
    auto a=vv.unchecked<2>();auto ids=loop.unchecked<1>();auto constraints=forbidden.unchecked<2>();int n=int(loop.size());std::vector<V> v(n);std::vector<int64_t> original(n);
    for(int i=0;i<n;++i){original[i]=ids(i);if(original[i]<0||original[i]>=vv.shape(0))throw std::invalid_argument("bad patch vertex index");for(int j=0;j<3;++j)v[i][j]=a(original[i],j);}
    std::vector<F> output;
    {
    py::gil_scoped_release release;
    std::unordered_set<uint64_t> present;
    auto key=[](int64_t a,int64_t b){return (uint64_t(std::min(a,b))<<32)|uint32_t(std::max(a,b));};
    for(py::ssize_t i=0;i<forbidden.shape(0);++i)present.insert(key(constraints(i,0),constraints(i,1)));
    auto allowed=[&](int i,int j){return j-i==1||(i==0&&j==n-1)||!present.count(key(original[i],original[j]));};
    std::vector<double> cost(size_t(n)*n,std::numeric_limits<double>::infinity());std::vector<int> split(size_t(n)*n,-1);
    for(int i=0;i<n-1;++i)cost[size_t(i)*n+i+1]=0;
    for(int span=2;span<n;++span)for(int i=0;i+span<n;++i){int j=i+span;if(!allowed(i,j))continue;V diagonal=vsub(v[j],v[i]);
        for(int k=i+1;k<j;++k){double subcost=cost[size_t(i)*n+k]+cost[size_t(k)*n+j];if(!std::isfinite(subcost))continue;
            V cross=vcross(vsub(v[k],v[i]),diagonal);double area=std::sqrt(vdot(cross,cross));
            if(area==0&&projection(v[i],v[k],v[j])<0)continue;
            double value=subcost+area;
            if(value<cost[size_t(i)*n+j]){cost[size_t(i)*n+j]=value;split[size_t(i)*n+j]=k;}}}
    if(!std::isfinite(cost[n-1]))throw std::runtime_error("3D patch cannot avoid degenerate triangles");
    std::vector<std::pair<int,int>> stack={{0,n-1}};
    while(!stack.empty()){auto [i,j]=stack.back();stack.pop_back();if(j-i<2)continue;int k=split[size_t(i)*n+j];if(k<0)throw std::runtime_error("invalid patch dynamic program");output.push_back({original[i],original[j],original[k]});stack.push_back({i,k});stack.push_back({k,j});}
    }
    py::array_t<int64_t> out({py::ssize_t(output.size()),py::ssize_t(3)});if(!output.empty())std::memcpy(out.mutable_data(),output.data(),output.size()*sizeof(F));return out;
}
