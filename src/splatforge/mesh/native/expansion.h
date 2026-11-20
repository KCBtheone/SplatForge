// SPDX-License-Identifier: Apache-2.0
// Owned error-free expansion arithmetic for the ordinary-exponent exact path.
// Extreme exponents use the arbitrary-precision dyadic path in geometry.cpp.
struct Expansion {
    double e[256];int n=0;
};
static void grow(Expansion& a,double x){
    double q=x;int count=0;
    for(int i=0;i<a.n;++i){double b=a.e[i],s=q+b,bv=s-q,av=s-bv;
        double error=(q-av)+(b-bv);if(error!=0)a.e[count++]=error;q=s;}
    if(q!=0)a.e[count++]=q;
    a.n=count;
}
static Expansion ediff(double a,double b){
    Expansion e;double x=a-b,bv=a-x,av=x+bv,br=bv-b,ar=a-av,error=ar+br;
    if(error)e.e[e.n++]=error;if(x)e.e[e.n++]=x;return e;
}
static Expansion esub(const Expansion&a,const Expansion&b){
    Expansion r;r.n=a.n;std::copy(a.e,a.e+a.n,r.e);for(int i=0;i<b.n;++i)grow(r,-b.e[i]);return r;
}
static Expansion emul(const Expansion&a,const Expansion&b){
    Expansion r;
    for(int i=0;i<a.n;++i)for(int j=0;j<b.n;++j){double product=a.e[i]*b.e[j],error=std::fma(a.e[i],b.e[j],-product);if(error)grow(r,error);if(product)grow(r,product);}
    return r;
}
static bool ordinary(const V&p){for(double x:p)if(x!=0&&(std::abs(x)<0x1p-200||std::abs(x)>0x1p200))return false;return true;}
static int esign(const Expansion&e){return !e.n?0:(e.e[e.n-1]>0?1:-1);}
static int exact2_expansion(const V&a,const V&b,const V&c,int x,int y){
    return esign(esub(emul(ediff(a[x],c[x]),ediff(b[y],c[y])),emul(ediff(a[y],c[y]),ediff(b[x],c[x]))));
}
static int exact3_expansion(const V&a,const V&b,const V&c,const V&d){
    Expansion aa[3],bb[3],cc[3],r;
    for(int k=0;k<3;++k){aa[k]=ediff(a[k],d[k]);bb[k]=ediff(b[k],d[k]);cc[k]=ediff(c[k],d[k]);}
    for(int k=0;k<3;++k){int j=(k+1)%3,l=(k+2)%3;Expansion t=emul(aa[k],esub(emul(bb[j],cc[l]),emul(bb[l],cc[j])));for(int i=0;i<t.n;++i)grow(r,t.e[i]);}
    return esign(r);
}
