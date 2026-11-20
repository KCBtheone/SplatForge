// Owned FP32 constrained-variance geometry. Derivative lanes are first-order
// differentials, not extra precision for the primal. No reference data is used.
#pragma once
#include <cmath>
#ifdef __CUDACC__
#define SF_HD __host__ __device__ __forceinline__
#else
#define SF_HD inline
#endif

SF_HD float rade_fma_scalar(float a,float b,float c){return fmaf(a,b,c);}
SF_HD double rade_fma_scalar(double a,double b,double c){return fma(a,b,c);}
SF_HD float rade_sqrt_scalar(float a){return sqrtf(a);}
SF_HD double rade_sqrt_scalar(double a){return sqrt(a);}
template<int D,typename S=float> struct RadeValue {
    S v; S d[D ? D : 1];
    SF_HD RadeValue(S x=S(0)):v(x) { for(int j=0;j<D;++j)d[j]=S(0); }
};
template<int D,typename S=float> SF_HD RadeValue<D,S> rade_variable(S x,int lane) {
    RadeValue<D,S> a(x);if(lane<D)a.d[lane]=S(1);return a;
}
template<int D,typename S> SF_HD RadeValue<D,S> operator+(const RadeValue<D,S>&a,const RadeValue<D,S>&b) {
    RadeValue<D,S> c(a.v+b.v);for(int j=0;j<D;++j)c.d[j]=a.d[j]+b.d[j];return c;
}
template<int D,typename S> SF_HD RadeValue<D,S> operator-(const RadeValue<D,S>&a,const RadeValue<D,S>&b) {
    RadeValue<D,S> c(a.v-b.v);for(int j=0;j<D;++j)c.d[j]=a.d[j]-b.d[j];return c;
}
template<int D,typename S> SF_HD RadeValue<D,S> operator-(const RadeValue<D,S>&a) {
    RadeValue<D,S> c(-a.v);for(int j=0;j<D;++j)c.d[j]=-a.d[j];return c;
}
template<int D,typename S> SF_HD RadeValue<D,S> operator*(const RadeValue<D,S>&a,const RadeValue<D,S>&b) {
    RadeValue<D,S> c(a.v*b.v);for(int j=0;j<D;++j)c.d[j]=rade_fma_scalar(a.d[j],b.v,a.v*b.d[j]);return c;
}
template<int D,typename S> SF_HD RadeValue<D,S> operator/(const RadeValue<D,S>&a,const RadeValue<D,S>&b) {
    RadeValue<D,S> c(a.v/b.v);for(int j=0;j<D;++j)c.d[j]=rade_fma_scalar(-c.v,b.d[j],a.d[j])/b.v;return c;
}
template<int D,typename S> SF_HD RadeValue<D,S> rade_fma(const RadeValue<D,S>&a,const RadeValue<D,S>&b,const RadeValue<D,S>&c) {
    RadeValue<D,S> z(rade_fma_scalar(a.v,b.v,c.v));
    for(int j=0;j<D;++j)z.d[j]=rade_fma_scalar(a.d[j],b.v,rade_fma_scalar(a.v,b.d[j],c.d[j]));return z;
}
template<int D,typename S> SF_HD RadeValue<D,S> rade_sqrt(const RadeValue<D,S>&a) {
    RadeValue<D,S> z(rade_sqrt_scalar(a.v));for(int j=0;j<D;++j)z.d[j]=a.d[j]/(S(2)*z.v);return z;
}
template<int D,typename S> SF_HD RadeValue<D,S> rade_floor(const RadeValue<D,S>&a,S floor) {
    return a.v>=floor?a:RadeValue<D,S>(floor);
}
template<int D,typename S> SF_HD RadeValue<D,S> rade_clip(const RadeValue<D,S>&a,S low,S high) {
    return a.v<low?RadeValue<D,S>(low):(a.v>high?RadeValue<D,S>(high):a);
}
template<int D,typename S> SF_HD RadeValue<D,S> rade_norm3(const RadeValue<D,S>*a) {
    return rade_sqrt(rade_fma(a[0],a[0],rade_fma(a[1],a[1],a[2]*a[2])));
}

template<int D,typename S> SF_HD void rade_reduce_coordinates(RadeValue<D,S> joint[6][3]) {
    using T=RadeValue<D,S>;T one(S(1));
    // Same primal row decisions and Householder program for both adjoints.
    for(int r=0;r<5;++r) {
        int best=r;S norm=S(-1);
        for(int j=r;j<6;++j) {
            S z=rade_fma_scalar(joint[j][0].v,joint[j][0].v,joint[j][1].v*joint[j][1].v);
            if(z>norm){norm=z;best=j;}
        }
        if(best!=r)for(int c=0;c<3;++c){T tmp=joint[r][c];joint[r][c]=joint[best][c];joint[best][c]=tmp;}
    }
    for(int col=0;col<2;++col) {
        T sum(S(0));for(int r=col;r<6;++r)sum=rade_fma(joint[r][col],joint[r][col],sum);
        T alpha=rade_sqrt(sum);if(joint[col][col].v>=0.f)alpha=-alpha;
        T first=joint[col][col]-alpha;
        T tau=(alpha-joint[col][col])/alpha;
        T reflector[6];reflector[col]=one;
        for(int r=col+1;r<6;++r)reflector[r]=joint[r][col]/first;
        for(int c=col+1;c<3;++c) {
            T dot=joint[col][c];for(int r=col+1;r<6;++r)dot=rade_fma(reflector[r],joint[r][c],dot);
            T weight=tau*dot;
            joint[col][c]=joint[col][c]-weight;
            for(int r=col+1;r<6;++r)joint[r][c]=rade_fma(-reflector[r],weight,joint[r][c]);
        }
        joint[col][col]=alpha;
    }
}

template<int D,typename S=float> SF_HD void rade_fields_math(const float*means,const float*quat,
        const float*scales,const float*view,const float*K,int width,int height,RadeValue<D,S>*out) {
    using T=RadeValue<D,S>;
    T m[3],q[4],s[3];
    for(int j=0;j<3;++j){m[j]=rade_variable<D,S>(S(means[j]),j);s[j]=rade_variable<D,S>(S(scales[j]),j+7);}
    for(int j=0;j<4;++j)q[j]=rade_variable<D,S>(S(quat[j]),j+3);
    T den=rade_floor(rade_fma(q[0],q[0],rade_fma(q[1],q[1],rade_fma(q[2],q[2],q[3]*q[3]))),S(1e-24));
    T two(2.f),one(1.f),zero(0.f);
    T n[9]={den-two*rade_fma(q[2],q[2],q[3]*q[3]),two*rade_fma(q[1],q[2],-q[0]*q[3]),two*rade_fma(q[1],q[3],q[0]*q[2]),
            two*rade_fma(q[1],q[2],q[0]*q[3]),den-two*rade_fma(q[1],q[1],q[3]*q[3]),two*rade_fma(q[2],q[3],-q[0]*q[1]),
            two*rade_fma(q[1],q[3],-q[0]*q[2]),two*rade_fma(q[2],q[3],q[0]*q[1]),den-two*rade_fma(q[1],q[1],q[2]*q[2])};
    T factor[3][3],camera[3];T trace(0.f);
    for(int r=0;r<3;++r) {
        camera[r]=rade_fma(T(view[4*r]),m[0],rade_fma(T(view[4*r+1]),m[1],rade_fma(T(view[4*r+2]),m[2],T(view[4*r+3]))));
        for(int c=0;c<3;++c) {
            T numerator=rade_fma(T(view[4*r]),n[c],rade_fma(T(view[4*r+1]),n[3+c],T(view[4*r+2])*n[6+c]));
            factor[r][c]=(numerator/den)*s[c];trace=rade_fma(factor[r][c],factor[r][c],trace);
        }
    }
    T lambda=rade_floor(T(S(1e-6))*rade_floor(trace,S(1e-8)),S(1e-8)),root=rade_sqrt(lambda);
    T raw_u=camera[0]/camera[2],raw_v=camera[1]/camera[2];
    T u=rade_clip(raw_u,-(S(K[2])+S(.15)*width)/S(K[0]),(S(width)-S(K[2])+S(.15)*width)/S(K[0]));
    T v=rade_clip(raw_v,-(S(K[5])+S(.15)*height)/S(K[4]),(S(height)-S(K[5])+S(.15)*height)/S(K[4]));
    T joint[6][3];
    for(int j=0;j<3;++j) {
        joint[j][0]=rade_fma(-u,factor[2][j],factor[0][j]);
        joint[j][1]=rade_fma(-v,factor[2][j],factor[1][j]);
        joint[j][2]=factor[2][j];
    }
    joint[3][0]=root;joint[3][1]=zero;joint[3][2]=zero;
    joint[4][0]=zero;joint[4][1]=root;joint[4][2]=zero;
    joint[5][0]=-root*u;joint[5][1]=-root*v;joint[5][2]=root;
    T t0,t1;
#ifdef SF_COORDINATE_PULLBACK
    if(D>0) {
        RadeValue<0,S> qr[6][3];
        for(int r=0;r<6;++r)for(int c=0;c<3;++c)qr[r][c].v=joint[r][c].v;
        rade_reduce_coordinates(qr);
        t1.v=-qr[1][2].v/qr[1][1].v;
        t0.v=rade_fma_scalar(-qr[0][1].v,t1.v,-qr[0][2].v)/qr[0][0].v;
        S residual[6];
        for(int r=0;r<6;++r)residual[r]=rade_fma_scalar(joint[r][0].v,t0.v,rade_fma_scalar(joint[r][1].v,t1.v,joint[r][2].v));
        // d(M^T(Mt+b))=0. Solve R^T R dt=-rhs with the same primal R;
        // no Gram matrix, inverse, detached chain or reference derivative.
        for(int lane=0;lane<D;++lane) {
            S rhs0=S(0),rhs1=S(0);
            for(int r=0;r<6;++r) {
                S dr=rade_fma_scalar(joint[r][0].d[lane],t0.v,rade_fma_scalar(joint[r][1].d[lane],t1.v,joint[r][2].d[lane]));
                rhs0=rade_fma_scalar(joint[r][0].d[lane],residual[r],rade_fma_scalar(joint[r][0].v,dr,rhs0));
                rhs1=rade_fma_scalar(joint[r][1].d[lane],residual[r],rade_fma_scalar(joint[r][1].v,dr,rhs1));
            }
            S y0=rhs0/qr[0][0].v,y1=rade_fma_scalar(-qr[0][1].v,y0,rhs1)/qr[1][1].v;
            t1.d[lane]=-y1/qr[1][1].v;
            t0.d[lane]=rade_fma_scalar(-qr[0][1].v,t1.d[lane],-y0)/qr[0][0].v;
        }
    } else
#endif
    {
        rade_reduce_coordinates(joint);
        t1=-joint[1][2]/joint[1][1];
        t0=rade_fma(-joint[0][1],t1,-joint[0][2])/joint[0][0];
    }
    T covector[3]={t0,t1,rade_fma(-u,t0,rade_fma(-v,t1,one))};
    T variance(0.f);
    for(int c=0;c<3;++c) {
        T a=rade_fma(factor[0][c],covector[0],rade_fma(factor[1][c],covector[1],factor[2][c]*covector[2]));
        variance=rade_fma(a,a,variance);
    }
    T cov_norm2=rade_fma(covector[0],covector[0],rade_fma(covector[1],covector[1],covector[2]*covector[2]));
    variance=rade_fma(lambda,cov_norm2,variance);
    T divisor=rade_sqrt(cov_norm2),eps=T(S(1e-12))*variance;if(divisor.v<eps.v)divisor=eps;
    T normal[3];for(int c=0;c<3;++c)normal[c]=covector[c]/divisor;
    T pairing=rade_floor(rade_fma(normal[0],u,rade_fma(normal[1],v,normal[2])),S(1e-7));
    T ray_norm2=rade_fma(u,u,rade_fma(v,v,one));
    T radius=rade_norm3(camera),length=radius;
    if(raw_u.v!=u.v||raw_v.v!=v.v)length=camera[2]*rade_sqrt(ray_norm2);
    T plane_factor=length/ray_norm2;
    T p0=rade_fma(one+v*v,normal[0],rade_fma(-u*v,normal[1],-u*normal[2]));
    T p1=rade_fma(-u*v,normal[0],rade_fma(one+u*u,normal[1],-v*normal[2]));
    out[0]=radius;out[1]=(p0/pairing)*plane_factor/T(K[0]);out[2]=(p1/pairing)*plane_factor/T(K[4]);
    for(int c=0;c<3;++c)out[3+c]=-normal[c];
}
