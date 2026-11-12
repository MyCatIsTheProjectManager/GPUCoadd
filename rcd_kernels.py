"""RCD demosaicing on the GPU: one CUDA kernel per 32 x 32 output tile, its five stages in shared
memory (the raw crop with a 10 px border; each stage over the border the next one reads), each
intermediate reflected at the image boundary. RCD is by Luis Sanz Rodriguez, see
THIRD-PARTY-NOTICES.txt.

The arithmetic helpers match rcd_fused's Jiterator operations; a register barrier at their
outputs keeps the same roundings, so the results are identical to rcd_fused.
"""

from cuda_kernels import Module

SOURCE = r"""
__device__ __forceinline__ float rounded(float v) {
    asm volatile("" : "+f"(v)); return v;
}
__device__ __forceinline__ float band(float a3,float a1,float b1,float b3,float a2,float b2,float c) {
    float d=a3-a1-b1+b3-3.0f*(a2+b2)+6.0f*c; return rounded(d*d);
}
__device__ __forceinline__ float ratio(float vm,float v,float vp,float hm,float h,float hp) {
    float vs=vm+v+vp;vs=vs<1e-10f?1e-10f:vs;
    float hs=hm+h+hp;hs=hs<1e-10f?1e-10f:hs;return rounded(vs/(vs+hs));
}
__device__ __forceinline__ float disc(float d,float a,float b,float c,float e) {
    float avg=.25f*(a+b+c+e);
    float x=.5f-d;x=x<0?-x:x;float y=.5f-avg;y=y<0?-y:y;
    return rounded(x<y?avg:d);
}
__device__ __forceinline__ float sum4(float a,float b,float c,float d) {return rounded(a+b+c+d);}
__device__ __forceinline__ float lpf(float c,float a,float b,float d,float e,float diag) {
    return rounded(c+.5f*(a+b+d+e)+.25f*diag);
}
__device__ __forceinline__ float ggrad(float a,float b,float c,float d,float n,float e,float f) {
    float x=a-b,y=c-d,z=n-e,w=d-f;
    return rounded(1e-5f+(x<0?-x:x)+(y<0?-y:y)+(z<0?-z:z)+(w<0?-w:w));
}
__device__ __forceinline__ float gest(float w1,float n1,float l1,float w2,float n2,float l2,float l) {
    float r1=1.0f+(l-l1)/(1e-5f+l+l1),r2=1.0f+(l-l2)/(1e-5f+l+l2);
    return rounded((w1*n1*r1+w2*n2*r2)/(w2+w1));
}
__device__ __forceinline__ float gsel(float gm,float c,float d,float h,float v) {
    return rounded(gm>0?c:d*(h-v)+v);
}
__device__ __forceinline__ float dgrad(float a,float b,float c,float d,float e) {
    float x=a-b,y=a-c,z=d-e;
    return rounded(1e-5f+(x<0?-x:x)+(y<0?-y:y)+(z<0?-z:z));
}
__device__ __forceinline__ float dest(float w1,float a1,float g1,float w2,float a2,float g2) {
    return rounded((w1*(a1-g1)+w2*(a2-g2))/(w1+w2));
}
__device__ __forceinline__ float other(float g,float d,float q,float p) {return rounded(g+d*(q-p)+p);}
__device__ __forceinline__ float rbsel(float own,float opp,float c,float o) {return rounded(own>0?c:(opp>0?o:0.0f));}
__device__ __forceinline__ float cgrad(float g,float g2,float c1,float c2,float c3) {
    float x=g-g2,y=c1-c2,z=c1-c3;return rounded(1e-5f+(x<0?-x:x)+(y<0?-y:y)+(z<0?-z:z));
}
__device__ __forceinline__ float csel(float gm,float g,float d,float h,float v,float ch) {
    return rounded(gm>0?g+d*(h-v)+v:ch);
}
__device__ __forceinline__ int reflect(int p,int n) {return p<0?-p:(p>=n?2*n-2-p:p);}

#define T 32
// a buffer over image rows/columns [t0 - halo, t0 + T + halo), read with each image's own reflection
struct Sh {
    const float* p; int y0,x0,W,h,w,y,x;
    __device__ __forceinline__ float operator()(int dy,int dx) const {
        return p[(reflect(y+dy,h)-y0)*W+(reflect(x+dx,w)-x0)];
    }
};
#define SH(buf,halo) Sh{buf,ty0-(halo),tx0-(halo),T+2*(halo),h,w,y,x}
// every position of the tile with this border inside the image; t indexes the buffer
#define FOR(halo) for(int t=threadIdx.x,W_=T+2*(halo);t<W_*W_;t+=blockDim.x) { \
    int y=ty0-(halo)+t/W_,x=tx0-(halo)+t%W_; if(y<0||y>=h||x<0||x>=w) continue;
#define END }
// the same, only red/blue (site 1) or green (site 0) positions
#define SITES(halo,site) for(int k=threadIdx.x,W_=T+2*(halo),HW=(W_+1)/2;k<W_*HW;k+=blockDim.x) { \
    int ly=k/HW,y=ty0-(halo)+ly,lx=2*(k%HW)+((((site)^gpar)+y+tx0-(halo))&1); if(lx>=W_) continue; \
    int x=tx0-(halo)+lx,t=ly*W_+lx; if(y<0||y>=h||x<0||x>=w) continue;
__device__ __forceinline__ float adu(float v,const float* peds,int batch) {
    return peds?__fsub_rn(__fmul_rn(v,65535.0f),peds[batch]):v;
}
extern "C" __global__ void rcd_tile(const float* raw,const float* gm,const float* rm,const float* bm,
                                   float* rgb,int gs,int rs,int bs,const float* peds,int h,int w) {
    __shared__ float sm[12236];
    float* R=sm;            // raw, border 10: 52 x 52
    float* X=R+2704;        // V/H bands (7), then P/Q bands (5), then low-pass (7), then red/blue (3)
    float* DVH=X+4232;      // V/H ratio, border 6: 44 x 44
    float* DPQ=DVH+1936;    // P/Q ratio, border 4: 40 x 40
    float* GR=DPQ+1600;     // greens, border 5: 42 x 42
    int tilesx=(w+T-1)/T,batch=blockIdx.y,gpar=gm[0]>0?0:1;  // parity of y + x at green sites
    int ty0=(blockIdx.x/tilesx)*T,tx0=(blockIdx.x%tilesx)*T;
    long plane=(long)h*w;
    const float* src=raw+batch*plane;
    FOR(10) R[t]=src[(long)y*w+x]; END
    __syncthreads();
    FOR(7) Sh s=SH(R,10);float c=s(0,0);
        X[t]=band(s(-3,0),s(-1,0),s(1,0),s(3,0),s(-2,0),s(2,0),c);
        X[2116+t]=band(s(0,-3),s(0,-1),s(0,1),s(0,3),s(0,-2),s(0,2),c);
    END
    __syncthreads();
    FOR(6) Sh v=SH(X,7),hor=SH(X+2116,7);
        DVH[t]=ratio(v(-1,0),v(0,0),v(1,0),hor(0,-1),hor(0,0),hor(0,1));
    END
    __syncthreads();
    SITES(5,1) Sh s=SH(R,10);float c=s(0,0);
        X[t]=band(s(-3,-3),s(-1,-1),s(1,1),s(3,3),s(-2,-2),s(2,2),c);
        X[1764+t]=band(s(-3,3),s(-1,1),s(1,-1),s(3,-3),s(-2,2),s(2,-2),c);
    END
    __syncthreads();
    SITES(4,1) Sh pd=SH(X,5),qd=SH(X+1764,5);
        DPQ[t]=ratio(pd(-1,-1),pd(0,0),pd(1,1),qd(-1,1),qd(0,0),qd(1,-1));
    END
    __syncthreads();
    SITES(7,1) Sh s=SH(R,10);
        X[t]=lpf(s(0,0),s(-1,0),s(1,0),s(0,-1),s(0,1),sum4(s(-1,-1),s(-1,1),s(1,-1),s(1,1)));
    END
    __syncthreads();
    SITES(5,0) GR[t]=R[(y-ty0+10)*(T+20)+(x-tx0+10)]; END
    SITES(5,1) Sh s=SH(R,10),L=SH(X,7),D=SH(DVH,6);float c=s(0,0),gv=gm[y*gs+x];
        float d=disc(D(0,0),D(-1,-1),D(-1,1),D(1,-1),D(1,1));
        float ng=ggrad(s(-1,0),s(1,0),c,s(-2,0),s(-1,0),s(-3,0),s(-4,0));
        float sg=ggrad(s(-1,0),s(1,0),c,s(2,0),s(1,0),s(3,0),s(4,0));
        float wg=ggrad(s(0,-1),s(0,1),c,s(0,-2),s(0,-1),s(0,-3),s(0,-4));
        float eg=ggrad(s(0,-1),s(0,1),c,s(0,2),s(0,1),s(0,3),s(0,4));
        float v=gest(sg,s(-1,0),L(-2,0),ng,s(1,0),L(2,0),L(0,0));
        float hor=gest(wg,s(0,1),L(0,2),eg,s(0,-1),L(0,-2),L(0,0));
        GR[t]=gsel(gv,c,d,hor,v);
    END
    __syncthreads();
    SITES(3,0) X[t]=0;X[1444+t]=0; END
    SITES(3,1) float own=rm[y*rs+x],opp=bm[y*bs+x];
        Sh s=SH(R,10),G=SH(GR,5),D=SH(DPQ,4);float g=G(0,0);
        float d=disc(D(0,0),D(-1,-1),D(-1,1),D(1,-1),D(1,1));
        float nw=dgrad(s(-1,-1),s(1,1),s(-3,-3),g,G(-2,-2));
        float ne=dgrad(s(-1,1),s(1,-1),s(-3,3),g,G(-2,2));
        float sw=dgrad(s(1,-1),s(-1,1),s(3,-3),g,G(2,-2));
        float se=dgrad(s(1,1),s(-1,-1),s(3,3),g,G(2,2));
        float p=dest(nw,s(1,1),G(1,1),se,s(-1,-1),G(-1,-1));
        float q=dest(ne,s(1,-1),G(1,-1),sw,s(-1,1),G(-1,1));
        float o=other(g,d,q,p),c=s(0,0);
        X[t]=rbsel(own,opp,c,o);
        X[1444+t]=rbsel(opp,own,c,o);
    END
    __syncthreads();
    SITES(0,1) long o=batch*3*plane+(long)y*w+x;
        rgb[o+plane]=adu(SH(GR,5)(0,0),peds,batch);
        rgb[o]=adu(SH(X,3)(0,0),peds,batch);
        rgb[o+2*plane]=adu(SH(X+1444,3)(0,0),peds,batch);
    END
    SITES(0,0) Sh G=SH(GR,5);float g=G(0,0),gv=gm[y*gs+x];
        long o=batch*3*plane+(long)y*w+x;
        rgb[o+plane]=adu(g,peds,batch);
        Sh D=SH(DVH,6);float d=disc(D(0,0),D(-1,-1),D(-1,1),D(1,-1),D(1,1));
        for(int ch=0;ch<2;ch++) {
            Sh C=SH(X+ch*1444,3);
            float ng=cgrad(g,G(-2,0),C(-1,0),C(1,0),C(-3,0));
            float sg=cgrad(g,G(2,0),C(1,0),C(-1,0),C(3,0));
            float wg=cgrad(g,G(0,-2),C(0,-1),C(0,1),C(0,-3));
            float eg=cgrad(g,G(0,2),C(0,1),C(0,-1),C(0,3));
            float v=dest(ng,C(1,0),G(1,0),sg,C(-1,0),G(-1,0));
            float hor=dest(eg,C(0,-1),G(0,-1),wg,C(0,1),G(0,1));
            rgb[o+ch*2*plane]=adu(csel(gv,g,d,hor,v,C(0,0)),peds,batch);
        }
    END
}
"""

_module = None


def module():
    global _module
    if _module is None:
        _module = Module(SOURCE, fmad=True)
    return _module


def rcd_demosaic_staged(torch, F, cfa, mr, mg, mb, pedestals=None):
    """pedestals: optional float32 tensor, one per batch item: the result is then
    rgb * 65535 - pedestal (ADU), as computing it in torch afterwards."""
    h, w = cfa.shape[-2:]
    if not cfa.is_cuda or cfa.dtype != torch.float32 or min(h, w) <= 4:
        from rcd_fused import rcd_demosaic_fused

        rgb = rcd_demosaic_fused(torch, F, cfa, mr, mg, mb)
        if pedestals is None:
            return rgb
        return rgb * 65535 - pedestals.reshape(-1, *([1] * (rgb.ndim - 1)))
    cfa = cfa.contiguous()
    masks = [m.to(dtype=cfa.dtype) for m in (mr, mg, mb)]
    masks = [m if m.stride(-1) == 1 else m.contiguous() for m in masks]
    rs, gs, bs = [m.stride(-2) for m in masks]
    batch = cfa.numel() // (h * w)
    rgb = torch.empty((batch, 3, h, w), device=cfa.device, dtype=cfa.dtype)
    peds = None if pedestals is None else pedestals.to(device=cfa.device, dtype=torch.float32).contiguous()
    if peds is not None and peds.numel() != batch:
        raise ValueError("One pedestal per batch item expected")
    tiles = ((h + 31) // 32) * ((w + 31) // 32)
    module().launch("rcd_tile", (tiles, batch), 512, [cfa, masks[1], masks[0], masks[2], rgb, gs, rs, bs, peds, h, w])
    return rgb.reshape(*cfa.shape[:-2], 3, h, w)


rcd_demosaic_staged.folds_scale = True  # warp_tile: calibrate_pad(unit=True), pedestals= here
