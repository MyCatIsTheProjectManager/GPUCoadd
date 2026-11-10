"""Experimental five-stage RCD: reflect each intermediate at its own image boundary.
RCD is by Luis Sanz Rodriguez, see THIRD-PARTY-NOTICES.txt.

The small arithmetic helpers match rcd_fused's Jiterator operations. A register
barrier at their outputs preserves the rounding boundaries between those kernels.
No padded image copies or full-image gradient/estimate temporaries are needed.
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
struct Image {
    const float* p; int h,w,y,x;
    __device__ __forceinline__ float operator()(int dy,int dx) const {
        return p[reflect(y+dy,h)*w+reflect(x+dx,w)];
    }
};
#define LOCATE int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; \
    int plane=h*w,batch=i/plane,j=i%plane,y=j/w,x=j%w

extern "C" __global__ void bands(const float* raw,float* bands,int n,int h,int w) {
    LOCATE;Image s={raw+batch*plane,h,w,y,x};float c=s(0,0);
    float* out=bands+batch*5*plane+j;
    out[0*plane]=band(s(-3,0),s(-1,0),s(1,0),s(3,0),s(-2,0),s(2,0),c);
    out[1*plane]=band(s(0,-3),s(0,-1),s(0,1),s(0,3),s(0,-2),s(0,2),c);
    out[2*plane]=band(s(-3,-3),s(-1,-1),s(1,1),s(3,3),s(-2,-2),s(2,2),c);
    out[3*plane]=band(s(-3,3),s(-1,1),s(1,-1),s(3,-3),s(-2,2),s(2,-2),c);
    out[4*plane]=lpf(c,s(-1,0),s(1,0),s(0,-1),s(0,1),sum4(s(-1,-1),s(-1,1),s(1,-1),s(1,1)));
}
extern "C" __global__ void directions(const float* bands,float* dirs,int n,int h,int w) {
    LOCATE;const float* p=bands+batch*5*plane;
    Image v={p,h,w,y,x},hor={p+plane,h,w,y,x},pd={p+2*plane,h,w,y,x},qd={p+3*plane,h,w,y,x};
    dirs[batch*2*plane+j]=ratio(v(-1,0),v(0,0),v(1,0),hor(0,-1),hor(0,0),hor(0,1));
    dirs[batch*2*plane+plane+j]=ratio(pd(-1,-1),pd(0,0),pd(1,1),qd(-1,1),qd(0,0),qd(1,-1));
}
extern "C" __global__ void green(const float* raw,const float* bands,const float* dirs,const float* gm,
                                float* greens,float* vh,int gs,int n,int h,int w) {
    LOCATE;Image s={raw+batch*plane,h,w,y,x},L={bands+(batch*5+4)*plane,h,w,y,x};
    Image D={dirs+batch*2*plane,h,w,y,x};float c=s(0,0);
    float d=disc(D(0,0),D(-1,-1),D(-1,1),D(1,-1),D(1,1));vh[i]=d;
    if(gm[y*gs+x]>0) {greens[i]=c;return;}
    float ng=ggrad(s(-1,0),s(1,0),c,s(-2,0),s(-1,0),s(-3,0),s(-4,0));
    float sg=ggrad(s(-1,0),s(1,0),c,s(2,0),s(1,0),s(3,0),s(4,0));
    float wg=ggrad(s(0,-1),s(0,1),c,s(0,-2),s(0,-1),s(0,-3),s(0,-4));
    float eg=ggrad(s(0,-1),s(0,1),c,s(0,2),s(0,1),s(0,3),s(0,4));
    float v=gest(sg,s(-1,0),L(-2,0),ng,s(1,0),L(2,0),L(0,0));
    float hor=gest(wg,s(0,1),L(0,2),eg,s(0,-1),L(0,-2),L(0,0));
    greens[i]=gsel(gm[y*gs+x],c,d,hor,v);
}
extern "C" __global__ void opposite(const float* raw,const float* greens,const float* dirs,
                                   const float* rm,const float* bm,float* rb,int rs,int bs,int n,int h,int w) {
    LOCATE;Image s={raw+batch*plane,h,w,y,x},G={greens+batch*plane,h,w,y,x};
    float own=rm[y*rs+x],opp=bm[y*bs+x];
    if(own<=0 && opp<=0) {rb[batch*2*plane+j]=0;rb[batch*2*plane+plane+j]=0;return;}
    Image D={dirs+(batch*2+1)*plane,h,w,y,x};float g=G(0,0);
    float d=disc(D(0,0),D(-1,-1),D(-1,1),D(1,-1),D(1,1));
    float nw=dgrad(s(-1,-1),s(1,1),s(-3,-3),g,G(-2,-2));
    float ne=dgrad(s(-1,1),s(1,-1),s(-3,3),g,G(-2,2));
    float sw=dgrad(s(1,-1),s(-1,1),s(3,-3),g,G(2,-2));
    float se=dgrad(s(1,1),s(-1,-1),s(3,3),g,G(2,2));
    float p=dest(nw,s(1,1),G(1,1),se,s(-1,-1),G(-1,-1));
    float q=dest(ne,s(1,-1),G(1,-1),sw,s(-1,1),G(-1,1));
    float o=other(g,d,q,p),c=s(0,0);
    rb[batch*2*plane+j]=rbsel(own,opp,c,o);
    rb[batch*2*plane+plane+j]=rbsel(opp,own,c,o);
}
// peds: null, or a pedestal per batch item: write v * 65535 - pedestal (ADU), two roundings as torch
__device__ __forceinline__ float adu(float v,const float* peds,int batch) {
    return peds?__fsub_rn(__fmul_rn(v,65535.0f),peds[batch]):v;
}
extern "C" __global__ void finish(const float* greens,const float* rb,const float* vh,const float* gm,
                                 float* rgb,int gs,const float* peds,int n,int h,int w) {
    LOCATE;Image G={greens+batch*plane,h,w,y,x};float g=G(0,0),d=vh[i];
    rgb[batch*3*plane+plane+j]=adu(g,peds,batch);
    if(gm[y*gs+x]<=0) {
        rgb[batch*3*plane+j]=adu(rb[batch*2*plane+j],peds,batch);
        rgb[batch*3*plane+2*plane+j]=adu(rb[batch*2*plane+plane+j],peds,batch);return;
    }
    for(int ch=0;ch<2;ch++) {
        Image C={rb+(batch*2+ch)*plane,h,w,y,x};
        float ng=cgrad(g,G(-2,0),C(-1,0),C(1,0),C(-3,0));
        float sg=cgrad(g,G(2,0),C(1,0),C(-1,0),C(3,0));
        float wg=cgrad(g,G(0,-2),C(0,-1),C(0,1),C(0,-3));
        float eg=cgrad(g,G(0,2),C(0,1),C(0,-1),C(0,3));
        float v=dest(ng,C(1,0),G(1,0),sg,C(-1,0),G(-1,0));
        float hor=dest(eg,C(0,-1),G(0,-1),wg,C(0,1),G(0,1));
        rgb[batch*3*plane+ch*2*plane+j]=adu(csel(gm[y*gs+x],g,d,hor,v,C(0,0)),peds,batch);
    }
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
    n = cfa.numel()
    batch = n // (h * w)
    alloc = lambda channels: torch.empty((batch, channels, h, w), device=cfa.device, dtype=cfa.dtype)
    bands, dirs = alloc(5), alloc(2)
    g, vh = alloc(1), alloc(1)
    rb, rgb = alloc(2), alloc(3)
    mod = module()
    launch = lambda name, args: mod.launch(name, (n + 127) // 128, 128, [*args, n, h, w])
    launch("bands", [cfa, bands])
    launch("directions", [bands, dirs])
    launch("green", [cfa, bands, dirs, masks[1], g, vh, gs])
    del bands
    launch("opposite", [cfa, g, dirs, masks[0], masks[2], rb, rs, bs])
    del dirs
    peds = None if pedestals is None else pedestals.to(device=cfa.device, dtype=torch.float32).contiguous()
    if peds is not None and peds.numel() != batch:
        raise ValueError("One pedestal per batch item expected")
    launch("finish", [g, rb, vh, masks[1], rgb, gs, peds])
    return rgb.reshape(*cfa.shape[:-2], 3, h, w)


rcd_demosaic_staged.folds_scale = True  # warp_tile: calibrate_pad(unit=True), pedestals= here
