# Features from saved AOI cubes (+ EMIT MASK files re-gathered on the same grid)
import numpy as np, netCDF4, glob
AOI=(109.80,41.72,110.10,41.84); RES=0.0005
NX=int(round((AOI[2]-AOI[0])/RES)); NY=int(round((AOI[3]-AOI[1])/RES))
glon=AOI[0]+(np.arange(NX)+.5)*RES; glat=AOI[3]-(np.arange(NY)+.5)*RES
feat_def={'F740':([738.2,745.7],723.3,783.0),'F800':([797.9,805.4],783.0,835.2),'F580':([581.7],566.8,604.0),'F870':([865.1],842.7,894.9)}
out={}
for a in sorted(glob.glob('data/emit_aoi_*.npz')):
    date=a.split('_')[-1][:8]; z=np.load(a); C=z['cube'].astype('float32'); wl=z['wl']
    bi=lambda w:int(np.argmin(abs(wl-w)))
    mf=glob.glob(f'data/emit/EMIT_L2A_MASK_001_{date}*.nc')[0]; m=netCDF4.Dataset(mf); m.set_auto_mask(False)
    gx=m['location/glt_x'][:]; gy=m['location/glt_y'][:]; gt=m.getncattr('geotransform')
    oj=((glon-gt[0])/gt[1]).astype(int); oi=((glat-gt[3])/gt[5]).astype(int); OI,OJ=np.meshgrid(oi,oj,indexing='ij')
    ins=(OI>=0)&(OJ>=0)&(OI<gx.shape[0])&(OJ<gx.shape[1]); X=np.zeros((NY,NX),int); Y=np.zeros((NY,NX),int)
    X[ins]=gx[OI[ins],OJ[ins]]; Y[ins]=gy[OI[ins],OJ[ins]]; v=ins&(X>0)&(Y>0)
    agg=m['mask'][:,:,7]; mk=np.ones((NY,NX)); mk[v]=agg[Y[v]-1,X[v]-1]
    r671=C[...,bi(671.1)]; r798=C[...,bi(797.9)]; ndvi=(r798-r671)/(r798+r671)
    g=C[...,bi(560)]; s=C[...,bi(1610)]; ndsi=(g-s)/(g+s)
    valid=v&(mk==0)&np.isfinite(r671)&(r671>0.03)&(ndvi<0.25)&(ndsi<0.4)
    res={'valid':valid,'ndvi':ndvi,'alb':r671}
    for n,(c,l,r) in feat_def.items():
        Rc=np.nanmean([C[...,bi(w)] for w in c],0); L=C[...,bi(l)]; Rr=C[...,bi(r)]
        t=(np.mean([wl[bi(w)] for w in c])-wl[bi(l)])/(wl[bi(r)]-wl[bi(l)])
        d=1-Rc/(L+(Rr-L)*t); d[~valid]=np.nan; res[n]=d.astype('float32')
    out[date]=res; print(date,'valid',round(float(valid.mean()),3),'cloud-flag frac',round(float((mk[v]>0).mean()),3),'ndvi p50',round(float(np.nanmedian(ndvi)),3))
np.savez_compressed('data/emit_nd.npz',dates=np.array(list(out)),glon=glon,glat=glat,**{f'{d}_{k}':v for d,r in out.items() for k,v in r.items()})
