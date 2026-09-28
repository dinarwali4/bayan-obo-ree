# Neodymium (REE) absorption mapping from EMIT L2A reflectance, orthorectified by GLT gather onto a common 0.0005 deg grid
import numpy as np, netCDF4, glob, json, sys
AOI=(109.80,41.72,110.10,41.84); RES=0.0005
NX=int(round((AOI[2]-AOI[0])/RES)); NY=int(round((AOI[3]-AOI[1])/RES))
glon=AOI[0]+(np.arange(NX)+.5)*RES; glat=AOI[3]-(np.arange(NY)+.5)*RES
feat_def={  # name: (centre band wl list, left shoulder, right shoulder)
 'F740':([738.2,745.7],723.3,783.0),
 'F800':([797.9,805.4],783.0,835.2),
 'F580':([581.7],566.8,604.0),
 'F870':([865.1],842.7,894.9)}
out={}
for f in sorted(glob.glob('data/emit/EMIT_L2A_RFL_001_*.nc')):
    date=f.split('_')[4][:8]
    import os
    if os.path.exists(f'data/emit_aoi_{date}.npz') and not os.path.exists(f): pass
    try: ds=netCDF4.Dataset(f); ds.set_auto_mask(False)
    except OSError: print(date,'incomplete, skipped'); continue
    wl=ds['sensor_band_parameters/wavelengths'][:]
    def bi(w): return int(np.argmin(abs(wl-w)))
    need=sorted({bi(w) for c,l,r in feat_def.values() for w in c+[l,r]}|{bi(671.1),bi(797.9),bi(560),bi(2200)})
    gx=ds['location/glt_x'][:]; gy=ds['location/glt_y'][:]; gt=ds.getncattr('geotransform')
    # grid mapping: for each target cell find ortho cell
    oj=((glon-gt[0])/gt[1]).astype(int); oi=((glat-gt[3])/gt[5]).astype(int)
    OI,OJ=np.meshgrid(oi,oj,indexing='ij')
    inside=(OI>=0)&(OJ>=0)&(OI<gx.shape[0])&(OJ<gx.shape[1])
    X=np.zeros((NY,NX),int); Y=np.zeros((NY,NX),int)
    X[inside]=gx[OI[inside],OJ[inside]]; Y[inside]=gy[OI[inside],OJ[inside]]
    v=inside&(X>0)&(Y>0)
    print(date,'AOI cells covered',round(v.mean(),3),flush=True)
    if v.mean()<0.2: continue
    rows=Y[v]-1; cols=X[v]-1; r0,r1=rows.min(),rows.max()+1
    import os
    aoi=f'data/emit_aoi_{date}.npz'
    if os.path.exists(aoi):
        full=np.load(aoi)['cube'].astype('float32')
    else:
        R=ds['reflectance'][r0:r1,:,:]
        full=np.full((NY,NX,R.shape[2]),np.nan,'float32'); full[v]=R[rows-r0,cols]
        np.savez_compressed(aoi,cube=full.astype('float16'),wl=wl)
    cube=full[:,:,need]
    cube[cube<=-9000]=np.nan
    m=netCDF4.Dataset(f.replace('_RFL_','_MASK_')); m.set_auto_mask(False)
    agg=m['mask'][:,:,7]; mk=np.zeros((NY,NX)); mk[v]=agg[rows,cols]
    B={w:cube[...,need.index(bi(w))] for w in [671.1,797.9,560,2200]}
    ndvi=(B[797.9]-B[671.1])/(B[797.9]+B[671.1])
    valid=v&(mk==0)&np.isfinite(B[671.1])&(B[671.1]>0.03)&(ndvi<0.25)
    res={'valid':valid,'ndvi':ndvi,'alb':B[671.1],'swir2200':B[2200]}
    for n,(c,l,r) in feat_def.items():
        Rc=np.nanmean([cube[...,need.index(bi(w))] for w in c],0)
        L=cube[...,need.index(bi(l))]; Rr=cube[...,need.index(bi(r))]
        wc=np.mean([wl[bi(w)] for w in c]); t=(wc-wl[bi(l)])/(wl[bi(r)]-wl[bi(l)])
        cont=L+(Rr-L)*t
        d=1-Rc/cont; d[~valid]=np.nan; res[n]=d.astype('float32')
    out[date]=res
    print(date,'valid',round(valid.mean(),3),{n:round(float(np.nanmedian(res[n])),4) for n in feat_def},flush=True)
np.savez_compressed('data/emit_nd.npz',dates=np.array(list(out)),glon=glon,glat=glat,
    **{f'{d}_{k}':v for d,r in out.items() for k,v in r.items()})
print('saved')
