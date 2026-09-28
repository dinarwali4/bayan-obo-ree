import numpy as np
from scipy import ndimage as ndi
z=np.load('data/emit_nd.npz'); dates=list(z['dates']); glon,glat=z['glon'],z['glat']; LON,LAT=np.meshgrid(glon,glat)
def box(a,b,c,d): return (LON>=a)&(LON<=c)&(LAT>=b)&(LAT<=d)
regions={'Main pit':box(109.955,41.792,109.972,41.803),'East pit':box(109.975,41.795,109.990,41.805),'Hot1 (E of East pit)':box(109.993,41.801,110.004,41.808),
 'West workings':box(109.885,41.786,109.910,41.796),'Town (neg)':box(109.955,41.765,109.985,41.778),'Steppe N (neg)':box(109.90,41.815,109.95,41.835),'Granite SE (neg)':box(110.03,41.73,110.09,41.76)}
def rz(a): med=np.nanmedian(a); mad=1.4826*np.nanmedian(abs(a-med)); return (a-med)/mad
HOT={}
for d in dates:
    cube=np.load(f'data/emit_aoi_{d}.npz'); wl=cube['wl']; C=cube['cube']
    bi=lambda w:int(np.argmin(abs(wl-w)))
    g=C[...,bi(560)].astype(float); s=C[...,bi(1610)].astype(float); ndsi=(g-s)/(g+s)
    print(f'== {d}: snow-like (NDSI>0.4) frac {np.nanmean(ndsi>0.4):.3f}')
    F740=z[f'{d}_F740']; F580=z[f'{d}_F580']; F800=z[f'{d}_F800']
    F740=np.where(ndsi>0.4,np.nan,F740); F580=np.where(ndsi>0.4,np.nan,F580)
    Z7,Z5=rz(F740),rz(F580)
    for n,m in regions.items(): print(f'   {n:22s} z740 {np.nanmedian(Z7[m]):+5.2f}  z580 {np.nanmedian(Z5[m]):+5.2f}')
    HOT[d]=(np.nan_to_num(Z7)>3)&(np.nan_to_num(Z5)>2)
    np.save(f'data/hot_{d}.npy',HOT[d]); np.save(f'data/z740_{d}.npy',Z7.astype('float32')); np.save(f'data/z580_{d}.npy',Z5.astype('float32'))
if len(dates)>=2:
    a,b=HOT[dates[0]],HOT[dates[1]]
    # allow 1-cell (≈50 m) geolocation tolerance
    bd=ndi.binary_dilation(b,iterations=1)
    inter=(a&bd).sum(); exp=a.sum()*bd.mean()
    print(f'hot {dates[0]}: {a.sum()}  hot {dates[1]}: {b.sum()}  overlap(±1 cell) {inter}  expected by chance {exp:.1f}  lift {inter/max(exp,1e-9):.1f}x')
    Za=np.load(f'data/z740_{dates[0]}.npy'); Zb=np.load(f'data/z740_{dates[1]}.npy'); m=np.isfinite(Za)&np.isfinite(Zb)
    print('pixelwise r(z740)',round(float(np.corrcoef(Za[m],Zb[m])[0,1]),3))
