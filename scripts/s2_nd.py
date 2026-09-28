import numpy as np, json
from pyproj import Transformer
f=np.load('data/s2_raw.npz'); S=f['stack'].astype('float32'); bn=list(f['bandnames']); B=f['bounds']; boa=f['boa']
print('scenes',len(S),'boa applied',set(boa.tolist()))
scl=S[:,bn.index('SCL')]; good=np.isin(scl,[4,5,7])
R=S[:,:10]/10000.0; R[~np.repeat(good[:,None],10,1)]=np.nan; R[R<=0]=np.nan
comp=np.nanmedian(R,0); b={k:comp[i] for i,k in enumerate(bn[:10])}
ndvi=(b['B08']-b['B04'])/(b['B08']+b['B04'])
t=(740-705)/(783-705)
F740=1-b['B06']/(b['B05']+(b['B07']-b['B05'])*t)
# per-scene F740 for persistence
F740s=1-R[:,5]/(R[:,4]+(R[:,6]-R[:,4])*t)
valid=np.isfinite(F740)&(ndvi<0.25)&(b['B04']>0.03)
H,W=F740.shape
print('valid frac',round(valid.mean(),3),'ndvi p50',np.nanmedian(ndvi).round(3))
np.savez_compressed('data/s2_nd.npz',F740=np.where(valid,F740,np.nan).astype('float32'),F740s=F740s.astype('float16'),ndvi=ndvi.astype('float32'),
    rgb=np.stack([b['B04'],b['B03'],b['B02']]).astype('float32'),B=B,valid=valid)
tr=Transformer.from_crs(32649,4326,always_xy=True)
r,c=np.mgrid[0:H,0:W]; lon,lat=tr.transform(B[0]+(c+.5)*20,B[3]-(r+.5)*20)
np.savez_compressed('data/s2_lonlat.npz',lon=lon.astype('float32'),lat=lat.astype('float32'))
def box(a,b_,cc,d): return (lon>=a)&(lon<=cc)&(lat>=b_)&(lat<=d)
regions={'Main pit':box(109.955,41.792,109.972,41.803),'East pit':box(109.975,41.795,109.990,41.805),'Hot1 E of East pit':box(109.993,41.801,110.004,41.808),
         'Town (neg)':box(109.955,41.765,109.985,41.778),'Steppe N (neg)':box(109.90,41.815,109.95,41.835),'Granite SE (neg)':box(110.03,41.73,110.09,41.76)}
for n,m in regions.items(): print(f'{n:20s} F740 median {np.nanmedian(np.where(valid,F740,np.nan)[m]):+.4f}  p90 {np.nanpercentile(np.where(valid,F740,np.nan)[m],90):+.4f}')
