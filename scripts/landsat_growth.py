import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pyproj import Transformer
L=np.load('data/landsat.npz'); yrs=L['years']; S=L['stack']; T=L['T']
n,_,H,W=S.shape
r,c=np.mgrid[0:H,0:W]; lon,lat=Transformer.from_crs(32649,4326,always_xy=True).transform(T[2]+(c+.5)*30,T[5]-(r+.5)*30)
zone=(lon>=109.925)&(lon<=110.03)&(lat>=41.783)&(lat<=41.818)     # pits + dumps, north of town
res={}
for k,y in enumerate(yrs):
    red,grn,blu,nir,sw1,sw2=S[k]
    vis=(red+grn+blu)/3; ndvi=(nir-red)/(nir+red); yel=(red-blu)/(red+blu)
    dist=(vis<0.115)&(yel<0.28)&(ndvi<0.2)&np.isfinite(vis)
    valid=np.isfinite(vis)&zone
    res[int(y)]=dict(km2=round(float((dist&zone).sum()*900/1e6),2),valid=round(float(valid.sum()/zone.sum()),3))
    np.save(f'data/disturbed_{y}.npy',dist&zone)
for y,v in res.items(): print(y,v)
json.dump(res,open('landsat_growth.json','w'),indent=1)
ys=[y for y in res if res[y]['valid']>0.9]; a=[res[y]['km2'] for y in ys]
fig,ax=plt.subplots(figsize=(8,3.6),dpi=130); ax.plot(ys,a,'-o',color='#8a3b12',ms=4); ax.fill_between(ys,a,color='#8a3b12',alpha=.12)
ax.set_ylabel('Dark mined ground, km²'); ax.set_title('Bayan Obo Main/East complex: pit + waste-dump footprint (Landsat, 30 m)',fontsize=10)
ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.25); plt.tight_layout(); plt.savefig('fig/footprint_growth.png')
