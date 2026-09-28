import numpy as np, json
from scipy import ndimage as ndi
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; LON,LAT=np.meshgrid(glon,glat); Z=np.load('data/dem_emitgrid.npy')
dates=['20230203','20230423','20231015','20230626']
def rz(a): med=np.nanmedian(a); mad=1.4826*np.nanmedian(abs(a-med)); return (a-med)/mad
Z5=np.stack([rz(e[f'{d}_F580']) for d in dates]); Z7=np.stack([rz(e[f'{d}_F740']) for d in dates]); D5=np.stack([e[f'{d}_F580'] for d in dates])
val=np.isfinite(Z5); nval=val.sum(0)
hit=(np.nan_to_num(Z5)>2)
nhit=hit.sum(0); mz=np.nanmean(np.where(val,Z5,np.nan),0); sup740=(np.nan_to_num(Z7)>1.5).sum(0)
pers=(nhit>=2)&(mz>2.0)&(nval>=2)
# chance level: circularly shift each date's hit map by random offsets and recount cells with >=2 hits
rng=np.random.default_rng(1); sims=[]
for _ in range(200):
    hs=np.stack([np.roll(np.roll(hit[k],rng.integers(20,200),0),rng.integers(20,500),1) for k in range(len(dates))])
    sims.append(int(((hs.sum(0)>=2)).sum()))
print('cells hit on >=2 dates: observed',int((nhit>=2).sum()),' chance (spatial shuffle) mean',np.mean(sims),'max',max(sims))
def box(a,b,c,d): return (LON>=a)&(LON<=c)&(LAT>=b)&(LAT<=d)
ctrl={'Town':box(109.955,41.765,109.985,41.778),'Steppe N':box(109.90,41.815,109.95,41.835),'Granite SE':box(110.03,41.73,110.09,41.76)}
for n,m in ctrl.items(): print(f'  persistent cells in control {n}: {int((pers&m).sum())} of {int(m.sum())}')
lab,n=ndi.label(pers,structure=np.ones((3,3))); sz=ndi.sum(pers,lab,range(1,n+1))
T=[]
for k in np.argsort(-sz)+1:
    m=lab==k
    if m.sum()<3: continue
    w=np.where(m,mz,-np.inf); i=np.unravel_index(np.argmax(w),w.shape)
    T.append(dict(cells=int(m.sum()),area_ha=round(float(m.sum()*0.0005*111320*np.cos(np.radians(41.8))*0.0005*110574/1e4),1),
      centroid=[round(float(LAT[m].mean()),5),round(float(LON[m].mean()),5)],peak=[round(float(LAT[i]),5),round(float(LON[i]),5)],
      peak_elev_m=int(Z[i]),mean_z580=round(float(np.nanmean(mz[m])),2),peak_z580=round(float(mz[i]),2),
      dates_hit_at_peak=int(nhit[i]),dates_valid_at_peak=int(nval[i]),F580_depth_pct=round(float(np.nanmean(D5[:,m]))*100,2),z740_support_dates=int(sup740[i])))
for t in T[:20]: print(t)
json.dump(T,open('targets.json','w'),indent=1)
np.savez_compressed('data/persist.npz',pers=pers,mz=mz.astype('float32'),nhit=nhit.astype('uint8'),nval=nval.astype('uint8'))
