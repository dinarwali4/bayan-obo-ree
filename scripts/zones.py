import numpy as np, json
from scipy import ndimage as ndi
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; LON,LAT=np.meshgrid(glon,glat); Zd=np.load('data/dem_emitgrid.npy')
dates=['20230203','20230423','20231015']
def rz(a): med=np.nanmedian(a); mad=1.4826*np.nanmedian(abs(a-med)); return (a-med)/mad
Z5=np.stack([rz(e[f'{d}_F580']) for d in dates]); Z7=np.stack([rz(e[f'{d}_F740']) for d in dates]); D5=np.stack([e[f'{d}_F580'] for d in dates])
red=[]
for d in dates:
    c=np.load(f'data/emit_aoi_{d}.npz'); wl=c['wl']; cu=c['cube']
    red.append(cu[...,int(np.argmin(abs(wl-650)))].astype(float)/cu[...,int(np.argmin(abs(wl-555)))].astype(float))
red=np.nanmean(red,0)
two=((np.nan_to_num(Z5)>2)&(np.nan_to_num(Z7)>2)).sum(0)
nd=(two>=2)&(red<1.20)
score=np.nanmean(np.where(np.isfinite(Z5),(Z5+Z7)/2,np.nan),0)
np.savez_compressed('data/nd_twoband.npz',nd=nd,score=score.astype('float32'),two=two.astype('uint8'),red=red.astype('float32'))
def box(a,b,c,d): return (LON>=a)&(LON<=c)&(LAT>=b)&(LAT<=d)
ctrl={'Town':box(109.955,41.765,109.985,41.778),'Steppe N':box(109.90,41.815,109.95,41.835),'Granite SE':box(110.03,41.73,110.09,41.76)}
for n,m in ctrl.items(): print(f'two-band Nd cells in control {n}: {int((nd&m).sum())}/{int(m.sum())}')
mine=box(109.950,41.788,110.010,41.810)
lab,n=ndi.label(nd&mine,structure=np.ones((3,3))); sz=ndi.sum(nd&mine,lab,range(1,n+1))
pix_ha=0.0005*111320*np.cos(np.radians(41.8))*0.0005*110574/1e4
Z=[]
for k in np.argsort(-sz)+1:
    m=lab==k
    if m.sum()<2: continue
    w=np.where(m,score,-np.inf); i=np.unravel_index(np.argmax(w),w.shape)
    Z.append(dict(cells=int(m.sum()),area_ha=round(float(m.sum()*pix_ha),1),centroid=[round(float(LAT[m].mean()),5),round(float(LON[m].mean()),5)],
      peak=[round(float(LAT[i]),5),round(float(LON[i]),5)],peak_score=round(float(score[i]),1),mean_score=round(float(np.nanmean(score[m])),1),
      mean_F580_pct=round(float(np.nanmean(D5[:,m]))*100,2),elev_range=[int(np.nanmin(Zd[m])),int(np.nanmax(Zd[m]))],
      bbox=[round(float(LON[m].min()),5),round(float(LAT[m].min()),5),round(float(LON[m].max()),5),round(float(LAT[m].max()),5)]))
for z in Z: print(z)
print('pixel area ha',round(pix_ha,3))
json.dump(Z,open('zones_insitu.json','w'),indent=1)
