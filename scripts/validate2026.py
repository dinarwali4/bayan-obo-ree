import numpy as np, json
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; LON,LAT=np.meshgrid(glon,glat)
d='20260410'
def rz(a): med=np.nanmedian(a); mad=1.4826*np.nanmedian(abs(a-med)); return (a-med)/mad
z5=rz(e[f'{d}_F580']); z7=rz(e[f'{d}_F740']); v=e[f'{d}_valid']
print('2026 valid frac',round(float(v.mean()),3))
N=np.load('data/nd_twoband.npz'); nd=N['nd']
def box(a,b,c,dd): return (LON>=a)&(LON<=c)&(LAT>=b)&(LAT<=dd)
groups={'2023 two-band Nd cells (all)':nd,'Z1 eastern excavation':nd&box(109.990,41.797,110.010,41.810),'Z2 Main pit':nd&box(109.959,41.796,109.973,41.805),
        'Town (control)':box(109.955,41.765,109.985,41.778),'Steppe N (control)':box(109.90,41.815,109.95,41.835),'Granite SE (control)':box(110.03,41.73,110.09,41.76)}
res={}
for n,m in groups.items():
    mm=m&np.isfinite(z5)
    res[n]=dict(n=int(mm.sum()),z580=round(float(np.nanmedian(z5[mm])),2),z740=round(float(np.nanmedian(z7[mm])),2),both_gt2=round(float(((z5>2)&(z7>2))[mm].mean()),3))
    print(f'{n:32s} n={res[n]["n"]:5d}  median z580 {res[n]["z580"]:+.2f}  z740 {res[n]["z740"]:+.2f}  frac both>2 {res[n]["both_gt2"]:.3f}')
T=json.load(open('targets_classified.json'))
for t in T:
    la,lo=t['peak']; i=int(round((glat[0]-la)/0.0005)); j=int(round((lo-glon[0])/0.0005)); sl=(slice(i-1,i+2),slice(j-1,j+2))
    t['val2026']=dict(z580=round(float(np.nanmax(z5[sl])),1) if np.isfinite(z5[sl]).any() else None,z740=round(float(np.nanmax(z7[sl])),1) if np.isfinite(z7[sl]).any() else None)
    print(t['id'],t['class'][0],t['val2026'])
json.dump(T,open('targets_classified.json','w'),indent=1); json.dump(res,open('validation_2026.json','w'),indent=1)
