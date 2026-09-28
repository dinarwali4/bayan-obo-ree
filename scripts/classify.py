import numpy as np, json
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']
dates=['20230203','20230423','20231015']
Z5={d:np.load(f'data/z580_{d}.npy') if False else None for d in dates}
def rz(a): med=np.nanmedian(a); mad=1.4826*np.nanmedian(abs(a-med)); return (a-med)/mad
Z7={d:rz(e[f'{d}_F740']) for d in dates}; Z5={d:rz(e[f'{d}_F580']) for d in dates}
C={d:np.load(f'data/emit_aoi_{d}.npz') for d in dates}
T=json.load(open('targets_strong.json'))
visual={'T01':'small compound (S of tailings)','T02':'processing plant (blue roofs, tanks)','T03':'red-roof buildings, W pit rim','T04':'industrial compound by tailings pond',
 'T05':'red-roof buildings','T06':'red-roof building at rail loading, S of Main pit','T07':'industrial compound by tailings pond','T08':'red-roof housing (town)',
 'T09':'Hot1 excavation benches','T10':'Main pit benches','T11':'Hot1 excavation benches','T12':'Main pit benches','T13':'industrial yard','T14':'dark stockpile by rail loop',
 'T15':'industrial site by pond (town NE)','T16':'East pit W rim haul area','T17':'Hot1 excavation benches','T18':'industrial compound by tailings pond','T19':'red-roof housing (town)',
 'T20':'Hot1 excavation benches','T21':'red-roof housing (town)'}
for t in T:
    la,lo=t['peak']; i=int(round((glat[0]-la)/0.0005)); j=int(round((lo-glon[0])/0.0005))
    sl=(slice(max(i-1,0),i+2),slice(max(j-1,0),j+2))
    z7=[float(np.nanmax(Z7[d][sl])) for d in dates]; z5=[float(np.nanmax(Z5[d][sl])) for d in dates]
    red=[]
    for d in dates:
        wl=C[d]['wl']; cu=C[d]['cube']; r=cu[sl][...,int(np.argmin(abs(wl-650)))].astype(float); g=cu[sl][...,int(np.argmin(abs(wl-555)))].astype(float)
        red.append(float(np.nanmean(r/g)))
    t['z740_by_date']=[round(x,1) for x in z7]; t['z740_dates_gt2']=int(sum(x>2 for x in z7)); t['redness_650_555']=round(float(np.mean(red)),2)
    t['surface']=visual[t['id']]
    roof='red-roof' in t['surface'] or 'housing' in t['surface']
    insitu=any(k in t['surface'] for k in ('pit','excavation'))
    t['class']=('A in-situ ore exposure' if insitu else 'C false positive (roof)' if roof else 'B processed / stockpiled REE material')
    print(f"{t['id']} {t['peak']} z580 {t['peak_z580']:5.1f} z740>2 on {t['z740_dates_gt2']}/3  red {t['redness_650_555']:.2f}  {t['class']:36s} {t['surface']}")
json.dump(T,open('targets_classified.json','w'),indent=1)
# does redness alone separate roofs from the rest?
import collections
g=collections.defaultdict(list)
for t in T: g[t['class'][0]].append(t['redness_650_555'])
print({k:(round(min(v),2),round(max(v),2)) for k,v in g.items()})
