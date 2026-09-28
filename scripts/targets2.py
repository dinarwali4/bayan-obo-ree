import numpy as np, json
from scipy.optimize import curve_fit
from scipy import ndimage as ndi
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; LON,LAT=np.meshgrid(glon,glat); Z=np.load('data/dem_emitgrid.npy')
P=np.load('data/persist.npz'); mz,nhit,nval=P['mz'],P['nhit'],P['nval']
strong=(mz>=5)&(nhit>=3)
lab,n=ndi.label(ndi.binary_dilation(strong,iterations=1),structure=np.ones((3,3)))
cubes={d:np.load(f'data/emit_aoi_{d}.npz') for d in ['20230203','20230423','20231015']}
g=lambda x,a,mu,sig:1-a*np.exp(-(x-mu)**2/(2*sig**2))
def width(m):
    out=[]
    for d,c in cubes.items():
        C=c['cube']; wl=c['wl']; sel=(wl>545)&(wl<625); w=wl[sel]; s=np.nanmean(C[m][:,sel].astype('float32'),0)
        i0,i1=np.argmin(abs(w-552)),np.argmin(abs(w-604)); cr=s/(s[i0]+(s[i1]-s[i0])*(w-w[i0])/(w[i1]-w[i0]))
        try: p,_=curve_fit(g,w,cr,p0=[0.03,578,10],bounds=([0,560,2],[0.3,600,60])); out.append((round(p[0]*100,1),round(p[1],1),round(2.355*p[2],1)))
        except Exception: out.append(None)
    return out
T=[]
for k in range(1,n+1):
    m=(lab==k)&strong
    if m.sum()==0: continue
    w=np.where(m,mz,-np.inf); i=np.unravel_index(np.argmax(w),w.shape)
    T.append(dict(peak=[round(float(LAT[i]),5),round(float(LON[i]),5)],cells=int(m.sum()),peak_z580=round(float(mz[i]),1),mean_z580=round(float(mz[m].mean()),1),
                  dates=f'{int(nhit[i])}/{int(nval[i])}',elev=int(Z[i]),fit_580=width(m)))
T=sorted(T,key=lambda t:-t['peak_z580'])
for j,t in enumerate(T): t['id']=f'T{j+1:02d}'; print(t)
json.dump(T,open('targets_strong.json','w'),indent=1)
