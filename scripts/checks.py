import numpy as np, rasterio
from rasterio.merge import merge
from scipy.optimize import curve_fit
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; LON,LAT=np.meshgrid(glon,glat)
srcs=[rasterio.open(f'data/dem_{t}.tif') for t in ('N41_00_E109_00','N41_00_E110_00')]
mos,T=merge(srcs,bounds=(109.80,41.72,110.10,41.84),res=(0.0005,0.0005)); Z=mos[0][:len(glat),:len(glon)]
np.save('data/dem_emitgrid.npy',Z)
def box(a,b,c,d): return (LON>=a)&(LON<=c)&(LAT>=b)&(LAT<=d)
for d in ['20230203','20230423']:
    z7=np.load(f'data/z740_{d}.npy'); z5=np.load(f'data/z580_{d}.npy')
    ctrl=box(109.80,41.72,110.10,41.84)&~box(109.93,41.78,110.03,41.815)&np.isfinite(z7)   # outside the mine complex
    print(d,'corr with elevation outside mine: z740 r=%.2f  z580 r=%.2f'%(np.corrcoef(z7[ctrl],Z[ctrl])[0,1],np.corrcoef(z5[ctrl],Z[ctrl])[0,1]))
    # width of the 580 feature at Hot1 vs a Gaussian fit
    c=np.load(f'data/emit_aoi_{d}.npz'); C=c['cube'].astype('float32'); wl=c['wl']; hot=np.load(f'data/hot_{d}.npy')
    for nm,m in (('Hot1',box(109.993,41.801,110.004,41.808)&hot),('Main pit',box(109.955,41.792,109.972,41.803))):
        sel=(wl>545)&(wl<625); w=wl[sel]; s=np.nanmean(C[m][:,sel],0)
        # local linear continuum between 552 and 604
        i0,i1=np.argmin(abs(w-552)),np.argmin(abs(w-604)); cont=s[i0]+(s[i1]-s[i0])*(w-w[i0])/(w[i1]-w[i0]); cr=s/cont
        g=lambda x,a,mu,sig:1-a*np.exp(-(x-mu)**2/(2*sig**2))
        try:
            p,_=curve_fit(g,w,cr,p0=[0.03,578,10],bounds=([0,560,2],[0.3,600,60]))
            print(f'   {nm}: 580 feature depth {p[0]*100:.1f}%  centre {p[1]:.1f} nm  FWHM {2.355*p[2]:.1f} nm  (Elev median {np.nanmedian(Z[m]):.0f} m)')
        except Exception as ex: print('  fit failed',nm,ex)
