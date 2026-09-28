import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; LON,LAT=np.meshgrid(glon,glat)
def box(a,b,c,d): return (LON>=a)&(LON<=c)&(LAT>=b)&(LAT<=d)
fig,axs=plt.subplots(1,2,figsize=(12,4.4),dpi=130)
for d,ax in zip(['20230203','20230423'],axs):
    z=np.load(f'data/emit_aoi_{d}.npz'); C=z['cube'].astype('float32'); wl=z['wl']
    hot=np.load(f'data/hot_{d}.npy')
    sets={'Hot1 hotspot pixels':box(109.993,41.801,110.004,41.808)&hot,'Main pit (all)':box(109.955,41.792,109.972,41.803),
          'Steppe control':box(109.90,41.815,109.95,41.835),'Granite control':box(110.03,41.73,110.09,41.76)}
    sel=(wl>540)&(wl<920)&~((wl>755)&(wl<772))
    cols=['#b3261e','#8a5a00','#6b8f71','#6b7b8c']
    for (n,m),col in zip(sets.items(),cols):
        s=np.nanmean(C[m][:, sel],0); w=wl[sel]
        # convex-hull-free continuum: straight line between 560 and 900
        i0,i1=np.argmin(abs(w-560)),np.argmin(abs(w-900)); cont=s[i0]+(s[i1]-s[i0])*(w-w[i0])/(w[i1]-w[i0])
        off=-0.06*list(sets).index(n)
        ax.plot(w,s/cont+off,color=col,lw=1.6,label=f'{n} (n={int(m.sum())})')
    for x in (580,625,680,745,800,865): ax.axvline(x,color='#b3261e',lw=.6,ls='--',alpha=.5); ax.text(x+1,1.075,str(x),fontsize=6.5,color='#b3261e')
    ax.axvspan(755,772,color='#ddd',alpha=.6); ax.text(757,0.78,'O₂-A\n(masked)',fontsize=6.5,color='#777')
    ax.set_ylim(0.76,1.09); ax.set_xlabel('Wavelength (nm)'); ax.set_title(f'EMIT {d[:4]}-{d[4:6]}-{d[6:]}: continuum-removed; dashed = Nd³⁺ bands',fontsize=9.5)
    ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
axs[0].set_ylabel('Relative reflectance (curves offset by 0.06)'); axs[1].legend(frameon=False,fontsize=7.5,loc='lower left')
plt.tight_layout(); plt.savefig('fig/emit_spectra.png')
print('ok')
