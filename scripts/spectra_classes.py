import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; T=json.load(open('targets_classified.json'))
def cells(ids):
    out=[]
    for t in T:
        if t['id'] in ids:
            la,lo=t['peak']; i=int(round((glat[0]-la)/0.0005)); j=int(round((lo-glon[0])/0.0005)); out+=[(i+a,j+b) for a in (-1,0,1) for b in (-1,0,1)]
    return tuple(np.array(out).T)
groups={'A: in-situ pit benches (T09–T12, T16, T17, T20)':(cells(['T09','T10','T11','T12','T16','T17','T20']),'#0a84a8'),
        'B: concentrator & tailings compounds (T02, T04, T07)':(cells(['T02','T04','T07']),'#b8860b'),
        'C: rejected red roofs (T03, T05, T08, T19)':(cells(['T03','T05','T08','T19']),'#c0392b')}
LON,LAT=np.meshgrid(glon,glat); steppe=np.where((LON>=109.90)&(LON<=109.95)&(LAT>=41.815)&(LAT<=41.835))
groups['Steppe control']=(steppe,'#6b8f71')
fig,axs=plt.subplots(1,2,figsize=(13,4.8),dpi=130)
for ax,d in zip(axs,['20230423','20260410']):
    c=np.load(f'data/emit_aoi_{d}.npz'); C=c['cube']; wl=c['wl']
    sel=(wl>540)&(wl<880)&~((wl>755)&(wl<772)); w=wl[sel]
    for k,(n,(idx,col)) in enumerate(groups.items()):
        s=np.nanmean(C[idx][:,sel].astype('float32'),0)
        # continuum = straight line across each Nd feature region's shoulders (hull approx: 560 & 870)
        i0,i1=np.argmin(abs(w-560)),np.argmin(abs(w-870)); cont=s[i0]+(s[i1]-s[i0])*(w-w[i0])/(w[i1]-w[i0])
        ax.plot(w,s/cont-0.07*k,color=col,lw=1.7,label=n)
    for x in (580,745,800): ax.axvline(x,color='#555',lw=.7,ls='--'); ax.text(x+2,1.06,f'Nd {x}',fontsize=7,color='#444')
    ax.axvspan(755,772,color='#ddd'); ax.set_ylim(0.70,1.09); ax.set_xlabel('Wavelength (nm)')
    ax.set_title(f'EMIT {d[:4]}-{d[4:6]}-{d[6:]}' + (' (independent test date)' if d.startswith('2026') else ''),fontsize=10)
    ax.spines[['top','right']].set_visible(False); ax.grid(alpha=.2)
axs[0].set_ylabel('Continuum-removed reflectance (offset 0.07)'); axs[1].legend(frameon=False,fontsize=7.2,loc='lower left')
fig.suptitle('Pit benches (blue) show narrow Nd dips at 575–580 and 745 nm on both dates. Roofs (red) trip the 580 nm index through a steep red-pigment edge but have no 745 nm band. Plant sites (gold) vary between dates.',fontsize=10)
plt.tight_layout(); plt.savefig('fig/spectra_classes.png'); print('ok')
