import numpy as np, json, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.patheffects as pe
from pyproj import Transformer
s=np.load('data/s2_nd.npz'); B=s['B']; rgb=s['rgb']; H,W=rgb.shape[1:]
fw=Transformer.from_crs(4326,32649,always_xy=True)
def rc(lon,lat): x,y=fw.transform(lon,lat); return (B[3]-y)/20,(x-B[0])/20
A=(109.85,41.755,110.08,41.815); (r0,c0),(r1,c1)=rc(A[0],A[3]),rc(A[2],A[1]); r0,r1,c0,c1=int(r0),int(r1),int(c0),int(c1)
img=np.clip(np.transpose(rgb,(1,2,0))[r0:r1,c0:c1]/0.30,0,1)**(1/1.4)
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; N=np.load('data/nd_twoband.npz')
fig,ax=plt.subplots(figsize=(14,6.6),dpi=130); ax.imshow(np.nan_to_num(img)*0.85)
# overlay two-band Nd cells as squares projected to S2 pixel coords
yy,xx=np.where(N['nd']); sc=N['score'][yy,xx]
lo=glon[xx]; la=glat[yy]; rr,cc=rc(lo,la)
m=ax.scatter(cc-c0,rr-r0,c=sc,s=5,marker='s',cmap='plasma',vmin=2,vmax=6,alpha=.9,linewidths=0)
cb=plt.colorbar(m,ax=ax,fraction=.025,pad=.01); cb.set_label('Two-band Nd score (mean z of 580 & 740 nm)',fontsize=8)
T=json.load(open('targets_classified.json'))
sty={'A':('o','#00e5ff','In-situ ore exposure'),'B':('s','#ffd60a','Processed / stockpiled REE'),'C':('x','#ff453a','Rejected (red roofs)')}
for t in T:
    k=t['class'][0]; mk,col,_=sty[k]; r,c=rc(t['peak'][1],t['peak'][0])
    ax.plot(c-c0,r-r0,mk,ms=7 if k!='C' else 6,mfc='none' if k!='C' else col,mec=col,mew=1.6)
    ax.text(c-c0+6,r-r0-6,t['id'],fontsize=6.5,color=col,path_effects=[pe.withStroke(linewidth=1.8,foreground='k')])
for k,(mk,col,lab) in sty.items(): ax.plot([],[],mk,mfc='none' if k!='C' else col,mec=col,mew=1.6,color=col,label=lab)
lbl={'Main pit':(109.965,41.8045),'East pit':(109.985,41.8075),'Z1 eastern excavation':(110.000,41.8115),'West pit':(109.878,41.8055),'Concentrator':(109.896,41.7885),'Tailings pond':(109.887,41.772),'Bayan Obo town':(109.970,41.762)}
for n,(lo,la) in lbl.items():
    r,c=rc(lo,la); ax.text(c-c0,r-r0,n,fontsize=8,color='white',ha='center',weight='bold',path_effects=[pe.withStroke(linewidth=2.2,foreground='k')])
for lo in np.arange(109.86,110.08,0.02):
    r,c=rc(lo,41.756); ax.text(c-c0,r1-r0-6,f'{lo:.2f}°E',fontsize=6.5,color='w',ha='center',path_effects=[pe.withStroke(linewidth=1.5,foreground='k')])
for la in np.arange(41.76,41.815,0.01):
    r,c=rc(109.851,la); ax.text(4,r-r0,f'{la:.2f}°N',fontsize=6.5,color='w',va='center',path_effects=[pe.withStroke(linewidth=1.5,foreground='k')])
ax.set_xlim(0,c1-c0); ax.set_ylim(r1-r0,0)
ax.legend(loc='lower right',fontsize=7.5,framealpha=.85); ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Neodymium absorption at Bayan Obo: EMIT 60 m cells with both the 580 nm and 740 nm bands on ≥2 of 3 dates (2023), over Sentinel-2',fontsize=9.5)
plt.tight_layout(); plt.savefig('fig/nd_map.png',bbox_inches='tight'); print('ok')
