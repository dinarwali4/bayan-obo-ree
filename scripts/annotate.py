import numpy as np, json
from PIL import Image, ImageDraw, ImageFont
G={'BayanObo_01_pits_2025-06':(41.7960,109.9715,1000/243.5),'BayanObo_02_pits_2010-09':(41.7960,109.9715,1000/244),'BayanObo_03_pits_2004-05':(41.7960,109.9715,1000/244),
   'BayanObo_04_pits_2019-06':(41.7960,109.9715,1000/244),'BayanObo_05_belt_2025-06':(41.785,109.895,5000/344),'BayanObo_06_mainpit_2025-06':(41.7975,109.9630,500/321),
   'BayanObo_07_eastpit_2025-06':(41.7995,109.9810,500/331),'BayanObo_08_hot1_2025-06':(41.8045,109.9986,400/323)}
MLAT=110574.0; MLON=111320*np.cos(np.radians(41.8))
F=lambda s,b=True: ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if b else '/System/Library/Fonts/Supplemental/Arial.ttf',s)
def g2p(f,la,lo):
    c_la,c_lo,mpp=G[f]; w,h=2398,1522
    return w/2+(lo-c_lo)*MLON/mpp, h/2-(la-c_la)*MLAT/mpp
T=json.load(open('targets_classified.json')); Z=json.load(open('zones_insitu.json'))
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; N=np.load('data/nd_twoband.npz'); nd,score=N['nd'],N['score']
import matplotlib.cm as cm; cmap=cm.get_cmap('plasma')
def halo(d,xy,txt,font,fill=(255,255,255),anchor='la'):
    x,y=xy
    for dx in (-2,-1,0,1,2):
        for dy in (-2,-1,0,1,2): d.text((x+dx,y+dy),txt,font=font,fill=(0,0,0),anchor=anchor)
    d.text(xy,txt,font=font,fill=fill,anchor=anchor)
COL={'A':(0,229,255),'B':(255,214,10),'C':(255,69,58)}
def annotate(f,title,sub,labels,show_cells=True,show_T=('A','B','C'),coord_T=(),zone_lbl=True,out=None):
    im=Image.open(f'ge/img/{f}.jpg').convert('RGB'); w,h=im.size; mpp=G[f][2]
    ov=Image.new('RGBA',im.size,(0,0,0,0)); d=ImageDraw.Draw(ov)
    if show_cells:
        half=0.00025; yy,xx=np.where(nd)
        for y,x in zip(yy,xx):
            la,lo=glat[y],glon[x]; x0,y0=g2p(f,la+half,lo-half); x1,y1=g2p(f,la-half,lo+half)
            if x1<0 or y1<0 or x0>w or y0>h: continue
            v=np.clip((score[y,x]-2)/4,0,1); r,g,b,_=[int(255*c) for c in cmap(v)]
            d.rectangle([x0,y0,x1,y1],fill=(r,g,b,120),outline=(r,g,b,200))
    im=Image.alpha_composite(im.convert('RGBA'),ov).convert('RGB'); d=ImageDraw.Draw(im)
    # cover Earth's 'Untitled Map' and 'Legend' boxes with a title panel
    d.rectangle([0,0,w,118],fill=(14,18,22))
    halo(d,(28,18),title,F(40)); d.text((30,70),sub,font=F(24,False),fill=(200,206,212))
    for t in T:
        k=t['class'][0]
        if k not in show_T: continue
        x,y=g2p(f,*t['peak'])
        if not (0<x<w and 118<y<h): continue
        r=14 if mpp>3 else 20
        if k=='C': d.line([x-r,y-r,x+r,y+r],fill=COL[k],width=5); d.line([x-r,y+r,x+r,y-r],fill=COL[k],width=5)
        elif k=='B': d.rectangle([x-r,y-r,x+r,y+r],outline=COL[k],width=5)
        else: d.ellipse([x-r,y-r,x+r,y+r],outline=COL[k],width=5)
        lab=t['id']
        halo(d,(x+r+6,y-r-4),lab,F(24 if mpp>3 else 28),fill=COL[k])
    for txt,(la,lo),sz in labels:
        x,y=g2p(f,la,lo)
        if 0<x<w and 118<y<h: halo(d,(x,y),txt,F(sz),anchor='mm')
    if coord_T:
        rows=[t for t in T if t['id'] in coord_T]; bx=w-620; by=140
        d.rectangle([bx-14,by-12,w-20,by+22+len(rows)*38],fill=(14,18,22))
        d.text((bx,by-2),'Peak coordinates (WGS84, ±60 m)',font=F(22),fill=(255,255,255))
        for i,t in enumerate(rows):
            d.text((bx,by+34+i*38),f"{t['id']}  {t['peak'][0]:.5f}°N  {t['peak'][1]:.5f}°E",font=F(24,False),fill=COL[t['class'][0]])
    # legend
    lx,ly=24,h-230; d.rectangle([lx-10,ly-12,lx+560,h-24],fill=(14,18,22))
    items=[('A','In-situ ore exposure (Nd 580+740 nm)'),('B','Processed / stockpiled REE material'),('C','Rejected: red roofs (580 nm only)')]
    for i,(k,txt) in enumerate(items):
        yy=ly+10+i*42; x=lx+14
        if k=='C': d.line([x-11,yy-11,x+11,yy+11],fill=COL[k],width=4); d.line([x-11,yy+11,x+11,yy-11],fill=COL[k],width=4)
        elif k=='B': d.rectangle([x-11,yy-11,x+11,yy+11],outline=COL[k],width=4)
        else: d.ellipse([x-11,yy-11,x+11,yy+11],outline=COL[k],width=4)
        d.text((x+26,yy-13),txt,font=F(22,False),fill=(235,238,240))
    if show_cells:
        for i in range(200):
            r,g,b,_=[int(255*c) for c in cmap(i/199)]; d.line([lx+4+i*2,ly+140,lx+4+i*2,ly+160],fill=(r,g,b))
        d.text((lx+4,ly+166),'EMIT 60 m Nd cells: score 2 → 6+',font=F(20,False),fill=(200,206,212))
    im.save(out or f'fig/annot_{f[9:]}.jpg',quality=90); return out or f'fig/annot_{f[9:]}.jpg'
L_pits=[('MAIN PIT',(41.8035,109.9630),34),('EAST PIT',(41.8075,109.9840),34),('Z1 EASTERN EXCAVATION',(41.8115,110.0000),30),('Bayan Obo town',(41.7780,109.9700),26),('waste dumps',(41.7880,109.9420),24),('waste dumps',(41.7870,110.0120),24)]
annotate('BayanObo_01_pits_2025-06','Bayan Obo: where the rare-earth signal is strongest','Google Earth / Airbus, 3 Jun 2025 · EMIT hyperspectral Nd cells (Feb/Apr/Oct 2023) · ±60 m',L_pits,coord_T=('T09','T11','T17','T20','T10','T12','T16'))
annotate('BayanObo_08_hot1_2025-06','Zone Z1: eastern excavation, the strongest in-situ Nd zone','Google Earth / Airbus, Jun 2025 · 134 ha of two-band Nd cells · peak 41.80425°N 109.99725°E',[('Z1',(41.8085,110.0000),40)],coord_T=('T09','T11','T17','T20'))
annotate('BayanObo_06_mainpit_2025-06','Zone Z2: Main pit','Google Earth / Airbus, Jun 2025 · 45 ha of two-band Nd cells · peak 41.79925°N 109.96525°E',[('MAIN PIT',(41.8035,109.9630),36)],coord_T=('T10','T12'))
annotate('BayanObo_07_eastpit_2025-06','East pit: zones Z3 (north wall) and Z4 (west rim)','Google Earth / Airbus, Jun 2025 · EMIT two-band Nd cells',[('EAST PIT',(41.8055,109.9830),36)],coord_T=('T16','T14'))
annotate('BayanObo_05_belt_2025-06','The 20 km Bayan Obo belt: pits, concentrator, tailings','Google Earth, Jun 2025 · strongest Nd signals are processed material at the concentrator and tailings',
 [('Main/East pits',(41.812,109.975),34),('West pit',(41.813,109.878),30),('Concentrator',(41.7985,109.8960),26),('Tailings',(41.765,109.888),26),('Bayan Obo town',(41.758,109.970),26)],coord_T=('T02','T04'))
print('annotated')
