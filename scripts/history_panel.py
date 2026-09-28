import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure
from scipy import ndimage as ndi
exec(open('annotate.py').read().split("L_pits=")[0])
nd2=ndi.binary_closing(ndi.binary_dilation(nd,iterations=1),iterations=2)
mine=np.zeros_like(nd2); 
LON,LAT=np.meshgrid(glon,glat); mine=(LON>=109.950)&(LON<=110.012)&(LAT>=41.788)&(LAT<=41.812)
cont=[c for c in measure.find_contours((nd2&mine).astype(float),0.5) if len(c)>25]
frames=[('BayanObo_03_pits_2004-05','May 2004'),('BayanObo_02_pits_2010-09','Sep 2010'),('BayanObo_04_pits_2019-06','Jun 2019'),('BayanObo_01_pits_2025-06','Jun 2025')]
tiles=[]
for f,lab in frames:
    im=Image.open(f'ge/img/{f}.jpg').convert('RGB'); d=ImageDraw.Draw(im)
    for c in cont:
        pts=[g2p(f,glat[0]-r*0.0005,glon[0]+cc*0.0005) for r,cc in c]
        d.line(pts+[pts[0]],fill=(0,229,255),width=6)
    d.rectangle([0,0,im.width,118],fill=(14,18,22)); halo(d,(28,22),lab,F(64))
    for txt,(la,lo) in (('Main',(41.8030,109.9630)),('East',(41.8065,109.9840)),('Z1',(41.8105,110.0000))):
        x,y=g2p(f,la,lo); halo(d,(x,y),txt,F(44),anchor='mm')
    tiles.append(im.crop((250,118,2398-150,1522-60)).resize((999,625)))
W,H=tiles[0].size; out=Image.new('RGB',(W*2+12,H*2+12+150),(14,18,22)); dd=ImageDraw.Draw(out)
dd.text((20,20),'Bayan Obo Main/East complex, 2004 → 2025 (Google Earth historical imagery)',font=F(40),fill=(255,255,255))
dd.text((20,78),'Cyan outline = today\'s EMIT two-band Nd zones, drawn on every epoch to show when mining reached them. Same frame and scale in all four panels.',font=F(24,False),fill=(200,206,212))
for i,t in enumerate(tiles): out.paste(t,((i%2)*(W+12),150+(i//2)*(H+12)))
out.save('fig/history_panel.jpg',quality=90); print(out.size)
