import numpy as np, json
from skimage import measure
from scipy import ndimage as ndi
from shapely.geometry import Polygon
e=np.load('data/emit_nd.npz'); glon,glat=e['glon'],e['glat']; N=np.load('data/nd_twoband.npz'); nd=N['nd']
T=json.load(open('targets_classified.json')); Z=json.load(open('zones_insitu.json'))
LON,LAT=np.meshgrid(glon,glat)
zdefs=[('Z1 Eastern excavation',(109.990,41.797,110.010,41.810)),('Z2 Main pit',(109.959,41.796,109.973,41.805)),('Z3 East pit, north wall',(109.984,41.801,109.989,41.807)),('Z4 East pit, west rim',(109.978,41.793,109.983,41.798))]
zinfo={0:Z[0],1:Z[1],2:Z[2],3:Z[3]}
def polys(mask):
    m=ndi.binary_closing(ndi.binary_dilation(mask,iterations=1),iterations=1); out=[]
    for c in measure.find_contours(np.pad(m.astype(float),1),0.5):
        pts=[(glon[0]+(cc-1)*0.0005,glat[0]-(r-1)*0.0005) for r,cc in c]
        p=Polygon(pts).simplify(0.0002)
        if p.is_valid and p.area>1e-7: out.append(p)
    return out
S={'A':'ffffe500','B':'ff0ad6ff','C':'ff3a45ff'}
k=['<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Bayan Obo REE (Nd) targets, RegenX 2026-09</name>',
   '<Style id="zone"><LineStyle><color>ffffe500</color><width>2.5</width></LineStyle><PolyStyle><color>40ffe500</color></PolyStyle></Style>']
for c,col in S.items(): k.append(f'<Style id="t{c}"><IconStyle><color>{col}</color><scale>1.0</scale><Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>')
k.append('<Folder><name>In-situ Nd zones (EMIT, both 580 and 740 nm on ≥2 of 3 dates in 2023)</name>')
for idx,(name,(a,b,c,d)) in enumerate(zdefs):
    zi=zinfo[idx]; m=nd&(LON>=a)&(LON<=c)&(LAT>=b)&(LAT<=d)
    for p in polys(m):
        co=' '.join(f'{x:.6f},{y:.6f},0' for x,y in p.exterior.coords)
        k.append(f'<Placemark><name>{name}</name><description>Area {zi["area_ha"]} ha · peak {zi["peak"][0]}N {zi["peak"][1]}E · mean 580 nm depth {zi["mean_F580_pct"]}% · elev {zi["elev_range"][0]}–{zi["elev_range"][1]} m</description><styleUrl>#zone</styleUrl><Polygon><tessellate>1</tessellate><outerBoundaryIs><LinearRing><coordinates>{co}</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark>')
k.append('</Folder>')
names={'A':'A — in-situ ore exposure','B':'B — processed / stockpiled REE material (variable signal)','C':'C — rejected (red roofs, 580 nm only)'}
for c in 'ABC':
    k.append(f'<Folder><name>{names[c]}</name>')
    for t in T:
        if t['class'][0]!=c: continue
        v=t.get('val2026',{})
        k.append(f'<Placemark><name>{t["id"]}</name><description>{t["surface"]}. Peak z580 {t["peak_z580"]} (2023, {t["dates"]} dates); z740&gt;2 on {t["z740_dates_gt2"]}/3 dates; 2026 check z580 {v.get("z580")} / z740 {v.get("z740")}. Coordinates ±60 m (EMIT 60 m pixel).</description><styleUrl>#t{c}</styleUrl><Point><coordinates>{t["peak"][1]},{t["peak"][0]},0</coordinates></Point></Placemark>')
    k.append('</Folder>')
k.append('</Document></kml>')
open('Bayan_Obo_REE_targets.kml','w').write('\n'.join(k)); print('ok')
