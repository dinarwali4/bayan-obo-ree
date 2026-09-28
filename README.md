# Bayan Obo neodymium map

Hyperspectral mapping of surface rare-earth (neodymium) exposure at the Bayan Obo REE–Fe–Nb mine, Baotou, Inner Mongolia, China (41.76–41.81°N, 109.85–110.07°E).

**Live report:** https://dinarwali4.github.io/bayan-obo-ree/

## Ranked in-situ zones
The EMIT satellite instrument shows both Nd³⁺ absorption bands (580 and 745 nm) on at least 2 of 3 dates in 2023. Coordinates are WGS84, ±60 m.

| Rank | Zone | Area | Peak |
|---|---|---|---|
| 1 | Z1 eastern excavation | 134 ha | 41.80425 N, 109.99725 E |
| 2 | Z2 Main pit | 45 ha | 41.79925 N, 109.96525 E |
| 3 | Z3 East pit, north wall | 12 ha | 41.80225 N, 109.98575 E |
| 4 | Z4 East pit, west rim | 9 ha | 41.79625 N, 109.97975 E |

**Validation.** The two-band rule flags zero cells on town, steppe or granite control ground. All 7 in-situ targets reconfirm on an independent EMIT pass on 10 Apr 2026.

**What was excluded.** The concentrator and tailings compounds carry strong but variable signals; they are processed material, not ore in place. Nine red-roof false positives were removed.

**Limits.** This maps relative surface abundance. It is not a grade or tonnage estimate. For grades, see USGS Bulletin 2143.

## Contents
| Path | What |
|---|---|
| `docs/` | Published report (GitHub Pages), annotated Google Earth imagery, and the KML |
| `Bayan_Obo_REE_targets.kml` | Zones Z1–Z4 and targets by class, for Google Earth |
| `data/` | Targets, zones, 2026 validation, and Landsat footprint series (JSON) |
| `figures/` | Full-resolution charts and maps |
| `scripts/` | Python pipeline |

## Pipeline
1. **Fetch:** `s2_fetch.py`, `landsat_fetch.py`, `dl.sh`. EMIT needs a NASA Earthdata token in `~/.edl_token`.
2. **Extract and compute features:** `emit_nd.py`, then `emit_feat.py`.
3. **Score and classify:** `persist.py`, `targets.py`, `targets2.py`, `classify.py`, `zones.py`, `validate2026.py`.
4. **Figures:** `fig_map.py`, `spectra_classes.py`, `annotate.py`, `history_panel.py`, `landsat_growth.py`.
5. **KML:** `make_kml.py`.

## Sources
- Chao et al. (1997), [USGS Bulletin 2143](https://pubs.usgs.gov/bul/b2143/). Public domain; figure 8 is reproduced.
- [Scientific Reports (2024)](https://www.nature.com/articles/s41598-024-71395-2): REE detection with EnMAP at Mountain Pass.
- Data: NASA EMIT L2A (LP DAAC), Sentinel-2 L2A, Landsat Collection 2, Copernicus GLO-30. Google Earth Pro imagery © Airbus / CNES / Maxar Technologies, used for annotation.

Prepared by RegenX.eco, September 2026.
