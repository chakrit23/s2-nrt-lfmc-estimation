# s2-nrt-lfmc-estimation
s2-nrt-lfmc-estimation: Sentinel-2–Based Heuristic Framework for Near-Real-Time Live Fuel Moisture Content Mapping in Northern Thailand s2-lfmc-thailand provides an open, reproducible implementation of a Sentinel-2–based workflow for estimating relative vegetation moisture conditions (Live Fuel Moisture Content, LFMC) across northern Thailand. The workflow follows the methodology described in the manuscript:

Chotamonsak, C., Lapyai, D., Thanadolmethaphorn, P. (2025). Towards Near Real-Time Estimation of Live Fuel Moisture Content from Sentinel-2 for Fire Management in Northern Thailand. Submitted to Fire.

This repository includes the full Python code used for preprocessing, spectral-index computation, normalization, moisture proxy integration, and heuristic LFMC estimation. It is intended to support transparency, reproducibility, and future development of a validated LFMC monitoring system for fire management applications in mainland Southeast Asia.

# s2-lfmc-thailand

Near-real-time **Live Fuel Moisture Content (LFMC)** mapping for northern Thailand using **Sentinel-2**.

This repository provides a single pipeline script:

- `generate_s2_lfmc_pipeline.py`

which:

- reads Sentinel-2 data (from **Sentinel Hub** or **local GeoTIFFs**),
- computes LFMC (%) using NDVI, NDII, MSI and an NDVI-based ETf proxy,
- generates **daily** LFMC maps and danger-class maps,
- aggregates LFMC to **weekly / biweekly / monthly** composites,
- exports all products as **PNG** maps and optional **GeoTIFFs**.

> Designed for operational fire management and seasonal dryness monitoring over northern Thailand, but easily adaptable to other regions.

---

## 1. Main features (LFMC only)

- **Daily LFMC grids**
  - Per-pixel LFMC (%) for a user-defined AOI and period.
  - Outputs:
    - PNG maps (`lfmc_YYYY-MM-DD.png`)
    - optional GeoTIFFs (`lfmc_YYYY-MM-DD.tif`)

- **LFMC danger classes**
  - Pixel-wise danger classes based on LFMC thresholds.
  - Outputs:
    - PNG + PDF map + histogram (`lfmc_danger_YYYY-MM-DD.png/.pdf`)
    - optional GeoTIFFs (`lfmc_danger_YYYY-MM-DD.tif`)

- **Temporal composites**
  - Weekly (`--agg weekly`)
  - Biweekly (`--agg biweekly`)
  - Monthly (`--agg monthly`)
  - Statistic: `--agg_stat mean` or `--agg_stat median`
  - Composite LFMC maps + danger-class maps for each period.

- **Flexible data sources**
  - `--s2_source sh` – directly from Sentinel Hub.
  - `--s2_source local` – local B04/B08/B11/SCL GeoTIFF stacks.

- **Boundary overlays**
  - Optional shapefile/GeoJSON overlay clipped to AOI.
  - Automatic province labels when suitable fields exist.

- **Parallel per-day processing**
  - `--workers N` with safe `spawn` start method.
  - Uses on-disk `.npy` grids to keep memory usage reasonable.

---

## 2. Installation

### 2.1. Python environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# .\venv\Scripts\activate  # Windows
Install dependencies:

pip install numpy pandas xarray matplotlib tqdm requests
pip install rasterio geopandas shapely
pip install sentinelhub
2.2. Sentinel Hub credentials (for --s2_source sh)

export SH_CLIENT_ID="your_client_id"
export SH_CLIENT_SECRET="your_client_secret"
3. Inputs

3.1. Area of interest (AOI)

All coordinates are WGS84 (EPSG:4326):

--aoi lon_min lat_min lon_max lat_max
Example (northern Thailand):

--aoi 97 17 101.5 21
3.2. Time range

Inclusive ISO dates:

--time_start 2024-03-01 --time_end 2024-03-31
3.3. Sentinel-2 source options

Option A: Sentinel Hub

--s2_source sh \
--size 360 360 \
--maxcc 0.2 \
--s2_scale 10000
--size width height – output grid size in pixels.
--maxcc – max cloud fraction (0–1).
--s2_scale – reflectance scaling (10000 for L2A).
Option B: Local GeoTIFFs

--s2_source local \
--size 720 720 \
--s2_scale 10000 \
--b04_tpl "data/S2/B04_*.tif" \
--b08_tpl "data/S2/B08_*.tif" \
--b11_tpl "data/S2/B11_*.tif" \
--scl_tpl "data/S2/SCL_*.tif"
Each template should match daily files with dates in the filename, e.g.:

data/S2/B04_20240301.tif
data/S2/B08_20240301.tif
data/S2/B11_20240301.tif
data/S2/SCL_20240301.tif
The script:

parses YYYYMMDD or YYYY-MM-DD from filenames,
reprojects and resamples each band to AOI and --size in EPSG:4326.
4. LFMC model and danger classes

4.1. Indices and normalization (per pixel)

For valid (non-cloud) pixels:

NDVI = (B08 − B04) / (B08 + B04)
NDII = (B08 − B11) / (B08 + B11)
MSI = B11 / B08
Then normalized to [0,1] with heuristic ranges:

NDVI_n: NDVI in [0.2, 0.8]
NDII_n: NDII in [0.0, 0.6]
MSI_n: MSI in [0.6, 1.4], inverted (higher MSI → drier)
Derived terms:

Soil moisture index: SM = 0.5 * NDII_n + 0.5 * MSI_n
ET fraction proxy: ETf = clip(1.25 * NDVI_n, 0, 1)
4.2. LFMC (%) equation

LFMC = 35 + 80*NDVI_n + 50*SM + 20*(ETf - 0.5)
LFMC is clipped to [0, 200] (%)
4.3. Danger-class thresholds

Class index	LFMC range (%)	Label	Color
0	< 80	Extreme Dry (<80)	#800000
1	80–100	Dry (80–100)	#FF4500
2	100–120	Moderate Dry (100–120)	#FFD700
3	120–140	Moderate Moist (120–140)	#9ACD32
4	> 140	Moist (>140)	#006400
Danger-class GeoTIFFs are int16 with configurable nodata (default -1).

5. Outputs

Given --outdir outputs_lfmc, typical structure:

outputs_lfmc/
  lfmc_grids/
    lfmc_YYYY-MM-DD.png                 # daily LFMC maps
    lfmc_YYYY-MM-DD.tif                 # daily LFMC GeoTIFFs (optional)
    composites/
      lfmc_YYYY-MM-DD_to_YYYY-MM-DD.png # composite LFMC maps
      lfmc_YYYY-MM-DD_to_YYYY-MM-DD.tif # composite LFMC GeoTIFFs (optional)
  lfmc_danger/
    lfmc_danger_YYYY-MM-DD.png          # daily danger map + histogram
    lfmc_danger_YYYY-MM-DD.pdf
    composites/
      lfmc_danger_YYYY-MM-DD_to_YYYY-MM-DD.png
      lfmc_danger_YYYY-MM-DD_to_YYYY-MM-DD.pdf
    tif/
      lfmc_danger_YYYY-MM-DD.tif        # daily danger-class GeoTIFFs (optional)
      composites/
        lfmc_danger_YYYY-MM-DD_to_YYYY-MM-DD.tif
  _tmp_np_grids/                        # temporary .npy LFMC grids (auto-cleaned)
6. Usage examples

6.1. Daily LFMC + weekly composites (local GeoTIFFs)

python generate_s2_lfmc_pipeline.py \
  --aoi 97 17 101.5 21 \
  --time_start 2024-03-01 \
  --time_end 2024-03-31 \
  --s2_source local \
  --size 720 720 \
  --s2_scale 10000 \
  --b04_tpl "data/S2/B04_*.tif" \
  --b08_tpl "data/S2/B08_*.tif" \
  --b11_tpl "data/S2/B11_*.tif" \
  --scl_tpl "data/S2/SCL_*.tif" \
  --outdir outputs_weekly \
  --workers 4 \
  --daily_outputs tif \
  --agg weekly \
  --agg_stat median
Produces:

daily LFMC GeoTIFFs,
daily LFMC danger maps,
weekly median LFMC composites and danger maps.
6.2. Monthly composites (Sentinel Hub)

python generate_s2_lfmc_pipeline.py \
  --aoi 97 17 101.5 21 \
  --time_start 2024-01-01 \
  --time_end 2024-04-30 \
  --s2_source sh \
  --size 360 360 \
  --maxcc 0.2 \
  --s2_scale 10000 \
  --outdir outputs_monthly \
  --workers 4 \
  --daily_outputs both \
  --agg monthly \
  --agg_stat mean
6.3. Generating all three aggregation scales

Run three times with different output directories:

# Weekly
python generate_s2_lfmc_pipeline.py ... \
  --agg weekly --agg_stat median --outdir outputs_weekly

# Biweekly
python generate_s2_lfmc_pipeline.py ... \
  --agg biweekly --agg_stat median --outdir outputs_biweekly

# Monthly
python generate_s2_lfmc_pipeline.py ... \
  --agg monthly --agg_stat mean --outdir outputs_monthly
(... = shared arguments such as --aoi, --time_start, --time_end, --s2_source, etc.)

6.4. Output control

--daily_outputs both|png|tif|none
--no_daily_danger – skip daily danger maps.
--no_overlay_daily – skip boundaries overlay on daily maps (composites still overlay).
--write_danger_tif – write danger-class GeoTIFFs for daily + composites.
--danger_tif_nodata N – nodata value for danger-class GeoTIFFs (default -1).
--keep_tmp – keep _tmp_np_grids/ for debugging.
6.5. Boundaries overlay

--boundaries data/shapes/th_province_4326.shp
Must be readable by geopandas.
Reprojected to EPSG:4326 and clipped to AOI.
Province labels taken from one of: prov_name, NAME_1, NAME_TH, name, NAME.
7. License

8. Contact

For questions or collaboration, please open an issue in this repository.
