#!/usr/bin/env python3
"""
s2-lfmc-thailand: LFMC mapping pipeline

Generates:
- Daily LFMC grids (PNG/TIFF, optional) + danger maps
- Weekly/biweekly/monthly LFMC composites + danger composites
- Province labels on boundary overlay (optional, clipped to AOI)
- Local Sentinel-2 mode (GeoTIFF templates) or Sentinel Hub mode
- Parallel per-day processing with spawn & maxtasksperchild=1
- On-disk .npy grids (parent-only plotting) + visible post-processing progress

Key flags:
  --workers N                  parallel day processing (default 1)
  --daily_outputs {both,png,tif,none}  control daily PNG/TIFF generation
  --no_daily_danger           skip daily danger-class PNG/PDF
  --no_overlay_daily          skip boundaries overlay on daily maps (faster)
  --write_danger_tif          also write GeoTIFF for danger classes (int16)
  --danger_tif_nodata N       nodata value for danger-class GeoTIFF (default -1)
  --keep_tmp                  keep temp .npy grids directory

Example:

python generate_s2_lfmc_pipeline.py \
  --aoi 97 17 101.5 21 \
  --time_start 2024-03-01 --time_end 2024-03-31 \
  --s2_source local --size 720 720 --s2_scale 10000 \
  --b04_tpl "data/S2/B04_*.tif" \
  --b08_tpl "data/S2/B08_*.tif" \
  --b11_tpl "data/S2/B11_*.tif" \
  --scl_tpl "data/S2/SCL_*.tif" \
  --outdir outputs_geotiff_local \
  --workers 4 \
  --daily_outputs tif \
  --agg weekly --agg_stat median
"""

# ---- Cap native thread pools EARLY (before numpy/rasterio loads) ----
import os as _os
_os.environ.setdefault("GDAL_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import re
import glob
import argparse
import uuid
import shutil
import multiprocessing as mp
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")  # headless-safe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects
from matplotlib.ticker import MaxNLocator, FuncFormatter
from tqdm import tqdm

# Sentinel Hub (only used in --s2_source sh)
from sentinelhub import SHConfig, SentinelHubRequest, DataCollection, MimeType, BBox, CRS

# GeoTIFF I/O (required for --s2_source local; optional for GeoTIFF export)
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
except Exception:
    rasterio = None

# Optional geopandas
try:
    import geopandas as gpd
    from shapely.geometry import box
except Exception:
    gpd = None
    box = None

# ---------- formatters ----------
def lon_formatter(x, _):
    return f"{int(round(x))}°E" if x >= 0 else f"{int(round(-x))}°W"

def lat_formatter(y, _):
    return f"{int(round(y))}°N" if y >= 0 else f"{int(round(-y))}°S"


# ---------------- Sentinel Hub evalscript ----------------
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands:["B04","B08","B11","SCL","dataMask"] }],
    output: [{ id:"bands", bands:4, sampleType:"FLOAT32" }],
    mosaicking: "SIMPLE"
  };
}
function goodSCL(scl){
  return !(scl===3||scl===6||scl===8||scl===9||scl===10||scl===11);
}
function evaluatePixel(s){
  if (s.dataMask===0 || !goodSCL(s.SCL)) return [NaN,NaN,NaN,0];
  return [s.B04,s.B08,s.B11,1];
}
"""


# ---------------- Utilities ----------------
def daterange(start_iso, end_iso):
    s = datetime.fromisoformat(start_iso)
    e = datetime.fromisoformat(end_iso)
    d = s
    while d <= e:
        yield d
        d += timedelta(days=1)


def require_sh():
    cfg = SHConfig()
    cfg.sh_client_id = _os.getenv("SH_CLIENT_ID", "")
    cfg.sh_client_secret = _os.getenv("SH_CLIENT_SECRET", "")
    if not cfg.sh_client_id or not cfg.sh_client_secret:
        raise RuntimeError("Set SH_CLIENT_ID / SH_CLIENT_SECRET")
    return cfg


def _nan_stats(a):
    a = np.asarray(a, float)
    if not np.isfinite(a).any():
        return (np.nan, np.nan, np.nan)
    return (float(np.nanmin(a)), float(np.nanmax(a)), float(np.nanmean(a)))


# ---------------- LFMC computation (shared) ----------------
def normalize_01(x, lo, hi, invert=False):
    y = (x - lo) / (hi - lo + 1e-9)
    y = np.clip(y, 0, 1)
    return 1 - y if invert else y


def _print_s2_stats(day_iso, s2_scale, B04, B08, B11, NDVI, NDII, MSI, LFMC):
    b04s = _nan_stats(B04)
    b08s = _nan_stats(B08)
    b11s = _nan_stats(B11)
    ndvis = _nan_stats(NDVI)
    ndiis = _nan_stats(NDII)
    msis = _nan_stats(MSI)
    lfmcs = _nan_stats(LFMC)
    print(
        f"[S2 {day_iso}] scale={s2_scale:g} "
        f"B04[{b04s[0]:.3f},{b04s[1]:.3f}] "
        f"B08[{b08s[0]:.3f},{b08s[1]:.3f}] "
        f"B11[{b11s[0]:.3f},{b11s[1]:.3f}] | "
        f"NDVI[{ndvis[0]:.2f},{ndvis[1]:.2f}] "
        f"NDII[{ndiis[0]:.2f},{ndiis[1]:.2f}] "
        f"MSI[{msis[0]:.2f},{msis[1]:.2f}] | "
        f"LFMC[{lfmcs[0]:.1f},{lfmcs[1]:.1f}]"
    )


# ---------------- Sentinel-2 from SH ----------------
def s2_lfmc_grid_and_mean_sh(cfg, d, aoi, s2_scale, size, maxcc):
    bbox = BBox(bbox=aoi, crs=CRS.WGS84)
    req = SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(d.strftime("%Y-%m-%d"), d.strftime("%Y-%m-%d")),
            maxcc=maxcc,
            mosaicking_order="leastCC"
        )],
        responses=[SentinelHubRequest.output_response("bands", MimeType.TIFF)],
        bbox=bbox, size=size, config=cfg
    )
    data = req.get_data()
    h, w = size[1], size[0]
    if not data or data[0] is None:
        lfmc_nan = np.full((h, w), np.nan, dtype=np.float32)
        print(f"[S2 {d.strftime('%Y-%m-%d')}] no data → NaN grid")
        return np.nan, lfmc_nan

    arr = np.asarray(data[0])
    if arr.ndim == 3 and arr.shape[-1] == 4:
        pass
    elif arr.ndim == 3 and arr.shape[0] == 4:
        arr = np.transpose(arr, (1, 2, 0))
    else:
        lfmc_nan = np.full((h, w), np.nan, dtype=np.float32)
        print(f"[S2 {d.strftime('%Y-%m-%d')}] bad shape {arr.shape} → NaN grid")
        return np.nan, lfmc_nan

    H, W, _ = arr.shape
    if (H, W) != (h, w):
        yi = (np.arange(h) * (H / max(h, 1))).astype(int).clip(0, max(H - 1, 0))
        xi = (np.arange(w) * (W / max(w, 1))).astype(int).clip(0, max(W - 1, 0))
        arr = arr[yi[:, None], xi[None, :], :]

    B04 = arr[:, :, 0].astype("float32") / float(s2_scale)
    B08 = arr[:, :, 1].astype("float32") / float(s2_scale)
    B11 = arr[:, :, 2].astype("float32") / float(s2_scale)
    V = arr[:, :, 3] > 0

    NDVI = np.where(V, (B08 - B04) / (B08 + B04 + 1e-6), np.nan)
    NDII = np.where(V, (B08 - B11) / (B08 + B11 + 1e-6), np.nan)
    MSI = np.where(V, B11 / (B08 + 1e-6), np.nan)

    NDVI_n = normalize_01(NDVI, 0.2, 0.8)
    NDII_n = normalize_01(NDII, 0.0, 0.6)
    MSI_n = normalize_01(MSI, 0.6, 1.4, invert=True)
    SM = 0.5 * NDII_n + 0.5 * MSI_n
    ETf = np.clip(1.25 * NDVI_n, 0, 1)
    LFMC = np.clip(35 + 80 * NDVI_n + 50 * SM + 20 * (ETf - 0.5), 0, 200)

    _print_s2_stats(d.strftime("%Y-%m-%d"), s2_scale, B04, B08, B11, NDVI, NDII, MSI, LFMC)
    mean_val = float(np.nanmean(LFMC[V])) if np.any(V) else np.nan
    return mean_val, LFMC.astype("float32")


# ---------------- Sentinel-2 from LOCAL GeoTIFFs ----------------
_DATE_PAT = re.compile(r'(\d{4}-\d{2}-\d{2}|\d{8})')

def _parse_date_from_name(path):
    m = _DATE_PAT.search(_os.path.basename(path))
    if not m:
        return None
    s = m.group(1)
    if len(s) == 8:   # YYYYMMDD
        return datetime.strptime(s, "%Y%m%d").date()
    else:             # YYYY-MM-DD
        return datetime.strptime(s, "%Y-%m-%d").date()


def _pick_daily_file(tpl, day):
    files = glob.glob(tpl)
    for p in files:
        d = _parse_date_from_name(p)
        if d == day.date():
            return p
    return None


def _build_target_grid(aoi, size):
    xmin, ymin, xmax, ymax = aoi
    width, height = size
    transform = from_bounds(xmin, ymin, xmax, ymax, width, height)
    return transform, (height, width)


def _read_reproject(src_path, dst_shape, dst_transform, resampling):
    with rasterio.open(src_path) as src:
        src_arr = src.read(1)
        src_tr = src.transform
        src_crs = src.crs
    dst = np.zeros(dst_shape, dtype=src_arr.dtype)
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_tr, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs="EPSG:4326",
        resampling=resampling, num_threads=1  # single-threaded to avoid deadlocks
    )
    return dst


def _scl_valid_mask(SCL):
    # Valid if not in {3,6,8,9,10,11} and not 0 (nodata)
    bad = (SCL == 3) | (SCL == 6) | (SCL == 8) | (SCL == 9) | (SCL == 10) | (SCL == 11) | (SCL == 0)
    return ~bad


def s2_lfmc_grid_and_mean_local(day_dt, aoi, size, scale, b04_tpl, b08_tpl, b11_tpl, scl_tpl):
    if rasterio is None:
        raise RuntimeError("rasterio is required for --s2_source local")

    transform, dst_shape = _build_target_grid(aoi, size)

    f_b04 = _pick_daily_file(b04_tpl, day_dt)
    f_b08 = _pick_daily_file(b08_tpl, day_dt)
    f_b11 = _pick_daily_file(b11_tpl, day_dt)
    f_scl = _pick_daily_file(scl_tpl, day_dt)

    if not all([f_b04, f_b08, f_b11, f_scl]):
        print(
            f"[Local {day_dt.strftime('%Y-%m-%d')}] missing file(s):",
            f"B04={bool(f_b04)} B08={bool(f_b08)} B11={bool(f_b11)} SCL={bool(f_scl)} → NaN grid"
        )
        return np.nan, np.full(dst_shape, np.nan, dtype=np.float32)

    B04 = _read_reproject(f_b04, dst_shape, transform, Resampling.bilinear).astype("float32") / float(scale)
    B08 = _read_reproject(f_b08, dst_shape, transform, Resampling.bilinear).astype("float32") / float(scale)
    B11 = _read_reproject(f_b11, dst_shape, transform, Resampling.bilinear).astype("float32") / float(scale)
    SCL = _read_reproject(f_scl, dst_shape, transform, Resampling.nearest)

    valid = _scl_valid_mask(SCL)
    NDVI = np.where(valid, (B08 - B04) / (B08 + B04 + 1e-6), np.nan)
    NDII = np.where(valid, (B08 - B11) / (B08 + B11 + 1e-6), np.nan)
    MSI = np.where(valid, B11 / (B08 + 1e-6), np.nan)

    NDVI_n = normalize_01(NDVI, 0.2, 0.8)
    NDII_n = normalize_01(NDII, 0.0, 0.6)
    MSI_n = normalize_01(MSI, 0.6, 1.4, invert=True)
    SM = 0.5 * NDII_n + 0.5 * MSI_n
    ETf = np.clip(1.25 * NDVI_n, 0, 1)
    LFMC = np.clip(35 + 80 * NDVI_n + 50 * SM + 20 * (ETf - 0.5), 0, 200)

    _print_s2_stats(day_dt.strftime("%Y-%m-%d"), scale, B04, B08, B11, NDVI, NDII, MSI, LFMC)
    mean_val = float(np.nanmean(LFMC[valid])) if np.any(valid) else np.nan
    return mean_val, LFMC.astype("float32")


# ---------------- Danger classes and plotting ----------------
DANGER_CLASSES = [
    (None, 80, "Extreme (<80)", "#800000"),
    (80, 100, "High (80-100)", "#FF4500"),
    (100, 120, "Moderate (100-120)", "#FFD700"),
    (120, 140, "Low (120-140)", "#9ACD32"),
    (140, None, "Very Low (>140)", "#006400"),
]


def classify_lfmc(grid):
    cls = np.full(grid.shape, -1, dtype=np.int16)
    for i, (lo, hi, _, _) in enumerate(DANGER_CLASSES):
        m = np.ones_like(grid, bool)
        if lo is not None:
            m &= grid >= lo
        if hi is not None:
            m &= grid < hi
        cls[m] = i
    return cls


def _draw_province_labels(ax, gdf):
    name_cols = ["prov_name", "NAME_1", "NAME_TH", "name", "NAME"]
    use_col = None
    if gdf is not None:
        for c in name_cols:
            if c in gdf.columns:
                use_col = c
                break
    if gdf is not None and use_col:
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            c = geom.centroid
            ax.text(
                c.x, c.y, str(row.get(use_col, "")),
                fontsize=6, ha="center", va="center", color="black",
                path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5, foreground="white")]
            )


def _apply_lonlat_ticks(ax, extent):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FuncFormatter(lon_formatter))
    ax.yaxis.set_major_formatter(FuncFormatter(lat_formatter))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")


def save_lfmc_png(lfmc, aoi, out_png, gdf=None, title_suffix=None, overlay=True):
    extent = [aoi[0], aoi[2], aoi[1], aoi[3]]
    fig = plt.figure(figsize=(5, 5), constrained_layout=True)
    ax = plt.gca()
    im = ax.imshow(
        lfmc, cmap="viridis", extent=extent,
        origin="upper", interpolation="nearest"
    )
    if overlay and gdf is not None:
        gdf.boundary.plot(ax=ax, color="black", lw=0.5)
        _draw_province_labels(ax, gdf)
    _apply_lonlat_ticks(ax, [extent[0], extent[1], extent[2], extent[3]])
    ttl = "LFMC (%)" + (f" — {title_suffix}" if title_suffix else "")
    ax.set_title(ttl)
    plt.colorbar(im, ax=ax, label="LFMC (%)")
    _os.makedirs(_os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)


def save_lfmc_tif(lfmc, aoi, out_tif):
    if rasterio is None:
        return
    _os.makedirs(_os.path.dirname(out_tif), exist_ok=True)
    h, w = lfmc.shape
    transform = from_bounds(aoi[0], aoi[1], aoi[2], aoi[3], w, h)
    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=np.nan,
        compress="deflate",
        tiled=True,
        blockxsize=min(256, w),
        blockysize=min(256, h),
    )
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(lfmc.astype("float32"), 1)


def save_danger_map_and_hist(cls, out_base, aoi=None, gdf=None, title_suffix=None, overlay=True):
    _os.makedirs(_os.path.dirname(out_base), exist_ok=True)
    colors = [c for _, _, _, c in DANGER_CLASSES]
    labels = [s for _, _, s, _ in DANGER_CLASSES]
    extent = [aoi[0], aoi[2], aoi[1], aoi[3]] if aoi else None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    ax1.imshow(
        cls,
        cmap=plt.matplotlib.colors.ListedColormap(colors),
        vmin=0, vmax=len(colors) - 1,
        extent=extent, origin="upper", interpolation="nearest"
    )
    if overlay and gdf is not None:
        gdf.boundary.plot(ax=ax1, color="black", lw=0.5)
        _draw_province_labels(ax1, gdf)
    if aoi is not None:
        _apply_lonlat_ticks(ax1, [extent[0], extent[1], extent[2], extent[3]])
    ttl = "LFMC Danger Classes" + (f" — {title_suffix}" if title_suffix else "")
    ax1.set_title(ttl)
    counts = [(cls == i).sum() for i in range(len(colors))]
    ax2.bar(range(len(colors)), counts, color=colors)
    ax2.set_xticks(range(len(colors)))
    ax2.set_xticklabels(labels, rotation=30, ha="right")
    ax2.set_ylabel("Pixel count")
    plt.savefig(out_base + ".png", dpi=300)
    plt.savefig(out_base + ".pdf")
    plt.close(fig)


def save_danger_tif(cls, aoi, out_tif, nodata_val=-1):
    """Write danger classes as int16 GeoTIFF with nodata."""
    if rasterio is None:
        return
    _os.makedirs(_os.path.dirname(out_tif), exist_ok=True)
    h, w = cls.shape
    transform = from_bounds(aoi[0], aoi[1], aoi[2], aoi[3], w, h)
    arr = cls.astype(np.int16, copy=False)
    profile = dict(
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype="int16",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata_val,
        compress="deflate",
        tiled=True,
        blockxsize=min(256, w),
        blockysize=min(256, h),
    )
    with rasterio.open(out_tif, "w", **profile) as dst:
        dst.write(arr, 1)


# ---------------- Aggregation helpers ----------------
def period_freq(kind):
    if kind == "weekly":
        return "W-MON"
    if kind == "biweekly":
        return "2W-MON"
    if kind == "monthly":
        return "ME"  # month-end
    return ""


# ---------------- Per-day worker + star-unpacker ----------------
def _process_one_day(
    day_iso, aoi, s2_source, size, s2_scale, maxcc,
    b04_tpl, b08_tpl, b11_tpl, scl_tpl, tmp_dir
):
    d = datetime.fromisoformat(day_iso)

    # Sentinel-2 → LFMC grid
    try:
        if s2_source == "sh":
            cfg = require_sh()
            lfmc_mean, grid = s2_lfmc_grid_and_mean_sh(
                cfg, d, aoi, s2_scale, tuple(size), maxcc
            )
        else:
            lfmc_mean, grid = s2_lfmc_grid_and_mean_local(
                d, aoi, tuple(size), s2_scale,
                b04_tpl, b08_tpl, b11_tpl, scl_tpl
            )
    except Exception as e:
        print(f"[S2 {day_iso}] failed: {e}")
        h, w = size[1], size[0]
        grid = np.full((h, w), np.nan, dtype=np.float32)
        lfmc_mean = np.nan

    # Save grid to tmp .npy
    _os.makedirs(tmp_dir, exist_ok=True)
    npy_path = _os.path.join(tmp_dir, f"lfmc_{day_iso}_{uuid.uuid4().hex}.npy")
    np.save(npy_path, grid.astype("float32"))

    return day_iso, lfmc_mean, npy_path


def _star(args):
    return _process_one_day(*args)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Sentinel-2 LFMC mapping pipeline")
    # Core period & AOI
    ap.add_argument(
        "--aoi", nargs=4, type=float, required=True,
        help="lon_min lat_min lon_max lat_max"
    )
    ap.add_argument("--time_start", type=str, required=True)
    ap.add_argument("--time_end", type=str, required=True)

    # S2 source & options
    ap.add_argument(
        "--s2_source", choices=["sh", "local"], default="sh",
        help="Use 'sh' (Sentinel Hub) or 'local' GeoTIFF bands"
    )
    ap.add_argument(
        "--size", nargs=2, type=int, default=[360, 360],
        help="output raster size (width height)"
    )
    ap.add_argument(
        "--maxcc", type=float, default=0.2,
        help="[SH only] max cloud cover fraction (0..1)"
    )
    ap.add_argument(
        "--s2_scale", type=float, default=10000.0,
        help="Reflectance scale factor (10000 or 1)"
    )

    # Local band templates (daily files, e.g., B04_YYYYMMDD.tif)
    ap.add_argument("--b04_tpl", type=str, default="", help='e.g., "data/S2/B04_*.tif"')
    ap.add_argument("--b08_tpl", type=str, default="", help='e.g., "data/S2/B08_*.tif"')
    ap.add_argument("--b11_tpl", type=str, default="", help='e.g., "data/S2/B11_*.tif"')
    ap.add_argument("--scl_tpl", type=str, default="", help='e.g., "data/S2/SCL_*.tif"')

    # Outputs & extras
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument(
        "--boundaries", type=str, default=None,
        help="Optional shapefile (.shp/.geojson) for overlay; it will be clipped to AOI"
    )
    ap.add_argument(
        "--keep_tmp", action="store_true",
        help="Keep _tmp_np_grids directory"
    )

    # Daily output control
    ap.add_argument(
        "--daily_outputs", choices=["both", "png", "tif", "none"], default="both",
        help="What daily outputs to write"
    )
    ap.add_argument(
        "--no_daily_danger", action="store_true",
        help="Skip daily danger-class PNG/PDF"
    )
    ap.add_argument(
        "--no_overlay_daily", action="store_true",
        help="Skip boundaries overlay on DAILY maps for speed"
    )
    ap.add_argument(
        "--write_danger_tif", action="store_true",
        help="Also write GeoTIFFs of danger classes (int16) for daily and composites"
    )
    ap.add_argument(
        "--danger_tif_nodata", type=int, default=-1,
        help="Nodata value for danger-class GeoTIFF (default -1)"
    )

    # Aggregation
    ap.add_argument(
        "--agg", choices=["weekly", "biweekly", "monthly"], default=None,
        help="Temporal aggregation for LFMC composites"
    )
    ap.add_argument(
        "--agg_stat", choices=["mean", "median"], default="mean",
        help="Statistic for LFMC composites when aggregating"
    )

    # Parallelism
    ap.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker processes for per-day processing"
    )

    args = ap.parse_args()

    _os.makedirs(args.outdir, exist_ok=True)

    aoi = args.aoi
    dates_dt = list(daterange(args.time_start, args.time_end))
    dates = [d.strftime("%Y-%m-%d") for d in dates_dt]

    # Use 'spawn' for rasterio/GDAL safety
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if args.s2_source == "local" and rasterio is None:
        raise RuntimeError("rasterio is required for --s2_source local")
    if args.s2_source == "local":
        if not (args.b04_tpl and args.b08_tpl and args.b11_tpl and args.scl_tpl):
            raise SystemExit(
                "When --s2_source local, provide --b04_tpl, --b08_tpl, --b11_tpl, --scl_tpl"
            )

    # Boundaries
    gdf = None
    if args.boundaries and gpd is not None and box is not None:
        print(f"✔ Loading boundaries: {args.boundaries}")
        gdf = gpd.read_file(args.boundaries).to_crs(4326)
        aoi_poly = box(*aoi)
        gdf = gdf.clip(aoi_poly)
        print(f"✔ Boundaries clipped to AOI: {len(gdf)} features")

    print(f"▶ Date range: {args.time_start} → {args.time_end}  ({len(dates)} days) | AOI={aoi}")
    if args.s2_source == "sh":
        print(f"▶ S2 source: Sentinel Hub | scale={args.s2_scale:g} | maxcc={args.maxcc}")
    else:
        print(f"▶ S2 source: LOCAL GeoTIFF | scale={args.s2_scale:g}")
        print(
            f"   TPL: B04='{args.b04_tpl}'  B08='{args.b08_tpl}'  "
            f"B11='{args.b11_tpl}'  SCL='{args.scl_tpl}'"
        )
    print(f"▶ Workers (spawn + maxtasksperchild=1): {args.workers}")

    # ----------------------------- Daily processing (parallel/serial) -----------------------------
    tmp_np_dir = _os.path.join(args.outdir, "_tmp_np_grids")
    _os.makedirs(tmp_np_dir, exist_ok=True)
    results = {}  # day_iso -> (lfmc_mean, npy_path)

    tasks = [
        (
            day_iso, aoi, args.s2_source, args.size, args.s2_scale, args.maxcc,
            args.b04_tpl, args.b08_tpl, args.b11_tpl, args.scl_tpl,
            tmp_np_dir
        )
        for day_iso in dates
    ]

    if args.workers and args.workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers, maxtasksperchild=1) as pool:
            for day_iso, lfmc_mean, npy_path in tqdm(
                pool.imap_unordered(_star, tasks, chunksize=1),
                total=len(tasks), desc="Processing days"
            ):
                results[day_iso] = (lfmc_mean, npy_path)
    else:
        for t in tqdm(tasks, desc="Processing days"):
            day_iso, lfmc_mean, npy_path = _star(t)
            results[day_iso] = (lfmc_mean, npy_path)

    # Collect npy paths
    npy_paths = {}
    for day_iso in sorted(results.keys()):
        _, npy_path = results[day_iso]
        npy_paths[day_iso] = npy_path

    # --- Daily products (parent, visible progress) ---
    if args.daily_outputs != "none" or args.write_danger_tif:
        print("▶ Writing daily products…")
        for day_iso in tqdm(sorted(npy_paths.keys()), desc="Daily outputs"):
            grid = np.load(npy_paths[day_iso])

            # PNG
            if args.daily_outputs in ("both", "png"):
                out_png = _os.path.join(args.outdir, "lfmc_grids", f"lfmc_{day_iso}.png")
                _os.makedirs(_os.path.dirname(out_png), exist_ok=True)
                save_lfmc_png(
                    grid, aoi, out_png,
                    gdf=(None if args.no_overlay_daily else gdf),
                    title_suffix=day_iso,
                    overlay=(not args.no_overlay_daily),
                )

            # TIF
            if args.daily_outputs in ("both", "tif"):
                if rasterio is not None:
                    out_tif = _os.path.join(args.outdir, "lfmc_grids", f"lfmc_{day_iso}.tif")
                    save_lfmc_tif(grid, aoi, out_tif)

            # Danger maps (PNG/PDF) + optional TIF
            if (not args.no_daily_danger) and args.daily_outputs in ("both", "png"):
                cls = classify_lfmc(grid)
                base = _os.path.join(args.outdir, "lfmc_danger", f"lfmc_danger_{day_iso}")
                save_danger_map_and_hist(
                    cls, base, aoi=aoi,
                    gdf=(None if args.no_overlay_daily else gdf),
                    title_suffix=day_iso,
                    overlay=(not args.no_overlay_daily),
                )
            if args.write_danger_tif:
                cls = classify_lfmc(grid)
                out_dtif = _os.path.join(
                    args.outdir, "lfmc_danger", "tif", f"lfmc_danger_{day_iso}.tif"
                )
                save_danger_tif(cls, aoi, out_dtif, nodata_val=args.danger_tif_nodata)

    # --------------------------- Aggregation & composites ---------------------------
    if args.agg:
        freq = period_freq(args.agg)
        if not freq:
            print(f"Unknown aggregation kind: {args.agg}")
        else:
            agg_fun = np.nanmean if args.agg_stat == "mean" else np.nanmedian

            # Composite grids per period (stream from .npy paths)
            print("▶ Building composites…")
            daily_series = pd.Series(
                list(npy_paths.keys()),
                index=pd.to_datetime(list(npy_paths.keys()))
            )
            period_groups = daily_series.resample(freq).apply(list)

            comp_dir = _os.path.join(args.outdir, "lfmc_grids", "composites")
            danger_comp_dir = _os.path.join(args.outdir, "lfmc_danger", "composites")
            _os.makedirs(comp_dir, exist_ok=True)
            _os.makedirs(danger_comp_dir, exist_ok=True)

            for period_end, dlist in tqdm(
                list(period_groups.items()), desc="Composite periods"
            ):
                if not dlist:
                    continue
                arrs = []
                for dstr in dlist:
                    p = npy_paths.get(dstr, "")
                    if p and _os.path.exists(p):
                        arrs.append(np.load(p))
                if not arrs:
                    continue

                stack = np.stack(arrs, axis=0)
                comp = agg_fun(stack, axis=0)

                period_start = min(dlist)
                tag = f"{period_start}_to_{period_end.strftime('%Y-%m-%d')}"

                # PNG + (optional) TIF for composites
                comp_png = _os.path.join(comp_dir, f"lfmc_{tag}.png")
                save_lfmc_png(comp, aoi, comp_png, gdf=gdf, title_suffix=tag, overlay=True)
                if rasterio is not None and args.daily_outputs in ("both", "tif"):
                    comp_tif = _os.path.join(comp_dir, f"lfmc_{tag}.tif")
                    save_lfmc_tif(comp, aoi, comp_tif)

                # Danger composite (PNG/PDF) + optional TIF
                cls = classify_lfmc(comp)
                save_danger_map_and_hist(
                    cls,
                    _os.path.join(danger_comp_dir, f"lfmc_danger_{tag}"),
                    aoi=aoi, gdf=gdf, title_suffix=tag, overlay=True
                )
                if args.write_danger_tif:
                    out_dtif = _os.path.join(
                        args.outdir, "lfmc_danger", "tif", "composites",
                        f"lfmc_danger_{tag}.tif"
                    )
                    save_danger_tif(cls, aoi, out_dtif, nodata_val=args.danger_tif_nodata)

    if not args.keep_tmp:
        shutil.rmtree(_os.path.join(args.outdir, "_tmp_np_grids"), ignore_errors=True)

    print("✅ Done.")


if __name__ == "__main__":
    main()
