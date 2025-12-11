#!/usr/bin/env python3
"""
download_s2_bands_sentinelhub_to_local.py
Sentinel‑2 L2A downloader (no sentinelhub client) — patched:
  • SCL uses dataMask → writes 0 outside valid data (no NaN in uint8)
  • --scl_nodata N : set nodata tag for SCL outputs (e.g., 0) via gdal_edit.py or rio
Keeps: parallel HTTP, retries, tiling, presets from earlier v2.2.

python download_s2_bands_sentinelhub_to_local.py mosaic   \
--bbox 97 17 101.5 21   \
--start 2024-01-01 \
--end 2024-01-07   \
--out_dir data/S2   \
--maxcc 60 \
--mosaic_order leastCC   \
--workers 8 --max_retries 5 \
--retry_base 1.7   \
--tile_px 1024 \
--bands B04,B08,B11,SCL   \
--scl_nodata 0

python download_s2_bands_sentinelhub_to_local.py daily   \
--bbox 97 17 101.5 21   \
--start 2024-01-01 \
--end 2024-04-30   \
--out_dir data/S2   \
--maxcc 60 \
--mosaic_order leastCC   \
--workers 8 --max_retries 5 \
--retry_base 1.7   \
--tile_px 1024 \
--bands B04,B08,B11,SCL   \
--scl_nodata 0

"""
from __future__ import annotations
import os, sys, json, math, argparse, time, shutil, subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional requests; fall back to urllib
try:
    import requests  # type: ignore
    HAVE_REQUESTS = True
except Exception:
    import urllib.request, urllib.parse, urllib.error
    HAVE_REQUESTS = False

DEFAULT_BASE_URL = "https://services.sentinel-hub.com"
TOKEN_PATH = "/oauth/token"
PROCESS_PATH = "/api/v1/process"
MAX_API_PX = 2500  # per side

# Resolution per band
BAND_RES = {
    # 10 m
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    # 20 m
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20, "SCL": 20,
    # 60 m
    "B01": 60, "B09": 60, "B10": 60,
}

def _eval_float(band: str) -> str:
    return f"""//VERSION=3
function setup(){{return{{input:[{{bands:[\"{band}\"]}}],output:{{bands:1,sampleType:\"FLOAT32\"}}}}}}
function evaluatePixel(s){{return [ s.{band} ];}}"""

EVAL_BAND = {b: _eval_float(b) for b in BAND_RES if b != "SCL"}

# Patched SCL evalscript: write 0 when dataMask==0
EVAL_SCL = """//VERSION=3
function setup() {
  return { input: [{ bands: ["SCL","dataMask"] }], output: { bands: 1, sampleType: "UINT8" } };
}
function evaluatePixel(s) {
  return [ s.dataMask === 1 ? s.SCL : 0 ];
}
"""

# ---------- HTTP wrappers with retries ----------
class _Resp:
    def __init__(self, status_code: int, body: bytes):
        self.status_code = status_code
        self._body = body
    def json(self):
        return json.loads(self._body.decode("utf-8"))
    @property
    def text(self):
        try: return self._body.decode("utf-8")
        except Exception: return str(self._body)
    @property
    def content(self):
        return self._body

def http_post_form(url: str, form: Dict[str, str], headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> _Resp:
    headers = headers or {}
    if HAVE_REQUESTS:
        r = requests.post(url, data=form, headers=headers, timeout=timeout)
        return _Resp(r.status_code, r.content)
    data = urllib.parse.urlencode(form).encode("utf-8")
    h = {"Content-Type": "application/x-www-form-urlencoded"}; h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _Resp(resp.getcode(), resp.read())
    except urllib.error.HTTPError as e:
        return _Resp(e.getcode(), e.read())

def http_post_json(url: str, payload: Dict, headers: Optional[Dict[str, str]] = None, timeout: int = 600,
                   max_retries: int = 3, retry_base: float = 1.5) -> _Resp:
    headers = headers or {}
    attempt = 0
    while True:
        attempt += 1
        if HAVE_REQUESTS:
            r = requests.post(url, json=payload, headers=headers, timeout=timeout)
            resp = _Resp(r.status_code, r.content)
        else:
            data = json.dumps(payload).encode("utf-8")
            h = {"Content-Type": "application/json"}; h.update(headers)
            req = urllib.request.Request(url, data=data, headers=h, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as rr:
                    resp = _Resp(rr.getcode(), rr.read())
            except urllib.error.HTTPError as e:
                resp = _Resp(e.getcode(), e.read())
        if resp.status_code in (429, 500, 502, 503, 504) and attempt <= max_retries:
            time.sleep((retry_base ** (attempt-1)) + 0.1 * attempt)
            continue
        return resp

# ---------- Core helpers ----------
def require_env(var: str) -> str:
    val = os.getenv(var)
    if not val:
        raise SystemExit(f"Missing env var {var}. Export SH_CLIENT_ID/SH_CLIENT_SECRET.")
    return val

def get_oauth_token(base_url: str) -> str:
    client_id = require_env("SH_CLIENT_ID"); client_secret = require_env("SH_CLIENT_SECRET")
    url = base_url.rstrip("/") + TOKEN_PATH
    resp = http_post_form(url, {"grant_type": "client_credentials", "client_id": client_id, "client_secret": client_secret})
    if resp.status_code != 200:
        raise SystemExit(f"Auth failed ({resp.status_code}): {resp.text}")
    tok = resp.json().get("access_token")
    if not tok: raise SystemExit("Auth response missing access_token")
    return tok

def mosaicking_api_value(order: str) -> str:
    order = (order or "leastCC").lower()
    return {"leastrecent": "LEAST_RECENT", "mostrecent": "MOST_RECENT"}.get(order, "LEAST_CLOUD_COVERAGE")

def meters_per_degree(lat_deg: float) -> Tuple[float, float]:
    m_per_deg_lat = 111_132.92 - 559.82 * math.cos(2*math.radians(lat_deg)) + 1.175 * math.cos(4*math.radians(lat_deg))
    m_per_deg_lon = 111_412.84 * math.cos(math.radians(lat_deg)) - 93.5 * math.cos(3*math.radians(lat_deg))
    return m_per_deg_lon, m_per_deg_lat

def clamp_size(width: int, height: int, max_px: int = MAX_API_PX) -> Tuple[int, int]:
    if width <= max_px and height <= max_px: return width, height
    f = min(max_px/float(width), max_px/float(height))
    return max(1, int(width * f)), max(1, int(height * f))

def size_from_bbox(bbox: Tuple[float, float, float, float], res_m: float, size_override: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    if size_override: return clamp_size(int(size_override[0]), int(size_override[1]))
    xmin, ymin, xmax, ymax = bbox
    midlat = (ymin + ymax) / 2.0
    dlon = max(0.0, xmax - xmin); dlat = max(0.0, ymax - ymin)
    m_per_deg_lon, m_per_deg_lat = meters_per_degree(midlat)
    width = max(1, int(round((dlon * m_per_deg_lon) / res_m)))
    height = max(1, int(round((dlat * m_per_deg_lat) / res_m)))
    return clamp_size(width, height)

def build_payload(evalscript: str, bbox: Tuple[float, float, float, float], t0: str, t1: str,
                  width: int, height: int, maxcc: int, mosaic_order: str) -> Dict:
    return {
        "input": {
            "bounds": {
                "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "S2L2A",
                "dataFilter": {
                    "timeRange": {"from": t0, "to": t1},
                    "maxCloudCoverage": max(0, min(100, int(maxcc)))
                },
                "processing": {"mosaicking": mosaicking_api_value(mosaic_order)}
            }]
        },
        "evalscript": evalscript,
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }]
        }
    }

def post_process(base_url: str, token: str, payload: Dict, max_retries=3, retry_base=1.5) -> bytes:
    url = base_url.rstrip("/") + PROCESS_PATH
    headers = {"Authorization": f"Bearer {token}"}
    resp = http_post_json(url, payload, headers=headers, timeout=600, max_retries=max_retries, retry_base=retry_base)
    if resp.status_code != 200:
        msg = resp.text
        try: msg = json.dumps(resp.json(), ensure_ascii=False)
        except Exception: pass
        raise RuntimeError(f"Process API failed ({resp.status_code}): {msg}")
    return resp.content

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def daterange(start: datetime, end_excl: datetime):
    cur = start
    while cur < end_excl:
        yield cur
        cur += timedelta(days=1)

# ---------- Tiling ----------
def need_tiling(bbox, res_m, max_px=MAX_API_PX, size_override=None):
    w, h = size_from_bbox(bbox, res_m, size_override=size_override)
    return (w > max_px or h > max_px)

def split_bbox_grid(bbox: Tuple[float, float, float, float], nx: int, ny: int) -> List[Tuple[int, int, Tuple[float, float, float, float]]]:
    xmin, ymin, xmax, ymax = bbox
    dx = (xmax - xmin) / nx; dy = (ymax - ymin) / ny
    tiles = []
    for iy in range(ny):
        for ix in range(nx):
            bx = (xmin + ix*dx, ymin + iy*dy, xmin + (ix+1)*dx, ymin + (iy+1)*dy)
            tiles.append((ix, iy, bx))
    return tiles

def choose_grid(bbox, res_m, tile_px):
    w, h = size_from_bbox(bbox, res_m)
    nx = max(1, math.ceil(w / tile_px)); ny = max(1, math.ceil(h / tile_px))
    return nx, ny

# ---------- Nodata tagging for SCL ----------
def set_nodata_tag(path: str, nodata_val: int):
    """Try gdal_edit.py or rio edit-info to set nodata tag on a file; ignore if tools missing."""
    gdal_edit = shutil.which("gdal_edit.py")
    if gdal_edit:
        try:
            subprocess.run([gdal_edit, "-a_nodata", str(nodata_val), path],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return
        except Exception:
            pass
    rio = shutil.which("rio")
    if rio:
        try:
            subprocess.run([rio, "edit-info", "--nodata", str(nodata_val), path],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return
        except Exception:
            pass

# ---------- Workers ----------
@dataclass
class Job:
    band: str
    date_tag: str
    t0: str
    t1: str
    bbox: Tuple[float, float, float, float]
    out_path: Path
    width: int
    height: int
    scl_nodata_val: Optional[int] = None  # only used when band == "SCL"

def run_jobs(base_url, token, jobs: List[Job], eval_by_band: Dict[str, str], maxcc: int, mosaic_order: str,
             workers: int, max_retries: int, retry_base: float, dry_run: bool):
    if not jobs: return
    ensure_dir(jobs[0].out_path.parent)
    if workers <= 1:
        for j in jobs:
            _do_one(base_url, token, j, eval_by_band, maxcc, mosaic_order, max_retries, retry_base, dry_run)
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_do_one, base_url, token, j, eval_by_band, maxcc, mosaic_order, max_retries, retry_base, dry_run) for j in jobs]
        for f in as_completed(futs):
            exc = f.exception()
            if exc: print(f"  ! job failed: {exc}")

def _do_one(base_url, token, job: Job, eval_by_band, maxcc, mosaic_order, max_retries, retry_base, dry_run):
    evalscript = eval_by_band[job.band]
    payload = build_payload(evalscript, job.bbox, job.t0, job.t1, job.width, job.height, maxcc, mosaic_order)
    if dry_run:
        print(f"DRY {job.band} {job.date_tag} {job.width}x{job.height} → {job.out_path.name}")
        return
    data = post_process(base_url, token, payload, max_retries=max_retries, retry_base=retry_base)
    job.out_path.write_bytes(data)
    print(f"  saved {job.out_path}")
    if job.band == "SCL" and job.scl_nodata_val is not None:
        set_nodata_tag(str(job.out_path), job.scl_nodata_val)

# ---------- CLI ----------
PRESETS = {
    "veg":  "B02,B03,B04,B05,B06,B07,B08,B8A",
    "atmos":"B01,B09,B10",
    "swir": "B11,B12",
    "qa":   "SCL",
    "lfmc": "B04,B08,B11,SCL",
    "fire": "B08,B11,B12",
    "soil": "B11,B12,B8A",
    "water":"B02,B03,B04,B08",
    "all":  "B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B10,B11,B12,SCL",
}

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fast S2 L2A downloader (B01..B12,SCL) via Sentinel Hub Process API", add_help=True)
    sub = p.add_subparsers(dest='mode')

    def add_common(ap):
        ap.add_argument('--bbox', nargs=4, type=float, metavar=('xmin','ymin','xmax','ymax'), help='Lon/Lat bbox (WGS84)')
        ap.add_argument('--start', help='Start YYYY-MM-DD'); ap.add_argument('--end', help='End YYYY-MM-DD inclusive')
        ap.add_argument('--out_dir', default='downloads', help='Output directory')
        ap.add_argument('--maxcc', type=int, default=80, help='Max cloud cover %')
        ap.add_argument('--mosaic_order', default='leastCC', choices=['mostRecent','leastCC','leastRecent'])
        ap.add_argument('--size', nargs=2, type=int, metavar=('WIDTH','HEIGHT'), help='Override output size')
        ap.add_argument('--base_url', default=DEFAULT_BASE_URL, help='Sentinel Hub base URL')
        ap.add_argument('--dry_run', action='store_true', help='Build payloads but do not call the API')
        ap.add_argument('--workers', type=int, default=6, help='Parallel HTTP workers')
        ap.add_argument('--max_retries', type=int, default=4)
        ap.add_argument('--retry_base', type=float, default=1.6)
        ap.add_argument('--bands', default='B04,B08,B11,SCL', help='Comma list of bands to fetch (ignored if --preset given)')
        ap.add_argument('--preset', choices=list(PRESETS.keys()), help='Convenience band groups')
        ap.add_argument('--tile_px', type=int, default=1024, help='Tile size when tiling is needed (px per side)')
        ap.add_argument('--scl_nodata', type=int, default=None, help='Set nodata tag for SCL outputs (e.g., 0)')

    d = sub.add_parser('daily', help='Per-day GeoTIFFs per band')
    add_common(d)

    m = sub.add_parser('mosaic', help='One mosaic per band for the whole range')
    add_common(m)

    t = sub.add_parser('selftest', help='Run offline tests')
    return p

def parse_args(argv: Optional[List[str]] = None):
    p = build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) == 0:
        return argparse.Namespace(mode='selftest')
    return p.parse_args(argv)

# ---------- Main ----------
def main():
    args = parse_args()
    if args.mode == 'selftest':
        print("Selftest OK"); return

    token = get_oauth_token(args.base_url)
    out_dir = Path(args.out_dir); ensure_dir(out_dir)
    bbox = tuple(args.bbox) if args.bbox else (97.0, 17.0, 102.0, 21.0)
    size_override = tuple(args.size) if args.size else None
    bands_str = PRESETS[args.preset] if getattr(args, "preset", None) else args.bands
    bands = [b.strip().upper() for b in bands_str.split(",") if b.strip()]
    # build eval map only for requested bands
    eval_by_band = {}
    for b in bands:
        if b == "SCL":
            eval_by_band[b] = EVAL_SCL
        else:
            if b not in EVAL_BAND:
                raise SystemExit(f"Unknown band '{b}'")
            eval_by_band[b] = EVAL_BAND[b]

    if args.mode == 'daily':
        start = datetime.fromisoformat(args.start); end = datetime.fromisoformat(args.end) + timedelta(days=1)
        for day in daterange(start, end):
            date_tag = day.strftime('%Y%m%d')
            t0 = day.strftime('%Y-%m-%dT00:00:00Z'); t1 = (day + timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
            print(f"[Daily] {date_tag} …")

            plan: List[Job] = []
            for band in bands:
                res_m = BAND_RES.get(band, 20)
                if need_tiling(bbox, res_m, size_override=size_override):
                    nx, ny = choose_grid(bbox, res_m, args.tile_px)
                    for ix, iy, sub in split_bbox_grid(bbox, nx, ny):
                        w, h = size_from_bbox(sub, res_m, size_override=size_override)
                        out_name = f"{band}_{date_tag}_x{ix}_y{iy}.tif"
                        kw = dict(scl_nodata_val=args.scl_nodata) if band == "SCL" else {}
                        plan.append(Job(band, date_tag, t0, t1, sub, out_dir/out_name, w, h, **kw))
                else:
                    w, h = size_from_bbox(bbox, res_m, size_override=size_override)
                    kw = dict(scl_nodata_val=args.scl_nodata) if band == "SCL" else {}
                    plan.append(Job(band, date_tag, t0, t1, bbox, out_dir/f"{band}_{date_tag}.tif", w, h, **kw))

            if len(plan):
                run_jobs(args.base_url, token, plan, eval_by_band, args.maxcc, args.mosaic_order,
                         args.workers, args.max_retries, args.retry_base, args.dry_run)

    elif args.mode == 'mosaic':
        t0 = f"{args.start}T00:00:00Z"; t1 = f"{args.end}T23:59:59Z"
        print(f"[Mosaic] {args.start} → {args.end}")
        for band in bands:
            res_m = BAND_RES.get(band, 20)
            if need_tiling(bbox, res_m, size_override=size_override):
                nx, ny = choose_grid(bbox, res_m, args.tile_px)
                plan: List[Job] = []
                for ix, iy, sub in split_bbox_grid(bbox, nx, ny):
                    w, h = size_from_bbox(sub, res_m, size_override=size_override)
                    out_name = f"{band}_{args.start}_to_{args.end}_{args.mosaic_order}_x{ix}_y{iy}.tif"
                    kw = dict(scl_nodata_val=args.scl_nodata) if band == "SCL" else {}
                    plan.append(Job(band, f"{args.start}_to_{args.end}", t0, t1, sub, out_dir/out_name, w, h, **kw))
                run_jobs(args.base_url, token, plan, eval_by_band, args.maxcc, args.mosaic_order,
                         args.workers, args.max_retries, args.retry_base, args.dry_run)
            else:
                w, h = size_from_bbox(bbox, res_m, size_override=size_override)
                out_name = f"{band}_{args.start}_to_{args.end}_{args.mosaic_order}.tif"
                kw = dict(scl_nodata_val=args.scl_nodata) if band == "SCL" else {}
                job = Job(band, f"{args.start}_to_{args.end}", t0, t1, bbox, out_dir/out_name, w, h, **kw)
                run_jobs(args.base_url, token, [job], eval_by_band, args.maxcc, args.mosaic_order,
                         args.workers, args.max_retries, args.retry_base, args.dry_run)
    else:
        print("Unknown mode"); sys.exit(2)

if __name__ == '__main__':
    main()
