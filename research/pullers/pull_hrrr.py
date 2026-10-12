"""Pull HRRR 2 m temperature (and dewpoint) forecasts at the 24 Kalshi airports from the University of Utah
HRRR-zarr archive on AWS (s3://hrrrzarr, public). Each zarr chunk is a 150x150 grid tile holding all 48 forecast
hours of a synoptic run, so one ~1 MB download serves every station in that tile.

Output: raw/hrrr/YYYYMMDD_HHz.csv with columns icao, fh, t2m_f, d2m_f (fh = forecast hour 1..48; valid = run + fh h).
Resumable: runs already on disk are skipped; missing runs are logged to raw/hrrr/_missing.txt.
Env: START (YYYY-MM-DD), END (inclusive), RUNS (e.g. "0,12"), WORKERS (default 16).
"""
import concurrent.futures as cf
import datetime as dt
import json
import os
import sys
import time
import urllib.error
import urllib.request

import numcodecs
import numpy as np
from pyproj import Proj

ROOT = os.environ.get("MLWX", "/home/claude/mlwx")
OUT = f"{ROOT}/raw/hrrr"
B = "https://hrrrzarr.s3.amazonaws.com/sfc"
COORDS = json.load(open("/home/claude/kalshi-wx-bot/wxbot/ml/station_coords.json"))
X0, Y0, DX = -2697520.14252193, -1587306.15255666, 3000.0
LCC = Proj("+proj=lcc +lat_0=38.5 +lon_0=-97.5 +lat_1=38.5 +lat_2=38.5 +x_0=0 +y_0=0 +R=6371229 +units=m +no_defs")
BLOSC = numcodecs.Blosc()
VARS = {"t2m_f": "2m_above_ground/TMP"}  # dewpoint ("d2m_f": "2m_above_ground/DPT") skipped to halve downloads

STATIONS = {}
for icao, c in COORDS.items():
    x, y = LCC(c["lon"], c["lat"])
    ix, iy = int(round((x - X0) / DX)), int(round((y - Y0) / DX))
    STATIONS[icao] = (iy // 150, ix // 150, iy % 150, ix % 150)
TILES = sorted({(v[0], v[1]) for v in STATIONS.values()})


def fetch(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed {url}")


def one_run(day: dt.date, hh: int):
    tag = f"{day:%Y%m%d}_{hh:02d}z"
    path = f"{OUT}/{tag}.csv"
    if os.path.exists(path):
        return tag, "skip"
    series = {icao: {} for icao in STATIONS}
    for col, var in VARS.items():
        leaf = var.split("/")[1]
        base = f"{B}/{day:%Y%m%d}/{tag}_fcst.zarr/{var}/{var.split('/')[0]}/{leaf}"
        meta = fetch(f"{base}/.zarray")
        if meta is None:
            if col == "t2m_f":
                return tag, "missing"
            continue
        meta = json.loads(meta)
        dtype, fill = meta["dtype"], meta.get("fill_value")  # float16 before ~2024-03, float32 after
        nt_meta = int(meta["chunks"][0])
        for ty, tx in TILES:
            raw = fetch(f"{base}/0.{ty}.{tx}")
            if raw is None:
                if col == "t2m_f":
                    return tag, "missing"
                continue
            a = np.frombuffer(BLOSC.decode(raw), dtype=dtype).astype(np.float32)
            nt = a.size // (150 * 150)
            if nt != nt_meta:
                return tag, f"error shape {nt} vs {nt_meta}"
            a = a.reshape(nt, 150, 150)
            for icao, (sy, sx, oy, ox) in STATIONS.items():
                if (sy, sx) == (ty, tx):
                    v = a[:, oy, ox].astype(float)
                    v[(v < 200) | (v > 340)] = np.nan  # fill values (-9999 / -10000) and anything non-physical (K)
                    series[icao][col] = (v - 273.15) * 9 / 5 + 32
    lines = ["icao,fh,t2m_f,d2m_f"]
    for icao, s in series.items():
        t, d = s.get("t2m_f"), s.get("d2m_f")
        if t is None:
            continue
        for i in range(len(t)):
            dv = d[i] if d is not None and i < len(d) else np.nan
            lines.append(f"{icao},{i + 1},{t[i]:.2f},{dv:.2f}")
    tmp = path + ".tmp"
    open(tmp, "w").write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return tag, "ok"


def main():
    os.makedirs(OUT, exist_ok=True)
    start = dt.date.fromisoformat(os.environ.get("START", "2024-01-01"))
    end = dt.date.fromisoformat(os.environ.get("END", "2026-09-30"))
    runs = [int(r) for r in os.environ.get("RUNS", "0,12").split(",")]
    workers = int(os.environ.get("WORKERS", "16"))
    jobs = []
    d = end
    while d >= start:  # newest first: the most relevant months land first
        for hh in runs:
            jobs.append((d, hh))
        d -= dt.timedelta(days=1)
    print(len(STATIONS), "stations in", len(TILES), "tiles;", len(jobs), "runs", flush=True)
    t0, n, miss = time.time(), 0, 0
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(one_run, d, hh): (d, hh) for d, hh in jobs}
        for f in cf.as_completed(futs):
            try:
                tag, st = f.result()
            except Exception as e:  # keep going; a rerun retries it
                tag, st = str(futs[f]), f"error {e}"
            n += 1
            if st == "missing":
                miss += 1
                open(f"{OUT}/_missing.txt", "a").write(tag + "\n")
            elif st.startswith("error"):
                print("ERR", tag, st, flush=True)
            if n % 50 == 0 or n == len(jobs):
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(jobs)} runs, {miss} missing, {n / (time.time() - t0):.2f} runs/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
