"""Re-pull HRRR 2 m temperature at chosen grid points for a few stations (round 2: KSFO's nearest HRRR grid point is
over San Francisco Bay, land fraction 0, which makes HRRR run 7 F cold there; the nearest land point is one cell west,
2.3 km from the station). Copies every other station from raw/hrrr/ so the output is a drop-in replacement.

Output: raw/hrrr_land/YYYYMMDD_HHz.csv (icao, fh, t2m_f), same runs as raw/hrrr. Resumable.
Env: POINTS (JSON {icao: [iy, ix]}, default the KSFO land point), WORKERS (default 16), MLWX.
"""
import concurrent.futures as cf
import glob
import json
import os
import time
import urllib.error
import urllib.request

import numcodecs
import numpy as np

ROOT = os.environ.get("MLWX", "/home/claude/mlwx")
SRC, OUT = f"{ROOT}/raw/hrrr", f"{ROOT}/raw/hrrr_land"
B = "https://hrrrzarr.s3.amazonaws.com/sfc"
POINTS = json.loads(os.environ.get("POINTS", '{"KSFO": [595, 177]}'))
BLOSC = numcodecs.Blosc()


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


def one_run(src_path):
    tag = os.path.basename(src_path)[:-4]
    out = f"{OUT}/{tag}.csv"
    if os.path.exists(out):
        return tag, "skip"
    day = tag[:8]
    base = f"{B}/{day}/{tag}_fcst.zarr/2m_above_ground/TMP/2m_above_ground/TMP"
    meta = fetch(f"{base}/.zarray")
    if meta is None:
        return tag, "missing"
    meta = json.loads(meta)
    vals = {}
    for icao, (iy, ix) in POINTS.items():
        raw = fetch(f"{base}/0.{iy // 150}.{ix // 150}")
        if raw is None:
            return tag, "missing"
        a = np.frombuffer(BLOSC.decode(raw), dtype=meta["dtype"]).astype(np.float32)
        nt = a.size // (150 * 150)
        a = a.reshape(nt, 150, 150)[:, iy % 150, ix % 150].astype(float)
        a[(a < 180) | (a > 340)] = np.nan
        vals[icao] = (a - 273.15) * 9 / 5 + 32
    # other stations unchanged (only the columns the readers use: icao, fh, t2m_f)
    rows = [",".join(ln.split(",")[:3]) for ln in open(src_path).read().splitlines()[1:]
            if ln and ln.split(",")[0] not in POINTS]
    for icao, v in vals.items():
        rows += [f"{icao},{i + 1},{t:.2f}" for i, t in enumerate(v)]
    open(out + ".tmp", "w").write("icao,fh,t2m_f\n" + "\n".join(rows) + "\n")
    os.replace(out + ".tmp", out)
    return tag, "ok"


def main():
    os.makedirs(OUT, exist_ok=True)
    jobs = sorted(glob.glob(f"{SRC}/*_*z.csv"), reverse=True)
    print(len(jobs), "runs", POINTS, flush=True)
    t0, n, miss = time.time(), 0, 0
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "16"))) as ex:
        for tag, st in ex.map(one_run, jobs):
            n += 1
            miss += st == "missing"
            if n % 500 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(jobs)} runs, {miss} missing, {n / (time.time() - t0):.2f} runs/s",
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
