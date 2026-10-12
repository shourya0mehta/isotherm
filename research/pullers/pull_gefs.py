"""Pull GEFS ensemble mean and spread of the 6-hour maximum 2 m temperature at the 24 Kalshi airports (round 2:
an ensemble spread on free data; the ECMWF ensemble has no mean/spread product, GEFS does).

Source: NOAA GEFS on AWS (s3://noaa-gefs-pds, public): gefs.YYYYMMDD/HH/atmos/pgrb2ap5/{geavg,gespr}.tHHz.pgrb2a.0p50.fFFF,
one byte-range request per field via the .idx (TMAX:2 m above ground, the max over the 6 hours ending at fFFF).
Output: raw/gefs/YYYYMMDD_HHz.csv with icao, fh, tmax_mean_f, tmax_sd_f (fh = 6..48; window (fh-6, fh]).
Resumable. Env: START, END (dates), RUNS (default "0,12"), WORKERS (default 16), MLWX.
"""
import concurrent.futures as cf
import datetime as dt
import json
import os
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = os.environ.get("MLWX", "/home/claude/mlwx")
OUT = f"{ROOT}/raw/gefs"
B = "https://noaa-gefs-pds.s3.amazonaws.com"
COORDS = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "wxbot", "ml", "station_coords.json")))
FHS = list(range(6, 49, 6))
VAR = "TMAX:2 m above ground"


def fetch(url, rng=None, tries=4):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {})
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                return None
            time.sleep(2 * (k + 1))
        except Exception:
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed {url}")


def field(day, hh, prod, fh):
    base = f"{B}/gefs.{day:%Y%m%d}/{hh:02d}/atmos/pgrb2ap5/{prod}.t{hh:02d}z.pgrb2a.0p50.f{fh:03d}"
    idx = fetch(base + ".idx")
    if idx is None:
        return None
    lines = idx.decode().splitlines()
    for i, line in enumerate(lines):
        if VAR in line:
            start = int(line.split(":")[1])
            end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
            return fetch(base, (start, end))
    return None


def decode(msg):
    import eccodes
    gid = eccodes.codes_new_from_message(msg)
    try:
        v = eccodes.codes_get_values(gid)
        ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
        la1 = eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees")
        lo1 = eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees")
        d = eccodes.codes_get(gid, "iDirectionIncrementInDegrees")
    finally:
        eccodes.codes_release(gid)
    return v.reshape(nj, ni), la1, lo1, d


def station_values(A, la1, lo1, d):
    """Bilinear interpolation at each airport (0.5 deg grid, latitude decreasing from la1)."""
    out = {}
    nj, ni = A.shape
    for icao, c in COORDS.items():
        y = (la1 - c["lat"]) / d
        x = ((c["lon"] % 360) - lo1) / d
        j0, i0 = int(np.floor(y)), int(np.floor(x))
        fy, fx = y - j0, x - i0
        j1, i1 = min(j0 + 1, nj - 1), (i0 + 1) % ni
        v = (A[j0, i0] * (1 - fy) * (1 - fx) + A[j0, i1] * (1 - fy) * fx + A[j1, i0] * fy * (1 - fx)
             + A[j1, i1] * fy * fx)
        out[icao] = float(v)
    return out


def one_run(day, hh):
    tag = f"{day:%Y%m%d}_{hh:02d}z"
    path = f"{OUT}/{tag}.csv"
    if os.path.exists(path):
        return tag, "skip"
    lines = ["icao,fh,tmax_mean_f,tmax_sd_f"]
    rows = {}
    for fh in FHS:
        vals = {}
        for prod in ("geavg", "gespr"):
            msg = field(day, hh, prod, fh)
            if msg is None:
                return tag, "missing"
            A, la1, lo1, d = decode(msg)
            vals[prod] = station_values(A, la1, lo1, d)
        for icao in COORDS:
            m = (vals["geavg"][icao] - 273.15) * 9 / 5 + 32
            sd = vals["gespr"][icao] * 9 / 5
            rows.setdefault(icao, []).append(f"{icao},{fh},{m:.2f},{sd:.2f}")
    for icao in COORDS:
        lines += rows.get(icao, [])
    open(path + ".tmp", "w").write("\n".join(lines) + "\n")
    os.replace(path + ".tmp", path)
    return tag, "ok"


def main():
    os.makedirs(OUT, exist_ok=True)
    start = dt.date.fromisoformat(os.environ.get("START", "2020-10-01"))
    end = dt.date.fromisoformat(os.environ.get("END", "2026-09-30"))
    runs = [int(r) for r in os.environ.get("RUNS", "0,12").split(",")]
    jobs, d = [], end
    while d >= start:  # newest first
        jobs += [(d, hh) for hh in runs]
        d -= dt.timedelta(days=1)
    print(len(jobs), "runs", flush=True)
    t0, n, miss = time.time(), 0, 0
    with cf.ThreadPoolExecutor(int(os.environ.get("WORKERS", "16"))) as ex:
        for tag, st in ex.map(lambda j: one_run(*j), jobs):
            n += 1
            miss += st == "missing"
            if st == "missing":
                open(f"{OUT}/_missing.txt", "a").write(tag + "\n")
            if n % 50 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(jobs)} runs, {miss} missing, {n / (time.time() - t0):.2f} runs/s",
                      flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
