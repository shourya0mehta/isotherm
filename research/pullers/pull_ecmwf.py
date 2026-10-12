"""ECMWF IFS open data (CC-BY-4.0, AWS mirror): 2 m temperature at our 24 stations for the steps that cover each
day's afternoon (valid 18Z, 21Z, 00Z+1). Byte-range fetch of the single 2t message per file; decode with eccodes;
bilinear + nearest point. Runs used: 00Z, 06Z, 12Z, 18Z. Resumable; one CSV per run-month.
Usage: python3 pull_ecmwf.py [start YYYY-MM-DD] [end YYYY-MM-DD] [threads]
"""
import csv, datetime as dt, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
import eccodes

B = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"
OUT = "raw/ecmwf"
os.makedirs(OUT, exist_ok=True)
ST = json.load(open("station_coords.json"))
NAMES = sorted(ST)
STEPS = {0: [18, 21, 24], 6: [12, 15, 18], 12: [6, 9, 12, 30, 33, 36], 18: [24, 27, 30]}
sess = threading.local()


def S():
    if not hasattr(sess, "s"):
        sess.s = requests.Session()
    return sess.s


def get(url, rng=None, tries=8):
    h = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
    for k in range(tries):
        try:
            r = S().get(url, headers=h, timeout=60)
        except Exception:
            time.sleep(2 + 2 * k); continue
        if r.status_code in (200, 206):
            if r.content[:5] == b"<?xml":  # throttling message
                time.sleep(2 + 2 * k); continue
            return r
        if r.status_code == 404:
            return None
        time.sleep(2 + 2 * k)
    return None


def bases(day: dt.date, hh: int):
    d = f"{day:%Y%m%d}"
    streams = ["oper"] if hh in (0, 12) else ["scda", "oper"]
    out = []
    for res in ("ifs/0p25", "0p4-beta"):
        for s in streams:
            out.append((f"{B}/{d}/{hh:02d}z/{res}/{s}/{d}{hh:02d}0000-{{step}}h-{s}-fc", res))
    return out


_w = {}


def weights(msg):
    """Cache bilinear weights per grid geometry."""
    ni, nj = eccodes.codes_get(msg, "Ni"), eccodes.codes_get(msg, "Nj")
    la0 = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
    lo0 = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")
    di = eccodes.codes_get(msg, "iDirectionIncrementInDegrees")
    dj = eccodes.codes_get(msg, "jDirectionIncrementInDegrees")
    key = (ni, nj, la0, lo0, di, dj)
    if key in _w:
        return _w[key]
    W = []
    for n in NAMES:
        lat, lon = ST[n]["lat"], ST[n]["lon"]
        x = ((lon - lo0) % 360.0) / di
        y = (la0 - lat) / dj
        i0, j0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - i0, y - j0
        idx = [(j0 * ni + (i0 % ni)), (j0 * ni + ((i0 + 1) % ni)), ((j0 + 1) * ni + (i0 % ni)), ((j0 + 1) * ni + ((i0 + 1) % ni))]
        w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
        near = idx[int(np.argmax(w))]
        W.append((idx, w, near))
    _w[key] = W
    return W


def one(day: dt.date, hh: int):
    fn = f"{OUT}/{day:%Y%m}_{hh:02d}.csv"
    rows = []
    base = None
    for b, res in bases(day, hh):
        if get(b.format(step=STEPS[hh][0]) + ".index") is not None:
            base = (b, res); break
    if base is None:
        return day, hh, 0
    for step in STEPS[hh]:
        r = get(base[0].format(step=step) + ".index")
        if r is None:
            continue
        rec = None
        for line in r.text.splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("param") == "2t" and d.get("levtype") == "sfc":
                rec = d; break
        if rec is None:
            continue
        g = get(base[0].format(step=step) + ".grib2", (rec["_offset"], rec["_offset"] + rec["_length"] - 1))
        if g is None:
            continue
        msg = eccodes.codes_new_from_message(g.content)
        try:
            v = eccodes.codes_get_values(msg)
            W = weights(msg)
        finally:
            eccodes.codes_release(msg)
        run = dt.datetime(day.year, day.month, day.day, hh)
        valid = run + dt.timedelta(hours=step)
        for n, (idx, w, near) in zip(NAMES, W):
            rows.append([run.isoformat(), step, valid.isoformat(), n, round(float(np.dot(v[idx], w)) - 273.15, 3),
                         round(float(v[near]) - 273.15, 3), base[1]])
    with lock:
        new = not os.path.exists(fn)
        with open(fn, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["run", "step", "valid", "icao", "t2m_bil", "t2m_near", "grid"])
            wr.writerows(rows)
        with open(f"{OUT}/_done.txt", "a") as f:
            f.write(f"{day:%Y-%m-%d} {hh}\n")
    return day, hh, len(rows)


lock = threading.Lock()
if __name__ == "__main__":
    a = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date(2023, 1, 18)
    b = dt.date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else dt.date(2026, 10, 1)
    nth = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    done = set()
    if os.path.exists(f"{OUT}/_done.txt"):
        done = set(open(f"{OUT}/_done.txt").read().split("\n"))
    jobs = []
    d = b
    while d >= a:  # newest first
        for hh in (0, 6, 12, 18):
            if f"{d:%Y-%m-%d} {hh}" not in done:
                jobs.append((d, hh))
        d -= dt.timedelta(days=1)
    print("jobs", len(jobs), flush=True)
    t0 = time.time(); n = 0
    with ThreadPoolExecutor(nth) as ex:
        for day, hh, k in ex.map(lambda j: one(*j), jobs):
            n += 1
            if n % 50 == 0:
                print(time.strftime("%H:%M:%S"), day, hh, k, f"{n / (time.time() - t0):.2f} jobs/s", flush=True)
    print("done", flush=True)
