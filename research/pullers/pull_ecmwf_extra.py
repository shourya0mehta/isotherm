"""ECMWF IFS open data, convection-related fields at our 24 stations: CAPE and 2 m dewpoint at the 21Z step, and
accumulated precipitation at the 18Z and 00Z+1 steps (daytime precip = difference). Same runs/steps as pull_ecmwf.py.
Output raw/ecmwf_x/{YYYYMM}_{HH}.csv: run, step, valid, icao, param, value. Resumable, newest first."""
import csv, datetime as dt, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, eccodes
import pull_ecmwf as P
OUT = "raw/ecmwf_x"; os.makedirs(OUT, exist_ok=True)
A = ["tp", "mx2t3"]; M = ["cape", "mucape", "2d", "tcc", "mx2t3"]  # field set grew over time; take what exists
WANT = {0: [(18, A), (21, M), (24, A)], 6: [(12, A), (15, M), (18, A)],
        12: [(6, A), (9, M), (12, A), (30, A), (33, M), (36, A)], 18: [(24, A), (27, M), (30, A)]}
lock = threading.Lock()

def one(day, hh):
    base = None
    for b, res in P.bases(day, hh):
        if P.get(b.format(step=WANT[hh][0][0]) + ".index") is not None:
            base = (b, res); break
    if base is None:
        return day, hh, 0
    rows = []
    for step, params in WANT[hh]:
        r = P.get(base[0].format(step=step) + ".index")
        if r is None:
            continue
        recs = {}
        for line in r.text.splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("levtype") == "sfc" and d.get("param") in params:
                recs[d["param"]] = d
        for prm, rec in recs.items():
            g = P.get(base[0].format(step=step) + ".grib2", (rec["_offset"], rec["_offset"] + rec["_length"] - 1))
            if g is None:
                continue
            msg = eccodes.codes_new_from_message(g.content)
            try:
                v = eccodes.codes_get_values(msg); W = P.weights(msg)
            finally:
                eccodes.codes_release(msg)
            run = dt.datetime(day.year, day.month, day.day, hh); valid = run + dt.timedelta(hours=step)
            for n, (idx, w, near) in zip(P.NAMES, W):
                rows.append([run.isoformat(), step, valid.isoformat(), n, prm, round(float(np.dot(v[idx], w)), 5)])
    with lock:
        fn = f"{OUT}/{day:%Y%m}_{hh:02d}.csv"; new = not os.path.exists(fn)
        with open(fn, "a", newline="") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["run", "step", "valid", "icao", "param", "value"])
            wr.writerows(rows)
        with open(f"{OUT}/_done.txt", "a") as f:
            f.write(f"{day:%Y-%m-%d} {hh}\n")
    return day, hh, len(rows)

if __name__ == "__main__":
    a = dt.date.fromisoformat(sys.argv[1]); b = dt.date.fromisoformat(sys.argv[2]); nth = int(sys.argv[3])
    done = set(open(f"{OUT}/_done.txt").read().split("\n")) if os.path.exists(f"{OUT}/_done.txt") else set()
    jobs = []; d = b
    while d >= a:
        for hh in (0, 6, 12, 18):
            if f"{d:%Y-%m-%d} {hh}" not in done:
                jobs.append((d, hh))
        d -= dt.timedelta(days=1)
    print("jobs", len(jobs), flush=True); t0 = time.time(); n = 0
    with ThreadPoolExecutor(nth) as ex:
        for day, hh, k in ex.map(lambda j: one(*j), jobs):
            n += 1
            if n % 50 == 0:
                print(time.strftime("%H:%M:%S"), day, hh, k, f"{n / (time.time() - t0):.2f} jobs/s", flush=True)
    print("done", flush=True)
