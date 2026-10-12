"""NBM's hourly runs for the live forecaster (round 5's fresher-NBM inputs; research/ml/pull_nbm_hourly.py built the
training archive with the same parser).

IEM's text feed carries the 00/06/12/18Z NBM cycles only. NOAA's bucket posts every hourly run's station text about
65-75 minutes after the hour. At a decision time `dec` the inputs come from the latest run whose file was posted at
least MARGIN before `dec` (training used the file's Last-Modified time the same way): its max for each target day
(TXN), the max's spread (XND) and the run's age in hours.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import pandas as pd

log = logging.getLogger("wxbot.nbm_hourly")
BUCKET = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
MARGIN = pd.Timedelta(minutes=5)


def parse(text: str, run: pd.Timestamp, stations) -> list:
    """(icao, D, txn, xnd) from one NBS bulletin: values in the 00Z columns of the TXN / XND rows are the max of the
    previous local day."""
    want = set(stations)
    out = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        icao = line[1:5]
        if icao not in want or "NBS GUIDANCE" not in line:
            continue
        rows = {b[1:4]: b for b in lines[i + 1:i + 30] if len(b) > 5}
        fhr, txn, xnd = rows.get("FHR"), rows.get("TXN"), rows.get("XND")
        if fhr is None or txn is None:
            continue
        for k in range((len(fhr) - 5) // 3 + 1):
            h = fhr[5 + 3 * k:8 + 3 * k].strip()
            v = txn[5 + 3 * k:8 + 3 * k].strip() if len(txn) >= 8 + 3 * k else ""
            if not h or not v:
                continue
            ft = run + pd.Timedelta(hours=int(h))
            if ft.hour != 0:
                continue
            s = xnd[5 + 3 * k:8 + 3 * k].strip() if xnd is not None and len(xnd) >= 8 + 3 * k else ""
            out.append((icao, (ft - pd.Timedelta(days=1)).normalize(), float(v), float(s) if s else float("nan")))
    return out


def url_for(run: pd.Timestamp) -> str:
    return f"{BUCKET}/blend.{run:%Y%m%d}/{run:%H}/text/blend_nbstx.t{run:%H}z"


def _get(url: str, head: bool = False, timeout: float = 120.0):
    import requests
    for k in range(3):
        try:
            r = (requests.head if head else requests.get)(url, timeout=timeout)
            if r.status_code in (200, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 * (k + 1))
    return None


RUN_HOURS = (1, 7, 10, 13, 16, 19)   # the runs the training archive holds


def _run(run: pd.Timestamp, dec: pd.Timestamp, stations, cache_dir: str) -> Optional[dict]:
    """One run's parsed rows {(icao, D): (txn, xnd)} if its file was posted at least MARGIN before dec, else None.
    Parsed runs are cached as small JSON files (their posting time included)."""
    cache = os.path.join(cache_dir, f"{run:%Y%m%d_%H}z.json")
    if os.path.exists(cache):
        d = json.load(open(cache))
        posted = pd.Timestamp(d["posted"])
        if posted + MARGIN > pd.Timestamp(dec):
            return None
        return {"run": run, "posted": posted, "rows": {(i, pd.Timestamp(D)): (t, x) for i, D, t, x in d["rows"]}}
    h = _get(url_for(run), head=True)
    if h is None or h.status_code != 200 or not h.headers.get("Last-Modified"):
        return None
    posted = pd.Timestamp(h.headers["Last-Modified"]).tz_convert("UTC").tz_localize(None)
    if posted + MARGIN > pd.Timestamp(dec):
        return None
    r = _get(url_for(run))
    if r is None or r.status_code != 200:
        return None
    rows = parse(r.text, run, stations)
    tmp = cache + ".tmp"
    json.dump({"run": str(run), "posted": str(posted),
               "rows": [(i, str(D.date()), t, None if x != x else x) for i, D, t, x in rows]}, open(tmp, "w"))
    os.replace(tmp, cache)
    return {"run": run, "posted": posted, "rows": {(i, D): (t, x) for i, D, t, x in rows}}


def latest(dec: pd.Timestamp, stations, cache_dir: str, max_back_h: int = 24, hours=RUN_HOURS,
           need=None) -> Optional[dict]:
    """The freshest value for each (icao, D) usable at decision time dec (naive UTC), as training chose it
    (research/ml/r5_nbm_fresh_feats.py: per station and day, the latest archived run posted at least MARGIN before
    the decision that carries that day's max; a run that starts after the day's max period began has no value for that
    day, so later in the day an older run supplies it). Returns {"run": newest run used, "posted": ..., "rows":
    {(icao, D): (txn, xnd, run)}}, or None if no run qualifies. `need` (a set of (icao, D)) stops the walk back once
    every needed day has a value."""
    os.makedirs(cache_dir, exist_ok=True)
    out, first = {}, None
    for back in range(1, max_back_h + 1):
        run = (pd.Timestamp(dec) - pd.Timedelta(hours=back)).floor("h")
        if run.hour not in hours:
            continue
        got = _run(run, dec, stations, cache_dir)
        if got is None:
            continue
        first = first or got
        for k, (t, x) in got["rows"].items():
            if k not in out:
                out[k] = (t, x, run)
        if need is not None and all(k in out for k in need):
            break
    if first is None:
        return None
    return {"run": first["run"], "posted": first["posted"], "rows": out}


def columns(df: pd.DataFrame, fresh: Optional[dict], dec_col: str = "_dec") -> pd.DataFrame:
    """nbf_txn, nbf_xnd, nbf_age_h for rows (icao, D) (NaN where no usable run has a value)."""
    out = pd.DataFrame(index=df.index, columns=["nbf_txn", "nbf_xnd", "nbf_age_h"], dtype=float)
    if not fresh:
        return out
    for i, r in zip(df.index, df.itertuples()):
        v = fresh["rows"].get((r.icao, pd.Timestamp(r.D)))
        if v is None:
            continue
        dec = pd.Timestamp(r.D) + pd.Timedelta(hours=int(r.h))
        run = v[2] if len(v) > 2 else fresh["run"]
        out.loc[i, "nbf_txn"] = v[0]
        out.loc[i, "nbf_xnd"] = v[1] if v[1] is not None else float("nan")
        out.loc[i, "nbf_age_h"] = (dec - run) / pd.Timedelta(hours=1)
    return out
