"""ECMWF AIFS ENS 2 m temperature at the stations, read live from ECMWF's open data: candidate C's extra inputs
(docs/HOLDOUT_DECJAN.md, amendment C1). The research built them from the dynamical.org archive of the same open data
(research/ml/r4_ens_features.py); this reads the source directly, the same way.

For one run: the control (cf) and the 50 perturbed members (pf) of 2 m temperature at the 6-hourly steps a decision
needs, each field fetched by its byte range from the run's .index files, at the nearest 0.25-degree grid point to
each station (the archive's rule). About 0.6 MB per field, so about 31 MB per step. Saved as
{root}/aifs_ens/YYYYMMDD_HHz.npz (t2m_c[station, step, member], stations, steps), merged across calls.

features() turns a run into the model's inputs exactly as r4_ens_features.features does: per member the highest value
over the steps inside the rest of the local day, then across members the mean, standard deviation, 10th and 90th
percentiles (F) and the run's age, for the latest run public at the decision (init + DELAY_H).
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("wxbot.aifs")
HOSTS = ["https://data.ecmwf.int/forecasts",                       # the last few days
         "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]  # AWS mirror: older runs
DELAY_H = 7            # the research's publication rule (06Z steps 24-54 were out by 12:49Z on Oct 7, 2026)
NMEMBERS = 51
MAX_STEP = 60          # the research archive kept steps up to 60 h
HOUR = np.timedelta64(1, "h")


def bitround(x: np.ndarray, keepbits: int = 7) -> np.ndarray:
    """The archive's lossy compression (dynamical.org stores 2 m temperature in C as float32 keeping 7 mantissa bits,
    rounded to nearest, ties to even, as numcodecs BitRound): applied to live values so C sees the inputs it was
    trained on (without it, member values differ by up to 0.125 C)."""
    b = np.asarray(x, np.float32).copy().view(np.uint32)
    drop = np.uint32(23 - keepbits)
    mask = np.uint32((0xFFFFFFFF >> int(drop)) << int(drop))
    half1 = np.uint32((1 << (int(drop) - 1)) - 1)
    b += ((b >> drop) & np.uint32(1)) + half1
    b &= mask
    return b.view(np.float32).astype(float)


def run_for(dec: pd.Timestamp) -> pd.Timestamp:
    """The latest 6-hourly run public at decision time dec (naive UTC): init <= dec - DELAY_H."""
    t = pd.Timestamp(dec) - pd.Timedelta(hours=DELAY_H)
    return t.floor("6h")


def steps_needed(init: pd.Timestamp, targets: list[tuple[pd.Timestamp, pd.Timestamp, int]]) -> list[int]:
    """6-hourly steps of run `init` valid inside any target's window [max(dec, start), start + 24 h).
    targets: (D, dec, lst_off)."""
    out = set()
    for D, dec, off in targets:
        start = pd.Timestamp(D) + pd.Timedelta(hours=-int(off))
        lo, hi = max(pd.Timestamp(dec), start), start + pd.Timedelta(hours=24)
        s = 0
        while s <= 360:
            v = init + pd.Timedelta(hours=s)
            if lo <= v < hi:
                out.add(s)
            s += 6
    return sorted(out)


def _nearest(lat0: float, lon0: float, d: float, ni: int, nj: int, lat: float, lon: float) -> int:
    """Flat index of the nearest grid point (regular lat-lon grid scanning west to east, north to south)."""
    j = int(round((lat0 - lat) / d))
    i = int(round(((lon - lon0) % 360.0) / d)) % ni
    return min(max(j, 0), nj - 1) * ni + i


def _get(url: str, headers=None, timeout: float = 60.0, tries: int = 3):
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
    return None


def _fields(base: str) -> list[dict]:
    """The 2 m temperature records of one .index file (one per member)."""
    r = _get(base + ".index")
    if r is None:
        return []
    out = []
    for line in r.text.splitlines():
        try:
            j = json.loads(line)
        except ValueError:
            continue
        if j.get("param") == "2t" and j.get("levtype") == "sfc":
            out.append(j)
    return out


def _decode_points(content: bytes, coords: dict, names: list[str]) -> Optional[np.ndarray]:
    """Values (C) at the stations' nearest grid points of one GRIB2 field."""
    import eccodes
    if content[:4] != b"GRIB":
        return None
    msg = eccodes.codes_new_from_message(content)
    try:
        v = eccodes.codes_get_values(msg)
        ni, nj = eccodes.codes_get(msg, "Ni"), eccodes.codes_get(msg, "Nj")
        lat0 = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")
        d = eccodes.codes_get(msg, "iDirectionIncrementInDegrees")
    finally:
        eccodes.codes_release(msg)
    idx = [_nearest(lat0, lon0, d, ni, nj, coords[n]["lat"], coords[n]["lon"]) for n in names]
    return np.asarray(v, float)[idx] - 273.15


def path_for(root: str, init: pd.Timestamp) -> str:
    return os.path.join(root, "aifs_ens", f"{init:%Y%m%d_%H}z.npz")


def load(root: str, init: pd.Timestamp) -> Optional[dict]:
    p = path_for(root, init)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {"t2m_c": z["t2m_c"], "stations": list(z["stations"]), "steps": [int(s) for s in z["steps"]]}


def fetch_run(init: pd.Timestamp, steps: list[int], root: str, coords: dict, workers: int = 4,
              fetch_bytes: Callable = None) -> dict:
    """Fetch the steps of run `init` not on disk yet (all 51 members each), merge and save. Returns the run's data.
    A step whose 51 members are not all published is left out and tried again on a later call."""
    names = sorted(coords)
    have = load(root, init) or {"t2m_c": np.zeros((len(names), 0, NMEMBERS)), "stations": names, "steps": []}
    todo = [s for s in steps if s not in have["steps"]]
    if not todo:
        return have
    recent = (pd.Timestamp.now("UTC").tz_localize(None) - init) < pd.Timedelta(days=3)
    hosts = HOSTS if recent else HOSTS[::-1]
    d = f"{init:%Y%m%d}"
    hh = f"{init:%H}"
    new_steps, new_vals = [], []
    for s in todo:
        got = None
        for host in hosts:
            recs = []
            for typ in ("cf", "pf"):
                base = f"{host}/{d}/{hh}z/aifs-ens/0p25/enfo/{d}{hh}0000-{s}h-enfo-{typ}"
                recs += [(base + ".grib2", r) for r in _fields(base)]
            if len(recs) != NMEMBERS:
                continue

            def one(item):
                url, r = item
                rng = {"Range": f"bytes={r['_offset']}-{r['_offset'] + r['_length'] - 1}"}
                if fetch_bytes is not None:
                    content = fetch_bytes(url, rng)
                else:
                    g = _get(url, headers=rng)
                    content = g.content if g is not None else b""
                return int(r.get("number") or 0), _decode_points(content, coords, names)
            with ThreadPoolExecutor(workers) as ex:
                res = list(ex.map(one, recs))
            if any(v is None for _, v in res):
                continue
            res.sort(key=lambda t: t[0])                     # control (0) first, then members 1-50
            got = np.stack([v for _, v in res], axis=1)      # station, member
            break
        if got is None:
            log.warning("AIFS ENS %s step %d not available (yet)", init, s)
            continue
        new_steps.append(s)
        new_vals.append(got)
    if new_steps:
        steps_all = have["steps"] + new_steps
        T = np.concatenate([have["t2m_c"], np.stack(new_vals, axis=1)], axis=1)
        order = np.argsort(steps_all)
        have = {"t2m_c": T[:, order, :], "stations": names, "steps": [steps_all[i] for i in order]}
        os.makedirs(os.path.join(root, "aifs_ens"), exist_ok=True)
        tmp = path_for(root, init) + ".tmp.npz"
        np.savez_compressed(tmp, t2m_c=have["t2m_c"].astype(np.float32), stations=np.array(names),
                            steps=np.array(have["steps"]))
        os.replace(tmp, path_for(root, init))
    return have


def features(rows: pd.DataFrame, run: dict, init: pd.Timestamp) -> pd.DataFrame:
    """aifs_mx, aifs_sd, aifs_q10, aifs_q90, aifs_age_h for rows (icao, D, h, lst_off), from one run, as
    research/ml/r4_ens_features.features computes them (members with any missing in-day step make the row NaN)."""
    cols = ["aifs_mx", "aifs_sd", "aifs_q10", "aifs_q90", "aifs_age_h"]
    out = pd.DataFrame(index=rows.index, columns=cols, dtype=float)
    if run is None or not run["steps"]:
        return out
    T = bitround(run["t2m_c"]) * 9 / 5 + 32                          # station, step, member (F)
    valid = np.datetime64(pd.Timestamp(init)) + np.array(run["steps"]) * HOUR
    for i, r in zip(rows.index, rows.itertuples()):
        if r.icao not in run["stations"]:
            continue
        k = run["stations"].index(r.icao)
        D = np.datetime64(pd.Timestamp(r.D))
        dec = D + int(r.h) * HOUR
        start = D + (-int(r.lst_off)) * HOUR
        lo, end = max(dec, start), start + 24 * HOUR
        grid = np.datetime64(pd.Timestamp(init)) + np.arange(0, MAX_STEP + 1, 6) * HOUR   # the archive's steps
        need = int(((grid >= lo) & (grid < end)).sum())
        inday = (valid >= lo) & (valid < end) & (np.array(run["steps"]) <= MAX_STEP)
        X = T[k][inday]                                              # in-day steps, member
        if need == 0 or int(inday.sum()) != need or not np.isfinite(X).all():
            continue                                                 # a step missing: no inputs, as in training
        M = X.max(axis=0)
        out.loc[i, cols] = [float(np.mean(M)), float(np.std(M)), float(np.percentile(M, 10)),
                            float(np.percentile(M, 90)), float((dec - np.datetime64(pd.Timestamp(init))) / HOUR)]
    return out


FEATS = ["aifs_mx_d", "aifs_sd", "aifs_q10_d", "aifs_q90_d", "aifs_age_h"]


def add_model_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """The model's view, as research/ml/r4_ens_features.add_model_columns: center and quantiles relative to the
    Tier-1 anchor, spread and age as they are."""
    for c in ("mx", "q10", "q90"):
        x[f"aifs_{c}_d"] = (df[f"aifs_{c}"] if f"aifs_{c}" in df else np.nan) - x["anchor"]
    x["aifs_sd"] = df["aifs_sd"] if "aifs_sd" in df else np.nan
    x["aifs_age_h"] = df["aifs_age_h"] if "aifs_age_h" in df else np.nan
    return x
