"""HRRR station features at each Tier-1 decision time (shared by research and the live bot).

Source: raw/hrrr/YYYYMMDD_HHz.csv, one file per run: icao, fh, t2m_f, d2m_f at the nearest 3 km grid point.
Research files come from the University of Utah HRRR-zarr archive (research/ml/../mlwx/pull_hrrr.py); live files
come from NOAA's HRRR GRIB2 on AWS (fetch_run below, one byte-range request per forecast hour, decoded with
eccodes). The two agree to 0.004 F at all 24 stations (checked on the 2026-09-30 12Z run).

Timing rule: a run counts at decision time only if init <= decision - LAG_H hours. Runs appear in the archive about
2h00m-2h20m after init (checked Oct 3, 2026: 12Z 14:01, 18Z 20:17, 00Z 02:16), so LAG_H = 3 leaves 40+ minutes of slack.
With synoptic runs this maps h-3 -> 18Z (day before), h3 -> 00Z, h9/h12 -> 06Z, h15/h18 -> 12Z, h21 -> 18Z.

Features (temperatures relative to the Tier-1 anchor so the trees see departures, like every other input):
  hrrr_rest_d     max HRRR 2 m temperature over the rest of the local-standard-time day, minus anchor
  hrrr_age_h      hours since the run's init
  hrrr_now_err    latest observed temperature minus HRRR at the observation time (HRRR running cold -> positive)
  hrrr_sofar_err  observed max so far today minus HRRR's max over the same already-observed hours
  hrrr_peak_lh    local hour of HRRR's rest-of-day maximum
  hrrr_trend      rest-of-day max from this run minus the same from the run six hours earlier
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import pandas as pd

LAG_H = 3
NFH = 48
HRRR_FEATS = ["hrrr_rest_d", "hrrr_age_h", "hrrr_now_err", "hrrr_sofar_err", "hrrr_peak_lh", "hrrr_trend"]
HOUR = np.timedelta64(1, "h")


def load_runs(root: str, since=None, sub: str = "hrrr", col: str = "t2m_f") -> pd.DataFrame:
    """Wide table indexed by (icao, init) with columns 1..48 (forecast hour) of 2 m temperature in F.
    root: a directory holding raw/hrrr/ (research) or hrrr/ (live). since: skip runs initialized before this time.
    sub/col: another field pulled the same way (round 2: sub="hrrr_dpt", col="d2m_f" for 2 m dewpoint)."""
    d0 = f"{root}/raw/{sub}" if os.path.isdir(f"{root}/raw/{sub}") else f"{root}/{sub}"
    parts = []
    for f in sorted(glob.glob(f"{d0}/*_*z.csv")):
        b = os.path.basename(f)[:-4]
        init = pd.Timestamp(b[:8]) + pd.Timedelta(hours=int(b[9:11]))
        if since is not None and init < pd.Timestamp(since):
            continue
        d = pd.read_csv(f, usecols=["icao", "fh", col])
        if d.empty:
            continue
        d["init"] = init
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=range(1, NFH + 1))
    H = pd.concat(parts, ignore_index=True)
    H["init"] = H["init"].astype("datetime64[ns]")
    W = H.set_index(["icao", "init", "fh"])[col].unstack("fh")
    return W.reindex(columns=range(1, NFH + 1)).sort_index()


def _pick_runs(icao: np.ndarray, target: np.ndarray, runs: pd.DataFrame, max_back_h: int) -> np.ndarray:
    """Latest init <= target per station (NaT if none within max_back_h)."""
    t = np.asarray(target, "datetime64[ns]")
    out = np.full(len(icao), np.datetime64("NaT", "ns"))
    ok = ~np.isnat(t)
    q = pd.DataFrame({"i": np.flatnonzero(ok), "icao": icao[ok], "t": t[ok]}).sort_values("t")
    r = runs.assign(init=runs["init"].astype("datetime64[ns]")).sort_values("init")
    m = pd.merge_asof(q, r, left_on="t", right_on="init", by="icao", direction="backward",
                      tolerance=pd.Timedelta(hours=max_back_h))
    out[m["i"].values] = m["init"].values.astype("datetime64[ns]")
    return out


def _window_max(A, init, lo, hi):
    """Max of A over forecast hours with valid time in [lo, hi); also the argmax valid time."""
    fh = np.arange(1, NFH + 1)[None, :]
    flo = ((lo - init) / HOUR)[:, None]
    fhi = ((hi - init) / HOUR)[:, None]
    ok = (fh >= flo) & (fh < fhi) & ~np.isnan(A)
    B = np.where(ok, A, -np.inf)
    j = B.argmax(axis=1)
    mx = B[np.arange(len(B)), j]
    has = ok.any(axis=1)
    mx = np.where(has, mx, np.nan)
    when = init + (j + 1) * HOUR
    return mx, when, has


def _interp(A, init, t):
    """HRRR value at time t (linear between forecast hours); NaN outside the run."""
    x = (t - init) / HOUR
    i0 = np.floor(x)
    ok = np.isfinite(x) & (i0 >= 1) & (i0 + 1 <= NFH)
    i0c = np.clip(np.nan_to_num(i0, nan=1), 1, NFH - 1).astype(int)
    a = A[np.arange(len(A)), i0c - 1]
    b = A[np.arange(len(A)), i0c]
    w = np.nan_to_num(x - i0)
    v = a + (b - a) * w
    exact = ok & (w == 0)
    v = np.where(exact, a, v)
    return np.where(ok, v, np.nan)


def compute(df: pd.DataFrame, W: pd.DataFrame, lag_h: int = LAG_H, synoptic_only: bool = False) -> pd.DataFrame:
    """df needs icao, D (date), h (decision hour UTC relative to D 00Z), lst_off, obs_max, obs_last, obs_age_h.
    Returns raw HRRR columns aligned to df.index: hrrr_rest, hrrr_age_h, hrrr_now_err, hrrr_sofar_err,
    hrrr_peak_lh, hrrr_trend (temperatures in F; hrrr_rest is absolute, model_features makes it anchor-relative)."""
    out = pd.DataFrame(index=df.index, columns=["hrrr_rest", "hrrr_age_h", "hrrr_now_err", "hrrr_sofar_err",
                                                "hrrr_peak_lh", "hrrr_trend"], dtype=float)
    if W.empty:
        return out
    runs = W.index.to_frame(index=False)
    if synoptic_only:
        runs = runs[runs["init"].dt.hour % 6 == 0]
    syn = runs[runs["init"].dt.hour % 6 == 0]
    icao = df["icao"].astype(str).values
    D = pd.to_datetime(df["D"]).values.astype("datetime64[ns]")
    dec = D + df["h"].values.astype(int) * HOUR
    start = D + (-df["lst_off"].values.astype(int)) * HOUR
    end = start + 24 * HOUR
    init = _pick_runs(icao, dec - lag_h * HOUR, runs, max_back_h=12)
    W = W.copy()
    W.index = W.index.set_levels(W.index.levels[1].astype("datetime64[ns]"), level=1)
    idx = pd.MultiIndex.from_arrays([icao, pd.DatetimeIndex(init)])
    A = W.reindex(idx).to_numpy(float)
    lo = np.maximum(dec, start)
    rest, when, has = _window_max(A, init, lo, end)
    out["hrrr_rest"] = rest
    out["hrrr_age_h"] = np.where(has, (dec - init) / HOUR, np.nan)
    loc = (pd.DatetimeIndex(when) + pd.to_timedelta(df["lst_off"].values, unit="h")).hour.values
    out["hrrr_peak_lh"] = np.where(has, loc, np.nan)
    t_obs = dec - (df["obs_age_h"].fillna(0).values * 3600).astype("int64").astype("timedelta64[s]")
    out["hrrr_now_err"] = df["obs_last"].values - _interp(A, init, t_obs)
    so_lo = np.maximum(start, init + HOUR)
    sofar, _, has_s = _window_max(A, init, so_lo, dec + HOUR)  # hours already observed (through the decision hour)
    sofar = np.where(has_s & (dec > start), sofar, np.nan)
    out["hrrr_sofar_err"] = df["obs_max"].values - sofar
    # run-to-run trend: the synoptic run six hours before this one, same window
    prev_init = _pick_runs(icao, init - HOUR, syn, max_back_h=7)
    Ap = W.reindex(pd.MultiIndex.from_arrays([icao, pd.DatetimeIndex(prev_init)])).to_numpy(float)
    prest, _, phas = _window_max(Ap, prev_init, lo, end)
    out["hrrr_trend"] = np.where(phas & has, rest - prest, np.nan)
    return out


TLE_FEATS = ["hrrr_tle_mean_d", "hrrr_tle_sd"]


def compute_tle(df: pd.DataFrame, W: pd.DataFrame, lag_h: int = LAG_H, k: int = 4) -> pd.DataFrame:
    """Round-2 candidate (not in the shipped model): an HRRR time-lagged ensemble. The last k synoptic runs
    (00/06/12/18Z, 48 h) allowed at decision time each give a max over the rest of the local day; their mean and
    spread say how much successive runs agree. A missing run is skipped (the chain looks back up to 13 hours).
    Returns hrrr_tle_mean (F, absolute), hrrr_tle_sd (F, population SD over >= 3 runs) and hrrr_tle_n."""
    out = pd.DataFrame(index=df.index, columns=["hrrr_tle_mean", "hrrr_tle_sd", "hrrr_tle_n"], dtype=float)
    if W.empty:
        return out
    runs = W.index.to_frame(index=False)
    syn = runs[runs["init"].dt.hour % 6 == 0]
    icao = df["icao"].astype(str).values
    D = pd.to_datetime(df["D"]).values.astype("datetime64[ns]")
    dec = D + df["h"].values.astype(int) * HOUR
    start = D + (-df["lst_off"].values.astype(int)) * HOUR
    end = start + 24 * HOUR
    lo = np.maximum(dec, start)
    W = W.copy()
    W.index = W.index.set_levels(W.index.levels[1].astype("datetime64[ns]"), level=1)
    target = dec - lag_h * HOUR
    vals = []
    for _ in range(k):
        init = _pick_runs(icao, target, syn, max_back_h=13)
        A = W.reindex(pd.MultiIndex.from_arrays([icao, pd.DatetimeIndex(init)])).to_numpy(float)
        mx, _, has = _window_max(A, init, lo, end)
        vals.append(np.where(has, mx, np.nan))
        target = init - HOUR  # the next member is strictly older (NaT stops the chain)
    V = np.vstack(vals).T
    n = np.isfinite(V).sum(axis=1)
    with np.errstate(all="ignore"):
        mean = np.nanmean(np.where(n[:, None] > 0, V, 0.0), axis=1)
        sd = np.nanstd(np.where(n[:, None] > 0, V, 0.0), axis=1)
    out["hrrr_tle_mean"] = np.where(n >= 2, mean, np.nan)
    out["hrrr_tle_sd"] = np.where(n >= 3, sd, np.nan)
    out["hrrr_tle_n"] = n.astype(float)
    return out


DPT_FEATS = ["hrrr_tdd_peak", "hrrr_td_now_err"]


def compute_dpt(df: pd.DataFrame, W: pd.DataFrame, Wd: pd.DataFrame, lag_h: int = LAG_H) -> pd.DataFrame:
    """Round-2 candidate (not in the shipped model): HRRR 2 m dewpoint from the same run compute() uses.
    hrrr_tdd_peak    HRRR's dewpoint depression (T - Td, F) at the hour of its rest-of-day maximum: dry air heats
                     faster, moist air caps the high
    hrrr_td_now_err  latest observed dewpoint minus HRRR's dewpoint at that time (HRRR too moist -> negative)"""
    out = pd.DataFrame(index=df.index, columns=["hrrr_tdd_peak", "hrrr_td_now_err"], dtype=float)
    if W.empty or Wd.empty:
        return out
    runs = W.index.to_frame(index=False)
    icao = df["icao"].astype(str).values
    D = pd.to_datetime(df["D"]).values.astype("datetime64[ns]")
    dec = D + df["h"].values.astype(int) * HOUR
    start = D + (-df["lst_off"].values.astype(int)) * HOUR
    end = start + 24 * HOUR
    init = _pick_runs(icao, dec - lag_h * HOUR, runs, max_back_h=12)
    idx = pd.MultiIndex.from_arrays([icao, pd.DatetimeIndex(init)])
    W = W.copy()
    W.index = W.index.set_levels(W.index.levels[1].astype("datetime64[ns]"), level=1)
    Wd = Wd.copy()
    Wd.index = Wd.index.set_levels(Wd.index.levels[1].astype("datetime64[ns]"), level=1)
    A = W.reindex(idx).to_numpy(float)
    Ad = Wd.reindex(idx).to_numpy(float)
    rest, when, has = _window_max(A, init, np.maximum(dec, start), end)
    j = np.clip(np.nan_to_num((when - init) / HOUR, nan=1).astype(int), 1, NFH) - 1
    td_peak = Ad[np.arange(len(Ad)), j]
    out["hrrr_tdd_peak"] = np.where(has, rest - td_peak, np.nan)
    if "obs_dwp" in df:
        t_obs = dec - (df["obs_age_h"].fillna(0).values * 3600).astype("int64").astype("timedelta64[s]")
        out["hrrr_td_now_err"] = df["obs_dwp"].values - _interp(Ad, init, t_obs)
    return out


TCC_FEATS = ["hrrr_tcc_now", "hrrr_tcc_err", "hrrr_tcc_rest"]
SKY_PCT = 25.0  # METAR lowest-layer category 0-4 (CLR..OVC) -> rough percent cover


def compute_tcc(df: pd.DataFrame, W: pd.DataFrame, Wc: pd.DataFrame, lag_h: int = LAG_H) -> pd.DataFrame:
    """Round-2 candidate (not in the shipped model): HRRR total cloud cover (%) from the same run compute() uses.
    hrrr_tcc_now   HRRR cloud cover at the latest observation time
    hrrr_tcc_err   observed sky (lowest METAR layer, 0-4 -> 0-100%) minus HRRR's cover then (clouds HRRR missed > 0)
    hrrr_tcc_rest  HRRR's mean cloud cover from the decision time to its forecast peak hour (the heating still left)"""
    out = pd.DataFrame(index=df.index, columns=["hrrr_tcc_now", "hrrr_tcc_err", "hrrr_tcc_rest"], dtype=float)
    if W.empty or Wc.empty:
        return out
    runs = W.index.to_frame(index=False)
    icao = df["icao"].astype(str).values
    D = pd.to_datetime(df["D"]).values.astype("datetime64[ns]")
    dec = D + df["h"].values.astype(int) * HOUR
    start = D + (-df["lst_off"].values.astype(int)) * HOUR
    end = start + 24 * HOUR
    init = _pick_runs(icao, dec - lag_h * HOUR, runs, max_back_h=12)
    idx = pd.MultiIndex.from_arrays([icao, pd.DatetimeIndex(init)])
    W = W.copy()
    W.index = W.index.set_levels(W.index.levels[1].astype("datetime64[ns]"), level=1)
    Wc = Wc.copy()
    Wc.index = Wc.index.set_levels(Wc.index.levels[1].astype("datetime64[ns]"), level=1)
    A = W.reindex(idx).to_numpy(float)
    Ac = Wc.reindex(idx).to_numpy(float)
    lo = np.maximum(dec, start)
    rest, when, has = _window_max(A, init, lo, end)
    fh = np.arange(1, NFH + 1)[None, :]
    flo = ((lo - init) / HOUR)[:, None]
    fpk = ((when - init) / HOUR)[:, None]
    m = (fh >= flo) & (fh <= fpk) & np.isfinite(Ac)
    with np.errstate(all="ignore"):
        out["hrrr_tcc_rest"] = np.where(has & m.any(axis=1), np.where(m, Ac, 0.0).sum(axis=1) / m.sum(axis=1), np.nan)
    t_obs = dec - (df["obs_age_h"].fillna(0).values * 3600).astype("int64").astype("timedelta64[s]")
    now = _interp(Ac, init, t_obs)
    out["hrrr_tcc_now"] = now
    if "obs_sky" in df:
        out["hrrr_tcc_err"] = df["obs_sky"].values * SKY_PCT - now
    return out


def add_tcc_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    for c in TCC_FEATS:
        x[c] = df[c] if c in df else np.nan
    return x


def add_dpt_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    for c in DPT_FEATS:
        x[c] = df[c] if c in df else np.nan
    return x


def add_tle_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Anchor-relative TLE features for the design matrix (round-2 variant only)."""
    x["hrrr_tle_mean_d"] = (df["hrrr_tle_mean"] if "hrrr_tle_mean" in df else np.nan) - x["anchor"]
    x["hrrr_tle_sd"] = df["hrrr_tle_sd"] if "hrrr_tle_sd" in df else np.nan
    return x


HRRR_LAG_FEATS = ["e_hrrr_m14", "e_prev_hrrr"]


def add_lag_errors(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """HRRR's recent misses at the station, using only reports public at decision time (same rule as
    features.add_lag_errors: D-1 counts from 12Z on D). e_hrrr = CLI high - HRRR max for the day as forecast at 03Z
    (the whole local day is still ahead at 03Z for every station)."""
    src = df if history is None else pd.concat([history, df], ignore_index=True)
    base = src[src["h"] == 3][["icao", "D", "y", "hrrr_rest"]].drop_duplicates(["icao", "D"], keep="first").copy()
    base["e_hrrr"] = base["y"] - base["hrrr_rest"]
    parts = []
    for icao, g in base.groupby("icao"):
        g = g.set_index("D").sort_index()
        full = pd.date_range(g.index.min(), max(g.index.max(), df["D"].max()), freq="D")
        g = g.reindex(full)
        for lag0 in (1, 2):
            f = pd.DataFrame(index=g.index)
            f["e_hrrr_m14"] = g["e_hrrr"].shift(lag0).rolling(14, min_periods=5).mean()
            f["icao"], f["lag0"] = icao, lag0
            parts.append(f.rename_axis("D").reset_index())
    t = pd.concat(parts, ignore_index=True)
    out = df.copy()
    out["lag0"] = np.where(out["h"] >= 12, 1, 2)
    out = out.drop(columns=[c for c in ("e_hrrr_m14", "hrrr_rest_prev") if c in out])
    out = out.merge(t, on=["icao", "D", "lag0"], how="left")
    prev = base[["icao", "D", "hrrr_rest"]].rename(columns={"hrrr_rest": "hrrr_rest_prev"})
    prev["D"] = prev["D"] + pd.Timedelta(days=1)
    out = out.merge(prev, on=["icao", "D"], how="left")
    out["e_prev_hrrr"] = out["obs_prev_max"] - out["hrrr_rest_prev"] if "obs_prev_max" in out else np.nan
    return out


def add_model_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Anchor-relative HRRR features for the Tier-1 design matrix (x from features.model_features)."""
    a = x["anchor"]
    for c in ["hrrr_rest", "hrrr_age_h", "hrrr_now_err", "hrrr_sofar_err", "hrrr_peak_lh", "hrrr_trend"]:
        if c not in df:
            df = df.assign(**{c: np.nan})
    x["hrrr_rest_d"] = df["hrrr_rest"] - a
    x["hrrr_age_h"] = df["hrrr_age_h"]
    x["hrrr_now_err"] = df["hrrr_now_err"]
    x["hrrr_sofar_err"] = df["hrrr_sofar_err"]
    x["hrrr_peak_lh"] = df["hrrr_peak_lh"]
    x["hrrr_trend"] = df["hrrr_trend"]
    for c in HRRR_LAG_FEATS:
        x[c] = df[c] if c in df else np.nan
    return x



# ------------------------------------------------------------------------------------------------ live fetching
NOAA = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
NX, NY = 1799, 1059
X0, Y0, DX = -2697520.14252193, -1587306.15255666, 3000.0  # HRRR Lambert grid, south-west corner


def lcc_xy(lon, lat):
    """HRRR's Lambert conformal projection (sphere R = 6371229 m, tangent at 38.5 N, centred on 97.5 W).
    Matches pyproj to nanometres, so the live bot needs no projection library."""
    R, phi1, lam0 = 6371229.0, np.radians(38.5), np.radians(-97.5)
    n = np.sin(phi1)
    Fc = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n
    rho = R * Fc / np.tan(np.pi / 4 + np.radians(lat) / 2) ** n
    rho0 = R * Fc / np.tan(np.pi / 4 + phi1 / 2) ** n
    t = n * (np.radians(lon) - lam0)
    return rho * np.sin(t), rho0 - rho * np.cos(t)


def station_index(coords: dict) -> dict:
    """{icao: flat index into the GRIB values array (j scans north, i fastest)}."""
    out = {}
    for icao, c in coords.items():
        x, y = lcc_xy(c["lon"], c["lat"])
        ix, iy = int(round((x - X0) / DX)), int(round((y - Y0) / DX))
        out[icao] = iy * NX + ix
    return out


def _get_retry(url, timeout, headers=None, tries=3):
    """GET with a short back-off on throttling (S3 answers 503 SlowDown) or network errors; 404 is final."""
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code in (200, 206, 404):
                return r
        except requests.RequestException:
            pass
        time.sleep(1.5 * (k + 1))
    return None


def _fetch_hour(url: str, timeout: float):
    """2 m temperature field of one HRRR GRIB2 file via its .idx byte range; None if not published (yet)."""
    import eccodes  # the bot already uses eccodes for ECMWF
    try:
        r = _get_retry(url + ".idx", timeout)
        if r is None or r.status_code != 200:
            return None
        lines = r.text.splitlines()
        rng = None
        for i, line in enumerate(lines):
            if ":TMP:2 m above ground:" in line:
                start = int(line.split(":")[1])
                end = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
                rng = f"bytes={start}-{end}"
                break
        if rng is None:
            return None
        g = _get_retry(url, timeout, headers={"Range": rng})
        if g is None or g.status_code not in (200, 206) or g.content[:4] != b"GRIB":
            return None
        msg = eccodes.codes_new_from_message(g.content)
        try:
            return eccodes.codes_get_values(msg)
        finally:
            eccodes.codes_release(msg)
    except Exception:
        return None


def fetch_run(day, hh: int, fhs, root: str, coords: dict, timeout: float = 60.0, workers: int = 8) -> int:
    """Fetch 2 m temperature for the given forecast hours of one HRRR run from NOAA's AWS bucket (about 1.2 MB per
    hour) and merge them into root/hrrr/YYYYMMDD_HHz.csv (the research layout). Returns the number of new hours;
    hours not published yet are skipped and picked up on a later call."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(os.path.join(root, "hrrr"), exist_ok=True)
    path = os.path.join(root, "hrrr", f"{day:%Y%m%d}_{hh:02d}z.csv")
    have = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["icao", "fh", "t2m_f", "d2m_f"])
    done = set(have["fh"].astype(int)) if len(have) else set()
    todo = sorted(set(int(f) for f in fhs) - done)
    if not todo:
        return 0
    idx = station_index(coords)
    base = f"{NOAA}/hrrr.{day:%Y%m%d}/conus/hrrr.t{hh:02d}z.wrfsfcf"
    with ThreadPoolExecutor(workers) as ex:
        fields = list(ex.map(lambda fh: _fetch_hour(f"{base}{fh:02d}.grib2", timeout), todo))
    rows = []
    for fh, v in zip(todo, fields):
        if v is None:
            continue
        for icao, k in idx.items():
            rows.append([icao, fh, round((float(v[k]) - 273.15) * 9 / 5 + 32, 2), np.nan])
    if rows:
        new = pd.DataFrame(rows, columns=["icao", "fh", "t2m_f", "d2m_f"])
        out = pd.concat([have, new], ignore_index=True) if len(have) else new
        out = out.drop_duplicates(["icao", "fh"], keep="last").sort_values(["icao", "fh"])
        out.to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
    return len(rows) // max(len(idx), 1)


def runs_needed(now, lst_offsets, lag_h: int = LAG_H, history_days: int = 16):
    """Which (run init, forecast hours) the live bot needs at `now` (UTC, tz-aware or naive UTC):
    the latest run allowed by the lag rule (synoptic, or the 09Z/15Z hourly runs), the synoptic run before it (trend),
    and the 00Z runs of the past `history_days` days (03Z decisions feed the HRRR error features)."""
    import datetime as dt
    now = pd.Timestamp(now).tz_localize(None) if pd.Timestamp(now).tzinfo is None else \
        pd.Timestamp(now).tz_convert("UTC").tz_localize(None)
    t = (now - pd.Timedelta(hours=lag_h)).floor("h")
    cands = [t - pd.Timedelta(hours=k) for k in range(0, 13)]
    latest = next(c for c in cands if c.hour in (0, 6, 9, 12, 15, 18))
    prev_syn = next(latest - pd.Timedelta(hours=k) for k in range(1, 8) if (latest - pd.Timedelta(hours=k)).hour % 6 == 0)
    last_end = max(-o for o in lst_offsets) + 24  # hours after D 00Z when the last local day ends
    out = {}
    for init in (latest, prev_syn):
        n_fh = 48 if init.hour % 6 == 0 else 18
        D = init.normalize() + (pd.Timedelta(days=1) if init.hour >= 18 else pd.Timedelta(0))
        end_fh = int(((D + pd.Timedelta(hours=last_end)) - init) / pd.Timedelta(hours=1))
        out[init] = list(range(1, min(n_fh, max(end_fh, 1)) + 1))
    for k in range(1, history_days + 1):
        init = (now.normalize() - pd.Timedelta(days=k))
        end_fh = min(48, last_end)
        out.setdefault(init, list(range(1, end_fh + 1)))
    return out
