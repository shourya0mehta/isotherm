"""Tier-1 v2 feature pipeline, shared by training (research/ml) and the live bot, so there is no train/serve skew.

Everything reads a "raw" directory with the training layout:
  mos/{MODEL}_{ICAO}_*.csv   IEM MOS archive CSVs (NBS, NBE, GFS, NAM, MEX, LAV)
  obs/asos_*.csv             IEM ASOS/METAR CSVs (all stations)
  cli/{ICAO}_*.json          IEM CLI JSON (labels, normals, lagged errors)
  ecmwf/*.csv                IFS 2 m temperature at stations (run, step, valid, icao, t2m_bil, ...)
A product counts at decision time tau only if runtime + publication lag <= tau; obs only if valid <= tau - 5 min.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

_ST = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "stations.json")))
HIGH = {v["icao"]: v for v in _ST.values() if v["kind"] == "high"}
DEC_H = [-3, 3, 9, 12, 15, 18, 21]
LAG_MIN = {"NBS": 75, "NBE": 90, "GFS": 240, "NAM": 210, "MEX": 270, "LAV": 40}
EC_LAG = {0: 455, 6: 390, 12: 455, 18: 390}
SKY = {"CLR": 0, "SKC": 0, "NSC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4}
K = 12
NCLS = 2 * K + 1
V5_DATE = pd.Timestamp("2026-05-01")
# NWS SCN26-47: the NAM MOS (MET) bulletins end with the Nov 3, 2026 12Z cycle. NAM is never missing in the training
# data, so from that cycle on every row gets the same pre-registered fallback (docs/HOLDOUT_DEC1.md, amendment A4):
# "gfs" = GFS MOS in NAM's place (walk-forward Jun-Sep 2026: MAE +0.016 F), "nan" = NAM missing (+0.027 F);
# research/ml/nam_retire.py, docs/results/nam_retirement.txt.
NAM_END = pd.Timestamp("2026-11-03 12:00")      # UTC, naive like D
NAM_FALLBACK = os.environ.get("WXBOT_NAM_FALLBACK", "gfs")
NAM_GFS_AGE_GAP_H = 4.4                          # typical NAM minus GFS MOS age at decision time (Jun-Sep 2026)


def nam_fallback(df: pd.DataFrame, mode: str = None) -> pd.DataFrame:
    """Rows decided at or after NAM_END get the NAM fallback (a copy is returned only when something changes)."""
    mode = mode or NAM_FALLBACK
    if "D" not in df or "h" not in df or "nam_n_x" not in df:
        return df
    tau = pd.to_datetime(df["D"]).dt.tz_localize(None) + pd.to_timedelta(df["h"], unit="h")
    late = (tau >= NAM_END).values
    if not late.any():
        return df
    df = df.copy()
    if mode == "gfs":
        df.loc[late, "nam_n_x"] = df.loc[late, "gfs_n_x"]
        df.loc[late, "nam_age_h"] = df.loc[late, "gfs_age_h"] + NAM_GFS_AGE_GAP_H
    else:
        df.loc[late, "nam_n_x"] = np.nan
        df.loc[late, "nam_age_h"] = np.nan
    return df


def c2f(c):
    return c * 1.8 + 32.0


def localday(ts: pd.Series, off_h: int) -> pd.Series:
    return (ts + pd.Timedelta(hours=off_h)).dt.tz_localize(None).dt.normalize()


def load_mos(raw: str, model: str, icao: str, cols: list[str]) -> pd.DataFrame:
    fs = [f for f in sorted(glob.glob(f"{raw}/mos/{model}_{icao}_*.csv")) if os.path.getsize(f) > 200]
    if not fs:
        e = pd.DataFrame({"runtime": pd.to_datetime([], utc=True), "ftime": pd.to_datetime([], utc=True)})
        for c in cols:
            e[c] = pd.Series(dtype=float)
        return e
    keep = set(["runtime", "ftime"] + cols)
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in keep) for f in fs], ignore_index=True)
    for c in cols:
        if c not in df:
            df[c] = np.nan
    df["runtime"] = pd.to_datetime(df["runtime"], utc=True, format="ISO8601")
    df["ftime"] = pd.to_datetime(df["ftime"], utc=True, format="ISO8601")
    return df.drop_duplicates(["runtime", "ftime"])


def _align(m: pd.DataFrame, f: pd.DataFrame):
    """merge_asof needs identical key dtypes; live frames mix second/microsecond/nanosecond resolutions."""
    m = m.copy()
    f = f.copy()
    m["D"] = pd.to_datetime(m["D"]).astype("datetime64[ns]")
    f["D"] = pd.to_datetime(f["D"]).astype("datetime64[ns]")
    m["avail"] = pd.to_datetime(m["avail"], utc=True).astype("datetime64[ns, UTC]")
    f["tau"] = pd.to_datetime(f["tau"], utc=True).astype("datetime64[ns, UTC]")
    return m, f


def pick(maxrows: pd.DataFrame, frame: pd.DataFrame, lag_min: int, prefix: str, vals: list[str]) -> pd.DataFrame:
    """For each (D, tau) in frame choose the latest run with runtime + lag <= tau that has a row for D."""
    m = maxrows.copy()
    m["avail"] = m["runtime"] + pd.Timedelta(minutes=lag_min)
    m, f = _align(m, frame)
    m = m.sort_values("avail")
    f = f.sort_values("tau")
    j = pd.merge_asof(f, m[["D", "avail", "runtime"] + vals], left_on="tau", right_on="avail", by="D",
                      direction="backward")
    j = j.rename(columns={**{v: f"{prefix}_{v}" for v in vals}, "runtime": f"{prefix}_run"}).drop(columns=["avail"])
    j[f"{prefix}_age_h"] = (j["tau"] - j[f"{prefix}_run"]).dt.total_seconds() / 3600
    return j


def pick_var_lag(table: pd.DataFrame, frame: pd.DataFrame, prefix: str, vals: list[str]) -> pd.DataFrame:
    m = table.copy()
    m["avail"] = m["runtime"] + pd.to_timedelta(m["lag"], unit="m")
    m, f = _align(m, frame)
    m = m.sort_values("avail")
    f = f.sort_values("tau")
    j = pd.merge_asof(f, m[["D", "avail", "runtime"] + vals], left_on="tau", right_on="avail", by="D",
                      direction="backward")
    j = j.rename(columns={"runtime": f"{prefix}_run"}).drop(columns=["avail"])
    j[f"{prefix}_age_h"] = (j["tau"] - j[f"{prefix}_run"]).dt.total_seconds() / 3600
    return j


class RawSource:
    """Caches the parsed obs and ECMWF tables of one raw directory."""

    def __init__(self, raw: str):
        self.raw = raw
        self._obs = None
        self._ec = None

    def obs(self) -> pd.DataFrame:
        if self._obs is not None:
            return self._obs
        parts = []
        for f in sorted(glob.glob(f"{self.raw}/obs/asos_*.csv")):
            parts.append(pd.read_csv(f, usecols=["station", "valid", "tmpf", "dwpf", "drct", "sknt", "skyc1", "mslp",
                                                     "metar"]))
        d = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["station", "valid", "tmpf", "dwpf", "drct", "sknt", "skyc1", "mslp", "metar"])
        d["valid"] = pd.to_datetime(d["valid"], utc=True)
        for c in ["tmpf", "dwpf", "drct", "sknt", "mslp"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        m = d["metar"].fillna("").astype(str)
        tg = m.str.extract(r" T([01])(\d{3})([01])(\d{3})")
        tc = tg[1].astype(float) / 10.0 * np.where(tg[0] == "1", -1, 1)
        d["t_prec"] = np.where(tc.notna(), c2f(tc), d["tmpf"])
        mx = m.str.extract(r"RMK.* 1([01])(\d{3})(?: |$)")
        mxc = mx[1].astype(float) / 10.0 * np.where(mx[0] == "1", -1, 1)
        hr, mi = d["valid"].dt.hour, d["valid"].dt.minute
        d["max6"] = np.where(mxc.notna() & hr.isin([0, 6, 12, 18]) & (mi >= 45)
                             | mxc.notna() & hr.isin([23, 5, 11, 17]) & (mi >= 45), c2f(mxc), np.nan)
        d["sky"] = d["skyc1"].map(SKY)
        d = d.drop(columns=["metar"]).drop_duplicates(["station", "valid"]).sort_values(["station", "valid"])
        self._obs = d.reset_index(drop=True)
        return self._obs

    def ecmwf_x(self) -> pd.DataFrame:
        if getattr(self, "_ecx", None) is not None:
            return self._ecx
        fs = sorted(glob.glob(f"{self.raw}/ecmwf_x/*.csv"))
        ex = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame(
            columns=["run", "step", "valid", "icao", "param", "value"])
        ex["run"] = pd.to_datetime(ex["run"], utc=True)
        ex["valid"] = pd.to_datetime(ex["valid"], utc=True)
        ex["param"] = ex["param"].replace({"mucape": "cape"})
        self._ecx = ex.drop_duplicates(["run", "step", "icao", "param"])
        return self._ecx

    def ecmwf(self) -> pd.DataFrame:
        if self._ec is not None:
            return self._ec
        fs = sorted(glob.glob(f"{self.raw}/ecmwf/*.csv"))
        ec = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame(
            columns=["run", "step", "valid", "icao", "t2m_bil"])
        ec["run"] = pd.to_datetime(ec["run"], utc=True)
        ec["valid"] = pd.to_datetime(ec["valid"], utc=True)
        self._ec = ec.drop_duplicates(["run", "step", "icao"])
        return self._ec


def obs_features(src: RawSource, icao: str, frame: pd.DataFrame) -> pd.DataFrame:
    off = int(HIGH[icao]["lst_offset_h"])
    o = src.obs()
    o = o[o["station"] == icao[1:]].copy()
    o = o[o["t_prec"].notna()].reset_index(drop=True)
    res = pd.DataFrame(index=frame.index)
    cols = ["obs_max", "obs_last", "obs_age_h", "obs_dwp", "obs_sknt", "obs_u", "obs_v", "obs_sky", "obs_dt3", "obs_dp3",
            "obs_prev_max"]
    if o.empty:
        for c in cols:
            res[c] = np.nan
        return res
    o["lday"] = localday(o["valid"], off)
    o["cummax"] = o.groupby("lday")["t_prec"].cummax()
    rep = o["valid"].dt.round("h")
    win_start_ok = localday(rep - pd.Timedelta(hours=6), off) == o["lday"]
    o["m6"] = np.where(win_start_ok, o["max6"], np.nan)
    # cummax skips NaN but leaves NaN at those rows: fill with -inf first so the 6-h max carries forward in the day
    o["cummax6"] = o.assign(m6=o["m6"].fillna(-np.inf)).groupby("lday")["m6"].cummax().replace(-np.inf, np.nan)
    t = o["valid"].values.astype("datetime64[ns]").astype(np.int64)
    q = (frame["tau"] - pd.Timedelta(minutes=5)).values.astype("datetime64[ns]").astype(np.int64)
    i = np.searchsorted(t, q, side="right") - 1
    ok = i >= 0
    ii = np.where(ok, i, 0)
    same = ok & (o["lday"].values[ii] == frame["D"].values)
    res["obs_max"] = np.where(same, np.fmax(o["cummax"].values[ii], np.nan_to_num(o["cummax6"].values[ii], nan=-999)),
                              np.nan)
    res["obs_max"] = res["obs_max"].where(res["obs_max"] > -900)
    res["obs_last"] = np.where(ok, o["t_prec"].values[ii], np.nan)
    res["obs_age_h"] = np.where(ok, (q - t[ii]) / 3.6e12, np.nan)
    res["obs_dwp"] = np.where(ok, o["dwpf"].values[ii], np.nan)
    res["obs_sknt"] = np.where(ok, o["sknt"].values[ii], np.nan)
    dr = np.where(ok, o["drct"].values[ii], np.nan)
    res["obs_u"] = -res["obs_sknt"] * np.sin(np.deg2rad(dr))
    res["obs_v"] = -res["obs_sknt"] * np.cos(np.deg2rad(dr))
    res["obs_sky"] = np.where(ok, o["sky"].values[ii], np.nan)
    q3 = q - 3 * 3600 * 10**9
    j = np.searchsorted(t, q3, side="right") - 1
    jj = np.where(j >= 0, j, 0)
    res["obs_dt3"] = np.where(ok & (j >= 0), o["t_prec"].values[ii] - o["t_prec"].values[jj], np.nan)
    res["obs_dp3"] = np.where(ok & (j >= 0), o["mslp"].values[ii] - o["mslp"].values[jj], np.nan)
    stale = res["obs_age_h"] > 3
    for c in ["obs_last", "obs_dwp", "obs_sknt", "obs_u", "obs_v", "obs_sky", "obs_dt3", "obs_dp3"]:
        res.loc[stale, c] = np.nan
    # yesterday's observed max (local day D-1), as far as it is known at tau: full day once D has started,
    # the running max while D-1 is still in progress (e.g. the 21Z-the-day-before decision)
    full = np.fmax(o["cummax"].values, np.nan_to_num(o["cummax6"].values, nan=-999))
    dmax = pd.Series(full, index=o.index).groupby(o["lday"]).max()
    prev = (frame["D"] - pd.Timedelta(days=1)).values
    in_d = ok & same
    in_prev = ok & (o["lday"].values[ii] == prev)
    res["obs_prev_max"] = np.where(in_d, dmax.reindex(prev).values,
                                   np.where(in_prev, full[ii], np.nan))
    res["obs_prev_max"] = res["obs_prev_max"].where(res["obs_prev_max"] > -900)
    return res


def ecmwf_table(src: RawSource, icao: str) -> pd.DataFrame:
    ec = src.ecmwf()
    e = ec[ec["icao"] == icao].copy()
    if e.empty:
        return pd.DataFrame({"D": pd.to_datetime([]), "runtime": pd.to_datetime([], utc=True), "ec_t18": [],
                             "ec_t21": [], "ec_t00": [], "ec_max": [], "lag": []})
    e["D"] = (e["valid"] - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
    e["slot"] = e["valid"].dt.hour.map({18: "ec_t18", 21: "ec_t21", 0: "ec_t00"})
    e["tf"] = c2f(e["t2m_bil"])
    w = e.pivot_table(index=["run", "D"], columns="slot", values="tf", aggfunc="first").reset_index()
    for c in ["ec_t18", "ec_t21", "ec_t00"]:
        if c not in w:
            w[c] = np.nan
    w["ec_max"] = w[["ec_t18", "ec_t21", "ec_t00"]].max(axis=1)
    w = w[w[["ec_t18", "ec_t21", "ec_t00"]].notna().sum(axis=1) >= 2]
    w = w.rename(columns={"run": "runtime"})
    w["lag"] = w["runtime"].dt.hour.map(EC_LAG)
    return w


def ecmwf_x_table(src: RawSource, icao: str) -> pd.DataFrame:
    """Per IFS run and day D: max of mx2t3 over the 18Z/21Z/00Z steps (F), and at 21Z: total cloud, CAPE, dewpoint (F);
    afternoon precipitation = tp(00Z+1) - tp(18Z) in mm. Field availability grew over time (missing -> NaN)."""
    ex = src.ecmwf_x()
    e = ex[ex["icao"] == icao].copy()
    cols = ["ecx_mx", "ecx_tcc", "ecx_cape", "ecx_2d", "ecx_tp"]
    if e.empty:
        out = pd.DataFrame({"D": pd.to_datetime([]), "runtime": pd.to_datetime([], utc=True), "lag": []})
        for c in cols:
            out[c] = []
        return out
    e["D"] = (e["valid"] - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()
    e["hr"] = e["valid"].dt.hour
    mx = e[e["param"] == "mx2t3"].groupby(["run", "D"])["value"].max().rename("mx")
    p21 = e[e["hr"] == 21].pivot_table(index=["run", "D"], columns="param", values="value", aggfunc="first")
    tp = e[e["param"] == "tp"].pivot_table(index=["run", "D"], columns="hr", values="value", aggfunc="first")
    w = pd.concat([mx, p21.reindex(columns=["tcc", "cape", "2d"]),
                   tp.reindex(columns=[0, 18]).rename(columns={0: "tp0", 18: "tp18"})], axis=1).reset_index()
    w = w.rename(columns={"run": "runtime"})
    w["ecx_mx"] = c2f(w["mx"] - 273.15)
    w["ecx_tcc"] = w["tcc"]
    w["ecx_cape"] = w["cape"]
    w["ecx_2d"] = c2f(w["2d"] - 273.15)
    w["ecx_tp"] = (w["tp0"] - w["tp18"]) * 1000.0
    w["lag"] = w["runtime"].dt.hour.map(EC_LAG)
    return w[["runtime", "D", "lag"] + cols]


def _cli_final(r: dict, off_h: int) -> bool:
    """True if the CLI product was issued after the local standard day ended (a final report). IEM sometimes keeps a
    preliminary same-day report (e.g. a 6 AM "so far today" CLI), whose high is not the day's high."""
    prod = str(r.get("product") or "")
    try:
        issued = pd.Timestamp(dt_from_prod(prod))
    except Exception:
        return True
    day_end = pd.Timestamp(r["valid"]) + pd.Timedelta(days=1) - pd.Timedelta(hours=off_h)  # LST midnight in UTC
    return issued >= day_end


def dt_from_prod(prod: str):
    return pd.to_datetime(prod[:12], format="%Y%m%d%H%M")


def cli_table(raw: str, icao: str) -> pd.DataFrame:
    off = int(HIGH[icao]["lst_offset_h"]) if icao in HIGH else 0
    cli = []
    for f in glob.glob(f"{raw}/cli/{icao}_*.json"):
        for r in json.load(open(f)):
            hi = r.get("high")
            if hi in (None, "M"):
                continue
            y = float(hi) if _cli_final(r, off) else np.nan
            cli.append({"D": pd.Timestamp(r["valid"]), "y": y,
                        "normal": float(r["high_normal"]) if r.get("high_normal") not in (None, "M") else np.nan})
    if not cli:
        return pd.DataFrame(columns=["y", "normal"], index=pd.DatetimeIndex([], name="D"))
    return pd.DataFrame(cli).drop_duplicates("D").set_index("D").sort_index()


def station_rows(src: RawSource, icao: str, frame: pd.DataFrame | None = None, start="2020-06-05") -> pd.DataFrame:
    """frame: optional DataFrame with columns D (naive date), h; default = every CLI day x DEC_H."""
    raw = src.raw
    off = int(HIGH[icao]["lst_offset_h"])
    cli = cli_table(raw, icao)
    if frame is None:
        days = cli.index[(cli.index >= start)]
        frame = pd.DataFrame([(d, h) for d in days for h in DEC_H], columns=["D", "h"])
    frame = frame[["D", "h"]].copy()
    frame["D"] = pd.to_datetime(frame["D"]).astype("datetime64[ns]")
    frame["tau"] = frame["D"].dt.tz_localize("UTC") + pd.to_timedelta(frame["h"], unit="h")
    nbs = load_mos(raw, "NBS", icao, ["txn", "xnd", "tmp", "dpt", "sky", "wsp", "wdr", "p06", "sol", "tsd", "gst"])
    mx = nbs[nbs["txn"].notna() & (nbs["ftime"].dt.hour == 0)].copy()
    mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
    out = pick(mx, frame, LAG_MIN["NBS"], "nbs", ["txn", "xnd"])
    h = nbs.copy()
    h["D"] = localday(h["ftime"], off)
    h["u"] = -h["wsp"] * np.sin(np.deg2rad(h["wdr"]))
    h["v"] = -h["wsp"] * np.cos(np.deg2rad(h["wdr"]))
    agg = h.groupby(["runtime", "D"]).agg(tmax=("tmp", "max"), n=("tmp", "count"), dpt=("dpt", "mean"),
                                          sky=("sky", "mean"), wsp=("wsp", "max"), u=("u", "mean"), v=("v", "mean"),
                                          p06=("p06", "max"), sol=("sol", "max"), tsd=("tsd", "mean"),
                                          gst=("gst", "max")).reset_index()
    out = pick(agg, out, LAG_MIN["NBS"], "nbh", ["tmax", "n", "dpt", "sky", "wsp", "u", "v", "p06", "sol", "tsd",
                                                 "gst"])
    nbe = load_mos(raw, "NBE", icao, ["txn", "xnd"])
    mx = nbe[nbe["txn"].notna() & (nbe["ftime"].dt.hour == 0)].copy()
    mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
    out = pick(mx, out, LAG_MIN["NBE"], "nbe", ["txn", "xnd"])
    for model, pre in (("GFS", "gfs"), ("NAM", "nam"), ("MEX", "mex")):
        g = load_mos(raw, model, icao, ["n_x"])
        mx = g[g["n_x"].notna() & (g["ftime"].dt.hour == 0)].copy()
        mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
        out = pick(mx, out, LAG_MIN[model], pre, ["n_x"])
    lav = load_mos(raw, "LAV", icao, ["tmp", "dpt", "wsp"])
    lav = lav[lav["runtime"].dt.hour.isin([0, 6, 12, 18])]  # the archive holds these cycles; live matches it
    lav["D"] = localday(lav["ftime"], off)
    la = lav.groupby(["runtime", "D"]).agg(tmax=("tmp", "max"), n=("tmp", "count"), dpt=("dpt", "mean"),
                                           wsp=("wsp", "max")).reset_index()
    out = pick(la, out, LAG_MIN["LAV"], "lav", ["tmax", "n", "dpt", "wsp"])
    ec = ecmwf_table(src, icao)
    out = pick_var_lag(ec, out, "ec", ["ec_t18", "ec_t21", "ec_t00", "ec_max"])
    if glob.glob(f"{raw}/ecmwf_x/*.csv"):
        ecx = ecmwf_x_table(src, icao)
        out = pick_var_lag(ecx, out, "ecx", ["ecx_mx", "ecx_tcc", "ecx_cape", "ecx_2d", "ecx_tp"])
    out = out.sort_values(["D", "h"]).reset_index(drop=True)
    ob = obs_features(src, icao, out)
    out = pd.concat([out, ob], axis=1)
    out = out.merge(cli, left_on="D", right_index=True, how="left")
    live = ~out["D"].isin(cli.index)
    if live.any() and len(cli):
        # live rows have no CLI yet: carry the climate normal for the same calendar day from the latest year
        nm = cli["normal"].dropna()
        key = nm.groupby([nm.index.month, nm.index.day]).last()
        fill = pd.Series([key.get((d.month, d.day), np.nan) for d in out["D"]], index=out.index)
        out.loc[live, "normal"] = fill[live]
    out["icao"] = icao
    out["series"] = HIGH[icao]["series"]
    out["lst_off"] = off
    return out


def add_lag_errors(df: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recent errors at the station from days whose CLI was public at tau (D-1 counts from 12Z on D).
    history: rows (any h) carrying y for past days, used live where df itself has no labels yet."""
    src = df if history is None else pd.concat([history, df], ignore_index=True)
    base = src[src["h"] == 3][["icao", "D", "y", "nbs_txn", "gfs_n_x", "ec_max", "normal"]].copy()
    base = base.drop_duplicates(["icao", "D"], keep="first")
    base["e_nbs"] = base["y"] - base["nbs_txn"]
    base["e_gfs"] = base["y"] - base["gfs_n_x"]
    base["e_ec"] = base["y"] - base["ec_max"]
    base["anom"] = base["y"] - base["normal"]
    feats = []
    for icao, g in base.groupby("icao"):
        g = g.set_index("D").sort_index()
        full = pd.date_range(g.index.min(), max(g.index.max(), df["D"].max()), freq="D")
        g = g.reindex(full)
        for lag0 in (1, 2):
            f = pd.DataFrame(index=g.index)
            for c in ("e_nbs", "e_gfs", "e_ec"):
                f[f"{c}_l"] = g[c].shift(lag0)
                f[f"{c}_m7"] = g[c].shift(lag0).rolling(7, min_periods=3).mean()
            f["e_nbs_m30"] = g["e_nbs"].shift(lag0).rolling(30, min_periods=10).mean()
            f["anom_l"] = g["anom"].shift(lag0)
            f["icao"] = icao
            f["lag0"] = lag0
            feats.append(f.rename_axis("D").reset_index())
    feats = pd.concat(feats, ignore_index=True)
    df = df.copy()
    df["lag0"] = np.where(df["h"] >= 12, 1, 2)
    out = df.merge(feats, on=["icao", "D", "lag0"], how="left")
    # yesterday's guidance (as issued for D-1 at its 03Z decision) to pair with yesterday's observed max
    prev = base[["icao", "D", "nbs_txn", "ec_max"]].rename(columns={"nbs_txn": "nbs_txn_prev", "ec_max": "ec_max_prev"})
    prev["D"] = prev["D"] + pd.Timedelta(days=1)
    out = out.merge(prev, on=["icao", "D"], how="left")
    if "obs_prev_max" in out:
        out["e_prev_nbs"] = out["obs_prev_max"] - out["nbs_txn_prev"]
        out["e_prev_ec"] = out["obs_prev_max"] - out["ec_max_prev"]
    return out


# ----------------------------------------------------------------------------------------------- model features
FEATS = ["h", "loc_h", "xnd", "nbs_age_h", "txn_d", "nbh_d", "nbh_n", "nbh_dpt_d", "nbh_sky", "nbh_wsp", "nbh_u",
         "nbh_v", "nbh_p06", "nbh_sol", "nbh_tsd", "nbh_gst", "nbe_d", "nbe_xnd", "gfs_d", "nam_d", "mex_d",
         "gfs_age_h", "nam_age_h", "lav_d", "lav_n", "lav_dpt_d", "lav_wsp", "ec_d", "ec18_d", "ec21_d", "ec00_d",
         "ec_age_h", "obs_max_d", "obs_last_d", "obs_age_h", "obs_dwp_d", "obs_sknt", "obs_u", "obs_v", "obs_sky",
         "obs_dt3", "obs_dp3", "norm_d", "e_nbs_l", "e_nbs_m7", "e_nbs_m30", "e_gfs_l", "e_gfs_m7", "e_ec_l",
         "e_ec_m7", "anom_l", "doy_s", "doy_c", "st", "spread", "anchor_frac", "base_d", "nbm_v5"]


FEATS_V3 = FEATS + ["ecx_mx_d", "ecx_tcc", "ecx_cape", "ecx_2d_d", "ecx_tp", "ecx_age_h", "prev_max_d", "e_prev_nbs",
                    "e_prev_ec"]


def anchor(df: pd.DataFrame):
    base = df["nbs_txn"].copy()
    base = base.fillna(df["gfs_n_x"]).fillna(df["lav_tmax"] - 1.7)
    a = np.fmax(base, df["obs_max"].fillna(-999))
    return a, base


# ----------------------------------------------------------------------- round 5: the NBM fix (candidate D, Oct 8)
# The market beats Tier-1 on the nights NBM departs from the other guidance, and since NBM v5.0.14 (Jul 28, 2026) those
# departures are mostly wrong (research report "Beating the market on daily highs"). Candidate D gives the trees NBM's
# departure from the guidance median, the median itself and a v5.0.14 flag, and optionally centers the classes on the
# median instead of NBM (anchor_mode "median"). Training (research/ml/tier1_r1_oos.py hrrr_d1 / hrrr_d2 and
# research/ml/train_d.py) and the live bot use these same functions.
V5014_DATE = pd.Timestamp("2026-07-28")
ROUND5_FEATS = ["nbm_dep", "gmed_d", "nbm_v5014"]


def guidance_median(df: pd.DataFrame) -> np.ndarray:
    """Median of the day's max from NBM, GFS MOS, HRRR (rest of the day), ECMWF and LAMP (LAMP with anchor()'s -1.7 F),
    over the sources present."""
    hr = df["hrrr_rest"] if "hrrr_rest" in df else pd.Series(np.nan, index=df.index)
    g = np.column_stack([df["nbs_txn"], df["gfs_n_x"], hr, df["ec_max"], df["lav_tmax"] - 1.7]).astype(float)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(np.where(np.isfinite(g), g, np.nan), axis=1)


def anchor_median(df: pd.DataFrame):
    base = pd.Series(guidance_median(df), index=df.index)
    base = base.fillna(df["nbs_txn"]).fillna(df["gfs_n_x"]).fillna(df["lav_tmax"] - 1.7)
    return np.fmax(base, df["obs_max"].fillna(-999)), base


def add_round5_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    med = guidance_median(df)
    x["nbm_dep"] = df["nbs_txn"] - med
    x["gmed_d"] = med - x["anchor"]
    x["nbm_v5014"] = (pd.to_datetime(df["D"]) >= V5014_DATE).astype(int)
    return x


NBM_FRESH_FEATS = ["nbf_d", "nbf_rev", "nbf_xnd", "nbf_age_h"]


def add_nbm_fresh_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """The freshest hourly NBM run public at the decision (raw columns nbf_txn, nbf_xnd, nbf_age_h; the bot reads it
    from NOAA's bucket, training from research/ml/pull_nbm_hourly.py): its max relative to the anchor, its revision of
    the 6-hourly run the other inputs use, its spread and its age."""
    for c in ("nbf_txn", "nbf_xnd", "nbf_age_h"):
        if c not in df:
            df = df.assign(**{c: np.nan})
    x["nbf_d"] = df["nbf_txn"] - x["anchor"]
    x["nbf_rev"] = df["nbf_txn"] - df["nbs_txn"]
    x["nbf_xnd"] = df["nbf_xnd"]
    x["nbf_age_h"] = df["nbf_age_h"]
    return x


NBP_FEATS = ["nbp_p50_d", "nbp_lo_d", "nbp_hi_d", "nbp_iqr", "nbp_skew", "nbp_mn_rev", "nbp_age_h"]


def add_nbp_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """NBM's probabilistic text (NBP, wxbot/ml/nbp.py; training from research/ml/pull_nbp.py), freshest run public at
    the decision (raw columns nbp_mn, nbp_sd, nbp_p10 ... nbp_p90, nbp_age_h): the median and the 10th / 90th
    percentiles relative to the anchor, the interquartile width, the skew ((p90 - p50) - (p50 - p10)), the
    probabilistic mean's revision of the deterministic max the other inputs use, and the run's age."""
    for c in ("nbp_mn", "nbp_p10", "nbp_p25", "nbp_p50", "nbp_p75", "nbp_p90", "nbp_age_h"):
        if c not in df:
            df = df.assign(**{c: np.nan})
    x["nbp_p50_d"] = df["nbp_p50"] - x["anchor"]
    x["nbp_lo_d"] = df["nbp_p10"] - x["anchor"]
    x["nbp_hi_d"] = df["nbp_p90"] - x["anchor"]
    x["nbp_iqr"] = df["nbp_p75"] - df["nbp_p25"]
    x["nbp_skew"] = (df["nbp_p90"] - df["nbp_p50"]) - (df["nbp_p50"] - df["nbp_p10"])
    x["nbp_mn_rev"] = df["nbp_mn"] - df["nbs_txn"]
    x["nbp_age_h"] = df["nbp_age_h"]
    return x


def model_features(df: pd.DataFrame, anchor_mode: str = "nbm") -> pd.DataFrame:
    df = nam_fallback(df)
    a, base = anchor_median(df) if anchor_mode == "median" else anchor(df)
    x = pd.DataFrame(index=df.index)
    x["anchor"] = a
    x["anchor_r"] = np.round(a)
    x["anchor_frac"] = a - np.round(a)
    x["base_d"] = base - a
    x["h"] = df["h"]
    x["loc_h"] = (df["h"] + df["lst_off"]) % 24
    x["xnd"] = df["nbs_xnd"]
    x["nbs_age_h"] = df["nbs_age_h"]
    x["txn_d"] = df["nbs_txn"] - a
    x["nbh_d"] = df["nbh_tmax"] - a
    x["nbh_dpt_d"] = df["nbh_dpt"] - a
    for c in ["nbh_n", "nbh_sky", "nbh_wsp", "nbh_u", "nbh_v", "nbh_p06", "nbh_sol", "nbh_tsd", "nbh_gst", "nbe_xnd",
              "gfs_age_h", "nam_age_h", "lav_n", "lav_wsp", "ec_age_h", "obs_age_h", "obs_sknt", "obs_u", "obs_v",
              "obs_sky", "obs_dt3", "obs_dp3", "e_nbs_l", "e_nbs_m7", "e_nbs_m30", "e_gfs_l", "e_gfs_m7", "e_ec_l",
              "e_ec_m7", "anom_l"]:
        x[c] = df[c]
    x["nbe_d"] = df["nbe_txn"] - a
    x["gfs_d"] = df["gfs_n_x"] - a
    x["nam_d"] = df["nam_n_x"] - a
    x["mex_d"] = df["mex_n_x"] - a
    x["lav_d"] = df["lav_tmax"] - a
    x["lav_dpt_d"] = df["lav_dpt"] - a
    x["ec_d"] = df["ec_max"] - a
    x["ec18_d"] = df["ec_t18"] - a
    x["ec21_d"] = df["ec_t21"] - a
    x["ec00_d"] = df["ec_t00"] - a
    x["obs_max_d"] = df["obs_max"] - a
    x["obs_last_d"] = df["obs_last"] - a
    x["obs_dwp_d"] = df["obs_dwp"] - a
    x["norm_d"] = df["normal"] - a
    doy = df["D"].dt.dayofyear
    x["doy_s"] = np.sin(2 * np.pi * doy / 365.25)
    x["doy_c"] = np.cos(2 * np.pi * doy / 365.25)
    x["st"] = df["icao"].astype("category")
    g = df[["nbs_txn", "gfs_n_x", "nam_n_x", "ec_max", "lav_tmax"]]
    x["spread"] = g.max(axis=1) - g.min(axis=1)
    x["nbm_v5"] = (df["D"] >= V5_DATE).astype(int)
    for c in ["ecx_mx", "ecx_tcc", "ecx_cape", "ecx_2d", "ecx_tp", "ecx_age_h"]:
        if c not in df:
            df = df.assign(**{c: np.nan})
    x["ecx_mx_d"] = df["ecx_mx"] - a
    x["ecx_2d_d"] = df["ecx_2d"] - a
    x["ecx_tcc"] = df["ecx_tcc"]
    x["ecx_cape"] = df["ecx_cape"]
    x["ecx_tp"] = df["ecx_tp"]
    x["ecx_age_h"] = df["ecx_age_h"]
    for c in ["obs_prev_max", "e_prev_nbs", "e_prev_ec"]:
        if c not in df:
            df = df.assign(**{c: np.nan})
    x["prev_max_d"] = df["obs_prev_max"] - a
    x["e_prev_nbs"] = df["e_prev_nbs"]
    x["e_prev_ec"] = df["e_prev_ec"]
    return x
