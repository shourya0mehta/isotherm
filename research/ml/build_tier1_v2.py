"""Tier-1 v2 dataset: everything free public guidance + observations said about each station-day, as of many
decision times (not just two). One row per (station, local date D, decision time tau).

Decision times (UTC, relative to D 00Z): -3 (21Z the day before), 3, 9, 12, 15, 18, 21.
A product counts only once it was public: runtime + publication lag <= tau.
Sources (all free): NBM NBS/NBE text (IEM), GFS MOS (MAV), NAM MOS (MET), GFS extended MOS (MEX), GFS-LAMP (LAV),
ECMWF IFS open data 2 m temperature (AWS mirror, CC-BY-4.0), ASOS/METAR observations (IEM). Labels: NWS CLI max.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from build_tier1 import HIGH, RAW, load, localday, pick  # noqa: E402

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
OUT = f"{MLWX}/tier1_v2.parquet"
DEC_H = [-3, 3, 9, 12, 15, 18, 21]
LAG_MIN = {"NBS": 75, "NBE": 90, "GFS": 240, "NAM": 210, "MEX": 270, "LAV": 40}
EC_LAG = {0: 455, 6: 390, 12: 455, 18: 390}
SKY = {"CLR": 0, "SKC": 0, "NSC": 0, "FEW": 1, "SCT": 2, "BKN": 3, "OVC": 4, "VV": 4}
_obs_cache = {}


def c2f(c):
    return c * 1.8 + 32.0


def obs_all() -> pd.DataFrame:
    if "all" in _obs_cache:
        return _obs_cache["all"]
    parts = []
    for f in sorted(glob.glob(f"{RAW}/obs/asos_*.csv")):
        parts.append(pd.read_csv(f, usecols=["station", "valid", "tmpf", "dwpf", "drct", "sknt", "skyc1", "mslp",
                                                 "metar"]))
    d = pd.concat(parts, ignore_index=True)
    d["valid"] = pd.to_datetime(d["valid"], utc=True)
    m = d["metar"].fillna("")
    tg = m.str.extract(r" T([01])(\d{3})([01])(\d{3})")
    tc = tg[1].astype(float) / 10.0 * np.where(tg[0] == "1", -1, 1)
    d["t_prec"] = np.where(tc.notna(), c2f(tc), d["tmpf"])
    mx = m.str.extract(r"RMK.* 1([01])(\d{3})(?: |$)")
    mxc = mx[1].astype(float) / 10.0 * np.where(mx[0] == "1", -1, 1)
    d["max6"] = np.where(mxc.notna() & d["valid"].dt.hour.isin([0, 6, 12, 18]) & (d["valid"].dt.minute >= 45)
                         | mxc.notna() & d["valid"].dt.hour.isin([23, 5, 11, 17]) & (d["valid"].dt.minute >= 45),
                         c2f(mxc), np.nan)
    d["sky"] = d["skyc1"].map(SKY)
    d = d.drop(columns=["metar"]).sort_values(["station", "valid"]).reset_index(drop=True)
    _obs_cache["all"] = d
    return d


def obs_features(icao: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Observed-to-date features at tau (obs valid <= tau - 5 min)."""
    off = int(HIGH[icao]["lst_offset_h"])
    o = obs_all()
    o = o[o["station"] == icao[1:]].copy()
    o = o[o["t_prec"].notna()].reset_index(drop=True)
    o["lday"] = localday(o["valid"], off)
    o["cummax"] = o.groupby("lday")["t_prec"].cummax()
    # 6-h max group: only windows that lie inside the local standard day of the report
    rep = o["valid"].dt.round("h")
    win_start_ok = localday(rep - pd.Timedelta(hours=6), off) == o["lday"]
    o["m6"] = np.where(win_start_ok, o["max6"], np.nan)
    o["cummax6"] = o.groupby("lday")["m6"].cummax()
    t = o["valid"].values.astype("datetime64[ns]").astype(np.int64)
    q = (frame["tau"] - pd.Timedelta(minutes=5)).values.astype("datetime64[ns]").astype(np.int64)
    i = np.searchsorted(t, q, side="right") - 1
    ok = i >= 0
    ii = np.where(ok, i, 0)
    res = pd.DataFrame(index=frame.index)
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
    # 3-hour tendencies
    q3 = q - 3 * 3600 * 10**9
    j = np.searchsorted(t, q3, side="right") - 1
    jj = np.where(j >= 0, j, 0)
    res["obs_dt3"] = np.where(ok & (j >= 0), o["t_prec"].values[ii] - o["t_prec"].values[jj], np.nan)
    res["obs_dp3"] = np.where(ok & (j >= 0), o["mslp"].values[ii] - o["mslp"].values[jj], np.nan)
    stale = res["obs_age_h"] > 3
    for c in ["obs_last", "obs_dwp", "obs_sknt", "obs_u", "obs_v", "obs_sky", "obs_dt3", "obs_dp3"]:
        res.loc[stale, c] = np.nan
    return res


def ecmwf_table(icao: str) -> pd.DataFrame:
    """Per IFS run and target day D: 2 m temperature (F) at 18Z, 21Z on D and 00Z on D+1 and their max."""
    if "ec" not in _obs_cache:
        fs = sorted(glob.glob(f"{RAW}/ecmwf/*.csv"))
        ec = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame(
            columns=["run", "step", "valid", "icao", "t2m_bil"])
        ec["run"] = pd.to_datetime(ec["run"], utc=True)
        ec["valid"] = pd.to_datetime(ec["valid"], utc=True)
        ec = ec.drop_duplicates(["run", "step", "icao"])
        _obs_cache["ec"] = ec
    ec = _obs_cache["ec"]
    e = ec[ec["icao"] == icao].copy()
    if e.empty:
        return pd.DataFrame(columns=["D", "runtime", "ec_t18", "ec_t21", "ec_t00", "ec_max", "lag"])
    e["D"] = (e["valid"] - pd.Timedelta(hours=1)).dt.tz_localize(None).dt.normalize()  # 00Z D+1 belongs to D
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


def pick_var_lag(table: pd.DataFrame, frame: pd.DataFrame, prefix: str, vals: list[str]) -> pd.DataFrame:
    m = table.copy()
    m["avail"] = m["runtime"] + pd.to_timedelta(m["lag"], unit="m")
    m = m.sort_values("avail")
    f = frame.sort_values("tau")
    j = pd.merge_asof(f, m[["D", "avail", "runtime"] + vals], left_on="tau", right_on="avail", by="D",
                      direction="backward")
    j = j.rename(columns={"runtime": f"{prefix}_run"}).drop(columns=["avail"])
    j[f"{prefix}_age_h"] = (j["tau"] - j[f"{prefix}_run"]).dt.total_seconds() / 3600
    return j


def station_rows(icao: str) -> pd.DataFrame:
    off = int(HIGH[icao]["lst_offset_h"])
    cli = []
    for f in glob.glob(f"{RAW}/cli/{icao}_*.json"):
        for r in json.load(open(f)):
            hi = r.get("high")
            if hi in (None, "M"):
                continue
            cli.append({"D": pd.Timestamp(r["valid"]), "y": float(hi),
                        "normal": float(r["high_normal"]) if r.get("high_normal") not in (None, "M") else np.nan})
    cli = pd.DataFrame(cli).drop_duplicates("D").set_index("D").sort_index()
    days = cli.index[(cli.index >= "2020-06-05")]
    frame = pd.DataFrame([(d, h, d.tz_localize("UTC") + pd.Timedelta(hours=h)) for d in days for h in DEC_H],
                         columns=["D", "h", "tau"])
    # NBM: TXN/XND for D from the latest run that still carries it
    nbs = load("NBS", icao, ["txn", "xnd", "tmp", "dpt", "sky", "wsp", "wdr", "p06", "sol", "tsd", "gst"])
    mx = nbs[nbs["txn"].notna() & (nbs["ftime"].dt.hour == 0)].copy()
    mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
    out = pick(mx, frame, LAG_MIN["NBS"], "nbs", ["txn", "xnd"])
    # NBM hourly fields over the rest of local day D from the latest run (any run)
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
    # NBM extended
    nbe = load("NBE", icao, ["txn", "xnd"])
    mx = nbe[nbe["txn"].notna() & (nbe["ftime"].dt.hour == 0)].copy()
    mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
    out = pick(mx, out, LAG_MIN["NBE"], "nbe", ["txn", "xnd"])
    # GFS MOS, NAM MOS, GFS extended MOS: daytime max for D
    for model, pre in (("GFS", "gfs"), ("NAM", "nam"), ("MEX", "mex")):
        g = load(model, icao, ["n_x"])
        mx = g[g["n_x"].notna() & (g["ftime"].dt.hour == 0)].copy()
        mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
        out = pick(mx, out, LAG_MIN[model], pre, ["n_x"])
    # LAMP: max over the rest of local day D from the latest run
    lav = load("LAV", icao, ["tmp", "dpt", "wsp"])
    lav["D"] = localday(lav["ftime"], off)
    la = lav.groupby(["runtime", "D"]).agg(tmax=("tmp", "max"), n=("tmp", "count"), dpt=("dpt", "mean"),
                                           wsp=("wsp", "max")).reset_index()
    out = pick(la, out, LAG_MIN["LAV"], "lav", ["tmax", "n", "dpt", "wsp"])
    # ECMWF IFS
    ec = ecmwf_table(icao)
    out = pick_var_lag(ec, out, "ec", ["ec_t18", "ec_t21", "ec_t00", "ec_max"])
    # observations to date
    out = out.sort_values(["D", "h"]).reset_index(drop=True)
    ob = obs_features(icao, out)
    out = pd.concat([out, ob], axis=1)
    out = out.merge(cli, left_on="D", right_index=True, how="left")
    out["icao"] = icao
    out["series"] = HIGH[icao]["series"]
    out["lst_off"] = off
    return out


def add_lag_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Recent errors at the station from days whose CLI was public at tau (D-1 counts from 12Z on D)."""
    base = df[df["h"] == 3][["icao", "D", "y", "nbs_txn", "gfs_n_x", "ec_max", "normal"]].copy()
    base["e_nbs"] = base["y"] - base["nbs_txn"]
    base["e_gfs"] = base["y"] - base["gfs_n_x"]
    base["e_ec"] = base["y"] - base["ec_max"]
    base["anom"] = base["y"] - base["normal"]
    feats = []
    for icao, g in base.groupby("icao"):
        g = g.set_index("D").sort_index().asfreq("D")
        for lag0 in (1, 2):
            f = pd.DataFrame(index=g.index)
            for c in ("e_nbs", "e_gfs", "e_ec"):
                f[f"{c}_l"] = g[c].shift(lag0)
                f[f"{c}_m7"] = g[c].shift(lag0).rolling(7, min_periods=3).mean()
            f["e_nbs_m30"] = g["e_nbs"].shift(lag0).rolling(30, min_periods=10).mean()
            f["anom_l"] = g["anom"].shift(lag0)
            f["icao"] = icao
            f["lag0"] = lag0
            feats.append(f.reset_index())
    feats = pd.concat(feats, ignore_index=True)
    df["lag0"] = np.where(df["h"] >= 12, 1, 2)
    return df.merge(feats, on=["icao", "D", "lag0"], how="left")


def main():
    parts = []
    for icao in sorted(HIGH):
        r = station_rows(icao)
        parts.append(r)
        print(icao, len(r), "nbs", round(r["nbs_txn"].notna().mean(), 3), "nam", round(r["nam_n_x"].notna().mean(), 3),
              "ec", round(r["ec_max"].notna().mean(), 3), "obs_max", round(r["obs_max"].notna().mean(), 3), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = add_lag_errors(df)
    df["doy"] = df["D"].dt.dayofyear
    df.to_parquet(OUT)
    print("wrote", OUT, df.shape)


if __name__ == "__main__":
    main()
