"""Tier-1 dataset: what free public guidance said about each station-day, as of each decision time.

One row per (station, local date D, decision time tau). Decision times are fixed in UTC on the target date:
  EB = 03:00Z on D (11 PM EDT the evening before), MO = 15:00Z on D (11 AM EDT the morning of).
A product counts only if it was available by tau: runtime + publication lag <= tau.
Sources (IEM MOS archive, free): NBM short-range (NBS: TXN max, XND spread, hourly fields), NBM extended (NBE),
GFS MOS (MAV, n_x), GFS extended MOS (MEX), GFS-LAMP hourly temperatures (LAV). Labels: NWS CLI daily max (IEM).
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RAW = os.environ.get("MLWX", "/home/claude/mlwx") + "/raw"
OUT = os.environ.get("MLWX", "/home/claude/mlwx") + "/tier1.parquet"
ST = json.load(open(os.path.join(os.path.dirname(__file__), "..", "..", "wxbot", "data", "stations.json")))
HIGH = {v["icao"]: v for v in ST.values() if v["kind"] == "high"}
DEC = {"EB": 3, "MO": 15}
LAG_MIN = {"NBS": 75, "NBE": 90, "GFS": 240, "MEX": 270, "LAV": 40}


def load(model: str, icao: str, cols: list[str]) -> pd.DataFrame:
    fs = [f for f in sorted(glob.glob(f"{RAW}/mos/{model}_{icao}_*.csv")) if os.path.getsize(f) > 200]
    if not fs:
        e = pd.DataFrame({"runtime": pd.to_datetime([], utc=True), "ftime": pd.to_datetime([], utc=True)})
        for c in cols:
            e[c] = pd.Series(dtype=float)
        return e
    keep = set(["runtime", "ftime"] + cols)
    df = pd.concat([pd.read_csv(f, usecols=lambda c: c in keep) for f in fs], ignore_index=True)
    df["runtime"] = pd.to_datetime(df["runtime"], utc=True, format="ISO8601")
    df["ftime"] = pd.to_datetime(df["ftime"], utc=True, format="ISO8601")
    return df.drop_duplicates(["runtime", "ftime"])


def pick(maxrows: pd.DataFrame, frame: pd.DataFrame, lag_min: int, prefix: str, vals: list[str]) -> pd.DataFrame:
    """For each (D, tau) in frame choose the latest run with runtime + lag <= tau that has a max row for D."""
    m = maxrows.copy()
    m["avail"] = m["runtime"] + pd.Timedelta(minutes=lag_min)
    m = m.sort_values("avail")
    f = frame.sort_values("tau")
    j = pd.merge_asof(f, m[["D", "avail", "runtime"] + vals], left_on="tau", right_on="avail", by="D",
                      direction="backward")
    j = j.rename(columns={**{v: f"{prefix}_{v}" for v in vals}, "runtime": f"{prefix}_run"}).drop(columns=["avail"])
    j[f"{prefix}_age_h"] = (j["tau"] - j[f"{prefix}_run"]).dt.total_seconds() / 3600
    return j


def localday(ts: pd.Series, off_h: int) -> pd.Series:
    return (ts + pd.Timedelta(hours=off_h)).dt.tz_localize(None).dt.normalize()


def station_rows(icao: str) -> pd.DataFrame:
    off = int(HIGH[icao]["lst_offset_h"])
    # labels
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
    frame = pd.DataFrame([(d, k, d.tz_localize("UTC") + pd.Timedelta(hours=h)) for d in days for k, h in DEC.items()],
                         columns=["D", "dec", "tau"])
    # NBM short-range: TXN/XND at the (D+1) 00Z column
    nbs = load("NBS", icao, ["txn", "xnd", "tmp", "dpt", "sky", "wsp", "p06", "sol", "tsd", "dsd", "gst"])
    mx = nbs[nbs["txn"].notna() & (nbs["ftime"].dt.hour == 0)].copy()
    mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
    out = pick(mx, frame, LAG_MIN["NBS"], "nbs", ["txn", "xnd"])
    # hourly NBS fields aggregated over the local standard day D, from the chosen run
    h = nbs.copy()
    h["D"] = localday(h["ftime"], off)
    agg = h.groupby(["runtime", "D"]).agg(nbs_tmax_h=("tmp", "max"), nbs_dpt=("dpt", "mean"), nbs_sky=("sky", "mean"),
                                          nbs_wsp=("wsp", "max"), nbs_p06=("p06", "max"), nbs_sol=("sol", "max"),
                                          nbs_tsd=("tsd", "mean"), nbs_gst=("gst", "max")).reset_index()
    agg = agg.rename(columns={"runtime": "nbs_run"})
    out = out.merge(agg, on=["nbs_run", "D"], how="left")
    # NBM extended
    nbe = load("NBE", icao, ["txn", "xnd"])
    mx = nbe[nbe["txn"].notna() & (nbe["ftime"].dt.hour == 0)].copy()
    mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
    out = pick(mx, out, LAG_MIN["NBE"], "nbe", ["txn", "xnd"])
    # GFS MOS and its extended version: n_x at the (D+1) 00Z column = daytime max
    for model, pre in (("GFS", "gfs"), ("MEX", "mex")):
        g = load(model, icao, ["n_x"])
        mx = g[g["n_x"].notna() & (g["ftime"].dt.hour == 0)].copy()
        mx["D"] = (mx["ftime"] - pd.Timedelta(days=1)).dt.tz_localize(None).dt.normalize()
        out = pick(mx, out, LAG_MIN[model], pre, ["n_x"])
    # LAMP hourly temperature: max over the local day from the latest run (and how many hours it covers)
    lav = load("LAV", icao, ["tmp", "dpt"])
    lav["D"] = localday(lav["ftime"], off)
    la = lav.groupby(["runtime", "D"]).agg(lav_tmax=("tmp", "max"), lav_n=("tmp", "count"),
                                           lav_dpt=("dpt", "mean")).reset_index()
    la = la[la["lav_n"] >= 6]
    out = pick(la.rename(columns={}), out, LAG_MIN["LAV"], "lav", ["lav_tmax", "lav_n", "lav_dpt"])
    out = out.rename(columns={"lav_lav_tmax": "lav_tmax", "lav_lav_n": "lav_n", "lav_lav_dpt": "lav_dpt"})
    # labels and normals
    out = out.merge(cli, left_on="D", right_index=True, how="left")
    out["icao"] = icao
    out["series"] = HIGH[icao]["series"]
    return out


def add_lag_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Recent forecast errors at the station (only days whose CLI is public at tau)."""
    eb = df[df["dec"] == "EB"][["icao", "D", "y", "nbs_txn", "gfs_n_x", "normal"]].copy()
    eb["e_nbs"] = eb["y"] - eb["nbs_txn"]
    eb["e_gfs"] = eb["y"] - eb["gfs_n_x"]
    eb["anom"] = eb["y"] - eb["normal"]
    feats = []
    for icao, g in eb.groupby("icao"):
        g = g.set_index("D").sort_index().asfreq("D")
        for lag0, dec in ((2, "EB"), (1, "MO")):  # CLI(D-1) is out by the morning of D, not by the evening before
            f = pd.DataFrame(index=g.index)
            f["e_nbs_l"] = g["e_nbs"].shift(lag0)
            f["e_nbs_m7"] = g["e_nbs"].shift(lag0).rolling(7, min_periods=3).mean()
            f["e_nbs_m30"] = g["e_nbs"].shift(lag0).rolling(30, min_periods=10).mean()
            f["e_gfs_l"] = g["e_gfs"].shift(lag0)
            f["e_gfs_m7"] = g["e_gfs"].shift(lag0).rolling(7, min_periods=3).mean()
            f["anom_l"] = g["anom"].shift(lag0)
            f["icao"] = icao
            f["dec"] = dec
            feats.append(f.reset_index())
    feats = pd.concat(feats, ignore_index=True)
    return df.merge(feats, on=["icao", "D", "dec"], how="left")


def main():
    parts = []
    for icao in sorted(HIGH):
        r = station_rows(icao)
        parts.append(r)
        print(icao, len(r), "rows; nbs coverage", round(r["nbs_txn"].notna().mean(), 3), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = add_lag_errors(df)
    df["doy"] = df["D"].dt.dayofyear
    df.to_parquet(OUT)
    print("wrote", OUT, df.shape)


if __name__ == "__main__":
    main()
