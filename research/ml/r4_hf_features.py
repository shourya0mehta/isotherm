"""Round 4, Track B part 2 (docs/results/r4_obs_speed.txt (e)): the NWS API's 5-minute observations as Tier-1 inputs.

The feed is simulated from the IEM ASOS 1-minute archive the way the NWS API reports it: the 5-minute average of the
1-minute readings, rounded to a whole F (ASOS; its daily maximum is the settled high), reported every 5 minutes in
whole degrees C, available FEED_LAG_MIN minutes after the reading (25 on Oct 7). For each Tier-1 row (icao, D, h),
from the readings public at the decision time:
  hf_lo_max  the largest whole F that every reading of the local day so far guarantees (lowest F that rounds to the
             reported C), i.e. a running max no settlement can fall below
  hf_last    the latest reading (F, middle of its C)          hf_trend  the latest minus the one 60 minutes before
  hf_n       readings so far (coverage)
Days without 1-minute data get NaN. Nothing dated Oct 1, 2026 or later is read. Writes mlwx/r4_hf_feats.parquet;
research/ml/tier1_r1_oos.py VARIANT=hrrr_hf adds them (relative to the anchor and to the METAR max so far).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

MLWX = os.environ.get("MLWX", "/home/claude/mlwx")
FEED_LAG_MIN = int(os.environ.get("FEED_LAG_MIN", "25"))
FEATS = ["hf_lo_max_d", "hf_gap", "hf_last_d", "hf_trend", "hf_n"]


def feed(stn3: str) -> pd.DataFrame:
    """The simulated 5-minute feed for one station: index = reading time (UTC), columns lo (guaranteed F), mid (F)."""
    parts = [pd.read_csv(f) for f in sorted(glob.glob(f"{MLWX}/raw/asos1min_r4/{stn3}_*.csv"))]
    if not parts:
        return pd.DataFrame(columns=["lo", "mid"])
    d = pd.concat(parts)
    t = pd.to_datetime(d["valid(UTC)"], utc=True)
    v = pd.Series(pd.to_numeric(d["tmpf"], errors="coerce").values, index=t).dropna()
    v = v[~v.index.duplicated()].sort_index()
    v = v[v.index < pd.Timestamp("2026-10-01", tz="UTC")]
    full = v.reindex(pd.date_range(v.index[0].floor("min"), v.index[-1].ceil("min"), freq="1min"))
    a5 = np.floor(full.rolling(5, min_periods=5).mean() + 0.5)
    a5 = a5[a5.index.minute % 5 == 0].dropna()
    c = np.floor((a5 - 32) * 5 / 9 + 0.5)
    return pd.DataFrame({"lo": np.ceil((c - 0.5) * 9 / 5 + 32 - 1e-9), "mid": c * 9 / 5 + 32}, index=a5.index)


def features(rows: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index, columns=["hf_lo_max", "hf_last", "hf_trend", "hf_n"], dtype=float)
    for icao, g in rows.groupby("icao"):
        f = feed(icao[1:])
        if f.empty:
            continue
        ts = f.index.values.astype("datetime64[ns]")
        lo, mid = f["lo"].values, f["mid"].values
        D = pd.to_datetime(g["D"]).values.astype("datetime64[ns]")
        dec = D + g["h"].values.astype(int) * np.timedelta64(1, "h")
        day0 = D + (-g["lst_off"].values.astype(int)) * np.timedelta64(1, "h")
        cut = dec - np.timedelta64(FEED_LAG_MIN, "m")
        i0 = np.searchsorted(ts, day0, side="left")           # first reading of the local day
        i1 = np.searchsorted(ts, cut, side="right") - 1       # last reading public at the decision
        for k, ix in enumerate(g.index):
            a, b = i0[k], i1[k]
            if b < a:
                continue
            seg = lo[a:b + 1]
            out.at[ix, "hf_lo_max"] = seg.max()
            out.at[ix, "hf_last"] = mid[b]
            out.at[ix, "hf_n"] = b - a + 1
            j = np.searchsorted(ts, ts[b] - np.timedelta64(60, "m"), side="left")
            if a <= j < b:
                out.at[ix, "hf_trend"] = mid[b] - mid[j]
    return out


def add_model_columns(x: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    x["hf_lo_max_d"] = (df["hf_lo_max"] if "hf_lo_max" in df else np.nan) - x["anchor"]
    x["hf_gap"] = (df["hf_lo_max"] if "hf_lo_max" in df else np.nan) - (df["obs_max"] if "obs_max" in df else np.nan)
    x["hf_last_d"] = (df["hf_last"] if "hf_last" in df else np.nan) - x["anchor"]
    x["hf_trend"] = df["hf_trend"] if "hf_trend" in df else np.nan
    x["hf_n"] = df["hf_n"] if "hf_n" in df else np.nan
    return x


def main():
    rows = pd.read_parquet(f"{MLWX}/tier1_v3.parquet", columns=["icao", "D", "h", "lst_off", "y", "obs_max"])
    rows = rows[(pd.to_datetime(rows["D"]) >= "2025-01-01") & (pd.to_datetime(rows["D"]) < "2026-10-01")]
    rows = rows.reset_index(drop=True)
    f = features(rows)
    out = pd.concat([rows[["icao", "D", "h"]], f], axis=1)
    out.to_parquet(f"{MLWX}/r4_hf_feats.parquet")
    have = out["hf_lo_max"].notna()
    print(f"coverage {have.mean():.1%} of {len(out)} rows (2025-01 to 2026-09)")
    sub = rows[have & rows["y"].notna()]
    over = out.loc[sub.index, "hf_lo_max"] > sub["y"]
    print(f"guaranteed max above the settled high (must be ~0): {over.mean():.3%}")
    gap = (out.loc[sub.index, "hf_lo_max"] - sub["obs_max"])
    for h in sorted(rows["h"].unique()):
        m = sub["h"] == h
        print(f"  h={h:+d}: feed shows more than the METARs on {(gap[m] > 0).mean():.1%} of rows")


if __name__ == "__main__":
    sys.exit(main())
