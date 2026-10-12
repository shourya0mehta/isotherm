"""Rain forecaster for Kalshi's daily rain markets (KXRAIN: "Will it rain in <city> on <date>?", YES when the NWS
climate report shows at least 0.01 in; a trace settles NO). Shared by training (research/rain) and the live bot, so
there is no train/serve skew: both read a "raw" directory in the Tier-1 layout (wxbot/ml/features.py).

For a station, local climate day D and decision time tau = D 00Z + h hours, the inputs are:
  guidance PoP / QPF from the latest run of each model available at tau (NBM NBS, GFS MOS, NAM MOS, LAMP, NBM NBE,
    GFS-extended MEX; publication lags as Tier-1): over the periods of D not yet ended at tau and over the whole day,
    the largest PoP, the chance of at least one wet period if periods were independent, the largest QPF category
  observations: hourly METAR precipitation since local midnight up to tau - 5 min, trace flags, hours left
  climatology: the climate report's daily normal precipitation, day of year, station
Model: LightGBM binary classifier (artifacts/rain_tier1.txt.gz, trained on data through Sep 30, 2026). Pool with the
market: p = sigmoid(a + b logit(mid) + c logit(model)), weights in artifacts/rain_meta.json (fit on Aug-Sep 2026
out-of-sample rows). Research, evidence and the pre-registered test: research/rain, docs/STRATEGY.md 4d.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from typing import Optional

import numpy as np
import pandas as pd

from .features import HIGH, LAG_MIN, _cli_final, load_mos, localday, pick

ART = os.path.join(os.path.dirname(__file__), "artifacts")
HOURS = [-3, 3, 9, 12, 15, 18, 21, 24]
# KXRAIN city code -> climate site; the stations the bot already follows (Chicago rain settles at O'Hare: not here)
RAIN_CITY = {"NYC": "KNYC", "AUS": "KAUS", "MIA": "KMIA", "DEN": "KDEN", "PHIL": "KPHL", "LAX": "KLAX", "LV": "KLAS",
             "NOLA": "KMSY", "SFO": "KSFO", "DC": "KDCA", "SEA": "KSEA", "BOS": "KBOS", "PHX": "KPHX", "ATL": "KATL",
             "MIN": "KMSP", "DAL": "KDFW", "SATX": "KSAT", "HOU": "KHOU", "OKC": "KOKC", "EWR": "KEWR", "TTN": "KTTN"}
# the other nine KXRAIN climate sites (research for a later version; the frozen Sleeve R uses RAIN_CITY only)
RAIN_CITY_EXTRA = {"CHI": "KORD", "MKE": "KMKE", "LEX": "KLEX", "CMH": "KCMH", "PVD": "KPVD", "CLL": "KCLL",
                   "PIT": "KPIT", "ABQ": "KABQ", "SGF": "KSGF"}
EXTRA_LST = {"KORD": -6, "KMKE": -6, "KLEX": -5, "KCMH": -5, "KPVD": -5, "KCLL": -6, "KPIT": -5, "KABQ": -7, "KSGF": -6}
PERIOD_MODELS = [  # model, PoP columns with period length (h), QPF columns with period length (h)
    ("NBS", {"p06": 6, "p12": 12}, {"q06": 6, "q12": 12}),
    ("GFS", {"p06": 6, "p12": 12}, {"q06": 6, "q12": 12}),
    ("NAM", {"p06": 6, "p12": 12}, {"q06": 6, "q12": 12}),
    ("NBE", {"p12": 12}, {"q12": 12}),
    ("MEX", {"p12": 12}, {"q12": 12}),
    ("LAV", {"p06": 6}, {}),
]
# version 2 (round-2 research, not frozen): LAMP's hourly fields too: 1-hour PoP, convection and lightning chances
PERIOD_MODELS_V2 = PERIOD_MODELS[:-1] + [("LAV", {"p06": 6, "p01": 1}, {"cp1": 1, "lp1": 1})]
DROP = {"D", "tau", "y", "amt", "city", "icao"}


def lst_offset(icao: str) -> int:
    return int(HIGH[icao]["lst_offset_h"]) if icao in HIGH else int(EXTRA_LST[icao])


def labels(raw: str, icao: str, off: int) -> pd.DataFrame:
    """Final climate reports only: y = precipitation >= 0.01 in (trace = 0), the day's normal precipitation."""
    rows = []
    for f in glob.glob(f"{raw}/cli/{icao}_*.json"):
        for r in json.load(open(f)):
            p = r.get("precip")
            if p in (None, "M") or not _cli_final(r, off):
                continue
            amt = 0.0 if p == "T" else float(p)
            nrm = r.get("precip_normal")
            rows.append({"D": pd.Timestamp(r["valid"]), "y": int(amt >= 0.01), "amt": amt,
                         "pnorm": float(nrm) if nrm not in (None, "M", "T") else np.nan})
    if not rows:
        return pd.DataFrame(columns=["y", "amt", "pnorm"], index=pd.DatetimeIndex([], name="D"))
    return pd.DataFrame(rows).drop_duplicates("D").set_index("D").sort_index()


def normals(raw: str, icao: str) -> dict:
    """(month, day) -> daily normal precipitation from any climate report on that calendar day (normals are fixed
    for a decade, so last year's value stands in for a day whose report does not exist yet)."""
    out = {}
    for f in sorted(glob.glob(f"{raw}/cli/{icao}_*.json")):
        for r in json.load(open(f)):
            nrm = r.get("precip_normal")
            if nrm in (None, "M", "T"):
                continue
            d = pd.Timestamp(r["valid"])
            out[(d.month, d.day)] = float(nrm)
    return out


def period_features(raw: str, frame: pd.DataFrame, model: str, pcols: dict, qcols: dict, icao: str,
                    off: int) -> pd.DataFrame:
    """Per frame row: PoP/QPF over the periods of D not yet ended at tau ("rem") and over the whole day ("day"),
    from the latest run available at tau."""
    pre = model.lower()
    cols = list(pcols) + list(qcols)
    m = load_mos(raw, model, icao, cols)
    if model == "LAV":
        m = m[m["runtime"].dt.hour.isin([0, 6, 12, 18])]   # the archive holds these cycles; live matches it
    long = []
    for c, L in {**pcols, **qcols}.items():
        x = m[["runtime", "ftime", c]].dropna().rename(columns={c: "v"})
        x["var"] = c
        x["D"] = localday(x["ftime"] - pd.Timedelta(hours=L / 2), off)
        long.append(x)
    P = pd.concat(long, ignore_index=True) if long else pd.DataFrame(columns=["runtime", "ftime", "v", "var", "D"])
    runs = P[["runtime", "D"]].drop_duplicates()
    runs["has"] = 1
    picked = pick(runs, frame[["D", "h", "tau"]], LAG_MIN[model], pre, ["has"])
    picked["row"] = np.arange(len(picked))
    P["D"] = pd.to_datetime(P["D"]).astype("datetime64[ns]")
    j = picked[["row", "D", "tau", f"{pre}_run"]].merge(P, left_on=[f"{pre}_run", "D"], right_on=["runtime", "D"],
                                                         how="inner")
    out = pd.DataFrame({"row": np.arange(len(picked))})
    for scope, jj in (("rem", j[j["ftime"] > j["tau"]]), ("day", j)):
        for c in pcols:
            s = jj[jj["var"] == c]
            g = s.groupby("row")["v"]
            out[f"{pre}_{c}_max_{scope}"] = out["row"].map(g.max())
            lg = np.log1p(-np.clip(s["v"].values.astype(float), 0, 99.9) / 100)
            out[f"{pre}_{c}_any_{scope}"] = out["row"].map(
                1 - np.exp(pd.Series(lg, index=s.index).groupby(s["row"]).sum()))
            if scope == "rem":
                out[f"{pre}_{c}_n_rem"] = out["row"].map(g.size()).fillna(0)
        for c in qcols:
            s = jj[jj["var"] == c]
            out[f"{pre}_{c}_max_{scope}"] = out["row"].map(s.groupby("row")["v"].max())
    out[f"{pre}_age_h"] = picked[f"{pre}_age_h"].values
    out["D"] = picked["D"].values
    out["h"] = picked["h"].values
    return out.drop(columns=["row"])


def load_obs(raw: str, icao: str) -> pd.DataFrame:
    code = icao[1:]
    parts = []
    for f in sorted(glob.glob(f"{raw}/obs/asos_*.csv")):
        try:
            d = pd.read_csv(f, usecols=["station", "valid", "p01i"], dtype={"p01i": str, "station": str})
        except ValueError:
            continue
        parts.append(d[d["station"].isin([code, icao])])
    if not parts:
        return pd.DataFrame({"valid": pd.to_datetime([]), "p": [], "tr": []})
    o = pd.concat(parts, ignore_index=True)
    o["valid"] = pd.to_datetime(o["valid"])
    o["tr"] = (o["p01i"] == "T").astype(float)
    o["p"] = pd.to_numeric(o["p01i"].replace({"T": "0", "M": None}), errors="coerce")
    return o[["valid", "p", "tr"]]


def obs_rain(frame: pd.DataFrame, obs: pd.DataFrame, off: int) -> pd.DataFrame:
    """Hourly METAR precipitation since local midnight, up to tau - 5 min (the routine report carries the hour's
    total, so the largest value reported within each clock hour counts once)."""
    o = obs.copy()
    o["hr"] = o["valid"].dt.floor("h")
    hr = o.groupby("hr").agg(p=("p", "max"), tr=("tr", "max")).reset_index().sort_values("hr")
    hr["cum_p"] = hr["p"].fillna(0).cumsum()
    hr["cum_tr"] = hr["tr"].fillna(0).cumsum()
    f = frame[["D", "h", "tau"]].copy()
    f["start"] = f["D"].dt.tz_localize("UTC") - pd.Timedelta(hours=off)
    f["end"] = f["start"] + pd.Timedelta(days=1)
    f["cut"] = f[["tau", "end"]].min(axis=1) - pd.Timedelta(minutes=5)
    t = hr["hr"].values.astype("datetime64[ns]")

    def cum_at(ts, col):
        i = np.searchsorted(t, ts.values.astype("datetime64[ns]"), side="right") - 1
        return np.where(i >= 0, hr[col].values[np.clip(i, 0, None)] if len(hr) else 0.0, 0.0)
    for col, name in (("cum_p", "obs_p"), ("cum_tr", "obs_tr")):
        a = cum_at(f["cut"].dt.tz_convert("UTC").dt.tz_localize(None), col)
        b = cum_at(f["start"].dt.tz_localize(None) - pd.Timedelta(minutes=5), col)
        f[name] = np.where(f["cut"] > f["start"], a - b, 0.0)
    f["obs_wet"] = (f["obs_p"] >= 0.01 - 1e-9).astype(int)
    f["obs_any"] = ((f["obs_p"] > 0) | (f["obs_tr"] > 0)).astype(int)
    f["hours_left"] = (f["end"] - f["tau"]).dt.total_seconds() / 3600
    return f[["obs_p", "obs_tr", "obs_wet", "obs_any", "hours_left"]]


def rows(raw: str, icao: str, frame: Optional[pd.DataFrame] = None, start: str = "2020-06-05",
         version: int = 1) -> pd.DataFrame:
    """Feature rows for one station. frame: columns D (naive date), h; default = every day with a final climate
    report since `start`, at every decision hour (training). Labels are attached where a final report exists."""
    off = lst_offset(icao)
    lab = labels(raw, icao, off)
    if frame is None:
        days = lab.index[lab.index >= start]
        frame = pd.DataFrame([(d, h) for d in days for h in HOURS], columns=["D", "h"])
    frame = frame[["D", "h"]].copy()
    frame["D"] = pd.to_datetime(frame["D"]).astype("datetime64[ns]")
    frame["tau"] = frame["D"].dt.tz_localize("UTC") + pd.to_timedelta(frame["h"], unit="h")
    frame = frame.sort_values(["D", "h"]).reset_index(drop=True)
    out = frame.copy()
    for model, pc, qc in (PERIOD_MODELS if version == 1 else PERIOD_MODELS_V2):
        out = out.merge(period_features(raw, frame, model, pc, qc, icao, off), on=["D", "h"], how="left")
    out = pd.concat([out.reset_index(drop=True), obs_rain(frame, load_obs(raw, icao), off).reset_index(drop=True)],
                    axis=1)
    out = out.merge(lab, left_on="D", right_index=True, how="left")
    if out["pnorm"].isna().any():
        nm = normals(raw, icao)
        fill = pd.Series([nm.get((d.month, d.day), np.nan) for d in out["D"]], index=out.index)
        out["pnorm"] = out["pnorm"].fillna(fill)
    out["icao"] = icao
    out["lst_off"] = off
    doy = out["D"].dt.dayofyear
    out["doy_s"], out["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return out


def feature_names(df: pd.DataFrame) -> list:
    f = [c for c in df.columns if c not in DROP and not c.endswith("_run") and c != "st"]
    return f + ["st"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class RainModel:
    """A frozen rain forecaster plus its pool weights. predict(raw, icao, D, h) -> P(rain) or None.
    suffix "" = Sleeve R (feature version 1, 21 cities); "_v2" = the R2 variant (version 2, all 30 cities)."""

    def __init__(self, art_dir: str = ART, suffix: str = ""):
        self.meta = json.load(open(os.path.join(art_dir, f"rain_meta{suffix}.json")))
        s = gzip.open(os.path.join(art_dir, f"rain_tier1{suffix}.txt.gz"), "rt").read()
        try:
            import lightgbm as lgb
            self.booster = lgb.Booster(model_str=s)
            self.engine = "lightgbm"
        except Exception:
            from .npgbm import NumpyBooster
            self.booster = NumpyBooster(s)
            self.engine = "numpy"
        self.features = self.meta["features"]
        self.stations = self.meta["stations"]     # category order used in training
        self.w = np.array(self.meta["pool"], float)
        self.version = int(self.meta.get("version", 1))

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        X = df.reindex(columns=self.features[:-1]).to_numpy(dtype=float)
        st = np.array([self.stations.index(x) if x in self.stations else np.nan for x in df["icao"]], float)
        return np.column_stack([X, st])

    def predict_rows(self, df: pd.DataFrame) -> np.ndarray:
        p = np.asarray(self.booster.predict(self.matrix(df)), float).reshape(-1)
        return np.clip(p, 1e-4, 1 - 1e-4)

    def predict(self, raw: str, icao: str, D, h: int) -> Optional[float]:
        df = rows(raw, icao, pd.DataFrame({"D": [pd.Timestamp(D)], "h": [int(h)]}), version=self.version)
        if df.empty:
            return None
        return float(self.predict_rows(df)[0])

    def pool(self, mid: float, p: float) -> float:
        z = self.w[0] + self.w[1] * logit(mid) + self.w[2] * logit(p)
        return float(1 / (1 + np.exp(-z)))
