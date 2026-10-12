"""Minute-level nowcast of Kalshi's hourly Weather Index (research/hourly/README.md).

For a settlement minute S and a decision time tau, the model sees the index up to event minute t0 = tau - 5 (the
publication delay), its recent path, the same clock window yesterday, and the time of day and year. It predicts the
change R(S) - R(t0) as a distribution over 0.2 C steps (0.36 F, the grid the index moves on when every member
reports) with a LightGBM multiclass model per city; P(value > k) sums the steps above k.

Models: wxbot/ml/artifacts/nowcast_<city>.txt.gz (trained by research/hourly/train_nowcast_final.py on the index
rebuilt from one-minute ASOS, 2024-01 to 2026-09). Live prediction uses LightGBM when it loads, else the numpy
evaluator (same outputs).
"""
from __future__ import annotations

import gzip
import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("wxbot.nowcast")

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DELAY = 5          # minutes between an index minute and its publication
STEP = 0.36        # 0.2 C in F
NB = 15            # classes -NB..+NB steps (clipped)
LATLON = {"miami": (25.8, -80.3), "nyc": (40.75, -73.9), "chicago": (41.9, -87.9), "la-coastal": (33.9, -118.35)}
TZ = {"miami": "America/New_York", "nyc": "America/New_York", "chicago": "America/Chicago",
      "la-coastal": "America/Los_Angeles"}
SERIES_CITY = {"KXTEMPMIAH": "miami", "KXTEMPNYCHS": "nyc", "KXTEMPCHIHS": "chicago", "KXTEMPLAXHS": "la-coastal"}
FEATS = ["m", "d5", "d10", "d20", "d30", "d60", "d120", "y_win", "y_lvl", "hr", "doy_s", "doy_c", "sun_S", "sun_0"]


def solar_elev(ts: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Approximate solar elevation (degrees) for UTC timestamps."""
    d = ts.dayofyear.to_numpy()
    h = ts.hour.to_numpy() + ts.minute.to_numpy() / 60.0
    decl = 23.44 * np.sin(np.radians(360 / 365 * (d - 81)))
    ha = 15 * (h + lon / 15.0 - 12)
    la, de, hr = np.radians(lat), np.radians(decl), np.radians(ha)
    return np.degrees(np.arcsin(np.sin(la) * np.sin(de) + np.cos(la) * np.cos(de) * np.cos(hr)))


def features(R: pd.Series, S: pd.DatetimeIndex, m: int, city: str):
    """Feature frame (FEATS columns plus x0) for settlement minutes S (naive UTC) at lead m minutes.
    R: the index in F on a naive-UTC minute index (published values, or the rebuilt series in research)."""
    t0 = S - pd.Timedelta(minutes=m + DELAY)
    g = lambda t: R.reindex(t).to_numpy(dtype=float)
    x0 = g(t0)
    f = {"x0": x0, "m": np.full(len(S), float(m))}
    for k in (5, 10, 20, 30, 60, 120):
        f[f"d{k}"] = x0 - g(t0 - pd.Timedelta(minutes=k))
    f["y_win"] = g(S - pd.Timedelta(hours=24)) - g(t0 - pd.Timedelta(hours=24))   # same clock window yesterday
    f["y_lvl"] = x0 - g(t0 - pd.Timedelta(hours=24))
    loc = S.tz_localize("UTC").tz_convert(TZ[city])
    f["hr"] = loc.hour.to_numpy() + 0.0
    doy = loc.dayofyear.to_numpy()
    f["doy_s"], f["doy_c"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    lat, lon = LATLON[city]
    f["sun_S"] = solar_elev(S, lat, lon)
    f["sun_0"] = solar_elev(t0, lat, lon)
    return pd.DataFrame(f, index=S), x0


def exceed(P: np.ndarray, x0: np.ndarray, k: np.ndarray) -> np.ndarray:
    """P(x0 + STEP*(c - NB) > k) for class probabilities P (n, 2NB+1)."""
    grid = (np.arange(2 * NB + 1) - NB) * STEP
    return (P * ((x0[:, None] + grid[None, :]) > k[:, None])).sum(1)


class Nowcaster:
    """Per-city models; predict(city, R, S, m) -> (class probabilities (2NB+1,), x0) or None."""

    def __init__(self, art_dir: str = ART):
        self.models = {}
        self.engine = "none"
        for city in LATLON:
            fn = os.path.join(art_dir, f"nowcast_{city}.txt.gz")
            if not os.path.exists(fn):
                continue
            s = gzip.open(fn, "rt").read()
            try:
                import lightgbm as lgb
                self.models[city] = lgb.Booster(model_str=s)
                self.engine = "lightgbm"
            except Exception:
                from .npgbm import NumpyBooster
                self.models[city] = NumpyBooster(s)
                self.engine = "numpy"
        meta_f = os.path.join(art_dir, "nowcast_meta.json")
        self.meta = json.load(open(meta_f)) if os.path.exists(meta_f) else {}

    def ok(self, city: str) -> bool:
        return city in self.models

    def predict(self, city: str, R: pd.Series, S: pd.Timestamp, m: int) -> Optional[tuple]:
        if city not in self.models:
            return None
        X, x0 = features(R, pd.DatetimeIndex([S]), m, city)
        if not np.isfinite(x0[0]):
            return None
        P = np.asarray(self.models[city].predict(X[FEATS].to_numpy(dtype=float)), float).reshape(1, -1)
        if P.shape[1] < 2 * NB + 1:
            P = np.pad(P, ((0, 0), (0, 2 * NB + 1 - P.shape[1])))
        return P[0], float(x0[0])
