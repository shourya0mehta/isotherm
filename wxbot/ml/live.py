"""Live Tier-1 v2 forecasts for the bot: fetch the same free products used in training into a local raw directory
(training file layout), build features with wxbot.ml.features (same code as training), predict integer PMFs.

Refresh policy: a 45-day history window once a day (needed for lagged-error features), plus a 2-day window at every
decision hour. Decision hours are fixed UTC offsets from the target day's 00Z, as in training.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import requests

from . import features as F

log = logging.getLogger("wxbot.ml")
UA = {"User-Agent": "wxbot/0.2 (student research; contact via GitHub)"}
IEM = "https://mesonet.agron.iastate.edu"
MODELS = ["NBS", "NBE", "GFS", "NAM", "MEX", "LAV"]
ART = os.path.join(os.path.dirname(__file__), "artifacts")
ECMWF_HOSTS = ["https://data.ecmwf.int/forecasts",  # ECMWF's own portal: last few days only
               "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com"]  # AWS mirror: full archive (older runs)
EC_STEPS = {0: [18, 21, 24], 6: [12, 15, 18], 12: [6, 9, 12, 30, 33, 36], 18: [24, 27, 30]}


_IEM_LOCK = threading.Lock()
_IEM_LAST = [0.0]
IEM_GAP_S = 2.3  # IEM allows about one request per 2 s per address; 1 s spacing drew HTTP 429s, 2.3 s none (Oct 3)
# cycles each product has in IEM's feed (checked live and in the archive, Oct 3): NBM text 00/06/12/18Z, NBM extended
# 00/12Z, LAMP hourly. Fetching only when a new cycle can exist (with the prefetch between hours) leaves the decision
# refresh with only observations and HRRR to fetch.
MOS_CYCLES = {"GFS": (0, 6, 12, 18), "NAM": (0, 6, 12, 18), "MEX": (0, 12), "NBS": (0, 6, 12, 18), "NBE": (0, 12),
              "LAV": tuple(range(24))}


def _new_cycle_since(model: str, last: float, now: float) -> bool:
    """True if a cycle of `model` became available (cycle time + its publication lag, as in training) after `last`.
    Skipping only when nothing new can exist keeps live inputs identical to what training saw at each decision."""
    cyc = MOS_CYCLES.get(model)
    if cyc is None or last <= 0:
        return True
    lag = F.LAG_MIN[model] * 60
    t = int((last - lag) // 3600) * 3600 - 86400
    while t <= now:
        if dt.datetime.fromtimestamp(t, dt.timezone.utc).hour in cyc and last < t + lag <= now:
            return True
        t += 3600
    return False


def _iem_get(url, params=None, timeout=90, tries=6):
    """IEM GET, paced across threads so the bot stays under IEM's per-address limit instead of retrying 429s."""
    for k in range(tries):
        with _IEM_LOCK:
            wait = IEM_GAP_S - (time.time() - _IEM_LAST[0])
            if wait > 0:
                time.sleep(wait)
            _IEM_LAST[0] = time.time()
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(5 * (k + 1))
                continue
        except Exception:
            pass
        time.sleep(min(2 * 2 ** k, 20))
    return None


def _get(url, params=None, timeout=90, tries=6, headers=None):
    """GET with backoff; S3 answers 503 SlowDown when throttling, which is retried, 404 is final."""
    for k in range(tries):
        try:
            r = requests.get(url, params=params, headers={**UA, **(headers or {})}, timeout=timeout)
            if r.status_code in (200, 206) and not r.content[:5] == b"<?xml":
                return r
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(min(2 * 2 ** k, 20))
    return None


class LiveRaw:
    def __init__(self, root: str, stations: list[str], ecmwf: bool = True):
        self.root = root
        self.stations = stations
        self.ecmwf = ecmwf   # the rain-only folder (extra climate sites) skips the ECMWF fields
        for d in ("mos", "obs", "cli", "ecmwf", "ecmwf_x"):
            os.makedirs(os.path.join(root, d), exist_ok=True)
        self.last_hist_day = self._read_hist_day()  # a restart the same day reuses that day's history files
        self.last_mos: dict = {}  # model -> time of its last successful fetch
        self.last_cli = 0.0
        self.last_refresh_ts = 0.0  # end of the last refresh (the rain shadow reuses fresh data)

    def _read_hist_day(self):
        try:
            d = dt.date.fromisoformat(open(os.path.join(self.root, "hist_day.txt")).read().strip())
        except (OSError, ValueError):
            return None
        need = [os.path.join(self.root, "obs", "asos_hist.csv")]
        return d if all(os.path.exists(f) for f in need) else None

    # ---------------------------------------------------------------- fetchers
    def _mos(self, model, icao, days, tag):
        now = dt.datetime.now(dt.timezone.utc)
        p = {"station": icao, "model": model, "sts": (now - dt.timedelta(days=days)).strftime("%Y-%m-%dT%H:00Z"),
             "ets": (now + dt.timedelta(hours=2)).strftime("%Y-%m-%dT%H:00Z"), "format": "csv"}
        r = _iem_get(f"{IEM}/cgi-bin/request/mos.py", p)
        if r is not None and len(r.text) > 100:
            with open(os.path.join(self.root, "mos", f"{model}_{icao}_{tag}.csv"), "w") as f:
                f.write(r.text)
            return True
        return False

    def _obs(self, days, tag):
        now = dt.datetime.now(dt.timezone.utc)
        ids = ",".join(s[1:] for s in self.stations)
        url = (f"{IEM}/cgi-bin/request/asos.py?station={ids}&data=tmpf,dwpf,drct,sknt,skyc1,skyl1,mslp,p01i,metar"
               f"&sts={(now - dt.timedelta(days=days)):%Y-%m-%dT%H:%MZ}&ets={(now + dt.timedelta(hours=1)):%Y-%m-%dT%H:%MZ}"
               f"&tz=Etc/UTC&format=onlycomma&missing=empty&trace=T&report_type=3,4")
        r = _iem_get(url, timeout=180)
        if r is not None and len(r.content) > 1000:
            open(os.path.join(self.root, "obs", f"asos_{tag}.csv"), "wb").write(r.content)

    def _cli(self, icao, year):
        r = _iem_get(f"{IEM}/json/cli.py", {"station": icao, "year": year})
        if r is not None:
            json.dump(r.json().get("results", []), open(os.path.join(self.root, "cli", f"{icao}_{year}.json"), "w"))

    def _ecmwf_run(self, day: dt.date, hh: int):
        """2 m temperature at the afternoon steps (ecmwf/) plus max-in-3h, cloud, CAPE, dewpoint, precipitation
        (ecmwf_x/), same layout as the training pulls. A run that is not published yet writes nothing."""
        import eccodes
        coords = json.load(open(os.path.join(os.path.dirname(__file__), "station_coords.json")))
        names = sorted(coords)
        d = f"{day:%Y%m%d}"
        fn = os.path.join(self.root, "ecmwf", f"{d}_{hh:02d}.csv")
        fnx = os.path.join(self.root, "ecmwf_x", f"{d}_{hh:02d}.csv")
        if os.path.exists(fn) and os.path.exists(fnx):
            return
        os.makedirs(os.path.dirname(fnx), exist_ok=True)
        steps = EC_STEPS[hh]
        mids = set(steps[1::3])  # the 21Z-valid step of each (run, day) triple
        rows, rows_x = [], []
        recent = (dt.datetime.now(dt.timezone.utc).date() - day).days <= 3
        hosts = ECMWF_HOSTS if recent else ECMWF_HOSTS[::-1]
        for host, stream in [(h_, s_) for h_ in hosts for s_ in (["oper"] if hh in (0, 12) else ["oper", "scda"])]:
            base = f"{host}/{d}/{hh:02d}z/ifs/0p25/{stream}/{d}{hh:02d}0000-{{step}}h-{stream}-fc"
            for step in steps:
                r = _get(base.format(step=step) + ".index", timeout=60, tries=3)
                if r is None:
                    break
                want = {"2t", "mx2t3"} | ({"cape", "mucape", "2d", "tcc"} if step in mids else {"tp"})
                recs = {}
                for line in r.text.splitlines():
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    if j.get("levtype") == "sfc" and j.get("param") in want:
                        recs[j["param"]] = j
                run = dt.datetime(day.year, day.month, day.day, hh)
                valid = run + dt.timedelta(hours=step)
                for prm, rec in recs.items():
                    g = _get(base.format(step=step) + ".grib2", timeout=60,
                             headers={"Range": f"bytes={rec['_offset']}-{rec['_offset'] + rec['_length'] - 1}"})
                    if g is None or g.content[:4] != b"GRIB":
                        continue
                    msg = eccodes.codes_new_from_message(g.content)
                    try:
                        v = eccodes.codes_get_values(msg)
                        ni = eccodes.codes_get(msg, "Ni")
                        la0 = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
                        lo0 = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")
                        di = eccodes.codes_get(msg, "iDirectionIncrementInDegrees")
                        dj = eccodes.codes_get(msg, "jDirectionIncrementInDegrees")
                    finally:
                        eccodes.codes_release(msg)
                    for n in names:
                        x = ((coords[n]["lon"] - lo0) % 360.0) / di
                        y = (la0 - coords[n]["lat"]) / dj
                        i0, j0 = int(np.floor(x)), int(np.floor(y))
                        fx, fy = x - i0, y - j0
                        idx = [j0 * ni + i0 % ni, j0 * ni + (i0 + 1) % ni, (j0 + 1) * ni + i0 % ni,
                               (j0 + 1) * ni + (i0 + 1) % ni]
                        w = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]
                        val = float(np.dot(v[idx], w))
                        if prm == "2t":
                            rows.append([run.isoformat(), step, valid.isoformat(), n, round(val - 273.15, 3), "",
                                         "ifs/0p25"])
                        else:
                            rows_x.append([run.isoformat(), step, valid.isoformat(), n, prm, round(val, 5)])
            if rows:
                break
        if rows:
            pd.DataFrame(rows, columns=["run", "step", "valid", "icao", "t2m_bil", "t2m_near", "grid"]).to_csv(fn, index=False)
        if rows_x:
            pd.DataFrame(rows_x, columns=["run", "step", "valid", "icao", "param", "value"]).to_csv(fnx, index=False)

    def refresh(self, full: bool = False):
        """full: 45-day history (daily); else the last 2 days (every decision hour)."""
        t0 = time.time()
        days, tag = (45, "hist") if full else (2, "live")
        now_s = time.time()
        due = [m for m in ("NBS", "LAV", "NBE", "GFS", "NAM", "MEX")
               if full or _new_cycle_since(m, self.last_mos.get(m, 0.0), now_s)]
        jobs = [(m, s) for m in due for s in self.stations]
        with ThreadPoolExecutor(3) as ex:  # requests are paced by _iem_get; threads only overlap the transfers
            ok = list(ex.map(lambda j: self._mos(j[0], j[1], days, tag), jobs))
        for m in due:
            if any(o for (mm, _), o in zip(jobs, ok) if mm == m):
                self.last_mos[m] = now_s
        t1 = time.time()
        self._obs(days, tag)
        t2 = time.time()
        today = dt.datetime.now(dt.timezone.utc).date()
        if full:
            with ThreadPoolExecutor(3) as ex:
                list(ex.map(lambda s: [self._cli(s, y) for y in (today.year - 1, today.year)], self.stations))
            self.last_cli = time.time()
            ec_jobs = [(today - dt.timedelta(days=k), 18) for k in range(1, 46)]
        else:
            if time.time() - self.last_cli >= 2.5 * 3600:  # climate reports change once a day; save IEM requests
                with ThreadPoolExecutor(3) as ex:
                    list(ex.map(lambda s: self._cli(s, today.year), self.stations))
                self.last_cli = time.time()
            ec_jobs = []
        ec_jobs += [(today - dt.timedelta(days=k), hh) for k in range(0, 3) for hh in (0, 6, 12, 18)]
        if not self.ecmwf:
            ec_jobs = []
        t3 = time.time()

        def ec(j):
            try:
                self._ecmwf_run(*j)
            except Exception as e:  # a run not out yet, or a transient error
                log.debug("ecmwf %s %s: %s", j[0], j[1], e)
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(ec, ec_jobs))
        if full:
            self.last_hist_day = today
            try:
                open(os.path.join(self.root, "hist_day.txt"), "w").write(today.isoformat())
            except OSError:
                pass
        log.info("ML raw refresh (%s): MOS %.0fs, obs %.0fs, CLI %.0fs, ECMWF %.0fs", tag, t1 - t0, t2 - t1, t3 - t2,
                 time.time() - t3)
        self.last_refresh_ts = time.time()


class Tier1Live:
    def __init__(self, raw_root: str, stations: Optional[list[str]] = None, model: str = "tier1_v2"):
        """model: the artifact name (tier1_v2 is the forecaster shipped Oct 3; round 5's candidate D ships as its own
        name and declares its anchor and extra inputs in its meta file)."""
        self.model_name = model
        self.meta = json.load(open(os.path.join(ART, f"{model}_meta.json")))
        self.anchor_mode = self.meta.get("anchor_mode", "nbm")
        self.round5 = bool(self.meta.get("round5", False))
        self.uses_nbf = any(f.startswith("nbf_") for f in self.meta.get("feats", []))   # round 5: hourly NBM
        self.uses_nbp = any(f.startswith("nbp_") for f in self.meta.get("feats", []))   # round 5: NBM percentiles
        txt = os.path.join(ART, f"{model}.txt")
        if os.path.exists(txt):
            model_str = open(txt).read()
        else:  # the repo ships the gzipped model
            import gzip
            model_str = gzip.decompress(open(txt + ".gz", "rb").read()).decode()
        try:  # LightGBM's library needs OpenMP; macOS without Homebrew lacks it
            import lightgbm as lgb
            self.booster = lgb.Booster(model_str=model_str)
            self.engine = "lightgbm"
        except (ImportError, OSError) as e:
            from .npgbm import NumpyBooster
            self.booster = NumpyBooster(model_str)  # same trees, same outputs (to ~1e-15), no compiled library
            self.engine = "numpy"
            log.info("Tier-1 on the built-in numpy tree evaluator (%s)", str(e).splitlines()[0][:120])
        self.parity_max_diff = self.parity_check()
        self.stations = stations or sorted(F.HIGH)
        self.raw = LiveRaw(raw_root, self.stations)
        # HRRR inputs: only fetched when the shipped model was trained with them
        self.uses_hrrr = any(f.startswith("hrrr_") or f in ("e_hrrr_m14", "e_prev_hrrr") for f in self.meta["feats"])
        self.lock = threading.Lock()
        self.cache: dict = {}  # (icao, date) -> {"h": h, "P": np.array, "anchor_r": float, "t": ts}

    def parity_check(self, tol: float = 1e-6) -> float:
        """Startup self-check (Oct 4 review): the engine in use (LightGBM or the numpy tree evaluator) must reproduce
        the probabilities LightGBM gave for a stored set of inputs (artifacts/tier1_parity.json.gz, made in the cloud
        with this model). A mismatch raises, so the bot runs without the model (market-anchored) instead of trading
        on wrong probabilities."""
        import gzip
        f = os.path.join(ART, "tier1_parity.json.gz")
        if os.path.exists(os.path.join(ART, f"{getattr(self, 'model_name', 'tier1_v2')}_parity.json.gz")):
            f = os.path.join(ART, f"{self.model_name}_parity.json.gz")
        if not os.path.exists(f):
            log.warning("no Tier-1 parity fixture: skipping the startup check")
            return float("nan")
        fx = json.loads(gzip.decompress(open(f, "rb").read()).decode())
        if fx.get("model_sha256") != self.meta.get("sha256"):
            log.warning("Tier-1 parity fixture is for another model: skipping the startup check")
            return float("nan")
        import pandas as pd
        X = pd.DataFrame([[np.nan if v is None else v for v in r] for r in fx["X"]], columns=fx["num"])
        X["st"] = pd.Categorical(fx["st"], categories=self.meta["cats"])
        X = X[fx["columns"]]
        P = np.asarray(self.booster.predict(X))
        d = float(np.abs(P - np.asarray(fx["P"])).max())
        if not d <= tol:
            raise RuntimeError(f"Tier-1 parity check failed on the {self.engine} engine: max difference {d:.2e}")
        log.info("Tier-1 parity check passed on the %s engine (max difference %.1e)", self.engine, d)
        return d

    @staticmethod
    def decision_hour(D: dt.date, now: dt.datetime) -> Optional[int]:
        d0 = dt.datetime(D.year, D.month, D.day, tzinfo=dt.timezone.utc)
        hs = [h for h in F.DEC_H if d0 + dt.timedelta(hours=h) <= now]
        return hs[-1] if hs else None

    def predict(self, targets: list[tuple[str, dt.date]], now: Optional[dt.datetime] = None) -> dict:
        """targets: (icao, local date D). Returns {(icao, D): {...}} at each target's latest decision hour."""
        now = now or dt.datetime.now(dt.timezone.utc)
        if self.raw.last_hist_day != now.date():
            self.raw.refresh(full=True)
        else:
            self.raw.refresh(full=False)
        src = F.RawSource(self.raw.root)
        out = {}
        by_st: dict[str, list] = {}
        for icao, D in targets:
            h = self.decision_hour(D, now)
            if h is not None:
                by_st.setdefault(icao, []).append((pd.Timestamp(D), h))
        rows = []
        for icao, lst in by_st.items():
            # past 44 days plus today at the 03Z decision: lagged errors and "yesterday" features need them
            hist_days = pd.date_range(now.date() - dt.timedelta(days=44), now.date())
            frame = pd.DataFrame([(d, 3) for d in hist_days] + lst, columns=["D", "h"]).drop_duplicates()
            r = F.station_rows(src, icao, frame)
            r["_target"] = r.set_index(["D", "h"]).index.isin(pd.MultiIndex.from_tuples(lst))
            rows.append(r)
        if not rows:
            return out
        df = pd.concat(rows, ignore_index=True)
        if self.uses_hrrr:
            df = self._add_hrrr(df, now)
        hist = df[~df["_target"]]
        df = F.add_lag_errors(df[df["_target"]].drop(columns=["_target"]), history=hist)
        if self.uses_hrrr:
            from . import hrrr as H
            df = H.add_lag_errors(df, history=hist)
        if getattr(self, "uses_nbf", False):
            df = self._add_nbm_fresh(df)
        if getattr(self, "uses_nbp", False):
            df = self._add_nbp(df)
        x = F.model_features(df, anchor_mode=getattr(self, "anchor_mode", "nbm"))
        if self.uses_hrrr:
            x = H.add_model_columns(x, df)
        if getattr(self, "round5", False):
            x = F.add_round5_columns(x, df)
        if getattr(self, "uses_nbf", False):
            x = F.add_nbm_fresh_columns(x, df)
        if getattr(self, "uses_nbp", False):
            x = F.add_nbp_columns(x, df)
        x["st"] = pd.Categorical(df["icao"], categories=self.meta["cats"])
        self.last_inputs = x[self.meta["feats"]].assign(icao=df["icao"].values, D=df["D"].values, h=df["h"].values)
        p = self.booster.predict(x[self.meta["feats"]])
        full = np.zeros((len(df), F.NCLS))
        full[:, self.meta["classes"]] = p
        full = np.clip(full, 1e-6, None)
        full /= full.sum(axis=1, keepdims=True)
        for i, r in enumerate(df.itertuples()):
            key = (r.icao, r.D.date())
            out[key] = {"h": int(r.h), "P": full[i], "anchor_r": float(x["anchor_r"].iloc[i]), "t": now.timestamp(),
                        "nbs_txn": float(r.nbs_txn) if pd.notna(r.nbs_txn) else None,
                        "obs_max": float(r.obs_max) if pd.notna(r.obs_max) else None}
        with self.lock:
            self.cache.update(out)
        return out

    def _add_nbm_fresh(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round 5: the freshest hourly NBM run public at each row's decision time (wxbot/ml/nbm_hourly.py); a failed
        fetch leaves the columns empty, as for rows before the training archive."""
        from . import nbm_hourly as NH
        parts = []
        dec = pd.to_datetime(df["D"]) + pd.to_timedelta(df["h"], unit="h")
        for t, g in df.groupby(dec):
            try:
                need = {(i, pd.Timestamp(d)) for i, d in zip(g["icao"], g["D"])}
                fresh = NH.latest(pd.Timestamp(t), self.stations, os.path.join(self.raw.root, "nbm_hourly"), need=need)
            except Exception as e:
                log.warning("hourly NBM unavailable for %s: %s", t, e)
                fresh = None
            parts.append(NH.columns(g, fresh))
        cols = pd.concat(parts).reindex(df.index) if parts else None
        if cols is not None:
            df = pd.concat([df.drop(columns=[c for c in cols.columns if c in df]), cols], axis=1)
        return df

    def _add_nbp(self, df: pd.DataFrame) -> pd.DataFrame:
        """Round 5 step 5: NBM's percentiles (NBP text, wxbot/ml/nbp.py) from the latest run public at each row's
        decision time that carries the row's day; a failed fetch leaves the columns empty."""
        from . import nbp as NP
        parts = []
        dec = pd.to_datetime(df["D"]) + pd.to_timedelta(df["h"], unit="h")
        for t, g in df.groupby(dec):
            try:
                need = {(i, pd.Timestamp(d)) for i, d in zip(g["icao"], g["D"])}
                got = NP.latest(pd.Timestamp(t), self.stations, os.path.join(self.raw.root, "nbp"), need=need)
            except Exception as e:
                log.warning("NBM percentiles unavailable for %s: %s", t, e)
                got = None
            parts.append(NP.frame_columns(g, got))
        cols = pd.concat(parts).reindex(df.index) if parts else None
        if cols is not None:
            df = pd.concat([df.drop(columns=[c for c in cols.columns if c in df]), cols], axis=1)
        return df

    def _add_hrrr(self, df: pd.DataFrame, now: dt.datetime) -> pd.DataFrame:
        """Fetch the HRRR runs this decision needs (NOAA GRIB2 byte ranges, cached) and attach the raw HRRR columns.
        A failed fetch leaves the columns empty; the trees route missing values like they did in training."""
        from . import hrrr as H
        try:
            coords = json.load(open(os.path.join(os.path.dirname(__file__), "station_coords.json")))
            offs = sorted({int(F.HIGH[s]["lst_offset_h"]) for s in self.stations if s in F.HIGH})
            t0 = time.time()
            n = 0
            for init, fhs in H.runs_needed(now, offs).items():
                fhs = list(fhs)
                # a few hours per call: fetch_run holds each decoded CONUS field (~15 MB) until its call returns, so a
                # restart's 37-hour catch-up peaked near 800 MB (Oct 7); four at a time keeps it near 60 MB
                for i in range(0, len(fhs), 4):
                    n += H.fetch_run(init.date(), init.hour, fhs[i:i + 4], self.raw.root, coords)
            W = H.load_runs(self.raw.root, since=pd.Timestamp(now.date()) - pd.Timedelta(days=20))
            feats = H.compute(df, W)
            log.info("HRRR: %d new forecast hours, %d runs on disk, %.0fs", n,
                     W.index.get_level_values("init").nunique() if len(W) else 0, time.time() - t0)
            return pd.concat([df.drop(columns=[c for c in feats.columns if c in df]), feats], axis=1)
        except Exception as e:
            log.warning("HRRR features unavailable this round: %s", e)
            return df

    def get(self, icao: str, D: dt.date) -> Optional[dict]:
        with self.lock:
            return self.cache.get((icao, D))


def bracket_pmf(brackets, P: np.ndarray, anchor_r: float, K: int = F.K) -> np.ndarray:
    f = []
    for b in brackets:
        lo, hi = max(b.lo, anchor_r - K), min(b.hi, anchor_r + K)
        js = [int(v - anchor_r + K) for v in range(int(np.ceil(lo)), int(np.floor(hi)) + 1)]
        f.append(float(P[js].sum()) if js else 0.0)
    f = np.clip(np.array(f), 1e-4, None)
    return f / f.sum()


def nbm_dep(nbs: Optional[float], mu_m: float, clip_f: float = 4.0) -> float:
    """The pool's disagreement covariate (round 5, research/ml/r5_pool_gate.py "nbmdep"): NBM's max minus the
    market's implied mean, clipped to +-clip_f and scaled to +-1; 0 when NBM is missing."""
    if nbs is None or not np.isfinite(nbs) or not np.isfinite(mu_m):
        return 0.0
    return float(np.clip(nbs - mu_m, -clip_f, clip_f) / clip_f)


class Pool:
    def __init__(self, path: str = os.path.join(ART, "pool_weights.json")):
        self.w = json.load(open(path))
        self.gate = self.w.get("gate") or None   # round 5: exponents shift with NBM's departure from the market

    def probs(self, brackets, market: dict, f: np.ndarray, h: int, nbs: Optional[float] = None) -> dict:
        k = self.w["hours"].index(h)
        a, b, th, ka = (self.w[n][k] for n in ("alpha", "beta", "theta", "kappa"))
        m = np.clip(np.array([market.get(br.ticker, 0.005) for br in brackets], float), 1e-4, None)
        m /= m.sum()
        lo = np.array([br.lo for br in brackets], float)
        hi = np.array([br.hi for br in brackets], float)
        c = np.where(lo < -1e5, hi - 1.0, np.where(hi > 1e5, lo + 1.0, (lo + hi) / 2.0))
        mu_m = float((m * c).sum())
        if self.gate:
            g = nbm_dep(nbs, mu_m, float(self.gate.get("clip_f", 4.0)))
            a += self.gate["alpha"][0] * g
            b += self.gate["beta"][0] * g
            th += self.gate["theta"][0] * g
            ka += self.gate["kappa"][0] * g
        z = (c - mu_m) / self.w.get("z_scale_f", 2.0)
        s = a * np.log(m) + b * np.log(f) + th * z + ka * z * z
        p = np.exp(s - s.max())
        p /= p.sum()
        return {br.ticker: float(v) for br, v in zip(brackets, p)}
