"""Main loop. Same code path for paper and live; only the execution layer differs.

paper: quotes are simulated against the live public order books + trade tape (queue-aware fills), positions and
       P&L are tracked locally, nothing is sent to Kalshi. No API key needed.
live:  quotes are reconciled to real post-only orders; fills/positions come from the authenticated API.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import signal
import sys
import time
import collections
from collections import defaultdict
from typing import Optional

import yaml

from .. import exchange_hours
from ..data import nbm as nbm_mod
from ..data import obs as obs_mod
from ..data.recorder_feed import RecorderFeed
from ..execution.exec_log import ExecLogConfig, ExecLogger
from ..execution.live import LiveOrderManager
from ..execution.paper import PaperBroker
from ..execution.taker import booked_size
from ..kalshi.auth import KalshiAuth
from ..kalshi.rest import DEMO, PROD, PROD_ALT, KalshiError, KalshiREST
from ..models.brackets import bracket_from_market
from ..models.fair_value import book_to_quote, ladder_fair_value
from ..risk import same_weather as same_weather_mod
from ..risk import shutoff as shutoff_mod
from ..risk.limits import (RiskConfig, RiskState, ThresholdLadder, cash_in_use, event_worst_case, filter_orders,
                           pending_cash)
from ..risk.shutoff import ShutoffConfig
from ..risk.stop_guard import StopGuard, StopGuardConfig, usd
from ..store.db import Store
from ..timeutil import et_day, parse_ts, to_et
from .alerts import Alerter
from ..strategy import sleeve_a, sleeve_b, sleeve_h, taker
from ..strategy.lip import snapshot_value
from ..strategy.types import DesiredOrder, EventBook, Position

log = logging.getLogger("wxbot")
MON = {m: i for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], 1)}
STATIONS = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "stations.json")))
try:  # per-station NBM evening-before skill (Aug 20-Oct 1, 2026): bias (obs - NBM) and RMSE in F
    NBM_SKILL = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "nbm_station_skill.json")))
except Exception:
    NBM_SKILL = {}


def parse_event_date(event_ticker: str) -> tuple[Optional[dt.date], Optional[int]]:
    """KXHIGHNY-26OCT02 -> (2026-10-02, None); KXTEMPMIAH-26OCT0214 -> (2026-10-02, 14)."""
    try:
        tag = event_ticker.split("-")[1]
        y, mon, d = 2000 + int(tag[:2]), MON[tag[2:5]], int(tag[5:7])
        hr = int(tag[7:9]) if len(tag) >= 9 else None
        return dt.date(y, mon, d), hr
    except Exception:
        return None, None


def pd_ts(epoch_s: float):
    """Naive-UTC pandas Timestamp at the minute of an epoch time (the nowcast's index convention)."""
    import pandas as pd
    return pd.Timestamp(int(epoch_s // 60) * 60, unit="s")


def ts_of(s: Optional[str]) -> float:
    return parse_ts(s)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def peak_memory_mb() -> Optional[float]:
    """The process's peak resident memory in MB (status peak_memory_mb): what a cloud host would need to hold. macOS
    reports ru_maxrss in bytes, Linux in kilobytes."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(r / (1024 * 1024) if sys.platform == "darwin" else r / 1024, 1)
    except Exception:
        return None


def transient_failure(e: Exception) -> bool:
    """The network or Kalshi was unreachable (no connection, a timeout, a 5xx or 429 after the client's retries):
    nothing the bot can fix, as opposed to an error of its own."""
    import requests
    if isinstance(e, (requests.RequestException, ConnectionError, TimeoutError)):
        return True
    return isinstance(e, KalshiError) and (e.status >= 500 or e.status == 429)


class Bot:
    def code_version(self) -> str:
        """The code this process runs, from .deployed_commit ("<commit> (code as <code>; ...)"): the demo gate counts
        two days on the code GO_LIVE will run, so a code change starts the count again."""
        try:
            txt = open(os.path.join(getattr(self, "base", "."), ".deployed_commit")).read().strip()
        except OSError:
            return "unknown"
        m = re.search(r"code as ([0-9a-f]{6,40})", txt)
        return m.group(1) if m else (txt.split() or ["unknown"])[0]

    def hint(self, what: str) -> str:
        """Commands named in alerts and logs, pointed at this bot: on the demo bot its own config, status file and stop
        button, so following a DEMO alert can never clear a halt of the real-money bot (Oct 4 review)."""
        demo = (getattr(self, "cfg", None) or {}).get("env") == "demo"
        conf = " --config config/demo.yaml" if demo else ""
        return {"reset-halt": f".venv/bin/python -m wxbot reset-halt{conf}",
                "reset-h-breaker": f".venv/bin/python -m wxbot reset-h-breaker{conf}",
                "reset-shutoff": f".venv/bin/python -m wxbot reset-shutoff{conf}",
                "stop": "DEMO_STOP.command" if demo else "BACK_TO_PAPER.command",
                "check": (".venv/bin/python scripts/live_check.py data/demo/demo_status.json data/demo/demo.sqlite"
                          if demo else ".venv/bin/python scripts/live_check.py"),
                "smoke": "DEMO_START.command (its order test)" if demo else f".venv/bin/python -m wxbot smoke{conf}",
                }[what]

    @property
    def alert_name(self) -> str:
        """How alert titles name this bot: DEMO on Kalshi's demo exchange (its mode is live), else its mode."""
        return "DEMO" if (getattr(self, "cfg", None) or {}).get("env") == "demo" else getattr(self, "mode", "paper")

    def __init__(self, cfg: dict, base_dir: str = "."):
        self.cfg = cfg
        self.base = base_dir
        self.mode = cfg.get("mode", "paper")
        auth = KalshiAuth.from_env()
        if self.mode == "live" and auth is None:
            raise SystemExit("live mode needs KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH in the environment")
        env = cfg.get("env", "prod")
        base_url = DEMO if env == "demo" else os.environ.get("KALSHI_BASE_URL", PROD)
        self.rest = KalshiREST(base_url=base_url, auth=auth)
        # WebSocket stream, shadow only (wxbot/kalshi/ws.py): compared with the REST polls, never traded on. Kalshi
        # authenticates every stream connection, so it needs the API key (live or demo)
        wcfg = cfg.get("ws", {}) or {}
        self.ws = None
        if wcfg.get("enabled") and auth is not None:
            from ..kalshi.ws import WS_DEMO, WS_PROD, KalshiWS
            self.ws = KalshiWS(auth, url=wcfg.get("url") or (WS_DEMO if env == "demo" else WS_PROD))
        self.ws_cmp = {"n": 0, "agree": 0, "lead_s": collections.deque(maxlen=2000),
                       "fill_lead_s": collections.deque(maxlen=500), "stuck": 0, "examples": [],
                       "rest_top": {}, "mismatch": {}}
        os.makedirs(os.path.join(base_dir, os.path.dirname(cfg["db_path"]) or "."), exist_ok=True)
        self.store = Store(os.path.join(base_dir, cfg["db_path"]))
        self.status_path = os.path.join(base_dir, cfg.get("status_path", "data/status.json"))
        a = cfg.get("sleeve_a", {})
        self.cfg_a = sleeve_a.SleeveAConfig(**{k: v for k, v in a.items()
                                                if k in sleeve_a.SleeveAConfig.__dataclass_fields__})
        # Sleeve A's own open-risk cap inside the shared one (Oct 9; docs/results/r12_x_priority_check.txt): its
        # resting orders filled the shared $150 at every decision hour of the Aug-Sep replay and crowded out the
        # takers. None (unset) = off: Sleeve A sees only the shared caps, as before.
        cap_a = a.get("max_total_open_risk_usd")
        self.a_open_risk_cap = float(cap_a) if cap_a is not None else None
        # ... and its own cash cap (full-size stage, docs/results/r12_x_priority_check.txt "Full size": at full caps the
        # shared cash cap binds before open risk, and the maker alone holds $400-560 of the $560). None = off.
        cash_a = a.get("max_cash_usd")
        self.a_cash_cap = float(cash_a) if cash_a is not None else None
        self.series_a_own = set(a.get("series", [])) if a.get("enabled", True) else set()
        # same-weather budget (Oct 11, 2026; wxbot/risk/same_weather.py, docs/results/r14_same_weather.txt Q2): one
        # worst-case budget per group of cities whose forecast misses move together, per market date, shared by the two
        # daily makers. Off unless same_weather.enabled
        self.cfg_same_weather = same_weather_mod.SameWeatherConfig.from_cfg(cfg.get("same_weather"))
        self.same_weather_usage: dict = {}
        # Sleeve T: the pooled-fair-value maker on thin, newly listed books (own small caps; replaces Sleeve A there)
        t = cfg.get("sleeve_t", {}) or {}
        self.cfg_t = sleeve_a.SleeveAConfig(**{k: v for k, v in t.items()
                                                if k in sleeve_a.SleeveAConfig.__dataclass_fields__})
        self.series_t = set(t.get("series", [])) if t.get("enabled", False) else set()
        # Rounding guard (round 7, after KXHIGHTOKC-26OCT08; wxbot/models/round_guard.py, docs/results/
        # r7_round_guard.txt): Sleeves A and T leave alone the two brackets either side of a line while the whole-C
        # 5-minute readings leave the day's running max (low ladders: min) on both sides of it and no exact reading
        # (T-group) or 6-hour group has settled it. The 5-minute readings come from the observation archive
        # (scripts/obs5_archive.py, every 5 minutes on the server -> data/obs5.sqlite), the exact ones from the
        # METAR feed the bot already polls and the archive. Off unless round_guard.enabled.
        g = cfg.get("round_guard", {}) or {}
        self.round_guard_on = bool(g.get("enabled", False))
        self.round_guard_sleeves = set(g.get("sleeves", ["A", "T"]))   # which of the two daily makers it applies to
        self.obs5_path = os.path.join(base_dir, g.get("obs5_db", "data/obs5.sqlite"))
        self.obs5_every_s = float(g.get("read_every_s", 60))
        self.obs5_stale_s = 60 * float(g.get("stale_minutes", 45))
        self.obs5: dict = {}            # icao -> [round_guard.Reading] of the last 36 hours, from data/obs5.sqlite
        self.obs5_newest = 0.0          # time of the newest reading in the archive (staleness warning)
        self.obs5_version = 0
        self.guard_last: dict = {}      # event -> tickers the guard blocked at its last decision (log on change)
        self.guard_cache: dict = {}     # event -> (data version, tickers)
        self.risk_t = RiskConfig(max_loss_per_bracket_usd=float(t.get("max_loss_per_bracket_usd", 15)),
                                 max_loss_per_event_usd=float(t.get("max_loss_per_event_usd", 25)),
                                 max_total_open_risk_usd=float(t.get("max_total_open_risk_usd", 100)))
        self.cash_cap_t = float(t.get("max_cash_usd", 100))
        b = cfg.get("sleeve_b", {})
        self.cfg_b = sleeve_b.SleeveBConfig(**{k: v for k, v in b.items()
                                                if k in sleeve_b.SleeveBConfig.__dataclass_fields__})
        self.b_shadow = bool(b.get("shadow", True))
        # Sleeve H: tail maker on the hourly Weather Index markets, shadow only (separate broker and table)
        hcfg = cfg.get("sleeve_h") or {}
        self.cfg_h = sleeve_h.SleeveHConfig.from_cfg(hcfg)
        self.h_on = bool(hcfg) and bool(hcfg.get("shadow", True))
        # Sleeve H breaker (real orders only): its state survives restarts; once tripped, real hourly orders stay off
        # until cleared by hand (python -m wxbot reset-h-breaker). The shadow keeps logging either way.
        self.load_h_breaker()
        r = cfg.get("risk", {})
        self.risk_cfg = RiskConfig(**{k: v for k, v in r.items() if k in RiskConfig.__dataclass_fields__})
        self.risk = RiskState(self.risk_cfg)
        # guards on the account stops (Oct 11, 2026; wxbot/risk/stop_guard.py), each off unless its risk key is set: a
        # daily halt lifted the same day once the day's settled result recovers (daily_resume_margin_usd), stops that
        # halt only after their condition has held a while (stop_confirm_s, drawdown_confirm_s), and the drawdown
        # stop's high point taken from start-of-day equity only (peak_from: start_of_day)
        self.stop_guard = StopGuard(StopGuardConfig.from_cfg(r))
        self.restore_halt()   # a hard or daily halt survives restarts (Oct 4 review, finding 2.6)
        # per-sleeve automatic shut-off (Oct 9, holdout draft A12; wxbot/risk/shutoff.py), off unless shutoff.enabled:
        # a sleeve switched off before a restart stays off from the first step (kv shutoff:<sleeve>)
        self.cfg_shutoff = ShutoffConfig.from_cfg(cfg.get("shutoff"))
        self.shutoff_state: dict = {}
        self.shutoff_off = self.load_shutoff()
        self.max_err = int(r.get("max_api_errors_per_min", 15))
        # account reads older than this block new orders (live): orders, balance and positions must all be fresh
        self.account_stale_s = float(r.get("account_stale_seconds", 30))
        self.stale_book_s = float(r.get("stale_book_seconds", 90))
        self.stale_weather_s = 60 * float(r.get("stale_weather_minutes", 180))
        self.series_a = list(a.get("series", [])) if a.get("enabled", True) else []
        self.series_a += [x for x in sorted(self.series_t) if x not in self.series_a]
        # watch-only series: priced and logged (ML snapshots, Sleeve X shadow) but never quoted by Sleeve A or T
        xw = cfg.get("taker_shadow", {}) or {}
        self.watch_only = set(xw.get("extra_series", [])) if xw.get("enabled") else set()
        self.series_a += [x for x in sorted(self.watch_only) if x not in self.series_a]
        self.w_nbm = float(a.get("w_nbm", 0.2))
        self.stations_needed = sorted({STATIONS[s]["icao"] for s in self.series_a if s in STATIONS})
        self.nbm = nbm_mod.NBMCache(set(self.stations_needed))
        self.paper = PaperBroker(late_fills=True) if self.mode == "paper" else None
        self.paper_b = PaperBroker(late_fills=True)  # shadow-only broker for Sleeve B (never sends orders)
        self.paper_h = PaperBroker(late_fills=True)  # shadow-only broker for Sleeve H (never sends orders)
        self.h_held: dict = defaultdict(list)  # hourly event -> [(ticker, side, price_cents, count)] filled, unsettled
        for _ts, ev_, tk_, side_, px_, n_ in self.store.hourly_open("H"):
            self.h_held[ev_].append((tk_, side_, int(px_), float(n_)))
        # Sleeve H2: the same rule, quoting a tail side only when the hourly nowcast agrees (shadow only)
        self.h2_on = self.h_on and self.cfg_h.nowcast
        self.paper_h2 = PaperBroker(late_fills=True)
        self.h2_held: dict = defaultdict(list)
        self.index_feed = None
        self.nowcaster = None
        if self.h_on and env == "prod":
            # the published index is recorded whenever the hourly shadow runs (H2 reads it; the store keeps it)
            try:
                from ..data.index_feed import IndexFeed
                from ..ml.nowcast import SERIES_CITY
                cities = sorted({SERIES_CITY[x] for x in self.cfg_h.series if x in SERIES_CITY})
                # Kalshi also publishes an index for cities without hourly markets yet (Houston): record it too, so a
                # nowcast has history the day markets list
                cities += [c for c in (hcfg.get("record_extra_index") or ["houston"]) if c not in cities]
                self.index_feed = IndexFeed(self.rest, self.store, cities)
            except Exception as e:
                log.warning("index feed unavailable: %s", e)
        if self.h2_on:
            for _ts, ev_, tk_, side_, px_, n_ in self.store.hourly_open("H2"):
                self.h2_held[ev_].append((tk_, side_, int(px_), float(n_)))
            try:
                from ..ml.nowcast import Nowcaster
                if self.index_feed is None:
                    raise RuntimeError("no index feed")
                self.nowcaster = Nowcaster()
                log.info("Sleeve H2 nowcast on (%s engine, models: %s)", self.nowcaster.engine,
                         ", ".join(sorted(self.nowcaster.models)) or "none")
            except Exception as e:
                log.warning("Sleeve H2 nowcast unavailable: %s", e)
                self.h2_on = False
        self.positions_b: dict[str, Position] = {}
        self.b_realized = 0.0
        # Our own copy of the public tape of settled events (results, 1-minute candles, trades) for the Dec 1 holdout
        # replays; Kalshi moves old data to /historical endpoints after about two months (wxbot/data/tape_archive.py)
        tcfg = cfg.get("tape_archive", {}) or {}
        self.tape = None
        self.tape_every_s = float(tcfg.get("every_s", 900))
        if tcfg.get("enabled", True) and env == "prod":
            from ..data.tape_archive import TapeArchiver
            since = dt.datetime.fromisoformat(str(tcfg.get("since", "2026-10-01"))).replace(
                tzinfo=dt.timezone.utc).timestamp()
            self.tape = TapeArchiver(os.path.join(os.path.dirname(self.status_path), "tape"),
                                     hourly_series=list(self.cfg_h.series) if self.h_on else [],
                                     daily_series=list(self.series_a) + [x for x in tcfg.get("extra_daily_series", [])
                                                                          if x not in self.series_a],
                                     since_ts=since,
                                     gap_s=float(tcfg.get("gap_s", 1.0)))
        rec_dir = cfg.get("recorder_dir")
        self.feed = RecorderFeed(rec_dir) if (self.mode == "paper" and rec_dir and os.path.isdir(rec_dir)) else None
        if self.feed:
            self.feed.skip_to_end()
        ex = cfg.get("execution", {}) or {}
        # patience: daily-maker orders do not chase a more aggressive price more than once per patient_reprice_s
        # (cautious moves are immediate); 0 = re-price on every change
        self.patient_s = float(ex.get("patient_reprice_s", 0) or 0)
        self.patient_sleeves = tuple(ex.get("patient_sleeves", ["A", "T"]))
        # dead-man expiry per sleeve (an order's sleeve is encoded in its client id): hourly orders 2 min, daily 5 min
        self.live = LiveOrderManager(self.rest, ttl_s=float(ex.get("order_ttl_s", 1200)), patient_s=self.patient_s,
                                     patient_sleeves=self.patient_sleeves,
                                     ttl_by_sleeve=ex.get("ttl_by_sleeve") or None,
                                     post_only_cooldown_s=float(ex.get("post_only_cooldown_s", 10))
                                     ) if self.mode == "live" else None
        self.events: dict[str, EventBook] = {}
        self.hourly: dict[str, EventBook] = {}
        self.metars: dict = {}
        self.index_point: Optional[dict] = None
        self.lip_programs: dict = {}
        self.positions: dict[str, Position] = {}
        self.fair_cache: dict[str, float] = {}
        self.last_trade_ts = int(time.time()) - 30
        self.last_book_ts = 0.0   # last book refresh (any market): timing stats only
        self.book_ts: dict[str, float] = {}   # ticker -> when its book was last actually received (freshness gates)
        self.acct_ts = {"sync": 0.0, "balance": 0.0, "positions": 0.0}   # last successful account reads (live)
        self.eq_readings: list = []          # recent settled-equity readings (stops use their median)
        self.eq_ts = 0.0                     # when the last one was taken
        self.recent_takes: list = []         # live taker fills not yet visible in a (lagged) positions read
        # does Kalshi's account-data timestamp move on its own (an exchange-wide clock: what an unsigned read shows,
        # Oct 4) or only with account activity? `python -m wxbot smoke` measures it with the real key
        self.as_of_moves_idle = (self.store.kv_get("api_observations") or {}).get("user_data_ts_moves_idle", True)
        self.next_due = defaultdict(float)
        self.running = True
        self.started = time.time()
        self.counters = defaultdict(int)
        self.lip_est_usd = 0.0
        self.lip_samples = 0
        self.ev_tickers: dict[str, list[str]] = {}  # event -> all its market tickers (kept after close)
        self.ev_tickers_miss: dict[str, float] = {}
        # the demo bot (config/demo.yaml, mode live on Kalshi's demo exchange) runs beside the main bot with the same
        # .env: its alerts say DEMO (alert_name), so a demo halt never reads like a real-money one, and it sends no
        # heartbeat, so it cannot keep an outside "is the bot alive" check quiet while the main bot is down
        self.alerts = Alerter(cfg.get("alerts"), demo=cfg.get("env") == "demo")
        self.netting = bool(r.get("assume_netting", False)) if self.mode == "paper" else False
        self.avail_balance: Optional[float] = None
        self.step_error_ts: list[float] = []
        # ML fair value: Tier-1 v2 forecaster pooled with the market (research/ml, docs/ML_RESEARCH.md)
        ml_cfg = cfg.get("ml", {}) or {}
        self.ml = self.pool = None
        self.ml_thread = None
        self.ml_key = None
        self.ml_logged: set = set()
        if ml_cfg.get("enabled"):
            try:
                from ..ml.live import Pool, Tier1Live
                self.ml = Tier1Live(os.path.join(base_dir, ml_cfg.get("raw_dir", "data/ml_raw")), self.stations_needed,
                                    model=ml_cfg.get("model", "tier1_v2"))
                # a named model's pool: its weekly refit's file in data/ once it exists, else the shipped one
                pool_live = ml_cfg.get("pool_live_path")
                pool_live = os.path.join(base_dir, pool_live) if pool_live else None
                if pool_live and os.path.exists(pool_live):
                    self.pool = Pool(pool_live)
                elif ml_cfg.get("pool_file"):
                    from ..ml.live import ART as _ART
                    self.pool = Pool(os.path.join(_ART, ml_cfg["pool_file"]))
                else:
                    self.pool = Pool()
                self.ml_sha = self.ml.meta.get("sha256", "")[:12]
                log.info("ML fair value on (model %s, trained through %s, engine %s)", self.ml_sha,
                         self.ml.meta.get("last_day"), getattr(self.ml, "engine", "?"))
            except Exception as e:  # never let the research model stop the bot
                log.warning("ML fair value disabled: %s", e)
                self.store.log("ERROR", f"ML fair value disabled (market-anchored prices only): {e}")
                self.ml = self.pool = None
        # weekly pool refit (pre-registered: scheduled refits allowed, design frozen): Mondays after 06:00 UTC
        self.refit_cfg = ml_cfg.get("weekly_refit", {}) or {}
        self.refit_thread = None
        # Sleeve X, shadow only: logs the model-taker trades the pre-registered rule would make; never sends orders
        x = cfg.get("taker_shadow", {}) or {}
        self.cfg_x = taker.TakerConfig(**{k: v for k, v in x.items() if k in taker.TakerConfig.__dataclass_fields__})
        self.cfg_x_raw = x
        self.x_seen: set = self.store.taker_shadow_keys() if self.cfg_x.enabled else set()
        # Sleeve R, shadow only: the model taker on the daily rain markets (wxbot/ml/rain.py, strategy/rain_taker.py)
        self.cfg_r = cfg.get("rain_shadow", {}) or {}
        self.rain = None
        self.rain_seen: set = set()
        self.rain_done: set = set()
        self.rain_pending: list = []
        self.rain_thread = None
        if self.cfg_r.get("enabled") and self.ml is not None and env == "prod":
            try:
                from ..ml.rain import RainModel
                self.rain = RainModel()
                self.rain_seen = self.store.rain_shadow_keys()
                log.info("Sleeve R rain shadow on (%s engine, model %s, %d cities)", self.rain.engine,
                         self.rain.meta["sha256"][:12], len(self.rain.meta["cities"]))
            except Exception as e:  # never let a research sleeve stop the bot
                log.warning("Sleeve R rain shadow unavailable: %s", e)
                self.rain = None
        # R2 (secondary variant, shadow only): version-2 rain model on all 30 cities; the nine extra climate sites get
        # their own raw folder (MOS, METARs, climate reports; no ECMWF)
        self.rain2 = None
        self.rain2_seen: set = set()
        self.rain_raw_extra = None
        if self.rain is not None and self.cfg_r.get("v2", True):
            try:
                from ..ml.live import LiveRaw
                from ..ml.rain import RAIN_CITY_EXTRA, RainModel
                self.rain2 = RainModel(suffix="_v2")
                self.rain2_seen = self.store.rain_shadow_keys("rain_shadow_v2")
                self.rain_raw_extra = LiveRaw(os.path.join(os.path.dirname(self.ml.raw.root), "ml_raw_rain"),
                                              sorted(set(RAIN_CITY_EXTRA.values())), ecmwf=False)
                log.info("Sleeve R2 rain shadow on (model %s, %d cities)", self.rain2.meta["sha256"][:12],
                         len(self.rain2.meta["cities"]))
            except Exception as e:
                log.warning("Sleeve R2 rain shadow unavailable: %s", e)
                self.rain2 = None
        # execution log (outside review, Oct 8; wxbot/execution/exec_log.py), off unless exec_log.enabled: the book
        # at each Sleeve X / R decision and at each X / E / R taker order, re-read at +1, +5 and +15 minutes, with
        # the order's answer. Its rows are kept in memory on the order path and written after the step, and the
        # re-reads run then too, through the read limiter without waiting on it: neither delays an order
        self.cfg_exec = ExecLogConfig.from_cfg(cfg.get("exec_log"))
        self.exec_log = ExecLogger(self.rest, self.cfg_exec, mode=self.mode) if self.cfg_exec.enabled else None

    # ------------------------------------------------------------ helpers
    def due(self, key: str, every: float) -> bool:
        now = time.time()
        if now >= self.next_due[key]:
            self.next_due[key] = now + every
            return True
        return False

    def event_tickers(self, ev: EventBook) -> list[str]:
        return [m["ticker"] for m in ev.markets]

    def cached_book(self, ticker: str) -> tuple:
        """(raw order book, when it was received) from the loop's last book refresh, or (None, None)."""
        e = ticker.rsplit("-", 1)[0]
        ev = (getattr(self, "events", None) or {}).get(e) or (getattr(self, "hourly", None) or {}).get(e)
        b = ev.books.get(ticker) if ev is not None else None
        return (b, (getattr(self, "book_ts", None) or {}).get(ticker)) if b is not None else (None, None)

    def tickers_for_event(self, event: str):
        """An event's outcome structure, including after it closed (needed for worst-case risk on held positions):
        a ThresholdLadder for nested 'above k' ladders (the hourly index events), else the list of its market
        tickers (exclusive brackets). Cached in memory and in the store (kv ev_struct:<event>), so a restart rebuilds
        it without guessing (Oct 4 review, finding 2.5: after a restart a closed hour's NO positions were valued as
        exclusive brackets, half their real worst case). None while unknown (lookup failed, retried every minute):
        the risk check then counts the event's full cost, and new orders on it are refused."""
        if event in self.ev_tickers:
            return self.ev_tickers[event]
        saved = self.store.kv_get(f"ev_struct:{event}")
        if saved:
            spec = ThresholdLadder(tuple((float(k), t) for k, t in saved["strikes"])) if saved.get("strikes") \
                else list(saved.get("tickers") or [])
            if spec:
                self.ev_tickers[event] = spec
                return spec
        if time.time() - self.ev_tickers_miss.get(event, 0) < 60:
            return None
        try:
            ms = self.rest.markets(event_ticker=event, limit=100)
            if ms:
                return self.remember_structure(event, ms)
        except Exception as e:
            log.warning("markets lookup failed for %s: %s", event, e)
        self.ev_tickers_miss[event] = time.time()
        return None

    def remember_structure(self, event: str, markets: list):
        """Store an event's outcome structure (see tickers_for_event) and return it."""
        lad = ThresholdLadder.from_markets(markets)
        nested = len(lad) >= 1 and len(lad) == len(markets)
        spec = lad if nested else [m["ticker"] for m in markets]
        if self.ev_tickers.get(event) != spec:
            self.ev_tickers[event] = spec
            self.store.kv_set(f"ev_struct:{event}", {"strikes": [list(x) for x in lad.strikes]} if nested
                              else {"tickers": spec})
        return spec

    def book_fresh(self, ev: EventBook, now: float, max_age: float) -> bool:
        """Every market of the event had its own book received within max_age seconds (a market missing from a
        refresh keeps its old quote, which must not pass for fresh: Oct 4 review, finding 2.7)."""
        bts = getattr(self, "book_ts", {})
        return bool(ev.markets) and all(now - bts.get(m["ticker"], 0.0) <= max_age for m in ev.markets)

    def account_fresh(self, now: float) -> bool:
        """Live: resting orders, balance and positions were all read successfully within account_stale_s. Without
        that the bot does not know its exposure or cash, so it places nothing new (cancels still go out)."""
        if self.mode != "live":
            return True
        acct = getattr(self, "acct_ts", {}) or {}
        lim = getattr(self, "account_stale_s", 30.0)
        if not all(now - acct.get(k, 0.0) <= lim for k in ("sync", "balance", "positions")):
            return False
        if getattr(self, "as_of_moves_idle", True) is False:
            return True   # the smoke test found the timestamp only moves with account activity: read times decide
        as_of = getattr(getattr(self, "live", None), "last_as_of", 0.0) or 0.0
        return not as_of or now - as_of <= lim   # reads that succeed but reflect stalled account data are not fresh

    # ----------------------------------------------------------- refreshers
    def refresh_universe(self):
        new = {}
        for s in self.series_a:
            st = STATIONS.get(s)
            if not st:
                continue
            try:
                evs = self.rest.events(s, status="open", nested=True, limit=10)
            except Exception as e:
                log.warning("events fetch failed for %s: %s (keeping previous)", s, e)
                for k, old in self.events.items():
                    if old.series == s and old.close_ts > time.time():
                        new[k] = old
                continue
            for e in evs:
                ms = [m for m in e.get("markets", []) if m.get("status") in ("active", "open")]
                if not ms or self.unsupported_event(e):
                    continue
                d, _ = parse_event_date(e["event_ticker"])
                try:
                    brs = [bracket_from_market(m) for m in ms]
                except ValueError:
                    continue
                old = self.events.get(e["event_ticker"])
                ev = EventBook(e["event_ticker"], s, st["kind"], st["icao"], d, ts_of(ms[0].get("close_time")),
                               ms, brs, old.quotes if old else {}, old.books if old else {})
                new[e["event_ticker"]] = ev
                self.remember_structure(e["event_ticker"], e.get("markets", []))
        self.events = new
        self.counters["universe"] += 1

    def unsupported_event(self, e: dict) -> Optional[str]:
        """Why an event's markets do not fit what the bot assumes, or None. The pricing, risk and order code assume
        whole-cent prices (price_level_structure linear_cent), exchange shard 0 (orders carry exchange_index 0) and
        a payout of exactly 0 or $1 (settlement_bounds_type default). Kalshi has been moving products to sub-cent
        ticks, other shards and floor settlements (changelog, Aug-Oct 2026); weather markets were all standard on
        Oct 4. An event that is not is skipped, with a warning, instead of being mispriced."""
        why = None
        if int(e.get("exchange_index") or 0) != 0:
            why = f"exchange shard {e.get('exchange_index')}"
        for m in e.get("markets", []):
            if why:
                break
            whole_cents = all(abs(float(r.get("step") or 0) - 0.01) < 1e-9 for r in m.get("price_ranges") or []) \
                and bool(m.get("price_ranges"))
            if (m.get("price_level_structure") or "linear_cent") != "linear_cent" and not whole_cents:
                why = f"price structure {m.get('price_level_structure')}"
            elif int(m.get("exchange_index") or 0) != 0:
                why = f"exchange shard {m.get('exchange_index')}"
            elif (m.get("settlement_bounds_type") or "default") != "default":
                why = f"settlement bounds {m.get('settlement_bounds_type')}"
        if why and self.due(f"unsupported:{e.get('event_ticker')}", 6 * 3600):
            msg = f"{e.get('event_ticker')} skipped: {why} (the bot assumes whole cents, shard 0, a 0/$1 payout)"
            log.warning(msg)
            self.store.log("WARN", msg)
        return why

    def refresh_hourly(self):
        """Open hourly Weather Index events: Miami for Sleeve B (+ its LIP programs), the Sleeve H series for the
        Sleeve H shadow. Called every minute near the top of the hour."""
        want_b = self.cfg_b.enabled or self.b_shadow
        series = (["KXTEMPMIAH"] if want_b else []) + [x for x in (self.cfg_h.series if self.h_on else ())
                                                         if not (want_b and x == "KXTEMPMIAH")]
        if not series:
            return
        try:
            hourly = {}
            for ser in series:
                for e in self.rest.events(ser, status="open", nested=True, limit=3):
                    ms = [m for m in e.get("markets", []) if m.get("status") in ("active", "open")]
                    if ms and not self.unsupported_event(e):
                        d, _ = parse_event_date(e["event_ticker"])
                        old = self.hourly.get(e["event_ticker"])
                        hourly[e["event_ticker"]] = EventBook(e["event_ticker"], ser, "hourly", ser + "-IDX", d,
                                                              ts_of(ms[0].get("close_time")), ms, [],
                                                              old.quotes if old else {}, old.books if old else {})
                        self.remember_structure(e["event_ticker"], e.get("markets", []))
            self.hourly = hourly
            if want_b:
                progs = self.rest.incentive_programs("active")
                now = time.time()
                self.lip_programs = {p["market_ticker"]: p for p in progs
                                     if p["market_ticker"].startswith("KXTEMPMIAH")
                                     and ts_of(p["start_date"]) <= now + 3600 and ts_of(p["end_date"]) >= now}
        except Exception as e:
            log.warning("hourly refresh failed: %s", e)

    def refresh_books(self):
        t0 = time.time()   # the books are at least this fresh
        tickers = [m["ticker"] for ev in self.events.values() for m in ev.markets]
        tickers += [m["ticker"] for ev in self.hourly.values() for m in ev.markets]
        if self.feed:
            books = self.feed.poll_books()
            missing = [t for t in tickers if t not in books]
            if missing:  # recorder universe may lag; fetch the gaps directly
                books.update(self.rest.orderbooks(missing))
        else:
            books = self.rest.orderbooks(tickers) if tickers else {}
        bts = getattr(self, "book_ts", None)
        if bts is None:
            bts = self.book_ts = {}
        for ev in list(self.events.values()) + list(self.hourly.values()):
            for m in ev.markets:
                b = books.get(m["ticker"])
                if b is None:
                    continue   # its old quote stays, and goes stale (book_fresh)
                bts[m["ticker"]] = t0
                ev.books[m["ticker"]] = b
                ev.quotes[m["ticker"]] = book_to_quote(b)
                if self.paper:
                    self.paper.on_book(m["ticker"], b, t0)
                if self.h_on and ev.kind == "hourly":
                    self.paper_h.on_book(m["ticker"], b, t0)
                    if self.h2_on:
                        self.paper_h2.on_book(m["ticker"], b, t0)
        self.last_book_ts = time.time()
        if getattr(self, "ws", None) is not None:
            # the stream is a shadow: nothing it does may fail the book refresh (round-9 audit)
            self.side_task("ws_markets", self.ws.set_markets, tickers)
            self.side_task("ws_compare", self.ws_compare, books, t0)

    def ws_fill_compare(self, fills: list, now: float):
        """Shadow: did the stream deliver each fill the poll found, and how much earlier or later?"""
        c = self.ws_cmp
        polled = c.setdefault("polled_fills", {})        # trade id -> when the poll had it, stream not yet
        seen = {e[2].get("trade_id"): e[0] for e in self.ws.events_since(now - 900, kinds={"fill"})}
        for raw in getattr(self.live, "last_new_raw", []) if fills else []:
            tid = raw.get("trade_id")
            if tid in seen:      # positive: the stream delivered it before this poll
                c["fill_lead_s"].append(round(time.time() - seen[tid], 3))
            elif tid:
                polled[tid] = time.time()
        for tid in [k for k in polled if k in seen]:      # negative: the stream delivered it after the poll
            c["fill_lead_s"].append(round(polled.pop(tid) - seen[tid], 3))
        for tid in [k for k, ts in polled.items() if now - ts > 900]:
            polled.pop(tid, None)
            c["fill_missing"] = c.get("fill_missing", 0) + 1   # the stream never delivered it

    def ws_compare(self, books: dict, t0: float):
        """Shadow check of the stream against this REST poll, market by market (top of book: best YES bid and best
        NO bid). Records how often they agree; for each top the poll finds new, how much earlier (positive) or later
        (negative, once the stream catches up) the stream had it; and tops that disagree for 15 s while the stream's
        top does not move (a book the stream got wrong)."""
        if not self.ws.healthy():
            return
        c = self.ws_cmp
        tops = self.ws.tops()
        late = c.setdefault("rest_first", {})          # ticker -> (top the poll saw first, when)
        for t, b in books.items():
            if t not in tops:
                continue
            q = book_to_quote(b)
            rest = (q.yes_bid, None if q.yes_ask is None else 100 - q.yes_ask)
            (wtop, wts) = tops[t]
            c["n"] += 1
            prev = c["rest_top"].get(t)
            c["rest_top"][t] = rest
            pend = late.get(t)
            if pend is not None and wtop == pend[0] and wts >= pend[1]:
                c["lead_s"].append(round(pend[1] - wts, 3))   # the stream caught up after the poll: negative
                late.pop(t, None)
            if wtop == rest:
                c["agree"] += 1
                c["mismatch"].pop(t, None)
                if prev is not None and prev != rest and wts and pend is None:
                    c["lead_s"].append(round(t0 - wts, 3))   # positive: the stream had it before the poll asked
                continue
            if prev is not None and prev != rest:
                late[t] = (rest, t0)                          # the poll saw a new top the stream does not show yet
            first = c["mismatch"].get(t)
            if first is None or first[1] != wts:
                c["mismatch"][t] = (t0, wts)                  # a new episode, or the stream's top moved: restart
                continue
            if t0 - first[0] >= 15:
                c["stuck"] += 1
                c["mismatch"][t] = (t0, wts)
                if len(c["examples"]) < 20:
                    c["examples"].append({"ticker": t, "rest": rest, "ws": wtop, "ts": int(t0)})
                if self.due("ws_stuck_note", 600):
                    self.store.log("WARN", f"websocket shadow: {t} REST top {rest} vs stream {wtop} for 15 s+")
        for t in [t for t, (_, ts) in late.items() if t0 - ts > 120]:
            late.pop(t, None)                                 # never caught up: counted by the stuck check

    def ws_status(self) -> Optional[dict]:
        if getattr(self, "ws", None) is None:
            return None
        c = self.ws_cmp
        q = lambda d, f: sorted(d)[min(len(d) - 1, int(f * len(d)))] if d else None
        return {**self.ws.stats(), "compared": c["n"], "agree_pct": round(100 * c["agree"] / c["n"], 2) if c["n"] else None,
                "lead_s_p50": q(c["lead_s"], 0.5), "lead_s_p90": q(c["lead_s"], 0.9), "lead_n": len(c["lead_s"]),
                "fill_lead_s_p50": q(c["fill_lead_s"], 0.5), "fill_lead_n": len(c["fill_lead_s"]),
                "fills_missing_from_stream": c.get("fill_missing", 0),
                "stuck_mismatches": c["stuck"], "examples": c["examples"][-5:]}

    def refresh_trades_paper(self):
        if self.feed:
            trades = self.feed.poll_trades()
        else:
            t1 = int(time.time()) - 2
            trades = self.rest.trades(min_ts=self.last_trade_ts, max_ts=t1, max_pages=60)
            self.last_trade_ts = t1
        watched = {m["ticker"] for ev in self.events.values() for m in ev.markets}
        watched |= {m["ticker"] for ev in self.hourly.values() for m in ev.markets}
        trades = [t for t in trades if t.get("ticker") in watched]
        trades.sort(key=lambda t: t.get("created_time", ""))
        for t in trades:
            if self.paper is not None:
                for f in self.paper.on_trade(t):
                    self.apply_fill(f)
            if self.h_on:
                for f in self.paper_h.on_trade(t):
                    self.record_h_fill(f)
            if self.h2_on:
                for f in self.paper_h2.on_trade(t):
                    self.record_h_fill(f, "H2")
            for f in self.paper_b.on_trade(t):
                p = self.positions_b.setdefault(f.ticker, Position(f.ticker, f.event))
                if f.side == "bid":
                    p.yes_qty += f.count
                    p.yes_cost += f.count * f.price_cents / 100.0
                else:
                    p.no_qty += f.count
                    p.no_cost += f.count * (100 - f.price_cents) / 100.0
                self.counters["b_fills"] += 1

    def apply_fill(self, f):
        p = self.positions.setdefault(f.ticker, Position(f.ticker, f.event))
        if f.side == "bid":
            p.yes_qty += f.count
            p.yes_cost += f.count * f.price_cents / 100.0
        else:
            p.no_qty += f.count
            p.no_cost += f.count * (100 - f.price_cents) / 100.0
        p.fees += f.fee
        self.store.add_fill(self.mode, f, self.fair_cache.get(f.ticker), tag=self.model_tag())
        self.counters["fills"] += 1
        log.info("FILL %s %s %s @%sc x%.0f (%s)", self.mode, f.ticker, f.side, f.price_cents, f.count, f.sleeve)

    def model_tag(self) -> str:
        """Forecaster hash and pool window behind the current prices, stored with every fill."""
        sha = getattr(self, "ml_sha", "") or ("off" if getattr(self, "ml", None) is None else "?")
        pool = getattr(self, "pool", None)
        last = (getattr(pool, "w", None) or {}).get("last_day", "") if pool is not None else ""
        return f"{sha}|pool {last}" if last else sha

    def sleeve_pnl_today(self) -> dict:
        """Realized P&L per sleeve for markets that settled since midnight ET, from the bot's own fills (cached for
        a minute)."""
        now = time.time()
        c = getattr(self, "_sleeve_pnl_cache", None)
        if c and now - c[0] < 60:
            return c[1]
        start = to_et(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        out = defaultdict(float)
        for sl, side, pc, n, fee, res in self.store.c.execute(
                "SELECT f.sleeve, f.side, f.price_cents, f.count, f.fee, s.result FROM fills f JOIN settlements s "
                "ON f.ticker = s.ticker WHERE f.mode = ? AND s.settled_ts >= ? AND s.result IN ('yes', 'no')",
                (self.mode, start)):
            won = (res == "yes") == (side == "bid")
            cost = pc / 100.0 if side == "bid" else (100 - pc) / 100.0
            out[sl or "?"] += n * ((1.0 if won else 0.0) - cost) - (fee or 0.0)
        out = dict(out)
        self._sleeve_pnl_cache = (now, out)
        return out

    def stopped_sleeves(self) -> set:
        """Sleeves whose realized loss today reached their own daily stop (risk.sleeve_daily_stop_usd): no new
        orders from them until the next ET day; the other sleeves keep running."""
        lim = (self.cfg.get("risk", {}) or {}).get("sleeve_daily_stop_usd") or {}
        if not lim:
            return set()
        day = et_day().isoformat()
        pnl = self.sleeve_pnl_today()
        out = set()
        for sl, cap in lim.items():
            if pnl.get(sl, 0.0) <= -float(cap):
                out.add(sl)
                key = f"sleeve_stop:{sl}:{day}"
                if not self.store.kv_get(key):
                    self.store.kv_set(key, {"pnl": round(pnl[sl], 2), "ts": time.time()})
                    msg = (f"Sleeve {sl} lost ${-pnl[sl]:.2f} today (stop ${float(cap):.0f}): no new {sl} orders "
                           "until tomorrow; the other sleeves keep running")
                    self.store.log("ERROR", msg)
                    self.alerts.send(f"sleeve_stop:{sl}",
                                     f"wxbot {self.alert_name}: sleeve {sl} stopped for the day", msg,
                                     priority="high", tags="warning")
        return out

    # ------------------------------------------------------- shut-off (A12)
    def load_shutoff(self) -> set:
        """Sleeves their shut-off rule switched off (kv shutoff:<sleeve>, latched until reset-shutoff): blocked from
        the first step after a restart. Empty unless shutoff.enabled."""
        c = getattr(self, "cfg_shutoff", None)
        if c is None or not c.enabled:
            return set()
        out = set()
        for s in c.sleeves:
            st = self.store.kv_get(f"shutoff:{s}") or {}
            if st.get("tripped"):
                out.add(s)
                self.store.log("WARN", f"Sleeve {s} is switched off by its shut-off rule (statistic ${st.get('S')} on "
                                       f"{st.get('day')}): no new {s} orders; {self.hint('reset-shutoff')} {s} turns it "
                                       "back on")
        return out

    def shutoff_blocked(self) -> set:
        """Sleeves that open no new positions because their shut-off rule switched them off (empty when off)."""
        c = getattr(self, "cfg_shutoff", None)
        if c is None or not c.enabled:
            return set()
        return set(getattr(self, "shutoff_off", None) or ())

    def check_shutoff(self, now: Optional[float] = None):
        """Each watched sleeve's statistic (wxbot/risk/shutoff.py) from the bot's own fills and settlements, after
        each settlement pass. When it reaches h the sleeve is switched off: latched in kv shutoff:<sleeve>, an ERROR
        line, an alert, the status file; its resting orders are cancelled by the next order update (no longer
        wanted) and its taker signals are logged but not sent; open positions are held. The count starts at the
        sleeve's `since` (config; else when the bot first checked it) or at the last reset-shutoff, the later one."""
        c = getattr(self, "cfg_shutoff", None)
        if c is None or not c.enabled:
            return
        now = time.time() if now is None else now
        status, off = {}, set()
        for s, rule in sorted(c.sleeves.items()):
            key = f"shutoff:{s}"
            st = self.store.kv_get(key) or {}
            if not st.get("first_ts"):
                st = {**st, "first_ts": now}
                self.store.kv_set(key, st)
            since = rule.since if rule.since is not None else float(st["first_ts"])
            if st.get("reset_ts"):
                since = max(since, float(st["reset_ts"]))
            ev = shutoff_mod.evaluate(self.store.c, self.mode, s, rule, since, now, c.grace_h)
            if ev["crossed_day"] and not st.get("tripped"):
                st = {**st, "tripped": True, "day": ev["crossed_day"], "ts": now, "S": ev["S_cross"],
                      "since_ts": since}
                self.store.kv_set(key, st)
                msg = (f"Sleeve {s} switched off by its shut-off rule: its running shortfall reached "
                       f"${ev['S_cross']:.2f} (limit ${rule.h:.2f}, k ${rule.k:.2f} a day) on {ev['crossed_day']}, "
                       f"{ev['days']} days from {ev['first_day']} (realized ${ev['pnl_usd']:+.2f}). No new {s} "
                       f"positions; open ones are held to settlement. After a review: {self.hint('reset-shutoff')} {s}")
                self.store.log("ERROR", msg)
                log.error(msg)
                if getattr(self, "alerts", None) is not None:
                    self.alerts.send(f"shutoff:{s}", f"wxbot {self.alert_name}: sleeve {s} switched off", msg,
                                     priority="high", tags="warning")
            if st.get("tripped"):
                off.add(s)
            status[s] = {"k": rule.k, "h": rule.h, "since": dt.datetime.fromtimestamp(
                since, dt.timezone.utc).isoformat(timespec="seconds"), **ev,
                         "switched_off": bool(st.get("tripped")), "switched_off_day": st.get("day")}
        self.shutoff_off, self.shutoff_state = off, status

    def refresh_live_state(self):
        """Live account state, each read on its own (a failure no longer passes silently: account_fresh() gates new
        orders on all three). Order: resting orders (merged, never erased by a lagged snapshot), positions, balance
        and equity, fills."""
        now = time.time()
        acct = self.__dict__.setdefault("acct_ts", {"sync": 0.0, "balance": 0.0, "positions": 0.0})
        sync_ok = False
        try:
            self.live.sync()
            acct["sync"] = now
            sync_ok = True
        except Exception as e:
            log.warning("order sync failed: %s", e)
        pos_ok = False
        try:
            mp, _ = self.rest.positions()
            positions, basis = {}, 0.0
            for m in mp:
                q = float(m.get("position_fp") or 0)
                # the cost of the position, a positive amount whichever side is held (position_fp carries the
                # side: negative = NO). Kalshi's docs give no sign for it; a NO position reported with a negative
                # cost would otherwise lower settled equity by twice its cost and read as negative risk
                cost = abs(float(m.get("market_exposure_dollars") or 0))
                ev = m["ticker"].rsplit("-", 1)[0]
                basis += cost
                if q > 0:
                    positions[m["ticker"]] = Position(m["ticker"], ev, yes_qty=q, yes_cost=cost)
                elif q < 0:
                    positions[m["ticker"]] = Position(m["ticker"], ev, no_qty=-q, no_cost=cost)
            # taker fills answered by the exchange but possibly not in this (lagged) positions read yet
            as_of = getattr(self.live, "last_as_of", 0.0) or (now - 10)
            self.recent_takes = [x for x in getattr(self, "recent_takes", []) if x[0] > now - 120]
            for ts, t, ev, side, n, cost in self.recent_takes:
                if ts > as_of:
                    p = positions.setdefault(t, Position(t, ev))
                    if side == "bid":
                        p.yes_qty += n
                        p.yes_cost += cost
                    else:
                        p.no_qty += n
                        p.no_cost += cost
            self.positions = positions
            self.positions_basis = basis
            acct["positions"] = now
            pos_ok = True
        except Exception as e:
            log.warning("positions refresh failed: %s", e)
        # equity for the stops: SETTLED equity = available cash + cash held by resting orders (when the available
        # balance leaves it out, measured once by `python -m wxbot smoke`) + the cost basis of open positions. It
        # moves only on settlements and fees, not on marks, so thin-book price swings cannot trip the stops (Oct 4
        # review, finding 2.8). The resting-order cash comes from the same orders snapshot that was just read (all
        # resting orders, manual ones too, exactly as the balance holds them), not from the bot's own order book,
        # which keeps a filled order for a sync or two and would count a fill twice beside the positions read. A
        # reading is taken only when orders, positions and balance were all read in this refresh. Stops and the peak
        # use the median of the last three readings. With the risk keys of Oct 11, 2026 (wxbot/risk/stop_guard.py)
        # each median is also one reading for the stop guard, and the peak can come from start-of-day equity only.
        g = getattr(self, "stop_guard", None)
        try:
            b = self.rest.balance()
            acct["balance"] = now
            sem = self.store.kv_get("balance_semantics") or {}
            if not sem and not getattr(self, "_sem_warned", False):
                self._sem_warned = True
                self.store.log("WARN", f"balance semantics not measured (run {self.hint('smoke')}): assuming the "
                                       "available balance leaves out cash held by resting orders")
            held = getattr(self.live, "snapshot_cash", 0.0) if sem.get("excludes_resting", True) else 0.0
            cash = (b.get("balance") or 0) / 100.0
            self.resting_held_usd = held
            self.avail_balance = cash
            self.live_equity_mtm = cash + held + (b.get("portfolio_value") or 0) / 100.0
            if pos_ok and sync_ok:
                eq = cash + held + getattr(self, "positions_basis", 0.0)
                self.live_equity = eq
                prev = getattr(self, "eq_readings", [])
                if prev and now - getattr(self, "eq_ts", now) > getattr(self, "account_stale_s", 30.0):
                    # readings from before a gap (an outage) would outvote the first new one in the median: start
                    # over, so the stops and new orders wait for three readings taken after it
                    prev = []
                    if g is not None:
                        g.restart()          # and the stop guard's count of readings toward a resume with them
                r = (prev + [eq])[-3:]
                self.eq_readings, self.eq_ts = r, now
                if len(r) == 3:
                    med = sorted(r)[1]
                    day = et_day().isoformat()
                    sod = self.store.kv_get("sod_equity", {})
                    if sod.get("day") != day:
                        sod = {"day": day, "equity": med}
                        self.store.kv_set("sod_equity", sod)
                    self.risk.realized_today = med - sod["equity"]
                    self.risk.equity = med
                    # the high point the drawdown stop measures from: raised by every reading (peak_from: reading, as
                    # before) or only by the start-of-day equity (start_of_day, Oct 11, 2026: one reading taken while
                    # a payout was in the balance but its position still in the list raised it by about $15)
                    top = sod["equity"] if g is not None and g.cfg.peak_from == "start_of_day" else med
                    self.risk.peak_equity = max(self.store.kv_get("peak_equity", top), top)
                    self.store.kv_set("peak_equity", self.risk.peak_equity)
                    if g is not None and g.cfg.guards_stops:
                        self.guard_reading(now, day)
        except Exception as e:
            log.warning("balance refresh failed: %s", e)
        fills = self.live.poll_fills()
        if getattr(self, "ws", None) is not None:
            # a shadow comparison: a failure here must never cost the fills just polled (they are recorded below)
            self.side_task("ws_fills", self.ws_fill_compare, fills, now)
        raws = list(getattr(self.live, "last_new_raw", []) or [])
        for i, f in enumerate(fills):
            try:
                f.event = f.ticker.rsplit("-", 1)[0]
                if self.store.add_fill(self.mode, f, self.fair_cache.get(f.ticker), tag=self.model_tag()):
                    self.counters["fills"] += 1
                    log.info("FILL live %s %s @%sc x%.2f (%s)", f.ticker, f.side, f.price_cents, f.count, f.sleeve)
            except Exception as e:
                # not recorded (a database error, an odd record): hand it back to the poll so the next one retries
                # it, instead of losing it from the bot's P&L (it is already marked as seen)
                log.error("could not record fill %s: %r (retried next poll)", f.order_ref, e)
                raw = raws[i] if i < len(raws) else None
                fid = (raw or {}).get("fill_id") or (raw or {}).get("trade_id")
                if fid:
                    self.live.seen_fill_ids.discard(fid)
                    self.live.unresolved_fills[fid] = raw

    def refresh_ml(self):
        """Re-forecast in a background thread whenever a new decision hour starts for any active high event."""
        if self.ml is None or (self.ml_thread and self.ml_thread.is_alive()):
            return
        now = dt.datetime.now(dt.timezone.utc)
        targets = sorted({(STATIONS[ev.series]["icao"], ev.local_date) for ev in self.events.values()
                          if ev.kind == "high" and ev.local_date and ev.series in STATIONS})
        key = tuple((icao, D, self.ml.decision_hour(D, now)) for icao, D in targets)
        if not targets or key == self.ml_key:
            return
        self.ml_key = key

        def work():
            try:
                t0 = time.time()
                out = self.ml.predict(targets, now)
                log.info("ML forecasts refreshed: %d station-days in %.0fs", len(out), time.time() - t0)
            except Exception as e:
                log.warning("ML refresh failed: %s", e)
                self.ml_key = None
        import threading
        self.ml_thread = threading.Thread(target=work, daemon=True)
        self.ml_thread.start()

    def maybe_prefetch_ml(self, now: Optional[dt.datetime] = None):
        """Between decision hours (minutes 42-54) fetch any new guidance cycles, and the daily 45-day history after
        midnight UTC, so the top-of-hour forecast refresh only needs observations and HRRR. The data are the same
        cycles the decision would fetch itself; only the timing moves off the decision path."""
        if self.ml is None or not (self.cfg.get("ml", {}) or {}).get("prefetch", True):
            return
        now = now or dt.datetime.now(dt.timezone.utc)
        if not (42 <= now.minute <= 54) or (self.ml_thread and self.ml_thread.is_alive()):
            return
        if not self.due("ml_prefetch", 40 * 60):
            return
        raw = self.ml.raw

        def work():
            try:
                raw.refresh(full=raw.last_hist_day != dt.datetime.now(dt.timezone.utc).date())
            except Exception as e:
                log.warning("ML prefetch failed: %s", e)
        import threading
        self.ml_thread = threading.Thread(target=work, daemon=True)
        self.ml_thread.start()

    def ml_stale(self, fc: dict, D, now: float) -> bool:
        """The forecaster missed the latest decision hour (its inputs failed to load, say): an older hour's forecast,
        pooled with that hour's weights against the current market, leans on stale information, so the market-anchored
        fair value stands until it catches up (round-7 pre-mortem). Each decision hour allows an hour for the refresh."""
        from ..ml.features import DEC_H
        d0 = dt.datetime(D.year, D.month, D.day, tzinfo=dt.timezone.utc).timestamp()
        past = [h for h in DEC_H if d0 + h * 3600 + 3600 <= now]
        if not past or int(fc.get("h", past[-1])) >= past[-1]:
            return False
        self.counters["ml_stale"] += 1
        if self.due(f"ml_stale_note:{D}", 3600):
            self.store.log("WARN", f"forecast for {D} is still from decision hour {fc.get('h')} (missed hour "
                                   f"{past[-1]}): using the market-anchored fair value until it updates")
        return True

    def apply_ml(self, ev: EventBook, fv, now: float):
        """Replace fair probabilities with the market x Tier-1 pool; keep observation kills and conflict flags."""
        from ..ml.live import bracket_pmf
        fc = self.ml.get(STATIONS[ev.series]["icao"], ev.local_date)
        if not fc or not fv.market:
            return fv
        if self.ml_stale(fc, ev.local_date, now):
            return fv
        f = bracket_pmf(ev.brackets, fc["P"], fc["anchor_r"])
        nbs = fc.get("nbs_txn")
        probs = self.pool.probs(ev.brackets, fv.market, f, fc["h"], nbs=nbs)
        for t in fv.dead:
            probs[t] = 0.0
        s = sum(probs.values()) or 1.0
        probs = {k: v / s for k, v in probs.items()}
        fv.disagreement = 0.5 * sum(abs(fv.market.get(b.ticker, 0.0) - probs.get(b.ticker, 0.0)) for b in ev.brackets)
        fv.probs = probs
        fv.notes = [n for n in fv.notes if "disagreement" not in n] + [f"ml pool h={fc['h']}"]
        k = (ev.event, fc["h"])
        if k not in self.ml_logged:
            self.ml_logged.add(k)
            rows = [(now, ev.event, fc["h"], b.ticker, fv.market.get(b.ticker), float(fb), probs.get(b.ticker),
                     self.ml_sha) for b, fb in zip(ev.brackets, f)]
            self.store.add_ml_snapshots(rows)
            self.store.add_ml_ctx([(ev.event, fc["h"], nbs, now)])
            cx = getattr(self, "cfg_x", None)
            if cx is not None and cx.enabled and ev.series not in getattr(self, "series_t", set()):
                self.shadow_taker(ev, probs, fc["h"], now)
        return fv

    def shadow_taker(self, ev: EventBook, probs: dict, h: int, now: float):
        """Sleeve X (shadow): log the taker trades the pre-registered rule would make at this decision hour.
        Only within 30 minutes of the decision time (a restart mid-hour would otherwise log at an off-rule time)."""
        if ev.local_date is not None:
            d0 = dt.datetime(ev.local_date.year, ev.local_date.month, ev.local_date.day, tzinfo=dt.timezone.utc)
            if now - (d0.timestamp() + int(h) * 3600) > 1800:
                return
        try:
            liquid, spr, msum, sig = taker.signals(ev.quotes, probs, self.cfg_x, books=ev.books)
        except Exception as e:
            log.warning("shadow taker failed for %s: %s", ev.event, e)
            return
        rows = []
        for x in sig:
            k = (ev.event, x["ticker"], x["side"])
            if k in self.x_seen:  # first signal per bracket and side only
                continue
            self.x_seen.add(k)
            rows.append((now, ev.event, x["ticker"], x["side"], x["price"], x["size"], x["avail"], x["p"], x["edge"],
                         int(h), self.ml_sha, spr, msum, x["intended"], x["depth"], x["booked"]))
            log.info("SHADOW TAKER %s buy %s @%.0fc x%g (p %.2f, edge %.1fc, h=%d; booked %g of depth %s)",
                     x["ticker"], x["side"].upper(), 100 * x["price"], x["size"], x["p"], 100 * x["edge"], h,
                     x["booked"], "unknown" if x["depth"] is None else f"{x['depth']:g}")
        if rows:
            self.store.add_taker_shadow(rows)
            if (getattr(self, "cfg_x_raw", None) or {}).get("trade") and self.x_priority() != "city":
                # taker_shadow.priority (Oct 9): held with the other signals of this decision time and sent by
                # flush_x_pending in priority order; their execution-log rows follow the orders there
                self.queue_x(rows, ev.local_date, now, ev.books, getattr(self, "book_ts", None))
                return
            if (getattr(self, "cfg_x_raw", None) or {}).get("trade"):
                for r in rows:   # (ts, event, ticker, side, price, size, ...): `size`, not the booked size, as before
                    self.execute_take("X", r[1], r[2], r[3], r[4], r[5], self.cfg_x_raw)
            xl = getattr(self, "exec_log", None)
            if xl is not None:   # after the orders: the decision's book, then its re-reads
                bts = getattr(self, "book_ts", None) or {}
                for r in rows:
                    self.side_task("exec_log", xl.record, self.store, "decision", f"X:{r[1]}:{r[2]}:{r[3]}", "X", r[1],
                                   r[2], r[3], r[4], ev.books.get(r[2]), bts.get(r[2]), None, now)

    def x_priority(self) -> str:
        """taker_shadow.priority: the order Sleeve X spends its daily budget in when signals compete ('city', the
        default and the old behaviour; 'hour'; 'edge'; wxbot/strategy/taker.priority_order). An unknown value counts
        as 'city', with a warning."""
        p = str((getattr(self, "cfg_x_raw", None) or {}).get("priority") or "city").strip().lower()
        if p not in taker.PRIORITIES:
            if not getattr(self, "_x_priority_warned", False):
                self._x_priority_warned = True
                log.warning("taker_shadow.priority %r unknown (city, hour or edge): city order", p)
            return "city"
        return p

    def queue_x(self, rows: list, D, now: float, books: Optional[dict] = None, book_ts: Optional[dict] = None):
        """Hold Sleeve X's new trades (shadow_taker rows) for flush_x_pending, with what the priority needs: the
        decision time (the event's date at 00Z plus h hours), h, the edge and the arrival order; and the book the
        decision saw, for the execution log."""
        pend = self.__dict__.setdefault("x_pending", [])
        d0 = dt.datetime(D.year, D.month, D.day, tzinfo=dt.timezone.utc).timestamp() if D is not None else None
        for r in rows:
            h = int(r[9])
            self.x_seq = getattr(self, "x_seq", 0) + 1
            pend.append({"row": tuple(r), "h": h, "edge": float(r[8]), "seq": self.x_seq, "queued": float(now),
                         "dec_ts": float(now) if d0 is None else d0 + h * 3600.0,
                         "book": (books or {}).get(r[2]), "book_ts": (book_ts or {}).get(r[2])})

    def flush_x_pending(self, now: Optional[float] = None, force: bool = False) -> int:
        """Send the held Sleeve X trades in priority order once the oldest has waited taker_shadow.priority_hold_s
        (default 5 s, two loop ticks: a forecast refresh that lands while one step prices the events would otherwise
        split one decision time's signals over two steps). Each goes through execute_take exactly as before (its caps,
        budget day and stops). A trade more than 30 minutes past its decision time is not sent, as shadow_taker would
        not have logged it then. Then the decisions' execution-log rows. Returns how many were sent."""
        pend = getattr(self, "x_pending", None)
        if not pend:
            return 0
        now = time.time() if now is None else now
        raw = getattr(self, "cfg_x_raw", None) or {}
        if not force and now - min(it["queued"] for it in pend) < float(raw.get("priority_hold_s", 5.0)):
            return 0
        self.x_pending = []
        items = taker.priority_order(pend, self.x_priority())
        n = 0
        for it in items:
            r = it["row"]
            if now - it["dec_ts"] > 1800:
                log.info("TAKE X %s %s not sent: held past 30 minutes after its decision time", r[2], r[3])
                continue
            self.execute_take("X", r[1], r[2], r[3], r[4], r[5], raw)
            n += 1
        xl = getattr(self, "exec_log", None)
        if xl is not None:   # after the orders: the decisions' books, then their re-reads
            for it in items:
                r = it["row"]
                self.side_task("exec_log", xl.record, self.store, "decision", f"X:{r[1]}:{r[2]}:{r[3]}", "X", r[1],
                               r[2], r[3], r[4], it["book"], it["book_ts"], None, r[0])
        return n

    def early_take(self, now: Optional[float] = None) -> int:
        """Sleeve E, the taker at h = -9 / -6 (Dec-Jan candidate C, accelerated by Shourya on Oct 8 after its untouched
        Mar-May check): send the frozen rule's trades that candidate C's job logged (wxbot/app/early_shadow.py, its own
        process and database, 15:05Z and 18:05Z) through Sleeve X's taker path and caps (execute_take: an immediate-
        or-cancel order at the logged price, its own daily budget and stop, every shared risk cap). Only signals
        logged within max_age_s count, so a bot that was down never takes an old price; each is acted on once."""
        cfg = self.cfg.get("early_taker") or {}
        now = now or time.time()
        path = cfg.get("db", "data/early_shadow_c.sqlite")
        if not os.path.isabs(path):
            path = os.path.join(getattr(self, "base", ".") or ".", path)
        if not os.path.exists(path):
            return 0
        import sqlite3
        from pathlib import Path
        try:
            c = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
            try:
                rows = c.execute("SELECT ts, event, ticker, side, price, size, h FROM taker_shadow WHERE ts >= ? "
                                 "ORDER BY ts", (now - float(cfg.get("max_age_s", 600)),)).fetchall()
            finally:
                c.close()
        except sqlite3.Error as e:
            if self.due("early_take_err", 3600):
                log.warning("early taker: cannot read %s: %s", path, e)
            return 0
        n = 0
        skip_h9 = bool(cfg.get("skip_h9_in_standard_time", False))
        for ts, ev, tk, side, price, size, h in rows:
            key = f"early_take:{ev}:{tk}:{side}"
            if self.store.kv_get(key):
                continue
            if skip_h9 and int(h) == -9 and self.h9_in_standard_time(ev, float(ts)):
                # winter (K3 option (a), Oct 9): from the end of daylight time to mid-March Kalshi lists the daily
                # markets at 15Z, the minute of the h = -9 decision, so its books are minutes old and no research
                # row covers them. The signal stays logged (the early shadow job's database, and this record); no order
                self.store.kv_set(key, {"ts": ts, "h": h, "skipped": "h = -9 in standard time"})
                log.info("EARLY TAKER %s %s not sent: h = -9 while US Eastern is on standard time (signal logged)",
                         tk, side.upper())
                continue
            self.store.kv_set(key, {"ts": ts, "h": h})
            log.info("EARLY TAKER %s buy %s @%.2f x%g (h=%d, logged %.0fs ago)", tk, side.upper(), float(price),
                     float(size), int(h), now - float(ts))
            self.execute_take("E", ev, tk, side, float(price), float(size), cfg)
            n += 1
        return n

    @staticmethod
    def h9_in_standard_time(event: str, logged_ts: float) -> bool:
        """Whether the h = -9 decision of this event (15Z the day before its date) falls while US Eastern is on
        standard time (wxbot.timeutil.et_standard_time); the time the signal was logged when the date cannot be read."""
        from ..timeutil import et_standard_time
        D, _ = parse_event_date(event)
        if D is None:
            return et_standard_time(logged_ts)
        d0 = dt.datetime(D.year, D.month, D.day, tzinfo=dt.timezone.utc).timestamp()
        return et_standard_time(d0 - 9 * 3600)

    def maybe_refit_pool(self, now: Optional[dt.datetime] = None):
        """Run the production pool refit once on the configured weekday (default Monday) after the configured UTC
        hour, in a background thread with its own public-API client; load the new weights only if the fit passes
        the sanity bounds (wxbot/ml/pool_fit.py). The attempt is recorded first, so a failure is not retried in a loop."""
        if self.ml is None or not self.refit_cfg.get("enabled"):
            return
        now = now or dt.datetime.now(dt.timezone.utc)
        if now.weekday() != int(self.refit_cfg.get("weekday", 0)) or now.hour < int(self.refit_cfg.get("after_utc_hour", 6)):
            return
        # the attempt is recorded in this database and in a file shared by the paper and live databases, so a switch
        # to live on a refit day does not refit again from the new (nearly empty) live database
        marker = os.path.join(os.path.dirname(getattr(self, "status_path", "") or self.store.path) or ".",
                              "pool_refit_date.txt")
        try:
            shared = open(marker).read().strip()
        except OSError:
            shared = ""
        if now.date().isoformat() in (self.store.kv_get("pool_refit_date"), shared):
            return
        if self.refit_thread and self.refit_thread.is_alive():
            return
        self.store.kv_set("pool_refit_date", now.date().isoformat())
        try:
            with open(marker, "w") as fh:
                fh.write(now.date().isoformat() + "\n")
        except OSError as e:
            log.warning("could not write %s: %s", marker, e)
        self.store.log("INFO", f"weekly pool refit started ({now.date()})")
        # the paper and live records are separate databases from the switch to real money on: read this one and the
        # others by their conventional names next to it (paper data/wxbot.sqlite, live data/live.sqlite), so the
        # window has no hole at the switch
        here = os.path.dirname(self.store.path) or "."
        db_path = [self.store.path] + [p for p in (os.path.join(here, n) for n in ("wxbot.sqlite", "live.sqlite"))
                                       if os.path.abspath(p) != os.path.abspath(self.store.path)]
        base_url, today = self.rest.base, now.date()
        sha = getattr(self, "ml_sha", None)
        from ..ml.live import ART
        w_path = self.refit_cfg.get("weights_path") or os.path.join(ART, "pool_weights.json")
        raw_dir = getattr(getattr(getattr(self, "ml", None), "raw", None), "root", None)
        base_path = self.refit_cfg.get("base_rows_path") or os.path.join(ART, "pool_rows_base.json.gz")

        def work():
            import gzip
            from ..ml import pool_fit
            from ..ml.live import Pool
            try:
                client = KalshiREST(base_url=base_url, auth=None, read_rate=20.0)
                cache_path = os.path.join(os.path.dirname(db_path[0]) or ".", "refit_markets_cache.json.gz")
                try:
                    cache = json.load(gzip.open(cache_path, "rt"))
                except Exception:
                    cache = {}

                def get_markets(ev):
                    if ev not in cache:
                        ms = client.markets(event_ticker=ev, limit=100)
                        if not ms:
                            # settled before Kalshi's historical cutoff (about two months back): only the archive
                            # endpoint lists it (matters if the cache file is ever lost)
                            ms = client.paged("/historical/markets", "markets", {"event_ticker": ev, "limit": 100})
                        if ms and all(m.get("result") in ("yes", "no") for m in ms):
                            cache[ev] = ms  # settled: never changes again
                        return ms
                    return cache[ev]
                base = json.load(gzip.open(base_path, "rt")) if os.path.exists(base_path) else []
                # the design stays what the loaded weights use (a gate is kept, never added or dropped by a refit)
                gate = ((getattr(getattr(self, "pool", None), "gate", None) or {}).get("cols") or [None])[0]
                if gate and raw_dir:
                    try:   # NBM's max for snapshots logged before the bot stored it (own database only)
                        n = pool_fit.backfill_ctx(db_path[0], raw_dir)
                        if n:
                            log.info("pool refit: NBM context filled for %d snapshot hours", n)
                    except Exception as e:
                        log.warning("pool refit: NBM context backfill failed: %s", e)
                out, msg = pool_fit.refit(db_path, base, today, get_markets, model_sha=sha,
                                          **({"gate": gate} if gate else {}))
                try:
                    with gzip.open(cache_path + ".tmp", "wt") as fh:
                        json.dump(cache, fh)
                    os.replace(cache_path + ".tmp", cache_path)
                except Exception:
                    pass
                if out is not None:
                    pool_fit.write_weights(out, w_path)
                    self.pool = Pool(w_path)
                    msg += f"; new weights loaded (alpha {out['alpha']}, beta {out['beta']}"
                    msg += f", gate {out['gate']['theta']})" if out.get("gate") else ")"
                log.info("Weekly pool refit: %s", msg)
            except Exception as e:
                log.warning("Weekly pool refit failed (weights unchanged): %s", e)
        import threading
        self.refit_thread = threading.Thread(target=work, daemon=True)
        self.refit_thread.start()

    def refresh_rain(self, now: Optional[float] = None):
        """Sleeve R (shadow): within 30 minutes after each rain decision hour, price the rain cities with the frozen
        rain model pooled with the market and log the taker trades the pre-registered rule would make. The work runs
        in a background thread that never overlaps the ML thread (both use the raw weather folder); the main loop
        writes the rows it hands back."""
        if self.rain is None:
            return
        if self.rain_pending:
            items, self.rain_pending = self.rain_pending, []
            by_table = defaultdict(list)
            for table, row in items:
                by_table[table].append(row)
            for table, rows in by_table.items():
                self.store.add_rain_shadow(rows, table)
            if self.cfg_r.get("trade"):
                for r in by_table.get("rain_shadow", []):   # (ts, event, ticker, city, day, h, side, price, size, ...)
                    self.execute_take("R", r[1], r[2], r[6], r[7], r[8], self.cfg_r)
            xl = getattr(self, "exec_log", None)
            if xl is not None:   # after the orders; no book at hand: read at the next pass, then the re-reads
                for r in by_table.get("rain_shadow", []):
                    self.side_task("exec_log", xl.record, self.store, "decision", f"R:{r[1]}:{r[2]}:{r[6]}", "R", r[1],
                                   r[2], r[6], r[7], None, None, None, r[0])
        if (self.rain_thread and self.rain_thread.is_alive()) or (self.ml_thread and self.ml_thread.is_alive()):
            return
        from ..ml.rain import HOURS
        now = now or time.time()
        today = dt.datetime.fromtimestamp(now, dt.timezone.utc).date()
        due = []
        for dd in (-1, 0, 1):
            D = today + dt.timedelta(days=dd)
            d0 = dt.datetime(D.year, D.month, D.day, tzinfo=dt.timezone.utc).timestamp()
            for h in HOURS:
                if 0 <= now - (d0 + h * 3600) <= 1800 and (D, h) not in self.rain_done:
                    due.append((D, h))
        if not due:
            return
        self.rain_done.update(due)
        raw = self.ml.raw
        base_url = self.rest.base
        margin = float(self.cfg_r.get("margin", 0.06))
        size = float(self.cfg_r.get("size", 5))

        rain2, extra = getattr(self, "rain2", None), getattr(self, "rain_raw_extra", None)

        def work():
            try:
                if time.time() - getattr(raw, "last_refresh_ts", 0.0) > 600:   # e.g. 00Z: no high-market refresh then
                    raw.refresh(full=False)
                client = KalshiREST(base_url=base_url, auth=None, read_rate=5.0)
                for D, h in due:
                    self.rain_pending.extend(("rain_shadow", r) for r in
                                             self.rain_signals(client, raw.root, D, h, margin, size))
            except Exception as e:
                log.warning("rain shadow failed: %s", e)
            if rain2 is None:
                return
            try:   # R2: the nine extra sites need their own fresh inputs
                today = dt.datetime.now(dt.timezone.utc).date()
                if extra.last_hist_day != today:
                    extra.refresh(full=True)
                elif time.time() - getattr(extra, "last_refresh_ts", 0.0) > 600:
                    extra.refresh(full=False)
                client = KalshiREST(base_url=base_url, auth=None, read_rate=5.0)
                for D, h in due:
                    self.rain_pending.extend(("rain_shadow_v2", r) for r in
                                             self.rain_signals(client, raw.root, D, h, margin, size, model=rain2,
                                                               seen=self.rain2_seen, extra_root=extra.root))
            except Exception as e:
                log.warning("rain shadow R2 failed: %s", e)
        import threading
        self.rain_thread = threading.Thread(target=work, daemon=True)
        self.rain_thread.start()
        self.ml_thread = self.rain_thread   # the ML refresh and prefetch wait for it (shared raw folder)

    def rain_signals(self, client, raw_root: str, D: dt.date, h: int, margin: float, size: float, model=None,
                     seen: Optional[set] = None, extra_root: Optional[str] = None) -> list:
        from ..ml.rain import RAIN_CITY_EXTRA
        from ..strategy import rain_taker
        model = model or self.rain
        seen = self.rain_seen if seen is None else seen
        cities = model.meta.get("cities") or {}
        extra_sites = set(RAIN_CITY_EXTRA.values())
        ev = f"KXRAIN-{D:%y%b%d}".upper()
        quotes, pp, info, sizes = {}, {}, {}, {}

        def displayed(v):   # the listing's size at the touch (yes_bid_size_fp / yes_ask_size_fp); None if absent
            try:
                return None if v in (None, "") else float(v)
            except (TypeError, ValueError):
                return None
        for m in client.markets(event_ticker=ev, limit=100):
            city = m["ticker"].rsplit("-", 1)[-1]
            icao = cities.get(city)
            if icao is None or m.get("status") not in ("active", "open"):
                continue
            root = extra_root if icao in extra_sites else raw_root
            if root is None:
                continue
            b = float(m.get("yes_bid_dollars") or 0.0)
            a = float(m.get("yes_ask_dollars") or 0.0)
            a = 1.0 if a <= 0 else a           # no ask listed
            mid = (a + b) / 2
            if not 0 < mid < 1:
                continue
            try:
                p = model.predict(root, icao, D, h)
            except Exception as e:
                log.debug("rain model %s %s: %s", icao, D, e)
                continue
            if p is None or p != p:
                continue
            quotes[m["ticker"]] = (b, a)
            pp[m["ticker"]] = model.pool(mid, p)
            info[m["ticker"]] = (city, p, mid)
            sizes[m["ticker"]] = (displayed(m.get("yes_bid_size_fp")), displayed(m.get("yes_ask_size_fp")))
        rows = []
        tag = "RAIN" if model is self.rain else "RAIN2"
        for x in rain_taker.signals(quotes, pp, margin):
            k = (x["ticker"], x["side"])
            if k in seen:   # first signal per market and side
                continue
            seen.add(k)
            city, p, mid = info[x["ticker"]]
            # depth seen: the contracts displayed at the touch the signal buys at (YES: the ask; NO at 1 - bid: the
            # YES bid), which are those at its price or better; booked = min(size, depth), 0 if zero or unknown
            depth = sizes[x["ticker"]][1 if x["side"] == "yes" else 0]
            rows.append((time.time(), ev, x["ticker"], city, D.isoformat(), int(h), x["side"], x["price"], size,
                         x["fee"], p, pp[x["ticker"]], mid, x["edge"], model.meta["sha256"][:12], float(size), depth,
                         booked_size(size, depth)))
            log.info("SHADOW %s %s buy %s @%.0fc x%g (model %.2f, pool %.2f, mid %.2f, edge %.1fc, h=%d; booked %g of "
                     "depth %s)", tag, x["ticker"], x["side"].upper(), 100 * x["price"], size, p, pp[x["ticker"]], mid,
                     100 * x["edge"], h, rows[-1][-1], "unknown" if depth is None else f"{depth:g}")
        log.info("%s shadow %s h=%d: %d cities priced, %d new signals", tag.lower(), ev, h, len(pp), len(rows))
        return rows

    def execute_take(self, sleeve: str, event: str, ticker: str, side: str, price: float, size: float,
                     trade_cfg: dict) -> Optional[dict]:
        """Taker order path for Sleeve X / R, used only when the sleeve's `trade` flag is on. Paper: fills against
        the current order book; live: an immediate-or-cancel limit order at the logged price (plus max_slip_c).
        Caps: contracts per signal, dollars spent per sleeve per budget day (fees included), and, since the Oct 4
        review (finding 2.4), the same per-bracket, per-event, total-risk and cash caps as the makers, with every
        resting order of ours counted. Live: nothing is sent while the account state is stale."""
        if not trade_cfg.get("trade") or self.risk.halted:
            return None
        import math
        import uuid
        from ..execution.taker import TakerExecutor, taker_fee
        from ..strategy.types import Fill
        now = time.time()
        if self.mode == "live" and not self.account_fresh(now):
            log.info("TAKE %s %s skipped: account state stale", sleeve, ticker)
            return None
        if self.mode == "live" and len(getattr(self, "eq_readings", [])) < 3:
            log.info("TAKE %s %s skipped: the stops have not checked three equity readings yet", sleeve, ticker)
            return None
        if hasattr(self, "cfg") and sleeve[:1] in self.stopped_sleeves():
            log.info("TAKE %s %s skipped: sleeve stopped for the day", sleeve, ticker)
            return None
        if sleeve[:1] in self.shutoff_blocked():
            log.info("TAKE %s %s skipped: sleeve switched off by its shut-off rule", sleeve, ticker)
            return None
        # the spend cap's day starts at 20:00 UTC, so the 21Z decision hour (the evening before: most signals and the
        # largest edges in the research) gets first claim on each day's budget instead of what is left after 03-18Z
        day = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=float(trade_cfg.get("budget_day_shift_h", 4))
                                                                 )).date().isoformat()
        key = f"take_spent:{sleeve}:{day}"
        spent = float(self.store.kv_get(key, 0.0) or 0.0)
        slip = int(trade_cfg.get("max_slip_c", 0))
        lim = min(round(price + slip / 100.0, 2), 0.99)
        cap_day = float(trade_cfg.get("trade_max_usd_day", 25.0))
        n = float(math.floor(min(float(trade_cfg.get("trade_size", size)), size)))
        if n < 1 or spent + n * lim + taker_fee(n, lim) > cap_day + 1e-9:
            # the frozen rule's size or nothing (the day budget counts fees); logged since Oct 9 (it used to be the one
            # silent refusal on this path, docs/results/r12_x_priority_check.txt)
            log.info("TAKE %s %s skipped: %s", sleeve, ticker, "size under one contract" if n < 1 else
                     f"day budget (${spent:.2f} of ${cap_day:.2f} spent, this order ${n * lim + taker_fee(n, lim):.2f})")
            return None
        # shared caps: the order as a maker-style order whose cost includes the fee (rounded up to the cent)
        fee_c = int(math.ceil(100 * taker_fee(n, lim) / n - 1e-9))
        cost_c = min(int(round(lim * 100)) + fee_c, 99)
        d = DesiredOrder(ticker, "bid" if side == "yes" else "ask", cost_c if side == "yes" else 100 - cost_c, n,
                         sleeve, 1.0 if side == "yes" else 0.0, reason="taker", event=event)
        pos_by_ev, base, tick_by_ev, open_risk, cash_used = self.risk_context([])
        kept, _ = filter_orders([d], pos_by_ev, tick_by_ev, self.risk_cfg, open_risk, cash_used=cash_used,
                                cash_cap=self.cash_cap(), netting=self.netting, base_pending=base)
        if not kept:
            log.info("TAKE %s %s skipped: shared caps (open risk $%.2f)", sleeve, ticker, open_risk)
            return None
        n = float(kept[0].count)
        ex = TakerExecutor(self.rest, self.mode, slip)
        book = None
        if self.mode != "live":
            try:
                book = self.rest.orderbook(ticker)
            except Exception as e:
                log.warning("book for %s failed: %s", ticker, e)
        r = ex.take(ticker, side, price, n, book, client_order_id=f"wxb-{sleeve[:1]}-tk{uuid.uuid4().hex[:16]}")
        log.info("TAKE %s %s buy %s @%.2f x%g -> %s %g", sleeve, ticker, side.upper(), price, n, r.status, r.filled)
        if r.order_id and self.live is not None:
            self.live.my_order_ids.add(r.order_id)
            self.live.order_sleeve[r.order_id] = sleeve[:1]
        if r.status == "error":
            # the order may have reached the exchange even though no answer came back: count it as spent in full
            # (the fill feed records whatever really filled), and as a position until a positions read newer than
            # the attempt shows what happened
            self.store.kv_set(key, spent + n * lim + taker_fee(n, lim))
            if self.mode == "live":
                self.recent_takes = getattr(self, "recent_takes", []) + [
                    (time.time(), ticker, event, "bid" if side == "yes" else "ask", n, n * lim + taker_fee(n, lim))]
        elif r.filled > 0 and r.avg_price is not None:
            self.store.kv_set(key, spent + r.filled * r.avg_price + r.fee)
            api_side = "bid" if side == "yes" else "ask"
            if self.mode == "live":   # fills arrive through the account's fill feed; until the positions read
                self.recent_takes = getattr(self, "recent_takes", []) + [   # shows them, they count from here
                    (time.time(), ticker, event, api_side, r.filled, r.filled * r.avg_price + r.fee)]
                p = self.positions.setdefault(ticker, Position(ticker, event))
                if api_side == "bid":
                    p.yes_qty += r.filled
                    p.yes_cost += r.filled * r.avg_price + r.fee
                else:
                    p.no_qty += r.filled
                    p.no_cost += r.filled * r.avg_price + r.fee
            else:
                pc = int(round(100 * r.avg_price)) if side == "yes" else int(round(100 - 100 * r.avg_price))
                self.apply_fill(Fill(time.time(), ticker, event, api_side, pc, r.filled,
                                     sleeve, is_taker=True, fee=r.fee, order_ref=r.order_id or ""))
        if getattr(self, "exec_log", None) is not None:   # after the order and its bookkeeping: never delays it
            self.side_task("exec_log", self.log_take, sleeve, event, ticker, side, price, r, book)
        return r.__dict__

    def log_take(self, sleeve: str, event: str, ticker: str, side: str, price: float, r, book: Optional[dict]):
        """Execution log row for a taker order (its answer) with the book at send time: paper, the book it was filled
        against; live, the loop's last read of that market, or none (then read at the next pass). The ref is the
        decision's for Sleeve X (and R), so the two share one set of re-reads."""
        bts = r.ts if book is not None else None
        if book is None:
            book, bts = self.cached_book(ticker)
        self.exec_log.record(self.store, "send", f"{sleeve[:1]}:{event}:{ticker}:{side}", sleeve[:1], event, ticker,
                             side, price, book, bts, dict(r.__dict__))

    def refresh_paper_report(self):
        """Rewrite data/paper_report.md once an hour in a child process (stdlib only, reads the database read-only),
        so the latest numbers are always on disk without blocking the loop."""
        script = os.path.join(self.base, "scripts", "paper_report.py")
        if not os.path.exists(script):
            return
        try:
            import subprocess
            import sys as _sys
            prev = getattr(self, "_report_proc", None)
            if prev is not None and prev.poll() is None:
                return          # last hour's report still running
            self._report_proc = subprocess.Popen([_sys.executable, script, self.store.path, self.status_path],
                                                 cwd=self.base, stdout=subprocess.DEVNULL,
                                                 stderr=subprocess.DEVNULL)
        except Exception as e:  # never let reporting stop the bot
            log.debug("paper report refresh failed: %s", e)

    def rain_status(self) -> dict:
        if getattr(self, "rain", None) is None:
            return {"on": False}
        pnl, cnt, n = self.store.rain_pnl()
        out = {"on": True, "model": self.rain.meta["sha256"][:12], "signals": n, "settled_contracts": cnt,
               "settled_pnl_usd": round(pnl, 2), "decision_hours_done": len(self.rain_done)}
        if getattr(self, "rain2", None) is not None:
            p2, c2, n2 = self.store.rain_pnl("rain_shadow_v2")
            out["R2"] = {"model": self.rain2.meta["sha256"][:12], "signals": n2, "settled_contracts": c2,
                         "settled_pnl_usd": round(p2, 2)}
        return out

    def settle_rain_shadow(self):
        """Record Kalshi's result for logged rain-shadow trades (a few markets per call, oldest first)."""
        for table in ("rain_shadow", "rain_shadow_v2"):
            for t in self.store.rain_shadow_open(older_than=time.time() - 8 * 3600, table=table)[:20]:
                try:
                    m = self.rest.market(t)
                except Exception:
                    continue
                if m.get("result") in ("yes", "no"):
                    self.store.set_rain_result(t, m["result"], time.time(), table=table)

    def settle_taker_shadow(self):
        """Record Kalshi's result for logged shadow-taker trades (a few markets per call, oldest first)."""
        for t in self.store.taker_shadow_open(older_than=time.time() - 6 * 3600)[:25]:
            try:
                m = self.rest.market(t)
            except Exception:
                continue
            if m.get("result") in ("yes", "no"):
                self.store.set_taker_result(t, m["result"], time.time())

    def refresh_weather(self):
        if self.due("nbm", self.cfg["loop"]["nbm_every_s"]):
            try:
                self.nbm.refresh()
            except Exception as e:
                log.warning("NBM refresh failed: %s", e)
        if self.due("metar", self.metar_interval()) and self.stations_needed:
            try:
                self.metars = obs_mod.fetch_metars(self.stations_needed, hours=30)
                self.metar_version = getattr(self, "metar_version", 0) + 1
            except Exception as e:
                log.warning("METAR fetch failed: %s", e)
        if self.metars and self.due("wx_stale_check", 600):
            newest = max((o.get("obsTime") or 0 for v in self.metars.values() for o in v), default=0)
            if newest and time.time() - newest > self.stale_weather_s:
                self.store.log("WARN", f"METAR feed stale: newest observation {int((time.time() - newest) / 60)} min "
                                       "old (dead-bracket detection runs on old data until it recovers; quoting continues)")
        if (self.cfg_b.enabled or self.b_shadow) and self.due("index", self.cfg["loop"]["index_every_s"]):
            # Sleeve B's input: a side task, so an odd answer cannot fail every live step (round-9 audit)
            self.index_point = self.side_task("index_point", obs_mod.kalshi_index_latest, self.rest, "miami")
        if getattr(self, "round_guard_on", False) and self.due("obs5", self.obs5_every_s):
            self.side_task("obs5", self.read_obs5)

    def read_obs5(self, now: Optional[float] = None):
        """The rounding guard's 5-minute readings: every row of the last 36 hours in the observation archive
        (data/obs5.sqlite: one indexed read, about 10,000 rows for 24 stations), as round_guard readings. Rows the
        archive first saw after `now` are left out (replays and tests; live they cannot exist). A missing archive or
        one with nothing new for round_guard.stale_minutes is logged once an hour: the guard then sees only the METARs,
        so it cannot catch whole-C ambiguity."""
        import sqlite3
        from pathlib import Path
        from ..models import round_guard
        now = time.time() if now is None else now
        rows = []
        if os.path.exists(self.obs5_path):
            since = dt.datetime.fromtimestamp(now - 36 * 3600, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M")
            c = sqlite3.connect(Path(self.obs5_path).resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
            try:
                rows = c.execute("SELECT station, ts, temp_c, raw, first_seen FROM obs5 WHERE ts >= ?",
                                 (since,)).fetchall()
            finally:
                c.close()
        out = defaultdict(list)
        newest = 0.0
        for stn, ts, temp, raw, seen in rows:
            if seen is not None and seen > now:
                continue
            t = parse_ts(ts)
            out[stn] += round_guard.readings_from_metar(raw or "", t, temp_c=temp)
            newest = max(newest, t)
        self.obs5, self.obs5_newest = dict(out), newest
        self.obs5_version += 1
        if now - newest > self.obs5_stale_s and self.due("obs5_stale", 3600):
            self.store.log("WARN", "round guard: " + (f"no 5-minute reading in the observation archive for "
                                                       f"{(now - newest) / 60:.0f} min" if newest else
                                                       f"no observation archive at {self.obs5_path}") +
                           " (is wxbot-obs5.timer running?); until it is back the guard sees only the METARs")

    def round_guard_skip(self, ev: EventBook, now: float) -> frozenset:
        """Tickers Sleeves A and T must leave alone on this ladder now (round_guard.guard on today's 5-minute readings
        and METARs), with a short log line whenever the set changes. Empty off the target day or with the guard off."""
        if not getattr(self, "round_guard_on", False) or ev.kind not in ("high", "low") or ev.local_date is None \
                or ev.series not in STATIONS or not self.target_day(ev, now):
            return frozenset()
        from ..models import round_guard
        version = (self.obs5_version, getattr(self, "metar_version", 0))
        hit = self.guard_cache.get(ev.event)
        if hit is not None and hit[0] == version:
            return hit[1]
        st = STATIONS[ev.series]
        t0, t1 = obs_mod.lst_day_bounds(ev.local_date, st["lst_offset_h"])
        rd = list(self.obs5.get(st["icao"], []))
        for o in self.metars.get(st["icao"]) or []:
            if o.get("obsTime") is not None:
                rd += round_guard.readings_from_metar(o.get("rawOb") or "", float(o["obsTime"]), temp_c=o.get("temp"))
        amb, cands = round_guard.guard(rd, ev.brackets, t0, t1, ev.kind)
        last = self.guard_last.get(ev.event, frozenset())
        if amb != last:
            def labels(ts):
                return " and ".join(b.label() for b in sorted(ev.brackets, key=lambda b: b.lo) if b.ticker in ts)
            if amb:
                self.store.log("INFO", f"{ev.event}: round guard skips {labels(amb)}: the {ev.kind} so far could be "
                                       f"{' or '.join(str(v) for v in cands)} F (whole-C 5-minute readings; no exact "
                                       "reading or 6-hour group settles it yet)")
            else:
                self.store.log("INFO", f"{ev.event}: round guard lifted on {labels(last)} ({ev.kind} so far "
                                       f"{'/'.join(str(v) for v in cands) or 'unknown'} F)")
            self.guard_last[ev.event] = amb
        self.guard_cache[ev.event] = (version, amb)
        return amb

    def round_guard_status(self):
        if not getattr(self, "round_guard_on", False):
            return None
        return {"blocked": {e: sorted(v) for e, v in self.guard_last.items() if v and e in self.events},
                "obs5_newest_age_min": round((time.time() - self.obs5_newest) / 60, 1) if self.obs5_newest else None}

    def metar_interval(self, minute: Optional[int] = None) -> float:
        """Seconds between METAR polls. Routine reports are taken at :51-:56 and reach the public feed about 3.5
        minutes later, so poll fast (loop.metar_fast_every_s) in the UTC minutes loop.metar_fast_window [from, to]
        (wrapping past :00) and every loop.metar_every_s otherwise (specials can land any time)."""
        L = self.cfg["loop"]
        fast = L.get("metar_fast_every_s")
        if not fast:
            return float(L["metar_every_s"])
        a, b = L.get("metar_fast_window", [53, 2])
        m = dt.datetime.now(dt.timezone.utc).minute if minute is None else minute
        inside = (a <= m <= b) if a <= b else (m >= a or m <= b)
        return float(fast if inside else L["metar_every_s"])

    # ------------------------------------------------------------- decisions
    def target_day(self, ev: EventBook, now: float) -> bool:
        st = STATIONS[ev.series]
        local_now = dt.datetime.fromtimestamp(now, dt.timezone.utc) + dt.timedelta(hours=st["lst_offset_h"])
        return ev.local_date == local_now.date()

    def decide(self) -> list[DesiredOrder]:
        now = time.time()
        desired: list[DesiredOrder] = []
        fv_rows = []
        reach_on = self.cfg_a.max_contracts_ahead is not None
        own_a = self.resting_own() if reach_on else None
        self.out_of_reach = []
        decided_used = defaultdict(float)
        for p in self.positions.values():
            decided_used[p.event] += p.yes_cost + p.no_cost
        for ev in self.events.values():
            if not ev.quotes or not self.book_fresh(ev, now, self.stale_book_s):
                continue
            st = STATIONS[ev.series]
            fc = self.nbm.forecast(st["icao"], ev.kind, ev.local_date) if ev.local_date else None
            obs_max = obs_min = None
            if self.target_day(ev, now) and self.metars.get(st["icao"]):
                ext = obs_mod.running_extremes(self.metars[st["icao"]], ev.local_date, st["lst_offset_h"])
                obs_max, obs_min = ext["max_f"], ext["min_f"]
            skill = NBM_SKILL.get(st["icao"], {}) if ev.kind == "high" else {}
            nbm_mean = (fc.mean + float(skill.get("bias", 0.0))) if fc else None
            sd_floor = float(skill["rmse"]) if skill.get("rmse") else None
            fv = ladder_fair_value(ev.brackets, ev.quotes, kind=ev.kind,
                                   nbm_mean=nbm_mean, nbm_sd=fc.sd if fc else None,
                                   hours_to_close=(ev.close_ts - now) / 3600, obs_max_f=obs_max, obs_min_f=obs_min,
                                   w_nbm=self.w_nbm, sd_floor=sd_floor)
            if self.ml is not None and ev.kind == "high":
                fv = self.apply_ml(ev, fv, now)
            for t, pr in fv.probs.items():
                self.fair_cache[t] = pr
                fv_rows.append((now, ev.event, t, pr, fv.market.get(t), fv.nbm.get(t), fv.disagreement))
            if fv.conflict and self.due(f"conflict:{ev.event}", 1800):
                self.store.log("WARN", f"{ev.event}: observation contradicts the market, not quoting "
                                       f"(obs max {obs_max}, {'; '.join(fv.notes)[:200]})")
            if ev.series in getattr(self, "watch_only", set()):
                continue  # priced for the logs only
            # rounding guard (off unless round_guard.enabled): a failure is a side error, never a step failure
            skip = frozenset()
            if getattr(self, "round_guard_on", False) and ("T" if ev.series in self.series_t else "A") in \
                    getattr(self, "round_guard_sleeves", {"A", "T"}):
                skip = self.side_task("round_guard", self.round_guard_skip, ev, now) or frozenset()
            if ev.series in self.series_t:
                orders, notes = sleeve_a.propose(ev, fv, now, self.cfg_t, decided_used[ev.event], skip_tickers=skip)
                for o in orders:
                    o.sleeve = "T"
            else:
                orders, notes = sleeve_a.propose(ev, fv, now, self.cfg_a, decided_used[ev.event], skip_tickers=skip,
                                                 resting=own_a, out_of_reach=self.out_of_reach)
            desired += orders
        if getattr(self, "x_pending", None):      # taker_shadow.priority only: X's held trades, in priority order
            self.flush_x_pending(now)
        if fv_rows and self.due("fvlog", 300):
            self.store.add_fair_values(fv_rows)
        # the shadow sleeves run as side tasks: a failure in one (bad data from Kalshi, say a program without a
        # target size) is logged and counted, never a step failure that would stop the real orders (round-7 review)
        if self.h_on:
            self.side_task("h_shadow", self.shadow_h_orders, now)
        if self.h2_on:
            self.side_task("h2_shadow", self.shadow_h_orders, now, "H2")
        desired += self.h_real_orders(now)
        # Sleeve B (real orders only if enabled; otherwise shadow-estimate rewards)
        if self.hourly and self.index_point and (self.cfg_b.enabled or self.b_shadow):
            if self.cfg_b.enabled:
                desired += self.sleeve_b_pass(now, real=True)
            elif self.b_shadow:
                self.side_task("b_shadow", self.sleeve_b_pass, now, False)
        return desired

    def sleeve_b_pass(self, now: float, real: bool) -> list:
        """Sleeve B on the Miami hourly ladders: its orders when real, otherwise the shadow reward estimate and the
        shadow paper orders."""
        out = []
        age_min = max((now - self.index_point["t"] / 1000.0) / 60.0, 0.0)
        for ev in self.hourly.values():
            if ev.series != "KXTEMPMIAH":
                continue
            ob, diag = sleeve_b.propose(ev, float(self.index_point["v"]), age_min, self.lip_programs, now, self.cfg_b)
            if real:
                out += ob
                continue
            if ob:
                self.shadow_lip(ev, ob, now)
            self.shadow_b_orders(ev, ob, now)
        return out

    def shadow_lip(self, ev: EventBook, ob: list[DesiredOrder], now: float):
        """Estimate reward $ a Sleeve B quote would earn over the next tick interval (not traded)."""
        rows = []
        dt_h = self.cfg["loop"]["tick_seconds"] / 3600.0
        for o in ob:
            prog = self.lip_programs.get(o.ticker)
            if not prog:
                continue
            book = ev.books.get(o.ticker, {})
            yes_lv = [(int(round(float(p) * 100)), float(q)) for p, q in (book.get("yes_dollars") or [])]
            no_lv = [(int(round(float(p) * 100)), float(q)) for p, q in (book.get("no_dollars") or [])]
            ours_y = [(1, o.count)] if o.side == "bid" else None
            ours_n = [(1, o.count)] if o.side == "ask" else None
            # target size and discount are nullable in Kalshi's schema: the same defaults as sleeve_b.propose
            sv = snapshot_value(yes_lv, no_lv, float(prog.get("target_size_fp") or self.cfg_b.target_default),
                                (prog.get("discount_factor_bps") or 5000) / 10000, ours_y, ours_n)
            hours = max((ts_of(prog["end_date"]) - ts_of(prog["start_date"])) / 3600, 1e-6)
            rate = (prog.get("period_reward") or 0) / 10000 / hours
            est = sv["frac"] * rate * dt_h
            self.lip_est_usd += est
            rows.append((now, o.ticker, int(sv["counts"]), sv["frac"], rate, est))
        self.lip_samples += 1
        if rows:
            self.store.add_lip(rows)

    def shadow_b_orders(self, ev: EventBook, ob: list[DesiredOrder], now: float):
        want = {o.key: o for o in ob}
        tick = {m["ticker"] for m in ev.markets}
        for o in self.paper_b.resting():
            if o.ticker in tick and ((o.ticker, o.side) not in want or want[(o.ticker, o.side)].price_cents != o.price_cents):
                self.paper_b.cancel(o.oid)
        for k, d in want.items():
            have = sum(o.remaining for o in self.paper_b.resting() if (o.ticker, o.side) == k)
            if have < 0.5 * d.count:
                self.paper_b.place(DesiredOrder(**{**d.__dict__, "count": d.count - have}), ev.books.get(d.ticker, {}), now)

    def settle_b(self):
        for t, p in list(self.positions_b.items()):
            if not (p.yes_qty or p.no_qty):
                continue
            try:
                m = self.rest.market(t)
            except Exception:
                continue
            res = m.get("result")
            if res in ("yes", "no"):
                pnl = (p.yes_qty if res == "yes" else 0) + (p.no_qty if res == "no" else 0) - p.yes_cost - p.no_cost
                self.b_realized += pnl
                self.store.log("INFO", f"shadowB settled {t} res={res} pnl={pnl:.2f}")
                self.positions_b[t] = Position(t, p.event)

    def _h(self, variant: str):
        return (self.paper_h, self.h_held) if variant == "H" else (self.paper_h2, self.h2_held)

    def record_h_fill(self, f, variant: str = "H"):
        _, held = self._h(variant)
        # shadow sizing: the paper order's open size, the trade print's contracts that reached it, and the booked
        # size, min of the two (the paper broker fills exactly that, so booked = count; 0 if a fill came without them)
        intended, depth = getattr(f, "intended", None), getattr(f, "depth", None)
        self.store.add_hourly_fill((f.ts, f.event, f.ticker, f.side, int(f.price_cents), float(f.count),
                                    intended, depth, booked_size(intended if intended is not None else f.count, depth)),
                                   variant)
        held[f.event].append((f.ticker, f.side, int(f.price_cents), float(f.count)))
        self.counters[f"{variant.lower()}_fills"] += 1
        log.info("SHADOW %s FILL %s %s @%sc x%.0f", variant, f.ticker, f.side, f.price_cents, f.count)

    def h2_probs(self, ev, now: float):
        """Nowcast P(yes) per strike of an hourly event, or None when the index history or model is missing."""
        from ..ml.nowcast import DELAY, SERIES_CITY, exceed
        city = SERIES_CITY.get(ev.series)
        if not city or self.nowcaster is None or self.index_feed is None or not self.nowcaster.ok(city):
            return None
        S = pd_ts(ev.close_ts)
        last = self.index_feed.latest(city)
        if last is None or now // 60 - last[0] > 12:   # no index, or the feed is stale
            return None
        # The model's lead m means "the last index value is at S - m - DELAY". Live points arrive about 5.5 minutes
        # after their minute and the feed polls once a minute, so anchor the lead on the newest point we hold
        # (asking for exactly now - 5 minutes would usually find nothing and the nowcast would never answer).
        m = int(ev.close_ts // 60) - int(last[0]) - DELAY
        if m < 1 or m > 60:
            return None
        key = (city, int(ev.close_ts), m, last[0])
        cache = self.__dict__.setdefault("_h2_cache", {})
        if key not in cache:   # one prediction per city, lead minute and new index point
            R = self.index_feed.as_series(city)
            cache[key] = self.nowcaster.predict(city, R, S, m)
            if len(cache) > 64:
                for k in list(cache)[:-32]:
                    del cache[k]
        out = cache[key]
        if out is None:
            return None
        P, x0 = out
        import numpy as np
        p_of = {}
        for mk in ev.markets:
            if mk.get("strike_type") == "greater" and mk.get("floor_strike") is not None:
                p_of[mk["ticker"]] = float(exceed(P[None, :], np.array([x0]), np.array([float(mk["floor_strike"])]))[0])
        return p_of

    def shadow_h_orders(self, now: float, variant: str = "H"):
        """Sleeve H shadow: requote the tails of every open hourly event on the variant's paper broker."""
        broker, held = self._h(variant)
        want = {}
        if True:
            for ev in self.hourly.values():
                if ev.series not in self.cfg_h.series or not ev.quotes or not self.book_fresh(ev, now,
                                                                                               self.stale_book_s):
                    continue
                p_of = None
                if variant == "H2":
                    try:
                        p_of = self.h2_probs(ev, now)
                    except Exception as e:  # never let the shadow break the loop
                        log.debug("H2 nowcast failed for %s: %s", ev.event, e)
                        p_of = None
                    if p_of is None:
                        continue
                for o in sleeve_h.propose(ev, now, self.cfg_h, held.get(ev.event, []), p_of=p_of, adverse=False):
                    want[o.key] = o   # adverse=False: the pre-registered shadow keeps its Oct 3 sizing
        books = {t: b for ev in self.hourly.values() for t, b in ev.books.items()}
        by_key = {}
        for o in broker.resting():
            by_key.setdefault((o.ticker, o.side), []).append(o)
        for k, lst in by_key.items():
            d = want.get(k)
            for o in lst:
                if d is None or o.price_cents != d.price_cents:
                    broker.cancel(o.oid)
        for k, d in want.items():
            lst = [o for o in by_key.get(k, []) if o.remaining > 0 and o.price_cents == d.price_cents]
            have = sum(o.remaining for o in lst)
            if have <= 0.5:
                broker.place(d, books.get(d.ticker, {}), now)
            elif have > d.count + 0.5:
                newest = max(lst, key=lambda o: o.placed_ts)
                broker.amend_size(newest.oid, max(newest.remaining - (have - d.count), 0))
            elif have < 0.5 * d.count:
                broker.place(DesiredOrder(**{**d.__dict__, "count": d.count - have}), books.get(d.ticker, {}), now)
        self.counters[f"{variant.lower()}_quotes"] = len(want)

    def settle_h(self, max_n: int = 30):
        """Record results for Sleeve H shadow fills once their hour has settled (fills are inside the hour, so a
        fill older than 65 minutes belongs to an hour that closed at least 5 minutes ago)."""
        done_events = set()
        for t in self.store.hourly_open_tickers(time.time() - 3900)[:max_n]:
            try:
                res = self.rest.market(t).get("result")
            except Exception:
                continue
            if res in ("yes", "no"):
                self.store.set_hourly_result(t, res, time.time())
                done_events.add(t.rsplit("-", 1)[0])
        if done_events:
            for variant in ("H", "H2"):
                _, held = self._h(variant)
                open_ev = {e for _ts, e, *_ in self.store.hourly_open(variant)}
                for e in done_events - open_ev:
                    held.pop(e, None)

    def load_h_breaker(self):
        self.h_breaker = self.store.kv_get("h_breaker", {}) or {}
        self.h_breaker_last = {}
        if self.cfg_h.enabled and self.h_breaker.get("tripped"):
            self.cfg_h.enabled = False
            self.store.log("WARN", "Sleeve H breaker is tripped (" + str(self.h_breaker.get("reason")) + "): real "
                                   f"hourly orders stay off; {self.hint('reset-h-breaker')} clears it")

    def h_real_orders(self, now: float) -> list:
        """Sleeve H with real orders (the main paper broker in paper mode, the exchange in live mode). Hourly quotes
        need fresher books than the daily ones: none for an event whose books are older than cfg_h.stale_book_s."""
        if not self.cfg_h.enabled:
            return []
        lim = min(self.stale_book_s, self.cfg_h.stale_book_s)
        out = []
        for ev in self.hourly.values():
            if ev.series in self.cfg_h.series and ev.quotes and self.book_fresh(ev, now, lim):
                out += sleeve_h.propose(ev, now, self.cfg_h, self.h_held_live(ev.event),
                                        filled=self.h_filled_gross(ev.event))
        return out

    def h_filled_gross(self, event: str):
        """Live: contracts filled per (ticker, side) in this hourly event from the bot's own fill record, for the
        per-strike size limit (net positions hide a YES and a NO fill on the same strike; review finding 2.12).
        Paper: None (the paper positions keep both sides)."""
        if self.mode != "live":
            return None
        out = defaultdict(float)
        for t, side, n in self.store.c.execute("SELECT ticker, side, SUM(count) FROM fills WHERE mode='live' AND "
                                               "event=? GROUP BY ticker, side", (event,)):
            out[(t, side)] += float(n or 0)
        return dict(out)

    def check_h_breaker(self):
        """Real hourly orders: switch them off once enough contracts have settled and too many lost or the P&L per
        contract is below the floor (sleeve_h.breaker_*). Runs after each settlement pass."""
        if not self.cfg_h.enabled:
            return
        res = {t: r for t, r in self.store.settled().items() if r in ("yes", "no")}
        s = sleeve_h.breaker_stats(self.store.fills(self.mode), res, self.cfg_h.series,
                                   float(self.h_breaker.get("since", 0.0)))
        self.h_breaker_last = s
        why = sleeve_h.breaker_verdict(s, self.cfg_h)
        if why:
            self.trip_h_breaker(why, s)

    def guarded_h_breaker(self):
        """check_h_breaker, fail-safe: while hourly orders are real, a check that cannot run switches them off (the
        breaker is their risk control); while they are shadow only, a failure is a side-task error."""
        try:
            self.check_h_breaker()
        except Exception as e:
            if self.cfg_h.enabled:
                log.exception("hourly breaker check failed")
                self.trip_h_breaker(f"breaker check failed ({e!r})"[:200], {})
            else:
                self.side_error("h_breaker", e)

    def trip_h_breaker(self, why: str, stats: dict):
        self.cfg_h.enabled = False  # the next execute() no longer wants hourly orders, so reconcile cancels them
        self.h_breaker = {**self.h_breaker, "tripped": True, "ts": time.time(), "reason": why, "stats": stats}
        self.store.kv_set("h_breaker", self.h_breaker)
        msg = (f"Sleeve H breaker tripped: {why}. Real hourly orders are off and their resting orders are being "
               f"cancelled; the daily sleeves keep running. Clear by hand: {self.hint('reset-h-breaker')}")
        self.store.log("ERROR", msg)
        log.error(msg)
        self.alerts.send("h_breaker", f"wxbot {self.alert_name}: hourly bot switched off", msg, priority="high",
                         tags="warning")

    def h_breaker_status(self) -> dict:
        c = self.cfg_h
        return {"real_orders_on": c.enabled, "tripped": bool(self.h_breaker.get("tripped")),
                "reason": self.h_breaker.get("reason"), "since": self.h_breaker.get("since"),
                "stats": self.h_breaker_last or self.h_breaker.get("stats"),
                "limits": {"min_contracts": c.breaker_min_contracts, "max_lost_share": c.breaker_max_wrong,
                           "min_pnl_c_per_contract": c.breaker_min_pnl_c}}

    def timing_status(self) -> dict:
        """How fresh our data is when we act: step time, book age at each order update, API round trips."""
        def pct(xs):
            if not xs:
                return None
            v = sorted(xs)
            return {"n": len(v), "p50": round(v[len(v) // 2], 2), "p95": round(v[min(len(v) - 1, int(0.95 * len(v)))], 2),
                    "max": round(v[-1], 2)}
        return {"step_s": pct(self.__dict__.get("t_step", ())), "book_age_at_orders_s": pct(self.__dict__.get("t_book_age", ())),
                "api_round_trip_ms": self.rest.rtt_ms() if hasattr(self.rest, "rtt_ms") else None}

    def h_held_live(self, event: str) -> list:
        """This event's held contracts (main paper broker or live account) as Sleeve H's (ticker, side, price, n)."""
        out = []
        for p in self.positions.values():
            if p.event != event:
                continue
            if p.yes_qty > 0:
                out.append((p.ticker, "bid", int(round(100 * p.yes_cost / p.yes_qty)), float(p.yes_qty)))
            if p.no_qty > 0:
                out.append((p.ticker, "ask", int(round(100 - 100 * p.no_cost / p.no_qty)), float(p.no_qty)))
        return out

    def h_status(self) -> dict:
        if not self.h_on:
            return {"on": False}
        day0 = time.time() - 86400
        out = {"on": True}
        for variant in (("H", "H2") if getattr(self, "h2_on", False) else ("H",)):
            _, held = self._h(variant)
            pnl, n = self.store.hourly_pnl(variant=variant)
            pnl1, _ = self.store.hourly_pnl(day0, variant=variant)
            v = variant.lower()
            out[variant] = {"quotes": self.counters.get(f"{v}_quotes", 0),
                            "fills_since_start": self.counters.get(f"{v}_fills", 0), "open_events": len(held),
                            "settled_pnl_usd": round(pnl, 2), "settled_contracts": n, "pnl_24h_usd": round(pnl1, 2)}
        if getattr(self, "index_feed", None) is not None:
            out["index_history_h"] = {c: round(self.index_feed.hours_of_history(c), 1) for c in self.index_feed.cities}
        return out

    def risk_context(self, desired: list[DesiredOrder]):
        """Everything the caps are checked against: positions and committed orders per event, each event's outcome
        structure, the open worst case and the cash in use. Committed orders (live) are the ones the exchange may
        still fill that `desired` does not stand for: orders being replaced or cancelled, unconfirmed cancels and
        unanswered placements (Oct 4 review: until then the risk filter saw positions plus desired orders only)."""
        pos_by_ev = defaultdict(list)
        for p in self.positions.values():
            pos_by_ev[p.event].append(p)
        base = defaultdict(list)
        if self.mode == "live" and self.live is not None:
            for o in self.live.committed(desired):
                base[o.event].append(o)
        tick_by_ev = {}
        for ev in self.events.values():
            tick_by_ev[ev.event] = self.ev_tickers.get(ev.event) or self.event_tickers(ev)
        for ev in self.hourly.values():
            tick_by_ev[ev.event] = self.ev_tickers.get(ev.event) or ThresholdLadder.from_markets(ev.markets)
        evs = set(pos_by_ev) | set(base)
        for e in evs:
            if e not in tick_by_ev:
                tick_by_ev[e] = self.tickers_for_event(e)   # None = unknown: full cost counts
        open_risk = sum(event_worst_case(pos_by_ev.get(e, []), base.get(e, []), tick_by_ev.get(e)) for e in evs)
        cash_used = cash_in_use(list(self.positions.values()), tick_by_ev, self.netting)
        cash_used += pending_cash([o for v in base.values() for o in v])
        return pos_by_ev, base, tick_by_ev, open_risk, cash_used

    def a_subcap(self, desired, pos_by_ev, base, tick_by_ev, cash_used, cash_cap, inc, rank, keep, cap):
        """Sleeve A's own open-risk cap inside the shared one (sleeve_a.max_total_open_risk_usd): its orders first pass
        a filter over its own cities' events with the total cap lowered to `cap`, so the takers keep the rest of the
        shared room (Aug-Sep replay at live caps: $110 of $150 for A added +$2.56 a day [+0.42, +3.47] week blocks,
        almost all of it to the takers, A's own earnings unchanged; docs/results/r12_x_priority_check.txt). The
        shared filter still runs on everything afterwards."""
        a_orders = [o for o in desired if o.sleeve == "A"]
        if not a_orders:
            return desired
        a_pos, a_base, a_risk, a_cash = self.sub_context(pos_by_ev, base, tick_by_ev,
                                                          getattr(self, "series_a_own", set()))
        risk_cap = float(cap) if cap is not None else self.risk_cfg.max_total_open_risk_usd
        risk_a = RiskConfig(**{**self.risk_cfg.__dict__, "max_total_open_risk_usd": risk_cap})
        cash_a = getattr(self, "a_cash_cap", None)     # sleeve_a.max_cash_usd: A's own cash, inside the shared cap
        cu, cc = (a_cash, min(float(cash_a), cash_cap)) if cash_a is not None else (cash_used, cash_cap)
        a_kept, _ = filter_orders(a_orders, a_pos, tick_by_ev, risk_a, a_risk, cash_used=cu, cash_cap=cc,
                                  netting=self.netting, base_pending=a_base, incumbents=inc, rank=rank, keep=keep)
        self.open_risk_a, self.cash_a = a_risk, a_cash
        return [o for o in desired if o.sleeve != "A"] + a_kept

    def same_weather_pass(self, desired, pos_by_ev, base, tick_by_ev, inc, keep) -> list:
        """The same-weather budget (same_weather:, Oct 11, 2026; wxbot/risk/same_weather.py) over the makers' orders:
        per group of cities and market date, the worst case of the group's events stays within the budget. Each order
        it cuts or drops is logged at INFO, at most every 15 minutes, as the taker's cap refusals are."""
        c = self.cfg_same_weather
        r = same_weather_mod.apply(desired, pos_by_ev, base, tick_by_ev, c, incumbents=inc, keep=keep)
        self.same_weather_usage = r.usage
        for o, got, g, used in r.refused:
            if self.due(f"same_weather_note:{o.ticker}:{o.side}", 900):
                what = "skipped" if got <= 0 else f"cut to {got:g}"
                log.info("QUOTE %s %s buy %s @%dc x%g %s: same-weather budget (%s $%.2f of $%g)", o.sleeve, o.ticker,
                         "YES" if o.side == "bid" else "NO", o.price_cents if o.side == "bid" else 100 - o.price_cents,
                         o.count, what, c.name(g), used, c.budget_usd)
        return r.kept

    def sub_context(self, pos_by_ev, base, tick_by_ev, series: set):
        """The same context restricted to events of the given series (a sleeve's own budget)."""
        keep = lambda e: e.split("-")[0] in series
        pos = {e: v for e, v in pos_by_ev.items() if keep(e)}
        bas = {e: v for e, v in base.items() if keep(e)}
        evs = set(pos) | set(bas)
        risk = sum(event_worst_case(pos.get(e, []), bas.get(e, []), tick_by_ev.get(e)) for e in evs)
        cash = cash_in_use([p for v in pos.values() for p in v], tick_by_ev, self.netting)
        cash += pending_cash([o for v in bas.values() for o in v])
        return pos, bas, risk, cash

    def cash_cap(self) -> float:
        cap = self.risk_cfg.max_cash_deployed_usd
        if self.mode == "live" and getattr(self, "live_equity", None):
            cap = min(cap, self.live_equity - self.risk_cfg.min_cash_reserve_usd)
        return cap

    def incumbent_keys(self) -> set:
        """(ticker, side, price) of orders already resting: the risk filter keeps them first (queue priority)."""
        if self.mode == "live" and self.live is not None:
            return {(o.ticker, o.side, o.price_cents) for o in self.live.resting_list() if o.state == "resting"}
        if self.paper is not None:
            return {(o.ticker, o.side, o.price_cents) for o in self.paper.resting()}
        return set()

    def resting_own(self) -> dict:
        """{(ticker, side, price): our resting contracts there that the displayed book includes} for the reachability
        rule: live orders by their size, paper orders as 0 (they are not in the real book); the key marks an order
        already resting."""
        own: dict = defaultdict(float)
        if self.mode == "live" and getattr(self, "live", None) is not None:
            for o in self.live.resting_list():
                if o.state == "resting":
                    own[(o.ticker, o.side, o.price_cents)] += o.remaining
        elif getattr(self, "paper", None) is not None:
            for o in self.paper.resting():
                own[(o.ticker, o.side, o.price_cents)] += 0.0
        return dict(own)

    def reach_rank(self, desired: list[DesiredOrder]) -> dict:
        """Sleeve A's quotes that a normal sweep can reach (Oct 7, first live day): {(ticker, side, price): contracts
        ahead (sleeve_a.reach)} for each A order with at most cfg_a.max_contracts_ahead ahead; the others are left
        out, and the risk filter gives the budget to the fewest ahead first. Sleeve A's proposals already leave out
        brackets out of reach (decide); this ranks across events, against the same books. An order whose book is
        missing ranks last among those kept."""
        cap = float(self.cfg_a.max_contracts_ahead)
        books = {t: b for ev in self.events.values() for t, b in ev.books.items()}
        own = self.resting_own()
        out, skipped = {}, list(getattr(self, "out_of_reach", []) or [])
        keep_on = bool(getattr(self.cfg_a, "keep_resting", False))
        ranked = {}       # keep_resting: the ranking values (sleeve_a.keep_rank) of the quotes kept
        for o in desired:
            if o.sleeve != "A":
                continue
            k = (o.ticker, o.side, o.price_cents)
            b = books.get(o.ticker)
            if b is None:
                out[k] = cap
                if keep_on:     # last among those kept, as above: a resting order's limit is twice the cutoff
                    ranked[k] = sleeve_a.keep_rank(2 * cap if k in own else cap, k in own,
                                                   self.cfg_a.keep_resting_ratio)
                continue
            eff = sleeve_a.reach(b, o.side, o.price_cents, o.ticker, own)
            if eff > cap:
                skipped.append((o.ticker, int(eff)))
                continue
            out[k] = eff
            if keep_on:     # the cutoff above is unchanged; only the order of the budget changes
                ranked[k] = sleeve_a.keep_rank(sleeve_a.contracts_ahead(b, o.side, o.price_cents, own.get(k, 0.0)),
                                               k in own, self.cfg_a.keep_resting_ratio)
        self.reach_stats = {"max_contracts_ahead": cap, "kept": len(out), "skipped": len(skipped),
                            "ahead_of_kept": sorted(int(v) for v in out.values())[:20],
                            "skipped_examples": sorted(skipped, key=lambda x: x[1])[:10]}
        return ranked if keep_on else out

    def keep_resting_sizes(self) -> Optional[dict]:
        """The resting orders the risk filter leaves alone (sleeve_a / sleeve_t keep_resting): {(ticker, side,
        price): (contracts resting, deadband)} for the sleeves with the option on; None when no sleeve has it, so
        filter_orders runs exactly as before."""
        cfgs = {s: c for s, c in (("A", getattr(self, "cfg_a", None)), ("T", getattr(self, "cfg_t", None)))
                if c is not None and getattr(c, "keep_resting", False)}
        if not cfgs:
            return None
        if self.mode == "live" and getattr(self, "live", None) is not None:
            rest = [o for o in self.live.resting_list() if o.state == "resting"]
        elif getattr(self, "paper", None) is not None:
            rest = self.paper.resting()
        else:
            rest = []
        size, sleeve = defaultdict(float), {}
        for o in rest:
            if o.sleeve in cfgs and o.remaining > 0:
                k = (o.ticker, o.side, o.price_cents)
                size[k] += o.remaining
                sleeve[k] = o.sleeve
        return {k: (r, sleeve_a.keep_deadband(r, cfgs[sleeve[k]])) for k, r in size.items()}

    def execute(self, desired: list[DesiredOrder]):
        now = time.time()
        if getattr(self, "last_book_ts", 0.0):
            self.__dict__.setdefault("t_book_age", collections.deque(maxlen=720)).append(now - self.last_book_ts)
        stopped = self.stopped_sleeves() if hasattr(self, "cfg") else set()
        stopped = stopped | self.shutoff_blocked()    # empty unless shutoff.enabled switched a sleeve off
        self.last_stopped = stopped
        if stopped:
            desired = [o for o in desired if o.sleeve not in stopped]
        pos_by_ev, base, tick_by_ev, open_risk, cash_used = self.risk_context(desired)
        cash_cap = self.cash_cap()
        inc = self.incumbent_keys()
        keep = self.keep_resting_sizes()     # None unless a daily maker keeps its resting orders (off by default)
        t_orders = [o for o in desired if o.sleeve == "T"]
        if t_orders:
            t_pos, t_base, t_risk, t_cash = self.sub_context(pos_by_ev, base, tick_by_ev, self.series_t)
            t_kept, _ = filter_orders(t_orders, t_pos, tick_by_ev, self.risk_t, t_risk, cash_used=t_cash,
                                      cash_cap=self.cash_cap_t, netting=self.netting, base_pending=t_base,
                                      incumbents=inc, keep=keep)
            desired = [o for o in desired if o.sleeve != "T"] + t_kept
            self.open_risk_t = t_risk
        h_orders = [o for o in desired if o.sleeve == "H"]
        if h_orders:
            h_pos, h_base, h_risk, h_cash = self.sub_context(pos_by_ev, base, tick_by_ev, set(self.cfg_h.series))
            risk_h = RiskConfig(max_loss_per_bracket_usd=self.cfg_h.cap_usd, max_loss_per_event_usd=self.cfg_h.cap_usd,
                                max_total_open_risk_usd=self.cfg_h.total_risk_usd)
            h_kept, _ = filter_orders(h_orders, h_pos, tick_by_ev, risk_h, h_risk, cash_used=h_cash,
                                      cash_cap=self.cfg_h.max_cash_usd, netting=self.netting, base_pending=h_base,
                                      incumbents=inc)
            desired = [o for o in desired if o.sleeve != "H"] + h_kept
            self.open_risk_h = h_risk
        cfg_a = getattr(self, "cfg_a", None)
        rank = self.reach_rank(desired) if cfg_a is not None and cfg_a.max_contracts_ahead is not None else None
        if rank is not None:          # quotes no normal sweep can reach are left out (Oct 7)
            desired = [o for o in desired if o.sleeve != "A" or (o.ticker, o.side, o.price_cents) in rank]
        if getattr(self, "a_open_risk_cap", None) is not None or getattr(self, "a_cash_cap", None) is not None:
            # off unless sleeve_a.max_total_open_risk_usd or sleeve_a.max_cash_usd is set
            desired = self.a_subcap(desired, pos_by_ev, base, tick_by_ev, cash_used, cash_cap, inc, rank, keep,
                                    getattr(self, "a_open_risk_cap", None))
        if getattr(getattr(self, "cfg_same_weather", None), "enabled", False):
            # after each maker's own caps, before the account's (off unless same_weather.enabled)
            desired = self.same_weather_pass(desired, pos_by_ev, base, tick_by_ev, inc, keep)
        kept, dropped = filter_orders(desired, pos_by_ev, tick_by_ev, self.risk_cfg, open_risk,
                                      cash_used=cash_used, cash_cap=cash_cap, netting=self.netting,
                                      base_pending=base, incumbents=inc, rank=rank, keep=keep)
        self.cash_used = cash_used
        if self.risk.halted:
            kept = []
        if self.mode == "paper":
            self.paper_reconcile(kept, now)
        else:
            expiry = {m["ticker"]: int(ev.close_ts) for ev in list(self.events.values()) + list(self.hourly.values())
                      for m in ev.markets}
            fresh = self.account_fresh(now)
            stops_ready = len(getattr(self, "eq_readings", [])) >= 3   # the stops have checked a median first
            allow_new = (fresh and stops_ready and self.avail_balance is not None
                         and self.avail_balance >= self.risk_cfg.min_cash_reserve_usd)
            if not allow_new and kept and self.due("no_new_note", 300):
                why = ("account state stale (orders, balance or positions not read in the last "
                       f"{self.account_stale_s:.0f} s, or Kalshi's account data lagging)" if not fresh else
                       "waiting for three equity readings before the first order" if not stops_ready else
                       "available balance unknown" if self.avail_balance is None else
                       f"available balance ${self.avail_balance:.2f} below the reserve")
                self.store.log("WARN", f"no new orders: {why}")
            stats = self.live.reconcile(kept, expiry, allow_new=allow_new, keep_alive=fresh)
            if stats.get("refusal_stop"):
                # the exchange refused order after order: treat trading as paused until a fresh status read says
                # otherwise (and restart the 10-minute wait before a status "open" overrides Kalshi's published
                # hours), so a pause with a failing or contradicting status costs three refusals per wait, never
                # an error-burst halt (round-9 review)
                self.trading_active, self.trading_active_ts, self.trading_off_by = False, now, "refusals"
                if "sched_closed_since" in self.__dict__:
                    self.sched_closed_since = now
                self.next_due["exch"] = now + 60
                # refusals that keep coming with the exchange open are the bot's problem: say so (once in 3 h)
                rs = [t for t in getattr(self, "refusal_stops", []) if t > now - 1800] + [now]
                self.refusal_stops = rs
                if len(rs) >= 8 and getattr(self, "alerts", None) is not None and self.due("refusal_alert", 3 * 3600):
                    self.alerts.send("refusals", f"wxbot {self.alert_name}: Kalshi keeps refusing orders",
                                     f"{len(rs)} times in 30 min Kalshi refused order after order. Last error: "
                                     f"{getattr(self.live, 'last_error', '')}"[:400] + ". Run the health check and "
                                     "send me the log.", priority="high", tags="warning")
            foa = getattr(self.live, "filled_on_arrival", 0)
            if foa > getattr(self, "foa_seen", 0):
                # a post-only order filled at once: the maker sleeves are taking liquidity, which they must never
                # do (the order is still counted until the account shows it; the Stage 2 gate fails on it)
                self.foa_seen = foa
                self.store.log("ERROR", f"post-only order filled on arrival ({foa} since start): post-only is not "
                                        "holding on Kalshi")
                if getattr(self, "alerts", None) is not None:
                    self.alerts.send("post_only", f"wxbot {self.alert_name}: a post-only order filled at once",
                                     f"{foa} since start. Maker orders should never take liquidity: check the log and "
                                     f"the Kalshi app ({self.hint('stop')} stops trading)", priority="high",
                                     tags="warning")
            if not fresh:
                # how long the bot has not known its account: the oldest of the three reads (a read that never
                # succeeded counts from the start), or how far Kalshi's account data lags when that is the reason
                t0 = getattr(self, "started", now)
                blind = now - min([v or t0 for v in getattr(self, "acct_ts", {}).values()] or [t0])
                as_of = getattr(self.live, "last_as_of", 0.0) or 0.0
                if as_of and getattr(self, "as_of_moves_idle", True) is not False:
                    blind = max(blind, now - as_of)
                if blind > 120 and getattr(self, "alerts", None) is not None and self.due("blind_alert", 1800):
                    self.alerts.send("blind", f"wxbot {self.alert_name}: account reads failing",
                                     f"orders, balance or positions not read for {blind / 60:.0f} min: no new orders, "
                                     "and resting ones are not kept alive (they lapse within their TTL)",
                                     priority="high", tags="warning")
            self.counters["placed"] += stats["placed"]
            self.add_activity(stats)
            self.last_reconcile = stats
            if self.live.errors_last_minute() > self.max_err:
                self.halt("API error burst: pausing 15 min", kind="pause", pause_s=900)
                self.live.errors.clear()
        if self.due("quotelog", 300):
            self.store.add_quotes(self.mode, kept, now)
        self.counters["quotes"] = len(kept)
        self.open_risk = open_risk
        self.watch_quoting(len(kept), now)

    def add_activity(self, stats: dict):
        """Cumulative order activity in this bot's database (kv live_activity: placed, amended, cancelled, expiry
        refreshes, refusals, and alive_s, the time actually running), saved every five minutes and kept across
        restarts, counted afresh for each code version. The demo gate reads it to see that the two-day demo test ran
        48 hours on the code GO_LIVE will run and really exercised placing, refreshing and cancelling."""
        now = time.time()
        act = getattr(self, "activity", None)
        if act is None:
            act = self.activity = dict(self.store.kv_get("live_activity", {}) or {})
        code = self.code_version() if not hasattr(self, "_code") else self._code
        self._code = code
        if act.get("code") != code:          # new code: the count (and the demo test's two days) starts again
            act.clear()
            act.update({"code": code, "since": now, "alive_s": 0.0})
        for k in ("placed", "amended", "grown", "cancelled", "refreshed", "refused", "rejected"):
            act[k] = int(act.get(k, 0) or 0) + int(stats.get(k) or 0)
        # running time: the gap since the last pass counts only when short (an asleep or stopped Mac does not)
        gap = now - float(act.get("ts") or now)
        act["alive_s"] = round(float(act.get("alive_s") or 0.0) + (gap if 0 < gap <= 60 else 0.0), 1)
        act["ts"] = now
        if self.due("activity_save", 300):
            self.store.kv_set("live_activity", act)

    def watch_quoting(self, n: int, now: float):
        """Live: an alert when the bot has wanted no order at all for two hours while it runs, unhalted, on an open
        exchange (full caps, every event skipped, account reads stale, no forecasts...): nothing else would say so
        (round-7 pre-mortem). Once per six hours."""
        stopped = getattr(self, "last_stopped", set())
        quoting = {"A"} | ({"T"} if getattr(self, "series_t", None) else set()) | (
            {"H"} if getattr(getattr(self, "cfg_h", None), "enabled", False) else set())
        if n > 0 or self.risk.halted or quoting <= stopped or not hasattr(self, "quoting_since"):
            self.quoting_since = now          # a halt, a sleeve's daily stop (or a pause, which skips this): known
            return
        if self.mode != "live" or now - self.quoting_since < 7200:
            return
        if getattr(self, "alerts", None) is not None and self.due("no_quotes_alert", 6 * 3600):
            self.alerts.send("no_quotes",
                             f"wxbot {self.alert_name}: no orders for {(now - self.quoting_since) / 3600:.0f} h",
                             "The bot is running but has wanted no order at all. Run the health check "
                             f"({self.hint('check')}): caps full, account reads stale, or every "
                             "event skipped are the usual reasons.", priority="default", tags="zzz")

    def paper_reconcile(self, kept: list[DesiredOrder], now: float):
        from ..execution.live import is_patient
        want = {o.key: o for o in kept}
        by_key = {}
        for o in self.paper.resting():
            by_key.setdefault((o.ticker, o.side), []).append(o)
        held = set()
        for k, lst in by_key.items():
            d = want.get(k)
            stale = [o for o in lst if d is not None and o.price_cents != d.price_cents]
            if stale and is_patient(d, stale, now, getattr(self, "patient_s", 0.0),
                                    getattr(self, "patient_sleeves", ("A", "T")), ts_attr="placed_ts"):
                held.add(k)
                continue
            for o in lst:
                if d is None or o.price_cents != d.price_cents:
                    self.paper.cancel(o.oid)
        books = {}
        for ev in list(self.events.values()) + list(self.hourly.values()):
            books.update(ev.books)
        for k, d in want.items():
            if k in held:
                continue
            lst = [o for o in by_key.get(k, []) if o.remaining > 0 and o.price_cents == d.price_cents]
            have = sum(o.remaining for o in lst)
            if have <= 0.5:
                self.paper.place(d, books.get(d.ticker, {}), now)
                self.counters["placed"] += 1
            elif have > d.count + 0.5:
                newest = max(lst, key=lambda o: o.placed_ts)
                self.paper.amend_size(newest.oid, max(newest.remaining - (have - d.count), 0))
            elif have < 0.5 * d.count:
                self.paper.place(DesiredOrder(**{**d.__dict__, "count": d.count - have}), books.get(d.ticker, {}), now)

    # ------------------------------------------------------------ settlement
    def settle(self):
        """Paper: realize P&L on markets that have a result and drop them from open positions.
        Live: Kalshi settles; we record its settlement records for the markets the bot traded."""
        if self.mode == "live":
            return self.settle_live()
        done = self.store.settled()
        for t, p in list(self.positions.items()):
            if not (p.yes_qty or p.no_qty):
                continue
            if t in done:
                self.positions.pop(t, None)
                continue
            try:
                m = self.rest.market(t)
            except Exception:
                continue
            res = m.get("result")
            if res in ("yes", "no"):
                self.store.set_settlement(t, t.rsplit("-", 1)[0], res, time.time(), m.get("expiration_value"))
                pnl = (p.yes_qty if res == "yes" else 0) + (p.no_qty if res == "no" else 0) - p.yes_cost - p.no_cost - p.fees
                self.risk.realized_today += pnl
                self.risk.realized_total += pnl
                self.positions.pop(t, None)
                self.fair_cache.pop(t, None)
                self.store.log("INFO", f"settled {t} result={res} pnl={pnl:.2f}")

    def settle_live(self):
        """Record settled markets the bot traded, with the bot's P&L computed from its own fills (Oct 4 review,
        finding 2.13: Kalshi's settlement record totals the whole account, manual trades included)."""
        try:
            recs = self.rest.settlements(min_ts=int(time.time()) - 4 * 86400)
        except Exception as e:
            log.warning("settlements fetch failed: %s", e)
            return
        mine = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])   # yes qty, yes cost, no qty, no cost, fees
        for (ts, mode, t, ev, side, pc, n, sl, tk, fee, ref, fair) in self.store.fills("live"):
            a = mine[t]
            if side == "bid":
                a[0] += n
                a[1] += n * pc / 100.0
            else:
                a[2] += n
                a[3] += n * (100 - pc) / 100.0
            a[4] += fee or 0.0
        for r in recs:
            t = r.get("ticker")
            if t not in mine:
                continue  # manual trades in the same account are not the bot's P&L
            yq, yc, nq, nc, fee = mine[t]
            res = r.get("market_result")
            if res == "yes":
                rev = yq
            elif res == "no":
                rev = nq
            else:
                rev = yc + nc   # void / no result: assume a refund at cost
            ts = ts_of(r.get("settled_time")) or time.time()
            ev = r.get("event_ticker") or t.rsplit("-", 1)[0]
            if self.store.add_live_settlement(t, ev, res, ts, rev, yc + nc, fee):
                self.store.set_settlement(t, ev, res, ts, r.get("value"))
                acct = float(r.get("revenue") or 0) / 100.0 - float(r.get("yes_total_cost_dollars") or 0) \
                    - float(r.get("no_total_cost_dollars") or 0) - float(r.get("fee_cost") or 0)
                self.store.log("INFO", f"settled {t} result={res} bot pnl={rev - yc - nc - fee:.2f} "
                                       f"(account record {acct:.2f})")
        self.risk.realized_total = self.store.live_pnl()

    def restore_paper_state(self):
        """Rebuild paper positions and realized P&L from the DB so a restart does not forget them."""
        settled = {t: (res, ts) for t, res, ts in
                   self.store.c.execute("SELECT ticker, result, settled_ts FROM settlements").fetchall()}
        pos: dict[str, Position] = {}
        for (ts, mode, t, ev, side, pc, n, sl, tk, fee, ref, fair) in self.store.fills("paper"):
            if sl == "B":
                continue
            p = pos.setdefault(t, Position(t, ev))
            if side == "bid":
                p.yes_qty += n
                p.yes_cost += n * pc / 100.0
            else:
                p.no_qty += n
                p.no_cost += n * (100 - pc) / 100.0
            p.fees += fee or 0.0
        today = et_day()
        for t, p in pos.items():
            if t in settled:
                res, sts = settled[t]
                pnl = (p.yes_qty if res == "yes" else 0) + (p.no_qty if res == "no" else 0) - p.yes_cost - p.no_cost - p.fees
                self.risk.realized_total += pnl
                if sts and et_day(sts) == today:
                    self.risk.realized_today += pnl
            else:
                self.positions[t] = p
        if pos:
            self.store.log("INFO", f"restored {len(self.positions)} open paper positions, "
                                   f"realized ${self.risk.realized_total:.2f}")

    HALT_RANK = {"pause": 1, "daily": 2, "hard": 3}


    def step_failed(self, e: Exception):
        """A step ended in an error of the bot's own (not the network's): count it; repeated failures alert, then
        pause, then halt (live)."""
        log.error("step failed: %s", e, exc_info=e)
        self.store.log("ERROR", f"step failed: {e!r}")
        self.counters["step_errors"] += 1
        self.step_error_ts = [t for t in self.step_error_ts if t > time.time() - 600] + [time.time()]
        if len(self.step_error_ts) >= 5:
            self.alerts.send("step_errors", f"wxbot {self.alert_name}: repeated errors",
                             f"{len(self.step_error_ts)} step failures in 10 min. Last: {e!r}"[:500],
                             priority="high", tags="rotating_light")
        if len(self.step_error_ts) >= 10 and self.live:
            # 10 failures in 10 minutes (the count used to be cumulative since start, so a long run could
            # hard-halt on scattered transient errors): pause 30 minutes; a second burst within 2 hours halts
            if time.time() - getattr(self, "last_err_pause", 0.0) < 7200:
                self.halt(f"repeated step failures again (last: {e!r})"[:300], kind="hard")
            else:
                self.last_err_pause = time.time()
                self.halt(f"repeated step failures: pausing 30 min (last: {e!r})"[:300], kind="pause",
                          pause_s=1800)
            self.step_error_ts = []

    def network_down(self, e: Exception):
        now = time.time()
        if not getattr(self, "net_down_since", None):
            self.net_down_since = now
            log.warning("network or Kalshi unreachable (nothing trades until it is back; resting orders lapse within "
                        "their TTL): %r", e)
            self.store.log("WARN", f"network or Kalshi unreachable: {e!r}"[:300])
        self.counters["net_errors"] += 1
        down = now - self.net_down_since
        if down > 900 and self.due("net_alert", 3600):
            self.alerts.send("net_down", f"wxbot {self.alert_name}: no connection for {down / 60:.0f} min",
                             f"Kalshi or the network has been unreachable since {time.strftime('%H:%M', time.localtime(self.net_down_since))}. "
                             "Nothing trades meanwhile and resting orders lapse on their own. Check the Mac's "
                             "Wi-Fi; the bot resumes by itself.", priority="high", tags="warning")

    def network_up(self):
        if getattr(self, "net_down_since", None):
            self.store.log("INFO", f"network back after {(time.time() - self.net_down_since) / 60:.0f} min")
            self.net_down_since = None

    def alert_upkeep(self):
        """Resend alerts the network swallowed, and remind about a hard halt that is still on."""
        try:
            if self.alerts.pending and self.due("alert_flush", 60):
                self.alerts.flush()
            self.remind_halt()
            # the 'alive' ping for an outside check, sent from the loop itself so that a paused exchange (Thursday
            # 3-5 AM) or a failing step does not look like a dead Mac (round-9 review)
            if self.due("heartbeat", 600):
                self.alerts.heartbeat(f"alive mode={self.mode} halted={int(self.risk.halted)} "
                                      f"fresh={int(self.account_fresh(time.time()))} ts={int(time.time())}")
        except Exception as e:
            log.debug("alert upkeep failed: %s", e)

    def remind_halt(self):
        """A hard halt waits for a person: a reminder every 4 hours between 8 AM and 10 PM ET."""
        if not (self.risk.halted and self.risk.halt_kind == "hard"):
            return
        if "halt_reminder" not in self.next_due:
            self.next_due["halt_reminder"] = time.time() + 4 * 3600     # the halt itself already alerted
            return
        if 8 <= to_et(time.time()).hour < 22 and self.due("halt_reminder", 4 * 3600):
            self.alerts.send("halt_reminder", f"wxbot {self.alert_name} is still halted",
                             f"{self.risk.halt_reason}. Nothing trades until you look at it and run: "
                             f"{self.hint('reset-halt')} ({self.hint('stop')} stops it instead).",
                             priority="high", tags="warning")

    def halt(self, why: str, kind: str = "hard", pause_s: float = 0.0):
        """Stop trading and cancel everything. Halts never downgrade (hard > daily > pause: an API-error pause after
        a hard halt used to replace it and resume 15 minutes later) and survive restarts (kv halt_latch; clear a hard
        halt by hand with python -m wxbot reset-halt). Oct 4 review, finding 2.6."""
        cur = self.risk.halt_kind if self.risk.halted else ""
        rank = self.HALT_RANK.get(kind, 3)
        if self.risk.halted and rank < self.HALT_RANK.get(cur, 3):
            log.warning("already halted (%s); not downgraded by: %s", cur, why)
            self.store.log("WARN", f"halt ({kind}) ignored: already halted ({cur}): {why}")
        else:
            self.risk.halted = True
            self.risk.halt_reason = why
            self.risk.halt_kind = kind
            if kind == "pause":
                self.risk.resume_ts = max(self.risk.resume_ts if cur == "pause" else 0.0, time.time() + pause_s)
            if kind == "daily":
                self.risk.halt_day = self.risk.halt_day or et_day().isoformat()
            self.store.kv_set("halt_latch", {"kind": kind, "reason": why, "ts": time.time(),
                                             "day": self.risk.halt_day, "resume_ts": self.risk.resume_ts})
        self.store.log("ERROR", f"HALT: {why}")
        log.error("HALT: %s", why)
        self.alerts.send(f"halt:{kind}", f"wxbot {self.alert_name} HALT ({kind})", why, priority="high", tags="warning")
        if self.live:
            left = self.live.cancel_all(why)
            if left != 0:
                msg = (f"after the halt {'some' if left < 0 else left} bot orders could not be confirmed cancelled; "
                       "the bot keeps retrying and they expire on their own within the order TTL. Check the Kalshi "
                       "app.")
                self.store.log("ERROR", msg)
                self.alerts.send("halt:leftover", f"wxbot {self.alert_name}: orders may still rest", msg,
                                 priority="high", tags="warning")

    def restore_halt(self):
        """A halt from before a restart: hard stays until reset-halt; daily until the next ET day; pause until its
        resume time."""
        h = self.store.kv_get("halt_latch") or {}
        kind = h.get("kind")
        if not kind:
            return
        if (kind == "daily" and h.get("day") != et_day().isoformat()) or \
                (kind == "pause" and time.time() >= float(h.get("resume_ts") or 0)):
            self.store.kv_set("halt_latch", {})
            return
        self.risk.halted, self.risk.halt_kind, self.risk.halt_reason = True, kind, h.get("reason", "")
        self.risk.halt_day, self.risk.resume_ts = h.get("day") or "", float(h.get("resume_ts") or 0)
        self.store.log("WARN", f"started halted ({kind}) from before the restart: {self.risk.halt_reason}")

    def maybe_resume(self, now: float, today: str) -> bool:
        if self.risk.maybe_resume(now, today):
            self.store.kv_set("halt_latch", {})
            self.store.log("INFO", "resumed after halt")
            return True
        return False

    # ------------------------------------------------------- stop guard (risk keys of Oct 11, 2026)
    def guard_reading(self, now: float, today: str):
        """One equity reading for the stop guard (wxbot/risk/stop_guard.py). Its notes (a stop's condition starting to
        hold while it must hold a while, or clearing before it halted) go to the log, so a phantom dip leaves a
        trace."""
        for note in self.stop_guard.observe(now, self.risk, today):
            log.warning(note)
            self.store.log("WARN", note)

    def daily_resumes(self, today: str) -> int:
        """Automatic resumes of a daily halt so far this ET day (kv daily_resumes, kept across restarts)."""
        d = self.store.kv_get("daily_resumes") or {}
        return int(d.get("n") or 0) if d.get("day") == today else 0

    def guarded_stop(self, today: str):
        """The account stops through the stop guard: RiskState.check_stops's two conditions, the drawdown stop first,
        each halting only once it has held on the readings for its confirmation time (0: at once)."""
        g = self.stop_guard
        kind = g.due(self.risk)
        if kind is None:
            return
        if kind == "daily":
            self.risk.halt_day = today        # as check_stops sets it (halt() would keep an older day)
        self.halt(g.reason(kind, self.risk, g.cfg.max_resumes_per_day - self.daily_resumes(today)), kind=kind)

    def maybe_resume_daily(self, today: str) -> bool:
        """risk.daily_resume_margin_usd: lift today's daily halt once the day's settled result is back at
        -(daily_loss_stop_usd - margin) or better on 3 readings in a row and, with risk.daily_resume_hold_s, has stayed
        there that long since the first of them; at most twice an ET day. Kalshi pays out by
        time zone, so a morning's first payouts are not the day (Oct 11, 2026: the losing Eastern bets paid out at
        7:10 AM and the stop halted the whole day at -$20.99; by 10:10, when the West had paid out too, the day stood
        at -$9.87). A hard halt (drawdown stop, QUIESCE, repeated failures) or a pause is never lifted here."""
        g = getattr(self, "stop_guard", None)
        if g is None or not (self.risk.halted and self.risk.halt_kind == "daily"):
            return False
        n = self.daily_resumes(today)
        if not g.resume_due(self.risk, today, n):
            return False
        self.risk.halted, self.risk.halt_reason, self.risk.halt_kind = False, "", ""
        self.store.kv_set("halt_latch", {})
        self.store.kv_set("daily_resumes", {"day": today, "n": n + 1, "ts": time.time()})
        g.above, g.above_since = 0, None
        msg = (f"Resumed: today's settled result recovered to {usd(self.risk.realized_today)} (daily stop "
               f"{usd(-self.risk.cfg.daily_loss_stop_usd)}; automatic resume {n + 1} of {g.cfg.max_resumes_per_day} "
               "today)")
        log.warning(msg)
        self.store.log("WARN", msg)
        al = getattr(self, "alerts", None)
        if al is not None:
            getattr(al, "last_sent", {}).pop("halt:daily", None)     # a daily halt after this one alerts at once
            al.send(f"resume:daily:{n + 1}", f"wxbot {self.alert_name} resumed", msg, priority="high",
                    tags="arrow_forward")
        return True

    def mark_to_market(self) -> float:
        mtm = 0.0
        for t, p in self.positions.items():
            fair = self.fair_cache.get(t)
            if fair is None:
                continue
            mtm += p.yes_qty * fair + p.no_qty * (1 - fair) - p.yes_cost - p.no_cost - p.fees
        return mtm

    def quiesce_requested(self) -> bool:
        """A file named QUIESCE in the data folder hard-halts the bot (cancelling every live order) without stopping
        the process: BACK_TO_PAPER writes it first, so nothing new is placed while it switches the config."""
        flag = os.path.join(os.path.dirname(self.status_path), "QUIESCE")
        if not os.path.exists(flag):
            return False
        try:
            os.remove(flag)
        except OSError:
            pass
        return True

    def restart_requested(self) -> bool:
        """A file named RESTART in the data folder makes the bot shut down cleanly (cancelling live orders) so its
        service manager (launchd / systemd, both set to keep it alive) starts it again on the newly deployed code."""
        flag = os.path.join(os.path.dirname(self.status_path), "RESTART")
        if not os.path.exists(flag):
            return False
        try:
            os.remove(flag)
        except OSError:
            pass
        log.warning("RESTART flag found: shutting down so the service manager restarts the bot")
        self.store.log("INFO", "restart requested via data/RESTART")
        self.running = False
        return True

    def write_status(self):
        """The status file is for people and the health check, never a reason for a step to fail: an error while
        building it is logged and counted (side task 'status'); a status file that stops updating shows as stale."""
        try:
            return self._write_status()
        except Exception as e:
            self.side_error("status", e)
            return {}

    def _write_status(self):
        st = {
            "ts": time.time(), "mode": self.mode, "uptime_min": round((time.time() - self.started) / 60, 1),
            "events": len(self.events), "hourly_events": len(self.hourly), "quotes": self.counters.get("quotes", 0),
            "a_reach": getattr(self, "reach_stats", None),
            "placed_total": self.counters.get("placed", 0), "fills_total": self.store.c.execute("SELECT COUNT(*) FROM fills WHERE mode=?", (self.mode,)).fetchone()[0],
            "fills_since_start": self.counters.get("fills", 0),
            "open_positions": sum(1 for p in self.positions.values() if p.yes_qty or p.no_qty),
            "open_risk_usd": round(getattr(self, "open_risk", 0.0), 2),
            "cash_in_positions_usd": round(getattr(self, "cash_used", 0.0), 2), "netting": self.netting,
            "available_balance_usd": self.avail_balance,
            "realized_usd": round(self.risk.realized_total, 2), "mtm_usd": round(self.mark_to_market(), 2),
            "halted": self.risk.halted, "halt_reason": self.risk.halt_reason, "halt_kind": self.risk.halt_kind,
            "nbm_cycle": self.nbm.nbs_cycle.isoformat() if self.nbm.nbs_cycle else None,
            "nbp_cycle": self.nbm.nbp_cycle.isoformat() if self.nbm.nbp_cycle else None,
            "sleeve_t_series": sorted(self.series_t), "sleeve_t_open_risk_usd": round(getattr(self, "open_risk_t", 0.0), 2),
            # Sleeve A's own caps (sleeve_a.max_total_open_risk_usd / max_cash_usd, Oct 9): None while they are off
            "sleeve_a_open_risk_usd": (round(getattr(self, "open_risk_a", 0.0), 2)
                                       if getattr(self, "a_open_risk_cap", None) is not None
                                       or getattr(self, "a_cash_cap", None) is not None else None),
            "sleeve_a_caps": ({"open_risk": getattr(self, "a_open_risk_cap", None), "cash": getattr(self, "a_cash_cap", None),
                               "cash_in_use": round(getattr(self, "cash_a", 0.0), 2)}
                              if getattr(self, "a_open_risk_cap", None) is not None
                              or getattr(self, "a_cash_cap", None) is not None else None),
            "taker_shadow_logged": len(self.x_seen),
            "sleeve_h_shadow": self.h_status(),
            "sleeve_h_breaker": self.h_breaker_status(),
            "timing": self.timing_status(),
            "ws_shadow": self.ws_status(),
            "side_task_errors": {k[9:]: v for k, v in self.counters.items() if k.startswith("side_err:")},
            "post_only_filled_on_arrival": getattr(getattr(self, "live", None), "filled_on_arrival", 0),
            "exchange_hours": self.schedule_status(),
            "trading_paused": ({"why": self.exchange_paused(time.time()),
                                "since_min": round((time.time() - self.paused_since) / 60, 1)}
                               if getattr(self, "paused_since", None) else None),
            "ml_stale_skips": self.counters.get("ml_stale", 0),
            "round_guard": self.round_guard_status(),
            "peak_memory_mb": peak_memory_mb(),
            "live_activity": dict(getattr(self, "activity", None) or {}) or None,
            "tape_archive": self.tape.status() if getattr(self, "tape", None) is not None else None,
            "rain_shadow": self.rain_status(),
            "sleeve_b_enabled": self.cfg_b.enabled, "sleeve_b_shadow_est_usd": round(self.lip_est_usd, 2),
            "sleeve_b_shadow_fills": self.counters.get("b_fills", 0),
            "sleeve_b_shadow_fill_pnl_usd": round(self.b_realized, 2),
            "api_requests": self.rest.n_requests, "api_errors": self.rest.n_errors, "last_api_error": self.rest.last_error,
            "heartbeat_topic": self.alerts.heartbeat_topic if getattr(self, "alerts", None) is not None else "",
            "ml_engine": getattr(self.ml, "engine", None) if getattr(self, "ml", None) is not None else None,
            "ml_parity_max_diff": getattr(self.ml, "parity_max_diff", None) if getattr(self, "ml", None) is not None
            else None,
            "model_tag": self.model_tag(),
        }
        if getattr(self, "exec_log", None) is not None:
            st["exec_log"] = self.exec_log.status()
        if getattr(getattr(self, "cfg_shutoff", None), "enabled", False):
            st["shutoff"] = {"switched_off": sorted(self.shutoff_blocked()),
                             "sleeves": getattr(self, "shutoff_state", None) or {},
                             "reset": f"{self.hint('reset-shutoff')} <sleeve>"}
        g = getattr(self, "stop_guard", None)
        if g is not None and g.cfg.enabled:      # the risk keys of Oct 11, 2026 (wxbot/risk/stop_guard.py)
            st["stop_guard"] = g.status(self.risk, self.daily_resumes(et_day().isoformat()))
        sw = getattr(self, "cfg_same_weather", None)
        if sw is not None and sw.enabled:        # the same-weather budget (Oct 11, 2026): per group, by market date
            st["same_weather"] = same_weather_mod.status(sw, getattr(self, "same_weather_usage", {}) or {})
        if self.live:
            rest_orders = self.live.resting_list()
            exp = [o.expires_ts for o in rest_orders if o.expires_ts]
            states = collections.Counter(o.state for o in rest_orders)
            now = time.time()
            st.update({"live_order_states": dict(states),
                       "account_fresh": self.account_fresh(now),
                       "equity_readings": len(getattr(self, "eq_readings", [])),
                       "account_read_age_s": {k: round(now - v, 1) for k, v in getattr(self, "acct_ts", {}).items()},
                       "live_equity_mtm_usd": getattr(self, "live_equity_mtm", None),
                       "last_reconcile": getattr(self, "last_reconcile", None),
                       "live_resting_orders": len(rest_orders),
                       "live_resting_cost_usd": round(sum(o.remaining * ((o.price_cents if o.side == "bid" else
                                                                          100 - o.price_cents) / 100.0)
                                                          for o in rest_orders), 2),
                       "live_min_minutes_to_expiry": round((min(exp) - time.time()) / 60, 1) if exp else None,
                       "live_equity_usd": getattr(self, "live_equity", None),
                       "live_equity_adds_resting_usd": getattr(self, "resting_held_usd", None),
                       "balance_semantics": self.store.kv_get("balance_semantics")})
        os.makedirs(os.path.dirname(self.status_path), exist_ok=True)
        with open(self.status_path, "w") as f:
            json.dump(st, f, indent=1)
        return st

    # ------------------------------------------------------------------ run
    def step(self):
        L = self.cfg["loop"]
        if self.due("universe", L["universe_every_s"]):
            self.refresh_universe()
        minute = dt.datetime.now(dt.timezone.utc).minute
        stale_hourly = any(ev.close_ts <= time.time() for ev in self.hourly.values()) or not self.hourly
        if self.due("hourly", 60 if (minute < 7 or stale_hourly) else 300):
            self.refresh_hourly()
        self.refresh_weather()
        self.refresh_ml()
        self.maybe_prefetch_ml()
        if self.due("books", L["books_every_s"]):
            self.refresh_books()
        if (self.mode == "paper" or self.h_on) and self.due("trades", L["trades_every_s"]):
            # in live mode the tape only feeds the hourly shadow: its failures never stop the live step
            if self.mode == "paper":
                self.refresh_trades_paper()
            else:
                self.side_task("trades", self.refresh_trades_paper)
        if self.h_on and self.due("h_settle", 600):
            self.side_task("h_settle", self.settle_h)
        if getattr(self, "index_feed", None) is not None and self.due("index_h", 60):
            self.side_task("index", self.index_feed.poll)
        if self.mode == "live" and self.due("positions", L["positions_every_s"]):
            self.refresh_live_state()
        if self.due("settle", L["settle_every_s"]):
            self.settle()
            self.side_task("settle_b", self.settle_b)
            self.guarded_h_breaker()
            if getattr(getattr(self, "cfg_shutoff", None), "enabled", False):
                self.side_task("shutoff", self.check_shutoff)
        if self.cfg_x.enabled and self.due("x_settle", 1800):
            self.side_task("x_settle", self.settle_taker_shadow)
        if getattr(self, "rain", None) is not None:
            self.side_task("rain", self.refresh_rain)
            if self.due("r_settle", 1800):
                self.side_task("r_settle", self.settle_rain_shadow)
        if (self.cfg.get("early_taker") or {}).get("enabled") and self.due("early_take", 30):
            self.early_take()
        if self.due("paper_report", 3600):
            self.side_task("paper_report", self.refresh_paper_report)
        if self.due("pool_refit_check", 1800):
            self.side_task("pool_refit", self.maybe_refit_pool)
        if self.mode == "paper":
            self.risk.equity = self.risk.realized_total + self.mark_to_market()
            self.risk.peak_equity = max(self.risk.peak_equity, self.risk.equity)
        today = et_day().isoformat()
        if getattr(self, "risk_day", None) != today:
            if getattr(self, "risk_day", None) is not None:
                # new ET trading day; live recomputes it from equity at the next reading. Without this reset in live
                # mode, the first step after midnight could resume a daily halt and halt again on yesterday's number
                self.risk.realized_today = 0.0
            self.risk_day = today
        g = getattr(self, "stop_guard", None)
        guarded = g is not None and g.cfg.guards_stops
        if guarded and self.mode == "paper":
            self.guard_reading(time.time(), today)       # paper has no account reads: each step is one reading
        self.maybe_resume(time.time(), today)
        if guarded:
            # the risk keys of Oct 11, 2026 (wxbot/risk/stop_guard.py): a daily halt can lift the same day, and each
            # stop halts only once its condition has held for its confirmation time
            self.maybe_resume_daily(today)
            if not self.risk.halted:
                self.guarded_stop(today)
        elif not self.risk.halted and self.risk.check_stops(today):
            self.halt(self.risk.halt_reason, kind=self.risk.halt_kind)
        if self.quiesce_requested():
            self.halt("stopped by hand (QUIESCE flag, e.g. BACK_TO_PAPER)", kind="hard")
        # exchange pauses (e.g. Thursday 3-5 AM ET maintenance): do not trade, do not count errors. Kalshi's
        # published hours stop the bot a minute before a scheduled pause; the live status (read every minute, and at
        # once after the exchange refuses several orders in a pass) catches unscheduled ones
        if self.due("schedule", 3600):
            self.side_task("schedule", self.refresh_schedule)
        if self.due("exch", 60):
            try:
                d = self.rest.exchange_status()
                self.trading_active = bool(d.get("trading_active", True)) and bool(d.get("exchange_active", True))
                self.trading_active_ts, self.trading_off_by = time.time(), "status"
            except Exception:
                pass
        now = time.time()
        if now - getattr(self, "trading_active_ts", now) > 300:
            self.trading_active = True     # status unread for 5 min: try (refusals stop a pass and re-read it)
        why = self.exchange_paused(now)
        if why:
            self.quoting_since = now       # not quoting while the exchange is paused is expected
            if "paused_since" not in self.__dict__:
                # start and end lines: the Stage 2 and demo gates leave paused time out of how long the bot took to
                # settle an unconfirmed cancel or an unanswered placement (nothing settles while Kalshi is paused)
                log.info("PAUSE start: %s; execution waits", why)
            since = self.__dict__.setdefault("paused_since", now)
            if self.due("pause_note", 600):
                self.store.log("INFO", f"exchange trading paused ({why}): skipping execution")
            if self.mode == "live" and now - since > 3 * 3600 and self.due("pause_alert", 6 * 3600):
                self.alerts.send("paused_long",
                                 f"wxbot {self.alert_name}: not trading for {(now - since) / 3600:.0f} h",
                                 f"{why}. The bot waits for the exchange; if Kalshi's site shows trading as open, "
                                 "run the health check and tell me.", priority="default", tags="hourglass")
            if self.due("status", 30):
                self.write_status()
            return
        since = self.__dict__.pop("paused_since", None)
        if since is not None:
            log.info("PAUSE end after=%.0fs: execution resumes", now - since)
        desired = self.decide()
        self.execute(desired)
        if self.due("status", 30):
            self.write_status()
        if self.due("dashboard", float(self.cfg.get("dashboard_every_s", 300))):
            try:
                from .dashboard import write_dashboard
                write_dashboard(self.cfg, base_dir=self.base)
            except Exception as e:
                log.warning("dashboard write failed: %s", e)
        pub = self.cfg.get("public_page", {}) or {}
        if pub.get("enabled") and self.ml is not None and self.due("public_page", float(pub.get("every_s", 600))):
            try:
                from .public_page import write as write_public
                with self.ml.lock:
                    cache = dict(self.ml.cache)
                stations = {v["icao"]: v for v in STATIONS.values() if v.get("kind") == "high"}
                st = {"mode": self.mode}
                write_public(os.path.join(self.base, pub.get("dir", "data/public")), cache, stations,
                             self.store.path, st)
            except Exception as e:
                log.warning("public page write failed: %s", e)
        self.side_task("daily_summary", self.maybe_daily_summary)

    def refresh_schedule(self):
        """Kalshi's published trading hours (GET /exchange/schedule; wxbot/exchange_hours.py). A schedule that keeps
        trading closed far longer than Kalshi's two hours a week (or cannot be read) is set aside, loudly: the bot
        then goes by the live exchange status alone (round-8 review)."""
        d = self.rest.exchange_schedule()
        if not d:
            return
        h = exchange_hours.closed_hours_per_week(d, time.time())
        if h is None or h > 12:
            self.schedule = None
            if self.due("schedule_odd", 24 * 3600):
                msg = ("Kalshi's published trading hours look unusual (" + ("unreadable" if h is None else
                       f"{h:.0f} h closed a week") + "): set aside; the bot goes by the live exchange status")
                log.error(msg)
                self.store.log("ERROR", msg)
                self.alerts.send("schedule_odd", f"wxbot {self.alert_name}: unusual exchange hours", msg)
            return
        self.schedule = d

    def exchange_paused(self, now: float) -> Optional[str]:
        """Why execution waits, or None: Kalshi's status says trading is off, or its published hours say closed (or
        closing within a minute). The hours are a head start, not an override: if a fresh status keeps saying
        trading is on ten minutes into a published closure, the hours are set aside until they say open again."""
        status_on = getattr(self, "trading_active", True)
        sched = exchange_hours.closed_now(getattr(self, "schedule", None), now)
        if sched:
            since = self.__dict__.setdefault("sched_closed_since", now)
            if status_on and now - getattr(self, "trading_active_ts", 0.0) < 120 and now - since > 600:
                if self.due("sched_contradiction_note", 3600):
                    self.store.log("WARN", "Kalshi's published hours say closed but its status says trading is on: "
                                           "going by the status")
                sched = False
        else:
            self.__dict__.pop("sched_closed_since", None)
        if not status_on:
            return ("Kalshi refused order after order" if getattr(self, "trading_off_by", "status") == "refusals"
                    else "Kalshi's status: trading paused")
        return "Kalshi's published hours" if sched else None

    def schedule_status(self) -> dict:
        now = time.time()
        c = getattr(self, "_sched_desc", None)
        if c is None or now - c[0] > 600:
            c = (now, exchange_hours.describe(getattr(self, "schedule", None), now))
            self._sched_desc = c
        return c[1]

    def exec_log_pass(self):
        """The execution log's work (exec_log.enabled only), after the step's orders: write the rows its orders and
        decisions left in memory, then the due re-reads, one batch book read if the read limiter can spare it now,
        else skipped. While the network is down, only the writing (the reads would just fail)."""
        xl = getattr(self, "exec_log", None)
        if xl is not None:
            xl.rest = getattr(self, "rest", xl.rest)      # always the bot's own client and read limiter
            self.side_task("exec_log", xl.flush if getattr(self, "net_down_since", None) else xl.run_due)

    def side_task(self, name: str, fn, *a, **kw):
        """Run a task the trading does not depend on (shadow sleeves and their settlement, reports, the weekly pool
        refit, which keeps the old pool when it fails). A failure is logged and counted, never a step failure, so a
        broken side feature cannot pause or halt live trading (found in the Oct 4 dress rehearsal of the Stage 1
        config: a failing trades read, which only the hourly shadow needs, ended every live step)."""
        try:
            return fn(*a, **kw)
        except Exception as e:
            self.side_error(name, e)
            return None

    def side_error(self, name: str, e: Exception):
        self.counters[f"side_err:{name}"] += 1
        if self.due(f"side_err_note:{name}", 900):
            log.warning("%s failed (side task, trading unaffected): %r", name, e)
            self.store.log("WARN", f"{name} failed (side task, trading unaffected): {e!r}"[:300])

    def maybe_daily_summary(self):
        if not self.alerts.enabled:
            return
        et = to_et(time.time())
        day = et.date().isoformat()
        if et.hour < self.alerts.summary_hour_et or self.store.kv_get("summary_day") == day:
            return
        self.store.kv_set("summary_day", day)
        start = et.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        n, c = self.store.c.execute("SELECT COUNT(*), COALESCE(SUM(count),0) FROM fills WHERE mode=? AND ts>=?",
                                    (self.mode, start)).fetchone()
        msg = (f"today ${self.risk.realized_today:+.2f} realized | total ${self.risk.realized_total:+.2f} | "
               f"{n} fills ({c:.0f} contracts) | open risk ${getattr(self, 'open_risk', 0.0):.2f} | "
               f"MTM ${self.mark_to_market():+.2f} | halted={self.risk.halted}")
        try:   # the logged-only sleeves, so the phone summary covers the whole test
            if self.h_on:
                hp, hn = self.store.hourly_pnl(start, variant="H")
                msg += f" | hourly shadow ${hp:+.2f} on {hn:.0f} ct"
            if getattr(self, "rain", None) is not None:
                rp, rn, rs = self.store.rain_pnl()
                msg += f" | rain shadow {rs} signals, settled ${rp:+.2f}"
                if getattr(self, "rain2", None) is not None:
                    rp2, _, rs2 = self.store.rain_pnl("rain_shadow_v2")
                    msg += f" (R2 {rs2}, ${rp2:+.2f})"
            if getattr(self, "cfg_x", None) is not None and self.cfg_x.enabled:
                msg += f" | taker shadow {len(self.x_seen)} signals"
            if getattr(getattr(self, "cfg_shutoff", None), "enabled", False):
                ss = getattr(self, "shutoff_state", None) or {}
                msg += " | shut-off " + ", ".join(
                    f"{s} {'OFF' if v.get('switched_off') else 'on'} ${v.get('S', 0):.0f}/{v.get('h', 0):.0f}"
                    for s, v in sorted(ss.items()))
        except Exception:
            pass
        self.alerts.send("daily", f"wxbot {self.alert_name} daily {day}", msg, priority="low", tags="bar_chart")

    def run(self, max_minutes: Optional[float] = None):
        signal.signal(signal.SIGTERM, lambda *a: setattr(self, "running", False))
        # Ctrl-C in a terminal: finish the step, then cancel and exit like a service stop (a KeyboardInterrupt would
        # skip the shutdown cancel and leave orders to their dead-man expiry)
        def on_int(*a):
            self.running = False
            signal.signal(signal.SIGINT, signal.default_int_handler)   # a second Ctrl-C interrupts a hung step
        signal.signal(signal.SIGINT, on_int)
        if self.live:
            left = self.live.cancel_all("startup clean slate")
            if left != 0:
                # still tracked: they count as exposure, block their keys and are re-cancelled every cycle; new orders
                # wait until the account reads are fresh (account_fresh)
                msg = (f"startup: {'an unknown number of' if left < 0 else left} bot orders could not be confirmed "
                       "cancelled; they count as exposure until they are")
                self.store.log("ERROR", msg)
                self.alerts.send("start:leftover", f"wxbot {self.alert_name}: orders from before the restart", msg,
                                 priority="high", tags="warning")
            try:
                cfgs = self.rest.netting().get("netting_configs") or []
                self.netting = any(c.get("enabled") and int(c.get("subaccount_number", -1)) == 0
                                   and int(c.get("exchange_index", 0)) == 0 for c in cfgs)
            except Exception as e:
                log.warning("netting lookup failed (assuming off): %s", e)
                self.netting = False
            self.store.log("INFO", f"collateral netting {'ON' if self.netting else 'OFF'}")
            # catch fills that happened while the bot was down (orders can fill until their dead-man expiry);
            # re-reading is safe because live fills are de-duplicated by Kalshi fill_id
            last = self.store.c.execute("SELECT MAX(ts) FROM fills WHERE mode='live'").fetchone()[0]
            self.live.last_fill_ts = int(last - 60) if last else int(time.time() - 86400)
        if self.mode == "paper":
            self.restore_paper_state()
        started = f"start mode={self.mode} code={self.code_version()}"
        self.store.log("INFO", started)
        log.info(started)          # the log file too: the gates read restarts from it
        self.alerts.send("start", f"wxbot {self.alert_name} started", f"{len(self.series_a)} series, "
                         f"{len(self.positions)} open positions restored", priority="low")
        t_end = time.time() + 60 * max_minutes if max_minutes else None
        if self.tape is not None:
            self.tape.start(self.tape_every_s)
        if getattr(self, "ws", None) is not None:
            self.ws.start()
        while self.running and (t_end is None or time.time() < t_end):
            t0 = time.time()
            try:
                self.step()
                self.network_up()
            except Exception as e:
                if transient_failure(e):
                    # the network or Kalshi is unreachable: nothing to fix in the bot, and nothing trades meanwhile
                    # (account reads go stale, resting orders lapse within their TTL), so it never counts toward
                    # the step-failure halt: a few minutes of Wi-Fi trouble used to hard-halt it (round-7 pre-mortem)
                    self.network_down(e)
                else:
                    self.step_failed(e)
            self.alert_upkeep()
            self.__dict__.setdefault("t_step", collections.deque(maxlen=720)).append(time.time() - t0)
            self.exec_log_pass()
            if self.restart_requested():
                break
            time.sleep(max(0.5, self.cfg["loop"]["tick_seconds"] - (time.time() - t0)))
        if self.tape is not None:
            self.tape.stop()
        if self.live:
            self.live.cancel_all("shutdown")
        if getattr(self, "exec_log", None) is not None:
            self.side_task("exec_log", self.exec_log.flush)     # after the cancels: rows still in memory are written
        if getattr(self, "ws", None) is not None:
            self.ws.stop()   # after the cancels: stopping it can take a few seconds
        self.write_status()
        self.store.log("INFO", "stopped")
        self.alerts.send("stop", f"wxbot {self.alert_name} stopped", "process exiting (restart it if this was not you)",
                         priority="default")
