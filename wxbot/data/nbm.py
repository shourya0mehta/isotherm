"""NOAA National Blend of Models (NBM v5) text bulletins: fetch + parse.

Free, public-domain, hourly. Files hold every station (~30 MB each), so we download a file once and parse
only the stations we trade. Sources (both free): AWS Open Data and NOMADS.

NBS (short-range, 3-hourly to 72h): TXN (max at the 00Z column, min at the 12Z column) + XND (its std dev),
    TMP/TSD per 3h, P06/P12 precip probabilities.
NBP (percentiles, issued at 00/01, 06/07, 12/13, 18/19Z): TXNMN mean, TXNSD sd, TXNP1/P2/P5/P7/P9 =
    10/25/50/75/90th percentiles, max at the 00Z column, min at the 12Z column.

The 00Z-column max covers the local daytime of the previous US calendar day; the 12Z-column min covers the
night ending that morning. That matches the CLI daily max well; for daily lows the CLI calendar-day min can also
fall late in the evening, so lows get a wider uncertainty floor downstream.
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import requests

log = logging.getLogger(__name__)

AWS = "https://noaa-nbm-grib2-pds.s3.amazonaws.com/blend.{d}/{h}/text/blend_{k}tx.t{h}z"
NOMADS = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/blend/prod/blend.{d}/{h}/text/blend_{k}tx.t{h}z"
CACHE = os.path.expanduser(os.environ.get("WXBOT_CACHE") or "~/.cache/wxbot/nbm")

HDR = re.compile(r"^\s*(K[A-Z0-9]{3})\s+NBM V[\d.]+ (NB[A-Z]) GUIDANCE\s+(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{2})(\d{2}) UTC")


def _url(kind: str, cycle: dt.datetime, src: str) -> str:
    tmpl = AWS if src == "aws" else NOMADS
    return tmpl.format(d=cycle.strftime("%Y%m%d"), h=cycle.strftime("%H"), k=kind)


def fetch(kind: str, cycle: dt.datetime, timeout: float = 60.0) -> Optional[str]:
    """kind: 'nbs' | 'nbp' | 'nbh' | 'nbe'. Returns text or None if not (yet) published."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, f"{kind}_{cycle:%Y%m%d%H}.txt")
    if os.path.exists(path) and os.path.getsize(path) > 100000:
        with open(path, "r", errors="ignore") as f:
            return f.read()
    for src in ("aws", "nomads"):
        try:
            r = requests.get(_url(kind, cycle, src), timeout=timeout)
            if r.status_code == 200 and len(r.content) > 100000:
                txt = r.content.decode("ascii", errors="ignore")
                # written whole, then renamed: the cache is shared by every bot on the Mac (the main bot and
                # the demo bot), and one must never read a file the other is halfway through writing
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    f.write(txt)
                os.replace(tmp, path)
                return txt
        except requests.RequestException as e:
            log.debug("nbm fetch %s %s failed: %s", kind, src, e)
    return None


def latest(kind: str, now: Optional[dt.datetime] = None, lookback_h: int = 8) -> tuple[Optional[dt.datetime], Optional[str]]:
    now = now or dt.datetime.now(dt.timezone.utc)
    base = now.replace(minute=0, second=0, microsecond=0)
    for k in range(lookback_h + 1):
        cyc = base - dt.timedelta(hours=k)
        txt = fetch(kind, cyc)
        if txt:
            return cyc, txt
    return None, None


def prune_cache(keep_hours: int = 48):
    if not os.path.isdir(CACHE):
        return
    cutoff = dt.datetime.now().timestamp() - keep_hours * 3600
    for fn in os.listdir(CACHE):
        p = os.path.join(CACHE, fn)
        try:
            if os.path.getmtime(p) < cutoff:
                os.remove(p)
        except OSError:
            pass        # another bot sharing the cache pruned it first


def station_blocks(text: str, stations: set[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    cur: Optional[str] = None
    for line in text.splitlines():
        m = HDR.match(line)
        if m:
            sid = m.group(1)
            cur = sid if sid in stations else None
            if cur:
                out[cur] = [line]
            continue
        if cur:
            if not line.strip():
                cur = None
                continue
            out[cur].append(line)
    return out


def _issue(header: str) -> dt.datetime:
    m = HDR.match(header)
    mo, d, y, hh, mm = int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))
    return dt.datetime(y, mo, d, hh, mm, tzinfo=dt.timezone.utc)


def _int(s: str) -> Optional[int]:
    s = s.strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        return None


@dataclass
class NBS:
    station: str
    issued: dt.datetime
    valid: list[dt.datetime]
    rows: dict[str, list[Optional[int]]] = field(default_factory=dict)


def parse_nbs(block: list[str]) -> NBS:
    issued = _issue(block[0])
    fhr_line = next(l for l in block if l.startswith(" FHR"))
    ncol = (len(fhr_line.rstrip()) - 5 + 2) // 3
    fhr = [_int(fhr_line[5 + 3 * j:8 + 3 * j]) for j in range(ncol)]
    valid = [issued + dt.timedelta(hours=h) if h is not None else None for h in fhr]
    rows = {}
    for l in block[1:]:
        lab = l[:5].strip()
        if not lab or lab in ("DT", "UTC", "FHR"):
            continue
        rows[lab] = [_int(l[5 + 3 * j:8 + 3 * j]) for j in range(ncol)]
    return NBS(block[0].split()[0], issued, valid, rows)


@dataclass
class NBP:
    station: str
    issued: dt.datetime
    valid: list[dt.datetime]
    rows: dict[str, list[Optional[int]]] = field(default_factory=dict)


def parse_nbp(block: list[str]) -> NBP:
    issued = _issue(block[0])
    fhr_line = next(l for l in block if l.startswith(" FHR"))
    body = fhr_line[7:]
    ngrp = body.count("|") + 1

    def vals(line: str) -> list[Optional[int]]:
        b = line[7:]
        out = []
        for g in range(ngrp):
            grp = b[8 * g:8 * g + 8]
            out.append(_int(grp[0:3]))
            out.append(_int(grp[4:7]))
        return out

    fhr = vals(fhr_line)
    valid = [issued + dt.timedelta(hours=h) if h is not None else None for h in fhr]
    rows = {}
    for l in block[1:]:
        lab = l[:7].strip()
        if not lab or lab in ("UTC", "FHR") or "|" not in l:
            continue
        if l.strip().startswith(("SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT")):
            continue
        rows[lab] = vals(l)
    return NBP(block[0].split()[0], issued, valid, rows)


@dataclass
class TempForecast:
    station: str
    kind: str  # 'high' | 'low'
    local_date: dt.date
    mean: float
    sd: float
    pct: dict  # {10: x, 25: x, 50: x, 75: x, 90: x} (may be empty)
    source: str
    issued: dt.datetime


def _target_valid(kind: str, local_date: dt.date) -> dt.datetime:
    # max for local date D is listed at (D+1) 00Z; min for local date D at D 12Z
    if kind == "high":
        d = local_date + dt.timedelta(days=1)
        return dt.datetime(d.year, d.month, d.day, 0, tzinfo=dt.timezone.utc)
    return dt.datetime(local_date.year, local_date.month, local_date.day, 12, tzinfo=dt.timezone.utc)


def forecast_from_nbp(nbp: NBP, kind: str, local_date: dt.date) -> Optional[TempForecast]:
    tv = _target_valid(kind, local_date)
    if tv not in nbp.valid:
        return None
    j = nbp.valid.index(tv)
    get = lambda lab: (nbp.rows.get(lab) or [None] * len(nbp.valid))[j]
    mean, sd = get("TXNMN"), get("TXNSD")
    if mean is None:
        return None
    pct = {10: get("TXNP1"), 25: get("TXNP2"), 50: get("TXNP5"), 75: get("TXNP7"), 90: get("TXNP9")}
    pct = {k: v for k, v in pct.items() if v is not None}
    return TempForecast(nbp.station, kind, local_date, float(mean), float(sd or 2), pct, "NBP", nbp.issued)


def forecast_from_nbs(nbs: NBS, kind: str, local_date: dt.date) -> Optional[TempForecast]:
    tv = _target_valid(kind, local_date)
    if tv not in nbs.valid:
        return None
    j = nbs.valid.index(tv)
    txn = (nbs.rows.get("TXN") or [None] * len(nbs.valid))[j]
    xnd = (nbs.rows.get("XND") or [None] * len(nbs.valid))[j]
    if txn is None:
        return None
    return TempForecast(nbs.station, kind, local_date, float(txn), float(xnd or 2), {}, "NBS", nbs.issued)


class NBMCache:
    """Keeps the latest parsed NBS/NBP for a station set; refresh() is cheap when nothing new is published."""

    def __init__(self, stations: set[str]):
        self.stations = set(stations)
        self.nbs: dict[str, NBS] = {}
        self.nbp: dict[str, NBP] = {}
        self.nbs_cycle: Optional[dt.datetime] = None
        self.nbp_cycle: Optional[dt.datetime] = None
        self.last_refresh: Optional[dt.datetime] = None

    def refresh(self, now: Optional[dt.datetime] = None):
        now = now or dt.datetime.now(dt.timezone.utc)
        cyc, txt = latest("nbs", now, lookback_h=6)
        if cyc and cyc != self.nbs_cycle and txt:
            blocks = station_blocks(txt, self.stations)
            self.nbs = {s: parse_nbs(b) for s, b in blocks.items()}
            self.nbs_cycle = cyc
        cyc, txt = latest("nbp", now, lookback_h=8)
        if cyc and cyc != self.nbp_cycle and txt:
            blocks = station_blocks(txt, self.stations)
            self.nbp = {s: parse_nbp(b) for s, b in blocks.items()}
            self.nbp_cycle = cyc
        self.last_refresh = now
        prune_cache()

    def forecast(self, station: str, kind: str, local_date: dt.date) -> Optional[TempForecast]:
        fc = None
        if station in self.nbs:
            fc = forecast_from_nbs(self.nbs[station], kind, local_date)
        if station in self.nbp:
            fp = forecast_from_nbp(self.nbp[station], kind, local_date)
            # prefer the percentile product when it is not older than the NBS run by >7h
            if fp and (fc is None or (fc.issued - fp.issued).total_seconds() < 7 * 3600):
                if fc is not None:
                    fp.mean = fc.mean  # freshest center from the hourly NBS run
                fc = fp
        return fc
