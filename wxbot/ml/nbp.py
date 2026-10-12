"""NBM probabilistic station text (NBP): the daily max's mean, spread and 10/25/50/75/90th percentiles.

Research plan step 5 ("Beating the market on daily highs", Oct 8, 2026): Tier-1 reads NBM's deterministic max and its
spread (xnd); NBP adds the shape of NBM's distribution (how far the 10th and 90th percentiles sit from the median),
which matters for the center brackets on nights the guidance disagrees.

NOAA's bucket posts blend_nbptx.tHHz for every hourly run, about 80 minutes after the run. Each file (about 35 MB)
holds fixed-width station blocks; the rows used here are TXNMN, TXNSD and TXNP1/P2/P5/P7/P9 (10/25/50/75/90th
percentiles) in the 00Z columns, which hold the max of the previous local day (as in the NBS text, nbm_hourly.parse).

Reading a whole file for 24 stations is wasteful, so `Reader` learns each station's byte offset from one full
download and then reads a small window around it in later files, checking every block's header (station, run date and
hour); any miss falls back to the full file. NBP is published for the 00, 01, 07, 12, 13 and 19Z runs only.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

BUCKET = "https://noaa-nbm-grib2-pds.s3.amazonaws.com"
ROWS = {"TXNMN": "mn", "TXNSD": "sd", "TXNP1": "p10", "TXNP2": "p25", "TXNP5": "p50", "TXNP7": "p75", "TXNP9": "p90"}
COLS = ["mn", "sd", "p10", "p25", "p50", "p75", "p90"]
BLOCK = 1400          # bytes from a station's header line through its TXN rows (about 11 lines of 81 characters)
_HDR = re.compile(r"^ ?(\w{4}) +NBM V[\d.]+ NBP GUIDANCE +(\d{1,2})/(\d{1,2})/(\d{4}) +(\d{2})00 UTC")


def url_for(run: pd.Timestamp) -> str:
    return f"{BUCKET}/blend.{run:%Y%m%d}/{run:%H}/text/blend_nbptx.t{run:%H}z"


def parse_block(text: str, run: pd.Timestamp) -> Optional[Tuple[str, Optional[List[tuple]]]]:
    """(icao, [(D, mn, sd, p10, p25, p50, p75, p90), ...]) from one station block starting at its header line; None
    if the header does not match `run`; (icao, None) if the text stops before the last TXN row is complete."""
    lines = text.lstrip("\n").splitlines()
    if not lines:
        return None
    m = _HDR.match(lines[0])
    if not m:
        return None
    icao, mo, dd, yy, hh = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
    if pd.Timestamp(year=yy, month=mo, day=dd, hour=hh) != pd.Timestamp(run):
        return None
    rows = {}
    for ln in lines[1:14]:
        key = ln[:7].strip()
        if key in ("UTC", "FHR") or key in ROWS:
            rows[key] = ln
    fhr, utc = rows.get("FHR"), rows.get("UTC")
    last = text.find("\n TXNP9")
    complete = all(k in rows for k in ROWS) and last >= 0 and text.find("\n", last + 1) > 0
    if fhr is None or utc is None or not complete:
        return icao, None
    out = []
    for mt in re.finditer(r"\d+", fhr[6:]):
        end = 6 + mt.end()
        if utc[max(0, end - 3):end].strip() != "00":
            continue
        vals = []
        for k in ROWS:
            ln = rows.get(k, "")
            s = ln[end - 3:end].strip() if len(ln) >= end else ""
            vals.append(float(s) if s.lstrip("-").isdigit() else float("nan"))
        if all(v != v for v in vals):
            continue
        valid = pd.Timestamp(run) + pd.Timedelta(hours=int(mt.group()))
        out.append(((valid - pd.Timedelta(days=1)).normalize(), *vals))
    return icao, out


def parse_file(data: bytes, run: pd.Timestamp, stations) -> Tuple[List[tuple], Dict[str, int]]:
    """All rows for `stations` from a whole file, plus each station's byte offset (start of its header line)."""
    rows, offsets = [], {}
    for icao in stations:
        i = data.find(b"\n " + icao.encode() + b" ")
        while i >= 0 and b"NBP GUIDANCE" not in data[i:i + 90]:
            i = data.find(b"\n " + icao.encode() + b" ", i + 1)
        if i < 0:
            continue
        got = parse_block(data[i:i + BLOCK].decode("ascii", "replace"), run)
        if got is None or got[1] is None:
            continue
        offsets[icao] = i
        rows += [(icao, *r) for r in got[1]]
    return rows, offsets


class Reader:
    """Range reads of NBP station blocks (thread-safe). Station offsets are learned from one full download and then
    tracked: each station is read from a window of +-WINDOW bytes around its last known offset (files of the same run
    type differ by a few bytes to a few KB), its header checked for the station, run date and hour, and the offset
    updated. Any station not found in its window sends that run to a full download, which re-learns every offset."""

    WINDOW = 24 * 1024

    def __init__(self, stations, timeout: float = 60.0):
        self.stations = list(stations)
        self.timeout = timeout
        self._offs: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._local = threading.local()
        self.stats = {"range": 0, "full": 0, "missing": 0}

    def _session(self):
        s = getattr(self._local, "s", None)
        if s is None:
            import requests
            s = requests.Session()
            self._local.s = s
        return s

    def _get(self, url: str, rng: Optional[Tuple[int, int]] = None):
        import requests
        hdr = {"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}
        for _ in range(4):
            try:
                r = self._session().get(url, headers=hdr, timeout=self.timeout)
                if r.status_code in (200, 206, 404, 416):
                    return r
            except requests.RequestException:
                self._local.s = None
                time.sleep(2)
        return None

    @staticmethod
    def _posted(r) -> Optional[pd.Timestamp]:
        lm = r.headers.get("Last-Modified")
        return pd.Timestamp(lm).tz_convert("UTC").tz_localize(None) if lm else None

    def _full(self, url: str, run: pd.Timestamp):
        r = self._get(url)
        if r is None or r.status_code != 200:
            with self._lock:
                self.stats["missing"] += 1
            return None
        rows, offs = parse_file(r.content, run, self.stations)
        with self._lock:
            self._offs.update(offs)
            self.stats["full"] += 1
        return rows, self._posted(r)

    def read(self, run: pd.Timestamp) -> Optional[Tuple[List[tuple], Optional[pd.Timestamp]]]:
        """([(icao, D, mn, sd, p10, p25, p50, p75, p90), ...], posted) for one run, or None if the file is missing."""
        url = url_for(run)
        with self._lock:
            offs = dict(self._offs)
        if len(offs) < len(self.stations):
            return self._full(url, run)
        rows, posted, learned = [], None, {}
        for icao in self.stations:
            a = max(0, offs[icao] - self.WINDOW)
            r = self._get(url, (a, offs[icao] + BLOCK + self.WINDOW))
            if r is None:
                return self._full(url, run)
            if r.status_code == 404:
                with self._lock:
                    self.stats["missing"] += 1
                return None
            if r.status_code not in (200, 206):
                return self._full(url, run)
            posted = posted or self._posted(r)
            body = r.content
            key = b"\n " + icao.encode() + b" "
            i = body.find(key)
            while i >= 0 and b"NBP GUIDANCE" not in body[i:i + 90]:
                i = body.find(key, i + 1)
            if i < 0:
                return self._full(url, run)
            got = parse_block(body[i:i + BLOCK].decode("ascii", "replace"), run)
            if got is None or got[0] != icao or got[1] is None:
                return self._full(url, run)
            rows += [(icao, *x) for x in got[1]]
            learned[icao] = a + i
        with self._lock:
            self._offs.update(learned)
            self.stats["range"] += 1
        return rows, posted


RUN_HOURS = (1, 7, 13, 19)        # the runs with NBP text that the training archive holds
MARGIN = pd.Timedelta(minutes=5)
_READERS: Dict[tuple, "Reader"] = {}


def latest(dec: pd.Timestamp, stations, cache_dir: str, need=None, max_back_h: int = 24, hours=RUN_HOURS,
           reader: Optional["Reader"] = None) -> Optional[dict]:
    """For each (icao, D), the latest archived-hour run posted at least MARGIN before dec that carries D's max, as
    training chose it (research/ml/r5_nbp_feats.py merge_asof). Returns {"rows": {(icao, D): (mn, sd, p10, p25, p50,
    p75, p90, run)}} or None. Parsed runs are cached as small JSON files; `need` stops the walk back early."""
    import json
    import os
    os.makedirs(cache_dir, exist_ok=True)
    key = tuple(stations)
    rd = reader or _READERS.setdefault(key, Reader(list(stations)))
    out = {}
    for back in range(1, max_back_h + 1):
        run = (pd.Timestamp(dec) - pd.Timedelta(hours=back)).floor("h")
        if run.hour not in hours:
            continue
        cache = os.path.join(cache_dir, f"{run:%Y%m%d_%H}z.json")
        if os.path.exists(cache):
            d = json.load(open(cache))
            posted = pd.Timestamp(d["posted"])
            rows = [(r[0], pd.Timestamp(r[1]), *r[2:]) for r in d["rows"]]
        else:
            got = rd.read(run)
            if not got or got[1] is None:
                continue
            rows, posted = got
            tmp = cache + ".tmp"
            json.dump({"run": str(run), "posted": str(posted),
                       "rows": [(r[0], str(pd.Timestamp(r[1]).date()), *[None if v != v else v for v in r[2:]])
                                for r in rows]}, open(tmp, "w"))
            os.replace(tmp, cache)
        if posted + MARGIN > pd.Timestamp(dec):
            continue
        for r in rows:
            k = (r[0], pd.Timestamp(r[1]))
            if k not in out:
                out[k] = (*[float("nan") if v is None else v for v in r[2:]], run)
        if need is not None and all(k in out for k in need):
            break
    return {"rows": out} if out else None


def frame_columns(df: pd.DataFrame, got: Optional[dict]) -> pd.DataFrame:
    """Raw nbp_mn ... nbp_p90 and nbp_age_h for rows (icao, D, h), as research/ml/r5_nbp_feats.py builds them."""
    cols = ["nbp_" + c for c in COLS] + ["nbp_age_h"]
    out = pd.DataFrame(index=df.index, columns=cols, dtype=float)
    if not got:
        return out
    for i, r in zip(df.index, df.itertuples()):
        v = got["rows"].get((r.icao, pd.Timestamp(r.D)))
        if v is None:
            continue
        dec = pd.Timestamp(r.D) + pd.Timedelta(hours=int(r.h))
        out.loc[i, cols[:-1]] = list(v[:7])
        out.loc[i, "nbp_age_h"] = (dec - v[7]) / pd.Timedelta(hours=1)
    return out


def columns(df: pd.DataFrame, latest: Optional[dict]) -> pd.DataFrame:
    """nbp_* columns for rows (icao, D) from one run's parsed rows ({(icao, D): (mn, sd, p10, ..., p90)})."""
    out = pd.DataFrame(index=df.index, columns=["nbp_" + c for c in COLS], dtype=float)
    if not latest:
        return out
    for i, r in zip(df.index, df.itertuples()):
        v = latest["rows"].get((r.icao, pd.Timestamp(r.D)))
        if v is not None:
            out.loc[i, :] = list(v)
    return out
