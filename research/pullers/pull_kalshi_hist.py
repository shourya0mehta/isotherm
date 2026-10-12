"""Hourly bid/ask candles for every settled Kalshi daily-high market (historical endpoint), newest first.
Phase 1: market lists for the newer KXHIGHT* series. Phase 2: candles. Resumable; polite rate with 429 backoff."""
import glob, json, os, sys, time, datetime as dt
import requests
B = os.environ.get("KHOST", "https://api.elections.kalshi.com/trade-api/v2")
PART = int(os.environ.get("KPART", "0")); NPART = int(os.environ.get("KNPART", "1"))
UA = {"User-Agent": "wxbot-research/0.1 (student research)"}
GAP = [float(os.environ.get("KGAP", "1.0"))]  # adaptive: grows on 429, shrinks slowly after successes
_last = [0.0]
_ok = [0]

def get(path, params, tries=10):
    for k in range(tries):
        w = GAP[0] - (time.time() - _last[0])
        if w > 0:
            time.sleep(w)
        _last[0] = time.time()
        try:
            r = requests.get(B + path, params=params, headers=UA, timeout=60)
        except Exception:
            time.sleep(3 * (k + 1)); continue
        if r.status_code == 200:
            _ok[0] += 1
            if _ok[0] % 100 == 0:
                GAP[0] = max(0.75, GAP[0] * 0.95)
            return r.json()
        if r.status_code == 429:
            GAP[0] = min(2.5, GAP[0] * 1.15)
            time.sleep(4 + 3 * k); continue
        if r.status_code == 404:
            return None
        time.sleep(3 * (k + 1))
    return None

def ts(s):
    return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())

SMALL = ["KXHIGHTATL", "KXHIGHTBOS", "KXHIGHTDAL", "KXHIGHTDC", "KXHIGHTEWR", "KXHIGHTHOU", "KXHIGHTLV", "KXHIGHTMIN",
         "KXHIGHTNOLA", "KXHIGHTOKC", "KXHIGHTPHX", "KXHIGHTSAN", "KXHIGHTSATX", "KXHIGHTSDF", "KXHIGHTSEA",
         "KXHIGHTSFO", "KXHIGHTTTN"]
os.makedirs("raw/kalshi_lists", exist_ok=True)
for s in SMALL:
    fn = f"raw/kalshi_lists/kalshi_hist_{s}.json"
    if os.path.exists(fn):
        continue
    out, cur = [], None
    for _ in range(60):
        p = {"series_ticker": s, "limit": 1000}
        if cur:
            p["cursor"] = cur
        d = get("/historical/markets", p)
        if not d:
            break
        out += d.get("markets", [])
        cur = d.get("cursor")
        if not cur or not d.get("markets"):
            break
    json.dump(out, open(fn, "w"))
    print("list", s, len(out), flush=True)

# order: every settled event since 2025-01-01 (all series) in a fixed random order, so whatever has landed is a
# uniform sample across cities and seasons; then older big-7 history newest first. Partition by event.
import zlib, random
BIG = ["KXHIGHNY", "KXHIGHCHI", "KXHIGHMIA", "KXHIGHAUS", "KXHIGHDEN", "KXHIGHPHIL", "KXHIGHLAX"]
mk = []
for fn in glob.glob("raw/kalshi_lists/kalshi_hist_*.json"):
    s = fn.split("kalshi_hist_")[1][:-5]
    for m in json.load(open(fn)):
        if m.get("result") not in ("yes", "no"):
            continue
        o, c = ts(m["open_time"]), ts(m["close_time"])
        ev = m["event_ticker"]
        if c >= ts("2025-01-02T00:00:00Z"):
            key = (0, random.Random(zlib.crc32(ev.encode())).random(), ev)
        else:
            key = (1, -o, ev)
        mk.append((key, s, m["ticker"], o, c))
mk.sort()
mk = [(k[0], k, s, t, o, c) for k, s, t, o, c in mk]
done = set()
for fn in glob.glob("raw/kcandles/*.jsonl"):
    for line in open(fn):
        try:
            done.add(json.loads(line)["t"])
        except Exception:
            pass
print("markets", len(mk), "already", len(done), flush=True)
n = 0
for tier, key, s, t, o, c in mk:
    if t in done or (zlib.crc32(key[2].encode()) % NPART) != PART:
        continue
    d = get(f"/historical/markets/{t}/candlesticks", {"start_ts": o, "end_ts": c, "period_interval": 60})
    if d is None:
        continue
    rows = []
    for x in d.get("candlesticks", []):
        yb = (x.get("yes_bid") or {}).get("close"); ya = (x.get("yes_ask") or {}).get("close")
        rows.append([x["end_period_ts"], float(yb) if yb is not None else None, float(ya) if ya is not None else None,
                     float(x.get("volume") or 0)])
    with open(f"raw/kcandles/{s}.p{PART}.jsonl", "a") as f:
        f.write(json.dumps({"t": t, "c": rows}) + "\n")
    n += 1
    if n % 100 == 0:
        print(time.strftime("%H:%M:%S"), "tier", tier, s, t, n, "gap", round(GAP[0], 2), flush=True)
print("done", n, flush=True)
