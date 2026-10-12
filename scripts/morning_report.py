"""Compile the overnight results into docs/MORNING_REPORT.md (paper bots, Sleeve B shadow, backtests, health)."""
import datetime as dt
import glob
import json
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from wxbot.kalshi.rest import KalshiREST  # noqa: E402

rest = KalshiREST(base_url="https://api.elections.kalshi.com/trade-api/v2", read_rate=20)
now = dt.datetime.now(dt.timezone.utc)
L = []
w = L.append


def _held(yq, nq):
    parts = ([f"YES x{yq:g}"] if yq else []) + ([f"NO x{nq:g}"] if nq else [])
    return " + ".join(parts) or "flat"


def paper_section(name, db, status_path):
    if not os.path.exists(db):
        return
    c = sqlite3.connect(db)
    st = json.load(open(status_path)) if os.path.exists(status_path) else {}
    fills = c.execute("SELECT ts,ticker,event,side,price_cents,count,sleeve,fair_yes FROM fills ORDER BY ts").fetchall()
    w(f"### {name}")
    first = c.execute("SELECT MIN(ts) FROM events_log WHERE msg LIKE 'start mode%'").fetchone()[0]
    since = dt.datetime.fromtimestamp(first, dt.timezone.utc).strftime("%H:%M UTC") if first else "?"
    w(f"- Paper since {since} (restarts keep state). Events tracked {st.get('events', '?')}, live quotes now "
      f"{st.get('quotes', '?')}, halted={st.get('halted')}. API errors are mostly 429s from this sandbox's shared IP.")
    pos = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    for ts, t, ev, side, pc, n, sl, fair in fills:
        if sl == "B":
            continue
        if side == "bid":
            pos[t][0] += n
            pos[t][1] += n * pc / 100
        else:
            pos[t][2] += n
            pos[t][3] += n * (100 - pc) / 100
    # resolve results directly from the exchange
    realized, unsettled_cost, n_set, n_open = 0.0, 0.0, 0, 0
    lines = []
    for t, (yq, yc, nq, nc) in sorted(pos.items()):
        try:
            m = rest.market(t)
        except Exception:
            m = {}
        res = m.get("result")
        if res in ("yes", "no"):
            pnl = (yq if res == "yes" else 0) + (nq if res == "no" else 0) - yc - nc
            realized += pnl
            n_set += 1
            lines.append(f"  - {t}: {_held(yq, nq)} cost ${yc + nc:.2f} -> settled {res}, P&L ${pnl:+.2f}")
        else:
            n_open += 1
            unsettled_cost += yc + nc
            lines.append(f"  - {t}: {_held(yq, nq)} cost ${yc + nc:.2f} -> open")
    tot = sum(r[5] for r in fills)
    w(f"- Fills: {len(fills)} ({tot:.0f} contracts) in {len(pos)} markets. Settled {n_set} (realized ${realized:+.2f}); "
      f"open {n_open} (cost ${unsettled_cost:.2f}).")
    if lines:
        w("- Positions:")
        L.extend(lines[:40])
    lip = c.execute("SELECT COUNT(*), SUM(est_usd), MIN(ts), MAX(ts) FROM lip_snaps").fetchone()
    if lip and lip[0]:
        hrs = (lip[3] - lip[2]) / 3600 if lip[3] and lip[2] else 0
        w(f"- Sleeve B shadow (NOT traded): est rewards ${lip[1]:.2f} over {hrs:.1f} h "
          f"(${(lip[1] / hrs) if hrs else 0:.1f}/h); shadow fills {st.get('sleeve_b_shadow_fills', 0)}, "
          f"shadow fill P&L ${st.get('sleeve_b_shadow_fill_pnl_usd', 0):+.2f}")
    w("")


w(f"# wxbot overnight report ({now:%Y-%m-%d %H:%M} UTC)")
w("")
w("Paper = simulated fills on the live public tape, no money. v1 quotes only near-certain outcomes, which mostly "
  "trade in the afternoon of the target day, so overnight fills are sparse by design. The replays below are the "
  "real evidence; the paper week on your machine checks that live fills look like them.")
w("")
w("## Paper trading on the live tape")
paper_section("v1 config (24 daily-high cities, decided-outcome maker)", "data/overnight_v1.sqlite", "data/overnight_v1_status.json")
paper_section("v0 config (5 cities, highs + lows, all favorites) for comparison", "data/overnight.sqlite", "data/overnight_status.json")
w("## Replays that decide the config (full grids in docs/results/)")
for fn, title in (("data/bt_joint_cash.txt", "Joint replay, Sept, 24 cities, cash constraint"),
                  ("data/bt_joint_aug24.txt", "August out-of-sample, 24 cities (calibration fit on Sept)"),
                  ("data/bt_joint_queue.txt", "Queue sensitivity: 1,500 contracts ahead instead of 300"),
                  ("data/bt_joint_aug.txt", "August out-of-sample, 7 cities"),
                  ("data/bt_joint_a2.txt", "Option A2 under the cash cap (why it stays off)")):
    if os.path.exists(fn):
        txt = [l for l in open(fn).read().strip().splitlines() if not l.startswith(("daily:", "  worst"))]
        if txt:
            w(f"**{title}**")
            w("```")
            L.extend(txt)
            w("```")
hb = "/home/claude/rec/data/heartbeat"
if os.path.exists(hb):
    age = now.timestamp() - float(open(hb).read())
    w(f"## Health\n- Recorder heartbeat age {age:.0f}s; recorded files: {len(glob.glob('/home/claude/rec/data/*.jsonl'))}")
os.makedirs("docs", exist_ok=True)
open("docs/MORNING_REPORT.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
