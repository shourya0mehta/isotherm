"""Where do v1 decided-outcome fills lose? P&L per contract by minute-of-hour, UTC hour, hours-to-close, price."""
import sys, glob, os, collections, math, json
sys.path.insert(0, '.')
from wxbot.backtest import sim
from wxbot.strategy.sleeve_a import SleeveAConfig
from wxbot.risk.limits import RiskConfig
EV='/home/claude/research/ev'; CD='/home/claude/research/cd'
sep=sorted(glob.glob(f'{EV}/KXHIGH*-26SEP*.json'))
def day(f): return int(os.path.basename(f).split('-')[1][5:7])
H1=[f for f in sep if day(f)<=15]; H2=[f for f in sep if day(f)>15]
cal1,_=sim.fit_calibration(H1, CD); cal2,_=sim.fit_calibration(H2, CD)
cfg=SleeveAConfig(decided_only=True, decided_prob=0.97, min_edge_c=2.0, decided_max_cost_usd=40, improve_if_edge_c=99)
risk=RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=40, max_total_open_risk_usd=10**9)
rows=[]
for test, cal in ((H1, cal2), (H2, cal1)):
    r=sim.run(test, CD, cal_map=cal, cfg_a=cfg, risk=risk, queue_ahead=300.0)
    rows += r.fill_log
json.dump(rows, open('data/fill_log_sep_v1.json','w'))
def pnl_c(row):
    ts, ev, t, side, pc, n, res, h = row
    win = (res == 'yes') == (side == 'bid')
    cost = pc if side == 'bid' else 100 - pc
    return (100 if win else 0) - cost
def table(name, key):
    agg = collections.defaultdict(lambda: [0.0, 0.0, collections.defaultdict(float)])
    for row in rows:
        k = key(row); n = row[5]; p = pnl_c(row)
        agg[k][0] += n; agg[k][1] += n * p; agg[k][2][row[1]] += n * p
    print(f"\n{name}")
    for k in sorted(agg):
        n, s, by_ev = agg[k]
        # event-clustered SE of the per-contract mean
        mean = s / n
        var = sum((v - mean * 0) ** 2 for v in by_ev.values())  # crude: SE of total / n
        se = math.sqrt(var) / n if n else 0
        print(f"  {str(k):>10s}: {n:8.0f} ct  {mean:+6.2f} c/ct  (se ~{se:.2f})  events {len(by_ev)}")
import datetime as dt
table("minute of hour (15-min buckets)", lambda r: dt.datetime.utcfromtimestamp(r[0]).minute // 15 * 15)
table("minute 45-59 split", lambda r: 'm50-56' if 50 <= dt.datetime.utcfromtimestamp(r[0]).minute <= 56 else 'other')
table("UTC hour (3h buckets)", lambda r: dt.datetime.utcfromtimestamp(r[0]).hour // 3 * 3)
table("hours to close", lambda r: '<6' if r[7] < 6 else '6-12' if r[7] < 12 else '12-24' if r[7] < 24 else '>24')
table("price paid", lambda r: (r[4] if r[3] == 'bid' else 100 - r[4]))
table("side", lambda r: 'YES fav' if r[3] == 'bid' else 'NO longshot')
