"""Per-city v1 P&L in each window (per-event replay, queue 300). Is any city negative everywhere?"""
import sys, glob, os, json, collections
sys.path.insert(0, '.')
from wxbot.backtest import sim
from wxbot.strategy.sleeve_a import SleeveAConfig
from wxbot.risk.limits import RiskConfig
EV='/home/claude/research/ev'; CD='/home/claude/research/cd'
sep=sorted(glob.glob(f'{EV}/KXHIGH*-26SEP*.json'))
aug=[f for f in sorted(glob.glob(f'{EV}/KXHIGH*-26AUG*.json')) if os.path.getsize(f) > 10]
def day(f): return int(os.path.basename(f).split('-')[1][5:7])
H1=[f for f in sep if day(f)<=15]; H2=[f for f in sep if day(f)>15]
cal1,_=sim.fit_calibration(H1, CD); cal2,_=sim.fit_calibration(H2, CD); calS,_=sim.fit_calibration(sep, CD)
cfg=SleeveAConfig(decided_only=True, decided_prob=0.97, min_edge_c=2.0, decided_max_cost_usd=40, improve_if_edge_c=99)
risk=RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=40, max_total_open_risk_usd=10**9)
res=collections.defaultdict(dict)
for name, files, cal in (('Sep1-15', H1, cal2), ('Sep16-30', H2, cal1), ('Aug', aug, calS)):
    r=sim.run(files, CD, cal_map=cal, cfg_a=cfg, risk=risk, queue_ahead=300.0)
    by=collections.defaultdict(lambda: [0.0, 0.0])
    for row in r.fill_log:
        ts, ev, t, side, pc, n, rr, h = row
        win = (rr == 'yes') == (side == 'bid')
        by[ev.split('-')[0]][0] += n
        by[ev.split('-')[0]][1] += n * ((100 if win else 0) - (pc if side == 'bid' else 100 - pc)) / 100
    for s,(n,p) in by.items():
        res[s][name]=(n,p)
print(f"{'series':14s} " + " ".join(f"{w:>22s}" for w in ('Sep1-15','Sep16-30','Aug')) + "   total")
neg_all=[]
for s in sorted(res):
    cells=[]; tot=0.0; negs=0; seen=0
    for w in ('Sep1-15','Sep16-30','Aug'):
        if w in res[s]:
            n,p=res[s][w]; cells.append(f"${p:7.2f} ({100*p/max(n,1):+5.2f}c)"); tot+=p; seen+=1; negs+= p<0
        else:
            cells.append(f"{'-':>22s}")
    print(f"{s:14s} " + " ".join(f"{c:>22s}" for c in cells) + f"   ${tot:7.2f}")
    if seen>=2 and negs==seen: neg_all.append(s)
print("negative in every window:", neg_all or "none")
