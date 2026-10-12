import sys, glob, os, json, time
sys.path.insert(0, '.')
from wxbot.backtest import sim
from wxbot.strategy.sleeve_a import SleeveAConfig
from wxbot.risk.limits import RiskConfig
EV='/home/claude/research/ev'; CD='/home/claude/research/cd'
files=sorted(glob.glob(f'{EV}/KXHIGH*.json'))
def day(f): return int(os.path.basename(f).split('-')[1][5:7])
train=[f for f in files if day(f)<=15]; test=[f for f in files if day(f)>15]
cal,n=sim.fit_calibration(train, CD)
print(f'calibration fit on Sep 1-15: {n} samples; test Sep 16-30: {len(test)} events')
FIVE={'KXHIGHNY','KXHIGHCHI','KXHIGHMIA','KXHIGHAUS','KXHIGHLAX'}
for label, cal_map in (('calibrated (train-fit)', cal), ('no calibration', None)):
    for qa in (0, 300, 1500):
        t0=time.time()
        r=sim.run(test, CD, cal_map=cal_map, queue_ahead=qa, series_allow=FIVE)
        pc = 100*r.pnl/r.contracts if r.contracts else 0
        print(f"{label:24s} queue_ahead={qa:5d}: fills={r.fills:5d} contracts={r.contracts:8.0f} cost=${r.cost:8.2f} "
              f"P&L=${r.pnl:8.2f} ({pc:+.2f}c/ct) maxDD=${r.max_dd:6.2f} days={len(r.daily)}  [{time.time()-t0:.0f}s]")
        if qa==300 and cal_map is not None:
            json.dump({'daily':r.daily,'by_event':r.by_event},open('data/backtest_sep16_30.json','w'),indent=0)
