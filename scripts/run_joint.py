import sys, glob, os, json, time
sys.path.insert(0, '.')
from wxbot.backtest import sim
from wxbot.backtest.joint import run_joint
from wxbot.strategy.sleeve_a import SleeveAConfig
from wxbot.risk.limits import RiskConfig
EV='/home/claude/research/ev'; CD='/home/claude/research/cd'
sep=sorted(glob.glob(f'{EV}/KXHIGH*-26SEP*.json'))
def day(f): return int(os.path.basename(f).split('-')[1][5:7])
H1=[f for f in sep if day(f)<=15]; H2=[f for f in sep if day(f)>15]
cal1,_=sim.fit_calibration(H1, CD); cal2,_=sim.fit_calibration(H2, CD)
cfg=SleeveAConfig(decided_only=True, decided_prob=0.97, min_edge_c=2.0, decided_max_cost_usd=40, improve_if_edge_c=99)
for label, risk in (('full caps ($350 open / $40 event)', RiskConfig(max_loss_per_bracket_usd=30, max_loss_per_event_usd=40, max_total_open_risk_usd=350)),
                    ('half caps ($150 open / $20 event)', RiskConfig(max_loss_per_bracket_usd=15, max_loss_per_event_usd=20, max_total_open_risk_usd=150))):
    for split,(test,cal) in (('Sep1-15',(H1,cal2)),('Sep16-30',(H2,cal1))):
        t0=time.time()
        r=run_joint(test, CD, cal_map=cal, cfg_a=cfg, risk=risk)
        print(f"{label:34s} {split:8s}: realized ${r['realized']:7.2f} over {r['days']} days (${r['realized']/max(r['days'],1):.2f}/day), "
              f"up {r['up_days']} / down {r['down_days']} days, maxDD ${r['max_dd']:.1f}, peak locked ${r['max_locked']:.0f}, "
              f"contracts {r['contracts']:.0f}  [{time.time()-t0:.0f}s]", flush=True)
        json.dump(r, open(f"data/joint_{split}_{label.split()[0]}.json","w"))
