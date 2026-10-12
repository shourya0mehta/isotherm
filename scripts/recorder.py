"""Overnight market-data recorder for replay/paper trading.
Writes JSONL (gz-free for simplicity) per hour into /home/claude/rec/data/."""
import json, time, os, sys, datetime, urllib.request, urllib.parse, traceback, threading
B="https://api.elections.kalshi.com/trade-api/v2"
OUT='/home/claude/rec/data'; os.makedirs(OUT,exist_ok=True)
_lock=threading.Lock(); _last=[0.0]; GAP=0.3
def get(path, params=None, tries=6):
    url=path if path.startswith('http') else B+path
    if params:
        q=urllib.parse.urlencode(params, doseq=True)
        url+=('&' if '?' in url else '?')+q
    for i in range(tries):
        with _lock:
            g=time.time()-_last[0]
            if g<GAP: time.sleep(GAP-g)
            _last[0]=time.time()
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'wxbot-recorder/0.1'})
            with urllib.request.urlopen(req,timeout=30) as r: return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code==404: return {'_err':404}
            time.sleep(1.5*(i+1))
        except Exception:
            time.sleep(1.5*(i+1))
    return {'_err':'fail'}
def fn(kind):
    h=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H')
    return f"{OUT}/{kind}_{h}.jsonl"
def write(kind, obj):
    with open(fn(kind),'a') as f: f.write(json.dumps(obj)+'\n')
SERIES=['KXHIGHNY','KXHIGHCHI','KXHIGHMIA','KXHIGHAUS','KXHIGHDEN','KXHIGHLAX','KXHIGHPHIL',
        'KXLOWTNYC','KXLOWTCHI','KXLOWTMIA','KXLOWTAUS','KXLOWTDEN','KXLOWTLAX','KXLOWTPHIL',
        'KXTEMPMIAH','KXRAIN']
PREFIX=tuple(['KXHIGH','KXLOWT','KXTEMP','KXRAIN'])
universe={}  # ticker -> market dict (slim)
def refresh_universe():
    new={}
    for s in SERIES:
        r=get('/events',{'series_ticker':s,'status':'open','limit':10,'with_nested_markets':'true'})
        for e in r.get('events',[]) or []:
            for m in e.get('markets',[]) or []:
                if m.get('status') not in ('active','open'): continue
                new[m['ticker']]={k:m.get(k) for k in ('ticker','event_ticker','strike_type','floor_strike','cap_strike','open_time','close_time','yes_sub_title','volume_fp','open_interest_fp','yes_bid_dollars','yes_ask_dollars','last_price_dollars')}
    if new:
        universe.clear(); universe.update(new)
    write('universe',{'ts':time.time(),'markets':list(universe.values())})
def snap_books():
    tks=list(universe.keys()); ts=time.time()
    for i in range(0,len(tks),100):
        chunk=tks[i:i+100]
        r=get('/markets/orderbooks',[('tickers',t) for t in chunk])
        for o in r.get('orderbooks',[]) or []:
            ob=o.get('orderbook_fp') or {}
            write('books',{'ts':ts,'t':o['ticker'],'y':ob.get('yes_dollars') or [],'n':ob.get('no_dollars') or []})
last_trade_ts=[int(time.time())-120]
def pull_trades():
    t0=last_trade_ts[0]; t1=int(time.time())-2
    cur=None; n=0; pages=0
    while True:
        p={'min_ts':t0,'max_ts':t1,'limit':1000}
        if cur: p['cursor']=cur
        r=get('/markets/trades',p); tr=r.get('trades',[]) or []; pages+=1
        for t in tr:
            if t['ticker'].startswith(PREFIX):
                write('trades',t); n+=1
        cur=r.get('cursor')
        if not cur or not tr or pages>200: break
    last_trade_ts[0]=t1
    return n
def pull_index():
    r=get('/live_data/weather/miami')
    ts=r.get('timeseries') or []
    if ts: write('mia_index',{'ts':time.time(),'config':r.get('config_version'),'points':ts[-15:]})
STATIONS='KNYC,KMDW,KMIA,KAUS,KDEN,KLAX,KPHL,KOPF,KFLL,KFXE,KPMP,KORD,KLGA,KJFK'
def pull_metar():
    try:
        url=f"https://aviationweather.gov/api/data/metar?ids={STATIONS}&format=json&hours=2"
        req=urllib.request.Request(url,headers={'User-Agent':'wxbot-recorder/0.1 (research)'})
        with urllib.request.urlopen(req,timeout=30) as r: d=json.load(r)
        write('metar',{'ts':time.time(),'obs':d})
    except Exception as e:
        write('errors',{'ts':time.time(),'where':'metar','err':str(e)})
def main():
    nxt={'u':0,'b':0,'t':0,'i':0,'m':0}
    while True:
        now=time.time()
        try:
            if now>=nxt['u']: refresh_universe(); nxt['u']=now+300
            if now>=nxt['b']: snap_books(); nxt['b']=now+20
            if now>=nxt['t']: pull_trades(); nxt['t']=now+30
            if now>=nxt['i']: pull_index(); nxt['i']=now+60
            if now>=nxt['m']: pull_metar(); nxt['m']=now+300
        except Exception as e:
            write('errors',{'ts':now,'err':traceback.format_exc()[-800:]})
        with open(f'{OUT}/heartbeat','w') as f: f.write(str(time.time()))
        time.sleep(1)
if __name__=='__main__': main()
