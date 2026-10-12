"""Hourly + special METAR observations for the 24 stations from IEM, one request per year (all stations)."""
import os, time, requests
UA = {"User-Agent": "wxbot-research/0.1 (student research)"}
IDS = "NYC,MDW,MIA,AUS,DEN,LAX,PHL,ATL,BOS,DFW,DCA,EWR,HOU,LAS,MSP,MSY,OKC,PHX,SAN,SAT,SDF,SEA,SFO,TTN"
for y in range(2020, 2027):
    fn = f"raw/obs/asos_{y}.csv"
    if os.path.exists(fn) and os.path.getsize(fn) > 1e6:
        continue
    ets = f"{y + 1}-01-01T00:00Z" if y < 2026 else "2026-10-03T00:00Z"
    url = (f"https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station={IDS}"
           f"&data=tmpf,dwpf,drct,sknt,skyc1,skyl1,mslp,p01i,metar&sts={y}-01-01T00:00Z&ets={ets}"
           f"&tz=Etc/UTC&format=onlycomma&missing=empty&trace=T&report_type=3,4")
    for k in range(8):
        try:
            r = requests.get(url, headers=UA, timeout=600)
            if r.status_code == 200 and len(r.content) > 1e5:
                open(fn, "wb").write(r.content)
                print(y, len(r.content), flush=True)
                break
            print(y, "status", r.status_code, flush=True)
        except Exception as e:
            print(y, "err", e, flush=True)
        time.sleep(15 * (k + 1))
    time.sleep(15)
print("done", flush=True)
