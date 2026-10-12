"""Station-level model guidance archive from IEM (NBM NBS/NBE, GFS MOS, NAM MOS, GFS-extended MEX, LAMP), 2020-2026.
Tiny CSVs, one request per station-model-quarter. Free, no key."""
import os, sys, time, requests, datetime as dt
UA = {"User-Agent": "wxbot-research/0.1 (student research)"}
ST = ['KATL','KAUS','KBOS','KDCA','KDEN','KDFW','KEWR','KHOU','KLAS','KLAX','KMDW','KMIA','KMSP','KMSY','KNYC',
      'KOKC','KPHL','KPHX','KSAN','KSAT','KSDF','KSEA','KSFO','KTTN']
MODELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["NBS", "GFS", "NAM", "MEX", "NBE", "LAV"]
start, end = dt.date(2020, 6, 1), dt.date(2026, 10, 3)
def chunks():
    d = start
    while d < end:
        e = min(d + dt.timedelta(days=92), end)
        yield d, e
        d = e
for model in MODELS:
    for s in ST:
        for a, b in chunks():
            fn = f"raw/mos/{model}_{s}_{a:%Y%m%d}.csv"
            if os.path.exists(fn) and os.path.getsize(fn) > 100:
                continue
            p = {"station": s, "model": model, "sts": f"{a:%Y-%m-%d}T00:00Z", "ets": f"{b:%Y-%m-%d}T00:00Z", "format": "csv"}
            for k in range(5):
                try:
                    r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py", params=p, headers=UA, timeout=300)
                    if r.status_code == 200:
                        open(fn, "w").write(r.text)
                        break
                    time.sleep(5 * (k + 1))
                except Exception as e:
                    time.sleep(5 * (k + 1))
            time.sleep(0.2)
        print(model, s, flush=True)
print("done", flush=True)
