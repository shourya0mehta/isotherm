"""NWS CLI daily climate reports (the settlement values) for the 24 Kalshi stations, 2015-2026, from IEM."""
import json, os, time, requests
UA = {"User-Agent": "wxbot-research/0.1 (student research)"}
ST = ['KATL','KAUS','KBOS','KDCA','KDEN','KDFW','KEWR','KHOU','KLAS','KLAX','KMDW','KMIA','KMSP','KMSY','KNYC',
      'KOKC','KPHL','KPHX','KSAN','KSAT','KSDF','KSEA','KSFO','KTTN']
for s in ST:
    for y in range(2015, 2027):
        fn = f"raw/cli/{s}_{y}.json"
        if os.path.exists(fn):
            continue
        for k in range(4):
            try:
                r = requests.get("https://mesonet.agron.iastate.edu/json/cli.py", params={"station": s, "year": y},
                                 headers=UA, timeout=90)
                if r.status_code == 200:
                    json.dump(r.json().get("results", []), open(fn, "w"))
                    break
            except Exception as e:
                time.sleep(3 * (k + 1))
        time.sleep(0.3)
    print(s, flush=True)
print("done")
