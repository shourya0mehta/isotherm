"""Round 5's hourly NBM reader (wxbot/ml/nbm_hourly.py): parser, run choice by posting time, columns."""
import json
import os

import pandas as pd

from wxbot.ml import nbm_hourly as NH

BULLETIN = """ KNYC    NBM V5.0 NBS GUIDANCE    8/15/2026  0100 UTC
 DT /AUG  15            /AUG  16                /AUG  17                /
 UTC  06 09 12 15 18 21 00 03 06 09 12 15 18 21 00 03 06 09 12 15 18 21 00
 FHR  05 08 11 14 17 20 23 26 29 32 35 38 41 44 47 50 53 56 59 62 65 68 71
 TXN                    83          69          80          70          84
 XND                     2           2           3           2           4
 TMP  73 70 71 78 82 83 77 75 72 71 73 79 79 77 74 74 73 73 75 79 83 84 80
"""


def test_parser_reads_each_days_max_and_spread():
    rows = NH.parse(BULLETIN, pd.Timestamp("2026-08-15 01:00"), ["KNYC"])
    assert rows == [("KNYC", pd.Timestamp("2026-08-15"), 83.0, 2.0), ("KNYC", pd.Timestamp("2026-08-16"), 80.0, 3.0),
                    ("KNYC", pd.Timestamp("2026-08-17"), 84.0, 4.0)]
    assert NH.parse(BULLETIN, pd.Timestamp("2026-08-15 01:00"), ["KMIA"]) == []


class Resp:
    def __init__(self, code, text="", lm=None):
        self.status_code, self.text = code, text
        self.headers = {"Last-Modified": lm} if lm else {}


def test_latest_takes_the_newest_archive_run_posted_in_time_and_caches_it(tmp_path, monkeypatch):
    posted = {"2026-08-15T01": "Sat, 15 Aug 2026 02:05:00 GMT"}
    calls = []

    def get(url, head=False, timeout=120.0):
        calls.append((url, head))
        key = url.split("blend.")[1][:8] + "T" + url.split("/")[-3]
        key = f"{key[:4]}-{key[4:6]}-{key[6:8]}T{key[9:11]}"
        if key not in posted:
            return Resp(404)
        return Resp(200, BULLETIN, posted[key])
    monkeypatch.setattr(NH, "_get", get)
    need = {("KNYC", pd.Timestamp("2026-08-15"))}
    f = NH.latest(pd.Timestamp("2026-08-15 03:00"), ["KNYC"], str(tmp_path), need=need)
    assert f["run"] == pd.Timestamp("2026-08-15 01:00")
    assert f["rows"][("KNYC", pd.Timestamp("2026-08-15"))] == (83.0, 2.0, pd.Timestamp("2026-08-15 01:00"))
    assert all(pd.Timestamp(u.split("blend.")[1][:8]) is not None for u, _ in calls)
    n = len(calls)
    assert NH.latest(pd.Timestamp("2026-08-15 03:00"), ["KNYC"], str(tmp_path), need=need)["run"] == f["run"]
    assert len(calls) == n
    # posted too late for an earlier decision: not used (02:05 + 5 min > 02:08)
    assert NH.latest(pd.Timestamp("2026-08-15 02:08"), ["KNYC"], str(tmp_path / "x")) is None
    df = pd.DataFrame({"icao": ["KNYC", "KNYC"], "D": pd.to_datetime(["2026-08-15", "2026-08-18"]), "h": [3, 3]})
    c = NH.columns(df, f)
    assert c.loc[0, "nbf_txn"] == 83.0 and c.loc[0, "nbf_age_h"] == 2.0 and pd.isna(c.loc[1, "nbf_txn"])


BULLETIN_13Z = """ KNYC    NBM V5.0 NBS GUIDANCE    8/15/2026  1300 UTC
 DT /AUG  15            /AUG  16                /AUG  17                /
 UTC  15 18 21 00 03 06 09 12 15 18 21 00 03 06 09 12 15 18 21 00
 FHR  02 05 08 11 14 17 20 23 26 29 32 35 38 41 44 47 50 53 56 59
 TXN                       70          81          71          85
 XND                        2           3           2           4
"""


def test_later_in_the_day_an_older_run_supplies_the_days_max(tmp_path, monkeypatch):
    """A 13Z run has no max for the day already under way (as in the archive): at 15Z the day's value comes from
    the 07Z or 01Z run, the next day's from the 13Z run, as research/ml/r5_nbm_fresh_feats.py chose them."""
    texts = {"01": BULLETIN, "13": BULLETIN_13Z}
    posted = {"01": "Sat, 15 Aug 2026 02:05:00 GMT", "13": "Sat, 15 Aug 2026 14:05:00 GMT"}

    def get(url, head=False, timeout=120.0):
        hh = url.split("/")[-3]
        if hh not in texts:
            return Resp(404)
        return Resp(200, texts[hh], posted[hh])
    monkeypatch.setattr(NH, "_get", get)
    need = {("KNYC", pd.Timestamp("2026-08-15")), ("KNYC", pd.Timestamp("2026-08-16"))}
    f = NH.latest(pd.Timestamp("2026-08-15 15:00"), ["KNYC"], str(tmp_path), need=need)
    assert f["run"] == pd.Timestamp("2026-08-15 13:00")
    assert f["rows"][("KNYC", pd.Timestamp("2026-08-15"))] == (83.0, 2.0, pd.Timestamp("2026-08-15 01:00"))
    assert f["rows"][("KNYC", pd.Timestamp("2026-08-16"))] == (81.0, 3.0, pd.Timestamp("2026-08-15 13:00"))
    df = pd.DataFrame({"icao": ["KNYC", "KNYC"], "D": pd.to_datetime(["2026-08-15", "2026-08-16"]), "h": [15, -9]})
    c = NH.columns(df, f)
    assert c.loc[0, "nbf_age_h"] == 14.0 and c.loc[0, "nbf_txn"] == 83.0
    assert c.loc[1, "nbf_txn"] == 81.0 and c.loc[1, "nbf_age_h"] == 2.0
