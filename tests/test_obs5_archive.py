"""The forward observation archive (scripts/obs5_archive.py): first-seen times kept, duplicates ignored."""
import importlib.util
import os
import sqlite3

spec = importlib.util.spec_from_file_location("obs5", os.path.join(os.path.dirname(__file__), "..", "scripts",
                                                                    "obs5_archive.py"))
O = importlib.util.module_from_spec(spec)
spec.loader.exec_module(O)


class Resp:
    def __init__(self, code, doc):
        self.status_code, self.doc = code, doc

    def json(self):
        return self.doc


def feat(ts, t):
    return {"properties": {"timestamp": ts, "temperature": {"value": t}, "dewpoint": {"value": None},
                           "maxTemperatureLast24Hours": {"value": None}, "rawMessage": ""}}


def test_archive_keeps_the_first_sighting_and_survives_errors(tmp_path):
    db = str(tmp_path / "o.sqlite")
    docs = {"KMDW": {"features": [feat("2026-10-08T02:30:00+00:00", 19), feat("2026-10-08T02:25:00+00:00", 19)]}}

    def get(url, params=None, headers=None, timeout=None):
        st = url.split("/")[-2]
        if st == "KLAX":
            raise O.requests.ConnectionError("down")
        return Resp(200 if st in docs else 503, docs.get(st, {}))
    assert O.run(db, get=get) == {"new": 2, "errors": len(O.STATIONS) - 1}
    first = sqlite3.connect(db).execute("SELECT first_seen FROM obs5 WHERE ts LIKE '%02:30%'").fetchone()[0]
    docs["KMDW"]["features"].insert(0, feat("2026-10-08T02:35:00+00:00", 20))
    assert O.run(db, get=get)["new"] == 1
    c = sqlite3.connect(db)
    assert c.execute("SELECT COUNT(*) FROM obs5").fetchone()[0] == 3
    assert c.execute("SELECT first_seen FROM obs5 WHERE ts LIKE '%02:30%'").fetchone()[0] == first
    assert c.execute("SELECT temp_c FROM obs5 WHERE ts LIKE '%02:35%'").fetchone()[0] == 20.0
