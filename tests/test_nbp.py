"""NBM probabilistic text (round 5 step 5): block parser, whole-file parse with offsets, windowed range reads that
follow a shifted layout, and the model columns."""
import numpy as np
import pandas as pd

from wxbot.ml import features as F
from wxbot.ml import nbp

BLOCK_KNYC = "\n".join([
    " KNYC    NBM V5.0 NBP GUIDANCE    9/06/2026  0100 UTC",
    "        MON 07| TUE 08| WED 09| THU 10| FRI 11| SAT 12| SUN 13| MON 14| TUE 15",
    " UTC    00  12| 00  12| 00  12| 00  12| 00  12| 00  12| 00  12| 00  12| 00  12",
    " FHR    23  35| 47  59| 71  83| 95 107|119 131|143 155|167 179|191 203|215 227",
    " TXNMN  73  61| 77  62| 80  66| 83  72| 83  65| 75  61| 76  64| 79  66| 79  66",
    " TXNSD   3   2|  1   2|  2   2|  4   3|  4   4|  5   5|  5   5|  5   5|  7   5",
    " TXNP1  69  59| 76  58| 78  63| 77  69| 78  60| 69  54| 70  57| 73  59| 71  59",
    " TXNP2  72  60| 77  60| 79  65| 80  70| 82  64| 72  57| 73  62| 76  64| 75  64",
    " TXNP5  73  61| 77  62| 80  66| 82  72| 83  66| 75  62| 76  65| 79  66| 78  66",
    " TXNP7  75  62| 78  63| 81  67| 85  74| 86  68| 79  65| 79  68| 81  69| 84  69",
    " TXNP9  76  63| 79  64| 83  68| 87  77| 88  69| 81  67| 83  71| 84  72| 87  72",
    " WSPP1   0   0|  0   0|  0   0|  4   5|  0   2|  0   0|  0   0|  0   0|  0   0",
]) + "\n"
RUN = pd.Timestamp("2026-09-06 01:00")


def _file(pad=0, other="KMIA"):
    """A fake bulletin: filler, KNYC's block, filler, a second station."""
    filler = (" XXXX    filler line" + " " * 60 + "\n") * (40 + pad)
    second = BLOCK_KNYC.replace("KNYC", other)
    return ("header\n" + filler + BLOCK_KNYC + "\n" + filler + second + "\n").encode()


def test_parse_block_reads_the_00z_columns_as_the_previous_local_day():
    icao, rows = nbp.parse_block("\n" + BLOCK_KNYC, RUN)
    assert icao == "KNYC" and len(rows) == 9
    D, mn, sd, p10, p25, p50, p75, p90 = rows[0]
    assert D == pd.Timestamp("2026-09-06") and (mn, sd, p10, p25, p50, p75, p90) == (73, 3, 69, 72, 73, 75, 76)
    assert rows[3][0] == pd.Timestamp("2026-09-09") and rows[3][1:] == (83, 4, 77, 80, 82, 85, 87)
    assert nbp.parse_block("\n" + BLOCK_KNYC, pd.Timestamp("2026-09-06 07:00")) is None   # another run's file


def test_parse_file_and_offsets():
    data = _file()
    rows, offs = nbp.parse_file(data, RUN, ["KNYC", "KMIA"])
    assert set(offs) == {"KNYC", "KMIA"} and len(rows) == 18
    assert data[offs["KNYC"]:offs["KNYC"] + 7] == b"\n KNYC "


class _Resp:
    def __init__(self, body, status, total):
        self.content, self.status_code = body, status
        self.headers = {"Last-Modified": "Sun, 06 Sep 2026 02:21:38 GMT", "Content-Range": f"bytes 0-1/{total}"}


def test_reader_follows_a_shifted_layout_by_range_reads():
    files = {}
    r = nbp.Reader(["KNYC", "KMIA"])
    calls = []

    def fake_get(url, rng=None):
        data = files[url]
        calls.append(rng)
        if rng is None:
            return _Resp(data, 200, len(data))
        return _Resp(data[rng[0]:rng[1] + 1], 206, len(data))

    r._get = fake_get
    run2 = RUN + pd.Timedelta(days=1)
    files[nbp.url_for(RUN)] = _file()
    files[nbp.url_for(run2)] = _file(pad=3).replace(b"9/06/2026", b"9/07/2026")   # shifted by 3 filler lines
    rows, posted = r.read(RUN)
    assert len(rows) == 18 and posted == pd.Timestamp("2026-09-06 02:21:38") and r.stats["full"] == 1
    rows2, _ = r.read(run2)
    assert len(rows2) == 18 and r.stats["range"] == 1 and all(c is not None for c in calls[1:])
    assert rows2[0][1] == pd.Timestamp("2026-09-07")


def test_nbp_columns():
    df = pd.DataFrame({"nbs_txn": [74.0, np.nan], "nbp_mn": [73.0, np.nan], "nbp_p10": [69.0, np.nan],
                       "nbp_p25": [72.0, np.nan], "nbp_p50": [73.0, np.nan], "nbp_p75": [75.0, np.nan],
                       "nbp_p90": [76.0, np.nan], "nbp_age_h": [2.0, np.nan]})
    x = F.add_nbp_columns(pd.DataFrame({"anchor": [74.0, 80.0]}), df)
    assert list(x.loc[0, F.NBP_FEATS]) == [-1.0, -5.0, 2.0, 3.0, -1.0, -1.0, 2.0]
    assert x.loc[1, F.NBP_FEATS].isna().all()


def test_latest_takes_each_days_newest_run_posted_in_time(tmp_path):
    """Per station and day, the newest run (of the archive's hours) posted 5 minutes before the decision that carries
    the day's max, as research/ml/r5_nbp_feats.py chose it; a run posted too late is skipped; parsed runs cached."""
    class FakeReader:
        def __init__(self):
            self.calls = []

        def read(self, run):
            self.calls.append(run)
            if run.hour == 1:
                return [("KNYC", pd.Timestamp("2026-09-06"), 73, 3, 69, 72, 73, 75, 76),
                        ("KNYC", pd.Timestamp("2026-09-07"), 77, 1, 76, 77, 77, 78, 79)], pd.Timestamp("2026-09-06 02:21")
            if run.hour == 7:          # posted too late for a 08:10 decision
                return [("KNYC", pd.Timestamp("2026-09-06"), 74, 3, 70, 72, 74, 75, 77)], pd.Timestamp("2026-09-06 08:20")
            return None

    rd = FakeReader()
    need = {("KNYC", pd.Timestamp("2026-09-06"))}
    got = nbp.latest(pd.Timestamp("2026-09-06 08:10"), ["KNYC"], str(tmp_path), need=need, reader=rd)
    assert got["rows"][("KNYC", pd.Timestamp("2026-09-06"))][:7] == (73, 3, 69, 72, 73, 75, 76)
    df = pd.DataFrame({"icao": ["KNYC", "KNYC"], "D": pd.to_datetime(["2026-09-06", "2026-09-08"]), "h": [8, 8]})
    c = nbp.frame_columns(df, got)
    assert c.loc[0, "nbp_p50"] == 73 and c.loc[0, "nbp_age_h"] == 7.0 and c.loc[1].isna().all()
    n = len(rd.calls)
    nbp.latest(pd.Timestamp("2026-09-06 08:10"), ["KNYC"], str(tmp_path), need=need, reader=rd)
    assert len(rd.calls) == n                      # both runs came from the cache this time
    later = nbp.latest(pd.Timestamp("2026-09-06 09:00"), ["KNYC"], str(tmp_path), need=need, reader=rd)
    assert later["rows"][("KNYC", pd.Timestamp("2026-09-06"))][4] == 74     # the 07Z run once it is public
