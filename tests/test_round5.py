"""Round 5 (candidate D) plumbing: the default model path is unchanged; the median anchor and the NBM-departure
inputs; a named model loads with its own parity rows."""
import numpy as np
import pandas as pd

from wxbot.ml import features as F


def rows():
    import re
    src = open(F.__file__).read()
    body = src[src.index("def model_features"):]
    cols = sorted(set(re.findall(r'"(\w+)"', body)))
    base = pd.DataFrame({c: [np.nan] * 3 for c in cols})
    base["icao"] = ["KNYC", "KMIA", "KLAX"]
    return base.assign(**_rows().to_dict("series"))


def _rows():
    return pd.DataFrame({
        "icao": ["KNYC", "KMIA", "KLAX"], "D": pd.to_datetime(["2026-07-01", "2026-08-15", "2026-08-15"]),
        "h": [3, 3, 15], "lst_off": [-5, -5, -8], "nbs_txn": [80.0, 88.0, np.nan], "gfs_n_x": [83.0, 90.0, 75.0],
        "hrrr_rest": [84.0, 89.0, 74.0], "ec_max": [79.0, 87.0, np.nan], "lav_tmax": [83.7, 91.7, 76.7],
        "obs_max": [np.nan, np.nan, 77.0], "nam_n_x": [82.0, 89.0, 74.0], "nam_age_h": [5.0, 5.0, 5.0],
        "gfs_age_h": [3.0, 3.0, 3.0]})


def test_guidance_median_and_anchor():
    df = rows()
    med = F.guidance_median(df)
    assert list(med) == [82.0, 89.0, 75.0]                    # LAMP enters at -1.7 F; missing sources skipped
    a, base = F.anchor_median(df)
    assert list(base) == [82.0, 89.0, 75.0] and list(a) == [82.0, 89.0, 77.0]   # floored at the max so far
    a0, _ = F.anchor(df)
    assert list(a0[:2]) == [80.0, 88.0]                       # NBM's anchor unchanged


def test_default_model_features_unchanged_and_round5_columns():
    df = rows()
    x0 = F.model_features(df)
    x1 = F.model_features(df, anchor_mode="nbm")
    pd.testing.assert_frame_equal(x0, x1)
    xm = F.model_features(df, anchor_mode="median")
    assert list(xm["anchor"]) == [82.0, 89.0, 77.0] and list(xm["txn_d"][:2]) == [-2.0, -1.0]
    x = F.add_round5_columns(xm, df)
    assert list(x["nbm_dep"][:2]) == [-2.0, -1.0] and list(x["gmed_d"]) == [0.0, 0.0, -2.0]
    assert list(x["nbm_v5014"]) == [0, 1, 1]


def test_a_named_model_loads_with_its_own_parity_rows(tmp_path):
    import os
    from wxbot.ml.live import ART, Tier1Live
    if not os.path.exists(os.path.join(ART, "tier1_r2b_meta.json")):
        return
    import gzip, shutil
    # Tier1Live reads an uncompressed or gzipped text model named after the artifact
    t = Tier1Live(str(tmp_path), ["KNYC"], model="tier1_r2b")
    assert t.parity_max_diff < 1e-9 and t.model_name == "tier1_r2b" and t.anchor_mode == "nbm" and not t.round5
