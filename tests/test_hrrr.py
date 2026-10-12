"""HRRR feature helpers: projection, run selection and the feature math on a synthetic run."""
import numpy as np
import pandas as pd

from wxbot.ml import hrrr as H


def test_projection_matches_grid_corner():
    # the HRRR grid's south-west corner (21.138 N, 122.72 W) maps to (X0, Y0)
    x, y = H.lcc_xy(-122.719528, 21.138123)
    assert abs(x - H.X0) < 50 and abs(y - H.Y0) < 50


def test_runs_needed_respects_lag():
    r = H.runs_needed(pd.Timestamp("2026-10-03 16:05"), [-5, -8], history_days=1)
    inits = sorted(r)
    assert pd.Timestamp("2026-10-03 12:00") in r  # 16:05 - 3 h -> the 12Z run, not 13Z or later
    assert all(i <= pd.Timestamp("2026-10-03 13:05") for i in inits)
    assert max(r[pd.Timestamp("2026-10-03 12:00")]) == 20  # through 08Z next day (end of the Pacific day)


def _synthetic(init, base):
    # 48 forecast hours of a smooth diurnal curve peaking at 21Z
    rows = []
    for fh in range(1, 49):
        t = init + pd.Timedelta(hours=fh)
        rows.append(["KNYC", init, fh, base + 10 * np.cos((t.hour - 21) / 24 * 2 * np.pi)])
    d = pd.DataFrame(rows, columns=["icao", "init", "fh", "t2m_f"])
    return d.set_index(["icao", "init", "fh"])["t2m_f"].unstack("fh")


def test_compute_rest_of_day_max_and_errors():
    W = pd.concat([_synthetic(pd.Timestamp("2026-08-15 12:00"), 80.0),
                   _synthetic(pd.Timestamp("2026-08-15 06:00"), 79.0)]).sort_index()
    df = pd.DataFrame({"icao": ["KNYC"], "D": [pd.Timestamp("2026-08-15")], "h": [15], "lst_off": [-5],
                       "obs_max": [86.0], "obs_last": [85.0], "obs_age_h": [0.1]})
    out = H.compute(df, W)
    assert abs(out["hrrr_rest"].iloc[0] - 90.0) < 1e-9  # the 21Z peak is still ahead at 15Z
    assert out["hrrr_age_h"].iloc[0] == 3.0  # 12Z run at a 15Z decision
    assert out["hrrr_peak_lh"].iloc[0] == 16  # 21Z = 16 local standard time
    assert abs(out["hrrr_trend"].iloc[0] - 1.0) < 1e-9  # 12Z run is 1 F warmer than the 06Z run
    assert np.isfinite(out["hrrr_now_err"].iloc[0]) and np.isfinite(out["hrrr_sofar_err"].iloc[0])
