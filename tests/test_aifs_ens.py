"""Candidate C's live AIFS ENS reader (wxbot/ml/aifs_ens.py): run choice, steps, the archive's rounding, inputs."""
import json
import os

import numpy as np
import pandas as pd

from wxbot.ml import aifs_ens as A


def test_run_choice_and_steps_match_the_research_rule():
    dec = pd.Timestamp("2026-10-08 15:00")
    assert A.run_for(dec) == pd.Timestamp("2026-10-08 06:00")              # 15Z - 7 h = 08Z -> the 06Z run
    assert A.run_for(pd.Timestamp("2026-10-08 18:00")) == pd.Timestamp("2026-10-08 06:00")
    assert A.run_for(pd.Timestamp("2026-10-08 19:00")) == pd.Timestamp("2026-10-08 12:00")
    D = pd.Timestamp("2026-10-09")
    steps = A.steps_needed(pd.Timestamp("2026-10-08 06:00"), [(D, dec, -5), (D, dec, -8)])
    assert steps == [24, 30, 36, 42, 48]                                   # 06Z D through 06Z D+1 (EST..PST days)


def test_bitround_matches_the_archive_and_is_idempotent():
    x = np.array([23.4, 22.37, 25.06, -3.3, 0.0, 30.1])
    r = A.bitround(x)
    assert list(r[:3]) == [23.375, 22.375, 25.0] and r[4] == 0.0
    assert np.array_equal(A.bitround(r), r)
    b = r.astype(np.float32).view(np.uint32) & 0xFFFF                    # 16 low mantissa bits dropped
    assert (b == 0).all()


def research_features(T_c, init, steps, rows):
    """research/ml/r4_ens_features.features for one run (archive leads 0..60 every 6 h)."""
    out = []
    leads = np.arange(0, 61, 6)
    full = np.full((T_c.shape[0], len(leads), T_c.shape[2]), np.nan)
    for j, s in enumerate(steps):
        full[:, list(leads).index(s), :] = T_c[:, j, :]
    T = full * 9 / 5 + 32
    H = np.timedelta64(1, "h")
    valid = np.datetime64(init) + leads * H
    for r in rows.itertuples():
        D = np.datetime64(r.D)
        dec, start = D + int(r.h) * H, D + (-int(r.lst_off)) * H
        inday = (valid >= max(dec, start)) & (valid < start + 24 * H)
        X = T[r.k]
        M = np.where(inday[:, None] & np.isfinite(X), X, -np.inf).max(axis=0)
        good = inday.any() and np.isfinite(M).all()
        out.append([np.nanmean(M), np.nanstd(M), np.nanpercentile(M, 10), np.nanpercentile(M, 90),
                    (dec - np.datetime64(init)) / H] if good else [np.nan] * 5)
    return np.array(out)


def test_features_equal_the_research_computation():
    rng = np.random.default_rng(1)
    init = pd.Timestamp("2026-09-15 06:00")
    steps = [24, 30, 36, 42, 48]
    T_c = A.bitround(rng.normal(25, 3, (3, len(steps), 51)))               # 3 stations, already archive-rounded
    run = {"t2m_c": T_c, "stations": ["KAAA", "KBBB", "KCCC"], "steps": steps}
    rows = pd.DataFrame({"icao": ["KAAA", "KBBB", "KCCC", "KAAA"], "D": pd.to_datetime(["2026-09-16"] * 4),
                         "h": [-9, -6, -9, 9], "lst_off": [-5, -8, -6, -5], "k": [0, 1, 2, 0]})
    got = A.features(rows, run, init)[["aifs_mx", "aifs_sd", "aifs_q10", "aifs_q90", "aifs_age_h"]].to_numpy(float)
    ref = research_features(T_c, init, steps, rows)
    assert np.allclose(got, ref, atol=1e-9, equal_nan=True)
    short = dict(run, steps=steps[:-1], t2m_c=T_c[:, :-1, :])               # a needed step missing: no inputs
    assert A.features(rows.iloc[[1]], short, init)["aifs_mx"].isna().all()


def test_fetch_run_reads_all_51_members_per_step_and_merges(tmp_path, monkeypatch):
    coords = {"KAAA": {"lat": 40.0, "lon": -74.0}, "KBBB": {"lat": 34.0, "lon": -118.25}}
    calls = []

    def fields(base):
        n = 1 if base.endswith("-cf") else 50
        return [{"param": "2t", "levtype": "sfc", "number": str(0 if n == 1 else i + 1), "_offset": i, "_length": 1}
                for i in range(n)]
    monkeypatch.setattr(A, "_fields", fields)
    monkeypatch.setattr(A, "_decode_points", lambda content, cs, names: np.array([float(content[0])] * len(names)))

    def fetch_bytes(url, rng):
        calls.append(url)
        return bytes([int(url.split("-")[-3][:-1]) % 250])                # the step as the "temperature"
    init = pd.Timestamp("2026-10-08 06:00")
    r = A.fetch_run(init, [24, 30], str(tmp_path), coords, fetch_bytes=fetch_bytes)
    assert r["steps"] == [24, 30] and r["t2m_c"].shape == (2, 2, 51) and len(calls) == 102
    assert np.allclose(r["t2m_c"][:, 0, :], 24) and np.allclose(r["t2m_c"][:, 1, :], 30)
    r2 = A.fetch_run(init, [24, 30, 36], str(tmp_path), coords, fetch_bytes=fetch_bytes)
    assert r2["steps"] == [24, 30, 36] and len(calls) == 153                # only the new step was fetched
    assert A.load(str(tmp_path), init)["steps"] == [24, 30, 36]


def test_nearest_point_on_the_open_data_grid():
    # ECMWF open data 0.25 deg: first point 90N, -180E, 1440 x 721
    k = A._nearest(90.0, -180.0, 0.25, 1440, 721, 40.78, -73.97)
    j, i = divmod(k, 1440)
    assert abs((90 - j * 0.25) - 40.78) <= 0.125 and abs((-180 + i * 0.25) - (-73.97)) <= 0.125
    k2 = A._nearest(90.0, 0.0, 0.25, 1440, 721, 40.78, -73.97)            # a 0..360 grid gives the same point
    j2, i2 = divmod(k2, 1440)
    assert j2 == j and abs((i2 * 0.25) - (360 - 73.97)) <= 0.125
