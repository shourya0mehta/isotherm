"""Frozen-artifact provenance for the two pre-registered holdouts (outside review of Oct 8, 2026).

docs/HOLDOUT_MANIFEST.json holds the sha256 of every frozen artifact the Dec 1 test (docs/HOLDOUT_DEC1.md, Oct 1 -
Nov 30) and the Feb 1 test (docs/HOLDOUT_DECJAN.md, Dec 1 - Jan 31) use: the forecasters, pool weights, the pool's
refit base rows, the rain models, the nowcast models and the early-shadow candidates B, C and G, plus the walk-forward
research rows outside the repo that fill the pool's window before Oct 1. Each scoring script is pinned by its sha256
and the git commit that last changed it before the manifest was written.

The scorers (research/ml/holdout_eval.py, research/hourly/holdout_h.py, research/rain/holdout_r.py) call require()
before anything else: a changed or missing artifact, research input or scoring script stops the scoring with the list
of differences. Library code the scorers share with the live bot (features, quoting, risk caps) is recorded under
"dependencies" for provenance and reported when it differs, without stopping the scoring: its effect on the frozen
forecasts is checked by the forecaster's own parity file (tier1_parity.json.gz) at the bot's start.

The Feb 1 set is the state before the Dec 1 freeze (candidates B and C as the early shadow runs them until then, spec
G, the tilt, candidate D as prepared). The freeze retrains A, B, C and D through Nov 30 (tag round2-2026-12-01) and
must add those artifacts with `write` before any Dec-Jan scoring.

Changing a pinned file before its holdout is scored is allowed only as a deliberate, dated act: rewrite the manifest
with `write --note "..."` in the same commit and say why in the holdout's file. The manifest keeps its history.
Score from a clean checkout of the scoring commit: a bot running in the same folder with the default refit path (a
config without ml.weekly_refit.weights_path) rewrites wxbot/ml/artifacts/pool_weights.json on Mondays, and the check
refuses that copy, as it should.

Usage:
  python3 research/ml/holdout_manifest.py check [dec1|feb1|all] [--no-external]   exit 1 on any difference
  python3 research/ml/holdout_manifest.py write --note "why"                       rewrite from the files as they are
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
MANIFEST = os.path.join(ROOT, "docs", "HOLDOUT_MANIFEST.json")
GATE_B_TAG = "gate-b-2026-10-03"
A = "wxbot/ml/artifacts/"

# ---------------------------------------------------------------------------------------------------- what is pinned
# path -> (what it is, the registered items that use it); taken from docs/HOLDOUT_DEC1.md and docs/HOLDOUT_DECJAN.md
ARTIFACTS = {
    A + "tier1_v2.txt.gz": ("frozen forecaster (tier1_v3+hrrr, trained through Sep 30)", ["dec1:1", "dec1:2", "dec1:3",
                                                                                          "dec1:4", "dec1:5", "dec1:A9",
                                                                                          "feb1:shipped"]),
    A + "tier1_v2_meta.json": ("frozen forecaster's features, classes and text sha256", ["dec1:1-5", "dec1:A9",
                                                                                         "feb1:shipped"]),
    A + "tier1_parity.json.gz": ("frozen forecaster's parity rows (the bot's start check)", ["dec1:4b", "dec1:5"]),
    A + "pool_weights.json": ("frozen pool weights at the freeze (BTh per hour)", ["dec1:4b", "dec1:5"]),
    A + "pool_rows_base.json.gz": ("frozen pool's refit base rows (research ladders before the bot's own)",
                                   ["dec1:4b", "dec1:5"]),
    A + "pool_weights_gate.json": ("tilted pool weights (T1, live from Stage 3)", ["dec1:A9", "dec1:A10",
                                                                                 "feb1:T1a"]),
    A + "pool_rows_base_gate.json.gz": ("tilted pool's refit base rows", ["dec1:A9", "dec1:A10", "feb1:T1a"]),
    A + "nowcast_meta.json": ("frozen hourly nowcast (Sleeve H2)", ["dec1:6"]),
    A + "nowcast_miami.txt.gz": ("frozen hourly nowcast, Miami", ["dec1:6"]),
    A + "nowcast_nyc.txt.gz": ("frozen hourly nowcast, New York", ["dec1:6"]),
    A + "nowcast_chicago.txt.gz": ("frozen hourly nowcast, Chicago", ["dec1:6"]),
    A + "nowcast_la-coastal.txt.gz": ("frozen hourly nowcast, LA coastal", ["dec1:6"]),
    A + "rain_tier1.txt.gz": ("frozen rain model (Sleeve R)", ["dec1:7"]),
    A + "rain_meta.json": ("frozen rain model's features and pool weights (Sleeve R)", ["dec1:7"]),
    A + "rain_tier1_v2.txt.gz": ("frozen rain model version 2 (R2)", ["dec1:7-R2"]),
    A + "rain_meta_v2.json": ("frozen rain model version 2's features and pool weights (R2)", ["dec1:7-R2"]),
    A + "tier1_r2b.txt.gz": ("candidate B, trained through Sep 30 (early shadow until the freeze)", ["feb1:B3"]),
    A + "tier1_r2b_meta.json": ("candidate B's meta", ["feb1:B3"]),
    A + "tier1_r2b_parity.json.gz": ("candidate B's parity rows", ["feb1:B3"]),
    A + "pool_weights_r2b.json": ("candidate B's pool weights", ["feb1:B3"]),
    A + "pool_rows_base_r2b.json.gz": ("candidate B's refit base rows", ["feb1:B3"]),
    A + "tier1_r2c.txt.gz": ("candidate C, trained through Sep 30 (early shadow until the freeze)", ["feb1:C3"]),
    A + "tier1_r2c_meta.json": ("candidate C's meta", ["feb1:C3"]),
    A + "tier1_r2c_parity.json.gz": ("candidate C's parity rows", ["feb1:C3"]),
    A + "pool_weights_r2c.json": ("candidate C's pool weights", ["feb1:C3"]),
    A + "pool_rows_base_r2c.json.gz": ("candidate C's refit base rows", ["feb1:C3"]),
    A + "pool_weights_r2cg.json": ("spec G: C's forecaster with the tilt (Sleeve E)", ["feb1:T1b"]),
    A + "pool_rows_base_r2cg.json.gz": ("spec G's refit base rows", ["feb1:T1b"]),
    A + "tier1_d4x.txt.gz": ("candidate D as prepared (trained through Sep 30; switch not approved)", ["feb1:D1"]),
    A + "tier1_d4x_meta.json": ("candidate D's meta", ["feb1:D1"]),
    A + "tier1_d4x_parity.json.gz": ("candidate D's parity rows", ["feb1:D1"]),
    A + "pool_weights_gate_d4x.json": ("candidate D's tilted pool weights", ["feb1:D1"]),
    A + "pool_rows_base_gate_d4x.json.gz": ("candidate D's refit base rows", ["feb1:D1"]),
}
# model text -> its meta (the meta's sha256 is the decompressed model text's)
MODEL_META = {A + "tier1_v2.txt.gz": A + "tier1_v2_meta.json", A + "rain_tier1.txt.gz": A + "rain_meta.json",
              A + "rain_tier1_v2.txt.gz": A + "rain_meta_v2.json", A + "tier1_r2b.txt.gz": A + "tier1_r2b_meta.json",
              A + "tier1_r2c.txt.gz": A + "tier1_r2c_meta.json", A + "tier1_d4x.txt.gz": A + "tier1_d4x_meta.json"}
# research inputs outside the repo (MLWX): the walk-forward rows the scorer's pool window reads before Oct 1
EXTERNAL = {
    "t2r1_hrrr.pkl": ("walk-forward Tier-2 rows of the shipped forecaster (pool window before Oct 1)",
                      ["dec1:1", "dec1:2", "dec1:3", "dec1:4", "dec1:A9"]),
    "tier1_r1_hrrr_oos.parquet": ("walk-forward Tier-1 ladders of the shipped forecaster (maker replay before Oct 1)",
                                  ["dec1:3", "dec1:A9"]),
}
SCRIPTS = {
    "dec1": ["research/ml/holdout_eval.py", "research/ml/holdout_manifest.py", "research/ml/refit_cadence.py",
             "research/ml/pool_r1.py", "research/ml/cgep.py", "research/ml/tier2_eval_v2.py",
             "research/ml/tier2_eval.py", "research/ml/build_tier2.py", "research/ml/build_tier2_v2.py",
             "research/ml/tier1_r1_oos.py", "research/ml/r1_prod_compare.py", "research/ml/taker_prod.py",
             "research/ml/fill_stress.py", "research/ml/pool_replay.py", "research/ml/pool_replay2.py",
             "research/ml/pool_replay_ablation.py", "research/ml/r5_maker_tilt.py", "wxbot/backtest/joint.py",
             "research/hourly/holdout_h.py", "research/hourly/live_replay.py", "research/hourly/maker_replay.py",
             "research/hourly/from_archive.py", "research/hourly/index.py", "research/hourly/nowcast.py",
             "research/hourly/market_eval.py", "research/rain/holdout_r.py", "research/rain/rain_vs_market.py",
             "scripts/paper_report.py", "scripts/ml_scorecard.py", "research/ml/r9_shutoff.py",
             "research/ml/r9_live_caps_replay.py", "research/ml/r8_exit_lib.py", "research/ml/r8_exit_report.py",
             "research/ml/r7_guard_lib.py", "research/ml/r7_obs5_pull.py"],
    "feb1": ["research/ml/holdout_manifest.py", "scripts/freeze.sh", "research/ml/tier1_r1_oos.py",
             "research/ml/train_final.py", "research/ml/train_r2b.py", "research/ml/train_d.py",
             "research/ml/r4_ens_features.py", "research/ml/early_eval.py", "research/ml/r4_c_vs_b.py",
             "research/ml/r5_pool_gate.py", "research/ml/r5_pool_gate_early.py", "research/ml/pool_r1.py",
             "research/ml/cgep.py", "research/ml/tier2_eval_v2.py", "research/ml/taker_prod.py",
             "scripts/paper_report.py"],
}
DEPENDENCIES = {
    "dec1": ["wxbot/ml/features.py", "wxbot/ml/hrrr.py", "wxbot/ml/npgbm.py", "wxbot/ml/live.py",
             "wxbot/ml/pool_fit.py", "wxbot/ml/rain.py", "wxbot/ml/nowcast.py", "wxbot/strategy/sleeve_a.py",
             "wxbot/strategy/sleeve_h.py", "wxbot/strategy/rain_taker.py", "wxbot/risk/limits.py",
             "wxbot/models/fair_value.py", "wxbot/models/brackets.py", "wxbot/execution/paper.py",
             "wxbot/store/db.py", "wxbot/data/tape_archive.py"],
    "feb1": ["wxbot/ml/features.py", "wxbot/ml/hrrr.py", "wxbot/ml/aifs_ens.py", "wxbot/ml/live.py",
             "wxbot/ml/pool_fit.py", "wxbot/app/early_shadow.py", "wxbot/store/db.py"],
}
HOLDOUT_DOC = {"dec1": "docs/HOLDOUT_DEC1.md", "feb1": "docs/HOLDOUT_DECJAN.md"}


def holdout_artifacts(h: str) -> list:
    return [p for p, (_, used) in ARTIFACTS.items() if any(u.startswith(h + ":") for u in used)]


def holdout_external(h: str) -> list:
    return [n for n, (_, used) in EXTERNAL.items() if any(u.startswith(h + ":") for u in used)]


# ---------------------------------------------------------------------------------------------------------- hashing
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(root, *args) -> str:
    try:
        return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def last_commit(root: str, rel: str) -> str:
    """The last commit that changed rel (empty when git or the history is not there)."""
    return _git(root, "log", "-1", "--format=%H", "--", rel)


def tag_sha256(root: str, tag: str, rel: str):
    """sha256 of rel as tag holds it, or None (not in the tag, or no git)."""
    try:
        out = subprocess.run(["git", "-C", root, "show", f"{tag}:{rel}"], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return hashlib.sha256(out).hexdigest()


# ---------------------------------------------------------------------------------------------------------- writing
def build(root: str = ROOT, mlwx: str | None = None, note: str = "", previous: dict | None = None) -> dict:
    """The manifest from the files as they are now (git supplies each script's last commit and the gate-B check)."""
    mlwx = mlwx or os.environ.get("MLWX", "/home/claude/mlwx")
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    files = {}
    for rel, (what, used) in ARTIFACTS.items():
        p = os.path.join(root, rel)
        e = {"kind": "artifact", "sha256": sha256_file(p), "bytes": os.path.getsize(p), "commit": last_commit(root, rel),
             "what": what, "used_by": used}
        g = tag_sha256(root, GATE_B_TAG, rel)
        if g is not None:
            e[f"same_as_{GATE_B_TAG}"] = g == e["sha256"]
        if rel in MODEL_META:
            text = gzip.decompress(open(p, "rb").read())
            meta = json.load(open(os.path.join(root, MODEL_META[rel])))
            e["text_sha256"] = hashlib.sha256(text).hexdigest()
            e["text_sha256_matches_meta"] = e["text_sha256"] == meta.get("sha256")
        files[rel] = e
    for kind, groups in (("script", SCRIPTS), ("dependency", DEPENDENCIES)):
        for h, rels in groups.items():
            for rel in rels:
                if rel in files:                     # pinned already (a script of the other holdout)
                    if h not in files[rel]["used_by"]:
                        files[rel]["used_by"].append(h)
                    continue
                p = os.path.join(root, rel)
                files[rel] = {"kind": kind, "sha256": sha256_file(p), "bytes": os.path.getsize(p),
                              "commit": last_commit(root, rel), "used_by": [h]}
    external = {}
    for name, (what, used) in EXTERNAL.items():
        p = os.path.join(mlwx, name)
        external[name] = {"sha256": sha256_file(p), "bytes": os.path.getsize(p), "what": what, "used_by": used,
                          "where": "MLWX (cloud workspace /home/claude/mlwx; backed up on the Mac, "
                                   "weather_trader/research_backup/)"}
    history = list((previous or {}).get("history") or [])
    history.append({"written": now, "repo_head": _git(root, "rev-parse", "HEAD"), "note": note})
    return {
        "about": "sha256 of every frozen artifact and scoring script the Dec 1 and Feb 1 holdout tests use. The scorers "
                 "check it first (research/ml/holdout_manifest.py require) and refuse on any difference. 'commit' is "
                 "the last commit that changed the file when the manifest was written; artifacts and scripts must "
                 "match exactly, dependencies are reported when they differ.",
        "written": now,
        "repo_head": _git(root, "rev-parse", "HEAD"),
        "note": note,
        "holdouts": {h: {"doc": HOLDOUT_DOC[h], "artifacts": holdout_artifacts(h), "external": holdout_external(h),
                         "scripts": SCRIPTS[h], "dependencies": DEPENDENCIES[h]} for h in ("dec1", "feb1")},
        "files": files,
        "external": external,
        "history": history,
    }


def write(note: str, root: str = ROOT, path: str = MANIFEST, mlwx: str | None = None) -> dict:
    prev = json.load(open(path)) if os.path.exists(path) else None
    m = build(root, mlwx, note, prev)
    bad = [rel for rel, e in m["files"].items() if e.get("text_sha256_matches_meta") is False]
    if bad:
        raise SystemExit(f"model text and meta sha256 disagree: {bad}")
    with open(path + ".tmp", "w") as f:
        json.dump(m, f, indent=1, sort_keys=False)
        f.write("\n")
    os.replace(path + ".tmp", path)
    return m


# --------------------------------------------------------------------------------------------------------- checking
class ManifestMismatch(SystemExit):
    """Raised by require(): a pinned file differs from the manifest (the scoring must not run)."""


def check(holdout: str = "dec1", root: str = ROOT, path: str = MANIFEST, mlwx: str | None = None,
          external: bool = True) -> tuple:
    """Compare the files with the manifest for one holdout ("dec1", "feb1" or "all"). Returns (problems, warnings,
    counts): problems are artifacts, research inputs (external=True) or scripts that are missing or differ;
    warnings are dependencies that differ."""
    m = json.load(open(path))
    mlwx = mlwx or os.environ.get("MLWX", "/home/claude/mlwx")
    hs = ("dec1", "feb1") if holdout == "all" else (holdout,)
    problems, warnings = [], []
    counts = {"artifacts": 0, "scripts": 0, "external": 0, "dependencies": 0}
    seen = set()
    for h in hs:
        spec = m["holdouts"][h]
        for group in ("artifacts", "scripts", "dependencies"):
            for rel in spec[group]:
                if (group, rel) in seen:
                    continue
                seen.add((group, rel))
                want = m["files"][rel]["sha256"]
                p = os.path.join(root, rel)
                got = sha256_file(p) if os.path.exists(p) else None
                counts[group] += 1
                if got == want:
                    continue
                msg = (f"{rel}: missing" if got is None else
                       f"{rel}: sha256 {got[:12]} but the manifest has {want[:12]} (pinned at commit "
                       f"{(m['files'][rel].get('commit') or '?')[:10]})")
                (warnings if group == "dependencies" else problems).append(msg)
        if external:
            for name in spec["external"]:
                if ("external", name) in seen:
                    continue
                seen.add(("external", name))
                counts["external"] += 1
                p = os.path.join(mlwx, name)
                want = m["external"][name]["sha256"]
                got = sha256_file(p) if os.path.exists(p) else None
                if got != want:
                    problems.append(f"MLWX/{name}: missing" if got is None else
                                    f"MLWX/{name}: sha256 {got[:12]} but the manifest has {want[:12]}")
    return problems, warnings, counts


def require(holdout: str = "dec1", root: str = ROOT, path: str = MANIFEST, mlwx: str | None = None,
            external: bool = True, out=print) -> dict:
    """The scorers' first step: stop (ManifestMismatch, a SystemExit) unless every pinned artifact, research input
    and scoring script matches the manifest; report dependencies that differ."""
    if not os.path.exists(path):
        raise ManifestMismatch(f"holdout manifest {path} not found: the scoring does not run without it")
    problems, warnings, counts = check(holdout, root, path, mlwx, external)
    for w in warnings:
        out(f"manifest: dependency changed since the manifest (reported, not blocking): {w}")
    if problems:
        raise ManifestMismatch("holdout manifest check FAILED, scoring refused:\n  " + "\n  ".join(problems))
    m = json.load(open(path))
    out(f"manifest check ({holdout}): {counts['artifacts']} artifacts, {counts['scripts']} scripts"
        + (f", {counts['external']} research inputs" if external else "")
        + f" match docs/HOLDOUT_MANIFEST.json (written {m['written']})")
    return counts


def main(argv):
    if not argv or argv[0] not in ("check", "write"):
        print(__doc__)
        return 2
    if argv[0] == "write":
        note = argv[argv.index("--note") + 1] if "--note" in argv else ""
        if not note:
            print("write needs --note \"why\" (the manifest keeps a dated history)")
            return 2
        m = write(note)
        print(f"wrote {MANIFEST}: {sum(e['kind'] == 'artifact' for e in m['files'].values())} artifacts, "
              f"{sum(e['kind'] == 'script' for e in m['files'].values())} scripts, "
              f"{sum(e['kind'] == 'dependency' for e in m['files'].values())} dependencies, "
              f"{len(m['external'])} research inputs")
        return 0
    h = next((a for a in argv[1:] if not a.startswith("--")), "all")
    try:
        require(h, external="--no-external" not in argv)
    except ManifestMismatch as e:
        print(e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
