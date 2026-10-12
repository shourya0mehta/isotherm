"""Pure-numpy predictor for LightGBM text models (gbdt, multiclass or binary/regression raw scores).

Why: LightGBM's compiled library needs OpenMP (libomp), which macOS lacks without Homebrew. Training stays on
LightGBM; live prediction only needs to walk the saved trees. Decision rules follow LightGBM's tree.h
(NumericalDecision / CategoricalDecision) so outputs match `Booster.predict` to floating-point precision.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

_K_ZERO = 1e-35


class _Tree:
    __slots__ = ("feat", "thr", "dtype", "left", "right", "leaf", "cat_b", "cat_t", "single")

    def __init__(self, kv: dict):
        self.leaf = np.array(kv["leaf_value"].split(), float)
        self.single = int(kv["num_leaves"]) <= 1
        if self.single:
            return
        self.feat = np.array(kv["split_feature"].split(), np.int64)
        self.thr = np.array([float(x) for x in kv["threshold"].split()], float)
        self.dtype = np.array(kv["decision_type"].split(), np.int64)
        self.left = np.array(kv["left_child"].split(), np.int64)
        self.right = np.array(kv["right_child"].split(), np.int64)
        if int(kv.get("num_cat", "0")) > 0:
            self.cat_b = np.array(kv["cat_boundaries"].split(), np.int64)
            self.cat_t = np.array(kv["cat_threshold"].split(), np.int64)
        else:
            self.cat_b = self.cat_t = None

    def leaf_index(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        if self.single:
            return np.zeros(n, np.int64)
        node = np.zeros(n, np.int64)
        while True:
            idx = np.flatnonzero(node >= 0)
            if idx.size == 0:
                return ~node
            nd = node[idx]
            fv = X[idx, self.feat[nd]]
            dt = self.dtype[nd]
            is_cat = (dt & 1) == 1
            mtype = (dt >> 2) & 3
            go_left = np.zeros(idx.size, bool)
            num = ~is_cat
            if num.any():
                v = fv[num]
                m = mtype[num]
                v = np.where(np.isnan(v) & (m != 2), 0.0, v)
                missing = ((m == 1) & (np.abs(v) <= _K_ZERO)) | ((m == 2) & np.isnan(v))
                default_left = (dt[num] & 2) != 0
                with np.errstate(invalid="ignore"):
                    go_left[num] = np.where(missing, default_left, v <= self.thr[nd[num]])
            if is_cat.any():
                v = fv[is_cat]
                m = mtype[is_cat]
                nan = np.isnan(v)
                v = np.where(nan & (m != 2), 0.0, v)  # NaN counts as category 0 unless NaN is its own branch
                ok = ~np.isnan(v) & (v >= 0)
                iv = np.where(ok, v, 0).astype(np.int64)
                ci = self.thr[nd[is_cat]].astype(np.int64)
                start, end = self.cat_b[ci], self.cat_b[ci + 1]
                word, bit = iv // 32, iv % 32
                inr = word < (end - start)
                w = self.cat_t[np.where(inr, start + word, 0)]
                go_left[is_cat] = ok & inr & (((w >> bit) & 1) == 1)
            node[idx] = np.where(go_left, self.left[nd], self.right[nd])


class NumpyBooster:
    def __init__(self, model_str: str):
        i = model_str.find("\nTree=")
        head, body = model_str[:i], model_str[i + 1:].split("end of trees")[0]
        hd = dict(line.split("=", 1) for line in head.splitlines() if "=" in line)
        self.num_class = int(hd.get("num_class", "1"))
        self.per_iter = int(hd.get("num_tree_per_iteration", "1"))
        self.feature_names = hd["feature_names"].split()
        self.objective = hd.get("objective", "")
        self.trees = []
        for block in re.split(r"^Tree=\d+\s*$", body, flags=re.M)[1:]:
            kv = dict(line.split("=", 1) for line in block.strip().splitlines() if "=" in line)
            if "leaf_value" in kv:
                self.trees.append(_Tree(kv))
        self.pandas_categorical = None
        m = re.search(r"pandas_categorical:(.*)\s*$", model_str)
        if m:
            try:
                self.pandas_categorical = json.loads(m.group(1))
            except ValueError:
                self.pandas_categorical = None

    def _matrix(self, X) -> np.ndarray:
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            pd = None
        if pd is not None and isinstance(X, pd.DataFrame):
            cols = []
            cat_i = 0
            for c in X.columns:
                s = X[c]
                if isinstance(s.dtype, pd.CategoricalDtype):
                    if self.pandas_categorical is not None and cat_i < len(self.pandas_categorical):
                        s = s.cat.set_categories(self.pandas_categorical[cat_i])
                    cat_i += 1
                    codes = s.cat.codes.to_numpy(np.float64)
                    codes[codes < 0] = np.nan
                    cols.append(codes)
                else:
                    cols.append(pd.to_numeric(s, errors="coerce").to_numpy(np.float64))
            return np.column_stack(cols) if cols else np.zeros((len(X), 0))
        return np.asarray(X, np.float64)

    def _py_tables(self):
        """Plain-list copies of the trees for fast single-row prediction (no categorical splits)."""
        if getattr(self, "_py", None) is not None:
            return self._py
        tabs = []
        for t in self.trees:
            if t.single:
                tabs.append((True, float(t.leaf[0]), None, None, None, None, None, None))
                continue
            if t.cat_b is not None or bool(np.any((t.dtype & 1) == 1)):
                self._py = False
                return self._py
            tabs.append((False, t.leaf.tolist(), t.feat.tolist(), t.thr.tolist(), t.dtype.tolist(),
                         t.left.tolist(), t.right.tolist(), None))
        self._py = tabs
        return self._py

    def _raw_py(self, row: list) -> list:
        out = [0.0] * self.per_iter
        isnan = math.isnan
        for i, (single, leaf, feat, thr, dt, left, right, _) in enumerate(self._py):
            if single:
                out[i % self.per_iter] += leaf
                continue
            node = 0
            while node >= 0:
                v = row[feat[node]]
                d = dt[node]
                mtype = (d >> 2) & 3
                if v != v and mtype != 2:   # NaN counts as zero unless NaN has its own branch
                    v = 0.0
                if (mtype == 1 and abs(v) <= _K_ZERO) or (mtype == 2 and v != v):
                    go_left = (d & 2) != 0
                else:
                    go_left = v <= thr[node]
                node = left[node] if go_left else right[node]
            out[i % self.per_iter] += leaf[~node]
        return out

    def raw_score(self, X) -> np.ndarray:
        M = self._matrix(X)
        if M.shape[0] <= 8 and self._py_tables():
            return np.array([self._raw_py([float(x) for x in row]) for row in M], float).reshape(M.shape[0], self.per_iter)
        out = np.zeros((M.shape[0], self.per_iter))
        for i, t in enumerate(self.trees):
            out[:, i % self.per_iter] += t.leaf[t.leaf_index(M)]
        return out

    def predict(self, X) -> np.ndarray:
        s = self.raw_score(X)
        if self.objective.startswith("multiclass"):
            s = s - s.max(axis=1, keepdims=True)
            e = np.exp(s)
            return e / e.sum(axis=1, keepdims=True)
        if self.objective.startswith("binary"):
            return 1.0 / (1.0 + np.exp(-s[:, 0]))
        return s[:, 0]
