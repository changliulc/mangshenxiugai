from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from run_dtw_clsmin_dba import dtw_distance_1d, dtw_warp_mv
from run_dtw_multi_quantile import build_templates_quantile


def _duration_bins(idx: np.ndarray, tlen: np.ndarray, templates_per_class: int) -> List[np.ndarray]:
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return [np.asarray([], dtype=int) for _ in range(int(templates_per_class))]
    order = np.argsort(tlen[idx], kind="mergesort")
    idx_sorted = idx[order]
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(idx_sorted, int(templates_per_class))]


def _medoid_index_from_ids(
    X_raw: np.ndarray,
    ids: np.ndarray,
    wR: float,
    step: float,
) -> int:
    ids = np.asarray(ids, dtype=int)
    if ids.size == 0:
        raise ValueError("medoid selection requires a non-empty id set")
    if ids.size == 1:
        return int(ids[0])

    L = int(X_raw.shape[2])
    w = max(1, int(round(float(wR) * float(L))))
    _ = dtw_distance_1d(X_raw[int(ids[0]), 3], X_raw[int(ids[0]), 3], w, float(step))

    costs = np.zeros((ids.size,), dtype=np.float64)
    for i in range(ids.size):
        xi = X_raw[int(ids[i]), 3]
        for j in range(i + 1, ids.size):
            d = float(dtw_distance_1d(xi, X_raw[int(ids[j]), 3], w, float(step)))
            costs[i] += d
            costs[j] += d
    return int(ids[int(np.argmin(costs))])


def _dba_template_from_ids(
    X_raw: np.ndarray,
    ids: np.ndarray,
    wR: float,
    step: float,
    n_iter: int,
    init_template: np.ndarray | None = None,
) -> np.ndarray:
    ids = np.asarray(ids, dtype=int)
    if ids.size == 0:
        raise ValueError("DBA template requires a non-empty id set")

    L = int(X_raw.shape[2])
    w = max(1, int(round(float(wR) * float(L))))
    if init_template is None:
        tpl = X_raw[ids].mean(axis=0).astype(np.float32)
    else:
        tpl = np.asarray(init_template, dtype=np.float32).copy()

    _ = dtw_warp_mv(X_raw[int(ids[0])].T, tpl.T, w, float(step))
    for _ in range(int(n_iter)):
        acc = np.zeros_like(tpl, dtype=np.float32)
        for idx in ids:
            warped = dtw_warp_mv(X_raw[int(idx)].T, tpl.T, w, float(step))
            acc += warped.T
        tpl = (acc / float(ids.size)).astype(np.float32)
    return tpl


def build_templates_binned_medoid(
    X_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    tlen: np.ndarray,
    templates_per_class: int,
    wR: float,
    step: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    X_raw = np.asarray(X_raw, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    idx_tr = np.asarray(idx_tr, dtype=int)
    tlen = np.asarray(tlen, dtype=int)

    C = int(np.max(y) + 1)
    K = int(templates_per_class)
    _, Ch, L = X_raw.shape

    templates = np.zeros((C, K, Ch, L), dtype=np.float32)
    source_indices: List[List[int]] = []
    bin_sizes: List[List[int]] = []
    for c in range(C):
        idx_c = idx_tr[y[idx_tr] == c]
        bins = _duration_bins(idx_c, tlen, K)
        class_sources: List[int] = []
        class_sizes: List[int] = []
        fallback = int(idx_c[len(idx_c) // 2])
        for k, ids_bin in enumerate(bins):
            ids_use = ids_bin if ids_bin.size > 0 else np.asarray([fallback], dtype=int)
            medoid_idx = _medoid_index_from_ids(X_raw, ids_use, wR=wR, step=step)
            templates[c, k] = X_raw[medoid_idx]
            class_sources.append(int(medoid_idx))
            class_sizes.append(int(ids_use.size))
        source_indices.append(class_sources)
        bin_sizes.append(class_sizes)

    return templates, {"family": "medoid", "source_indices": source_indices, "bin_sizes": bin_sizes}


def build_templates_binned_dba(
    X_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    tlen: np.ndarray,
    templates_per_class: int,
    wR: float,
    step: float,
    n_iter: int = 5,
) -> Tuple[np.ndarray, Dict[str, object]]:
    X_raw = np.asarray(X_raw, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    idx_tr = np.asarray(idx_tr, dtype=int)
    tlen = np.asarray(tlen, dtype=int)

    C = int(np.max(y) + 1)
    K = int(templates_per_class)
    _, Ch, L = X_raw.shape

    templates = np.zeros((C, K, Ch, L), dtype=np.float32)
    init_indices: List[List[int]] = []
    bin_sizes: List[List[int]] = []
    for c in range(C):
        idx_c = idx_tr[y[idx_tr] == c]
        bins = _duration_bins(idx_c, tlen, K)
        class_init: List[int] = []
        class_sizes: List[int] = []
        fallback = int(idx_c[len(idx_c) // 2])
        for k, ids_bin in enumerate(bins):
            ids_use = ids_bin if ids_bin.size > 0 else np.asarray([fallback], dtype=int)
            medoid_idx = _medoid_index_from_ids(X_raw, ids_use, wR=wR, step=step)
            init_tpl = X_raw[medoid_idx]
            templates[c, k] = _dba_template_from_ids(
                X_raw,
                ids_use,
                wR=wR,
                step=step,
                n_iter=n_iter,
                init_template=init_tpl,
            )
            class_init.append(int(medoid_idx))
            class_sizes.append(int(ids_use.size))
        init_indices.append(class_init)
        bin_sizes.append(class_sizes)

    return templates, {
        "family": "dba",
        "init_medoid_indices": init_indices,
        "bin_sizes": bin_sizes,
        "dba_iter": int(n_iter),
    }


def build_templates_family(
    X_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    tlen: np.ndarray,
    templates_per_class: int,
    family: str,
    wR: float,
    step: float,
    dba_iter: int = 5,
) -> Tuple[np.ndarray, Dict[str, object]]:
    family = str(family)
    if family == "raw_quantile":
        templates = build_templates_quantile(X_raw, y, idx_tr, tlen, templates_per_class=templates_per_class)
        return templates, {"family": family}
    if family == "medoid":
        return build_templates_binned_medoid(
            X_raw,
            y,
            idx_tr,
            tlen,
            templates_per_class=templates_per_class,
            wR=wR,
            step=step,
        )
    if family == "dba":
        return build_templates_binned_dba(
            X_raw,
            y,
            idx_tr,
            tlen,
            templates_per_class=templates_per_class,
            wR=wR,
            step=step,
            n_iter=dba_iter,
        )
    raise ValueError(f"Unsupported family: {family}")


def build_dtw_multi_family(
    X_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    tlen: np.ndarray,
    templates_per_class: int = 4,
    family: str = "raw_quantile",
    wR: float = 0.15,
    step: float = 0.05,
    dba_iter: int = 5,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    X_raw = np.asarray(X_raw, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    N, Ch, L = X_raw.shape
    C = int(np.max(y) + 1)
    K = int(templates_per_class)

    templates, meta = build_templates_family(
        X_raw,
        y,
        idx_tr,
        tlen,
        templates_per_class=K,
        family=family,
        wR=wR,
        step=step,
        dba_iter=dba_iter,
    )

    X_multi = np.zeros((N, Ch * C * K, L), dtype=np.float32)
    w = max(1, int(round(float(wR) * float(L))))
    _ = dtw_warp_mv(X_raw[0].T, templates[0, 0].T, w, float(step))

    col = 0
    for c in range(C):
        for k in range(K):
            tpl = templates[c, k]
            for i in range(N):
                ali = dtw_warp_mv(X_raw[i].T, tpl.T, w, float(step))
                X_multi[i, col : col + Ch, :] = ali.T
            col += Ch

    meta = dict(meta)
    meta.update(
        {
            "family": family,
            "templates_per_class": int(K),
            "wR": float(wR),
            "step": float(step),
            "templates_shape": list(templates.shape),
        }
    )
    return X_multi, templates, meta
