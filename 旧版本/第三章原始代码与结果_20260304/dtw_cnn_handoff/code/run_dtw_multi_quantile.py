"""DTW MultiTemplate builder with duration-quantile template selection.

This module is used by Chapter-3 experiments to construct the *DTW-MultiTemplate*
representation:

- For each class, select K templates according to *duration quantiles* (proxy for
  speed-induced time-scale variation).
- For every sample, align its 4-channel sequence [x,y,z,mag] to each template
  using DTW (path computed on mag channel; the full 3-axis signals are warped
  along that path).
- Concatenate all aligned sequences along the channel dimension and feed into a
  1D-CNN.

Important implementation notes (bug-fix / consistency):

- The low-level DTW warping kernel is `run_dtw_multi_sweep.dtw_warp_mv(X, Y, w, step_penalty)`
  where **X/Y must be (T, C)** (time-first) and `w` is an integer Sakoe-Chiba
  half-window radius.
- In the thesis codebase we store sequences as **(C, L)** or batches as
  **(N, C, L)**. Therefore, we must transpose before/after calling the kernel.
- Earlier drafts incorrectly called the kernel with keyword args (wR=..., step=...)
  and without transposing; this file is the corrected, stable version.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from run_dtw_multi_sweep import dtw_warp_mv


def _pick_quantile_indices(idx: np.ndarray, tlen: np.ndarray, K: int) -> List[int]:
    """Pick K indices whose durations are closest to K target quantiles.

    We avoid extreme tails to reduce outlier templates.
    """
    idx = np.asarray(idx, dtype=int)
    if idx.size == 0:
        return []

    d = tlen[idx].astype(float)

    if K <= 1:
        q_targets = [0.50]
    else:
        q_targets = np.linspace(0.15, 0.85, K)

    picks: List[int] = []
    for q in q_targets:
        target = float(np.quantile(d, q))
        j = int(idx[np.argmin(np.abs(d - target))])
        picks.append(j)

    # de-duplicate while keeping order
    seen = set()
    uniq: List[int] = []
    for j in picks:
        if j not in seen:
            uniq.append(j)
            seen.add(j)

    # fill if duplicates happen
    if len(uniq) < K:
        remain = [int(i) for i in idx.tolist() if int(i) not in seen]
        med = float(np.median(d))
        remain.sort(key=lambda i: abs(float(tlen[i]) - med))
        for i in remain:
            uniq.append(int(i))
            if len(uniq) >= K:
                break

    return uniq[:K]


def build_templates_quantile(
    X_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    tlen: np.ndarray,
    templates_per_class: int,
) -> np.ndarray:
    """Return templates array of shape (C, K, Ch, L)."""

    X_raw = np.asarray(X_raw, dtype=np.float32)
    y = np.asarray(y, dtype=int)
    idx_tr = np.asarray(idx_tr, dtype=int)
    tlen = np.asarray(tlen, dtype=int)

    C = int(np.max(y) + 1)
    K = int(templates_per_class)
    _, Ch, L = X_raw.shape

    templates = np.zeros((C, K, Ch, L), dtype=np.float32)
    for c in range(C):
        idx_c = idx_tr[y[idx_tr] == c]
        picks = _pick_quantile_indices(idx_c, tlen, K)
        for k, j in enumerate(picks):
            templates[c, k] = X_raw[j]

    return templates


def build_dtw_multi_quantile(
    X_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    tlen: np.ndarray,
    templates_per_class: int = 4,
    wR: float = 0.15,
    step: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build DTW multi-template aligned representation using quantile templates.

    Parameters
    ----------
    X_raw: np.ndarray
        Shape (N, Ch, L), Ch=4 for [x,y,z,mag].
    templates_per_class: int
        Number of templates per class (K).
    wR: float
        Window radius ratio. `w = round(wR * L)`.
    step: float
        Step penalty in DTW.

    Returns
    -------
    X_multi: np.ndarray
        Shape (N, Ch*(C*K), L)
    templates: np.ndarray
        Shape (C, K, Ch, L)
    """

    X_raw = np.asarray(X_raw, dtype=np.float32)
    y = np.asarray(y, dtype=int)

    N, Ch, L = X_raw.shape
    C = int(np.max(y) + 1)
    K = int(templates_per_class)

    templates = build_templates_quantile(X_raw, y, idx_tr, tlen, templates_per_class=K)

    X_multi = np.zeros((N, Ch * C * K, L), dtype=np.float32)

    # DTW window radius (Sakoe-Chiba band)
    w = int(np.round(float(wR) * float(L)))
    w = max(1, w)

    # warmup numba JIT
    _ = dtw_warp_mv(X_raw[0].T, templates[0, 0].T, w, float(step))

    col = 0
    for c in range(C):
        for k in range(K):
            T = templates[c, k]
            for i in range(N):
                ali = dtw_warp_mv(X_raw[i].T, T.T, w, float(step))  # (L,Ch)
                X_multi[i, col : col + Ch, :] = ali.T
            col += Ch

    return X_multi, templates
