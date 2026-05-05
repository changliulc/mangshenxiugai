"""Edge (端侧) feature extraction for REALPACK (all-real) data.

This file mirrors the MATLAB implementation used in Chapter 3 edge baselines:

- ch3_feature_names_allreal_v4.m
- ch3_extract_features_allreal_v4.m
- ch3_zero_crossings.m
- ch3_peak_count_and_loc.m

We keep feature ORDER consistent with MATLAB so that the selected feature list can be
ported back to MATLAB one-to-one.

Terminology note for thesis writing:
MATLAB code historically called b(t)=||[x(t),y(t),z(t)]||_2 "合成幅值"。
In the thesis, rename it to "矢量模值/模值序列" to avoid confusion with "synthetic samples".
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def feature_names_allreal_v4(fs: int = 50) -> list[str]:
    """Return feature names in the exact order of MATLAB ch3_feature_names_allreal_v4.m."""
    names: list[str] = []
    # length & duration
    names.append("len_N")
    names.append(f"len_T_sec_fs{fs}")

    sig_names = ["x", "y", "z", "b"]  # b is vector magnitude
    stat_names = ["mean", "std", "max", "min", "p2p", "energy", "rms", "absarea"]
    for s in sig_names:
        for st in stat_names:
            names.append(f"{s}_{st}")

    # derived features (same order as MATLAB)
    names += [
        "b_energy_density",
        "z_energy_frac",
        "p2p_z_over_b",
        "db_maxabs",
        "db_meanabs",
        "db_std",
        "zc_x",
        "zc_y",
        "zc_z",
        "b_num_peaks",
        "b_peak_loc_ratio",
        "z_sum",
        "b_max_over_mean",
        "b_std_over_mean",
    ]
    assert len(names) == 48, f"Feature name count mismatch: {len(names)} != 48"
    return names


def zero_crossings(a: np.ndarray) -> float:
    """Count sign changes excluding zeros (MATLAB ch3_zero_crossings)."""
    a = np.asarray(a).reshape(-1)
    if a.size < 2:
        return 0.0
    s = np.sign(a).astype(np.int8)
    # MATLAB uses: s(s==0)=1
    s[s == 0] = 1
    return float(np.sum((s[:-1] * s[1:]) < 0))


def peak_count_and_loc_ratio(b: np.ndarray) -> tuple[float, float]:
    """Peak count and main peak location ratio (MATLAB ch3_peak_count_and_loc)."""
    b = np.asarray(b).reshape(-1)
    T = int(b.size)
    if T == 0:
        return 0.0, 0.0
    p = int(np.argmax(b)) + 1  # MATLAB 1-index
    loc_ratio = float(p / T)
    if T < 3:
        return 0.0, loc_ratio
    thr = float(np.mean(b) + 0.5 * np.std(b, ddof=0))
    mid = b[1:-1]
    num = int(np.sum((mid > b[:-2]) & (mid > b[2:]) & (mid > thr)))
    return float(num), float(loc_ratio)


def _basic_stats(v: np.ndarray) -> tuple[float, float, float, float, float, float, float, float]:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return (0.0,) * 8
    amax = float(np.max(v))
    amin = float(np.min(v))
    p2p = amax - amin
    energy = float(np.sum(v * v))
    rms = float(np.sqrt(np.mean(v * v)))
    absarea = float(np.sum(np.abs(v)))
    return (
        float(np.mean(v)),
        float(np.std(v, ddof=0)),
        amax,
        amin,
        float(p2p),
        energy,
        float(rms),
        float(absarea),
    )


def extract_features_allreal_v4(xyz: np.ndarray, fs: int = 50) -> np.ndarray:
    """Compute the 48-dim feature vector for one event.

    Args:
        xyz: array (T,3) with axes x,y,z.
        fs: sampling frequency (Hz), default 50.

    Returns:
        feats: (48,) float64
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (T,3), got {xyz.shape}")

    T = int(xyz.shape[0])
    x = xyz[:, 0].astype(np.float64)
    y = xyz[:, 1].astype(np.float64)
    z = xyz[:, 2].astype(np.float64)
    b = np.sqrt(x * x + y * y + z * z)

    feats: list[float] = []
    # length & duration
    feats.append(float(T))
    feats.append(float(T / fs))

    fx = _basic_stats(x)
    fy = _basic_stats(y)
    fz = _basic_stats(z)
    fb = _basic_stats(b)
    feats += list(fx + fy + fz + fb)

    # derived
    e_x = fx[5]
    e_y = fy[5]
    e_z = fz[5]
    e_b = fb[5]
    p2p_z = fz[4]
    p2p_b = fb[4]

    b_energy_density = e_b / (T + EPS)
    z_energy_frac = e_z / (e_x + e_y + e_z + EPS)
    p2p_z_over_b = p2p_z / (p2p_b + EPS)

    if T >= 2:
        db = np.diff(b)
        db_maxabs = float(np.max(np.abs(db)))
        db_meanabs = float(np.mean(np.abs(db)))
        db_std = float(np.std(db, ddof=0))
    else:
        db_maxabs = db_meanabs = db_std = 0.0

    zc_x = zero_crossings(x)
    zc_y = zero_crossings(y)
    zc_z = zero_crossings(z)

    b_num_peaks, b_peak_loc_ratio = peak_count_and_loc_ratio(b)

    z_sum = float(np.sum(z))
    b_mean = fb[0]
    b_std = fb[1]
    b_max = fb[2]
    b_max_over_mean = float(b_max / (b_mean + EPS))
    b_std_over_mean = float(b_std / (b_mean + EPS))

    feats += [
        float(b_energy_density),
        float(z_energy_frac),
        float(p2p_z_over_b),
        float(db_maxabs),
        float(db_meanabs),
        float(db_std),
        float(zc_x),
        float(zc_y),
        float(zc_z),
        float(b_num_peaks),
        float(b_peak_loc_ratio),
        float(z_sum),
        float(b_max_over_mean),
        float(b_std_over_mean),
    ]

    feats_arr = np.asarray(feats, dtype=np.float64)
    if feats_arr.size != 48:
        raise RuntimeError(f"Feature dim mismatch: {feats_arr.size} != 48")
    return feats_arr
