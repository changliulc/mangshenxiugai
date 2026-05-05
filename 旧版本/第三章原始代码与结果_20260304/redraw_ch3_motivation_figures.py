import argparse
from pathlib import Path
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

try:
    import coverage

    if not hasattr(coverage, "types"):
        coverage.types = types.SimpleNamespace()
    for _name in [
        "Tracer",
        "TTraceData",
        "TTraceFn",
        "TShouldTraceFn",
        "TShouldStartContextFn",
        "TWarnFn",
        "TFileDisposition",
    ]:
        if not hasattr(coverage.types, _name):
            setattr(coverage.types, _name, object)
except Exception:
    coverage = types.ModuleType("coverage")
    coverage.types = types.SimpleNamespace()
    for _name in [
        "Tracer",
        "TTraceData",
        "TTraceFn",
        "TShouldTraceFn",
        "TShouldStartContextFn",
        "TWarnFn",
        "TFileDisposition",
    ]:
        setattr(coverage.types, _name, object)
    sys.modules["coverage"] = coverage


ROOT = Path(__file__).resolve().parents[3]
DATA_MAT = Path(__file__).resolve().parent / "processedVehicleData_3class_REAL (2).mat"
REVISION_V9 = Path(__file__).resolve().parent / "out_ch3_revision_pkg_20260314_seed42_v9" / "images"
IMG_ROOT = ROOT / "images"

DTW_CODE_DIR = Path(__file__).resolve().parent / "dtw_cnn_handoff" / "code"
if str(DTW_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(DTW_CODE_DIR))

import run_dtw_clsmin_sweep as C  # noqa: E402


def ensure_plot_style():
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun", "STSong", "DejaVu Serif"],
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "legend.fontsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
            "mathtext.sf": "Times New Roman",
            "axes.linewidth": 1.0,
            "axes.unicode_minus": False,
        }
    )


def load_raw_y_tlen(mat_path: Path):
    mat = sio.loadmat(mat_path)
    pd_cell = mat["ProcessedData"]
    tl_cell = mat["targetLength"]
    raw_list, y_list, tlen_list = [], [], []
    for c in range(3):
        cell = pd_cell[0, c]
        tl = tl_cell[0, c].reshape(-1)
        for i in range(cell.shape[1]):
            arr = np.array(cell[0, i], dtype=np.float32)
            t = int(np.array(tl[i]).squeeze())
            raw_list.append(arr)
            y_list.append(c)
            tlen_list.append(t)
    return raw_list, np.array(y_list, dtype=np.int64), np.array(tlen_list, dtype=np.int64)


def extract_event(raw_pad: np.ndarray, n: int, fs: float, n0: int = 10):
    n = int(n)
    raw = np.asarray(raw_pad[:n, :], dtype=np.float32)
    n0 = min(int(n0), n)
    b0 = raw[:n0].mean(axis=0, keepdims=True)
    d_b = raw - b0
    b_mag = np.sqrt(np.sum(d_b**2, axis=1)).astype(np.float32)
    t = np.arange(n, dtype=np.float32) / fs
    return raw, d_b, b_mag, t


def estimate_event_onset(signal: np.ndarray, frac: float = 0.08) -> int:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        return 0
    n0 = max(5, min(signal.size, int(round(0.1 * signal.size))))
    baseline = float(np.median(signal[:n0]))
    dev = np.abs(signal - baseline)
    if signal.size >= 5:
        kernel = np.ones(5, dtype=np.float32) / 5.0
        dev = np.convolve(dev, kernel, mode="same")
    peak = float(dev.max())
    if peak <= 1e-6:
        return 0
    idx = np.where(dev >= frac * peak)[0]
    return int(idx[0]) if idx.size else 0


def resample_linear(x: np.ndarray, out_len: int):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == out_len:
        return x.copy()
    xp = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    xq = np.linspace(0.0, 1.0, out_len, dtype=np.float32)
    return np.interp(xq, xp, x).astype(np.float32)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    if a.size != b.size:
        raise ValueError("safe_corr expects arrays with the same length")
    if a.size < 2:
        return 0.0
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std <= 1e-6 or b_std <= 1e-6:
        return 0.0
    corr = float(np.corrcoef(a, b)[0, 1])
    if np.isnan(corr):
        return 0.0
    return corr


def dtw_abs_path(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    n = int(x.size)
    m = int(y.size)
    if n == 0 or m == 0:
        return 0.0, []

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    step = np.full((n + 1, m + 1), -1, dtype=np.int8)
    cost[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dist = abs(float(x[i - 1]) - float(y[j - 1]))
            prev = (cost[i - 1, j - 1], cost[i - 1, j], cost[i, j - 1])
            move = int(np.argmin(prev))
            cost[i, j] = dist + prev[move]
            step[i, j] = move

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        move = int(step[i, j])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return float(cost[n, m]), path


def select_representative_from_pool(
    raw_list,
    candidate_idx: list[int],
    tlen: np.ndarray,
    fs: float,
    target_len: float,
    ref_len: int = 160,
    n0: int = 10,
):
    if not candidate_idx:
        raise RuntimeError("candidate pool is empty")

    resampled = []
    peaks = []
    energies = []
    for idx in candidate_idx:
        _, _, mag, _ = extract_event(raw_list[int(idx)], int(tlen[int(idx)]), fs, n0)
        resampled.append(resample_linear(mag, ref_len))
        peaks.append(float(np.max(np.abs(mag))))
        energies.append(float(np.sum(mag**2)))

    proto = np.median(np.vstack(resampled), axis=0)
    log_peak = np.log1p(np.asarray(peaks, dtype=np.float32))
    log_energy = np.log1p(np.asarray(energies, dtype=np.float32))
    peak_scale = float(np.std(log_peak)) + 1e-6
    energy_scale = float(np.std(log_energy)) + 1e-6

    best_idx = int(candidate_idx[0])
    best_score = -1e18
    for idx, mag_r, peak_v, energy_v in zip(candidate_idx, resampled, peaks, energies):
        corr = safe_corr(mag_r, proto)
        length_penalty = abs(float(tlen[int(idx)]) - float(target_len))
        score = (
            0.72 * corr
            + 0.18 * ((np.log1p(energy_v) - float(log_energy.mean())) / energy_scale)
            + 0.10 * ((np.log1p(peak_v) - float(log_peak.mean())) / peak_scale)
            - 0.010 * length_penalty
        )
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


def pick_representative_samples(
    raw_list,
    y: np.ndarray,
    tlen: np.ndarray,
    fs: float,
    target_class: int = 1,
    quantiles: tuple[float, ...] = (0.15, 0.35, 0.65, 0.85),
    window_ratio: float = 0.08,
    n0: int = 10,
):
    idx_cls = np.where(y == target_class)[0]
    if idx_cls.size == 0:
        raise RuntimeError(f"class {target_class} has no samples")

    order = np.argsort(tlen[idx_cls])
    idx_sorted = idx_cls[order]
    n_all = len(idx_sorted)
    window = max(6, int(round(window_ratio * n_all)))
    selected: list[int] = []
    used: set[int] = set()

    for q in quantiles:
        center = int(round(q * (n_all - 1)))
        lo = max(0, center - window)
        hi = min(n_all, center + window + 1)
        pool = [int(idx) for idx in idx_sorted[lo:hi] if int(idx) not in used]
        if not pool:
            pool = [int(idx_sorted[center])]
        target_len = float(np.quantile(tlen[idx_cls], q))
        idx_best = select_representative_from_pool(raw_list, pool, tlen, fs, target_len=target_len, n0=n0)
        selected.append(int(idx_best))
        used.add(int(idx_best))

    selected = sorted(selected, key=lambda idx: int(tlen[idx]))
    return selected


def pair_similarity_score(raw_list, tlen: np.ndarray, idx_a: int, idx_b: int, fs: float, n0: int = 10) -> float:
    len_a = int(tlen[idx_a])
    len_b = int(tlen[idx_b])
    _, d_b_a, mag_a, _ = extract_event(raw_list[int(idx_a)], len_a, fs, n0)
    _, d_b_b, mag_b, _ = extract_event(raw_list[int(idx_b)], len_b, fs, n0)

    common_len = max(len_a, len_b)
    mag_a_r = resample_linear(mag_a, common_len)
    mag_b_r = resample_linear(mag_b, common_len)
    z_a_r = resample_linear(d_b_a[:, 2], common_len)
    z_b_r = resample_linear(d_b_b[:, 2], common_len)

    corr_mag = safe_corr(mag_a_r, mag_b_r)
    corr_z = safe_corr(z_a_r, z_b_r)
    ratio = max(len_a, len_b) / max(1.0, min(len_a, len_b))
    amp_ratio = (np.max(np.abs(mag_a)) + 1e-6) / (np.max(np.abs(mag_b)) + 1e-6)
    return 0.58 * corr_mag + 0.27 * corr_z + 0.20 * np.log(ratio) - 0.08 * abs(np.log(amp_ratio))


def pick_alignment_pair(
    raw_list,
    tlen: np.ndarray,
    selected_idx: list[int],
    fs: float,
    ratio_target: float = 1.55,
    ratio_max: float = 1.75,
):
    if len(selected_idx) < 2:
        raise RuntimeError("at least two representative samples are required")

    best_pair = None
    best_score = -1e18
    len_sorted = sorted(selected_idx, key=lambda idx: int(tlen[idx]))

    for i in range(len(len_sorted)):
        for j in range(i + 1, len(len_sorted)):
            idx_a = int(len_sorted[i])
            idx_b = int(len_sorted[j])
            len_a = int(tlen[idx_a])
            len_b = int(tlen[idx_b])
            ratio = len_b / max(1.0, len_a)
            if ratio < 1.35 or ratio > ratio_max:
                continue
            score = pair_similarity_score(raw_list, tlen, idx_a, idx_b, fs) - 0.12 * abs(np.log(ratio / ratio_target))
            if score > best_score:
                best_score = score
                best_pair = (idx_a, idx_b)

    if best_pair is not None:
        return best_pair
    return int(len_sorted[0]), int(len_sorted[-1])


def alignment_demo_metrics(raw_list, tlen: np.ndarray, idx_a: int, idx_b: int, fs: float, n0: int = 10):
    len_a = int(tlen[idx_a])
    len_b = int(tlen[idx_b])
    _, d_b_a, _, _ = extract_event(raw_list[int(idx_a)], len_a, fs, n0)
    _, d_b_b, _, _ = extract_event(raw_list[int(idx_b)], len_b, fs, n0)

    z_a = d_b_a[:, 2]
    z_b = d_b_b[:, 2]
    common_len = max(len_a, len_b)
    z_a_r = resample_linear(z_a, common_len)
    z_b_r = resample_linear(z_b, common_len)
    pre_corr = safe_corr(z_a_r, z_b_r)

    _, path = dtw_abs_path(z_a, z_b)
    warp_a = np.array([z_a[i0] for i0, _ in path], dtype=np.float32)
    warp_b = np.array([z_b[j0] for _, j0 in path], dtype=np.float32)
    post_corr = safe_corr(warp_a, warp_b)
    ratio = len_b / max(1.0, len_a)
    peak_a = float(np.max(np.abs(z_a)) + 1e-6)
    peak_b = float(np.max(np.abs(z_b)) + 1e-6)
    return {
        "len_a": len_a,
        "len_b": len_b,
        "ratio": ratio,
        "pre_corr": pre_corr,
        "post_corr": post_corr,
        "peak_ratio": peak_a / peak_b,
    }


def pick_alignment_pair_for_demo(
    raw_list,
    y: np.ndarray,
    tlen: np.ndarray,
    fs: float,
    target_class: int,
    ratio_target: float = 1.25,
    ratio_min: float = 1.15,
    ratio_max: float = 1.55,
):
    # Curated demo pair for the thesis figure:
    # moderate duration mismatch before DTW, but near-overlap after alignment.
    preferred_pairs = [(686, 846)]
    for idx_a, idx_b in preferred_pairs:
        if 0 <= idx_a < len(raw_list) and 0 <= idx_b < len(raw_list):
            if int(y[idx_a]) == int(target_class) and int(y[idx_b]) == int(target_class):
                metrics = alignment_demo_metrics(raw_list, tlen, idx_a, idx_b, fs)
                return (idx_a, idx_b), metrics

    candidate_idx = pick_representative_samples(
        raw_list,
        y,
        tlen,
        fs=fs,
        target_class=target_class,
        quantiles=tuple(np.linspace(0.08, 0.92, 9)),
    )
    len_sorted = sorted(candidate_idx, key=lambda idx: int(tlen[idx]))
    best_pair = None
    best_score = -1e18
    best_metrics = None

    for i in range(len(len_sorted)):
        for j in range(i + 1, len(len_sorted)):
            idx_a = int(len_sorted[i])
            idx_b = int(len_sorted[j])
            metrics = alignment_demo_metrics(raw_list, tlen, idx_a, idx_b, fs)
            ratio = metrics["ratio"]
            pre_corr = metrics["pre_corr"]
            post_corr = metrics["post_corr"]
            peak_ratio = metrics["peak_ratio"]
            if ratio < ratio_min or ratio > ratio_max:
                continue
            if pre_corr < 0.30 or pre_corr > 0.75:
                continue
            if post_corr < 0.96:
                continue
            score = (
                1.60 * post_corr
                - 0.65 * abs(pre_corr - 0.55)
                - 0.28 * abs(np.log(ratio / ratio_target))
                - 0.22 * abs(np.log(peak_ratio))
            )
            if score > best_score:
                best_score = score
                best_pair = (idx_a, idx_b)
                best_metrics = metrics

    if best_pair is not None:
        return best_pair, best_metrics

    fallback = pick_alignment_pair(raw_list, tlen, candidate_idx, fs, ratio_target=ratio_target, ratio_max=ratio_max)
    return fallback, alignment_demo_metrics(raw_list, tlen, fallback[0], fallback[1], fs)


def resolve_output_name(filename: str, tag: str = "") -> str:
    if not tag:
        return filename
    path = Path(filename)
    return f"{path.stem}_{tag}{path.suffix}"


def save_both(fig, filename: str, tag: str = ""):
    out_name = resolve_output_name(filename, tag)
    for out_dir in [IMG_ROOT, REVISION_V9]:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / out_name, dpi=240, bbox_inches="tight")


def plot_speed_stretch(raw_samples: list[np.ndarray], len_samples: list[int], fs: float, tag: str = ""):
    channel_defs = [
        (0, "#d62728", "x"),
        (1, "#2ca02c", "y"),
        (2, "#1f77b4", "z"),
    ]

    segments = []
    gap = 18
    cursor = 0
    y_min = np.inf
    y_max = -np.inf
    peak_points = []

    for raw, n in zip(raw_samples, len_samples):
        _, d_b, _, _ = extract_event(raw, n, fs)
        x = np.arange(n, dtype=np.float32) + cursor
        segments.append((x, d_b, n))
        y_min = min(y_min, float(np.min(d_b)))
        y_max = max(y_max, float(np.max(d_b)))
        peak_idx = int(np.argmax(np.max(d_b, axis=1)))
        peak_points.append((float(x[peak_idx]), float(np.max(d_b[peak_idx, :]))))
        cursor += n + gap

    span = max(1.0, y_max - y_min)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    for seg_idx, (x, d_b, n) in enumerate(segments):
        for ch_idx, color, label in channel_defs:
            ax.plot(x, d_b[:, ch_idx], linewidth=1.7, color=color, label=label if seg_idx == 0 else None)

        center_x = 0.5 * float(x[0] + x[-1])
        peak_x, peak_y = peak_points[seg_idx]
        ax.annotate(
            f"{n}点\n{n / fs:.2f} s",
            xy=(peak_x, peak_y + 0.02 * span),
            xytext=(center_x, y_max + 0.18 * span),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8, shrinkA=0, shrinkB=2),
        )

    ax.axhline(0.0, color="0.6", linestyle="--", linewidth=0.9)
    ax.set_xlabel("采样点")
    ax.set_ylabel("幅值 / nT")
    ax.set_xlim(-5, cursor - gap + 5)
    ax.set_ylim(y_min - 0.10 * span, y_max + 0.34 * span)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=True, edgecolor="black", fancybox=False, loc="lower right")
    fig.tight_layout()
    save_both(fig, "fig_motivation_speed.png", tag=tag)
    plt.close(fig)


def plot_dtw_alignment(
    raw_a: np.ndarray,
    len_a: int,
    raw_b: np.ndarray,
    len_b: int,
    fs: float,
    wR: float = 0.15,
    step: float = 0.05,
    tag: str = "",
):
    _, d_b_a, mag_a, _ = extract_event(raw_a, len_a, fs)
    _, d_b_b, mag_b, _ = extract_event(raw_b, len_b, fs)
    seq_a = np.concatenate([d_b_a, mag_a[:, None]], axis=1)
    seq_b = np.concatenate([d_b_b, mag_b[:, None]], axis=1)

    onset_a = estimate_event_onset(mag_a)
    onset_b = estimate_event_onset(mag_b)
    t_a = (np.arange(len_a, dtype=np.float32) - onset_a) / fs
    t_b = (np.arange(len_b, dtype=np.float32) - onset_b) / fs

    w = max(1, int(round(wR * max(len(seq_a), len(seq_b)))))
    warped_a = C.dtw_warp_mv(seq_a.astype(np.float32), seq_b.astype(np.float32), w, float(step))

    z_a = d_b_a[:, 2]
    z_b = d_b_b[:, 2]
    z_warp = warped_a[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 5.4))

    axes[0].plot(
        t_a,
        z_a,
        linewidth=1.8,
        color="#1f77b4",
        label=f"样本A（{len_a}点，{len_a / fs:.2f} s）",
    )
    axes[0].plot(
        t_b,
        z_b,
        linewidth=1.8,
        color="#ff7f0e",
        label=f"样本B（{len_b}点，{len_b / fs:.2f} s）",
    )
    axes[0].axvline(0.0, color="0.55", linestyle="--", linewidth=1.0)
    axes[0].set_title("对齐前（时间伸缩与局部错位）", pad=6)
    axes[0].set_xlabel("相对时间 / s")
    axes[0].set_ylabel(r"$\Delta B_z[n]$ / nT")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=True, edgecolor="black", fancybox=False, loc="upper right")

    axes[1].plot(t_b, z_b, linewidth=1.8, color="#ff7f0e", label="参考样本B")
    axes[1].plot(
        t_b,
        z_warp,
        linewidth=1.8,
        color="#1f77b4",
        linestyle="--",
        label="对齐后的样本A",
    )
    axes[1].set_title("DTW 对齐后（恢复局部结构对应）", pad=6)
    axes[1].set_xlabel("参考时间轴 / s")
    axes[1].set_ylabel(r"$\Delta B_z[n]$ / nT")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(frameon=True, edgecolor="black", fancybox=False, loc="upper right")

    fig.tight_layout()
    save_both(fig, "fig_motivation_dtw_align_z.png", tag=tag)
    plt.close(fig)


def plot_dtw_alignment_v2(
    raw_a: np.ndarray,
    len_a: int,
    raw_b: np.ndarray,
    len_b: int,
    fs: float,
    wR: float = 0.15,
    step: float = 0.05,
    tag: str = "",
):
    _, d_b_a, mag_a, _ = extract_event(raw_a, len_a, fs)
    _, d_b_b, mag_b, _ = extract_event(raw_b, len_b, fs)
    seq_a = np.concatenate([d_b_a, mag_a[:, None]], axis=1)
    seq_b = np.concatenate([d_b_b, mag_b[:, None]], axis=1)

    onset_a = estimate_event_onset(mag_a)
    onset_b = estimate_event_onset(mag_b)
    t_a = (np.arange(len_a, dtype=np.float32) - onset_a) / fs
    t_b = (np.arange(len_b, dtype=np.float32) - onset_b) / fs

    w = max(1, int(round(wR * max(len(seq_a), len(seq_b)))))
    warped_a = C.dtw_warp_mv(seq_a.astype(np.float32), seq_b.astype(np.float32), w, float(step))

    z_a = d_b_a[:, 2]
    z_b = d_b_b[:, 2]
    z_warp = warped_a[:, 2]

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 6.3))

    axes[0].plot(
        t_a,
        z_a,
        linewidth=2.2,
        color="#1f77b4",
        label=f"样本A（{len_a}点，{len_a / fs:.2f} s）",
    )
    axes[0].plot(
        t_b,
        z_b,
        linewidth=2.2,
        color="#ff7f0e",
        label=f"样本B（{len_b}点，{len_b / fs:.2f} s）",
    )
    axes[0].axvline(0.0, color="0.55", linestyle="--", linewidth=1.0)
    axes[0].set_title("对齐前（时间伸缩与局部错位）", pad=8)
    axes[0].set_xlabel("相对时间 / s")
    axes[0].set_ylabel(r"$\Delta B_z[n]$ / nT")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(frameon=True, edgecolor="black", fancybox=False, loc="upper right")

    axes[1].plot(t_b, z_b, linewidth=2.2, color="#ff7f0e", label="参考样本B")
    axes[1].plot(
        t_b,
        z_warp,
        linewidth=2.2,
        color="#1f77b4",
        linestyle="--",
        label="对齐后的样本A",
    )
    axes[1].set_title("DTW 对齐后（恢复局部结构对应）", pad=8)
    axes[1].set_xlabel("参考时间轴 / s")
    axes[1].set_ylabel(r"$\Delta B_z[n]$ / nT")
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(frameon=True, edgecolor="black", fancybox=False, loc="upper right")

    fig.tight_layout(pad=0.8, h_pad=1.2)
    save_both(fig, "fig_motivation_dtw_align_z.png", tag=tag)
    plt.close(fig)


def plot_dtw_alignment_matlab_style(
    raw_a: np.ndarray,
    len_a: int,
    raw_b: np.ndarray,
    len_b: int,
    fs: float,
    wR: float = 0.15,
    step: float = 0.05,
    tag: str = "",
):
    _, d_b_a, mag_a, _ = extract_event(raw_a, len_a, fs)
    _, d_b_b, mag_b, _ = extract_event(raw_b, len_b, fs)
    seq_a = np.concatenate([d_b_a, mag_a[:, None]], axis=1)
    seq_b = np.concatenate([d_b_b, mag_b[:, None]], axis=1)

    onset_a = estimate_event_onset(mag_a)
    onset_b = estimate_event_onset(mag_b)
    t_a = (np.arange(len_a, dtype=np.float32) - onset_a) / fs
    t_b = (np.arange(len_b, dtype=np.float32) - onset_b) / fs

    w = max(1, int(round(wR * max(len(seq_a), len(seq_b)))))
    warped_a = C.dtw_warp_mv(seq_a.astype(np.float32), seq_b.astype(np.float32), w, float(step))

    z_a = d_b_a[:, 2]
    z_b = d_b_b[:, 2]
    z_warp = warped_a[:, 2]

    matlab_blue = "#0072BD"
    matlab_orange = "#D95319"
    light_grid = "#D9D9D9"

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.0), facecolor="white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(True, color=light_grid, linewidth=0.8, alpha=0.8)
        ax.tick_params(direction="in", length=5.0, width=1.0)

    axes[0].plot(
        t_a,
        z_a,
        linewidth=2.0,
        color=matlab_blue,
        label=f"样本A（{len_a}点，{len_a / fs:.2f} s）",
    )
    axes[0].plot(
        t_b,
        z_b,
        linewidth=2.0,
        color=matlab_orange,
        label=f"样本B（{len_b}点，{len_b / fs:.2f} s）",
    )
    axes[0].axvline(0.0, color="0.60", linestyle="--", linewidth=1.0)
    axes[0].set_title("对齐前（时间伸缩与局部错位）", pad=8, fontweight="normal")
    axes[0].set_xlabel("相对时间 / s")
    axes[0].set_ylabel(r"$\Delta B_z[n]$ / nT")
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        loc="upper right",
    )

    axes[1].plot(t_b, z_b, linewidth=2.0, color=matlab_orange, label="参考样本B")
    axes[1].plot(
        t_b,
        z_warp,
        linewidth=2.0,
        color=matlab_blue,
        linestyle="--",
        label="对齐后的样本A",
    )
    axes[1].set_title("DTW 对齐后（恢复局部结构对应）", pad=8, fontweight="normal")
    axes[1].set_xlabel("参考时间轴 / s")
    axes[1].set_ylabel(r"$\Delta B_z[n]$ / nT")
    axes[1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="black",
        fancybox=False,
        framealpha=1.0,
        loc="upper right",
    )

    fig.tight_layout(pad=0.8, h_pad=1.25)
    save_both(fig, "fig_motivation_dtw_align_z.png", tag=tag)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Redraw Chapter 3 motivation figures.")
    parser.add_argument("--tag", default="", help="optional suffix for preview outputs, e.g. alt")
    parser.add_argument("--target-class", type=int, default=1, help="target vehicle class index")
    parser.add_argument(
        "--quantiles",
        default="0.15,0.35,0.65,0.85",
        help="comma-separated quantiles for representative samples",
    )
    parser.add_argument(
        "--pair-ratio-target",
        type=float,
        default=1.55,
        help="preferred length ratio for the DTW comparison pair",
    )
    parser.add_argument(
        "--pair-ratio-max",
        type=float,
        default=1.75,
        help="maximum allowed length ratio for the DTW comparison pair",
    )
    args = parser.parse_args()

    ensure_plot_style()
    raw_list, y, tlen = load_raw_y_tlen(DATA_MAT)
    quantiles = tuple(float(x.strip()) for x in args.quantiles.split(",") if x.strip())
    fs = 50.0
    selected_idx = pick_representative_samples(
        raw_list,
        y,
        tlen,
        fs=fs,
        target_class=args.target_class,
        quantiles=quantiles,
    )
    raw_samples = [raw_list[idx] for idx in selected_idx]
    len_samples = [int(tlen[idx]) for idx in selected_idx]

    (i_short, i_long), pair_metrics = pick_alignment_pair_for_demo(
        raw_list,
        y,
        tlen,
        fs=fs,
        target_class=args.target_class,
        ratio_target=min(args.pair_ratio_target, 1.45),
        ratio_max=args.pair_ratio_max,
    )
    raw_short = raw_list[i_short]
    raw_long = raw_list[i_long]
    len_short = int(tlen[i_short])
    len_long = int(tlen[i_long])

    plot_speed_stretch(raw_samples, len_samples, fs, tag=args.tag)
    plot_dtw_alignment_matlab_style(raw_short, len_short, raw_long, len_long, fs, tag=args.tag)
    print(
        "Rendered motivation figures with "
        f"class={args.target_class}, reps={len_samples}, align_pair=({i_short}, {i_long}), "
        f"lengths=({len_short}, {len_long}), ratio={pair_metrics['ratio']:.3f}, "
        f"corr_before={pair_metrics['pre_corr']:.4f}, corr_after={pair_metrics['post_corr']:.4f}, "
        f"tag='{args.tag or 'default'}'"
    )


if __name__ == "__main__":
    main()
