# -*- coding: utf-8 -*-
"""Run Chapter-3 template strategy ablations.

The goal is to keep the classifier and training protocol fixed while changing
only the DTW/template representation:

- linear fixed-length CNN baseline
- DTW-ClsMin with mean templates
- DTW-ClsMin with DBA templates
- DTW multi-template with duration-quantile representative templates
- random template pool control with the same template count as the final method

The DTW builders are imported from the original Chapter-3 code to preserve the
thesis method口径; training and metrics use the current supplementary common
pipeline.
"""

from __future__ import annotations

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np

import ch3_common as C
from models import build_model, count_parameters


OLD_CH3_DIR = Path(
    r"D:\xidian_Master\研究生论文\毕业论文\xduts-main\非核心文件\绘图\第三章\20260304"
)
OLD_DTW_CODE_DIR = OLD_CH3_DIR / "dtw_cnn_handoff" / "code"
for _path in (OLD_CH3_DIR, OLD_DTW_CODE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Some legacy DTW scripts patch numba through the optional coverage package.
# The experiments do not use coverage itself, so provide a tiny local stub when
# the package is absent.
if "coverage" not in sys.modules:
    coverage_stub = types.ModuleType("coverage")
    coverage_stub.types = types.SimpleNamespace()
    sys.modules["coverage"] = coverage_stub

import run_cnn_dtw_experiment as DTW_MEAN  # noqa: E402
import run_dtw_clsmin_dba as DTW_DBA  # noqa: E402
import run_dtw_multi_quantile as DTW_QUANTILE  # noqa: E402
from run_dtw_multi_sweep import dtw_warp_mv  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 3 DTW template ablations.")
    parser.add_argument("--mat", default=C.MAIN_DATA_DEFAULT, help="MAT file with ProcessedData and targetLength.")
    parser.add_argument("--out", default="outputs/template_ablation", help="Output directory.")
    parser.add_argument("--preset", choices=sorted(C.TRAIN_PRESETS), default="paper_fair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=176)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--wR", type=float, default=0.10)
    parser.add_argument("--step", type=float, default=0.0)
    parser.add_argument("--dba_iter", type=int, default=10)
    parser.add_argument("--K_list", default="1,2,3,4")
    parser.add_argument("--random_K", type=int, default=2)
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Enable train-set minority oversampling. Default follows the thesis pipeline: shuffle only.",
    )
    return parser.parse_args()


def build_random_template_pool(
    X_raw: np.ndarray,
    idx_tr: np.ndarray,
    total_templates: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    replace = int(total_templates) > int(len(idx_tr))
    picks = rng.choice(idx_tr, size=int(total_templates), replace=replace)
    return X_raw[picks].astype(np.float32, copy=True)


def build_dtw_template_pool(
    X_raw: np.ndarray,
    templates: np.ndarray,
    wR: float,
    step: float,
) -> np.ndarray:
    X_raw = np.asarray(X_raw, dtype=np.float32)
    templates = np.asarray(templates, dtype=np.float32)
    N, ch, length = X_raw.shape
    n_templates = int(templates.shape[0])
    window = max(1, int(round(float(wR) * float(length))))
    _ = dtw_warp_mv(X_raw[0].T, templates[0].T, window, float(step))

    X_out = np.zeros((N, ch * n_templates, length), dtype=np.float32)
    for i in range(N):
        parts = []
        Xi = X_raw[i].T
        for template in templates:
            aligned = dtw_warp_mv(Xi, template.T, window, float(step))
            parts.append(aligned.T)
        X_out[i] = np.concatenate(parts, axis=0)
    return X_out


def build_linear_template_residual_pool(
    X_raw: np.ndarray,
    templates: np.ndarray,
) -> np.ndarray:
    """Build a multi-template control representation without DTW alignment.

    It uses the same template pool and channel count as DTW multi-template, but
    each block is the point-wise residual between the fixed-length sample and a
    fixed-length template. This keeps the multi-template view while removing the
    nonlinear DTW warping step.
    """

    X_raw = np.asarray(X_raw, dtype=np.float32)
    templates = np.asarray(templates, dtype=np.float32)
    if templates.ndim == 4:
        n_templates = int(templates.shape[0] * templates.shape[1])
        templates_flat = templates.reshape(n_templates, templates.shape[2], templates.shape[3])
    elif templates.ndim == 3:
        templates_flat = templates
        n_templates = int(templates.shape[0])
    else:
        raise ValueError(f"Expected templates with ndim 3 or 4, got shape {templates.shape}")

    N, ch, length = X_raw.shape
    if templates_flat.shape[1:] != (ch, length):
        raise ValueError(
            f"Template shape {templates_flat.shape[1:]} does not match sample shape {(ch, length)}"
        )

    X_out = np.zeros((N, ch * n_templates, length), dtype=np.float32)
    col = 0
    for template in templates_flat:
        X_out[:, col : col + ch, :] = X_raw - template[None, :, :]
        col += ch
    return X_out


def train_representation(
    method_name: str,
    X_rep_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    args: argparse.Namespace,
    build_seconds: float,
) -> tuple[dict, list[dict], list[dict], dict]:
    mu, sd = C.compute_norm_stats(X_rep_raw, idx_tr)
    X_rep = C.normalize_with_stats(X_rep_raw, mu, sd)

    C.set_seed(args.seed)
    model = build_model("fixed_cnn_1d", in_channels=int(X_rep.shape[1]), n_classes=3)
    param_count = count_parameters(model)
    started = time.time()
    result = C.train_model(
        model=model,
        X=X_rep,
        y=y,
        idx_tr=idx_tr,
        idx_va=idx_va,
        idx_te=idx_te,
        train_config=C.TRAIN_PRESETS[args.preset],
        seed=args.seed,
        batch_size=args.batch_size,
        oversample=bool(args.oversample),
        device_name=args.device,
    )
    train_seconds = time.time() - started

    metric = C.metric_row(
        experiment="template_ablation",
        validation_type="stratified_random",
        preset=args.preset,
        oversample=bool(args.oversample),
        run_id=0,
        seed=args.seed,
        model_name=method_name,
        param_count=param_count,
        result=result,
    )

    per_class_rows: list[dict] = []
    for row in result["test"]["per_class"]:
        row_out = dict(row)
        row_out.update(
            {
                "experiment": "template_ablation",
                "validation_type": "stratified_random",
                "run_id": 0,
                "seed": args.seed,
                "model": method_name,
            }
        )
        per_class_rows.append(row_out)

    confusion_rows: list[dict] = []
    cm = np.asarray(result["test"]["confusion_matrix"], dtype=int)
    for true_class in range(cm.shape[0]):
        confusion_rows.append(
            {
                "experiment": "template_ablation",
                "validation_type": "stratified_random",
                "run_id": 0,
                "seed": args.seed,
                "model": method_name,
                "true_class": true_class,
                "pred_small": int(cm[true_class, 0]),
                "pred_medium": int(cm[true_class, 1]),
                "pred_large": int(cm[true_class, 2]),
            }
        )

    detail = {
        "method": method_name,
        "in_channels": int(X_rep.shape[1]),
        "param_count": int(param_count),
        "build_seconds": float(build_seconds),
        "train_seconds": float(train_seconds),
        "result": result,
    }
    return metric, per_class_rows, confusion_rows, detail


def main() -> None:
    args = parse_args()
    out_dir = C.ensure_dir(args.out)
    dataset = C.load_dataset(args.mat, require_source=False)
    idx_tr, idx_va, idx_te = C.stratified_split(
        dataset.y,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    np.savez(out_dir / f"split_indices_seed{args.seed}.npz", idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te)

    X_raw = C.build_sequence_tensor(dataset.xyz_list, length=args.L)
    y = dataset.y
    tlen = dataset.target_length
    k_list = [int(item) for item in args.K_list.split(",") if item.strip()]

    all_metric_rows: list[dict] = []
    all_per_class_rows: list[dict] = []
    all_confusion_rows: list[dict] = []
    details: list[dict] = []

    representations: list[tuple[str, np.ndarray, float]] = []
    representations.append(("linear_fixed_cnn", X_raw.copy(), 0.0))

    started = time.time()
    X_mean = DTW_MEAN.build_dtw_clsmin(X_raw.copy(), y, idx_tr, wR=args.wR, step=args.step, seed=args.seed)
    representations.append(("dtw_clsmin_mean", X_mean, time.time() - started))

    started = time.time()
    X_dba = DTW_DBA.build_dtw_clsmin_dba(
        X_raw.copy(),
        y,
        idx_tr,
        wR=args.wR,
        step=args.step,
        n_iter=args.dba_iter,
    )
    representations.append(("dtw_clsmin_dba", X_dba, time.time() - started))

    for K in k_list:
        started = time.time()
        X_multi, _templates = DTW_QUANTILE.build_dtw_multi_quantile(
            X_raw.copy(),
            y,
            idx_tr,
            tlen,
            templates_per_class=K,
            wR=args.wR,
            step=args.step,
        )
        representations.append((f"dtw_multi_quantile_K{K}", X_multi, time.time() - started))

        started = time.time()
        X_linear_multi = build_linear_template_residual_pool(X_raw.copy(), _templates)
        representations.append((f"linear_multi_residual_K{K}", X_linear_multi, time.time() - started))

    total_random_templates = 3 * int(args.random_K)
    started = time.time()
    random_templates = build_random_template_pool(
        X_raw.copy(),
        idx_tr,
        total_templates=total_random_templates,
        seed=args.seed + 2026,
    )
    X_random = build_dtw_template_pool(X_raw.copy(), random_templates, wR=args.wR, step=args.step)
    representations.append((f"dtw_random_pool_K{args.random_K}", X_random, time.time() - started))

    for method_name, X_rep, build_seconds in representations:
        metric, per_class, confusion, detail = train_representation(
            method_name=method_name,
            X_rep_raw=X_rep,
            y=y,
            idx_tr=idx_tr,
            idx_va=idx_va,
            idx_te=idx_te,
            args=args,
            build_seconds=build_seconds,
        )
        all_metric_rows.append(metric)
        all_per_class_rows.extend(per_class)
        all_confusion_rows.extend(confusion)
        details.append(detail)

    summary_rows = C.aggregate_metric_rows(all_metric_rows)
    C.write_csv(out_dir / "metrics.csv", all_metric_rows, C.METRIC_FIELDS)
    C.write_csv(out_dir / "summary.csv", summary_rows, C.SUMMARY_FIELDS)
    C.write_csv(
        out_dir / "per_class_metrics.csv",
        all_per_class_rows,
        [
            "experiment",
            "validation_type",
            "run_id",
            "seed",
            "model",
            "class_id",
            "class_name",
            "precision",
            "recall",
            "f1",
            "support",
        ],
    )
    C.write_csv(
        out_dir / "confusion_matrices.csv",
        all_confusion_rows,
        [
            "experiment",
            "validation_type",
            "run_id",
            "seed",
            "model",
            "true_class",
            "pred_small",
            "pred_medium",
            "pred_large",
        ],
    )
    C.write_json(
        out_dir / "results.json",
        {
            "experiment": "template_ablation",
            "validation_type": "stratified_random",
            "mat_path": str(args.mat),
            "L": int(args.L),
            "preset": args.preset,
            "train_preset": C.TRAIN_PRESETS[args.preset],
            "oversample": bool(args.oversample),
            "batch_size": int(args.batch_size),
            "dtw": {
                "wR": float(args.wR),
                "step": float(args.step),
                "dba_iter": int(args.dba_iter),
                "K_list": k_list,
                "random_K": int(args.random_K),
            },
            "split_meta": {
                "seed": int(args.seed),
                "split_hash": C.split_hash(idx_tr, idx_va, idx_te),
                "split_counts": {
                    "train": C.split_counts(y, idx_tr),
                    "val": C.split_counts(y, idx_va),
                    "test": C.split_counts(y, idx_te),
                },
            },
            "metrics": all_metric_rows,
            "summary": summary_rows,
            "details": details,
        },
    )


if __name__ == "__main__":
    main()
