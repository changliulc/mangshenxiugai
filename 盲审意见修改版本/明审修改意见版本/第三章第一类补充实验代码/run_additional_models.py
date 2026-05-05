# -*- coding: utf-8 -*-
"""Run additional Chapter-3 model baselines.

This script extends the already completed fixed CNN/LSTM experiments with:

- fixed-length GRU
- fixed-length lightweight Transformer Encoder
- fixed-length ResNet1D
- masked CNN on padded variable-length event sequences
- masked LSTM on padded variable-length event sequences

All models use the same split, train-only normalization, optimizer preset,
early stopping rule, and metrics. Masked models are reported separately by model
name because their input is padded variable-length data rather than the
linearly resampled 176-point sequence.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import ch3_common as C
from models import build_model, count_parameters


FIXED_MODELS = {"gru_h64", "transformer_d64", "resnet1d"}
MASKED_PREFIXES = ("masked_",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Additional Chapter 3 model baselines.")
    parser.add_argument("--mat", default=C.MAIN_DATA_DEFAULT, help="MAT file with ProcessedData and targetLength.")
    parser.add_argument("--out", default="outputs/additional_models", help="Output directory.")
    parser.add_argument(
        "--models",
        default="gru_h64,transformer_d64,resnet1d,masked_cnn_1d,masked_lstm_h64",
        help="Comma-separated model names.",
    )
    parser.add_argument("--preset", choices=sorted(C.TRAIN_PRESETS), default="paper_fair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--L", type=int, default=176, help="Fixed sequence length.")
    parser.add_argument(
        "--masked_max_length",
        type=int,
        default=0,
        help="Optional max length for masked models. 0 means pad to dataset max length.",
    )
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Enable train-set minority oversampling. Default follows the thesis pipeline: shuffle only.",
    )
    return parser.parse_args()


def is_masked_model(model_name: str) -> bool:
    return model_name.startswith(MASKED_PREFIXES)


def input_kind(model_name: str) -> str:
    return "masked_variable" if is_masked_model(model_name) else "fixed_resampled"


def train_one_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    args: argparse.Namespace,
    run_id: int,
    run_seed: int,
) -> tuple[dict, list[dict], list[dict], dict]:
    C.set_seed(run_seed)
    in_channels = 4
    model = build_model(model_name, in_channels=in_channels, n_classes=3)
    param_count = count_parameters(model)
    started = time.time()
    result = C.train_model(
        model=model,
        X=X,
        y=y,
        idx_tr=idx_tr,
        idx_va=idx_va,
        idx_te=idx_te,
        train_config=C.TRAIN_PRESETS[args.preset],
        seed=run_seed,
        batch_size=args.batch_size,
        oversample=bool(args.oversample),
        device_name=args.device,
    )
    elapsed = time.time() - started

    metric = C.metric_row(
        experiment="additional_models",
        validation_type="stratified_random",
        preset=args.preset,
        oversample=bool(args.oversample),
        run_id=run_id,
        seed=run_seed,
        model_name=model_name,
        param_count=param_count,
        result=result,
    )

    per_class_rows: list[dict] = []
    for row in result["test"]["per_class"]:
        row_out = dict(row)
        row_out.update(
            {
                "experiment": "additional_models",
                "validation_type": "stratified_random",
                "run_id": run_id,
                "seed": run_seed,
                "model": model_name,
            }
        )
        per_class_rows.append(row_out)

    confusion_rows: list[dict] = []
    cm = np.asarray(result["test"]["confusion_matrix"], dtype=int)
    for true_class in range(cm.shape[0]):
        confusion_rows.append(
            {
                "experiment": "additional_models",
                "validation_type": "stratified_random",
                "run_id": run_id,
                "seed": run_seed,
                "model": model_name,
                "true_class": true_class,
                "pred_small": int(cm[true_class, 0]),
                "pred_medium": int(cm[true_class, 1]),
                "pred_large": int(cm[true_class, 2]),
            }
        )

    model_result = {
        "model": model_name,
        "input_kind": input_kind(model_name),
        "param_count": param_count,
        "train_seconds": float(elapsed),
        "result": result,
    }
    return metric, per_class_rows, confusion_rows, model_result


def main() -> None:
    args = parse_args()
    out_dir = C.ensure_dir(args.out)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("At least one model name is required.")

    dataset = C.load_dataset(args.mat, require_source=False)
    X_fixed_raw = C.build_sequence_tensor(dataset.xyz_list, length=args.L)
    masked_max = None if int(args.masked_max_length) <= 0 else int(args.masked_max_length)
    X_masked_raw, masked_lengths = C.build_masked_sequence_tensor(dataset.xyz_list, max_length=masked_max)

    all_metric_rows: list[dict] = []
    all_per_class_rows: list[dict] = []
    all_confusion_rows: list[dict] = []
    all_model_results: list[dict] = []
    split_meta: list[dict] = []

    for run_id in range(int(args.n_runs)):
        run_seed = int(args.seed + run_id)
        idx_tr, idx_va, idx_te = C.stratified_split(
            dataset.y,
            seed=run_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        np.savez(
            out_dir / f"split_indices_seed{run_seed}.npz",
            idx_tr=idx_tr,
            idx_va=idx_va,
            idx_te=idx_te,
        )
        split_meta.append(
            {
                "run_id": run_id,
                "seed": run_seed,
                "split_hash": C.split_hash(idx_tr, idx_va, idx_te),
                "split_counts": {
                    "train": C.split_counts(dataset.y, idx_tr),
                    "val": C.split_counts(dataset.y, idx_va),
                    "test": C.split_counts(dataset.y, idx_te),
                },
            }
        )

        mu_f, sd_f = C.compute_norm_stats(X_fixed_raw, idx_tr)
        X_fixed = C.normalize_with_stats(X_fixed_raw, mu_f, sd_f)
        mu_m, sd_m = C.compute_masked_norm_stats(X_masked_raw, idx_tr)
        X_masked = C.normalize_masked_with_stats(X_masked_raw, mu_m, sd_m)

        run_results: list[dict] = []
        for model_name in model_names:
            X = X_masked if is_masked_model(model_name) else X_fixed
            metric, per_class, confusion, model_result = train_one_model(
                model_name=model_name,
                X=X,
                y=dataset.y,
                idx_tr=idx_tr,
                idx_va=idx_va,
                idx_te=idx_te,
                args=args,
                run_id=run_id,
                run_seed=run_seed,
            )
            all_metric_rows.append(metric)
            all_per_class_rows.extend(per_class)
            all_confusion_rows.extend(confusion)
            run_results.append(model_result)
        all_model_results.append({"run_id": run_id, "seed": run_seed, "models": run_results})

    summary_rows = C.aggregate_metric_rows(all_metric_rows)
    C.write_csv(out_dir / "metrics.csv", all_metric_rows, C.METRIC_FIELDS)
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
    C.write_csv(out_dir / "summary.csv", summary_rows, C.SUMMARY_FIELDS)
    C.write_json(
        out_dir / "results.json",
        {
            "experiment": "additional_models",
            "validation_type": "stratified_random",
            "mat_path": str(args.mat),
            "L": int(args.L),
            "masked_max_length": int(X_masked_raw.shape[2]),
            "masked_length_min_median_max": [
                int(masked_lengths.min()),
                int(np.median(masked_lengths)),
                int(masked_lengths.max()),
            ],
            "preset": args.preset,
            "train_preset": C.TRAIN_PRESETS[args.preset],
            "oversample": bool(args.oversample),
            "batch_size": int(args.batch_size),
            "n_runs": int(args.n_runs),
            "models": model_names,
            "model_input_kind": {name: input_kind(name) for name in model_names},
            "class_counts": np.bincount(dataset.y, minlength=3).astype(int).tolist(),
            "split_meta": split_meta,
            "metrics": all_metric_rows,
            "summary": summary_rows,
            "model_results": all_model_results,
        },
    )


if __name__ == "__main__":
    main()
