# -*- coding: utf-8 -*-
"""Run source-group validation for Chapter 3 supplementary experiments.

This is a sourceIndex-grouped stability validation. It is not a strict
cross-road generalization test unless a true road/site label is provided.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import ch3_common as C
from models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chapter 3 sourceIndex-grouped validation.")
    parser.add_argument(
        "--mat",
        default=C.SOURCE_GROUP_DATA_DEFAULT,
        help="MAT file with ProcessedData, targetLength, and sourceIndex.",
    )
    parser.add_argument("--out", default="outputs/source_group_validation", help="Output directory.")
    parser.add_argument("--models", default="fixed_cnn_1d,lstm_h64,lstm_h32")
    parser.add_argument("--preset", choices=sorted(C.TRAIN_PRESETS), default="paper_fair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_runs", type=int, default=1)
    parser.add_argument("--L", type=int, default=176)
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="Enable train-set minority oversampling. Default keeps the original pipeline shuffle-only setting.",
    )
    return parser.parse_args()


def run_once(
    args: argparse.Namespace,
    dataset: C.DatasetBundle,
    X_raw: np.ndarray,
    model_names: list[str],
    run_id: int,
    run_seed: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    if dataset.source_index is None:
        raise ValueError("sourceIndex is required for source-group validation.")

    idx_tr, idx_va, idx_te = C.source_group_split(
        dataset.y,
        dataset.source_index,
        seed=run_seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    split_id = C.split_hash(idx_tr, idx_va, idx_te)
    mu, sd = C.compute_norm_stats(X_raw, idx_tr)
    X = C.normalize_with_stats(X_raw, mu, sd)

    out_dir = C.ensure_dir(Path(args.out))
    np.savez(
        out_dir / f"source_group_split_seed{run_seed}.npz",
        idx_tr=idx_tr,
        idx_va=idx_va,
        idx_te=idx_te,
    )

    metric_rows: list[dict] = []
    per_class_rows: list[dict] = []
    confusion_rows: list[dict] = []
    model_results: list[dict] = []
    preset = C.TRAIN_PRESETS[args.preset]
    oversample = bool(args.oversample)

    for model_name in model_names:
        # Match the thesis CNN pipeline: bind model initialization to the run seed.
        C.set_seed(run_seed)
        model = build_model(model_name, in_channels=4, n_classes=3)
        param_count = count_parameters(model)
        start = time.time()
        result = C.train_model(
            model=model,
            X=X,
            y=dataset.y,
            idx_tr=idx_tr,
            idx_va=idx_va,
            idx_te=idx_te,
            train_config=preset,
            seed=run_seed,
            batch_size=args.batch_size,
            oversample=oversample,
            device_name=args.device,
        )
        elapsed = time.time() - start

        metric_rows.append(
            C.metric_row(
                experiment="source_group_validation",
                validation_type="source_group",
                preset=args.preset,
                oversample=oversample,
                run_id=run_id,
                seed=run_seed,
                model_name=model_name,
                param_count=param_count,
                result=result,
            )
        )

        for row in result["test"]["per_class"]:
            row_out = dict(row)
            row_out.update(
                {
                    "experiment": "source_group_validation",
                    "validation_type": "source_group",
                    "run_id": run_id,
                    "seed": run_seed,
                    "model": model_name,
                }
            )
            per_class_rows.append(row_out)

        cm = np.asarray(result["test"]["confusion_matrix"], dtype=int)
        for true_class in range(cm.shape[0]):
            confusion_rows.append(
                {
                    "experiment": "source_group_validation",
                    "validation_type": "source_group",
                    "run_id": run_id,
                    "seed": run_seed,
                    "model": model_name,
                    "true_class": true_class,
                    "pred_small": int(cm[true_class, 0]),
                    "pred_medium": int(cm[true_class, 1]),
                    "pred_large": int(cm[true_class, 2]),
                }
            )

        model_results.append(
            {
                "model": model_name,
                "param_count": param_count,
                "train_seconds": float(elapsed),
                "result": result,
            }
        )

    split_meta = {
        "run_id": run_id,
        "seed": run_seed,
        "split_hash": split_id,
        "split_counts": {
            "train": C.split_counts(dataset.y, idx_tr),
            "val": C.split_counts(dataset.y, idx_va),
            "test": C.split_counts(dataset.y, idx_te),
        },
        "note": "Grouped by (class, sourceIndex); not a strict cross-road split.",
    }
    return metric_rows, per_class_rows, confusion_rows, model_results, split_meta


def main() -> None:
    args = parse_args()
    out_dir = C.ensure_dir(args.out)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    if not model_names:
        raise ValueError("At least one model name is required.")

    dataset = C.load_dataset(
        args.mat,
        require_source=True,
        require_grouped_source=False,
    )
    X_raw = C.build_sequence_tensor(dataset.xyz_list, length=args.L)

    all_metric_rows: list[dict] = []
    all_per_class_rows: list[dict] = []
    all_confusion_rows: list[dict] = []
    all_model_results: list[dict] = []
    split_meta: list[dict] = []

    for run_id in range(int(args.n_runs)):
        run_seed = int(args.seed + run_id)
        metric_rows, per_class_rows, confusion_rows, model_results, split = run_once(
            args=args,
            dataset=dataset,
            X_raw=X_raw,
            model_names=model_names,
            run_id=run_id,
            run_seed=run_seed,
        )
        all_metric_rows.extend(metric_rows)
        all_per_class_rows.extend(per_class_rows)
        all_confusion_rows.extend(confusion_rows)
        all_model_results.append({"run_id": run_id, "seed": run_seed, "models": model_results})
        split_meta.append(split)

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
            "experiment": "source_group_validation",
            "validation_type": "source_group",
            "validation_note": (
                "This is sourceIndex-grouped validation. It should not be described as "
                "strict cross-road generalization without true site labels."
            ),
            "mat_path": str(args.mat),
            "L": int(args.L),
            "preset": args.preset,
            "train_preset": C.TRAIN_PRESETS[args.preset],
            "oversample": bool(args.oversample),
            "batch_size": int(args.batch_size),
            "n_runs": int(args.n_runs),
            "models": model_names,
            "class_counts": np.bincount(dataset.y, minlength=3).astype(int).tolist(),
            "split_meta": split_meta,
            "metrics": all_metric_rows,
            "summary": summary_rows,
            "model_results": all_model_results,
        },
    )


if __name__ == "__main__":
    main()
