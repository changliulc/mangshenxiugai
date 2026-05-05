# -*- coding: utf-8 -*-
"""Collect supplementary experiment outputs into thesis-friendly CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import ch3_common as C


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Chapter 3 supplementary experiment outputs.")
    parser.add_argument("--in_dir", default="outputs", help="Directory containing result subfolders.")
    parser.add_argument("--out", default="tables_required", help="Directory for summary tables.")
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_dict(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = C.ensure_dir(args.out)

    result_files = sorted(in_dir.rglob("results.json"))
    if not result_files:
        raise FileNotFoundError(f"No results.json found under {in_dir}")

    per_run_rows: list[dict] = []
    summary_rows: list[dict] = []
    provenance_rows: list[dict] = []

    for result_file in result_files:
        payload = load_json(result_file)
        metrics_file = result_file.parent / "metrics.csv"
        summary_file = result_file.parent / "summary.csv"

        if metrics_file.exists():
            for row in read_csv_dict(metrics_file):
                row["result_dir"] = str(result_file.parent)
                per_run_rows.append(row)
        if summary_file.exists():
            for row in read_csv_dict(summary_file):
                row["experiment"] = payload.get("experiment", "")
                row["validation_type"] = payload.get("validation_type", "")
                row["preset"] = payload.get("preset", "")
                row["oversample"] = int(bool(payload.get("oversample", False)))
                row["result_dir"] = str(result_file.parent)
                summary_rows.append(row)

        provenance_rows.append(
            {
                "result_dir": str(result_file.parent),
                "experiment": payload.get("experiment", ""),
                "validation_type": payload.get("validation_type", ""),
                "preset": payload.get("preset", ""),
                "oversample": int(bool(payload.get("oversample", False))),
                "L": payload.get("L", ""),
                "n_runs": payload.get("n_runs", ""),
                "mat_path": payload.get("mat_path", ""),
            }
        )

    C.write_csv(
        out_dir / "per_run_metrics_all.csv",
        per_run_rows,
        C.METRIC_FIELDS + ["result_dir"],
    )
    C.write_csv(
        out_dir / "required_experiment_summary.csv",
        summary_rows,
        [
            "experiment",
            "validation_type",
            "preset",
            "oversample",
            "model",
            "runs",
            "param_count",
            "test_acc_mean",
            "test_acc_std",
            "test_macro_f1_mean",
            "test_macro_f1_std",
            "val_macro_f1_mean",
            "val_macro_f1_std",
            "result_dir",
        ],
    )
    C.write_csv(
        out_dir / "result_provenance.csv",
        provenance_rows,
        [
            "result_dir",
            "experiment",
            "validation_type",
            "preset",
            "oversample",
            "L",
            "n_runs",
            "mat_path",
        ],
    )


if __name__ == "__main__":
    main()

