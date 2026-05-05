from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
DTW_CODE_DIR = THIS_DIR / "dtw_cnn_handoff" / "code"
for _path in (THIS_DIR, DTW_CODE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_ch3_experiment_lift as R1
import run_ch3_experiment_lift_r2 as R2
import run_ch3_thesis_pipeline as TP
import run_dtw_multi_sweep as M


K_GRID = (2, 3, 4)
FAMILY_PRESETS = {
    "raw_quantile": ("D1", "D2"),
    "medoid": ("D3",),
    "dba": ("D5",),
}
WR_STEP_GRID = {
    "raw_quantile": (
        (0.10, 0.00),
        (0.12, 0.00),
        (0.15, 0.00),
        (0.15, 0.05),
        (0.20, 0.00),
    ),
    "medoid": (
        (0.10, 0.00),
        (0.10, 0.05),
        (0.15, 0.00),
        (0.15, 0.05),
    ),
    "dba": (
        (0.10, 0.00),
        (0.15, 0.00),
        (0.15, 0.05),
        (0.20, 0.00),
    ),
}


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def preset_map() -> dict[str, dict]:
    return {preset["name"]: dict(preset) for preset in R2.DTW_PRESETS}


def choose_with_eps(rows: list[dict], key: str, eps: float, tie_key):
    ranked = sorted(rows, key=lambda row: -row[key])
    best = ranked[0][key]
    close = [row for row in ranked if best - row[key] <= eps]
    close.sort(key=tie_key)
    return close[0]


def evaluate_candidate(
    X_raw: np.ndarray,
    y: np.ndarray,
    tlen: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    family: str,
    preset: dict,
    K: int,
    wR: float,
    step: float,
    seed: int,
    proto_dba_iter: int,
) -> dict:
    dataset = R2.build_family_dataset(
        X_raw,
        y,
        idx_tr,
        tlen,
        family=family,
        K=K,
        wR=wR,
        step=step,
        dba_iter=proto_dba_iter,
    )
    result = R1.run_cnn_experiment(
        dataset["X_multi"],
        y,
        idx_tr,
        idx_va,
        idx_te,
        in_ch=dataset["in_ch"],
        seed=seed,
        preset=preset,
    )
    return {
        "family": family,
        "preset_name": preset["name"],
        "lr": float(preset["lr"]),
        "weight_decay": float(preset["weight_decay"]),
        "max_epochs": int(preset["max_epochs"]),
        "patience": int(preset["patience"]),
        "K": int(K),
        "wR": float(wR),
        "step": float(step),
        "build_s": float(dataset["build_s"]),
        "in_ch": int(dataset["in_ch"]),
        "val_acc": float(result["metrics"]["val_acc"]),
        "val_f1": float(result["metrics"]["val_f1"]),
        "test_acc": float(result["metrics"]["test_acc"]),
        "test_f1": float(result["metrics"]["test_f1"]),
        "best_epoch": int(result["metrics"]["best_epoch"]),
        "train_s": float(result["train_s"]),
        "metrics": result["metrics"],
        "prototype_meta": dataset["meta"],
    }


def summarize_rows(rows: list[dict]) -> dict:
    best_by_val = choose_with_eps(
        rows,
        key="val_f1",
        eps=0.005,
        tie_key=lambda row: (int(row["K"]), abs(float(row["wR"]) - 0.10), abs(float(row["step"]) - 0.00)),
    )
    best_by_test = sorted(
        rows,
        key=lambda row: (-float(row["test_f1"]), -float(row["val_f1"]), int(row["K"])),
    )[0]
    best_per_family = {}
    for family in sorted({str(row["family"]) for row in rows}):
        fam_rows = [row for row in rows if str(row["family"]) == family]
        best_per_family[family] = {
            "best_by_val": choose_with_eps(
                fam_rows,
                key="val_f1",
                eps=0.005,
                tie_key=lambda row: (int(row["K"]), abs(float(row["wR"]) - 0.10), abs(float(row["step"]) - 0.00)),
            ),
            "best_by_test": sorted(
                fam_rows,
                key=lambda row: (-float(row["test_f1"]), -float(row["val_f1"]), int(row["K"])),
            )[0],
        }
    return {
        "best_by_val": best_by_val,
        "best_by_test": best_by_test,
        "best_per_family": best_per_family,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=176)
    parser.add_argument("--proto_dba_iter", type=int, default=5)
    args = parser.parse_args()

    TP.ensure_plot_style()
    mat_path = args.mat if args.mat is not None else R1.default_mat_path()
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    if args.out_dir is None:
        tag = date.today().strftime("%Y%m%d")
        out_dir = R1.ensure_unique_out_dir(THIS_DIR / f"out_ch3_dtw_tune_kle4_{tag}_seed{args.seed}")
    else:
        out_dir = R1.ensure_unique_out_dir(args.out_dir)

    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    xyz_list, y, tlen = TP.load_xyz_y_from_mat(str(mat_path))
    idx_tr, idx_va, idx_te = TP.stratified_split(y, seed=args.seed)
    np.savez(results_dir / f"split_indices_seed{args.seed}.npz", idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te)

    X_raw = M.build_baseline_data(xyz_list, args.L)
    preset_by_name = preset_map()

    rows: list[dict] = []
    started = time.time()
    for family, preset_names in FAMILY_PRESETS.items():
        for preset_name in preset_names:
            preset = preset_by_name[preset_name]
            for K in K_GRID:
                for wR, step in WR_STEP_GRID[family]:
                    row = evaluate_candidate(
                        X_raw,
                        y,
                        tlen,
                        idx_tr,
                        idx_va,
                        idx_te,
                        family=family,
                        preset=preset,
                        K=K,
                        wR=wR,
                        step=step,
                        seed=args.seed,
                        proto_dba_iter=args.proto_dba_iter,
                    )
                    rows.append(row)
                    pd.DataFrame(
                        [
                            {
                                "family": item["family"],
                                "preset_name": item["preset_name"],
                                "K": item["K"],
                                "wR": item["wR"],
                                "step": item["step"],
                                "in_ch": item["in_ch"],
                                "val_acc": item["val_acc"],
                                "val_f1": item["val_f1"],
                                "test_acc": item["test_acc"],
                                "test_f1": item["test_f1"],
                                "best_epoch": item["best_epoch"],
                                "train_s": item["train_s"],
                                "build_s": item["build_s"],
                            }
                            for item in rows
                        ]
                    ).to_csv(results_dir / "dtw_tune_kle4.csv", index=False)
                    print(
                        f"[{len(rows):02d}] family={family} preset={preset_name} "
                        f"K={K} wR={wR:.2f} step={step:.2f} "
                        f"val_f1={row['val_f1']:.4f} test_f1={row['test_f1']:.4f}"
                    )

    summary = summarize_rows(rows)
    summary["grid"] = {
        "K_grid": list(K_GRID),
        "family_presets": {key: list(val) for key, val in FAMILY_PRESETS.items()},
        "wr_step_grid": {key: [list(x) for x in val] for key, val in WR_STEP_GRID.items()},
        "proto_dba_iter": int(args.proto_dba_iter),
        "elapsed_s": float(time.time() - started),
    }
    json_dump(results_dir / "summary.json", summary)
    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
