from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import run_ch3_experiment_lift as R1
import run_ch3_thesis_pipeline as TP


THIS_DIR = Path(__file__).resolve().parent
BASELINE_SUMMARY = THIS_DIR / "out_ch3_lift_r2_20260314_seed42" / "results" / "summary.json"
DTW_SUMMARY = THIS_DIR / "out_ch3_dtw_tune_kle4_20260314_seed42" / "results" / "summary.json"
LABELS_CN = ["小型车", "中型车", "大型车"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=THIS_DIR / "out_confusion_preview_matlab_style_20260314",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    TP.ensure_plot_style()

    baseline_summary = read_json(BASELINE_SUMMARY)
    dtw_summary = read_json(DTW_SUMMARY)

    mat_path = R1.default_mat_path()
    _, y, _ = TP.load_xyz_y_from_mat(str(mat_path))
    _, _, idx_te = TP.stratified_split(y, seed=42)
    y_true = y[idx_te]

    y_pred_baseline = np.asarray(baseline_summary["baseline"]["metrics"]["y_pred_test"], dtype=int)
    y_pred_dtw = np.asarray(dtw_summary["best_by_val"]["metrics"]["y_pred_test"], dtype=int)

    TP.plot_confusion(
        y_true,
        y_pred_baseline,
        LABELS_CN,
        "线性定长卷积网络基线",
        str(args.out_dir / "cm_cnn_baseline.png"),
        normalize=False,
    )
    TP.plot_confusion(
        y_true,
        y_pred_dtw,
        LABELS_CN,
        "DTW 多模板增强方法",
        str(args.out_dir / "cm_dtw_multi_cnn.png"),
        normalize=False,
    )

    payload = {
        "baseline_summary": str(BASELINE_SUMMARY),
        "dtw_summary": str(DTW_SUMMARY),
        "mat_path": str(mat_path),
        "outputs": [
            str(args.out_dir / "cm_cnn_baseline.png"),
            str(args.out_dir / "cm_dtw_multi_cnn.png"),
        ],
    }
    (args.out_dir / "preview_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Preview saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
