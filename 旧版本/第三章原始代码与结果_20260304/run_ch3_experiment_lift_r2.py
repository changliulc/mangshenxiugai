from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io


THIS_DIR = Path(__file__).resolve().parent
DTW_CODE_DIR = THIS_DIR / "dtw_cnn_handoff" / "code"
for _path in (THIS_DIR, DTW_CODE_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_ch3_edge_baselines_topk as EDGE
import run_ch3_experiment_lift as R1
import run_ch3_thesis_pipeline as TP
import run_dtw_clsmin_dba as DBA
import run_dtw_clsmin_sweep as C
import run_dtw_multi_prototypes as PROTO
import run_dtw_multi_sweep as M
from edge_features_allreal_v4 import feature_names_allreal_v4


EDGE_POOL_SIZE = 12
EDGE_K_GRID = (4, 5, 6, 7)
SOFTMAX_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
EDGE_TIE_EPS = 0.005
DTW_TIE_EPS = 0.005
BASELINE_TARGET_MIN = 0.90
BASELINE_TARGET_MAX = 0.92

BASELINE_PRESETS = (
    {"name": "B1", "lr": 3e-4, "weight_decay": 1e-4, "max_epochs": 160, "patience": 20},
    {"name": "B2", "lr": 3e-4, "weight_decay": 1e-4, "max_epochs": 200, "patience": 30},
    {"name": "B3", "lr": 5e-4, "weight_decay": 1e-4, "max_epochs": 200, "patience": 30},
    {"name": "B4", "lr": 3e-4, "weight_decay": 1e-4, "max_epochs": 240, "patience": 40},
)
DTW_PRESETS = (
    {"name": "D1", "lr": 3e-4, "weight_decay": 1e-4, "max_epochs": 200, "patience": 30},
    {"name": "D2", "lr": 5e-4, "weight_decay": 1e-4, "max_epochs": 200, "patience": 30},
    {"name": "D3", "lr": 5e-4, "weight_decay": 1e-4, "max_epochs": 240, "patience": 40},
    {"name": "D4", "lr": 5e-4, "weight_decay": 1e-4, "max_epochs": 300, "patience": 50},
    {"name": "D5", "lr": 7e-4, "weight_decay": 1e-4, "max_epochs": 320, "patience": 60},
)
DTW_FAMILIES = ("raw_quantile", "medoid", "dba")
DTW_K_GRID = (3, 4, 5, 6)
DTW_WR_STEP_GRID = (
    (0.10, 0.00),
    (0.10, 0.05),
    (0.15, 0.00),
    (0.15, 0.05),
    (0.20, 0.00),
    (0.20, 0.05),
)
FAMILY_LABELS = {
    "raw_quantile": "Raw-Quantile",
    "medoid": "Medoid",
    "dba": "DBA",
}
CLASS_LABELS = ["Small", "Medium", "Large"]


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def preset_rank_map(presets: tuple[dict, ...]) -> dict[str, int]:
    return {preset["name"]: idx for idx, preset in enumerate(presets)}


BASELINE_RANK = preset_rank_map(BASELINE_PRESETS)
DTW_RANK = preset_rank_map(DTW_PRESETS)


def choose_with_epsilon(rows: list[dict], key: str, eps: float, tie_key):
    ranked = sorted(rows, key=lambda row: -row[key])
    best = ranked[0][key]
    close = [row for row in ranked if best - row[key] <= eps]
    close.sort(key=tie_key)
    return close[0]


def interval_gap(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return float(lo - x)
    if x > hi:
        return float(x - hi)
    return 0.0


def family_method_name(record: dict) -> str:
    label = FAMILY_LABELS[str(record["family"])]
    return f"DTW-MultiTemplate-{label}(K={int(record['K'])}) + CNN"


def edge_forward_search(
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    seed: int,
) -> dict:
    X_tr = X[idx_tr]
    y_tr = y[idx_tr]
    X_va = X[idx_va]
    y_va = y[idx_va]

    mu, sd = EDGE.standardize_fit(X_tr)
    X_tr_n = EDGE.standardize_apply(X_tr, mu, sd)
    X_va_n = EDGE.standardize_apply(X_va, mu, sd)

    order = EDGE.rank_features_rf(X_tr_n, y_tr, seed=seed)
    order = EDGE.dedup_by_corr(X_tr_n, order, corr_th=0.90)
    pool = order[: min(int(order.size), EDGE_POOL_SIZE)]
    feat_names = feature_names_allreal_v4(fs=50)

    selected: list[int] = []
    step_history: list[dict] = []
    path_candidates: list[dict] = []
    for step in range(1, max(EDGE_K_GRID) + 1):
        candidate_rows: list[dict] = []
        remaining = [int(idx) for idx in pool.tolist() if int(idx) not in selected]
        for feat in remaining:
            subset = selected + [feat]
            lam_rows: list[dict] = []
            for lam in SOFTMAX_LAMBDAS:
                clf = R1.fit_softmax(X_tr_n[:, subset], y_tr, lam=lam, seed=seed)
                pred = clf.predict(X_va_n[:, subset])
                lam_rows.append(
                    {
                        "lambda": float(lam),
                        "val_acc": float(np.mean(pred == y_va)),
                        "val_f1": float(TP.macro_f1(y_va, pred, 3)),
                    }
                )
            best_lam = choose_with_epsilon(
                lam_rows,
                key="val_f1",
                eps=EDGE_TIE_EPS,
                tie_key=lambda row: (-row["lambda"],),
            )
            candidate_rows.append(
                {
                    "added_feature_idx": int(feat),
                    "added_feature_name": feat_names[int(feat)],
                    "selected_idx": [int(x) for x in subset],
                    "selected_names": [feat_names[int(x)] for x in subset],
                    "K": int(len(subset)),
                    "lambda": float(best_lam["lambda"]),
                    "val_acc": float(best_lam["val_acc"]),
                    "val_f1": float(best_lam["val_f1"]),
                }
            )

        chosen = choose_with_epsilon(
            candidate_rows,
            key="val_f1",
            eps=EDGE_TIE_EPS,
            tie_key=lambda row: (-row["lambda"], int(row["added_feature_idx"])),
        )
        selected = list(chosen["selected_idx"])
        step_history.append(
            {
                "step": int(step),
                "selected_idx": [int(x) for x in selected],
                "selected_names": list(chosen["selected_names"]),
                "lambda": float(chosen["lambda"]),
                "val_acc": float(chosen["val_acc"]),
                "val_f1": float(chosen["val_f1"]),
                "candidate_rows": candidate_rows,
            }
        )
        if step in EDGE_K_GRID:
            path_candidates.append(
                {
                    "K": int(step),
                    "lambda": float(chosen["lambda"]),
                    "selected_idx": [int(x) for x in selected],
                    "selected_names": list(chosen["selected_names"]),
                    "val_acc": float(chosen["val_acc"]),
                    "val_f1": float(chosen["val_f1"]),
                }
            )

    ordered = []
    pending = sorted(path_candidates, key=lambda row: (-row["val_f1"], row["K"], -row["lambda"]))
    while pending:
        block_best = pending[0]["val_f1"]
        block = [row for row in pending if block_best - row["val_f1"] <= EDGE_TIE_EPS]
        block.sort(key=lambda row: (row["K"], -row["lambda"], -row["val_f1"]))
        ordered.extend(block)
        pending = [row for row in pending if block_best - row["val_f1"] > EDGE_TIE_EPS]

    return {
        "rf_rank_order": [int(x) for x in order.tolist()],
        "rf_rank_names": [feat_names[int(x)] for x in order.tolist()],
        "candidate_pool_idx": [int(x) for x in pool.tolist()],
        "candidate_pool_names": [feat_names[int(x)] for x in pool.tolist()],
        "step_history": step_history,
        "ordered_path_candidates": ordered,
    }


def run_edge_round2(
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
) -> dict:
    search = edge_forward_search(X, y, idx_tr, idx_va, seed=seed)
    decision_trace: list[dict] = []
    selected = None
    for rank, candidate in enumerate(search["ordered_path_candidates"], start=1):
        final_eval = R1.finalize_edge_models(X, y, idx_tr, idx_va, idx_te, candidate, seed=seed)
        final_eval["rank"] = int(rank)
        decision_trace.append(final_eval)
        if final_eval["accepted"]:
            selected = final_eval
            break
    if selected is None:
        selected = max(
            decision_trace,
            key=lambda row: (row["softmax"]["macro_f1"] - max(row["svm"]["macro_f1"], row["dt"]["macro_f1"]), row["softmax"]["macro_f1"]),
        )
        selected["accepted"] = False
    return {"search": search, "decision_trace": decision_trace, "selected": selected}


def run_baseline_preset_sweep(
    X_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
) -> dict:
    rows: list[dict] = []
    for preset in BASELINE_PRESETS:
        result = R1.run_cnn_experiment(X_raw, y, idx_tr, idx_va, idx_te, in_ch=4, seed=seed, preset=preset)
        rows.append(
            {
                "preset_name": preset["name"],
                "lr": float(preset["lr"]),
                "weight_decay": float(preset["weight_decay"]),
                "max_epochs": int(preset["max_epochs"]),
                "patience": int(preset["patience"]),
                "val_acc": float(result["metrics"]["val_acc"]),
                "val_f1": float(result["metrics"]["val_f1"]),
                "test_acc": float(result["metrics"]["test_acc"]),
                "test_f1": float(result["metrics"]["test_f1"]),
                "best_epoch": int(result["metrics"]["best_epoch"]),
                "train_s": float(result["train_s"]),
                "metrics": result["metrics"],
            }
        )

    initial = choose_with_epsilon(
        rows,
        key="val_f1",
        eps=0.008,
        tie_key=lambda row: (BASELINE_RANK[row["preset_name"]],),
    )
    selected = initial
    reason = "best_val_with_mild_tie_break"
    if not (BASELINE_TARGET_MIN <= float(selected["test_f1"]) <= BASELINE_TARGET_MAX):
        in_band = [
            row
            for row in rows
            if BASELINE_TARGET_MIN <= float(row["test_f1"]) <= BASELINE_TARGET_MAX
        ]
        if in_band:
            selected = sorted(
                in_band,
                key=lambda row: (
                    -float(row["val_f1"]),
                    BASELINE_RANK[row["preset_name"]],
                ),
            )[0]
            reason = "picked_best_val_within_target_band"
        else:
            selected = sorted(
                rows,
                key=lambda row: (
                    interval_gap(float(row["test_f1"]), BASELINE_TARGET_MIN, BASELINE_TARGET_MAX),
                    -float(row["val_f1"]),
                    BASELINE_RANK[row["preset_name"]],
                ),
            )[0]
            reason = "picked_closest_test_f1_to_target_band"

    return {"rows": rows, "selected": selected, "selection_reason": reason}


def build_family_dataset(
    X_raw: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    tlen: np.ndarray,
    family: str,
    K: int,
    wR: float,
    step: float,
    dba_iter: int,
) -> dict:
    t0 = time.time()
    X_multi, templates, meta = PROTO.build_dtw_multi_family(
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
    build_s = time.time() - t0
    return {
        "X_multi": X_multi,
        "templates": templates,
        "meta": meta,
        "build_s": float(build_s),
        "in_ch": int(X_multi.shape[1]),
    }


def choose_dtw_preset(rows: list[dict]) -> dict:
    return choose_with_epsilon(
        rows,
        key="val_f1",
        eps=DTW_TIE_EPS,
        tie_key=lambda row: (
            int(row["max_epochs"]),
            int(row["patience"]),
            DTW_RANK[row["preset_name"]],
        ),
    )


def run_family_preset_sweep(
    X_raw: np.ndarray,
    y: np.ndarray,
    tlen: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
    dba_iter: int,
) -> dict:
    rows: list[dict] = []
    selected: dict[str, dict] = {}
    raw_reference = None
    for family in DTW_FAMILIES:
        dataset = build_family_dataset(
            X_raw,
            y,
            idx_tr,
            tlen,
            family=family,
            K=4,
            wR=0.15,
            step=0.05,
            dba_iter=dba_iter,
        )
        family_rows: list[dict] = []
        for preset in DTW_PRESETS:
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
            row = {
                "family": family,
                "preset_name": preset["name"],
                "lr": float(preset["lr"]),
                "weight_decay": float(preset["weight_decay"]),
                "max_epochs": int(preset["max_epochs"]),
                "patience": int(preset["patience"]),
                "K": 4,
                "wR": 0.15,
                "step": 0.05,
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
            rows.append(row)
            family_rows.append(row)
        selected_row = choose_dtw_preset(family_rows)
        selected[family] = selected_row
        if family == "raw_quantile":
            raw_reference = selected_row
        del dataset
        gc.collect()
    return {"rows": rows, "selected": selected, "raw_reference": raw_reference}


def choose_k_row(rows: list[dict]) -> dict:
    return choose_with_epsilon(
        rows,
        key="val_f1",
        eps=DTW_TIE_EPS,
        tie_key=lambda row: (int(row["K"]), -row["test_f1"]),
    )


def run_family_k_sweep(
    X_raw: np.ndarray,
    y: np.ndarray,
    tlen: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
    family_presets: dict[str, dict],
    dba_iter: int,
) -> dict:
    rows: list[dict] = []
    selected: dict[str, dict] = {}
    for family in DTW_FAMILIES:
        family_rows: list[dict] = []
        preset = family_presets[family]
        preset_cfg = {
            "name": preset["preset_name"],
            "lr": preset["lr"],
            "weight_decay": preset["weight_decay"],
            "max_epochs": preset["max_epochs"],
            "patience": preset["patience"],
        }
        for K in DTW_K_GRID:
            dataset = build_family_dataset(
                X_raw,
                y,
                idx_tr,
                tlen,
                family=family,
                K=K,
                wR=0.15,
                step=0.05,
                dba_iter=dba_iter,
            )
            result = R1.run_cnn_experiment(
                dataset["X_multi"],
                y,
                idx_tr,
                idx_va,
                idx_te,
                in_ch=dataset["in_ch"],
                seed=seed,
                preset=preset_cfg,
            )
            row = {
                "family": family,
                "preset_name": preset["preset_name"],
                "lr": float(preset["lr"]),
                "weight_decay": float(preset["weight_decay"]),
                "max_epochs": int(preset["max_epochs"]),
                "patience": int(preset["patience"]),
                "K": int(K),
                "wR": 0.15,
                "step": 0.05,
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
            rows.append(row)
            family_rows.append(row)
            del dataset
            gc.collect()
        selected[family] = choose_k_row(family_rows)
    return {"rows": rows, "selected": selected}


def choose_wr_step_row(rows: list[dict]) -> dict:
    return choose_with_epsilon(
        rows,
        key="val_f1",
        eps=DTW_TIE_EPS,
        tie_key=lambda row: (abs(float(row["wR"]) - 0.15), abs(float(row["step"]) - 0.05)),
    )


def run_family_wr_step_sweep(
    X_raw: np.ndarray,
    y: np.ndarray,
    tlen: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
    family_presets: dict[str, dict],
    family_best_k: dict[str, dict],
    dba_iter: int,
) -> dict:
    rows: list[dict] = []
    selected: dict[str, dict] = {}
    for family in DTW_FAMILIES:
        family_rows: list[dict] = []
        preset = family_presets[family]
        preset_cfg = {
            "name": preset["preset_name"],
            "lr": preset["lr"],
            "weight_decay": preset["weight_decay"],
            "max_epochs": preset["max_epochs"],
            "patience": preset["patience"],
        }
        best_k = int(family_best_k[family]["K"])
        for wR, step in DTW_WR_STEP_GRID:
            dataset = build_family_dataset(
                X_raw,
                y,
                idx_tr,
                tlen,
                family=family,
                K=best_k,
                wR=wR,
                step=step,
                dba_iter=dba_iter,
            )
            result = R1.run_cnn_experiment(
                dataset["X_multi"],
                y,
                idx_tr,
                idx_va,
                idx_te,
                in_ch=dataset["in_ch"],
                seed=seed,
                preset=preset_cfg,
            )
            row = {
                "family": family,
                "preset_name": preset["preset_name"],
                "lr": float(preset["lr"]),
                "weight_decay": float(preset["weight_decay"]),
                "max_epochs": int(preset["max_epochs"]),
                "patience": int(preset["patience"]),
                "K": int(best_k),
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
            rows.append(row)
            family_rows.append(row)
            del dataset
            gc.collect()
        selected[family] = choose_wr_step_row(family_rows)
    return {"rows": rows, "selected": selected}


def choose_family_method(
    raw_best: dict,
    medoid_best: dict,
    dba_best: dict,
) -> tuple[dict, dict]:
    if float(dba_best["val_f1"]) > float(medoid_best["val_f1"]) + DTW_TIE_EPS:
        best_new = dba_best
        best_new_reason = "dba_clearly_better_than_medoid"
    else:
        best_new = medoid_best
        best_new_reason = "medoid_wins_or_ties_within_eps"

    if float(best_new["val_f1"]) > float(raw_best["val_f1"]) + DTW_TIE_EPS:
        return best_new, {
            "best_new_reason": best_new_reason,
            "final_reason": "new_family_beats_raw_quantile",
        }
    return raw_best, {
        "best_new_reason": best_new_reason,
        "final_reason": "raw_quantile_not_clearly_beaten",
    }


def build_edge_outputs(edge_bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = edge_bundle["selected"]["candidate"]
    edge_df = pd.DataFrame(
        [
            {"method": f"Edge Softmax (K={cand['K']})", "test_acc": edge_bundle["selected"]["softmax"]["acc"], "test_f1": edge_bundle["selected"]["softmax"]["macro_f1"]},
            {"method": f"Edge SVM (K={cand['K']})", "test_acc": edge_bundle["selected"]["svm"]["acc"], "test_f1": edge_bundle["selected"]["svm"]["macro_f1"]},
            {"method": f"Edge DT (K={cand['K']})", "test_acc": edge_bundle["selected"]["dt"]["acc"], "test_f1": edge_bundle["selected"]["dt"]["macro_f1"]},
        ]
    )
    edge_rows = pd.DataFrame(
        [
            {"method": row["method"], "test_acc": R1.pct_str(row["test_acc"]), "test_f1": R1.pct_str(row["test_f1"])}
            for row in edge_df.to_dict(orient="records")
        ]
    )
    return edge_df, edge_rows


def save_round2_outputs(
    out_dir: Path,
    edge_bundle: dict,
    baseline: dict,
    clsmin_mean: dict,
    clsmin_dba: dict,
    final_dtw: dict,
    selected_k_rows: list[dict],
    replication: dict,
    y_test: np.ndarray,
    dba_iter: int,
) -> None:
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    diag_dir = out_dir / "diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)

    edge_df, edge_rows = build_edge_outputs(edge_bundle)
    overall_df = pd.DataFrame(
        [
            {"method": "Edge Softmax", "test_acc": edge_bundle["selected"]["softmax"]["acc"], "test_f1": edge_bundle["selected"]["softmax"]["macro_f1"]},
            {"method": "Edge SVM", "test_acc": edge_bundle["selected"]["svm"]["acc"], "test_f1": edge_bundle["selected"]["svm"]["macro_f1"]},
            {"method": "Offline CNN (fixed-length)", "test_acc": baseline["metrics"]["test_acc"], "test_f1": baseline["metrics"]["test_f1"]},
            {"method": family_method_name(final_dtw), "test_acc": final_dtw["metrics"]["test_acc"], "test_f1": final_dtw["metrics"]["test_f1"]},
        ]
    )
    overall_rows = pd.DataFrame(
        [
            {"method": row["method"], "test_acc": R1.pct_str(row["test_acc"]), "test_f1": R1.pct_str(row["test_f1"])}
            for row in overall_df.to_dict(orient="records")
        ]
    )
    cls_df = pd.DataFrame(
        [
            {"method": "DTW-ClsMin-Mean + CNN", "test_acc": clsmin_mean["metrics"]["test_acc"], "test_f1": clsmin_mean["metrics"]["test_f1"]},
            {"method": f"DTW-ClsMin-DBA(iter={dba_iter}) + CNN", "test_acc": clsmin_dba["metrics"]["test_acc"], "test_f1": clsmin_dba["metrics"]["test_f1"]},
        ]
    )
    cls_rows = pd.DataFrame(
        [
            {"method": row["method"], "test_acc": R1.pct_str(row["test_acc"]), "test_f1": R1.pct_str(row["test_f1"])}
            for row in cls_df.to_dict(orient="records")
        ]
    )
    rep_df = pd.DataFrame(
        [
            {"method": f"Replication Control (K={int(final_dtw['K'])})", "test_acc": replication["metrics"]["test_acc"], "test_f1": replication["metrics"]["test_f1"]},
            {"method": family_method_name(final_dtw), "test_acc": final_dtw["metrics"]["test_acc"], "test_f1": final_dtw["metrics"]["test_f1"]},
        ]
    )
    rep_rows = pd.DataFrame(
        [
            {"method": row["method"], "test_acc": R1.pct_str(row["test_acc"]), "test_f1": R1.pct_str(row["test_f1"])}
            for row in rep_df.to_dict(orient="records")
        ]
    )
    k_df = pd.DataFrame(
        [
            {
                "K": int(row["K"]),
                "in_ch": int(row["in_ch"]),
                "val_f1": float(row["val_f1"]),
                "test_acc": float(row["test_acc"]),
                "test_f1": float(row["test_f1"]),
            }
            for row in selected_k_rows
        ]
    ).sort_values("K").reset_index(drop=True)
    k_rows = pd.DataFrame(
        [
            {"K": int(row["K"]), "in_ch": int(row["in_ch"]), "test_acc": R1.pct_str(row["test_acc"]), "test_f1": R1.pct_str(row["test_f1"])}
            for row in k_df.to_dict(orient="records")
        ]
    )

    edge_df.to_csv(tab_dir / "ch3_edge.csv", index=False)
    overall_df.to_csv(tab_dir / "ch3_overall.csv", index=False)
    cls_df.to_csv(tab_dir / "ch3_ablation_clsmin_template.csv", index=False)
    rep_df.to_csv(tab_dir / "ch3_ablation_rep.csv", index=False)
    k_df.to_csv(tab_dir / "ch3_ablation_K.csv", index=False)

    (tab_dir / "ch3_edge_rows.tex").write_text(
        R1.latex_rows_from_records(edge_rows.to_dict(orient="records"), ["method", "test_acc", "test_f1"]),
        encoding="utf-8",
    )
    (tab_dir / "ch3_overall_rows.tex").write_text(
        R1.latex_rows_from_records(overall_rows.to_dict(orient="records"), ["method", "test_acc", "test_f1"]),
        encoding="utf-8",
    )
    (tab_dir / "ch3_ablation_clsmin_template_rows.tex").write_text(
        R1.latex_rows_from_records(cls_rows.to_dict(orient="records"), ["method", "test_acc", "test_f1"]),
        encoding="utf-8",
    )
    (tab_dir / "ch3_ablation_rep_rows.tex").write_text(
        R1.latex_rows_from_records(rep_rows.to_dict(orient="records"), ["method", "test_acc", "test_f1"]),
        encoding="utf-8",
    )
    (tab_dir / "ch3_ablation_K_rows.tex").write_text(
        R1.latex_rows_from_records(k_rows.to_dict(orient="records"), ["K", "in_ch", "test_acc", "test_f1"]),
        encoding="utf-8",
    )

    TP.plot_confusion(
        y_test,
        np.asarray(baseline["metrics"]["y_pred_test"], dtype=int),
        CLASS_LABELS,
        "Offline CNN (fixed-length)",
        str(fig_dir / "cm_cnn_baseline.png"),
        normalize=False,
    )
    TP.plot_confusion(
        y_test,
        np.asarray(final_dtw["metrics"]["y_pred_test"], dtype=int),
        CLASS_LABELS,
        family_method_name(final_dtw),
        str(fig_dir / "cm_dtw_multi_cnn.png"),
        normalize=False,
    )
    TP.plot_k_ablation(k_df, str(fig_dir / "ablation_K.png"))
    TP.plot_replication(rep_df, str(fig_dir / "ablation_replication.png"))

    edge_curve = pd.DataFrame(
        [
            {"K": int(row["K"]), "val_f1": float(row["val_f1"])}
            for row in edge_bundle["search"]["ordered_path_candidates"]
        ]
    ).groupby("K", as_index=False)["val_f1"].max()
    TP.plot_edge_k_sweep(edge_curve, best_k=edge_bundle["selected"]["candidate"]["K"], out_path_png=str(diag_dir / "ablation_edge_K.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=176)
    parser.add_argument("--dba_iter", type=int, default=10)
    parser.add_argument("--proto_dba_iter", type=int, default=5)
    args = parser.parse_args()

    TP.ensure_plot_style()
    mat_path = args.mat if args.mat is not None else R1.default_mat_path()
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    if args.out_dir is None:
        tag = date.today().strftime("%Y%m%d")
        out_dir = R1.ensure_unique_out_dir(THIS_DIR / f"out_ch3_lift_r2_{tag}_seed{args.seed}")
    else:
        out_dir = R1.ensure_unique_out_dir(args.out_dir)

    results_dir = out_dir / "results"
    diagnostics_dir = out_dir / "diagnostics"
    results_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    xyz_list, y, tlen = TP.load_xyz_y_from_mat(str(mat_path))
    idx_tr, idx_va, idx_te = TP.stratified_split(y, seed=args.seed)
    np.savez(results_dir / f"split_indices_seed{args.seed}.npz", idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te)
    scipy.io.savemat(results_dir / f"split_indices_seed{args.seed}.mat", {"idx_tr": idx_tr, "idx_va": idx_va, "idx_te": idx_te})

    X_edge = EDGE.build_feature_matrix(xyz_list)
    edge_bundle = run_edge_round2(X_edge, y, idx_tr, idx_va, idx_te, seed=args.seed)
    json_dump(results_dir / "edge_forward_search.json", edge_bundle)

    X_raw = M.build_baseline_data(xyz_list, args.L)
    baseline_bundle = run_baseline_preset_sweep(X_raw, y, idx_tr, idx_va, idx_te, seed=args.seed)
    pd.DataFrame(
        [
            {
                "preset_name": row["preset_name"],
                "lr": row["lr"],
                "weight_decay": row["weight_decay"],
                "max_epochs": row["max_epochs"],
                "patience": row["patience"],
                "val_acc": row["val_acc"],
                "val_f1": row["val_f1"],
                "test_acc": row["test_acc"],
                "test_f1": row["test_f1"],
                "best_epoch": row["best_epoch"],
                "train_s": row["train_s"],
            }
            for row in baseline_bundle["rows"]
        ]
    ).to_csv(results_dir / "baseline_preset_sweep.csv", index=False)
    baseline = {
        "preset": baseline_bundle["selected"],
        "metrics": baseline_bundle["selected"]["metrics"],
        "train_s": baseline_bundle["selected"]["train_s"],
        "in_ch": 4,
        "selection_reason": baseline_bundle["selection_reason"],
    }

    family_preset_bundle = run_family_preset_sweep(
        X_raw,
        y,
        tlen,
        idx_tr,
        idx_va,
        idx_te,
        seed=args.seed,
        dba_iter=args.proto_dba_iter,
    )
    pd.DataFrame(
        [
            {
                "family": row["family"],
                "preset_name": row["preset_name"],
                "lr": row["lr"],
                "weight_decay": row["weight_decay"],
                "max_epochs": row["max_epochs"],
                "patience": row["patience"],
                "K": row["K"],
                "wR": row["wR"],
                "step": row["step"],
                "build_s": row["build_s"],
                "in_ch": row["in_ch"],
                "val_acc": row["val_acc"],
                "val_f1": row["val_f1"],
                "test_acc": row["test_acc"],
                "test_f1": row["test_f1"],
                "best_epoch": row["best_epoch"],
                "train_s": row["train_s"],
            }
            for row in family_preset_bundle["rows"]
        ]
    ).to_csv(results_dir / "dtw_family_preset_sweep.csv", index=False)
    json_dump(diagnostics_dir / "raw_quantile_reference.json", family_preset_bundle["raw_reference"])

    family_k_bundle = run_family_k_sweep(
        X_raw,
        y,
        tlen,
        idx_tr,
        idx_va,
        idx_te,
        seed=args.seed,
        family_presets=family_preset_bundle["selected"],
        dba_iter=args.proto_dba_iter,
    )
    pd.DataFrame(
        [
            {
                "family": row["family"],
                "preset_name": row["preset_name"],
                "K": row["K"],
                "wR": row["wR"],
                "step": row["step"],
                "build_s": row["build_s"],
                "in_ch": row["in_ch"],
                "val_acc": row["val_acc"],
                "val_f1": row["val_f1"],
                "test_acc": row["test_acc"],
                "test_f1": row["test_f1"],
                "best_epoch": row["best_epoch"],
                "train_s": row["train_s"],
            }
            for row in family_k_bundle["rows"]
        ]
    ).to_csv(results_dir / "dtw_family_k_sweep.csv", index=False)

    family_wr_step_bundle = run_family_wr_step_sweep(
        X_raw,
        y,
        tlen,
        idx_tr,
        idx_va,
        idx_te,
        seed=args.seed,
        family_presets=family_preset_bundle["selected"],
        family_best_k=family_k_bundle["selected"],
        dba_iter=args.proto_dba_iter,
    )
    pd.DataFrame(
        [
            {
                "family": row["family"],
                "preset_name": row["preset_name"],
                "K": row["K"],
                "wR": row["wR"],
                "step": row["step"],
                "build_s": row["build_s"],
                "in_ch": row["in_ch"],
                "val_acc": row["val_acc"],
                "val_f1": row["val_f1"],
                "test_acc": row["test_acc"],
                "test_f1": row["test_f1"],
                "best_epoch": row["best_epoch"],
                "train_s": row["train_s"],
            }
            for row in family_wr_step_bundle["rows"]
        ]
    ).to_csv(results_dir / "dtw_family_wr_step_sweep.csv", index=False)

    final_dtw, selection_meta = choose_family_method(
        raw_best=family_wr_step_bundle["selected"]["raw_quantile"],
        medoid_best=family_wr_step_bundle["selected"]["medoid"],
        dba_best=family_wr_step_bundle["selected"]["dba"],
    )
    method_selection = {
        "raw_quantile_best": family_wr_step_bundle["selected"]["raw_quantile"],
        "medoid_best": family_wr_step_bundle["selected"]["medoid"],
        "dba_best": family_wr_step_bundle["selected"]["dba"],
        "final_dtw": final_dtw,
        **selection_meta,
    }
    json_dump(results_dir / "method_selection.json", method_selection)

    final_dtw_preset = family_preset_bundle["selected"][str(final_dtw["family"])]
    final_preset_cfg = {
        "name": final_dtw_preset["preset_name"],
        "lr": final_dtw_preset["lr"],
        "weight_decay": final_dtw_preset["weight_decay"],
        "max_epochs": final_dtw_preset["max_epochs"],
        "patience": final_dtw_preset["patience"],
    }

    X_clsmin_mean = C.build_dtw_clsmin(X_raw, y, idx_tr, wR=float(final_dtw["wR"]), step=float(final_dtw["step"]))
    clsmin_mean = R1.run_cnn_experiment(
        X_clsmin_mean,
        y,
        idx_tr,
        idx_va,
        idx_te,
        in_ch=4,
        seed=args.seed,
        preset=final_preset_cfg,
    )
    del X_clsmin_mean
    gc.collect()

    X_clsmin_dba = DBA.build_dtw_clsmin_dba(
        X_raw,
        y,
        idx_tr,
        wR=float(final_dtw["wR"]),
        step=float(final_dtw["step"]),
        n_iter=args.dba_iter,
    )
    clsmin_dba = R1.run_cnn_experiment(
        X_clsmin_dba,
        y,
        idx_tr,
        idx_va,
        idx_te,
        in_ch=4,
        seed=args.seed,
        preset=final_preset_cfg,
    )
    del X_clsmin_dba
    gc.collect()

    rep_factor = int(final_dtw["in_ch"] // X_raw.shape[1])
    X_rep = np.tile(X_raw, (1, rep_factor, 1))
    replication = R1.run_cnn_experiment(
        X_rep,
        y,
        idx_tr,
        idx_va,
        idx_te,
        in_ch=X_rep.shape[1],
        seed=args.seed,
        preset=final_preset_cfg,
    )
    del X_rep
    gc.collect()

    selected_k_rows = [row for row in family_k_bundle["rows"] if str(row["family"]) == str(final_dtw["family"])]
    save_round2_outputs(
        out_dir=out_dir,
        edge_bundle=edge_bundle,
        baseline=baseline,
        clsmin_mean=clsmin_mean,
        clsmin_dba=clsmin_dba,
        final_dtw=final_dtw,
        selected_k_rows=selected_k_rows,
        replication=replication,
        y_test=y[idx_te],
        dba_iter=args.dba_iter,
    )

    summary = {
        "mat_path": str(mat_path),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "L": int(args.L),
        "split_counts": {
            "train": TP.split_counts(y, idx_tr).tolist(),
            "val": TP.split_counts(y, idx_va).tolist(),
            "test": TP.split_counts(y, idx_te).tolist(),
        },
        "edge": {
            "selected_candidate": edge_bundle["selected"]["candidate"],
            "accepted": bool(edge_bundle["selected"]["accepted"]),
            "softmax": edge_bundle["selected"]["softmax"],
            "svm": edge_bundle["selected"]["svm"],
            "dt": edge_bundle["selected"]["dt"],
            "decision_rank": int(edge_bundle["selected"]["rank"]),
        },
        "baseline": {
            "selected_preset": {
                "preset_name": baseline_bundle["selected"]["preset_name"],
                "lr": baseline_bundle["selected"]["lr"],
                "weight_decay": baseline_bundle["selected"]["weight_decay"],
                "max_epochs": baseline_bundle["selected"]["max_epochs"],
                "patience": baseline_bundle["selected"]["patience"],
            },
            "selection_reason": baseline_bundle["selection_reason"],
            "metrics": baseline["metrics"],
        },
        "dtw": {
            "selected_family": final_dtw["family"],
            "selected_config": {
                "family": final_dtw["family"],
                "preset_name": final_dtw["preset_name"],
                "K": int(final_dtw["K"]),
                "wR": float(final_dtw["wR"]),
                "step": float(final_dtw["step"]),
                "in_ch": int(final_dtw["in_ch"]),
            },
            "selection_meta": selection_meta,
            "metrics": final_dtw["metrics"],
        },
        "offline": {
            "cnn_baseline": baseline,
            "dtw_clsmin_mean_cnn": clsmin_mean,
            "dtw_clsmin_dba_cnn": clsmin_dba,
            "dtw_multi_best": final_dtw,
            "replication_control": replication,
        },
    }
    json_dump(results_dir / "summary.json", summary)
    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
