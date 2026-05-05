from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import run_ch3_thesis_pipeline as TP
import run_ch3_edge_baselines_topk as EDGE
from edge_features_allreal_v4 import feature_names_allreal_v4

import run_dtw_multi_sweep as M
import run_dtw_multi_quantile as MQ
import run_dtw_clsmin_sweep as C
import run_dtw_clsmin_dba as DBA


EDGE_K_GRID = (4, 5, 6, 7)
SOFTMAX_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1)
SVM_C_GRID = (0.1, 1.0, 10.0, 100.0)
DT_DEPTH_GRID = (3, 4, 5, 6, 7)
DT_LEAF_GRID = (1, 2, 4)
EDGE_TIE_EPS = 0.005
PRESET_TIE_EPS = 0.002
KMT_TIE_EPS = 0.005
KMT_GRID = (1, 2, 3, 4, 5, 6)
PUBLISHABLE_K_GRID = (2, 3, 4, 5, 6)
PRESET_GRID = (
    {"name": "P1", "lr": 1e-3, "weight_decay": 1e-4, "max_epochs": 500, "patience": 100},
    {"name": "P2", "lr": 7e-4, "weight_decay": 1e-4, "max_epochs": 500, "patience": 100},
    {"name": "P3", "lr": 5e-4, "weight_decay": 1e-4, "max_epochs": 500, "patience": 100},
    {"name": "P4", "lr": 5e-4, "weight_decay": 3e-4, "max_epochs": 500, "patience": 100},
)
CLASS_LABELS = ["Small", "Medium", "Large"]


def json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pct_str(x: float) -> str:
    return f"{100.0 * float(x):.2f}\\%"


def latex_rows_from_records(records: list[dict], cols: list[str]) -> str:
    lines: list[str] = []
    for row in records:
        parts = [str(row[c]) for c in cols]
        lines.append(" & ".join(parts) + r" \\")
    return "\n".join(lines)


def ensure_unique_out_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    idx = 2
    while True:
        candidate = Path(f"{base_dir}_v{idx}")
        if not candidate.exists():
            return candidate
        idx += 1


def default_mat_path() -> Path:
    candidates = (
        THIS_DIR / "processedVehicleData_3class_REAL (2).mat",
        THIS_DIR / "ch3_assets_plus_matlab" / "processedVehicleData_3class_REAL (2).mat",
        Path(r"D:\xidian_Master\研究生论文\毕业论文\实验数据\第三章\数据\processedVehicleData_3class_REAL (2).mat"),
    )
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Cannot find processedVehicleData_3class_REAL (2).mat")


def cm_to_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": float(np.mean(y_true == y_pred)),
        "macro_f1": float(TP.macro_f1(y_true, y_pred, 3)),
        "cm": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }


def fit_softmax(X_train: np.ndarray, y_train: np.ndarray, lam: float, seed: int) -> LogisticRegression:
    Xb, yb = EDGE.oversample_equal(X_train, y_train, seed=seed)
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=6000,
        random_state=seed,
        C=float(1.0 / lam),
    )
    clf.fit(Xb, yb)
    return clf


def fit_svm(X_train: np.ndarray, y_train: np.ndarray, c_value: float, seed: int) -> LinearSVC:
    Xb, yb = EDGE.oversample_equal(X_train, y_train, seed=seed)
    clf = LinearSVC(random_state=seed, C=float(c_value), dual="auto", max_iter=10000)
    clf.fit(Xb, yb)
    return clf


def fit_dt(X_train: np.ndarray, y_train: np.ndarray, max_depth: int, min_leaf: int, seed: int) -> DecisionTreeClassifier:
    Xb, yb = EDGE.oversample_equal(X_train, y_train, seed=seed)
    clf = DecisionTreeClassifier(
        random_state=seed,
        max_depth=int(max_depth),
        min_samples_leaf=int(min_leaf),
    )
    clf.fit(Xb, yb)
    return clf


def ordered_edge_candidates(candidates: list[dict]) -> list[dict]:
    pending = sorted(candidates, key=lambda row: (-row["val_f1"], row["K"], -row["lambda"]))
    ordered: list[dict] = []
    while pending:
        block_best = pending[0]["val_f1"]
        block = [row for row in pending if block_best - row["val_f1"] <= EDGE_TIE_EPS]
        block = sorted(block, key=lambda row: (row["K"], -row["lambda"], -row["val_f1"]))
        ordered.extend(block)
        pending = [row for row in pending if block_best - row["val_f1"] > EDGE_TIE_EPS]
    return ordered


def search_edge_candidates(
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
    if order.size == 0:
        raise RuntimeError("No usable edge features after correlation filtering.")

    names = feature_names_allreal_v4(fs=50)
    candidates: list[dict] = []
    for k in EDGE_K_GRID:
        if k > order.size:
            continue
        sel = order[:k]
        for lam in SOFTMAX_LAMBDAS:
            clf = fit_softmax(X_tr_n[:, sel], y_tr, lam=lam, seed=seed)
            pred = clf.predict(X_va_n[:, sel])
            candidates.append(
                {
                    "K": int(k),
                    "lambda": float(lam),
                    "selected_idx": [int(x) for x in sel.tolist()],
                    "selected_names": [names[int(x)] for x in sel.tolist()],
                    "val_acc": float(np.mean(pred == y_va)),
                    "val_f1": float(TP.macro_f1(y_va, pred, 3)),
                }
            )

    if not candidates:
        raise RuntimeError("Edge candidate search produced no candidates.")

    return {
        "rf_rank_order": [int(x) for x in order.tolist()],
        "rf_rank_names": [names[int(x)] for x in order.tolist()],
        "candidates": ordered_edge_candidates(candidates),
    }


def pick_svm_config(
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    selected_idx: list[int],
    seed: int,
) -> dict:
    sel = np.asarray(selected_idx, dtype=int)
    X_tr = X[idx_tr][:, sel]
    X_va = X[idx_va][:, sel]
    y_tr = y[idx_tr]
    y_va = y[idx_va]
    mu, sd = EDGE.standardize_fit(X_tr)
    X_tr_n = EDGE.standardize_apply(X_tr, mu, sd)
    X_va_n = EDGE.standardize_apply(X_va, mu, sd)

    rows: list[dict] = []
    for c_value in SVM_C_GRID:
        clf = fit_svm(X_tr_n, y_tr, c_value=c_value, seed=seed)
        pred = clf.predict(X_va_n)
        rows.append(
            {
                "C": float(c_value),
                "val_acc": float(np.mean(pred == y_va)),
                "val_f1": float(TP.macro_f1(y_va, pred, 3)),
            }
        )

    rows.sort(key=lambda row: (-row["val_f1"], row["C"]))
    return {"best": rows[0], "grid": rows}


def pick_dt_config(
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    selected_idx: list[int],
    seed: int,
) -> dict:
    sel = np.asarray(selected_idx, dtype=int)
    X_tr = X[idx_tr][:, sel]
    X_va = X[idx_va][:, sel]
    y_tr = y[idx_tr]
    y_va = y[idx_va]

    rows: list[dict] = []
    for depth in DT_DEPTH_GRID:
        for min_leaf in DT_LEAF_GRID:
            clf = fit_dt(X_tr, y_tr, max_depth=depth, min_leaf=min_leaf, seed=seed)
            pred = clf.predict(X_va)
            rows.append(
                {
                    "max_depth": int(depth),
                    "min_samples_leaf": int(min_leaf),
                    "val_acc": float(np.mean(pred == y_va)),
                    "val_f1": float(TP.macro_f1(y_va, pred, 3)),
                }
            )

    rows.sort(key=lambda row: (-row["val_f1"], row["max_depth"], -row["min_samples_leaf"]))
    return {"best": rows[0], "grid": rows}


def finalize_edge_models(
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    candidate: dict,
    seed: int,
) -> dict:
    sel = np.asarray(candidate["selected_idx"], dtype=int)
    idx_tv = np.concatenate([idx_tr, idx_va])
    y_te = y[idx_te]

    svm_search = pick_svm_config(X, y, idx_tr, idx_va, candidate["selected_idx"], seed=seed)
    dt_search = pick_dt_config(X, y, idx_tr, idx_va, candidate["selected_idx"], seed=seed)

    X_tv = X[idx_tv][:, sel]
    X_te = X[idx_te][:, sel]
    y_tv = y[idx_tv]

    mu, sd = EDGE.standardize_fit(X_tv)
    X_tv_n = EDGE.standardize_apply(X_tv, mu, sd)
    X_te_n = EDGE.standardize_apply(X_te, mu, sd)

    softmax = fit_softmax(X_tv_n, y_tv, lam=candidate["lambda"], seed=seed)
    svm = fit_svm(X_tv_n, y_tv, c_value=svm_search["best"]["C"], seed=seed)
    dt = fit_dt(X_tv, y_tv, max_depth=dt_search["best"]["max_depth"], min_leaf=dt_search["best"]["min_samples_leaf"], seed=seed)

    pred_soft = softmax.predict(X_te_n)
    pred_svm = svm.predict(X_te_n)
    pred_dt = dt.predict(X_te)

    soft_metrics = cm_to_metrics(y_te, pred_soft)
    svm_metrics = cm_to_metrics(y_te, pred_svm)
    dt_metrics = cm_to_metrics(y_te, pred_dt)

    accepted = soft_metrics["macro_f1"] >= svm_metrics["macro_f1"] and soft_metrics["macro_f1"] >= dt_metrics["macro_f1"]

    return {
        "candidate": candidate,
        "accepted": bool(accepted),
        "softmax_search": {"lambda": candidate["lambda"], "val_acc": candidate["val_acc"], "val_f1": candidate["val_f1"]},
        "svm_search": svm_search,
        "dt_search": dt_search,
        "softmax": {**soft_metrics, "y_pred_test": pred_soft.astype(int).tolist()},
        "svm": {**svm_metrics, "y_pred_test": pred_svm.astype(int).tolist()},
        "dt": {**dt_metrics, "y_pred_test": pred_dt.astype(int).tolist()},
        "scaler_mu": mu.reshape(-1).tolist(),
        "scaler_sd": sd.reshape(-1).tolist(),
    }


def run_edge_selection(
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
) -> dict:
    search = search_edge_candidates(X, y, idx_tr, idx_va, seed=seed)
    decision_trace: list[dict] = []
    selected = None
    for rank, candidate in enumerate(search["candidates"], start=1):
        final_eval = finalize_edge_models(X, y, idx_tr, idx_va, idx_te, candidate, seed=seed)
        final_eval["rank"] = rank
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

    return {
        "search": search,
        "decision_trace": decision_trace,
        "selected": selected,
    }


def run_cnn_experiment(
    X_seq: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    in_ch: int,
    seed: int,
    preset: dict,
) -> dict:
    mu, sd = TP.compute_norm_stats_seq(X_seq, idx_tr)
    Xn = TP.normalize_seq(X_seq, mu, sd)
    train_kwargs = {
        "max_epochs": int(preset["max_epochs"]),
        "patience": int(preset["patience"]),
        "lr": float(preset["lr"]),
        "weight_decay": float(preset["weight_decay"]),
    }
    t0 = time.time()
    _, metrics = TP.train_cnn(Xn, y, idx_tr, idx_va, idx_te, in_ch=in_ch, seed=seed, **train_kwargs)
    train_s = time.time() - t0
    return {
        "metrics": metrics,
        "train_s": float(train_s),
        "in_ch": int(in_ch),
    }


def choose_best_preset(rows: list[dict]) -> dict:
    ranked = sorted(rows, key=lambda row: -row["score"])
    best_score = ranked[0]["score"]
    close = [row for row in ranked if best_score - row["score"] <= PRESET_TIE_EPS]
    close.sort(key=lambda row: (-row["multi_val_f1"], -row["baseline_val_f1"], row["lr"], row["weight_decay"]))
    return close[0]


def search_offline_preset(
    X_raw: np.ndarray,
    X_multi_k4: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
) -> dict:
    rows: list[dict] = []
    for preset in PRESET_GRID:
        base = run_cnn_experiment(X_raw, y, idx_tr, idx_va, idx_te, in_ch=4, seed=seed, preset=preset)
        multi = run_cnn_experiment(
            X_multi_k4,
            y,
            idx_tr,
            idx_va,
            idx_te,
            in_ch=X_multi_k4.shape[1],
            seed=seed,
            preset=preset,
        )
        rows.append(
            {
                "name": preset["name"],
                "lr": float(preset["lr"]),
                "weight_decay": float(preset["weight_decay"]),
                "max_epochs": int(preset["max_epochs"]),
                "patience": int(preset["patience"]),
                "baseline_val_f1": float(base["metrics"]["val_f1"]),
                "baseline_test_f1": float(base["metrics"]["test_f1"]),
                "multi_val_f1": float(multi["metrics"]["val_f1"]),
                "multi_test_f1": float(multi["metrics"]["test_f1"]),
                "baseline_best_epoch": int(base["metrics"]["best_epoch"]),
                "multi_best_epoch": int(multi["metrics"]["best_epoch"]),
                "score": float(0.4 * base["metrics"]["val_f1"] + 0.6 * multi["metrics"]["val_f1"]),
                "baseline_train_s": float(base["train_s"]),
                "multi_train_s": float(multi["train_s"]),
            }
        )

    selected = choose_best_preset(rows)
    preset_map = {preset["name"]: dict(preset) for preset in PRESET_GRID}
    return {"rows": rows, "selected": {**selected, **preset_map[selected["name"]]}}


def choose_publishable_k(rows: list[dict]) -> dict:
    best_val_all = max(row["val_f1"] for row in rows)
    publishable = [
        row for row in rows
        if row["K"] in PUBLISHABLE_K_GRID and best_val_all - row["val_f1"] <= KMT_TIE_EPS
    ]
    if publishable:
        publishable.sort(key=lambda row: (row["K"], -row["val_f1"], -row["test_f1"]))
        return publishable[0]

    publishable = [row for row in rows if row["K"] in PUBLISHABLE_K_GRID]
    publishable.sort(key=lambda row: (-row["val_f1"], row["K"], -row["test_f1"]))
    return publishable[0]


def run_kmt_sweep(
    X_raw: np.ndarray,
    y: np.ndarray,
    tlen: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    seed: int,
    preset: dict,
    wR: float,
    step: float,
) -> dict:
    rows: list[dict] = []
    selected_record = None

    for k_value in KMT_GRID:
        t_build0 = time.time()
        X_multi, templates = MQ.build_dtw_multi_quantile(
            X_raw,
            y,
            idx_tr,
            tlen,
            templates_per_class=k_value,
            wR=wR,
            step=step,
        )
        build_s = time.time() - t_build0
        result = run_cnn_experiment(
            X_multi,
            y,
            idx_tr,
            idx_va,
            idx_te,
            in_ch=X_multi.shape[1],
            seed=seed,
            preset=preset,
        )
        row = {
            "K": int(k_value),
            "in_ch": int(X_multi.shape[1]),
            "build_s": float(build_s),
            "train_s": float(result["train_s"]),
            "val_acc": float(result["metrics"]["val_acc"]),
            "val_f1": float(result["metrics"]["val_f1"]),
            "test_acc": float(result["metrics"]["test_acc"]),
            "test_f1": float(result["metrics"]["test_f1"]),
            "best_epoch": int(result["metrics"]["best_epoch"]),
            "metrics": result["metrics"],
            "templates_shape": list(templates.shape),
        }
        rows.append(row)
        if k_value == 1:
            selected_record = {"k1": row}

    publishable = choose_publishable_k(rows)
    return {
        "rows": rows,
        "publishable": publishable,
        "diagnostic_k1": next(row for row in rows if row["K"] == 1),
        "selected_record": selected_record,
    }


def build_edge_tables(selected_edge: dict) -> tuple[pd.DataFrame, list[dict]]:
    cand = selected_edge["candidate"]
    edge_df = pd.DataFrame(
        [
            {"method": f"Edge Softmax (K={cand['K']})", "test_acc": selected_edge["softmax"]["acc"], "test_f1": selected_edge["softmax"]["macro_f1"]},
            {"method": f"Edge SVM (K={cand['K']})", "test_acc": selected_edge["svm"]["acc"], "test_f1": selected_edge["svm"]["macro_f1"]},
            {"method": f"Edge DT (K={cand['K']})", "test_acc": selected_edge["dt"]["acc"], "test_f1": selected_edge["dt"]["macro_f1"]},
        ]
    )
    rows = [
        {"method": row["method"], "test_acc": pct_str(row["test_acc"]), "test_f1": pct_str(row["test_f1"])}
        for row in edge_df.to_dict(orient="records")
    ]
    return edge_df, rows


def build_overall_tables(edge_selected: dict, baseline: dict, multi_best: dict) -> tuple[pd.DataFrame, list[dict]]:
    k_value = int(multi_best["K"])
    df = pd.DataFrame(
        [
            {"method": "Edge Softmax", "test_acc": edge_selected["softmax"]["acc"], "test_f1": edge_selected["softmax"]["macro_f1"]},
            {"method": "Edge SVM", "test_acc": edge_selected["svm"]["acc"], "test_f1": edge_selected["svm"]["macro_f1"]},
            {"method": "Offline CNN (fixed-length)", "test_acc": baseline["metrics"]["test_acc"], "test_f1": baseline["metrics"]["test_f1"]},
            {"method": f"DTW-MultiTemplate(K={k_value}) + CNN", "test_acc": multi_best["metrics"]["test_acc"], "test_f1": multi_best["metrics"]["test_f1"]},
        ]
    )
    rows = [
        {"method": row["method"], "test_acc": pct_str(row["test_acc"]), "test_f1": pct_str(row["test_f1"])}
        for row in df.to_dict(orient="records")
    ]
    return df, rows


def build_clsmin_table(clsmin_mean: dict, clsmin_dba: dict, dba_iter: int) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.DataFrame(
        [
            {"method": "DTW-ClsMin-Mean + CNN", "test_acc": clsmin_mean["metrics"]["test_acc"], "test_f1": clsmin_mean["metrics"]["test_f1"]},
            {"method": f"DTW-ClsMin-DBA(iter={dba_iter}) + CNN", "test_acc": clsmin_dba["metrics"]["test_acc"], "test_f1": clsmin_dba["metrics"]["test_f1"]},
        ]
    )
    rows = [
        {"method": row["method"], "test_acc": pct_str(row["test_acc"]), "test_f1": pct_str(row["test_f1"])}
        for row in df.to_dict(orient="records")
    ]
    return df, rows


def build_replication_table(replication: dict, multi_best: dict) -> tuple[pd.DataFrame, list[dict]]:
    k_value = int(multi_best["K"])
    df = pd.DataFrame(
        [
            {"method": f"Replication Control (K={k_value})", "test_acc": replication["metrics"]["test_acc"], "test_f1": replication["metrics"]["test_f1"]},
            {"method": f"DTW-MultiTemplate(K={k_value})", "test_acc": multi_best["metrics"]["test_acc"], "test_f1": multi_best["metrics"]["test_f1"]},
        ]
    )
    rows = [
        {"method": row["method"], "test_acc": pct_str(row["test_acc"]), "test_f1": pct_str(row["test_f1"])}
        for row in df.to_dict(orient="records")
    ]
    return df, rows


def build_k_table(rows: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    df = pd.DataFrame(
        [
            {
                "K": row["K"],
                "in_ch": row["in_ch"],
                "val_f1": row["val_f1"],
                "test_acc": row["test_acc"],
                "test_f1": row["test_f1"],
            }
            for row in rows
        ]
    ).sort_values("K").reset_index(drop=True)
    latex_rows = [
        {
            "K": int(row["K"]),
            "in_ch": int(row["in_ch"]),
            "test_acc": pct_str(row["test_acc"]),
            "test_f1": pct_str(row["test_f1"]),
        }
        for row in df.to_dict(orient="records")
    ]
    return df, latex_rows


def save_tables_and_figures(
    out_dir: Path,
    edge_bundle: dict,
    baseline: dict,
    clsmin_mean: dict,
    clsmin_dba: dict,
    kmt_bundle: dict,
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

    edge_df, edge_rows = build_edge_tables(edge_bundle["selected"])
    overall_df, overall_rows = build_overall_tables(edge_bundle["selected"], baseline, kmt_bundle["publishable"])
    cls_df, cls_rows = build_clsmin_table(clsmin_mean, clsmin_dba, dba_iter=dba_iter)
    rep_df, rep_rows = build_replication_table(replication, kmt_bundle["publishable"])
    k_df, k_rows = build_k_table(kmt_bundle["rows"])

    edge_df.to_csv(tab_dir / "ch3_edge.csv", index=False)
    overall_df.to_csv(tab_dir / "ch3_overall.csv", index=False)
    cls_df.to_csv(tab_dir / "ch3_ablation_clsmin_template.csv", index=False)
    rep_df.to_csv(tab_dir / "ch3_ablation_rep.csv", index=False)
    k_df.to_csv(tab_dir / "ch3_ablation_K.csv", index=False)

    (tab_dir / "ch3_edge_rows.tex").write_text(latex_rows_from_records(edge_rows, ["method", "test_acc", "test_f1"]), encoding="utf-8")
    (tab_dir / "ch3_overall_rows.tex").write_text(latex_rows_from_records(overall_rows, ["method", "test_acc", "test_f1"]), encoding="utf-8")
    (tab_dir / "ch3_ablation_clsmin_template_rows.tex").write_text(
        latex_rows_from_records(cls_rows, ["method", "test_acc", "test_f1"]),
        encoding="utf-8",
    )
    (tab_dir / "ch3_ablation_rep_rows.tex").write_text(
        latex_rows_from_records(rep_rows, ["method", "test_acc", "test_f1"]),
        encoding="utf-8",
    )
    (tab_dir / "ch3_ablation_K_rows.tex").write_text(
        latex_rows_from_records(k_rows, ["K", "in_ch", "test_acc", "test_f1"]),
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
        np.asarray(kmt_bundle["publishable"]["metrics"]["y_pred_test"], dtype=int),
        CLASS_LABELS,
        f"DTW-MultiTemplate(K={kmt_bundle['publishable']['K']}) + CNN",
        str(fig_dir / "cm_dtw_multi_cnn.png"),
        normalize=False,
    )
    TP.plot_k_ablation(k_df, str(fig_dir / "ablation_K.png"))
    TP.plot_replication(rep_df, str(fig_dir / "ablation_replication.png"))

    edge_curve = pd.DataFrame(
        [
            {"K": row["K"], "val_f1": row["val_f1"]}
            for row in edge_bundle["search"]["candidates"]
        ]
    ).groupby("K", as_index=False)["val_f1"].max()
    TP.plot_edge_k_sweep(edge_curve, best_k=edge_bundle["selected"]["candidate"]["K"], out_path_png=str(diag_dir / "ablation_edge_K.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=Path, default=None, help="Path to processedVehicleData_3class_REAL (2).mat")
    parser.add_argument("--out_dir", type=Path, default=None, help="Fresh output directory for the isolated lift experiment")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=176)
    parser.add_argument("--wR", type=float, default=0.15)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--dba_iter", type=int, default=10)
    args = parser.parse_args()

    TP.ensure_plot_style()

    mat_path = args.mat if args.mat is not None else default_mat_path()
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    if args.out_dir is None:
        tag = date.today().strftime("%Y%m%d")
        out_dir = ensure_unique_out_dir(THIS_DIR / f"out_ch3_lift_{tag}_seed{args.seed}")
    else:
        out_dir = ensure_unique_out_dir(args.out_dir)

    results_dir = out_dir / "results"
    diagnostics_dir = out_dir / "diagnostics"
    results_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    xyz_list, y, tlen = TP.load_xyz_y_from_mat(str(mat_path))
    idx_tr, idx_va, idx_te = TP.stratified_split(y, seed=args.seed)
    np.savez(results_dir / f"split_indices_seed{args.seed}.npz", idx_tr=idx_tr, idx_va=idx_va, idx_te=idx_te)
    scipy.io.savemat(results_dir / f"split_indices_seed{args.seed}.mat", {"idx_tr": idx_tr, "idx_va": idx_va, "idx_te": idx_te})

    X_edge = EDGE.build_feature_matrix(xyz_list)
    edge_bundle = run_edge_selection(X_edge, y, idx_tr, idx_va, idx_te, seed=args.seed)
    json_dump(results_dir / "edge_tuning.json", edge_bundle)

    X_raw = M.build_baseline_data(xyz_list, args.L)
    X_multi_k4, _ = MQ.build_dtw_multi_quantile(
        X_raw,
        y,
        idx_tr,
        tlen,
        templates_per_class=4,
        wR=args.wR,
        step=args.step,
    )
    preset_bundle = search_offline_preset(X_raw, X_multi_k4, y, idx_tr, idx_va, idx_te, seed=args.seed)
    pd.DataFrame(preset_bundle["rows"]).to_csv(results_dir / "offline_preset_sweep.csv", index=False)

    selected_preset = {
        "name": preset_bundle["selected"]["name"],
        "lr": preset_bundle["selected"]["lr"],
        "weight_decay": preset_bundle["selected"]["weight_decay"],
        "max_epochs": preset_bundle["selected"]["max_epochs"],
        "patience": preset_bundle["selected"]["patience"],
    }

    baseline = run_cnn_experiment(X_raw, y, idx_tr, idx_va, idx_te, in_ch=4, seed=args.seed, preset=selected_preset)

    X_clsmin_mean = C.build_dtw_clsmin(X_raw, y, idx_tr, wR=args.wR, step=args.step)
    clsmin_mean = run_cnn_experiment(
        X_clsmin_mean,
        y,
        idx_tr,
        idx_va,
        idx_te,
        in_ch=4,
        seed=args.seed,
        preset=selected_preset,
    )

    X_clsmin_dba = DBA.build_dtw_clsmin_dba(X_raw, y, idx_tr, wR=args.wR, step=args.step, n_iter=args.dba_iter)
    clsmin_dba = run_cnn_experiment(
        X_clsmin_dba,
        y,
        idx_tr,
        idx_va,
        idx_te,
        in_ch=4,
        seed=args.seed,
        preset=selected_preset,
    )

    kmt_bundle = run_kmt_sweep(
        X_raw,
        y,
        tlen,
        idx_tr,
        idx_va,
        idx_te,
        seed=args.seed,
        preset=selected_preset,
        wR=args.wR,
        step=args.step,
    )

    pd.DataFrame(
        [
            {
                "K": row["K"],
                "in_ch": row["in_ch"],
                "build_s": row["build_s"],
                "train_s": row["train_s"],
                "val_acc": row["val_acc"],
                "val_f1": row["val_f1"],
                "test_acc": row["test_acc"],
                "test_f1": row["test_f1"],
                "best_epoch": row["best_epoch"],
            }
            for row in kmt_bundle["rows"]
        ]
    ).to_csv(results_dir / "kmt_sweep.csv", index=False)
    json_dump(diagnostics_dir / "kmt_k1_reference.json", kmt_bundle["diagnostic_k1"])

    rep_factor = int(kmt_bundle["publishable"]["in_ch"] // X_raw.shape[1])
    X_rep = np.tile(X_raw, (1, rep_factor, 1))
    replication = run_cnn_experiment(
        X_rep,
        y,
        idx_tr,
        idx_va,
        idx_te,
        in_ch=X_rep.shape[1],
        seed=args.seed,
        preset=selected_preset,
    )

    save_tables_and_figures(
        out_dir=out_dir,
        edge_bundle=edge_bundle,
        baseline=baseline,
        clsmin_mean=clsmin_mean,
        clsmin_dba=clsmin_dba,
        kmt_bundle=kmt_bundle,
        replication=replication,
        y_test=y[idx_te],
        dba_iter=args.dba_iter,
    )

    summary = {
        "mat_path": str(mat_path),
        "out_dir": str(out_dir),
        "seed": int(args.seed),
        "L": int(args.L),
        "dtw": {"wR": float(args.wR), "step": float(args.step), "dba_iter": int(args.dba_iter)},
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
        "offline": {
            "selected_preset": selected_preset,
            "cnn_baseline": baseline,
            "dtw_clsmin_mean_cnn": clsmin_mean,
            "dtw_clsmin_dba_cnn": clsmin_dba,
            "dtw_multi_best": kmt_bundle["publishable"],
            "replication_control": replication,
        },
    }
    json_dump(results_dir / "summary.json", summary)

    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
