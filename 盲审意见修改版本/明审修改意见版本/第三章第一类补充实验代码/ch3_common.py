# -*- coding: utf-8 -*-
"""Shared utilities for Chapter 3 supplementary classification experiments.

The functions in this file intentionally keep data preparation, splitting,
normalization, oversampling, training, and metrics identical across CNN and
LSTM baselines. Model architecture should be the only core difference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import scipy.io
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


CLASS_NAMES = ("small", "medium", "large")

MAIN_DATA_DEFAULT = (
    r"D:\xidian_Master\研究生论文\毕业论文\xduts-main\非核心文件\绘图\第三章"
    r"\20260304\processedVehicleData_3class_REAL (2).mat"
)

SOURCE_GROUP_DATA_DEFAULT = MAIN_DATA_DEFAULT

TRAIN_PRESETS = {
    "legacy_fast": dict(lr=1e-3, weight_decay=0.0, max_epochs=30, patience=6),
    "paper_fair": dict(lr=5e-4, weight_decay=1e-4, max_epochs=200, patience=30),
}


@dataclass
class DatasetBundle:
    xyz_list: list[np.ndarray]
    y: np.ndarray
    target_length: np.ndarray
    source_index: np.ndarray | None


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_csv(path: str | Path, rows: Iterable[dict], fieldnames: list[str]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    rows = list(rows)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _unwrap_single_cell(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.item()
    return x


def _class_cell(mat_value: np.ndarray, class_id: int):
    cell = mat_value[0, class_id]
    return _unwrap_single_cell(cell)


def _to_1d_numeric(x) -> np.ndarray:
    x = _unwrap_single_cell(x)
    return np.asarray(x).reshape(-1)


def _fix_xyz_shape(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D event array, got shape {arr.shape}")
    if arr.shape[1] == 3:
        return arr
    if arr.shape[0] == 3:
        return arr.T
    raise ValueError(f"Expected event array with 3 axes, got shape {arr.shape}")


def load_dataset(
    mat_path: str | Path,
    require_source: bool = False,
    require_grouped_source: bool = False,
) -> DatasetBundle:
    """Load ProcessedData/targetLength and optional sourceIndex from a .mat file."""

    mat_path = Path(mat_path)
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    if "ProcessedData" not in mat or "targetLength" not in mat:
        raise KeyError("MAT file must contain ProcessedData and targetLength.")

    has_source = "sourceIndex" in mat
    if require_source and not has_source:
        raise KeyError(
            "sourceIndex is required for source-group validation. "
            "Use a REAL data file that contains sourceIndex."
        )

    xyz_list: list[np.ndarray] = []
    y_list: list[int] = []
    tlen_list: list[int] = []
    source_list: list[int] = []

    pd_cell = mat["ProcessedData"]
    tl_cell = mat["targetLength"]
    src_cell = mat.get("sourceIndex")

    for class_id in range(3):
        events = _class_cell(pd_cell, class_id)
        lengths = _to_1d_numeric(tl_cell[0, class_id]).astype(int)
        sources = None
        if src_cell is not None:
            sources = _to_1d_numeric(src_cell[0, class_id]).astype(int)

        n_events = events.shape[1] if getattr(events, "ndim", 0) == 2 else len(events)
        if lengths.size != n_events:
            raise ValueError(
                f"class {class_id}: targetLength size {lengths.size} "
                f"does not match event count {n_events}"
            )
        if sources is not None and sources.size != n_events:
            raise ValueError(
                f"class {class_id}: sourceIndex size {sources.size} "
                f"does not match event count {n_events}"
            )

        for i in range(n_events):
            raw_event = events[0, i] if getattr(events, "ndim", 0) == 2 else events[i]
            arr = _fix_xyz_shape(raw_event)
            valid_len = int(lengths[i])
            valid_len = max(1, min(valid_len, arr.shape[0]))
            xyz_list.append(arr[:valid_len, :])
            y_list.append(class_id)
            tlen_list.append(valid_len)
            if sources is not None:
                source_list.append(int(sources[i]))

    y = np.asarray(y_list, dtype=np.int64)
    tlen = np.asarray(tlen_list, dtype=np.int64)
    source = np.asarray(source_list, dtype=np.int64) if source_list else None

    if require_grouped_source and source is not None and not has_repeated_source_groups(y, source):
        raise ValueError(
            "sourceIndex exists but every class/source group has only one sample. "
            "This is not useful for strict source-group validation. Use a data "
            "file whose sourceIndex represents repeated same-source groups."
        )

    return DatasetBundle(
        xyz_list=xyz_list,
        y=y,
        target_length=tlen,
        source_index=source,
    )


def has_repeated_source_groups(y: np.ndarray, source: np.ndarray) -> bool:
    for class_id in np.unique(y):
        src = source[y == class_id]
        _, counts = np.unique(src, return_counts=True)
        if np.any(counts > 1):
            return True
    return False


def resample_linear(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape[0] == length:
        return x.copy()
    if x.shape[0] == 1:
        return np.repeat(x, repeats=length, axis=0)

    old_t = np.linspace(0.0, 1.0, x.shape[0])
    new_t = np.linspace(0.0, 1.0, length)
    out = np.empty((length, x.shape[1]), dtype=np.float32)
    for dim in range(x.shape[1]):
        out[:, dim] = np.interp(new_t, old_t, x[:, dim]).astype(np.float32)
    return out


def make_four_channel(xyz: np.ndarray, length: int) -> np.ndarray:
    resized = resample_linear(xyz, length)
    mag = np.sqrt(np.sum(resized * resized, axis=1, dtype=np.float32))
    out = np.empty((length, 4), dtype=np.float32)
    out[:, 0:3] = resized
    out[:, 3] = mag
    return out.T


def build_sequence_tensor(xyz_list: list[np.ndarray], length: int = 176) -> np.ndarray:
    X = np.empty((len(xyz_list), 4, length), dtype=np.float32)
    for i, xyz in enumerate(xyz_list):
        X[i] = make_four_channel(xyz, length)
    return X


def make_four_channel_variable(xyz: np.ndarray) -> np.ndarray:
    """Build an unresampled 4-channel event sequence as (4, T)."""

    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape (T,3), got {xyz.shape}")
    mag = np.sqrt(np.sum(xyz * xyz, axis=1, dtype=np.float32))
    out = np.empty((4, xyz.shape[0]), dtype=np.float32)
    out[0:3, :] = xyz.T
    out[3, :] = mag
    return out


def build_masked_sequence_tensor(
    xyz_list: list[np.ndarray],
    max_length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad variable-length events and append a binary mask channel.

    Returns
    -------
    X: np.ndarray
        Shape (N, 5, Lmax). Channels 0:4 are x/y/z/magnitude and channel 4 is
        the valid-position mask.
    lengths: np.ndarray
        Valid length after optional truncation.
    """

    if not xyz_list:
        raise ValueError("xyz_list is empty.")
    raw_lengths = np.asarray([int(x.shape[0]) for x in xyz_list], dtype=np.int64)
    if np.any(raw_lengths <= 0):
        raise ValueError("All events must contain at least one valid sample.")
    Lmax = int(max_length) if max_length is not None else int(raw_lengths.max())
    if Lmax <= 0:
        raise ValueError(f"max_length must be positive, got {Lmax}")

    X = np.zeros((len(xyz_list), 5, Lmax), dtype=np.float32)
    lengths = np.minimum(raw_lengths, Lmax).astype(np.int64)
    for i, xyz in enumerate(xyz_list):
        seq = make_four_channel_variable(xyz)
        n = int(lengths[i])
        X[i, 0:4, :n] = seq[:, :n]
        X[i, 4, :n] = 1.0
    return X, lengths


def compute_masked_norm_stats(X: np.ndarray, idx_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate channel normalization from valid positions in the train split."""

    data = X[idx_tr, 0:4, :]
    mask = X[idx_tr, 4:5, :]
    denom = mask.sum(axis=(0, 2), keepdims=True).astype(np.float32)
    denom[denom < 1.0] = 1.0
    mu = (data * mask).sum(axis=(0, 2), keepdims=True).astype(np.float32) / denom
    var = (((data - mu) * mask) ** 2).sum(axis=(0, 2), keepdims=True).astype(np.float32) / denom
    sd = np.sqrt(var).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def normalize_masked_with_stats(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    out = X.astype(np.float32, copy=True)
    mask = out[:, 4:5, :]
    out[:, 0:4, :] = ((out[:, 0:4, :] - mu) / sd) * mask
    out[:, 4:5, :] = mask
    return out.astype(np.float32, copy=False)


def stratified_split(
    y: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_tr: list[int] = []
    idx_va: list[int] = []
    idx_te: list[int] = []
    for class_id in np.unique(y):
        idx = np.where(y == class_id)[0]
        rng.shuffle(idx)
        n_total = len(idx)
        n_train = int(round(train_ratio * n_total))
        n_val = int(round(val_ratio * n_total))
        idx_tr.extend(idx[:n_train].tolist())
        idx_va.extend(idx[n_train:n_train + n_val].tolist())
        idx_te.extend(idx[n_train + n_val:].tolist())
    return np.asarray(idx_tr), np.asarray(idx_va), np.asarray(idx_te)


def _split_groups_by_count(
    groups: np.ndarray,
    counts: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[set[int], set[int], set[int]]:
    total = int(counts.sum())
    train_target = int(round(train_ratio * total))
    val_target = int(round(val_ratio * total))

    train_groups: list[int] = []
    val_groups: list[int] = []
    test_groups: list[int] = []
    train_count = 0
    val_count = 0

    for group, count in zip(groups.tolist(), counts.tolist()):
        if train_count < train_target:
            train_groups.append(int(group))
            train_count += int(count)
        elif val_count < val_target:
            val_groups.append(int(group))
            val_count += int(count)
        else:
            test_groups.append(int(group))

    if not val_groups and test_groups:
        val_groups.append(test_groups.pop(0))
    if not test_groups and val_groups:
        test_groups.append(val_groups.pop())

    return set(train_groups), set(val_groups), set(test_groups)


def source_group_split(
    y: np.ndarray,
    source: np.ndarray,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx_tr: list[int] = []
    idx_va: list[int] = []
    idx_te: list[int] = []

    for class_id in np.unique(y):
        cls_idx = np.where(y == class_id)[0]
        cls_source = source[cls_idx]
        groups, counts = np.unique(cls_source, return_counts=True)
        order = rng.permutation(len(groups))
        groups = groups[order]
        counts = counts[order]
        tr_groups, va_groups, te_groups = _split_groups_by_count(
            groups, counts, train_ratio=train_ratio, val_ratio=val_ratio
        )

        for idx in cls_idx.tolist():
            group = int(source[idx])
            if group in tr_groups:
                idx_tr.append(idx)
            elif group in va_groups:
                idx_va.append(idx)
            elif group in te_groups:
                idx_te.append(idx)
            else:
                raise RuntimeError(f"Unassigned source group: {group}")

    return np.asarray(idx_tr), np.asarray(idx_va), np.asarray(idx_te)


def split_counts(y: np.ndarray, indices: np.ndarray) -> list[int]:
    return np.bincount(y[indices], minlength=3).astype(int).tolist()


def split_hash(idx_tr: np.ndarray, idx_va: np.ndarray, idx_te: np.ndarray) -> str:
    h = hashlib.sha256()
    for name, arr in (("train", idx_tr), ("val", idx_va), ("test", idx_te)):
        h.update(name.encode("ascii"))
        h.update(np.asarray(arr, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def compute_norm_stats(X: np.ndarray, idx_tr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = X[idx_tr].mean(axis=(0, 2), keepdims=True).astype(np.float32)
    sd = X[idx_tr].std(axis=(0, 2), keepdims=True).astype(np.float32)
    sd[sd < 1e-6] = 1.0
    return mu, sd


def normalize_with_stats(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((X - mu) / sd).astype(np.float32)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(true), int(pred)] += 1
    return cm


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3) -> list[dict]:
    rows: list[dict] = []
    for class_id in range(n_classes):
        tp = int(np.sum((y_true == class_id) & (y_pred == class_id)))
        fp = int(np.sum((y_true != class_id) & (y_pred == class_id)))
        fn = int(np.sum((y_true == class_id) & (y_pred != class_id)))
        support = int(np.sum(y_true == class_id))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": support,
            }
        )
    return rows


def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    class_rows = per_class_metrics(y_true, y_pred, n_classes=3)
    macro_f1 = float(np.mean([row["f1"] for row in class_rows]))
    acc = float(np.mean(y_true == y_pred))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": class_rows,
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes=3).tolist(),
    }


def build_balanced_sampler(y_train: np.ndarray, seed: int) -> WeightedRandomSampler:
    counts = np.bincount(y_train, minlength=3).astype(np.float64)
    if np.any(counts == 0):
        raise ValueError(f"Cannot oversample when a class is absent: counts={counts.tolist()}")
    class_weights = counts.max() / counts
    sample_weights = class_weights[y_train]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(len(y_train)),
        replacement=True,
        generator=generator,
    )


def _make_train_loader(
    X_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    oversample: bool,
    seed: int,
) -> DataLoader:
    ds = SeqDataset(X_train, y_train)
    if oversample:
        sampler = build_balanced_sampler(y_train, seed)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)


def predict_model(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(SeqDataset(X, np.zeros(X.shape[0], dtype=np.int64)), batch_size=batch_size)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_model(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    idx_te: np.ndarray,
    train_config: dict,
    seed: int,
    batch_size: int = 64,
    oversample: bool = False,
    device_name: str = "auto",
) -> dict:
    # The caller sets the seed before model initialization. Do not reset it here;
    # the thesis pipeline lets initialization and DataLoader shuffling share the
    # same RNG stream.
    torch.set_num_threads(1)
    device = resolve_device(device_name)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    train_loader = _make_train_loader(
        X[idx_tr],
        y[idx_tr],
        batch_size=batch_size,
        oversample=oversample,
        seed=seed,
    )

    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale_epochs = 0

    for epoch in range(1, int(train_config["max_epochs"]) + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        pred_va = predict_model(model, X[idx_va], device=device)
        val_metrics = summarize_metrics(y[idx_va], pred_va)
        val_f1 = float(val_metrics["macro_f1"])

        if val_f1 > best_f1 + 1e-6:
            best_f1 = val_f1
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(train_config["patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_va = predict_model(model, X[idx_va], device=device)
    pred_te = predict_model(model, X[idx_te], device=device)
    val_metrics = summarize_metrics(y[idx_va], pred_va)
    test_metrics = summarize_metrics(y[idx_te], pred_te)
    return {
        "best_epoch": int(best_epoch),
        "val": val_metrics,
        "test": test_metrics,
        "y_pred_val": pred_va.astype(int).tolist(),
        "y_pred_test": pred_te.astype(int).tolist(),
        "device": str(device),
    }


def metric_row(
    experiment: str,
    validation_type: str,
    preset: str,
    oversample: bool,
    run_id: int,
    seed: int,
    model_name: str,
    param_count: int,
    result: dict,
) -> dict:
    return {
        "experiment": experiment,
        "validation_type": validation_type,
        "preset": preset,
        "oversample": int(bool(oversample)),
        "run_id": int(run_id),
        "seed": int(seed),
        "model": model_name,
        "param_count": int(param_count),
        "best_epoch": int(result["best_epoch"]),
        "val_acc": float(result["val"]["accuracy"]),
        "val_macro_f1": float(result["val"]["macro_f1"]),
        "test_acc": float(result["test"]["accuracy"]),
        "test_macro_f1": float(result["test"]["macro_f1"]),
    }


def aggregate_metric_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)

    out: list[dict] = []
    for model_name, model_rows in sorted(grouped.items()):
        test_acc = np.asarray([row["test_acc"] for row in model_rows], dtype=float)
        test_f1 = np.asarray([row["test_macro_f1"] for row in model_rows], dtype=float)
        val_f1 = np.asarray([row["val_macro_f1"] for row in model_rows], dtype=float)
        out.append(
            {
                "model": model_name,
                "runs": len(model_rows),
                "param_count": int(model_rows[0]["param_count"]),
                "test_acc_mean": float(test_acc.mean()),
                "test_acc_std": float(test_acc.std(ddof=1)) if len(test_acc) > 1 else 0.0,
                "test_macro_f1_mean": float(test_f1.mean()),
                "test_macro_f1_std": float(test_f1.std(ddof=1)) if len(test_f1) > 1 else 0.0,
                "val_macro_f1_mean": float(val_f1.mean()),
                "val_macro_f1_std": float(val_f1.std(ddof=1)) if len(val_f1) > 1 else 0.0,
            }
        )
    return out


METRIC_FIELDS = [
    "experiment",
    "validation_type",
    "preset",
    "oversample",
    "run_id",
    "seed",
    "model",
    "param_count",
    "best_epoch",
    "val_acc",
    "val_macro_f1",
    "test_acc",
    "test_macro_f1",
]

SUMMARY_FIELDS = [
    "model",
    "runs",
    "param_count",
    "test_acc_mean",
    "test_acc_std",
    "test_macro_f1_mean",
    "test_macro_f1_std",
    "val_macro_f1_mean",
    "val_macro_f1_std",
]
