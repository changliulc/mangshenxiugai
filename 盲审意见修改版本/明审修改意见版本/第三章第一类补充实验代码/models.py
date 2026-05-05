# -*- coding: utf-8 -*-
"""Model definitions for Chapter 3 supplementary experiments."""

from __future__ import annotations

import re

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class FixedCNN1D(nn.Module):
    """Fixed-length 1D-CNN baseline used for linear-resampled sequences."""

    def __init__(self, in_channels: int = 4, n_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TwoLayerLSTM(nn.Module):
    """Two-layer unidirectional LSTM for the same fixed-length 4-channel input."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 64,
        n_classes: int = 3,
        lstm_dropout: float = 0.2,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input from common pipeline is (batch, channels, length).
        seq = x.transpose(1, 2)
        out, _ = self.lstm(seq)
        return self.head(out[:, -1, :])


class TwoLayerGRU(nn.Module):
    """Two-layer unidirectional GRU for the same fixed-length 4-channel input."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 64,
        n_classes: int = 3,
        gru_dropout: float = 0.2,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        out, _ = self.gru(seq)
        return self.head(out[:, -1, :])


class TransformerEncoder1D(nn.Module):
    """Lightweight Transformer encoder on the fixed-length 4-channel sequence."""

    def __init__(
        self,
        in_channels: int = 4,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        n_classes: int = 3,
        dropout: float = 0.2,
        max_len: int = 256,
    ):
        super().__init__()
        self.proj = nn.Linear(in_channels, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = x.transpose(1, 2)
        h = self.proj(seq)
        h = h + self.pos[:, : h.shape[1], :]
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))


class ResNet1D(nn.Module):
    """Small ResNet-style 1D-CNN baseline for fixed-length sequences."""

    def __init__(self, in_channels: int = 4, n_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(
            ResidualBlock1D(32, 32, stride=1),
            ResidualBlock1D(32, 64, stride=2),
            ResidualBlock1D(64, 128, stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))


class MaskedCNN1D(nn.Module):
    """CNN for padded variable-length events.

    Input shape is (batch, 5, length): first four channels are x/y/z/magnitude,
    and the last channel is a binary valid-position mask.
    """

    def __init__(self, in_channels: int = 4, n_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data = x[:, :4, :]
        mask = x[:, 4:5, :].clamp(0.0, 1.0)
        h = self.features(data) * mask
        denom = mask.sum(dim=2).clamp_min(1.0)
        pooled = h.sum(dim=2) / denom
        return self.head(pooled)


class MaskedTwoLayerLSTM(nn.Module):
    """Two-layer LSTM using valid lengths from the padding mask."""

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 64,
        n_classes: int = 3,
        lstm_dropout: float = 0.2,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(hidden_size, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        data = x[:, :4, :].transpose(1, 2)
        mask = x[:, 4, :].clamp(0.0, 1.0)
        lengths = mask.sum(dim=1).clamp_min(1).to(torch.long).cpu()
        packed = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.head(h_n[-1])


def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


def build_model(model_name: str, in_channels: int = 4, n_classes: int = 3) -> nn.Module:
    if model_name == "fixed_cnn_1d":
        return FixedCNN1D(in_channels=in_channels, n_classes=n_classes)

    match = re.fullmatch(r"lstm_h(\d+)", model_name)
    if match:
        hidden_size = int(match.group(1))
        return TwoLayerLSTM(
            in_channels=in_channels,
            hidden_size=hidden_size,
            n_classes=n_classes,
            lstm_dropout=0.2,
            head_dropout=0.2,
        )

    match = re.fullmatch(r"gru_h(\d+)", model_name)
    if match:
        hidden_size = int(match.group(1))
        return TwoLayerGRU(
            in_channels=in_channels,
            hidden_size=hidden_size,
            n_classes=n_classes,
            gru_dropout=0.2,
            head_dropout=0.2,
        )

    match = re.fullmatch(r"transformer_d(\d+)", model_name)
    if match:
        d_model = int(match.group(1))
        return TransformerEncoder1D(
            in_channels=in_channels,
            d_model=d_model,
            nhead=4,
            num_layers=2,
            dim_feedforward=2 * d_model,
            n_classes=n_classes,
            dropout=0.2,
        )

    if model_name == "resnet1d":
        return ResNet1D(in_channels=in_channels, n_classes=n_classes)

    if model_name == "masked_cnn_1d":
        return MaskedCNN1D(in_channels=4, n_classes=n_classes)

    match = re.fullmatch(r"masked_lstm_h(\d+)", model_name)
    if match:
        hidden_size = int(match.group(1))
        return MaskedTwoLayerLSTM(
            in_channels=4,
            hidden_size=hidden_size,
            n_classes=n_classes,
            lstm_dropout=0.2,
            head_dropout=0.2,
        )

    raise ValueError(
        f"Unknown model name: {model_name}. "
        "Use fixed_cnn_1d, lstm_h64, lstm_h32, gru_h64, "
        "transformer_d64, resnet1d, masked_cnn_1d, or masked_lstm_h64."
    )
