"""PyTorch models: 1D-CNN with self-attention, 3D-CNN, a small student
network for knowledge distillation, and multi-task heads."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv1(cin, cout, k):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout),
                         nn.ReLU(inplace=True), nn.MaxPool1d(2))


class Encoder1D(nn.Module):
    """Three conv blocks + multi-head self-attention over spectral positions."""

    def __init__(self, n_bands: int, width: int = 64, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.Sequential(_conv1(1, width // 2, 7), _conv1(width // 2, width, 5), _conv1(width, width * 2, 3))
        d = width * 2
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)
        self.out_dim = d
        self.last_attn = None

    def forward(self, x):                 # x: (N, B)
        h = self.convs(x.unsqueeze(1))    # (N, C, L)
        h = h.transpose(1, 2)             # (N, L, C)
        a, w = self.attn(h, h, h, need_weights=True)
        self.last_attn = w.detach()
        h = self.norm(h + self.drop(a))
        return h.mean(dim=1)              # global average over positions


class Encoder3D(nn.Module):
    """Two 3-D conv blocks over (bands, height, width) + pooling."""

    def __init__(self, n_bands: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=(7, 3, 3), padding=(3, 1, 1)), nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 1, 1)),
            nn.Conv3d(16, 32, kernel_size=(5, 3, 3), padding=(2, 1, 1)), nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((8, 1, 1)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 8, 128), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.out_dim = 128

    def forward(self, x):                 # x: (N, h, w, B)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)   # (N, 1, B, h, w)
        return self.fc(self.net(x))


class Student1D(nn.Module):
    """Small 1D-CNN (two conv blocks, ~10k parameters) for distillation."""

    def __init__(self, n_bands: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(_conv1(1, 8, 7), _conv1(8, 16, 5), nn.AdaptiveAvgPool1d(8), nn.Flatten(),
                                 nn.Linear(16 * 8, 32), nn.ReLU(inplace=True), nn.Linear(32, n_classes))

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class Net(nn.Module):
    """Encoder + one linear head per task. `heads` maps task name -> n_outputs."""

    def __init__(self, encoder: nn.Module, heads: dict[str, int], dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({k: nn.Linear(encoder.out_dim, n) for k, n in heads.items()})

    def forward(self, x):
        z = self.drop(self.encoder(x))
        return {k: h(z) for k, h in self.heads.items()}


class SingleHead(nn.Module):
    """Wrap a plain classifier so it returns {task: logits} like Net."""

    def __init__(self, model: nn.Module, task: str):
        super().__init__()
        self.model, self.task = model, task

    def forward(self, x):
        return {self.task: self.model(x)}


def ordinal_loss(logits: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
    """Cumulative-link (CORAL-style) loss. logits: (N, K); level in 0..K."""
    k = torch.arange(logits.shape[1], device=logits.device)
    target = (level.unsqueeze(1) > k).float()
    return F.binary_cross_entropy_with_logits(logits, target)


def ordinal_predict(logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(logits) > 0.5).sum(dim=1)


def kd_loss(student_logits, teacher_logits, y, T: float, lam: float):
    soft = F.kl_div(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean") * T * T
    return lam * soft + (1 - lam) * F.cross_entropy(student_logits, y)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
