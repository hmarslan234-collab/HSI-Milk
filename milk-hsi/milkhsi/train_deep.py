"""Training / evaluation loops for the deep models on the shared grouped folds."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold

from .data import patch_steps
from .preprocess import BandMinMax, scatter_step


class Front:
    """Scatter correction + min-max fitted on the training spectra of one fold.
    Applied to mean spectra (N x B) or to every pixel of patches (N x h x w x B)."""

    def __init__(self, cfg):
        self.sc = scatter_step(cfg["preprocess"]["scatter"])
        self.mm = BandMinMax() if cfg["preprocess"]["minmax"] else None

    def fit(self, X):
        if self.sc is not None:
            X = self.sc.fit(X).transform(X)
        if self.mm is not None:
            self.mm.fit(X)
        return self

    def __call__(self, X):
        shape = X.shape
        X = X.reshape(-1, shape[-1])
        if self.sc is not None:
            X = self.sc.transform(X)
        if self.mm is not None:
            X = self.mm.transform(X)
        return np.asarray(X, dtype=np.float32).reshape(shape)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Batcher:
    """Yields preprocessed mini-batches of spectra or patches."""

    def __init__(self, idx, X, patches, raw_wl, front, cfg, use_patches, targets, shuffle, seed=0):
        self.idx, self.X, self.P, self.raw_wl = np.asarray(idx), X, patches, raw_wl
        self.front, self.cfg, self.use_patches = front, cfg, use_patches
        self.targets, self.shuffle = targets, shuffle
        self.bs = cfg["deep"]["batch_size"]
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        idx = self.rng.permutation(self.idx) if self.shuffle else self.idx
        for s in range(0, len(idx), self.bs):
            b = np.sort(idx[s:s + self.bs])
            if self.use_patches:
                x = self.front(patch_steps(self.P[b], self.raw_wl, self.cfg))
            else:
                x = self.front(self.X[b])
            yield b, torch.from_numpy(x), {k: torch.from_numpy(v[b]) for k, v in self.targets.items()}


def val_split(train_idx, y, groups, seed):
    """Hold out ~1/7 of the training *samples* for early stopping."""
    import pandas as pd
    per_class = pd.Series(y[train_idx]).groupby(groups[train_idx]).first().value_counts().min()
    k = int(max(2, min(7, per_class)))
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    a, b = next(sgkf.split(train_idx, y[train_idx], groups[train_idx]))
    return train_idx[a], train_idx[b]


def compute_loss(out, tgt, cfg, kinds):
    """kinds: task -> 'ce' or 'ord'. Species weighted by alpha, others by beta (Eq. 3)."""
    a, b = cfg["deep"]["mtl_alpha"], cfg["deep"]["mtl_beta"]
    loss = 0.0
    for t, logits in out.items():
        y = tgt[t].long()
        if kinds[t] == "ord":
            from .models_deep import ordinal_loss
            l = ordinal_loss(logits, y)
        else:
            l = F.cross_entropy(logits, y)
        loss = loss + (a if t == "species" else b) * l
    return loss


def fit_net(net, tr_batches, va_batches, cfg, kinds, log=print):
    dev = device()
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["deep"]["lr"], weight_decay=cfg["deep"]["weight_decay"])
    best, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
    for ep in range(cfg["deep"]["epochs"]):
        net.train()
        for _, x, t in tr_batches:
            x = x.to(dev)
            t = {k: v.to(dev) for k, v in t.items()}
            opt.zero_grad()
            loss = compute_loss(net(x), t, cfg, kinds)
            loss.backward()
            opt.step()
        net.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for _, x, t in va_batches:
                x = x.to(dev)
                t = {k: v.to(dev) for k, v in t.items()}
                vl += compute_loss(net(x), t, cfg, kinds).item() * len(x)
                n += len(x)
        vl /= max(n, 1)
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg["deep"]["patience"]:
                break
    net.load_state_dict(best_state)
    log(f"      stopped at epoch {ep + 1}, best val loss {best:.4f}")
    return net


@torch.no_grad()
def predict(net, batches, n_total):
    dev = device()
    net.eval().to(dev)
    out, t0, n = {}, time.time(), 0
    for b, x, _ in batches:
        o = net(x.to(dev))
        for k, v in o.items():
            arr = out.setdefault(k, np.zeros((n_total, v.shape[1]), np.float32))
            arr[b] = v.cpu().numpy()
        n += len(b)
    ms = 1000 * (time.time() - t0) / max(n, 1)
    return out, ms


def softmax_np(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
