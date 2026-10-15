"""Deep models on the same grouped folds as the classical models.

Models
  cnn1d   1D-CNN with self-attention on ROI mean spectra, one network per task
  cnn3d   3D-CNN on 7 x 7 x B ROI patches, one network per task
  mtl1d   1D-CNN encoder shared by all tasks (multi-task, Eq. 3)
  mtl3d   3D-CNN encoder shared by all tasks (multi-task, Eq. 3)
  kd      small 1D student distilled from the cnn3d teacher of the same fold

    python scripts/04_run_deep.py --models cnn1d cnn3d mtl3d kd
    python scripts/04_run_deep.py --models cnn1d --tasks species --repeats 1   # quick check

Requires PyTorch (GPU recommended for cnn3d / mtl3d). Results go to the same
results/folds, results/oof folders as step 03, so step 05 treats all models alike.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from milkhsi.data import encode, load_config, load_rois, make_splits, raw_wavelengths  # noqa: E402
from milkhsi.labels import task_target                                                  # noqa: E402
from milkhsi.metrics import aggregate_by_sample, score                                  # noqa: E402
from milkhsi.models_deep import (Encoder1D, Encoder3D, Net, SingleHead, Student1D,      # noqa: E402
                                 kd_loss, n_params, ordinal_predict)
from milkhsi.train_deep import (Batcher, Front, device, fit_net, predict,               # noqa: E402
                                softmax_np, val_split)


def fold_rows(task, y, groups, te, proba, sp, extra):
    m_roi = score(y[te], proba[te])
    ys, ps = aggregate_by_sample(groups[te], y[te], proba[te])
    m_smp = score(ys, ps)
    return {"repeat": sp["repeat"], "fold": sp["fold"], **extra,
            **{f"roi_{k}": v for k, v in m_roi.items()}, **{f"sample_{k}": v for k, v in m_smp.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--models", nargs="*", default=["cnn1d", "cnn3d", "mtl1d", "mtl3d", "kd"])
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--repeats", type=int, help="use only the first N repeats (quick runs)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    torch.manual_seed(cfg["cv"]["seed"])
    need_patches = any(m in a.models for m in ("cnn3d", "mtl3d", "kd"))
    X, wl, meta, P = load_rois(cfg, with_patches=need_patches)
    rwl = raw_wavelengths(cfg)
    groups = meta["sample_id"].to_numpy()
    tasks = [t for t in (a.tasks or cfg["tasks"]) if meta[{"level": "level_class"}.get(t, t)].nunique() > 1]
    res = Path(cfg["paths"]["results"])
    for d in ("folds", "oof", "classes"):
        (res / d).mkdir(parents=True, exist_ok=True)
    splits = make_splits(meta, cfg)
    if a.repeats:
        splits = [s for s in splits if s["repeat"] < a.repeats]

    Y = {t: encode(task_target(meta, t)) for t in tasks}
    for t, (_, cl) in Y.items():
        (res / "classes" / f"{t}.txt").write_text("\n".join(map(str, cl)))
    # multi-task targets: species, binary, adulterant (CE) + level 0..10 (ordinal)
    mt = {"species": encode(meta["species"])[0], "binary": encode(meta["binary"])[0],
          "adulterant": encode(meta["adulterant"])[0], "level": meta["level"].to_numpy()}
    adul_classes = encode(meta["adulterant"])[1]
    print(f"device: {device()}  |  {len(X)} ROIs, {X.shape[1]} bands  |  tasks: {tasks}")

    rows = {}
    oofs = {}
    for sp in splits:
        tr_all, te = np.asarray(sp["train"]), np.asarray(sp["test"])
        strat = (meta["species"] + "|" + meta["adulterant"]).to_numpy()
        tr, va = val_split(tr_all, strat, groups, seed=cfg["cv"]["seed"] + sp["fold"])
        front = Front(cfg).fit(X[tr])
        nb = X.shape[1]
        print(f"-- repeat {sp['repeat']} fold {sp['fold']}: train {len(tr)} / val {len(va)} / test {len(te)} ROIs")

        def batches(idx, patches, targets, shuffle):
            return Batcher(idx, X, P, rwl, front, cfg, patches, targets, shuffle, seed=sp["fold"])

        teacher = {}
        for model in a.models:
            if model in ("cnn1d", "cnn3d"):
                use_p = model == "cnn3d"
                for t in tasks:
                    y, cl = Y[t]
                    tg = {t: y}
                    enc = Encoder3D(nb, cfg["deep"]["dropout"]) if use_p else Encoder1D(nb, dropout=cfg["deep"]["dropout"])
                    net = Net(enc, {t: len(cl)}, cfg["deep"]["dropout"])
                    t0 = time.time()
                    net = fit_net(net, batches(tr, use_p, tg, True), batches(va, use_p, tg, False), cfg, {t: "ce"})
                    fit_s = time.time() - t0
                    out, ms = predict(net, batches(te, use_p, tg, False), len(X))
                    proba = softmax_np(out[t])
                    if model == "cnn3d":
                        teacher[t] = net
                    r = fold_rows(t, y, groups, te, proba, sp, {"fit_s": fit_s, "infer_ms_per_roi": ms,
                                                                  "params": n_params(net)})
                    rows.setdefault((t, model), []).append(r)
                    if sp["repeat"] == 0:
                        oofs.setdefault((t, model), np.full((len(X), len(cl)), np.nan))[te] = proba[te]
                    print(f"   {model:6s} {t:10s} roi acc {r['roi_accuracy']:.3f}  sample acc {r['sample_accuracy']:.3f}")
            elif model in ("mtl1d", "mtl3d"):
                use_p = model == "mtl3d"
                heads = {"species": int(mt["species"].max()) + 1, "binary": 2, "adulterant": len(adul_classes), "level": 10}
                kinds = {"species": "ce", "binary": "ce", "adulterant": "ce", "level": "ord"}
                enc = Encoder3D(nb, cfg["deep"]["dropout"]) if use_p else Encoder1D(nb, dropout=cfg["deep"]["dropout"])
                net = Net(enc, heads, cfg["deep"]["dropout"])
                t0 = time.time()
                net = fit_net(net, batches(tr, use_p, mt, True), batches(va, use_p, mt, False), cfg, kinds)
                fit_s = time.time() - t0
                out, ms = predict(net, batches(te, use_p, mt, False), len(X))
                for t in tasks:
                    y, cl = Y[t]
                    if t == "level":
                        # combine predicted adulterant with predicted ordinal level
                        a_pred = np.asarray(adul_classes)[out["adulterant"].argmax(1)]
                        l_pred = ordinal_predict(torch.from_numpy(out["level"])).numpy().clip(1, 10)
                        lab = np.where(a_pred == "pure", "pure",
                                       np.char.add(np.char.add(a_pred.astype(str), "_"),
                                                   np.char.zfill(l_pred.astype(str), 2)))
                        idx = {c: i for i, c in enumerate(cl)}
                        pred = np.array([idx.get(s, -1) for s in lab])
                        proba = np.zeros((len(X), len(cl)))
                        ok = pred >= 0
                        proba[np.where(ok)[0], pred[ok]] = 1.0     # hard prediction; AUC/ECE not defined
                    else:
                        proba = softmax_np(out[t])
                    r = fold_rows(t, y, groups, te, proba, sp, {"fit_s": fit_s, "infer_ms_per_roi": ms,
                                                                  "params": n_params(net)})
                    if t == "level":
                        lv = meta["level"].to_numpy()
                        r["level_mae"] = float(np.abs(ordinal_predict(torch.from_numpy(out["level"][te])).numpy()
                                                      - lv[te]).mean())
                        r["roi_roc_auc"] = r["sample_roc_auc"] = r["roi_ece"] = r["sample_ece"] = np.nan
                    rows.setdefault((t, model), []).append(r)
                    if sp["repeat"] == 0:
                        oofs.setdefault((t, model), np.full((len(X), len(cl)), np.nan))[te] = proba[te]
                    print(f"   {model:6s} {t:10s} roi acc {r['roi_accuracy']:.3f}  sample acc {r['sample_accuracy']:.3f}")
            elif model == "kd":
                for t in tasks:
                    if t not in teacher:
                        print("   kd needs cnn3d in the same run; skipping")
                        break
                    y, cl = Y[t]
                    # teacher logits on train + val ROIs
                    tl, _ = predict(teacher[t], batches(np.r_[tr, va], True, {t: y}, False), len(X))
                    tlog = tl[t]
                    st = Student1D(nb, len(cl)).to(device())
                    opt = torch.optim.Adam(st.parameters(), lr=cfg["deep"]["lr"])
                    T, lam = cfg["deep"]["kd_temperature"], cfg["deep"]["kd_lambda"]
                    best, best_state, bad = np.inf, {k: v.clone() for k, v in st.state_dict().items()}, 0
                    t0 = time.time()
                    for ep in range(cfg["deep"]["epochs"]):
                        st.train()
                        for b, x, tg in batches(tr, False, {t: y}, True):
                            opt.zero_grad()
                            loss = kd_loss(st(x.to(device())), torch.from_numpy(tlog[b]).to(device()),
                                           tg[t].long().to(device()), T, lam)
                            loss.backward()
                            opt.step()
                        st.eval()
                        with torch.no_grad():
                            vl = np.mean([kd_loss(st(x.to(device())), torch.from_numpy(tlog[b]).to(device()),
                                                  tg[t].long().to(device()), T, lam).item()
                                          for b, x, tg in batches(va, False, {t: y}, False)])
                        if vl < best - 1e-4:
                            best, best_state, bad = vl, {k: v.clone() for k, v in st.state_dict().items()}, 0
                        else:
                            bad += 1
                            if bad >= cfg["deep"]["patience"]:
                                break
                    st.load_state_dict(best_state)
                    fit_s = time.time() - t0
                    out, ms = predict(SingleHead(st, t), batches(te, False, {t: y}, False), len(X))
                    proba = softmax_np(out[t])
                    r = fold_rows(t, y, groups, te, proba, sp, {"fit_s": fit_s, "infer_ms_per_roi": ms,
                                                                  "params": n_params(st)})
                    rows.setdefault((t, "kd"), []).append(r)
                    if sp["repeat"] == 0:
                        oofs.setdefault((t, "kd"), np.full((len(X), len(cl)), np.nan))[te] = proba[te]
                    print(f"   kd     {t:10s} roi acc {r['roi_accuracy']:.3f}  sample acc {r['sample_accuracy']:.3f}")

        # write after every fold so a long run can be inspected or resumed by hand
        for (t, m), rr in rows.items():
            pd.DataFrame(rr).to_csv(res / "folds" / f"{t}__{m}.csv", index=False)
        for (t, m), o in oofs.items():
            np.save(res / "oof" / f"{t}__{m}.npy", o)


if __name__ == "__main__":
    main()
