"""Train a D-MPNN on a binary classification CSV (default: SD1).

Hyperparameters match PROJECT.md (hidden=300, depth=3, ffn=2, dropout=0,
30 epochs). Random 80/10/10 split. Per-epoch val AUC, best checkpoint saved.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

from chemprop.args import TrainArgs
from chemprop.data import (
    MoleculeDataLoader,
    MoleculeDatapoint,
    MoleculeDataset,
    StandardScaler,
)
from chemprop.models import MoleculeModel
from chemprop.nn_utils import NoamLR
from chemprop.train import evaluate_predictions, predict as predict_fn


def load_csv(path: Path) -> list[MoleculeDatapoint]:
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    data = []
    for r in rows:
        data.append(
            MoleculeDatapoint(
                smiles=[r["smiles"]],
                targets=[float(r["active"])],
                features_generator=["rdkit_2d_normalized"],
            )
        )
    return data


def split_random(data: list, seed: int, ratios=(0.8, 0.1, 0.1)) -> tuple[list, list, list]:
    rng = random.Random(seed)
    idx = list(range(len(data)))
    rng.shuffle(idx)
    n = len(data)
    n_tr = int(n * ratios[0])
    n_va = int(n * ratios[1])
    tr = [data[i] for i in idx[:n_tr]]
    va = [data[i] for i in idx[n_tr : n_tr + n_va]]
    te = [data[i] for i in idx[n_tr + n_va :]]
    return tr, va, te


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default="data/sd1_train.csv")
    p.add_argument("--save_dir", default="runs/sd1")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    cli = p.parse_args()

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)

    save_dir = Path(cli.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    args = TrainArgs()
    args.epochs = cli.epochs
    args.batch_size = cli.batch_size
    args.seed = cli.seed
    args.num_workers = cli.num_workers
    args.data_path = cli.data_path
    args.save_dir = str(save_dir)

    print(f"device: {args.device}")
    print(f"loading {cli.data_path}")
    data = load_csv(Path(cli.data_path))
    print(f"  {len(data):,} datapoints")

    train_data, val_data, test_data = split_random(data, args.seed)
    print(f"  split: train={len(train_data)}  val={len(val_data)}  test={len(test_data)}")

    train_ds = MoleculeDataset(train_data)
    val_ds = MoleculeDataset(val_data)
    test_ds = MoleculeDataset(test_data)

    feat_scaler = train_ds.normalize_features()
    val_ds.normalize_features(feat_scaler)
    test_ds.normalize_features(feat_scaler)

    args.train_data_size = len(train_ds)
    args.features_size = train_ds.features_size()
    print(f"  features_size: {args.features_size}")

    train_loader = MoleculeDataLoader(
        dataset=train_ds,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        seed=args.seed,
    )
    val_loader = MoleculeDataLoader(
        dataset=val_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )
    test_loader = MoleculeDataLoader(
        dataset=test_ds, batch_size=args.batch_size, num_workers=args.num_workers
    )

    model = MoleculeModel(args).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model params: {n_params:,}")

    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": args.init_lr, "weight_decay": 0}])
    steps_per_epoch = max(1, args.train_data_size // args.batch_size)
    scheduler = NoamLR(
        optimizer=optimizer,
        warmup_epochs=[args.warmup_epochs],
        total_epochs=[args.epochs],
        steps_per_epoch=steps_per_epoch,
        init_lr=[args.init_lr],
        max_lr=[args.max_lr],
        final_lr=[args.final_lr],
    )
    loss_func = nn.BCEWithLogitsLoss(reduction="none")

    best_val = -float("inf")
    best_epoch = -1
    n_iter = 0
    metrics = ["auc", "prc-auc", "accuracy"]
    history: list[dict] = []

    bar = "─" * 56
    print(f"\n{bar}\n {'ep':>3}  {'train_loss':>10}  {'val_auc':>8}  {'val_prc':>8}  {'val_acc':>8}  {'lr':>8}\n{bar}")

    for epoch in range(args.epochs):
        # train one epoch, capture running mean loss
        model.train()
        loss_sum = n_batches = 0
        for batch in train_loader:
            mol_batch = batch.batch_graph()
            features_batch = batch.features()
            target_batch = batch.targets()
            ad_batch, af_batch, bf_batch, dw_batch = batch.atom_descriptors(), batch.atom_features(), batch.bond_features(), batch.data_weights()
            mask = torch.tensor([[x is not None for x in tb] for tb in target_batch], dtype=torch.bool)
            targets = torch.tensor([[0 if x is None else x for x in tb] for tb in target_batch], dtype=torch.float)
            data_weights = torch.tensor(dw_batch).unsqueeze(1)

            model.zero_grad()
            preds = model(mol_batch, features_batch, ad_batch, af_batch, bf_batch)
            dev = preds.device
            mask, targets, data_weights = mask.to(dev), targets.to(dev), data_weights.to(dev)
            loss = loss_func(preds, targets) * data_weights * mask
            loss = loss.sum() / mask.sum()
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss.item()
            n_batches += 1
            n_iter += len(batch)

        train_loss = loss_sum / max(1, n_batches)

        val_preds = predict_fn(model=model, data_loader=val_loader, disable_progress_bar=True)
        val_scores = evaluate_predictions(
            preds=val_preds,
            targets=val_loader.targets,
            num_tasks=args.num_tasks,
            metrics=metrics,
            dataset_type=args.dataset_type,
        )
        v_auc = float(np.nanmean(val_scores["auc"]))
        v_prc = float(np.nanmean(val_scores["prc-auc"]))
        v_acc = float(np.nanmean(val_scores["accuracy"]))
        lr = scheduler.get_lr()[0]

        flag = " "
        if v_auc > best_val:
            best_val = v_auc
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "args": args.as_dict(),
                    "feature_scaler": {
                        "means": feat_scaler.means if feat_scaler else None,
                        "stds": feat_scaler.stds if feat_scaler else None,
                    },
                },
                save_dir / "model.pt",
            )
            flag = "*"

        print(f" {epoch + 1:>3}  {train_loss:>10.4f}  {v_auc:>8.4f}  {v_prc:>8.4f}  {v_acc:>8.4f}  {lr:>8.2e} {flag}")
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_auc": v_auc, "val_prc": v_prc, "val_acc": v_acc, "lr": lr})

    print(bar)
    print(f"best val auc={best_val:.4f} at epoch {best_epoch + 1}")

    # save history csv
    with open(save_dir / "history.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    # history plot
    epochs_arr = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs_arr, [h["train_loss"] for h in history], marker="o", label="train loss")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].set_title("train loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(epochs_arr, [h["val_auc"] for h in history], marker="o", label="AUC")
    axes[1].plot(epochs_arr, [h["val_prc"] for h in history], marker="s", label="PRC-AUC")
    axes[1].plot(epochs_arr, [h["val_acc"] for h in history], marker="^", label="accuracy")
    axes[1].axvline(best_epoch + 1, color="k", ls="--", alpha=0.3, label=f"best ep {best_epoch + 1}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("score"); axes[1].set_title("val metrics"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "history.png", dpi=120)
    plt.close(fig)
    print(f"wrote {save_dir / 'history.png'}")

    # test on best checkpoint
    ckpt = torch.load(save_dir / "model.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_preds_list = predict_fn(model=model, data_loader=test_loader, disable_progress_bar=True)
    test_scores = evaluate_predictions(
        preds=test_preds_list,
        targets=test_loader.targets,
        num_tasks=args.num_tasks,
        metrics=metrics,
        dataset_type=args.dataset_type,
    )
    t_auc = float(np.nanmean(test_scores["auc"]))
    t_prc = float(np.nanmean(test_scores["prc-auc"]))
    t_acc = float(np.nanmean(test_scores["accuracy"]))
    print(f"test: auc={t_auc:.4f}  prc-auc={t_prc:.4f}  acc={t_acc:.4f}")

    with open(save_dir / "test_scores.csv", "w") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for m, v in zip(metrics, [t_auc, t_prc, t_acc]):
            w.writerow([m, v])

    # ROC + PR on test set
    test_preds = np.asarray(test_preds_list)[:, 0]
    test_targets = np.asarray([d.targets[0] for d in test_data], dtype=float)
    fpr, tpr, _ = roc_curve(test_targets, test_preds)
    prec, rec, _ = precision_recall_curve(test_targets, test_preds)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label=f"AUC={t_auc:.3f}"); axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR"); axes[0].set_title("ROC (test)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(rec, prec, label=f"PRC-AUC={t_prc:.3f}")
    axes[1].axhline(test_targets.mean(), color="k", ls="--", alpha=0.3, label=f"base rate={test_targets.mean():.3f}")
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].set_title("Precision–Recall (test)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "test_curves.png", dpi=120)
    plt.close(fig)
    print(f"wrote {save_dir / 'test_curves.png'}")


if __name__ == "__main__":
    main()
