"""
Training and evaluation script for Chess GCN.

Usage:
    python train.py                          # 50K samples, auto device
    python train.py --num-samples 341000     # full dataset
    python train.py --epochs 100 --batch-size 128
"""

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader

from data import load_and_sample, build_graphs, WDL_K
from model import ChessGATv2


# ── Device selection ─────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


# ── Learning rate schedule: linear warmup + cosine decay ─────────────────────

def get_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)


# ── Training step ────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits, *_ = model(batch)
        target = batch.y.view(-1, 3)

        # KL divergence loss (soft WDL targets)
        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, target, reduction="batchmean")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * target.size(0)
        pred_class = logits.argmax(dim=1)
        true_class = target.argmax(dim=1)
        correct += (pred_class == true_class).sum().item()
        total += target.size(0)

    return total_loss / total, correct / total


# ── Evaluation step ──────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_cp_true = []
    all_values_pred = []

    for batch in loader:
        batch = batch.to(device)
        logits, *_ = model(batch)
        target = batch.y.view(-1, 3)

        log_probs = F.log_softmax(logits, dim=1)
        loss = F.kl_div(log_probs, target, reduction="batchmean")

        total_loss += loss.item() * target.size(0)

        probs = F.softmax(logits, dim=1)
        pred_class = probs.argmax(dim=1)
        true_class = target.argmax(dim=1)
        correct += (pred_class == true_class).sum().item()
        total += target.size(0)

        # Derived value: P(win) - P(loss)
        value_pred = probs[:, 0] - probs[:, 2]
        value_true = target[:, 0] - target[:, 2]

        all_preds.extend(pred_class.cpu().tolist())
        all_targets.extend(true_class.cpu().tolist())
        all_values_pred.extend(value_pred.cpu().tolist())

        if hasattr(batch, "cp"):
            all_cp_true.extend(batch.cp.view(-1).cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total

    # Value MAE
    values_pred_t = torch.tensor(all_values_pred)
    values_true_approx = torch.tensor([
        2.0 / (1.0 + math.exp(-cp / WDL_K)) - 1.0 for cp in all_cp_true
    ]) if all_cp_true else None
    value_mae = (values_pred_t - values_true_approx).abs().mean().item() if values_true_approx is not None else None

    # Pearson correlation
    correlation = None
    if values_true_approx is not None and len(values_true_approx) > 1:
        vp = values_pred_t - values_pred_t.mean()
        vt = values_true_approx - values_true_approx.mean()
        correlation = (vp * vt).sum() / (vp.norm() * vt.norm() + 1e-8)
        correlation = correlation.item()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "value_mae": value_mae,
        "correlation": correlation,
        "preds": all_preds,
        "targets": all_targets,
        "cp_true": all_cp_true,
        "values_pred": all_values_pred,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("WDL Accuracy")
    axes[1].legend()

    if history["val_corr"] and history["val_corr"][0] is not None:
        axes[2].plot(history["val_corr"], label="Val Correlation", color="green")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Pearson r")
        axes[2].set_title("Value Correlation")
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(targets, preds, save_path):
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Win", "Draw", "Loss"])
    disp.plot(cmap="Blues")
    plt.title("WDL Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_eval_scatter(cp_true, values_pred, save_path):
    if not cp_true:
        return
    # Convert predicted value back to rough centipawns
    cp_pred = []
    for v in values_pred:
        v_clamped = max(-0.999, min(0.999, v))
        p_win = (v_clamped + 1.0) / 2.0
        p_loss = 1.0 - p_win
        if p_loss < 1e-6:
            cp_pred.append(5000)
        elif p_win < 1e-6:
            cp_pred.append(-5000)
        else:
            cp_pred.append(WDL_K * math.log(p_win / p_loss))

    # Clamp true cp for display
    cp_true_clamped = [max(-5000, min(5000, c)) for c in cp_true]

    plt.figure(figsize=(8, 8))
    plt.scatter(cp_true_clamped, cp_pred, alpha=0.05, s=2)
    plt.plot([-5000, 5000], [-5000, 5000], "r--", linewidth=1, label="Perfect")
    plt.xlabel("True Evaluation (cp)")
    plt.ylabel("Predicted Evaluation (cp)")
    plt.title("Predicted vs True Evaluation")
    plt.legend()
    plt.xlim(-5000, 5000)
    plt.ylim(-5000, 5000)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Evaluation scatter plot saved to {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Chess GCN")
    parser.add_argument("--data", type=str, default="dataset_eval.csv")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--hidden", type=int, default=192)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-self-edges", action="store_true")
    parser.add_argument("--no-check-feature", action="store_true")
    parser.add_argument("--wdl-k", type=float, default=200.0)
    args = parser.parse_args()

    project_dir = Path(__file__).parent
    device = get_device()

    # Apply feature flags
    import data as data_module
    if args.no_self_edges or args.no_check_feature:
        _orig_ftg = data_module.fen_to_graph
        se = not args.no_self_edges
        cf = not args.no_check_feature
        data_module.fen_to_graph = lambda fen, wdl=None, **kw: _orig_ftg(fen, wdl, self_edges=se, check_feature=cf)
    if args.wdl_k != 200.0:
        data_module.WDL_K = args.wdl_k

    # ── Data loading ──
    print(f"\nLoading dataset ({args.num_samples} samples)...")
    csv_path = project_dir / args.data
    df = load_and_sample(csv_path, num_samples=args.num_samples)
    print(f"  Loaded {len(df)} positions")

    cache_name = f"processed_graphs_{args.num_samples}.pt"
    cache_path = None if args.no_cache else project_dir / cache_name

    graphs = build_graphs(df, cache_path=cache_path)
    print(f"  {len(graphs)} graphs ready")

    # ── Split: 80/10/10 ──
    train_graphs, temp_graphs = train_test_split(graphs, test_size=0.2, random_state=42)
    val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)
    print(f"  Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    # ── Model ──
    node_dim = 21 if args.no_check_feature else 22
    model = ChessGATv2(
        node_dim=node_dim,
        hidden=args.hidden,
        heads=args.heads,
        num_blocks=args.blocks,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {num_params:,} trainable parameters")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_scheduler(optimizer, args.warmup, args.epochs)

    # ── Training loop ──
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_corr": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = project_dir / "best_model.pt"

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...\n")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_result = evaluate(model, val_loader, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_result["loss"])
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_result["accuracy"])
        history["val_corr"].append(val_result["correlation"])

        corr_str = f"{val_result['correlation']:.4f}" if val_result["correlation"] else "N/A"
        mae_str = f"{val_result['value_mae']:.4f}" if val_result["value_mae"] else "N/A"

        marker = ""
        if val_result["loss"] < best_val_loss:
            best_val_loss = val_result["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            marker = " *"
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.3f} | "
            f"Val Loss: {val_result['loss']:.4f}  Acc: {val_result['accuracy']:.3f} | "
            f"Corr: {corr_str}  MAE: {mae_str} | "
            f"LR: {lr:.2e}{marker}"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # ── Save training curves ──
    plot_training_curves(history, project_dir / "training_curves.png")

    # ── Final evaluation on test set ──
    print("\n" + "=" * 60)
    print("Final evaluation on test set")
    print("=" * 60)

    model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=device))
    test_result = evaluate(model, test_loader, device)

    print(f"  Test Loss:       {test_result['loss']:.4f}")
    print(f"  Test WDL Acc:    {test_result['accuracy']:.3f}")
    if test_result["value_mae"] is not None:
        print(f"  Value MAE:       {test_result['value_mae']:.4f}")
    if test_result["correlation"] is not None:
        print(f"  Correlation:     {test_result['correlation']:.4f}")

    # Confusion matrix
    plot_confusion_matrix(test_result["targets"], test_result["preds"], project_dir / "confusion_matrix.png")

    # Scatter plot
    plot_eval_scatter(test_result["cp_true"], test_result["values_pred"], project_dir / "eval_scatter.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
