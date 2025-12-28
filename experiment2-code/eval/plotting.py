import matplotlib.pyplot as plt
import numpy as np
import os

plt.style.use("seaborn-v0_8-whitegrid")  # clean academic style

# Consistent color palette
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

def save_fig(fig, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight")  # high-resolution
    plt.close(fig)

def plot_metrics(history, save_path="results/metrics_curve.png"):
    losses = [h["total_loss"] for h in history]
    cls_losses = [h["classification_loss"] for h in history]
    logic_losses = [h["logic_loss"] for h in history]
    align_losses = [h["alignment_loss"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(losses, label="Total Loss", color=COLORS[0], linewidth=2)
    ax.plot(cls_losses, label="Classification Loss", color=COLORS[1], linewidth=2, linestyle="--")
    ax.plot(logic_losses, label="Logic Loss", color=COLORS[2], linewidth=2, linestyle="-.")
    ax.plot(align_losses, label="Alignment Loss", color=COLORS[3], linewidth=2, linestyle=":")

    ax.set_xlabel("Epoch", fontsize=12, fontweight="bold")
    ax.set_ylabel("Loss Value", fontsize=12, fontweight="bold")
    ax.set_title("Training Loss Curves", fontsize=14, fontweight="bold")
    ax.legend(frameon=True, fontsize=10)

    save_fig(fig, save_path)

def plot_shap(importance, save_path="results/shap_importance.png"):
    indices = np.argsort(-importance)[:20]  # top-20 features
    sorted_imp = importance[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(sorted_imp)), sorted_imp, color=COLORS[0])
    ax.set_xticks(range(len(sorted_imp)))
    ax.set_xticklabels(indices, rotation=45, fontsize=9)
    ax.set_xlabel("Feature Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Relative Importance", fontsize=12, fontweight="bold")
    ax.set_title("Top-20 Feature Importances", fontsize=14, fontweight="bold")

    save_fig(fig, save_path)

def plot_rule_timeline(rule_activations, save_path="results/rule_timeline.png"):
    fig, ax = plt.subplots(figsize=(10, 4))
    cax = ax.imshow(rule_activations.T, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_xlabel("Sample Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Rule Index", fontsize=12, fontweight="bold")
    ax.set_title("Symbolic Rule Activation Timeline", fontsize=14, fontweight="bold")

    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label("Activation Level", fontsize=11, fontweight="bold")

    save_fig(fig, save_path)