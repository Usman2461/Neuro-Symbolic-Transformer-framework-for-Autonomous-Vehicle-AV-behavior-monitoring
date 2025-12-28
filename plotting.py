import matplotlib.pyplot as plt
import numpy as np
import os

def plot_metrics(history, save_path="results/metrics_curve.png"):
    losses = [h["total_loss"] for h in history]
    cls_losses = [h["classification_loss"] for h in history]
    logic_losses = [h["logic_loss"] for h in history]
    align_losses = [h["alignment_loss"] for h in history]

    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Total Loss")
    plt.plot(cls_losses, label="Classification Loss")
    plt.plot(logic_losses, label="Logic Loss")
    plt.plot(align_losses, label="Alignment Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_shap(importance, save_path="results/shap_importance.png"):
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(importance)), importance)
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.title("SHAP-like Feature Importance")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_rule_timeline(rule_activations, save_path="results/rule_timeline.png"):
    plt.figure(figsize=(10, 5))
    plt.imshow(rule_activations.T, aspect="auto", cmap="Blues")
    plt.colorbar(label="Activation (0/1)")
    plt.xlabel("Sample Index")
    plt.ylabel("Rule Index")
    plt.title("Symbolic Rule Activations")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()