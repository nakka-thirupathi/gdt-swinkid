"""
Visualization utilities for GDT-SwinKid (Kidney CT)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import Config


def plot_confusion_matrices(
    train_results,
    test_results,
    labels=Config.LABELS,
    save_path="results"
):
    """Plot confusion matrices for train and test sets"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(
        train_results["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title(
        f"Training Confusion Matrix\nAccuracy: {train_results['accuracy']:.2f}%"
    )
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    sns.heatmap(
        test_results["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[1],
    )
    axes[1].set_title(
        f"Test Confusion Matrix\nAccuracy: {test_results['accuracy']:.2f}%"
    )
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, "confusion_matrices.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_performance_comparison(
    train_results,
    test_results,
    save_path="results"
):
    """Plot macro-averaged performance metrics"""

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]

    train_vals = [
        train_results["accuracy"],
        train_results["precision"],
        train_results["recall"],
        train_results["f1_score"],
        train_results["auc"],
    ]

    test_vals = [
        test_results["accuracy"],
        test_results["precision"],
        test_results["recall"],
        test_results["f1_score"],
        test_results["auc"],
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, train_vals, width, label="Train")
    bars2 = ax.bar(x + width / 2, test_vals, width, label="Test")

    ax.set_ylabel("Score (%)")
    ax.set_title("GDT-SwinKid Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, "performance_comparison.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_roc_curve(
    train_results,
    test_results,
    train_labels,
    test_labels,
    save_path="results"
):
    """
    Plot One-vs-Rest ROC curves for each class (test set)
    """

    y_test_bin = label_binarize(
        test_labels, classes=list(range(Config.NUM_CLASSES))
    )

    plt.figure(figsize=(10, 7))

    for i, label in enumerate(Config.LABELS):
        fpr, tpr, _ = roc_curve(
            y_test_bin[:, i], test_results["proba"][:, i]
        )
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{label} (AUC = {roc_auc:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest) – Kidney CT")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, "roc_curve.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_feature_importance(
    classifier,
    latent_dim=Config.LATENT_DIM,
    save_path="results"
):
    """Plot XGBoost feature importance"""

    importance = classifier.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(
        range(latent_dim),
        importance[indices],
        alpha=0.7,
    )
    plt.xlabel("Latent Feature Index")
    plt.ylabel("Importance Score")
    plt.title("XGBoost Feature Importance (Latent Space)")
    plt.xticks(range(latent_dim), indices)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, "feature_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_sample_predictions(
    x_test,
    test_labels,
    test_pred,
    labels=Config.LABELS,
    save_path="results",
):
    """Visualize correct vs incorrect predictions"""

    correct_idx = np.where(test_pred == test_labels)[0]
    incorrect_idx = np.where(test_pred != test_labels)[0]

    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle(
        "GDT-SwinKid Sample Predictions (Kidney CT)",
        fontsize=16,
        fontweight="bold",
    )

    for i, idx in enumerate(correct_idx[:6]):
        axes[0, i].imshow(x_test[idx], cmap="gray")
        axes[0, i].set_title(
            f"Pred: {labels[test_pred[idx]]}\nTrue: {labels[test_labels[idx]]}",
            fontsize=9,
            color="green",
        )
        axes[0, i].axis("off")

    for i, idx in enumerate(incorrect_idx[:6]):
        axes[1, i].imshow(x_test[idx], cmap="gray")
        axes[1, i].set_title(
            f"Pred: {labels[test_pred[idx]]}\nTrue: {labels[test_labels[idx]]}",
            fontsize=9,
            color="red",
        )
        axes[1, i].axis("off")

    for i in range(len(incorrect_idx), 6):
        axes[1, i].axis("off")

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(
        os.path.join(save_path, "sample_predictions.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
