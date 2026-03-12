import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """Plot and optionally save a confusion matrix heatmap."""
    if labels is None:
        labels = ['Normal', 'Attack']

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Confusion matrix saved to {save_path}")

    plt.close(fig)
    return fig


def plot_feature_importance(model, feature_names=None, top_n=15, save_path=None):
    """Plot the top-N most important features from a tree-based model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    if feature_names is not None:
        names = [feature_names[i] for i in indices]
    else:
        names = [f"Feature {i}" for i in indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importances[indices], y=names, ax=ax, palette='viridis')
    ax.set_title(f'Top {top_n} Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()

    if save_path:
        out = Path(save_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"Feature importance plot saved to {save_path}")

    plt.close(fig)
    return fig