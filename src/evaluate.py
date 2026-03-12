import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib


def load_model(model_path: str = 'models/random_forest.pkl'):
    """Load a trained model from disk."""
    return joblib.load(model_path)


def load_test_data(data_dir: str = 'data/processed'):
    """Load the preprocessed test arrays."""
    d = Path(data_dir)
    X_test = np.load(d / 'X_test.npy')
    y_test = np.load(d / 'y_test.npy')
    return X_test, y_test


def compute_metrics(y_true, y_pred):
    """Return a dict of evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def format_confusion_matrix(cm):
    """Format confusion matrix as a markdown table."""
    lines = [
        "| | Predicted Normal | Predicted Attack |",
        "|---|---|---|",
        f"| **Actual Normal** | {cm[0][0]} | {cm[0][1]} |",
        f"| **Actual Attack** | {cm[1][0]} | {cm[1][1]} |",
    ]
    return "\n".join(lines)


def save_report(metrics, cm, report_path: str = 'reports/results.md'):
    """Write evaluation results to a markdown file."""
    out = Path(report_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, 'w') as f:
        f.write("# Cyber Threat Detector — Evaluation Results\n\n")
        f.write("## Model\n\nRandom Forest (100 trees, max_depth=20)\n\n")
        f.write("## Metrics\n\n")
        f.write(f"| Metric | Score |\n")
        f.write(f"|---|---|\n")
        for name, value in metrics.items():
            f.write(f"| {name.capitalize()} | {value:.4f} |\n")
        f.write(f"\n## Confusion Matrix\n\n")
        f.write(format_confusion_matrix(cm))
        f.write("\n")

    print(f"Report saved to {report_path}")


def evaluate(
    model_path: str = 'models/random_forest.pkl',
    data_dir: str = 'data/processed',
    report_path: str = 'reports/results.md',
):
    """Run full evaluation pipeline."""
    print("Loading model and test data...")
    model = load_model(model_path)
    X_test, y_test = load_test_data(data_dir)

    print("Generating predictions...")
    y_pred = model.predict(X_test)

    print("Computing metrics...")
    metrics = compute_metrics(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    for name, value in metrics.items():
        print(f"  {name.capitalize():>10}: {value:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Attack']))

    save_report(metrics, cm, report_path)
    return metrics


if __name__ == "__main__":
    evaluate()