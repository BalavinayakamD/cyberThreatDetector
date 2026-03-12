# Cyber Threat Detector — Evaluation Results

## Model

Random Forest (100 trees, max_depth=20)

## Metrics

| Metric | Score |
|---|---|
| Accuracy | 0.9004 |
| Precision | 0.9885 |
| Recall | 0.8638 |
| F1 | 0.9219 |

## Confusion Matrix

| | Predicted Normal | Predicted Attack |
|---|---|---|
| **Actual Normal** | 54799 | 1201 |
| **Actual Attack** | 16259 | 103082 |
