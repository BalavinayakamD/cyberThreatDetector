import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib


def load_processed_data(data_dir: str = 'data/processed'):
    """Load the preprocessed numpy arrays saved by preprocess.py."""
    d = Path(data_dir)
    X_train = np.load(d / 'X_train.npy')
    y_train = np.load(d / 'y_train.npy')
    X_test = np.load(d / 'X_test.npy')
    y_test = np.load(d / 'y_test.npy')
    return X_train, y_train, X_test, y_test


def build_model():
    """Create a Random Forest classifier with sensible defaults."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )


def tune_model(X_train, y_train):
    """Run RandomizedSearchCV to find optimal hyperparameters."""
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [15, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None],
        'max_features': ['sqrt', 'log2'],
    }

    base = RandomForestClassifier(random_state=42, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_distributions,
        n_iter=20,
        scoring='f1',
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    print(f"\n  Best F1 (CV): {search.best_score_:.4f}")
    print(f"  Best params:  {search.best_params_}")
    return search.best_estimator_


def train(data_dir: str = 'data/processed', model_dir: str = 'models', tune: bool = False):
    """Load data, train the model, and save it to disk."""
    print("Loading processed data...")
    X_train, y_train, X_test, y_test = load_processed_data(data_dir)
    print(f"  Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")

    if tune:
        print("Tuning hyperparameters (RandomizedSearchCV, 20 iter, 3-fold CV)...")
        model = tune_model(X_train, y_train)
    else:
        print("Training Random Forest model...")
        model = build_model()
        model.fit(X_train, y_train)

    # Quick sanity check
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")

    # Save model
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path = out / 'random_forest.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model


if __name__ == "__main__":
    import sys
    do_tune = '--tune' in sys.argv
    train(tune=do_tune)