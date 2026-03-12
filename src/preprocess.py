import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from data_loader import load_dataset

# Columns to drop — IPs are not useful features, timestamps are raw
COLUMNS_TO_DROP = ['id', 'srcip', 'dstip', 'Stime', 'Ltime']

# Categorical columns that need to be converted to numbers
CATEGORICAL_COLUMNS = ['proto', 'state', 'service']

# The target column we want to predict (0 = normal, 1 = attack)
TARGET_COLUMN = 'label'


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    return df.drop(columns=cols)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Fill missing numbers with the median, missing text with 'unknown'
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('unknown')
    return df


def encode_categorical(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Convert text columns into numbers using LabelEncoder.
    
    fit=True  → learn the encoding from data (use on training set)
    fit=False → apply existing encoding (use on test set)
    """
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].astype(str).map(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    return df, encoders


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Standardize features so they all have mean=0 and std=1.
    Fit ONLY on training data to avoid data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_features_target(df: pd.DataFrame):
    """Separate input features (X) from the label we want to predict (y)."""
    # Also drop 'attack_cat' — it's another form of the label, not a feature
    X = df.drop(columns=[TARGET_COLUMN, 'attack_cat'], errors='ignore')
    y = df[TARGET_COLUMN]
    return X, y


def preprocess(train_path: str, test_path: str, output_dir: str = 'data/processed'):
    print("Loading data...")
    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    print("Dropping irrelevant columns...")
    train_df = drop_irrelevant_columns(train_df)
    test_df = drop_irrelevant_columns(test_df)

    print("Handling missing values...")
    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    print("Encoding categorical columns...")
    train_df, encoders = encode_categorical(train_df, fit=True)
    test_df, _ = encode_categorical(test_df, encoders=encoders, fit=False)

    print("Splitting features and labels...")
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    print("Scaling features...")
    X_train, X_test, scaler = scale_features(X_train, X_test)

    # Save processed data
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / 'X_train.npy', X_train)
    np.save(out / 'X_test.npy', X_test)
    np.save(out / 'y_train.npy', y_train)
    np.save(out / 'y_test.npy', y_test)

    # Save scaler and encoders so train.py and evaluate.py can reuse them
    joblib.dump(scaler, out / 'scaler.pkl')
    joblib.dump(encoders, out / 'encoders.pkl')

    print(f"Done. Processed data saved to {output_dir}/")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")


if __name__ == "__main__":
    preprocess(
        train_path='data/raw/unsw-nb15/UNSW_NB15_training-set.csv',
        test_path='data/raw/unsw-nb15/UNSW_NB15_testing-set.csv'
    )