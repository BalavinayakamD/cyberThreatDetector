import pandas as pd
from pathlib import Path

def load_dataset(path :str) -> pd.DataFrame:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"The file at {path} does not exist.")
    
    df = pd.read_csv(file_path)
    return df

def inspect_dataset(df: pd.DataFrame) -> None:
    print("Data Shape : ", df.shape)
    print("Columns :" , df.columns.tolist())
    print("Missing Vlaues : " , df.isnull().sum())
    print("Data Types : \n", df.dtypes)

if __name__ == "__main__":
    TRAIN_DATA_PATH = 'data/raw/unsw-nb15/UNSW_NB15_training-set.csv'
    TEST_DATA_PATH = 'data/raw/unsw-nb15/UNSW_NB15_testing-set.csv'

    train_df = load_dataset(TRAIN_DATA_PATH)
    test_df = load_dataset(TEST_DATA_PATH)

    print("Training Dataset Overview: ")
    inspect_dataset(train_df)
    print("\n\nTesting Dataset Overview: ")
    inspect_dataset(test_df)