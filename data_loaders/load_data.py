from ast import literal_eval

import pandas as pd


def load_data(train_file_path: str) -> pd.DataFrame:
    df = pd.read_csv(train_file_path)
    df["choices"] = df["choices"].apply(literal_eval)
    df["question_plus"] = df["question_plus"].fillna("")
    return df
