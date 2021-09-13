import pandas as pd


def get_data(path: str):
    return pd.read_csv(path)
