import numpy as np
import pandas as pd


def clean_returns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_index().sort_index(axis=1)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1)
    return df


def align_dataframes(*dfs):
    valid = [df for df in dfs if df is not None]
    if not valid:
        return dfs

    common_index = valid[0].index
    common_cols = valid[0].columns

    for df in valid[1:]:
        common_index = common_index.intersection(df.index)
        common_cols = common_cols.intersection(df.columns)

    common_index = common_index.sort_values()
    common_cols = common_cols.sort_values()

    aligned = []
    for df in dfs:
        if df is None:
            aligned.append(None)
        else:
            aligned.append(df.loc[common_index, common_cols].copy())

    return aligned
