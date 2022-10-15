from ast import Call
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple

# function to add lags to dataframe
def create_lag_creator(
    num_lags: int, col_name: str
) -> Callable[[pd.core.frame.DataFrame], pd.core.frame.DataFrame]:
    def lag_creator(df: pd.core.frame.DataFrame):
        for num in range(num_lags):
            df[f"{col_name}_lag{num+1}"] = df[col_name].shift(num)
        return df

    return lag_creator


# function to add the label
def add_label_buy_close(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df["tomorrow_close"] = df.loc[:, "close"].shift(-1)
    df["label"] = np.where(
        df.loc[:, "close"] >= df.loc[:, "tomorrow_close"], "SELL", "BUY"
    )
    return df.drop("tomorrow_close", axis=1)


# function to return a dataframe with the columns we want to keep for training
def create_cols_to_keep(
    list_cols: List[str],
) -> Callable[[pd.core.frame.DataFrame], pd.core.frame.DataFrame]:
    def cols_to_keep(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        return df[list_cols]

    return cols_to_keep


# function to split x and y for model
def split_X_Y(
    df: pd.core.frame.DataFrame,
) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.Series]:
    df = df.copy()
    X = df.drop("label", axis=1)
    Y = df.iloc[:, -1]
    return X, Y

