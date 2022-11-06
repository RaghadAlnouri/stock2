from typing import Callable, List, Tuple

import numpy as np
import pandas as pd


# function to add lags to dataframe
def create_lag_creator(num_lags: int, col_name: str = 'close', skip: int = 4) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    function to separate parameters from function in pipeline
    :param num_lags: number of lags to add
    :param col_name: column name to use to get the lags (usually close price)
    :param skip: in range function to skip increments
    :return: function to process dataframe
    """
    def lag_creator(df: pd.DataFrame) -> pd.DataFrame:
        for num in range(1, num_lags, skip):
            df[f"{col_name}_lag{num}"] = df[col_name].shift(num)
        return df

    return lag_creator


# function to add the label
# 0 = SELL, 1 = BUY
def add_label_buy_close(df: pd.DataFrame) -> pd.DataFrame:
    """
    function to add a label 0 for SELL and 1 for BUY, if prices are the same
    defaults to SELL
    :param df: dataframe
    :return: df with label added
    """
    df["tomorrow_close"] = df.loc[:, "close"].shift(-1)
    df["label"] = np.where(df.loc[:, "close"] >= df.loc[:, "tomorrow_close"], 0, 1)
    return df.drop("tomorrow_close", axis=1)


# function to return a dataframe with the columns we want to keep for training
def create_cols_to_keep(list_cols: List[str]) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    list of columns to keep, used to drop any unused columns
    :param list_cols: list of columns to keep
    :return: function to be used in pipeline
    """
    def cols_to_keep(df: pd.DataFrame) -> pd.DataFrame:
        return df[list_cols].copy()

    return cols_to_keep


# function to split x and y for model
def create_splitter(col_label: str) -> Callable[[pd.DataFrame], Tuple[pd.DataFrame, pd.Series]]:
    """
    function to separate X & y parameters to use in a model
    :param col_label: label of the y column - usually "label"
    :return: function to return tuple of dataframe & series
    """
    def split_x_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        df = df.copy()
        x = df.drop(col_label, axis=1)
        y = df[col_label]
        return x, y

    return split_x_y


# simple nan remover
def remove_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    function to remove rows with NaN values
    :param df:
    :return: df with NaNs removed
    """
    return df.dropna()


# add fourier transform features
def create_fourier_transformer(col: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    function to add angle and absolute columns from a fourier fransform
    :param col: column to use for calculation usually close price
    :return: function to add fourier features
    """
    def fourier_transform_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['fft'] = np.fft.fft(np.asarray(df[col].tolist()))
        df['absolute'] = df['fft'].apply(lambda x: np.abs(x))
        df['angle'] = df['fft'].apply(lambda x: np.angle(x))
        return df.drop('fft', axis=1)

    return fourier_transform_features


def cci(df: pd.DataFrame, ndays: int = 20) -> pd.DataFrame:
    """
    function to add CCI features
    :param df:
    :param ndays: number of days to use in calclulations
    :return: df with CCI feature added
    """
    df = df.copy()
    df['TP'] = (df['high'] + df['low'] + df['close']) / 3
    df['sma'] = df['TP'].rolling(ndays).mean()
    df['mad'] = df['TP'].rolling(ndays).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (df['TP'] - df['sma']) / (0.015 * df['mad'])
    return df.drop(['TP', 'sma', 'mad'], axis=1)


# function EMA adder
def create_ema_adder(spans: List[int], col: str = 'close') -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    function to add Exponential Moving Average
    :param spans: list of integers to calculate the EMA and add as features
    :param col: column to use for calculations usually close price
    :return: function to add EMA features
    """
    def ema_adder(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for span in spans:
            df[f'ema_{span}'] = df[col].ewm(span=span).mean().fillna(0)
        return df

    return ema_adder


def create_diff_features(num_diffs: int) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    function to add diffs from close price
    :param num_diffs: number of diffs to cycle through and add as features
    :return: function to add diffs to dataframe
    """
    def diff_features(df: pd.DataFrame) -> pd.DataFrame:
        for i in range(1, num_diffs + 1):
            df[f'diff_close_lag{i}'] = df['close'] - df['close'].shift(i)

        return df

    return diff_features


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    """
    function to add MACD to a dataframe using close price
    :param df:
    :return: dataframe with diffs added as features
    """
    df = df.copy()
    df['macd'] = df[["close"]].ewm(span=12).mean().fillna(0) - df[["close"]].ewm(span=24).mean().fillna(0)
    return df


def add_days_months_years_from_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    function to add the month, year and day of year to dataframe
    :param df:
    :return: dataframe with date features added
    """
    df['day_of_year'] = df.index.dayofyear
    df['month'] = df.index.month
    df['year'] = df.index.year
    return df


def create_sin_cos_transformer(period: int, col: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """
    function to transform int dates into a curcular sin and cos features
    :param period: max period in date
    :param col: data column to transform
    :return: function to add transformed dates to dataframe
    """
    def sin_cos_transformer(df: pd.DataFrame) -> pd.DataFrame:
        df[f'{col}_sin'] = df[col].apply(lambda x: np.sin(x / period * 2 * np.pi))
        df[f'{col}_cos'] = df[col].apply(lambda x: np.cos(x / period * 2 * np.pi))
        return df

    return sin_cos_transformer
