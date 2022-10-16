from typing import Callable, List

import numpy as np
import pandas as pd

from src.algo.create_model import create_pipeline, create_logistic_regression_learner
from src.business_logic.constants import NUM_LAGS


# function to add lags to dataframe
def create_lag_creator(num_lags: int, col_name: str) -> Callable[[pd.core.frame.DataFrame], pd.core.frame.DataFrame]:
    def lag_creator(df: pd.core.frame.DataFrame):
        for num in range(num_lags):
            df[f"{col_name}_lag{num + 1}"] = df[col_name].shift(num)
        return df

    return lag_creator


# function to add the label
def add_label_buy_close(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    df["tomorrow_close"] = df.loc[:, "close"].shift(-1)
    df["label"] = np.where(df.loc[:, "close"] >= df.loc[:, "tomorrow_close"], "SELL", "BUY")
    return df.drop("tomorrow_close", axis=1)


# function to return a dataframe with the columns we want to keep for training
def create_cols_to_keep(list_cols: List[str]) -> Callable[[pd.core.frame.DataFrame], pd.core.frame.DataFrame]:
    def cols_to_keep(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        return df[list_cols]

    return cols_to_keep


# function to split x and y for model
def create_splitter(col_label: str):
    def split_x_y(df):
        df = df.copy()
        x = df.drop(col_label, axis=1)
        y = df[col_label]
        return x, y

    return split_x_y


# simple nan remover
def remove_nans(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    return df.dropna()


# create training preprocessing pipeline
def create_preprocess_pipeline_train(train_data_fetcher):
    preprocess_pipeline_train = create_pipeline([train_data_fetcher,
                                                 create_lag_creator(NUM_LAGS, "close"),
                                                 add_label_buy_close,
                                                 remove_nans,
                                                 create_cols_to_keep(["close", "close_lag1", "close_lag2", "close_lag3",
                                                                      "close_lag4", "close_lag5", "label",
                                                                      ])
                                                 ]
                                                )
    return preprocess_pipeline_train


# create prediction preprocessing pipeline
def create_preprocess_pipeline_predict(predict_data_fetcher):
    preprocess_pipeline_predict = create_pipeline([predict_data_fetcher,
                                                   create_lag_creator(NUM_LAGS, "close"),
                                                   remove_nans,
                                                   create_cols_to_keep(["close", "close_lag1", "close_lag2",
                                                                        "close_lag3", "close_lag4", "close_lag5"])
                                                   ]
                                                  )
    return preprocess_pipeline_predict


def create_predictor(ticker, train_data_fetcher, predict_data_fetcher):
    preprocess_pipeline_train = create_preprocess_pipeline_train(train_data_fetcher)
    preprocess_pipeline_predict = create_preprocess_pipeline_predict(predict_data_fetcher)
    pipeline_lr_creator = create_pipeline_lr_creator(preprocess_pipeline_train)
    pipeline_create_prediction = create_pipeline_create_prediction(preprocess_pipeline_predict, pipeline_lr_creator,
                                                                   ticker)
    return pipeline_create_prediction


# function to create LR pipeline, this pipeline requires the pipeline from the training pipeline function
def create_pipeline_lr_creator(preprocess_pipeline_train):
    pipeline_lr_creator = create_pipeline([preprocess_pipeline_train,
                                           create_splitter('label'),
                                           create_logistic_regression_learner()]
                                          )
    return pipeline_lr_creator


# final prediction pipeline which depends on lr creator and predict preprocessor
def create_pipeline_create_prediction(preprocess_pipeline_predict, pipeline_lr_creator, ticker):
    pipeline_create_prediction = create_pipeline([preprocess_pipeline_predict,
                                                  pipeline_lr_creator(ticker)])
    return pipeline_create_prediction


def create_predictor(ticker, train_data_fetcher, predict_data_fetcher):
    preprocess_pipeline_train = create_preprocess_pipeline_train(train_data_fetcher)
    preprocess_pipeline_predict = create_preprocess_pipeline_predict(predict_data_fetcher)
    pipeline_lr_creator = create_pipeline_lr_creator(preprocess_pipeline_train)
    pipeline_create_prediction = create_pipeline_create_prediction(preprocess_pipeline_predict, pipeline_lr_creator,
                                                                   ticker)
    return pipeline_create_prediction
