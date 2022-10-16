import pandas as pd
import numpy as np
import configparser

from src.algo.add_features import (
    create_lag_creator,
    create_cols_to_keep,
    add_label_buy_close,
    split_X_Y,
    remove_nans,
)
from src.algo.create_model import create_pipeline, create_logistic_regression_learner
from src.IO.get_data import create_data_fetcher

NUM_LAGS = 5
ROOT_BUCKET = "mat-ycng228-stock2-bucket-proj"


def get_version():
    config = configparser.ConfigParser()
    config.read("application.conf")
    return config["DEFAULT"]["version"]


def get_bucket_name():
    version = get_version()
    return f'{ROOT_BUCKET}_{version.replace(".", "")}'


# create training preprocessing pipeline
def create_preprocess_pipeline_train():
    preprocess_pipeline_train = create_pipeline(
        [
            create_data_fetcher(NUM_LAGS),
            create_lag_creator(NUM_LAGS, "close"),
            add_label_buy_close,
            remove_nans,
            create_cols_to_keep(
                [
                    "close",
                    "close_lag1",
                    "close_lag2",
                    "close_lag3",
                    "close_lag4",
                    "close_lag5",
                    "label",
                ]
            ),
        ]
    )
    return preprocess_pipeline_train


# create prediction preprocessing pipeline
def create_preprocess_pipeline_predict():
    preprocess_pipeline_predict = create_pipeline(
        [
            create_data_fetcher(NUM_LAGS, last=True),
            create_lag_creator(NUM_LAGS, "close"),
            remove_nans,
            create_cols_to_keep(
                [
                    "close",
                    "close_lag1",
                    "close_lag2",
                    "close_lag3",
                    "close_lag4",
                    "close_lag5",
                ]
            ),
        ]
    )
    return preprocess_pipeline_predict


# function to create LR pipeline, this pipeline requires the pipeline from the training pipeline function
def create_pipeline_lr_creator(preprocess_pipeline_train):
    pipeline_lr_creator = create_pipeline(
        [preprocess_pipeline_train, split_X_Y, create_logistic_regression_learner()]
    )
    return pipeline_lr_creator


# final prediction pipeline which depends on lr creator and predict preprocessor
def create_pipeline_create_prediction(
    preprocess_pipeline_predict, pipeline_lr_creator, ticker
):
    pipeline_create_prediction = create_pipeline(
        [preprocess_pipeline_predict, pipeline_lr_creator(ticker)]
    )
    return pipeline_create_prediction
