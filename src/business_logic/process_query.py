import configparser

import dill

from src.IO.get_data import create_data_fetcher
from src.IO.storage_tools import get_model_from_bucket, upload_file_to_bucket, create_bucket
from src.algo.add_features import create_predictor
from src.business_logic.constants import NUM_LAGS, ROOT_BUCKET


def get_version():
    config = configparser.ConfigParser()
    config.read("application.conf")
    return config["DEFAULT"]["version"]


def get_bucket_name() -> str:
    version = get_version()
    return f'{ROOT_BUCKET}_{version.replace(".", "")}'


def get_model_filename_from_ticker(ticker):
    return f'{ticker}.dill'


def business_logic_get_model(ticker):
    bucket_name = get_bucket_name()
    create_bucket(bucket_name)
    model_filename = get_model_filename_from_ticker(ticker)
    model = get_model_from_bucket(model_filename, bucket_name)
    if model is None:
        train_data_fetcher = create_data_fetcher(NUM_LAGS)
        predict_data_fetcher = create_data_fetcher(NUM_LAGS, True)
        model = create_predictor(ticker, train_data_fetcher, predict_data_fetcher)
        with open(model_filename, 'wb') as f:
            dill.dump(model, f)
        upload_file_to_bucket(model_filename, bucket_name)
    return model


def get_prediction(model, ticker):
    return model(ticker)[-1]
