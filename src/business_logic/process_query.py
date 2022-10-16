import configparser

from src.IO.get_data import create_data_fetcher
from src.IO.storage_tools import get_model_from_bucket
from src.algo.add_features import create_preprocess_pipeline_train, create_preprocess_pipeline_predict
from src.business_logic.constants import NUM_LAGS, ROOT_BUCKET


def get_version():
    config = configparser.ConfigParser()
    config.read("application.conf")
    return config["DEFAULT"]["version"]


def get_bucket_name():
    version = get_version()
    return f'{ROOT_BUCKET}_{version.replace(".", "")}'


def get_model_filename_from_ticker(ticker):
    return f'{ticker}.pkl'





def create_business_logic(ticker):
    bucket_name = get_bucket_name()
    model_filename = get_model_filename_from_ticker(ticker)
    model = get_model_from_bucket(model_filename, bucket_name)
    if model is None:
        train_data_fetcher = create_data_fetcher(NUM_LAGS)
        predict_data_fetcher = create_data_fetcher(NUM_LAGS, True)


