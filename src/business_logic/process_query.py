import pandas as pd
import numpy as np
import configparser


def get_version():
    config = configparser.ConfigParser()
    config.read("application.conf")
    return config["DEFAULT"]["version"]


def get_bucket_name():
    version = get_version()
    return f'{ROOT_BUCKET}_{version.replace(".", "")}'


def create_business_logic(ticker):
    # check for model in bucket