import pandas as pd


# 1 import data from sp500_stocks_data.csv

# 2 get list of sp500 from yahoo_fin module

# 3 loop through list creating a model for each ticker, training on data < 2022-06-01 and predicting on 2022-06-01 -
# 2022-09-01

# 4 open file to add evaluation data of the model using train and test balanced accuracy

def eval_data_fetcher_train(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df_train = df[df['ticker'] == ticker]

def main():
