from typing import Callable
import pandas as pd
import yahoo_fin.stock_info as si
from sklearn.metrics import balanced_accuracy_score

from src.algo import create_pipeline


def create_eval_data_fetcher(ticker: str, train: bool = True) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def eval_data_fetcher(df: pd.DataFrame) -> pd.DataFrame:
        df = df[df['ticker'] == ticker].copy()
        if train:
            df_train = df.loc[:'2022-05-31', :]
            return df_train
        else:
            df_test = df.loc['2022-06-01':, :]
            return df_test

    return eval_data_fetcher


def main():
    df = pd.read_csv('src/evaluations/sp500_stocks_data.csv', index_col=0, parse_dates=True)
    sp500 = si.tickers_sp500()
    train_predictions = []
    test_predictions = []
    # for ticker in sp500:
    train_data_fetcher = create_eval_data_fetcher('AAPL')
    test_data_fetcher = create_eval_data_fetcher('AAPL', train=False)
    train_predictor = create_pipeline.create_predictor(df, train_data_fetcher, train_data_fetcher)
    test_predictor = create_pipeline.create_predictor(df, train_data_fetcher, test_data_fetcher)
    train_predictor(df)

if __name__ == "__main__":
    main()
