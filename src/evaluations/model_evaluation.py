from typing import Callable
import pandas as pd
import numpy as np
import csv
import yahoo_fin.stock_info as si
from sklearn.metrics import balanced_accuracy_score

from src.algo import create_pipeline
from src.algo.create_pipeline import create_preprocess_pipeline_train


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
    train_balanced_accuracy_score = list()
    test_balanced_accuracy_score = list()

    for ticker in sp500:
        # getting predictions for train and test
        train_data_fetcher = create_eval_data_fetcher(ticker)
        test_data_fetcher = create_eval_data_fetcher(ticker, train=False)

        if train_data_fetcher(df).shape[0] == 0:
            continue

        train_predictor = create_pipeline.create_predictor(df, train_data_fetcher, train_data_fetcher)
        test_predictor = create_pipeline.create_predictor(df, train_data_fetcher, test_data_fetcher)
        train_predictions = train_predictor(df)
        test_predictions = test_predictor(df)

        # get labels after fitting through preprocessor
        train_preprocessor = create_preprocess_pipeline_train(train_data_fetcher)
        test_preprocessor = create_preprocess_pipeline_train(test_data_fetcher)

        train_labels = train_preprocessor(df)['label']
        test_labels = test_preprocessor(df)['label']

        # get balanced accuracy score
        train_balanced_accuracy_score.append(balanced_accuracy_score(train_labels, train_predictions))
        test_balanced_accuracy_score.append(balanced_accuracy_score(test_labels, test_predictions))

    model_name = 'Baseline Logistic Regression Model'
    with open('src/evaluations/balanced_accuracy_scores.csv', 'a') as csvfile:
        score_writer = csv.writer(csvfile, delimiter=',')
        score_writer.writerow([model_name, np.mean(train_balanced_accuracy_score), np.mean(test_balanced_accuracy_score)])


if __name__ == "__main__":
    main()
