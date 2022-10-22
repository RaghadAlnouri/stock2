import pandas as pd


# 1 import data from sp500_stocks_data.csv

# 2 get list of sp500 from yahoo_fin module

# 3 loop through list creating a model for each ticker, training on data < 2022-06-01 and predicting on 2022-06-01 -
# 2022-09-01

# 4 open file to add evaluation data of the model using train and test balanced accuracy

def eval_data_fetcher(df: pd.DataFrame, ticker: str, train: bool = True) -> pd.DataFrame:
    df = df[df['ticker'] == ticker].copy()
    if train:
        df_train = df.loc[:'2022-06-01', :]
        return df_train
    else:
        df_test = df.loc['2022-06-01':, :]
        return df_test


def main():
    df = pd.read_csv('evaluations/sp500_stocks_data.csv', index_col=0, parse_dates=True)
    print(df.shape)
    print(df.loc['2022-05-29':'2022-05-31'])


if __name__ == "__main__":
    main()
