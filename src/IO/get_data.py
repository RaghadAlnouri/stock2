from typing import Callable
import yahoo_fin.stock_info as si
import datetime
import pandas as pd


def create_data_fetcher(days: int, last: bool = False) -> Callable[[str], pd.DataFrame]:
    def data_fetcher(ticker: str) -> pd.DataFrame:
        if last:
            now = datetime.datetime.now()
            start_date = now - datetime.timedelta(days=days + 2)
            return si.get_data(ticker, start_date)
        return si.get_data(ticker)

    return data_fetcher


if __name__ == '__main__':
    train_data_fetcher = create_data_fetcher(5, True)
    assert len(train_data_fetcher('ABBV')) == 5
