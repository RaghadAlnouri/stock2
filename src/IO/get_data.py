import yahoo_fin.stock_info as si
import datetime


def create_data_fetcher(last: bool = False):
    def data_fetcher(ticker: str):
        if last:
            now = datetime.datetime.now()
            start_date = now - datetime.timedelta(days=30)
            return si.get_data(ticker, start_date)
        return si.get_data(ticker)
    return data_fetcher