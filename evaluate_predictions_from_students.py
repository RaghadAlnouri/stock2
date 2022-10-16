import pandas as pd
import requests
from yahoo_fin import stock_info as si
import random
import logging
from datetime import date
from pyspark import SparkContext

random.seed(42)

def get_snp500_list(size = 5):

    return random.sample(si.tickers_sp500(),size)


def get_prediction(query):
    log = logging.getLogger()
    log.warning(f'running {query}')
    name, url, ticker = query
    try:
        response = requests.get(url.replace('<ticker>', ticker))
        pred = response.content.lower().decode('UTF-8').strip()
    except Exception as e:
        log.warning(f'{e}')
        pred = None
    log.warning(f'results: {name, ticker, pred}')
    return name, ticker, pred


if __name__ == '__main__':
    N = 5
    spark = True
    df = pd.read_csv('data/endpoints.csv')
    list_ticker = get_snp500_list(N)
    queries = []
    for ticker in list_ticker:
        # i = 0
        for _, row in df.iterrows():
            # if i == 2:
            #     break
            queries.append((row['name'], row['url'], ticker))
            # i += 1
    random.shuffle(queries)
    if spark:
        sc = SparkContext('local[*]')
        rdd = sc.parallelize(queries)
        preds = rdd.map(get_prediction).collect()
    else:
        preds = map(get_prediction, queries)

    predictions = pd.DataFrame(columns=['name', 'ticker', 'prediction'], data=preds)
    date_formated = date.today().strftime("%d-%m-%Y")
    predictions.to_csv(f'predictions-{date_formated}.csv')