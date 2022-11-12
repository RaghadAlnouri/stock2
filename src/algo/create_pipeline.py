from src.algo.add_features import create_lag_creator, add_label_buy_close, remove_nans, create_cols_to_keep, \
    create_splitter, cci, create_ema_adder, create_diff_features, \
    add_days_months_years_from_index, create_sin_cos_transformer, add_macd, create_fourier_transformer
from src.algo.create_model import create_pipeline, create_model_learner
from src.business_logic.constants import NUM_LAGS


# create training preprocessing pipeline
def create_preprocess_pipeline_train(train_data_fetcher):
    preprocess_pipeline_train = create_pipeline([train_data_fetcher,
                                                 add_days_months_years_from_index,
                                                 create_sin_cos_transformer(365, 'day_of_year'),
                                                 create_sin_cos_transformer(12, 'month'),
                                                 create_lag_creator(NUM_LAGS),
                                                 create_fourier_transformer('close'),
                                                 add_macd,
                                                 cci,
                                                 create_ema_adder([10, 20, 30]),
                                                 create_diff_features(5),
                                                 add_label_buy_close,
                                                 remove_nans,
                                                 create_cols_to_keep(["close", 'year', 'month_sin', 'month_cos',
                                                                      'day_of_year_sin', 'day_of_year_cos',
                                                                      "CCI", "angle", "absolute", "macd",
                                                                      "close_lag1", "close_lag5",
                                                                      "close_lag9", "close_lag13", "close_lag17",
                                                                      "close_lag21", "close_lag25", "ema_10",
                                                                      "ema_20", "ema_30", 'diff_close_lag1',
                                                                      'diff_close_lag2', 'diff_close_lag3',
                                                                      'diff_close_lag4', 'diff_close_lag5',
                                                                      "label", ])

                                                 ]
                                                )
    return preprocess_pipeline_train


# create prediction preprocessing pipeline
def create_preprocess_pipeline_predict(predict_data_fetcher):
    preprocess_pipeline_predict = create_pipeline([predict_data_fetcher,
                                                   add_days_months_years_from_index,
                                                   create_sin_cos_transformer(365, 'day_of_year'),
                                                   create_sin_cos_transformer(12, 'month'),
                                                   create_lag_creator(NUM_LAGS, "close"),
                                                   create_fourier_transformer('close'),
                                                   add_macd,
                                                   cci,
                                                   create_ema_adder([10, 20, 30]),
                                                   create_diff_features(5),
                                                   remove_nans,
                                                   create_cols_to_keep(["close", 'year', 'month_sin', 'month_cos',
                                                                        'day_of_year_sin', 'day_of_year_cos',
                                                                        "CCI", "angle", "absolute", "macd",
                                                                        "close_lag1", "close_lag5",
                                                                        "close_lag9", "close_lag13", "close_lag17",
                                                                        "close_lag21", "close_lag25", "ema_10",
                                                                        "ema_20", "ema_30", 'diff_close_lag1',
                                                                        'diff_close_lag2', 'diff_close_lag3',
                                                                        'diff_close_lag4', 'diff_close_lag5', ])

                                                   ]
                                                  )
    return preprocess_pipeline_predict


# function to create LR pipeline, this pipeline requires the pipeline from the training pipeline function
def create_pipeline_model_creator(preprocess_pipeline_train):
    pipeline_model_creator = create_pipeline([preprocess_pipeline_train,
                                              create_splitter('label'),
                                              create_model_learner(False)]
                                             )
    return pipeline_model_creator


# final prediction pipeline which depends on lr creator and predict preprocessor
def create_pipeline_create_prediction(preprocess_pipeline_predict, pipeline_model_creator, ticker):
    pipeline_create_prediction = create_pipeline([preprocess_pipeline_predict,
                                                  pipeline_model_creator(ticker)])
    return pipeline_create_prediction


def create_predictor(ticker, train_data_fetcher, predict_data_fetcher):
    preprocess_pipeline_train = create_preprocess_pipeline_train(train_data_fetcher)
    preprocess_pipeline_predict = create_preprocess_pipeline_predict(predict_data_fetcher)
    pipeline_model_creator = create_pipeline_model_creator(preprocess_pipeline_train)
    pipeline_create_prediction = create_pipeline_create_prediction(preprocess_pipeline_predict, pipeline_model_creator,
                                                                   ticker)
    return pipeline_create_prediction
