from src.algo.add_features import create_lag_creator, add_label_buy_close, remove_nans, create_cols_to_keep, \
    create_splitter, create_fourier_transformer, cci
from src.algo.create_model import create_pipeline, create_logistic_regression_learner
from src.business_logic.constants import NUM_LAGS


# create training preprocessing pipeline
def create_preprocess_pipeline_train(train_data_fetcher):
    preprocess_pipeline_train = create_pipeline([train_data_fetcher,
                                                 create_lag_creator(NUM_LAGS, "close"),
                                                 create_fourier_transformer('close'),
                                                 cci,
                                                 add_label_buy_close,
                                                 remove_nans,
                                                 create_cols_to_keep(["close", "absolute", "CCI", "angle", "close_lag1",
                                                                      "close_lag2", "close_lag3", "close_lag4",
                                                                      "close_lag5", "label",
                                                                      ])
                                                 ]
                                                )
    return preprocess_pipeline_train


# create prediction preprocessing pipeline
def create_preprocess_pipeline_predict(predict_data_fetcher):
    preprocess_pipeline_predict = create_pipeline([predict_data_fetcher,
                                                   create_lag_creator(NUM_LAGS, "close"),
                                                   create_fourier_transformer('close'),
                                                   cci,
                                                   remove_nans,
                                                   create_cols_to_keep(["close", "absolute", "angle", "CCI",
                                                                        "close_lag1", "close_lag2", "close_lag3",
                                                                         "close_lag4", "close_lag5"])

                                                   ]
                                                  )
    return preprocess_pipeline_predict


# function to create LR pipeline, this pipeline requires the pipeline from the training pipeline function
def create_pipeline_lr_creator(preprocess_pipeline_train):
    pipeline_lr_creator = create_pipeline([preprocess_pipeline_train,
                                           create_splitter('label'),
                                           create_logistic_regression_learner()]
                                          )
    return pipeline_lr_creator


# final prediction pipeline which depends on lr creator and predict preprocessor
def create_pipeline_create_prediction(preprocess_pipeline_predict, pipeline_lr_creator, ticker):
    pipeline_create_prediction = create_pipeline([preprocess_pipeline_predict,
                                                  pipeline_lr_creator(ticker)])
    return pipeline_create_prediction


def create_predictor(ticker, train_data_fetcher, predict_data_fetcher):
    preprocess_pipeline_train = create_preprocess_pipeline_train(train_data_fetcher)
    preprocess_pipeline_predict = create_preprocess_pipeline_predict(predict_data_fetcher)
    pipeline_lr_creator = create_pipeline_lr_creator(preprocess_pipeline_train)
    pipeline_create_prediction = create_pipeline_create_prediction(preprocess_pipeline_predict, pipeline_lr_creator,
                                                                   ticker)
    return pipeline_create_prediction
