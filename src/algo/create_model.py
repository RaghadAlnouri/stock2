from typing import Any, Callable, List
from sklearn.linear_model import LogisticRegression

from src.algo.add_features import create_splitter, create_preprocess_pipeline_train, create_preprocess_pipeline_predict


# pipeline creator
def create_pipeline(list_functions):
    def pipeline(input: Any) -> Any:
        res = input
        for function in list_functions:
            res = function(res)
        return res

    return pipeline


# function to train and predict on data
def create_logistic_regression_learner():
    lr = LogisticRegression()

    def train_lr_on(training_set):
        X, Y = training_set
        lr.fit(X, Y)

        def predict_lr_on(external_data):
            return lr.predict(external_data)

        return predict_lr_on

    return train_lr_on


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


def create_prediction_pipeline(ticker, train_data_fetcher, predict_data_fetcher):
    preprocess_pipeline_train = create_preprocess_pipeline_train(train_data_fetcher)
    preprocess_pipeline_predict = create_preprocess_pipeline_predict(predict_data_fetcher)
    pipeline_lr_creator = create_pipeline_lr_creator(preprocess_pipeline_train)
    pipeline_create_prediction = create_pipeline_create_prediction(preprocess_pipeline_predict, pipeline_lr_creator,
                                                                   ticker)
    return pipeline_create_prediction


