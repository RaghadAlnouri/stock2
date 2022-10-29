from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# pipeline creator
def create_pipeline(list_functions):
    def pipeline(inputs: Any) -> Any:
        res = inputs
        for function in list_functions:
            res = function(res)
        return res

    return pipeline


# function to train and predict on data
def create_model_learner():
    rf = RandomForestClassifier()

    def train_model_on(training_set):
        X, Y = training_set
        rf.fit(X, Y)

        def predict_model_on(external_data):
            return rf.predict(external_data)

        return predict_model_on

    return train_model_on
