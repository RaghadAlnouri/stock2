from typing import Any
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline


# pipeline creator
def create_pipeline(list_functions):
    def pipeline(inputs: Any) -> Any:
        res = inputs
        for function in list_functions:
            res = function(res)
        return res

    return pipeline


# function to train and predict on data
def create_model_learner(scale: bool = True):
    steps = list()
    if scale:
        steps.append(('scaler', MaxAbsScaler()))
    steps.append(('model', LogisticRegression()))
    model = Pipeline(steps)

    def train_model_on(training_set):
        X, Y = training_set
        model.fit(X, Y)

        def predict_model_on(external_data):
            return model.predict(external_data)

        return predict_model_on

    return train_model_on
