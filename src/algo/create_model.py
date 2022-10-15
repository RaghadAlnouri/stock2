from typing import Any, Callable, List
from sklearn.linear_model import LogisticRegression

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
