from typing import Any

from sklearn.neighbors import KNeighborsClassifier


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
    model = KNeighborsClassifier()

    def train_model_on(training_set):
        X, Y = training_set
        model.fit(X, Y)

        def predict_model_on(external_data):
            return model.predict(external_data)

        return predict_model_on

    return train_model_on
