from flask import Flask
from src.business_logic.process_query import (
    create_preprocess_pipeline_train,
    create_preprocess_pipeline_predict,
    create_pipeline_lr_creator,
    create_pipeline_create_prediction,
)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return f'Flask has opened a server, to predict use "/get_stock_val/<ticker>"!\n'


@app.route("/get_stock_val/<ticker>", methods=["GET"])
def get_stock_val(ticker: str) -> float:
    preprocess_pipeline_train = create_preprocess_pipeline_train()
    preprocess_pipeline_predict = create_preprocess_pipeline_predict()
    pipeline_lr_creator = create_pipeline_lr_creator(preprocess_pipeline_train)
    pipeline_create_prediction = create_pipeline_create_prediction(
        preprocess_pipeline_predict, pipeline_lr_creator, ticker
    )
    res = pipeline_create_prediction(ticker)[-1]
    return f"{res}"


if __name__ == "__main__":
    # When running python app.py this will run
    # on the cloud a webserver such as Gunicorn will serve the app
    app.run(host="localhost", port=8080, debug=True)

