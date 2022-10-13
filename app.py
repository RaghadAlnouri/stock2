from flask import Flask
from src.business_logic.process_query import predict

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return f'Flask has opened a server, to predict use "/get_stock_val/<ticker>"!\n'


@app.route("/get_stock_val/<ticker>", methods=["GET"])
def get_stock_val(ticker: str) -> float:
    return predict(ticker)


if __name__ == "__main__":
    # When running python app.py this will run
    # on the cloud a webserver such as Gunicorn will serve the app
    app.run(host="localhost", port=8080, debug=True)

