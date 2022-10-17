from flask import Flask
import git

from src.business_logic.process_query import business_logic_get_model, get_prediction

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return f'Flask has opened a server, to predict use "/get_stock_val/<ticker>"!\n'


@app.route("/get_stock_val/<ticker>", methods=["GET"])
def get_stock_val(ticker: str) -> str:
    model = business_logic_get_model(ticker)
    prediction = get_prediction(model, ticker)
    return f'{prediction}'


@app.route('/getversion/')
def getversion():
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    return f'{sha}\n'


if __name__ == "__main__":
    # When running python app.py this will run
    # on the cloud a webserver such as Gunicorn will serve the app
    app.run(host="localhost", port=8080, debug=True)
