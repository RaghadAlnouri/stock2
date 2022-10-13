from flask import Flask

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello():
    return f'Flask has opened a server, try "/another_page"!\n'


@app.route("/get_stock_val/<ticker>", methods=["GET"])
def get_stock_val(ticker):
    return


if __name__ == "__main__":
    # When running python app.py this will run
    # on the cloud a webserver such as Gunicorn will serve the app
    app.run(host="localhost", port=8080, debug=True)

