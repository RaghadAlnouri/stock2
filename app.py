from flask import Flask

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return f'Flask has opened a server, try "/another_page"!\n'

@app.route('/another_page', methods=['GET'])
