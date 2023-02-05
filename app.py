from flask import Flask

from train_model import train_model

app = Flask(__name__)

@app.route('/')
def hello_world():
    train_model()

    return 'test'

