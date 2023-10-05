import json
import pandas as pd
from flask import Flask, render_template, jsonify, request
import json
from flask import Flask, jsonify, request

from P7_Scoring.input import load_data, load_model


app = Flask(__name__)
app.config["DEBUG"] = True


data = load_data()
model = load_model()


@app.route('/')
def hello():
    return "<h1>ASK</h1><p>Hello, World! this application running on port=8000</p>"


@app.route('/client_data', methods=['GET'])
def get_data():
    return jsonify(data)

@app.route('/client_data/<int:id>', methods=['GET'])
def get_data_by_id(id: int):
    data = get_data(id)
    if data is None:
        return jsonify({'error': 'Client does not exist'}), 404
    return jsonify(data)

def get_data(id):
    return next(data.loc[data.index == id]), None)



@app.route('/client_score', methods=['GET'])
def get_score():
    return jsonify(employees)


@app.route('/dashboard/')
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(host="localhost", port=8000)
