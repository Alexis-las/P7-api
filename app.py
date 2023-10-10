import json
import pandas as pd
from flask import Flask, render_template, jsonify, request

from P7_Scoring.input import load_data, load_model

import lime
from lime import lime_tabular


app = Flask(__name__)
app.config["DEBUG"] = True


data = load_data()
model = load_model()


@app.route('/')
def hello():
    return "<h1>P7 FLASK API launched</h1><p>Database and model loaded... </p>"


@app.route('/client', methods=['GET'])
def get_client_list():
    # return json.loads(pd.Series(data.index.sort_values()).to_json())
    return jsonify(pd.Series(data.index.sort_values()).to_dict())


#
# @app.route('/client_data', methods=['GET'])
# def get_data_all():
#    return jsonify(data)


@app.route('/client_data/<int:id>', methods=['GET'])
def get_data_by_id(id: int):
    if verif_client_id(id):
        client_data = json.loads(data.loc[data.index == id].to_json())
        #return jsonify({'data': client_data})
        return jsonify(client_data)
    else:
        return jsonify({'error': 'Client does not exist'}), 404


def verif_client_id(client_id):
    if client_id in data.index:
        return True
    else:
        return False


@app.route('/client_score/<int:id>', methods=['GET'])
def get_score(id: int):
    if verif_client_id(id):
        score = model.predict_proba(data.loc[data.index == id].drop(columns=['TARGET']))
        return jsonify({'score': round(score[0][1]*100,2)}), 200
    else:
        return jsonify({'error': 'Client does not exist'}), 404


# Explanation LIME
@app.route('/client_explain/<int:id>', methods=['GET'])
def get_explanation(id: int):
    if verif_client_id(id):
        explainer = lime_tabular.LimeTabularExplainer(data.drop(columns=['TARGET']),
                                                      mode="classification",
                                                      feature_names=data.drop(columns=['TARGET']).columns,
                                                      discretize_continuous=False,
                                                      verbose=True
                                                     )
        explanation = explainer.explain_instance(data.drop(columns=['TARGET']).loc[id].values,
                                                 model.predict_proba,
                                                 num_features=10)
        return (explanation.as_html()), 200
    else:
        return jsonify({'error': 'Client does not exist'}), 404


@app.route('/dashboard/')
def dashboard():
    return render_template("dashboard.html")


if __name__ == "__main__":
    app.run(host="localhost", port=8000)
