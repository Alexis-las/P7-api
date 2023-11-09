import json
import pandas as pd
from flask import Flask, request, jsonify, make_response, render_template
from P7_Scoring.input import load_data, load_model
from P7_Scoring.custom_metrics import business_income
from P7_Scoring.custom_metrics import business_cost
from lime import lime_tabular
import base64
from io import BytesIO
import matplotlib
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import seaborn as sns
matplotlib.use('Agg')

app = Flask(__name__)
app.config["DEBUG"] = True

# ------ Param config
param = {'business_threshold': 44.44,
         'feature_dist_num_bins': 50}

# ------ Load Data + Model
data = load_data()
target = data['TARGET']
data.drop(columns=['TARGET'], inplace=True)
model = load_model()


@app.route('/init', methods=['GET'])
def init_run():
    return {'client_list': list(data.index.sort_values()[:]),
            'feature_list': list(data.columns),
            'business_threshold': param['business_threshold'],
            }


@app.route('/client_data/<int:id>', methods=['GET'])
def get_data_by_id(id: int):
    if verif_client_id(id):
        client_data = data.to_dict(orient='index')[id]
        return {'data': client_data}
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
        score = model.predict_proba(data.loc[data.index == id])
        return jsonify({'score': round(score[0][1] * 100, 2)})
    else:
        return jsonify({'error': 'Client does not exist'}), 404


@app.route('/client_explain/', methods=['GET'])  # <int:id>-<int:nb_feature>
def get_explanation():  # id: int, nb_feature=None
    client_id = request.args.get('id', type=int)
    nb_feature = request.args.get('nb_feat', type=int)
    if nb_feature is None: nb_feature = len(data.columns)
    if verif_client_id(client_id):
        explainer = lime_tabular.LimeTabularExplainer(data,
                                                      mode="classification",
                                                      feature_names=data.columns,
                                                      discretize_continuous=False,
                                                      class_names=['Ok', 'Risky'],
                                                      verbose=True
                                                      )
        explanation = explainer.explain_instance(data.loc[client_id].values,
                                                 model.predict_proba,
                                                 num_features=nb_feature)
        expl_list = explanation.as_list()
        explanation_feature = [expl_list[i][0] for i in range(len(expl_list))]
        explanation_value = [expl_list[i][1] * 100 for i in range(len(expl_list))]

        fig = Figure()
        ax = fig.subplots()
        sns.barplot(x=explanation_value, y=explanation_feature, ax=ax)
        ax.set_title('Local features importance')
        fig.tight_layout()
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        graph = base64.b64encode(buf.getbuffer()).decode("ascii")
        return {'graph': f"<img src='data:image/png;base64,{graph}'/>",
                'feat': explanation_feature}
    else:
        return jsonify({'error': 'Client does not exist',
                        'client_id': client_id}), 404


@app.route('/model_importance/', methods=['GET'])  # <int:id>-<int:nb_feature>
def get_model_importance():  # id: int, nb_feature=None
    nb_feature = request.args.get('nb_feat', type=int)
    if nb_feature is None: nb_feature = len(data.columns)
    importance = pd.Series(model['reg'].coef_[0], index=data.columns)
    importance = importance.sort_values(ascending=False, key=abs)[:nb_feature]

    fig = Figure()
    ax = fig.subplots()
    sns.barplot(x=importance.values, y=importance.index, ax=ax)
    ax.set_title('Global features importance')
    fig.tight_layout()
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    graph = base64.b64encode(buf.getbuffer()).decode("ascii")
    return {'graph': f"<img src='data:image/png;base64,{graph}'/>"}


@app.route('/features_dist/', methods=['GET'])
def get_features_dist():
    client_id = request.args.get('id', type=int)
    feature = request.args.get('feat', type=str)

    # importance = pd.Series(model.best_estimator_['reg'].coef_[0], index=data.columns)
    # data_col = [(col.replace(' ', '_').replace('-', '_')).upper() for col in data.columns]

    #
    df = pd.concat([data[feature], target, target.map({0: "Ok", 1: "Risky"})], axis=1)
    df.columns = [feature, 'TARGET', 'TARGET_label']

    # Calculate proportion of target=Risky in each bin
    df['bin'] = pd.cut(df[feature], param['feature_dist_num_bins'])
    bin_proportions = df.groupby(by='bin', observed=False)['TARGET'].mean()*100
    # Calculate center of each interval
    bin_centers = [interval.mid for interval in bin_proportions.index]

    fig = Figure()
    ax1 = fig.add_subplot(1, 1, 1)
    # Axe 1: Feature histogram
    sns.histplot(x=df[feature], hue=df['TARGET_label'], bins=param['feature_dist_num_bins'], ax=ax1)
    ax1.set_xlabel(feature)
    # ax1.legend(loc='upper right')
    # Axe 2: Proportion of 'Risky client' + Client position
    ax2 = ax1.twinx()
    sns.lineplot(x=bin_centers, y=bin_proportions, color='darkorange',
                 ax=ax2)  # color='red', marker='o', label='Proportion target=1',
    ax2.vlines(x=df[feature][client_id],
               ymin=0,
               ymax=max(bin_proportions),
               linestyles='--')
    ax2.set_ylabel('Risky Proportion (in %)', color='darkorange')
    # ax2.legend(loc='upper left')
    ax1.set_title(f"Distribution de {feature} par type de client")
    fig.tight_layout()
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    graph = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{graph}'/>"


if __name__ == "__main__":
    app.run(host="localhost", port=8000)



@app.route('/features', methods=['GET'])
def get_features():
    importance = pd.Series(model.best_estimator_['reg'].coef_[0], index=data.columns)
    return {'list': list(model.best_estimator_['reg'].coef_[0])}

