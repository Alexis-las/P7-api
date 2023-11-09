import pandas as pd
import pickle
import os
# from P7_Scoring.custom_metrics import business_income
# from P7_Scoring.custom_metrics import business_cost


def load_data(path="input/app_train_final.csv"):
    # Import data
    full_path = os.path.abspath(path)
    data = pd.read_csv(full_path, sep=',', index_col='SK_ID_CURR', low_memory=False)
    return data


def load_model(path="input/modele_Final_LR_business_cost.pickle"):
#def load_model(path="input/modele_LR_business_score.pickle"):
    return pickle.load(open(path, 'rb'))
