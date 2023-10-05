import pandas as pd
import pickle

def load_data(path = "Input/app_train_final_20.csv"):
    # Import data
    return pd.read_csv(path, sep=',', index_col='SK_ID_CURR', low_memory=False)

def load_model(path = "Input/modele_LGBM_business_score.pickle"):
    return pickle.load(open(path, 'rb'))
