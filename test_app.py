import pytest
from app import app

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
import pickle
matplotlib.use('Agg')

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_init_run(client):
    response = client.get('/init')
    assert response.status_code == 200
    data = response.get_json()
    assert 'client_list' in data
    assert 'feature_list' in data
    assert 'business_threshold' in data

def test_get_data_by_id(client):
    response = client.get('/client_data/1')
    assert response.status_code == 200
    data = response.get_json()
    assert 'data' in data

def test_get_data_by_invalid_id(client):
    response = client.get('/client_data/9999')
    assert response.status_code == 404
    data = response.get_json()
    assert 'error' in data

# Ajoutez d'autres tests pour les autres routes ici

if __name__ == '__main__':
    pytest.main()