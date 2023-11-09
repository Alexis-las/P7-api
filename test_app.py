import pytest
from app import app


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


test_data = [(267240), (412292), (111788)]


@pytest.mark.parametrize("SK_ID_CURR", test_data)
def test_get_data_by_id(SK_ID_CURR, client):
    response = client.get(f'/client_data/{SK_ID_CURR}')
    assert response.status_code == 200
    data = response.get_json()
    assert 'data' in data


def test_get_data_by_invalid_id(client):
    response = client.get('/client_data/9999')
    assert response.status_code == 404
    data = response.get_json()
    assert 'error' in data


test_score = [
    (232380, 58.00),
    (180439, 47.76),
    (200709, 51.75)
    ]


@pytest.mark.parametrize("SK_ID_CURR, expected", test_score)
def test_get_score(SK_ID_CURR, expected, client):
    response = client.get(f'/client_score/{SK_ID_CURR}')
    assert response.status_code == 200
    data = response.get_json()
    assert 'score' in data
    assert data['score'] == expected

