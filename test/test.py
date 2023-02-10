import requests
from app.torch_utils import transform_smile, get_prediction
import pytest
#resp = requests.post("http://127.0.0.1:5000/predict")
resp = requests.post("http://localhost:5000/predict", data={"smile":"Clc1ccc(cc1)C(c2ccc(Cl)cc2)C(Cl)(Cl)Cl"})
print(resp.result)



def test_main_predict():
    try:
        assert resp.status_code == 200
        reponse_json = resp.json()
        assert reponse_json['error'] == False
        assert isinstance(reponse_json['results']['pred'], str)

    except AssertionError:
        print(resp.status_code)
        print(resp.json())
        raise


def login(client, username, password):
    return client.post('/login', data=dict(
        username=username,
        password=password
    ), follow_redirects=True)


def logout(client):
    return client.get('/logout', follow_redirects=True)

def setUp(self, client):
    # Login here
    client.post('/login', data={'username': 'a', 'password': 'b'})

def test_foo_with_client(self, client):
    # Check if the session is logged in
    self.assertIn('userId')    # Assertion succeeds

def tearDown(self, client):
    # Logout here, though there isn't really a need - since session is cleared for the next test method
    client.get('/logout')
