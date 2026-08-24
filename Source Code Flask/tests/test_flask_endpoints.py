import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app


class DummyModel:
    def predict(self, df):
        return [0.1 for _ in range(7)]


def test_health_and_versions_endpoints(monkeypatch):
    app = create_app()
    client = app.test_client()
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'status' in data and data['status'] == 'ok'

    resp = client.get('/versions')
    assert resp.status_code == 200
    # versions may be empty
    _ = resp.get_json()

def test_predict_endpoint(monkeypatch):
    app = create_app()
    client = app.test_client()
    # Patch load_model to return a dummy model
    import app as module
    class M:
        def predict(self, df):
            return [0.5 for _ in range(7)]
    monkeypatch.setattr(module, 'load_model', lambda version, base_path=None: M())
    encoder_last_60 = [{"close": 1950.0, "open": 1949.0, "tickvol": 50000, "usd_close": 104.0, "usd_open": 103.9, "gspc_close": 5200.0, "gspc_open": 5199.0, "oil_dcoilwtico": 78.0, "fed_rate_pct": 0.0525, "cpi_cpiaucsl": 310.0} for _ in range(60)]
    future_7 = [{"usd_close": 104.0, "usd_open": 103.9, "gspc_close": 5200.0, "gspc_open": 5199.0, "oil_dcoilwtico": 78.0, "fed_rate_pct": 0.0525, "cpi_cpiaucsl": 310.0} for _ in range(7)]
    payload = {"version": "version_0", "encoder_last_60": encoder_last_60, "future_7": future_7}
    resp = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'predictions' in data
    assert len(data['predictions']) == 7


def test_api_market_data_endpoint(monkeypatch):
    app = create_app()
    client = app.test_client()
    import app as module

    def fake_fetch_real_time_data():
        return [{
            'usd_close': 104.0,
            'usd_open': 104.0,
            'gspc_close': 5200.0,
            'gspc_open': 5200.0,
            'oil_dcoilwtico': 78.0,
            'fed_rate_pct': 0.0525,
            'cpi_cpiaucsl': 310.0
        } for _ in range(60)]

    monkeypatch.setattr(module, 'fetch_real_time_data', fake_fetch_real_time_data)
    resp = client.get('/api/market-data')
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['success'] is True
    assert 'payload' in data
    assert len(data['payload']['encoder_last_60']) == 60
    assert len(data['payload']['future_7']) == 7


def test_predict_endpoint_invalid_input():
    app = create_app()
    client = app.test_client()
    # invalid encoder_last_60 length
    payload = {
        "version": "version_0",
        "encoder_last_60": [{"USD": 1.0}] * 5,
        "future_7": [{"USD": 1.0}] * 7,
    }
    resp = client.post('/predict', data=json.dumps(payload), content_type='application/json')
    assert resp.status_code == 400

def test_status_endpoint():
    app = create_app()
    client = app.test_client()
    resp = client.get('/status')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'versions' in data
    assert 'full_inference_available' in data
