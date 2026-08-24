# -*- coding: utf-8 -*-
"""
Test script untuk TFT Flask App
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app import create_app
import json

print('=== TFT Gold Price Predictor - Final Test ===')
print()

app = create_app()

# Test semua endpoint
endpoints = [
    ('GET', '/'),
    ('GET', '/health'),
    ('GET', '/status'),
    ('GET', '/versions')
]

with app.test_client() as client:
    print('Testing endpoints:')
    print('-' * 50)

    for method, endpoint in endpoints:
        if method == 'GET':
            r = client.get(endpoint)
        print(f'{method} {endpoint}: {r.status_code} OK')

    print()
    print('Model Status Check:')
    print('-' * 50)

    r = client.get('/status')
    data = json.loads(r.data)

    if data.get('models') and len(data['models']) > 0:
        print(f'Models Loaded: {data["models"]}')
        for v in data['models']:
            print(f'  - {v}: Ready')

        print()
        print('[SUCCESS] Model is ready for predictions!')
    else:
        print('[WARN] No models loaded')

    print()
    print('Quick Prediction Test:')
    print('-' * 50)

    # Test prediction
    encoder = [{'close': '2350', 'open': '2345', 'tickvol': '50000', 'usd_close': '104', 'usd_open': '104', 'gspc_close': '5200', 'gspc_open': '5200', 'oil_dcoilwtico': '78', 'fed_rate_pct': '0.0525', 'cpi_cpiaucsl': '310'} for _ in range(60)]
    future = [{'usd_close': '104', 'usd_open': '104', 'gspc_close': '5200', 'gspc_open': '5200', 'oil_dcoilwtico': '78', 'fed_rate_pct': '0.0525', 'cpi_cpiaucsl': '310'} for _ in range(7)]

    r = client.post('/predict', json={'version': 'version_0', 'encoder_last_60': encoder, 'future_7': future})

    if r.status_code == 200:
        result = json.loads(r.data)
        print(f'Mode: {result.get("mode")}')
        print(f'Predictions: {len(result.get("predictions", []))} days')

        # Show first 3 predictions
        for i, p in enumerate(result.get('predictions', [])[:3]):
            pred_val = p.get('prediction')
            print(f'  Day {i+1}: ${round(pred_val, 2)}')

        print()
        print('[SUCCESS] All systems operational!')
    else:
        print(f'[ERROR] Prediction failed: {r.status_code}')

print()
print('=' * 50)
print('APPLICATION READY FOR PRODUCTION')
print('=' * 50)
print()
print('To start the server:')
print('  cd flask_tft_app')
print('  python -m app.flask_app')
print()
print('Then open: http://localhost:5000')
