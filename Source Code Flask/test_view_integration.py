#!/usr/bin/env python
"""Test View Integration"""
import sys
sys.path.insert(0, '.')
from app import create_app

app = create_app()

with app.test_client() as client:
    # Check HTML components
    rv = client.get('/')
    html = rv.data.decode('utf-8')

    print('=== VIEW INTEGRATION CHECK ===')
    has_chart = 'id="chart"' in html
    has_table = 'resultTable' in html
    has_button = 'btnPredict' in html
    has_status_api = 'fetch(/status)' in html
    has_predict_api = 'fetch(/predict)' in html
    has_plotly = 'plotly' in html

    print(f'1. Index page loads: {rv.status_code == 200}')
    print(f'2. Has chart div: {has_chart}')
    print(f'3. Has result table: {has_table}')
    print(f'4. Has predict button: {has_button}')
    print(f'5. Fetch status API: {has_status_api}')
    print(f'6. Fetch predict API: {has_predict_api}')
    print(f'7. Has Plotly: {has_plotly}')

    # Test predict
    result = client.post('/predict', json={
        'version': 'version_0',
        'encoder_last_60': [{'close': 1950.0 + i, 'open': 1949.0 + i, 'tickvol': 50000, 'usd_close': 104.0, 'usd_open': 103.9, 'gspc_close': 5200.0, 'gspc_open': 5199.0, 'oil_dcoilwtico': 78.0, 'fed_rate_pct': 0.0525, 'cpi_cpiaucsl': 310.0} for i in range(60)],
        'future_7': [{'usd_close': 104.0, 'usd_open': 103.9, 'gspc_close': 5200.0, 'gspc_open': 5199.0, 'oil_dcoilwtico': 78.0, 'fed_rate_pct': 0.0525, 'cpi_cpiaucsl': 310.0}] * 7
    })

    resp = result.get_json()
    print(f'\n8. Predict API works: {result.status_code == 200}')
    print(f'9. TFT model mode: {resp.get("mode") == "TFT-model"}')
    print(f'10. Has 7 predictions: {len(resp.get("predictions", [])) == 7}')

    all_ok = (
        rv.status_code == 200 and
        'id="chart"' in html and
        'fetch(/predict)' in html and
        result.status_code == 200 and
        resp.get('mode') == 'TFT-model'
    )

    print(f'\n=== STATUS: {"READY" if all_ok else "NOT READY"} ===')

    if not all_ok:
        print('\nMissing components:')
        if not ('id="chart"' in html):
            print('  - Chart div not found')
        if not ('fetch(/predict)' in html):
            print('  - Predict API call not found')
        if resp.get('mode') != 'TFT-model':
            print(f'  - Mode is {resp.get("mode")}, not TFT-model')
