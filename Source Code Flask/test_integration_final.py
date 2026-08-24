#!/usr/bin/env python
"""Test View Integration - Complete Check"""
import sys
sys.path.insert(0, '.')
from app import create_app

app = create_app()

with app.test_client() as client:
    rv = client.get('/')
    html = rv.data.decode('utf-8')

    print('=== FRONTEND-BACKEND INTEGRATION CHECK ===')

    # Check components
    checks = {
        'Index page': rv.status_code == 200,
        'Chart div': 'id="chart"' in html,
        'Results table': 'resultsTable' in html,
        'Predict button': 'btnPredict' in html,
        'Status API': "fetch('/status')" in html,
        'Predict API': "fetch('/predict'" in html,
        'Plotly chart': 'plotly' in html,
        'renderChart()': 'renderChart' in html,
        'renderTable()': 'renderTable' in html,
    }

    for i, (name, passed) in enumerate(checks.items(), 1):
        status = 'OK' if passed else 'MISSING'
        print(f'{i}. {name}: {status}')

    # Test predict endpoint
    result = client.post('/predict', json={
        'version': 'version_0',
        'encoder_last_60': [{'close': 1950.0 + i, 'open': 1949.0 + i, 'tickvol': 50000, 'usd_close': 104.0, 'usd_open': 103.9, 'gspc_close': 5200.0, 'gspc_open': 5199.0, 'oil_dcoilwtico': 78.0, 'fed_rate_pct': 0.0525, 'cpi_cpiaucsl': 310.0} for i in range(60)],
        'future_7': [{'usd_close': 104.0, 'usd_open': 103.9, 'gspc_close': 5200.0, 'gspc_open': 5199.0, 'oil_dcoilwtico': 78.0, 'fed_rate_pct': 0.0525, 'cpi_cpiaucsl': 310.0}] * 7
    })

    resp = result.get_json()

    print(f'\n=== API ENDPOINT CHECK ===')
    api_checks = {
        'Predict endpoint works': result.status_code == 200,
        'Returns TFT-model mode': resp.get('mode') == 'TFT-model',
        'Has 7 predictions': len(resp.get('predictions', [])) == 7,
        'Predictions have values': all('prediction' in p for p in resp.get('predictions', [])),
    }

    for i, (name, passed) in enumerate(api_checks.items(), 1):
        status = 'OK' if passed else 'FAIL'
        print(f'{i}. {name}: {status}')

    # Final status
    all_passed = all(checks.values()) and all(api_checks.values())

    print(f'\n=== FINAL STATUS: {"READY" if all_passed else "NOT READY"} ===')

    if all_passed:
        print('\nView is fully integrated with TFT backend!')
        print('Open http://localhost:5000 to use the web UI')
    else:
        print('\nMissing components:')
        for name, passed in {**checks, **api_checks}.items():
            if not passed:
                print(f'  - {name}')
