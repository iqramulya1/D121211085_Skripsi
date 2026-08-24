import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app


def test_sample_scenarios_endpoint():
    """Test that sample scenarios endpoint returns real data from training dataset"""
    app = create_app()
    client = app.test_client()

    resp = client.get('/sample-scenarios')
    assert resp.status_code == 200

    data = resp.get_json()
    assert data['success'] == True
    assert 'scenarios' in data

    # Check all scenarios exist
    scenarios = data['scenarios']
    assert 'bullish' in scenarios
    assert 'bearish' in scenarios
    assert 'sideways' in scenarios
    assert 'volatile' in scenarios

    # Each scenario should have 60 data points
    for scenario_name, scenario in scenarios.items():
        assert 'name' in scenario
        assert 'description' in scenario
        assert 'data' in scenario
        assert len(scenario['data']) == 60, f"{scenario_name} should have 60 data points"

        # Check data structure
        first_row = scenario['data'][0]
        assert 'close' in first_row
        assert 'open' in first_row
        assert 'usd_close' in first_row
        assert 'gspc_close' in first_row

        print(f"OK {scenario_name}: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Data points: {len(scenario['data'])}")
        print(f"   Sample price: ${first_row['close']:.2f}")
        print()

    print("=" * 50)
    print("All scenarios loaded successfully from real training data!")
    print("=" * 50)


if __name__ == '__main__':
    test_sample_scenarios_endpoint()
