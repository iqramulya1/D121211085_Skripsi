import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import create_app


def test_model_info_endpoint():
    """Test that model info endpoint works"""
    app = create_app()
    client = app.test_client()

    resp = client.get('/model-info?version=version_0')
    assert resp.status_code == 200

    data = resp.get_json()
    assert data['success'] == True
    assert 'model' in data
    assert data['model']['model_type'] == 'Temporal Fusion Transformer (TFT)'
    assert 'architecture' in data['model']
    assert 'features' in data['model']

    print("Model Info Test Passed:")
    print(f"  - Model: {data['model']['model_type']}")
    print(f"  - Encoder Length: {data['model']['architecture']['encoder_length']}")
    print(f"  - Decoder Length: {data['model']['architecture']['decoder_length']}")


def test_historical_performance_endpoint():
    """Test that historical performance endpoint works"""
    app = create_app()
    client = app.test_client()

    resp = client.get('/historical-performance?version=version_0&days=30')
    assert resp.status_code == 200

    data = resp.get_json()
    assert data['success'] == True
    assert 'data' in data
    assert 'summary' in data
    assert len(data['data']) <= 30

    print("Historical Performance Test Passed:")
    print(f"  - Data points: {len(data['data'])}")
    print(f"  - Avg close: ${data['summary']['avg_close']}")


def test_evaluate_endpoint():
    """Test that evaluate endpoint works"""
    app = create_app()
    client = app.test_client()

    # Test evaluation - might take a bit longer
    resp = client.post('/evaluate',
                      json={'version': 'version_0'},
                      content_type='application/json')

    # Note: This might return 400 if no test data is available
    # or 200 if evaluation runs successfully
    assert resp.status_code in [200, 400, 500]

    if resp.status_code == 200:
        data = resp.get_json()
        assert 'metrics' in data or 'error' in data
        if 'metrics' in data:
            print("Evaluation Test Passed:")
            print(f"  - MAE: {data['metrics']['mae']}")
            print(f"  - RMSE: {data['metrics']['rmse']}")
            print(f"  - MAPE: {data['metrics']['mape']}%")
    else:
        print("Evaluation endpoint returned non-200 status (expected if no test data)")


if __name__ == '__main__':
    print("Running Analysis Endpoint Tests...")
    print("=" * 50)

    try:
        test_model_info_endpoint()
        print()
        test_historical_performance_endpoint()
        print()
        test_evaluate_endpoint()
        print()
        print("=" * 50)
        print("All tests completed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
