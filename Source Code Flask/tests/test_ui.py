import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_ui_index_page_loads():
    """Test bahwa UI index page bisa di-load"""
    from app import create_app
    
    app = create_app()
    client = app.test_client()
    
    # Load homepage
    resp = client.get('/')
    assert resp.status_code == 200
    
    # Cek mengandung elemen UI penting
    html_content = resp.data.decode('utf-8')
    assert 'version' in html_content
    assert '60 Days' in html_content
    assert '7 Days' in html_content
    assert 'Forecast' in html_content


def test_ui_has_plotly():
    """Test bahwa UI menggunakan Plotly"""
    from app import create_app
    
    app = create_app()
    client = app.test_client()
    
    resp = client.get('/')
    html_content = resp.data.decode('utf-8')
    
    # Cek Plotly CDN
    assert 'plotly' in html_content.lower()


def test_ui_has_download_button():
    """Test bahwa UI punya tombol download"""
    from app import create_app
    
    app = create_app()
    client = app.test_client()
    
    resp = client.get('/')
    html_content = resp.data.decode('utf-8')
    
    # Cek download functionality
    assert 'download' in html_content.lower() or 'csv' in html_content.lower()


def test_invalid_csv_has_visible_inline_validation():
    """CSV invalid harus ditolak dengan feedback di dekat area upload."""
    from app import create_app

    app = create_app()
    client = app.test_client()

    resp = client.get('/')
    html_content = resp.data.decode('utf-8')

    assert 'id="uploadFeedback"' in html_content
    assert 'role="alert"' in html_content
    assert 'id="uploadArea" role="button" tabindex="0"' in html_content
    assert 'REQUIRED_CSV_COLUMNS' in html_content
    assert 'missingColumns' in html_content
    assert 'rejectUploadedFile(validationError)' in html_content


def test_sample_scenario_fallback_enables_forecast_data():
    """Fallback skenario harus mengisi 60 data, bukan hanya preview."""
    from app import create_app

    app = create_app()
    client = app.test_client()

    resp = client.get('/')
    html_content = resp.data.decode('utf-8')

    assert html_content.count('class="btn btn-secondary scenario-button"') == 4
    assert 'const generatedData = generateSampleData();' in html_content
    assert 'encoderData = generatedData;' in html_content
    assert "currentMode = 'sample';" in html_content
    assert 'markSelectedScenario(type);' in html_content


def test_run_evaluation_is_not_shown_in_model_analysis():
    """Model Analysis hanya menampilkan info model dan data historis."""
    from app import create_app

    app = create_app()
    client = app.test_client()

    resp = client.get('/')
    html_content = resp.data.decode('utf-8')

    assert 'Run Evaluation' not in html_content
    assert 'runModelEvaluation' not in html_content
    assert 'evaluationResults' not in html_content
    assert 'Model Info' in html_content
    assert 'Historical Data' in html_content
