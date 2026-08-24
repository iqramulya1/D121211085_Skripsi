import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import build_input_dataframe  # Panggil dari __init__.py yang sudah benar


def test_build_input_dataframe_shape_and_columns():
    """Test build_input_dataframe dengan format NO_IDR (sesuai config saat ini)"""
    # Format data sesuai NO_IDR version - tanpa IDR
    encoder_last_60 = [
        {
            "close": 1900.0,
            "open": 1899.0,
            "tickvol": 50000,
            "usd_close": 104.0,
            "usd_open": 103.9,
            "gspc_close": 5200.0,
            "gspc_open": 5199.0,
            "oil_dcoilwtico": 78.0,
            "fed_rate_pct": 0.0525,
            "cpi_cpiaucsl": 310.0
        }
        for _ in range(60)
    ]
    future_7 = [
        {
            "usd_close": 104.0,
            "usd_open": 103.9,
            "gspc_close": 5200.0,
            "gspc_open": 5199.0,
            "oil_dcoilwtico": 78.0,
            "fed_rate_pct": 0.0525,
            "cpi_cpiaucsl": 310.0
        }
        for _ in range(7)
    ]

    df = build_input_dataframe(encoder_last_60, future_7)

    # Test shape
    assert df.shape[0] == 67, f"Expected 67 rows, got {df.shape[0]}"

    # Essential columns untuk NO_IDR version (target boleh NaN untuk inference)
    required_cols = ["time_idx", "group_id", "close", "open", "tickvol",
                     "usd_close", "usd_open", "gspc_close", "gspc_open",
                     "oil_dcoilwtico", "fed_rate_pct", "cpi_cpiaucsl"]

    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Test categorical columns
    assert "day_of_week" in df.columns
    assert "month" in df.columns
    assert "year_num" in df.columns

    print("✅ build_input_dataframe test passed")
