"""
Konfigurasi TFT Model sesuai training notebook kode_tft_iqra_no_idr.ipynb
"""

# TimeSeriesDataSet Configuration dari Cell 5
MAX_ENCODER_LENGTH = 60
MAX_PREDICTION_LENGTH = 7
MIN_ENCODER_LENGTH = 30  # max_encoder_length // 2
MIN_PREDICTION_LENGTH = 1

# Target column
TARGET = "close"
GROUP_IDS = ["group_id"]

# Static categoricals
STATIC_CATEGORICALS = ["group_id"]

# Time varying known categoricals
TIME_VARYING_KNOWN_CATEGORICALS = ["day_of_week", "month"]

# Time varying known reals
TIME_VARYING_KNOWN_REALS = ["time_idx", "year_num"]

# Unknown reals (semua fitur yang diprediksi)
# Dari Cell 4-5 training notebook - NO IDR VERSION
UNKNOWN_REALS = [
    "open", "close", "tickvol",
    "close_lag1", "close_lag5", "close_ma7", "close_ma20", "close_std7",
    "log_close", "log_close_lag1", "log_close_ma7",
    "close_pct_from_ma20",
    "gold_ret1", "gold_ret5", "gold_vol5", "gold_vol20",
    "usd_open", "usd_close", "usd_adj_close", "usd_close_lag1", "usd_ret1",
    "gspc_open", "gspc_close", "gspc_close_lag1", "gspc_ret1",
    # NO IDR features in no_idr version
    "oil_dcoilwtico", "oil_lag1", "oil_ret1",
    "fed_rate_pct", "fed_target_rate_from_pct", "fed_target_rate_to_pct",
    "cpi_cpiaucsl",
]

# Default values untuk features yang mungkin missing
DEFAULT_VALUES = {
    "open": 1900.0,
    "close": 1900.0,
    "tickvol": 0,
    "close_lag1": 1900.0,
    "close_lag5": 1900.0,
    "close_ma7": 1900.0,
    "close_ma20": 1900.0,
    "close_std7": 50.0,
    "log_close": 7.55,
    "log_close_lag1": 7.55,
    "log_close_ma7": 7.55,
    "close_pct_from_ma20": 0.0,
    "gold_ret1": 0.0,
    "gold_ret5": 0.0,
    "gold_vol5": 0.02,
    "gold_vol20": 0.02,
    "usd_open": 90.0,
    "usd_close": 90.0,
    "usd_adj_close": 90.0,
    "usd_close_lag1": 90.0,
    "usd_ret1": 0.0,
    "gspc_open": 3800.0,
    "gspc_close": 3800.0,
    "gspc_close_lag1": 3800.0,
    "gspc_ret1": 0.0,
    "oil_dcoilwtico": 55.0,
    "oil_lag1": 55.0,
    "oil_ret1": 0.0,
    "fed_rate_pct": 0.09,
    "fed_target_rate_from_pct": 0.09,
    "fed_target_rate_to_pct": 0.25,
    "cpi_cpiaucsl": 260.0,
}

# Mapping untuk input API ke internal columns
API_COLUMN_MAPPING = {
    "close": "close",
    "open": "open",
    "tickvol": "tickvol",
    "usd_close": "usd_close",
    "usd_open": "usd_open",
    "gspc_close": "gspc_close",
    "gspc_open": "gspc_open",
    "oil_dcoilwtico": "oil_dcoilwtico",
    "fed_rate_pct": "fed_rate_pct",
    "cpi_cpiaucsl": "cpi_cpiaucsl",
}
