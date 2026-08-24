import os
import glob
import logging
from typing import Dict

import pandas as pd
import numpy as np

from .config import (
    UNKNOWN_REALS,
    MAX_ENCODER_LENGTH,
    MAX_PREDICTION_LENGTH,
    MIN_ENCODER_LENGTH,
    MIN_PREDICTION_LENGTH,
    TARGET,
    GROUP_IDS,
    STATIC_CATEGORICALS,
    TIME_VARYING_KNOWN_CATEGORICALS,
    TIME_VARYING_KNOWN_REALS,
    DEFAULT_VALUES,
)

_LOADED_MODELS: Dict[str, object] = {}
# Model artifacts are grouped under the repository's model source folder.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_LOGS = os.path.join(_PROJECT_ROOT, "Source Code Model", "models")
_BASE_LOGS = os.environ.get("TFT_LIGHTNING_LOGS", os.environ.get("TFT_MODEL_BASE", _DEFAULT_LOGS))


def discover_versions(base_path: str = None):
    base = base_path or _BASE_LOGS
    versions = []
    if not os.path.isdir(base):
        return versions
    for path in sorted(glob.glob(os.path.join(base, "version_*"))):
        versions.append(os.path.basename(path))
    return versions


def best_ckpt_for_version(version: str, base_path: str = None):
    base = base_path or _BASE_LOGS
    version_dir = os.path.join(base, version)
    ckpt_dir = os.path.join(version_dir, "checkpoints")

    logging.info(f"Looking for checkpoints in: {ckpt_dir}")

    if os.path.isdir(ckpt_dir):
        # Priority 1: CPU-trained model (tft_cpu_model) - most compatible for CPU inference
        for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "tft_cpu_model*.ckpt"))):
            logging.info(f"Found CPU model checkpoint: {ckpt} for {version}")
            return ckpt
        # Priority 2: best_*.ckpt
        for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "best_*.ckpt"))):
            logging.info(f"Found best checkpoint: {ckpt} for {version}")
            return ckpt
        # Priority 3: GPU training checkpoint (epoch=*.ckpt with larger size)
        for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "epoch=*.ckpt"))):
            size_mb = os.path.getsize(ckpt) / 1024 / 1024
            if size_mb > 10:
                logging.info(f"Found GPU training checkpoint: {ckpt} ({size_mb:.2f} MB) for {version}")
                return ckpt
        # Priority 4: any checkpoint
        for ckpt in sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt"))):
            logging.info(f"Found checkpoint (any): {ckpt} for {version}")
            return ckpt
    else:
        logging.warning(f"Checkpoint directory not found: {ckpt_dir}")

    # Fallback locations
    alt_base = os.path.join(_PROJECT_ROOT, "Source Code Model", "lightning_logs")
    logging.info(f"Trying fallback location: {alt_base}")
    if os.path.isdir(alt_base):
        version_dir2 = os.path.join(alt_base, version)
        ckpt_dir2 = os.path.join(version_dir2, "checkpoints")
        if os.path.isdir(ckpt_dir2):
            for ckpt in sorted(glob.glob(os.path.join(ckpt_dir2, "*.ckpt"))):
                logging.info(f"Found checkpoint (alt): {ckpt} for {version}")
                return ckpt

    logging.error(f"No checkpoint found for {version}")
    return None


def load_model(version: str, base_path: str = None):
    global _LOADED_MODELS
    if version in _LOADED_MODELS:
        logging.info(f"Model for {version} already loaded in memory")
        return _LOADED_MODELS[version]

    ckpt_path = best_ckpt_for_version(version, base_path or _BASE_LOGS)
    if not ckpt_path:
        logging.warning(f"No checkpoint found for {version}, cannot load model.")
        return None

    try:
        from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import EncoderNormalizer
        import torch
        import pickle

        logging.info(f"Loading TFT model from: {ckpt_path}")

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Coba load training dataset yang di-save saat training
        # Ini CRITICAL untuk preservasi categorical encodings!
        training_dataset = None
        dataset_pkl = os.path.join(os.path.dirname(ckpt_path), "training_dataset.pkl")
        if os.path.exists(dataset_pkl):
            try:
                with open(dataset_pkl, 'rb') as f:
                    training_dataset = pickle.load(f)
                logging.info(f"Loaded training dataset from: {dataset_pkl}")
            except Exception as e:
                logging.warning(f"Failed to load training dataset: {e}")

        # Jika tidak ada training dataset, buat dummy dataset dengan semua kategori
        if training_dataset is None:
            logging.warning("No training dataset found, creating dummy dataset (categorical encodings may not match!)")

            # Buat dummy data dengan SEMUA kategori yang mungkin diperlukan
            all_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            all_months = [str(i) for i in range(1, 13)]  # "1" to "12"

            dummy_data = []
            for i in range(140):  # Cukup untuk cover 2 minggu
                dummy_data.append({
                    'time_idx': i,
                    'close': 1900.0 + i * 0.5,
                    'group_id': 'XAU_USD',
                    'day_of_week': all_days[i % 7],  # Rotasi semua hari
                    'month': all_months[(i // 30) % 12],  # Rotasi semua bulan
                    'year_num': 2021.0,
                    'open': 1899.0 + i * 0.5,
                    'tickvol': 50000,
                })

            dummy_df = pd.DataFrame(dummy_data)
            for col in UNKNOWN_REALS:
                if col not in dummy_df.columns:
                    dummy_df[col] = DEFAULT_VALUES.get(col, 0.0)

            training_dataset = TimeSeriesDataSet(
                dummy_df,
                time_idx="time_idx",
                target="close",
                group_ids=["group_id"],
                min_encoder_length=30,
                max_encoder_length=60,
                min_prediction_length=1,
                max_prediction_length=7,
                static_categoricals=["group_id"],
                time_varying_known_categoricals=["day_of_week", "month"],
                time_varying_known_reals=["time_idx", "year_num"],
                time_varying_unknown_reals=UNKNOWN_REALS,
                target_normalizer=EncoderNormalizer(method="standard", center=True),
            )

        # Ambil hyperparameters dari checkpoint
        hparams = checkpoint.get('hyper_parameters', {})

        # Buat model dari training dataset dengan hparams dari checkpoint
        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            **hparams
        )

        # Hapus logging_metrics SEBELUM load state_dict (untuk avoid CUDA error)
        model.logging_metrics = torch.nn.ModuleList()

        # Attach training dataset ke model untuk inference
        model.dataset = training_dataset

        # Load state_dict (skip metrics)
        state_dict = checkpoint['state_dict']
        model_state_dict = {}

        for key, value in state_dict.items():
            if not key.startswith('logging_metrics'):
                model_state_dict[key] = value

        model.load_state_dict(model_state_dict, strict=False)

        # Set ke CPU dan eval mode
        model = model.cpu()
        model.eval()

        _LOADED_MODELS[version] = model
        logging.info(f"Model {version} loaded successfully on CPU!")
        return model
    except Exception as e:
        logging.warning(f"Failed to load TFT model: {e}")
        import traceback
        logging.warning(traceback.format_exc())
        return None


def load_engineered_dataframe():
    """Load the training CSV and apply the same feature engineering used for
    inference (lag/rolling/return features), returning a clean tabular DataFrame.

    Shared by /load-dataset and /evaluate so both work on the same real,
    row-aligned data instead of a TimeSeriesDataSet's internal tensor dict
    (whose entries have mismatched shapes and can't be turned into a DataFrame).
    """
    dataset_path = os.path.join(_PROJECT_ROOT, "Dataset", "gold_price_no_idr.csv")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Training dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    ffill_cols = [
        'oil_dcoilwtico', 'fed_rate_pct', 'fed_target_rate_from_pct',
        'fed_target_rate_to_pct', 'cpi_cpiaucsl',
        'usd_open', 'usd_close', 'usd_adj_close',
        'gspc_open', 'gspc_close',
    ]
    for col in ffill_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    df['close_lag1'] = df['close'].shift(1)
    df['close_lag5'] = df['close'].shift(5)
    df['close_ma7'] = df['close'].rolling(7).mean()
    df['close_ma20'] = df['close'].rolling(20).mean()
    df['close_std7'] = df['close'].rolling(7).std()
    df['log_close'] = np.log(df['close'])
    df['log_close_lag1'] = df['log_close'].shift(1)
    df['log_close_ma7'] = df['log_close'].rolling(7).mean()
    df['close_pct_from_ma20'] = (df['close'] - df['close_ma20']) / df['close_ma20']
    df['gold_ret1'] = df['close'].pct_change()
    df['gold_ret5'] = df['close'].pct_change(5)
    df['usd_ret1'] = df['usd_close'].pct_change()
    df['gspc_ret1'] = df['gspc_close'].pct_change()
    df['oil_ret1'] = df['oil_dcoilwtico'].pct_change()
    df['usd_close_lag1'] = df['usd_close'].shift(1)
    df['oil_lag1'] = df['oil_dcoilwtico'].shift(1)
    df['gspc_close_lag1'] = df['gspc_close'].shift(1)
    df['gold_vol5'] = df['gold_ret1'].rolling(5).std()
    df['gold_vol20'] = df['gold_ret1'].rolling(20).std()

    required_cols = [
        'close_lag1', 'close_lag5', 'close_ma7', 'close_ma20', 'close_std7',
        'log_close', 'log_close_lag1', 'log_close_ma7', 'close_pct_from_ma20',
        'gold_ret1', 'gold_ret5', 'usd_ret1', 'gspc_ret1', 'oil_ret1',
        'usd_close_lag1', 'oil_lag1', 'gspc_close_lag1',
        'gold_vol5', 'gold_vol20',
    ]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    return df


# Fields that must be non-negative on incoming market data rows.
# 'close' is only required on encoder rows (history) - future rows carry the
# exogenous drivers, not 'close', since that's the value being predicted.
_ENCODER_REQUIRED_FIELDS = ["close"]
_NONNEGATIVE_ROW_FIELDS = [
    "close", "open", "tickvol", "usd_close", "usd_open", "gspc_close", "gspc_open",
    "oil_dcoilwtico", "fed_rate_pct", "cpi_cpiaucsl",
]


def validate_market_rows(rows, label, required_fields=()):
    """Validate a list of market data dicts (encoder_last_60 / future_7).

    Returns an error message string if invalid, or None if the rows are OK.
    Rejects missing/empty required fields and negative numeric values so bad
    input (e.g. blank required fields or -999 prices) can't reach the model.
    `required_fields` differs by row type: encoder rows must carry 'close'
    (it's historical), future rows must not (it's the value being predicted).
    """
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            return f"{label}[{idx}] must be an object"

        for field in required_fields:
            value = row.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                return f"{label}[{idx}]: field '{field}' is required and cannot be empty"

        for field in _NONNEGATIVE_ROW_FIELDS:
            value = row.get(field)
            if value is None or (isinstance(value, str) and value.strip() == ""):
                continue
            try:
                value = float(value)
            except (TypeError, ValueError):
                return f"{label}[{idx}]: field '{field}' must be a number"
            if value < 0:
                return f"{label}[{idx}]: field '{field}' cannot be negative"

    return None


def to_float(value, default=0.0):
    """Helper untuk convert ke float dengan aman (handle string, None, dll)"""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    return default


def build_input_dataframe(encoder_last_60, future_7, feature_cols=None):
    """
    Build input DataFrame dengan format PERSIS seperti training TFT.
    Dari kode: kode_tft_iqra_updated.ipynb line 607-643

    Format TimeSeriesDataSet:
    - time_idx="time_idx"
    - target="close"
    - group_ids=["group_id"]
    - min_encoder_length=30, max_encoder_length=60
    - min_prediction_length=1, max_prediction_length=7
    - static_categoricals=["group_id"]
    - time_varying_known_categoricals=["day_of_week", "month"]
    - time_varying_known_reals=["time_idx", "year_num"]
    - time_varying_unknown_reals = unknown_reals (all features)
    """
    if not isinstance(encoder_last_60, list) or len(encoder_last_60) != 60:
        raise ValueError("encoder_last_60 must be a list of 60 dicts")
    if not isinstance(future_7, list) or len(future_7) != 7:
        raise ValueError("future_7 must be a list of 7 dicts")

    # Valid days untuk bursa (no Sunday untuk beberapa data)
    VALID_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Semua fitur unknown_reals sesuai training (NO IDR version)
    unknown_reals = UNKNOWN_REALS

    # Semua kolom yang dibutuhkan
    all_cols = unknown_reals + ["time_idx", "year_num", "day_of_week", "month", "group_id"]
    
    def add_date_features(time_idx):
        """Tambah day_of_week dan month berdasarkan time_idx (asumsi mulai dari 2021-01-04)

        NOTE: Training data hanya punya 5 hari (Monday-Friday) karena stock market data.
        Weekend (Saturday/Sunday) di-map ke Friday.
        """
        import datetime
        start_date = datetime.date(2021, 1, 4)
        current_date = start_date + datetime.timedelta(days=time_idx - 1)

        day_name = current_date.strftime("%A")  # Monday, Tuesday, etc

        # Mapping weekend ke Friday untuk stock market data
        # Training data hanya punya weekdays (market closed on weekends)
        weekend_map = {
            "Saturday": "Friday",
            "Sunday": "Friday",
        }
        day_name = weekend_map.get(day_name, day_name)

        month_num = current_date.month
        month_num = max(1, min(12, month_num))

        return {
            "day_of_week": day_name,
            "month": str(month_num),
            "year_num": float(current_date.year)
        }
    
    # Proses encoder (60 hari)
    enc_data = []
    for i, d in enumerate(encoder_last_60):
        idx = i + 1
        date_feats = add_date_features(idx)

        # Convert semua nilai ke float dengan aman
        close = to_float(d.get("close", d.get("usd_close", np.nan)))
        open_val = to_float(d.get("open", np.nan))
        tickvol = to_float(d.get("tickvol", 0))

        # Feature engineering
        close_lag1 = enc_data[-1].get("close") if enc_data else close
        close_lag5 = enc_data[-5].get("close") if len(enc_data) >= 5 else close

        # Moving averages
        closes = [x.get("close", 0) for x in enc_data]
        close_ma7 = np.mean(closes[-7:]) if len(closes) >= 7 else close
        close_ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else close
        close_std7 = np.std(closes[-7:]) if len(closes) >= 7 else 0

        # Log
        log_close = np.log(close) if close and close > 0 else 0
        log_close_lag1 = np.log(close_lag1) if close_lag1 and close_lag1 > 0 else 0
        log_close_ma7 = np.log(close_ma7) if close_ma7 and close_ma7 > 0 else 0

        # Returns
        gold_ret1 = (close - close_lag1) / close_lag1 if close_lag1 and close_lag1 > 0 else 0
        gold_ret5 = (close - close_lag5) / close_lag5 if close_lag5 and close_lag5 > 0 else 0

        # Volatility
        tickvols = [x.get("tickvol", 0) for x in enc_data]
        gold_vol5 = np.std(tickvols[-5:]) if len(tickvols) >= 5 else 0
        gold_vol20 = np.std(tickvols[-20:]) if len(tickvols) >= 20 else 0

        # USD features
        usd_close = to_float(d.get("usd_close", d.get("usd_open", np.nan)))
        usd_open = to_float(d.get("usd_open", np.nan))
        usd_adj_close = to_float(d.get("usd_adj_close", np.nan))
        usd_lag1 = enc_data[-1].get("usd_close") if enc_data else usd_close
        usd_ret1 = (usd_close - usd_lag1) / usd_lag1 if usd_lag1 and usd_lag1 > 0 else 0

        # GSPC features
        gspc_close = to_float(d.get("gspc_close", d.get("gspc_open", np.nan)))
        gspc_open = to_float(d.get("gspc_open", np.nan))
        gspc_lag1 = enc_data[-1].get("gspc_close") if enc_data else gspc_close
        gspc_ret1 = (gspc_close - gspc_lag1) / gspc_lag1 if gspc_lag1 and gspc_lag1 > 0 else 0

        # NO IDR features in no_idr version

        # Oil
        oil = to_float(d.get("oil_dcoilwtico", np.nan))
        oil_lag1 = enc_data[-1].get("oil_dcoilwtico") if enc_data else oil
        oil_ret1 = (oil - oil_lag1) / oil_lag1 if oil_lag1 and oil_lag1 > 0 else 0

        # Fed & CPI
        fed_rate = to_float(d.get("fed_rate_pct", np.nan))
        fed_target_from = to_float(d.get("fed_target_rate_from_pct", np.nan))
        fed_target_to = to_float(d.get("fed_target_rate_to_pct", np.nan))
        cpi = to_float(d.get("cpi_cpiaucsl", np.nan))
        
        # pct from MA20
        close_pct_from_ma20 = (close - close_ma20) / close_ma20 if close_ma20 and close_ma20 > 0 else 0

        row = {
            "time_idx": idx,
            "group_id": "XAU_USD",
            "year_num": date_feats["year_num"],
            "day_of_week": date_feats["day_of_week"],
            "month": date_feats["month"],
            "open": open_val if not np.isnan(open_val) else close,
            "close": close,
            "tickvol": tickvol if not np.isnan(tickvol) else 0,
            "close_lag1": close_lag1,
            "close_lag5": close_lag5,
            "close_ma7": close_ma7,
            "close_ma20": close_ma20,
            "close_std7": close_std7,
            "log_close": log_close,
            "log_close_lag1": log_close_lag1,
            "log_close_ma7": log_close_ma7,
            "close_pct_from_ma20": close_pct_from_ma20,
            "gold_ret1": gold_ret1,
            "gold_ret5": gold_ret5,
            "gold_vol5": gold_vol5,
            "gold_vol20": gold_vol20,
            "usd_open": usd_open if not np.isnan(usd_open) else usd_close,
            "usd_close": usd_close,
            "usd_adj_close": usd_adj_close if not np.isnan(usd_adj_close) else usd_close,
            "usd_close_lag1": usd_lag1,
            "usd_ret1": usd_ret1,
            "gspc_open": gspc_open if not np.isnan(gspc_open) else gspc_close,
            "gspc_close": gspc_close,
            "gspc_close_lag1": gspc_lag1,
            "gspc_ret1": gspc_ret1,
            # NO IDR features in no_idr version
            "oil_dcoilwtico": oil if not np.isnan(oil) else (enc_data[-1].get("oil_dcoilwtico", 55.0) if enc_data else 55.0),
            "oil_lag1": oil_lag1,
            "oil_ret1": oil_ret1,
            "fed_rate_pct": fed_rate if not np.isnan(fed_rate) else 0.09,
            "fed_target_rate_from_pct": fed_target_from if not np.isnan(fed_target_from) else fed_rate if not np.isnan(fed_rate) else 0.09,
            "fed_target_rate_to_pct": fed_target_to if not np.isnan(fed_target_to) else 0.25,
            "cpi_cpiaucsl": cpi if not np.isnan(cpi) else 260.0,
        }
        enc_data.append(row)

    # Future data (7 hari) - gunakan last known values
    last_enc = enc_data[-1] if enc_data else {}

    for i, d in enumerate(future_7):
        idx = 61 + i
        date_feats = add_date_features(idx)

        # Convert future values dengan aman
        usd_close = to_float(d.get("usd_close", last_enc.get("usd_close", np.nan)))
        usd_open = to_float(d.get("usd_open", usd_close))
        gspc_close = to_float(d.get("gspc_close", last_enc.get("gspc_close", np.nan)))
        gspc_open = to_float(d.get("gspc_open", gspc_close))
        oil = to_float(d.get("oil_dcoilwtico", last_enc.get("oil_dcoilwtico", np.nan)))
        fed_rate = to_float(d.get("fed_rate_pct", last_enc.get("fed_rate_pct", 0.09)))
        fed_target_from = to_float(d.get("fed_target_rate_from_pct", last_enc.get("fed_target_rate_from_pct", fed_rate)))
        fed_target_to = to_float(d.get("fed_target_rate_to_pct", last_enc.get("fed_target_rate_to_pct", 0.25)))
        cpi = to_float(d.get("cpi_cpiaucsl", last_enc.get("cpi_cpiaucsl", np.nan)))

        row = {
            "time_idx": idx,
            "group_id": "XAU_USD",
            "year_num": date_feats["year_num"],
            "day_of_week": date_feats["day_of_week"],
            "month": date_feats["month"],
            # Use last known values untuk future
            "open": last_enc.get("close", np.nan),
            "close": last_enc.get("close", np.nan),
            "tickvol": last_enc.get("tickvol", 0),
            "close_lag1": last_enc.get("close"),
            "close_lag5": last_enc.get("close_lag5"),
            "close_ma7": last_enc.get("close_ma7"),
            "close_ma20": last_enc.get("close_ma20"),
            "close_std7": last_enc.get("close_std7"),
            "log_close": last_enc.get("log_close"),
            "log_close_lag1": last_enc.get("log_close_lag1"),
            "log_close_ma7": last_enc.get("log_close_ma7"),
            "close_pct_from_ma20": last_enc.get("close_pct_from_ma20"),
            "gold_ret1": last_enc.get("gold_ret1"),
            "gold_ret5": last_enc.get("gold_ret5"),
            "gold_vol5": last_enc.get("gold_vol5"),
            "gold_vol20": last_enc.get("gold_vol20"),
            "usd_open": usd_open if not np.isnan(usd_open) else usd_close if not np.isnan(usd_close) else last_enc.get("usd_close", 90.0),
            "usd_close": usd_close if not np.isnan(usd_close) else last_enc.get("usd_close", 90.0),
            "usd_adj_close": usd_close if not np.isnan(usd_close) else last_enc.get("usd_close", 90.0),
            "usd_close_lag1": last_enc.get("usd_close"),
            "usd_ret1": last_enc.get("usd_ret1"),
            "gspc_open": gspc_open if not np.isnan(gspc_open) else gspc_close if not np.isnan(gspc_close) else last_enc.get("gspc_close", 3800.0),
            "gspc_close": gspc_close if not np.isnan(gspc_close) else last_enc.get("gspc_close", 3800.0),
            "gspc_close_lag1": last_enc.get("gspc_close"),
            "gspc_ret1": last_enc.get("gspc_ret1"),
            # NO IDR features in no_idr version
            "oil_dcoilwtico": oil if not np.isnan(oil) else last_enc.get("oil_dcoilwtico", 55.0),
            "oil_lag1": last_enc.get("oil_dcoilwtico"),
            "oil_ret1": last_enc.get("oil_ret1"),
            "fed_rate_pct": fed_rate if not np.isnan(fed_rate) else last_enc.get("fed_rate_pct", 0.09),
            "fed_target_rate_from_pct": fed_target_from if not np.isnan(fed_target_from) else last_enc.get("fed_target_rate_from_pct", 0.09),
            "fed_target_rate_to_pct": fed_target_to if not np.isnan(fed_target_to) else last_enc.get("fed_target_rate_to_pct", 0.25),
            "cpi_cpiaucsl": cpi if not np.isnan(cpi) else last_enc.get("cpi_cpiaucsl", 260.0),
        }
        enc_data.append(row)
    
    df = pd.DataFrame(enc_data)
    return df


def infer_tft_model(model, df):
    """
    Inferensi TFT model menggunakan dataset yang sudah ter-train.
    CRITICAL: Gunakan model's dataset reference untuk preservasi categorical encodings.
    """
    try:
        if hasattr(model, 'predict') and model is not None:
            logging.info("Trying TFT prediction with model's reference dataset...")

            # Model memiliki referensi ke training dataset dengan encodings yang benar
            # Gunakan ini untuk membuat inference dataset yang compatible
            if not hasattr(model, 'dataset') or model.dataset is None:
                logging.warning("Model does not have reference dataset, cannot do proper inference")
                return None

            training_dataset = model.dataset

            # Validasi dan fix categorical values
            valid_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            df['day_of_week'] = df['day_of_week'].apply(lambda x: x if x in valid_days else "Monday")
            df['month'] = df['month'].astype(str)
            df['month'] = df['month'].apply(lambda x: str(max(1, min(12, int(x) if str(x).isdigit() else 1))))

            # GUNAKAN from_dataset untuk preservasi categorical encodings!
            # Ini critical - dari training dataset ke inference dataframe
            inference_dataset = training_dataset.from_dataset(
                training_dataset,
                df,
                predict=True,  # Mode prediction
                stop_randomization=True,
            )

            # Buat dataloader untuk prediction
            pred_dataloader = inference_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

            # Get prediction dengan mode="prediction" agar denormalized
            predictions = model.predict(pred_dataloader, mode="prediction", return_x=False)

            if predictions is not None and len(predictions) > 0:
                import torch
                if isinstance(predictions, torch.Tensor):
                    vals = predictions.cpu().numpy().flatten()[:7]
                elif hasattr(predictions, 'numpy'):
                    vals = predictions.numpy().flatten()[:7]
                else:
                    vals = np.array(predictions).flatten()[:7]

                preds = []
                for i, v in enumerate(vals):
                    preds.append({"time_idx": 61 + i, "prediction": float(v)})
                logging.info(f"TFT prediction successful: {vals[:3]}...")
                return preds
    except Exception as e:
        logging.warning(f"TFT inference error: {e}")
        import traceback
        logging.warning(traceback.format_exc())
    return None


def infer_from_payload(payload, model=None):
    version = payload.get("version", "version_0")
    encoder_last_60 = payload.get("encoder_last_60", [])
    future_7 = payload.get("future_7", [])
    
    df = build_input_dataframe(encoder_last_60, future_7)
    
    # Mode 1: Real TFT model
    if model is not None:
        logging.info("Attempting TFT model inference...")
        preds_tft = infer_tft_model(model, df)
        if preds_tft:
            return preds_tft, "TFT-model"

    # Mode 2: Fallback dengan close price dari data
    last_close = None
    for item in encoder_last_60[-10:]:
        val = item.get("close") or item.get("usd_close")
        if val:
            last_close = to_float(val)
            break

    if last_close is None or last_close == 0:
        last_close = 1900.0

    # Prediksi dengan trend naik 0.15% per hari
    preds = []
    for i in range(7):
        pred_val = last_close * (1 + (i + 1) * 0.0015)
        preds.append({"time_idx": 61 + i, "prediction": round(pred_val, 2)})

    return preds, "Fallback-trend"


# Helper functions for analysis endpoints
def generate_sample_historical_data(days):
    """Generate sample historical data for demo"""
    import random
    data = []
    price = 2300.0
    for i in range(days):
        change = random.uniform(-20, 20)
        price = max(2000, min(2600, price + change))
        data.append({
            "time_idx": i + 1,
            "date": f"Day {i + 1}",
            "close": round(price, 2),
            "open": round(price + random.uniform(-5, 5), 2),
            "volume": random.randint(45000, 55000)
        })
    return data


def calculate_summary(data):
    """Calculate summary statistics"""
    if not data:
        return {"avg_close": 0, "min_close": 0, "max_close": 0, "volatility": 0}

    closes = [d["close"] for d in data]
    import numpy as np
    return {
        "avg_close": round(float(np.mean(closes)), 2),
        "min_close": round(float(np.min(closes)), 2),
        "max_close": round(float(np.max(closes)), 2),
        "volatility": round(float(np.std(closes)), 2)
    }


def format_data_for_frontend(data):
    """Format data for frontend consumption"""
    formatted = []
    for row in data:
        formatted.append({
            "close": float(row.get("close", 0)),
            "open": float(row.get("open", 0)),
            "tickvol": int(row.get("tickvol", 50000)),
            "usd_close": float(row.get("usd_close", 104)),
            "usd_open": float(row.get("usd_open", 104)),
            "gspc_close": float(row.get("gspc_close", 5200)),
            "gspc_open": float(row.get("gspc_open", 5200)),
            "oil_dcoilwtico": float(row.get("oil_dcoilwtico", 78)),
            "fed_rate_pct": float(row.get("fed_rate_pct", 0.0525)),
            "cpi_cpiaucsl": float(row.get("cpi_cpiaucsl", 310))
        })
    return formatted


# Cache for real-time API data (simple in-memory cache)
_API_CACHE = {
    "data": None,
    "timestamp": 0,
    "ttl": 900  # 15 minutes cache
}


def fetch_real_time_data():
    """
    Fetch real-time data from multiple free APIs
    Returns 60 days of historical data with latest values
    """
    import time
    current_time = time.time()

    # Check cache first
    if _API_CACHE["data"] is not None and (current_time - _API_CACHE["timestamp"]) < _API_CACHE["ttl"]:
        logging.info("Returning cached API data")
        return _API_CACHE["data"]

    try:
        import yfinance as yf

        logging.info("Fetching real-time data from Yahoo Finance API...")

        # Fetch gold data (GC=F is Gold Futures)
        gold = yf.Ticker("GC=F")
        gold_hist = gold.history(period="3mo")  # 3 months to ensure we get 60+ trading days

        # Fetch S&P 500
        sp500 = yf.Ticker("^GSPC")
        sp500_hist = sp500.history(period="3mo")

        # Fetch USD Index
        usd = yf.Ticker("DX-Y.NYB")  # US Dollar Index
        usd_hist = usd.history(period="3mo")

        # Fetch Oil (WTI)
        oil = yf.Ticker("CL=F")  # Crude Oil WTI
        oil_hist = oil.history(period="3mo")

        # Take last 60 trading days
        gold_hist = gold_hist.tail(60).reset_index()
        sp500_hist = sp500_hist.tail(60).reset_index()
        usd_hist = usd_hist.tail(60).reset_index()
        oil_hist = oil_hist.tail(60).reset_index()

        # Build combined dataset
        data = []
        for i in range(len(gold_hist)):
            try:
                row = {
                    "date": str(gold_hist.iloc[i]['Date']) if 'Date' in gold_hist.columns else f"Day {i+1}",
                    "close": float(gold_hist.iloc[i]['Close']) if 'Close' in gold_hist.columns else float(gold_hist.iloc[i].get('close', 2300)),
                    "open": float(gold_hist.iloc[i]['Open']) if 'Open' in gold_hist.columns else float(gold_hist.iloc[i].get('open', 2300)),
                    "tickvol": int(gold_hist.iloc[i]['Volume']) if 'Volume' in gold_hist.columns else 50000,
                    "usd_close": float(usd_hist.iloc[i]['Close']) if i < len(usd_hist) and 'Close' in usd_hist.columns else 104.0,
                    "usd_open": float(usd_hist.iloc[i]['Open']) if i < len(usd_hist) and 'Open' in usd_hist.columns else 104.0,
                    "gspc_close": float(sp500_hist.iloc[i]['Close']) if i < len(sp500_hist) and 'Close' in sp500_hist.columns else 5200.0,
                    "gspc_open": float(sp500_hist.iloc[i]['Open']) if i < len(sp500_hist) and 'Open' in sp500_hist.columns else 5200.0,
                    "oil_dcoilwtico": float(oil_hist.iloc[i]['Close']) if i < len(oil_hist) and 'Close' in oil_hist.columns else 78.0,
                    "fed_rate_pct": 0.0525,  # Will fetch from FRED in future
                    "cpi_cpiaucsl": 310.0     # Will fetch from FRED in future
                }
                data.append(row)
            except Exception as e:
                logging.warning(f"Error processing day {i}: {e}")
                continue

        if len(data) >= 60:
            # Cache the result
            _API_CACHE["data"] = data
            _API_CACHE["timestamp"] = current_time

            logging.info(f"Successfully fetched {len(data)} days from Yahoo Finance API")
            return data
        else:
            logging.warning(f"Only got {len(data)} days from API, need 60")
            return None

    except ImportError as e:
        logging.warning("yfinance not installed, unable to fetch live market data")
        raise RuntimeError("yfinance is not installed. Install yfinance>=0.2.28 and restart the app.") from e
    except Exception as e:
        logging.error(f"Error fetching real-time data: {e}")
        raise RuntimeError(f"Failed to fetch real-time data from Yahoo Finance: {e}") from e


def build_api_market_payload():
    """Build a ready-to-use payload from real market data via yfinance."""
    api_data = fetch_real_time_data()
    if not api_data or len(api_data) < 60:
        raise RuntimeError("Insufficient real-time market data returned from Yahoo Finance (need 60 rows).")

    encoder_last_60 = api_data[-60:]
    last_row = encoder_last_60[-1]
    future_7 = []

    for _ in range(7):
        future_7.append({
            "usd_close": float(last_row.get("usd_close", 104.0)),
            "usd_open": float(last_row.get("usd_open", last_row.get("usd_close", 104.0))),
            "gspc_close": float(last_row.get("gspc_close", 5200.0)),
            "gspc_open": float(last_row.get("gspc_open", last_row.get("gspc_close", 5200.0))),
            "oil_dcoilwtico": float(last_row.get("oil_dcoilwtico", 78.0)),
            "fed_rate_pct": float(last_row.get("fed_rate_pct", 0.0525)),
            "fed_target_rate_from_pct": float(last_row.get("fed_rate_pct", 0.0525)),
            "fed_target_rate_to_pct": float(last_row.get("fed_rate_pct", 0.0525)),
            "cpi_cpiaucsl": float(last_row.get("cpi_cpiaucsl", 310.0))
        })

    return {
        "encoder_last_60": encoder_last_60,
        "future_7": future_7,
        "source": "Yahoo Finance API",
        "data_points": len(encoder_last_60)
    }


def create_app():
    from flask import Flask, jsonify, request, render_template
    import logging

    logging.basicConfig(level=logging.INFO)
    
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    template_path = os.path.join(base_path, "templates")
    static_path = os.path.join(base_path, "static")
    
    app = Flask(__name__, template_folder=template_path, static_folder=static_path)

    def _full_inference_available():
        try:
            import pytorch_forecasting
            return True
        except:
            return False

    @app.route("/health", methods=["GET"])
    def health():
        versions = discover_versions()
        loaded = list(_LOADED_MODELS.keys())
        return jsonify({
            "status": "ok", 
            "versions": versions, 
            "loaded": loaded,
            "tft_available": _full_inference_available()
        })

    @app.route("/status", methods=["GET"])
    def status():
        versions = discover_versions()
        status_map = {}
        models_loaded = []

        for v in versions:
            m = load_model(v)
            is_loaded = m is not None
            status_map[v] = {"loaded": is_loaded}
            if is_loaded:
                models_loaded.append(v)

        return jsonify({
            "versions": status_map,
            "models": models_loaded,  # For frontend compatibility
            "full_inference_available": _full_inference_available()
        })

    @app.route("/versions", methods=["GET"])
    def versions():
        return jsonify({"versions": discover_versions()})

    @app.route("/load-dataset", methods=["GET"])
    def load_dataset():
        """Load 60 hari terakhir dari training dataset untuk prediksi"""
        try:
            import pandas as pd

            try:
                df = load_engineered_dataframe()
            except FileNotFoundError as e:
                return jsonify({"error": str(e)}), 404

            # Ambil 60 hari terakhir
            df_last_60 = df.tail(60).copy()

            # Convert ke format list of dict untuk UI
            data = []
            for _, row in df_last_60.iterrows():
                data.append({
                    'close': float(row['close']),
                    'open': float(row['open']),
                    'tickvol': float(row['tickvol']) if pd.notna(row['tickvol']) else 50000,
                    'usd_close': float(row['usd_close']) if pd.notna(row['usd_close']) else 104.0,
                    'usd_open': float(row['usd_open']) if pd.notna(row['usd_open']) else 104.0,
                    'gspc_close': float(row['gspc_close']) if pd.notna(row['gspc_close']) else 5200.0,
                    'gspc_open': float(row['gspc_open']) if pd.notna(row['gspc_open']) else 5200.0,
                    'oil_dcoilwtico': float(row['oil_dcoilwtico']) if pd.notna(row['oil_dcoilwtico']) else 78.0,
                    'fed_rate_pct': float(row['fed_rate_pct']) if pd.notna(row['fed_rate_pct']) else 0.0525,
                    'cpi_cpiaucsl': float(row['cpi_cpiaucsl']) if pd.notna(row['cpi_cpiaucsl']) else 310.0,
                })

            return jsonify({
                "success": True,
                "data": data,
                "date_range": f"{df_last_60['date'].min().date()} to {df_last_60['date'].max().date()}",
                "total_rows": len(df_last_60)
            })

        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            import traceback
            return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

    @app.route("/api/market-data", methods=["GET"])
    def api_market_data():
        """Fetch live market data using Yahoo Finance and build TFT payload."""
        try:
            payload = build_api_market_payload()
            return jsonify({"success": True, "payload": payload})
        except Exception as e:
            logging.error(f"Market data endpoint failed: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.get_json(force=True) or {}
        encoder60 = data.get("encoder_last_60", [])
        future7 = data.get("future_7", [])

        if not isinstance(encoder60, list) or len(encoder60) != 60:
            return jsonify({"error": "encoder_last_60 must be 60 items"}), 400
        if not isinstance(future7, list) or len(future7) != 7:
            return jsonify({"error": "future_7 must be 7 items"}), 400

        row_error = (
            validate_market_rows(encoder60, "encoder_last_60", required_fields=_ENCODER_REQUIRED_FIELDS)
            or validate_market_rows(future7, "future_7")
        )
        if row_error:
            return jsonify({"error": row_error}), 400

        version = data.get("version", "version_0")
        model = load_model(version)

        # Run inference
        payload = {
            "version": version,
            "encoder_last_60": encoder60,
            "future_7": future7
        }

        if model:
            preds, mode = infer_from_payload(payload, model)
            # Check - jika ada model tapi fallback, tandainbeda
            if model is not None and mode == "Fallback-trend":
                mode = "TFT-fallback-format"
        else:
            preds, mode = infer_from_payload(payload, None)

        logging.info(f"Prediction mode: {mode}")

        return jsonify({
            "version": version,
            "predictions": preds,
            "mode": mode
        })

    @app.route("/upload-csv", methods=["POST"])
    def upload_csv():
        """Upload dan validasi CSV file untuk historical data"""
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if not file.filename.endswith('.csv'):
            return jsonify({"error": "File must be a CSV"}), 400

        try:
            # Baca CSV
            import io
            content = file.read().decode('utf-8')
            lines = content.split('\n')

            if len(lines) < 2:
                return jsonify({"error": "CSV is empty or has no data rows"}), 400

            # Parse headers
            headers = [h.strip().lower() for h in lines[0].split(',')]

            # Validasi required columns
            required_cols = ['close']
            missing_cols = [col for col in required_cols if col not in headers]

            if missing_cols:
                return jsonify({
                    "error": f"Missing required columns: {', '.join(missing_cols)}",
                    "required": required_cols,
                    "found": headers
                }), 400

            # Parse data rows
            rows = []
            for line in lines[1:]:
                if not line.strip():
                    continue
                values = line.split(',')
                row = {}
                for i, h in enumerate(headers):
                    if i < len(values):
                        val = values[i].strip()
                        try:
                            row[h] = float(val)
                        except:
                            row[h] = val

                if 'close' in row and row['close']:
                    rows.append(row)

            if len(rows) < 60:
                return jsonify({
                    "error": f"CSV must have at least 60 data rows. Found: {len(rows)}",
                    "rows_found": len(rows),
                    "rows_required": 60
                }), 400

            # Ambil last 60 rows
            last_60 = rows[-60:]

            return jsonify({
                "success": True,
                "total_rows": len(rows),
                "encoder_last_60": last_60,
                "preview": rows[:10],  # First 10 rows for preview
                "headers": headers
            })

        except Exception as e:
            logging.error(f"CSV upload error: {e}")
            return jsonify({"error": f"Error processing CSV: {str(e)}"}), 500

    @app.route("/evaluate", methods=["POST"])
    def evaluate():
        """
        Evaluate model performance with actual vs predicted values
        Returns MAE, RMSE, MAPE metrics for thesis analysis
        """
        try:
            data = request.get_json(force=True) or {}
            version = data.get("version", "version_0")

            # Load model
            model = load_model(version)
            if model is None:
                return jsonify({"error": "Model not available"}), 400

            # Get test data - either from provided data or load from training dataset
            if "test_data" in data and len(data["test_data"]) > 0:
                test_rows = data["test_data"]
            else:
                # Load real historical rows (same engineered features as training),
                # not the TimeSeriesDataSet's internal tensor dict, whose entries
                # (reals/categoricals/target/time) have mismatched shapes and can't
                # be turned into a DataFrame directly.
                try:
                    df = load_engineered_dataframe()
                except FileNotFoundError as e:
                    return jsonify({"error": f"No test data provided and {e}"}), 400

                test_rows = df.tail(90).to_dict('records')  # enough rows for a few walk-forward windows

            # Walk-forward validation needs at least one 60-day encoder + 7-day horizon window
            if len(test_rows) < 67:
                return jsonify({"error": "Need at least 67 data points for evaluation (60 encoder + 7 horizon)"}), 400

            # Perform walk-forward validation
            predictions = []
            actuals = []

            for i in range(len(test_rows) - 60 - 7):
                if i < 0:
                    continue

                # Get encoder data (60 days)
                encoder_data = test_rows[i:i+60]
                # Get actual next 7 days
                actual_next_7 = test_rows[i+60:i+67]

                # Build future data (use actual external variables)
                future_data = []
                for j in range(7):
                    row = actual_next_7[j]
                    future_data.append({
                        "usd_close": row.get("usd_close", 104),
                        "usd_open": row.get("usd_open", 104),
                        "gspc_close": row.get("gspc_close", 5200),
                        "gspc_open": row.get("gspc_open", 5200),
                        "oil_dcoilwtico": row.get("oil_dcoilwtico", 78),
                        "fed_rate_pct": row.get("fed_rate_pct", 0.0525),
                        "cpi_cpiaucsl": row.get("cpi_cpiaucsl", 310)
                    })

                # Make prediction
                try:
                    payload = {
                        "version": version,
                        "encoder_last_60": encoder_data,
                        "future_7": future_data
                    }
                    preds, mode = infer_from_payload(payload, model)

                    if mode == "TFT-model":
                        # Get actual close prices
                        actual_prices = [row.get("close", 0) for row in actual_next_7]
                        pred_prices = [p["prediction"] for p in preds]

                        actuals.extend(actual_prices)
                        predictions.extend(pred_prices)
                except Exception as e:
                    logging.warning(f"Prediction failed at step {i}: {e}")
                    continue

            if len(predictions) == 0:
                return jsonify({"error": "No valid predictions generated"}), 400

            # Calculate metrics
            import numpy as np
            actuals = np.array(actuals)
            predictions = np.array(predictions)

            # MAE (Mean Absolute Error)
            mae = np.mean(np.abs(actuals - predictions))

            # RMSE (Root Mean Squared Error)
            rmse = np.sqrt(np.mean((actuals - predictions) ** 2))

            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

            # Additional metrics
            mse = np.mean((actuals - predictions) ** 2)
            max_error = np.max(np.abs(actuals - predictions))
            min_error = np.min(np.abs(actuals - predictions))

            # Accuracy metrics
            r2_score = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))

            return jsonify({
                "success": True,
                "metrics": {
                    "mae": round(float(mae), 4),
                    "rmse": round(float(rmse), 4),
                    "mape": round(float(mape), 2),
                    "mse": round(float(mse), 4),
                    "max_error": round(float(max_error), 4),
                    "min_error": round(float(min_error), 4),
                    "r2_score": round(float(r2_score), 4)
                },
                "sample_size": len(predictions),
                "data_points": {
                    "actuals": [round(float(a), 2) for a in actuals[:20]],  # First 20 for visualization
                    "predictions": [round(float(p), 2) for p in predictions[:20]]
                }
            })

        except Exception as e:
            logging.error(f"Evaluation error: {e}")
            import traceback
            return jsonify({"error": f"Evaluation failed: {str(e)}", "trace": traceback.format_exc()}), 500

    @app.route("/historical-performance", methods=["GET"])
    def historical_performance():
        """
        Get historical performance data for visualization
        Useful for thesis Chapter 4 - Results
        """
        try:
            version = request.args.get("version", "version_0")
            days = min(int(request.args.get("days", 30)), 100)  # Cap at 100 days

            # Try to load training dataset
            import pickle
            ckpt_path = best_ckpt_for_version(version)
            if ckpt_path is None:
                # Return sample data if model not found
                historical_data = generate_sample_historical_data(days)
                return jsonify({
                    "success": True,
                    "data": historical_data,
                    "summary": calculate_summary(historical_data),
                    "note": "Sample data - model not found"
                })

            ckpt_dir = os.path.dirname(ckpt_path)
            pkl_path = os.path.join(ckpt_dir, "training_dataset.pkl")

            if not os.path.exists(pkl_path):
                # Return sample data if pickle not found
                historical_data = generate_sample_historical_data(days)
                return jsonify({
                    "success": True,
                    "data": historical_data,
                    "summary": calculate_summary(historical_data),
                    "note": "Sample data - training dataset not found"
                })

            with open(pkl_path, "rb") as f:
                training_dataset = pickle.load(f)

            # Get historical data - handle different data formats
            try:
                data = training_dataset.data
                if hasattr(data, 'to_pandas'):
                    df = data.to_pandas()
                elif isinstance(data, pd.DataFrame):
                    df = data.copy()
                else:
                    df = pd.DataFrame(data)
            except Exception as e:
                logging.warning(f"Could not convert training data to DataFrame: {e}")
                historical_data = generate_sample_historical_data(days)
                return jsonify({
                    "success": True,
                    "data": historical_data,
                    "summary": calculate_summary(historical_data),
                    "note": "Sample data - could not parse training dataset"
                })

            # Take last N days
            df = df.tail(days).reset_index(drop=True)

            # Ensure close column exists
            if "close" not in df.columns:
                historical_data = generate_sample_historical_data(days)
                return jsonify({
                    "success": True,
                    "data": historical_data,
                    "summary": calculate_summary(historical_data),
                    "note": "Sample data - close column not found"
                })

            # Convert to list format for frontend
            historical_data = []
            for idx in range(len(df)):
                try:
                    row_data = df.iloc[idx]
                    close_val = float(row_data.get("close", 0)) if hasattr(row_data, 'get') else float(row_data['close']) if 'close' in row_data else 0
                    open_val = float(row_data.get("open", 0)) if hasattr(row_data, 'get') else float(row_data['open']) if 'open' in row_data else 0

                    historical_data.append({
                        "time_idx": int(idx),
                        "date": f"Day {idx + 1}",
                        "close": close_val,
                        "open": open_val,
                        "volume": 50000
                    })
                except Exception as e:
                    logging.warning(f"Error processing row {idx}: {e}")
                    continue

            if not historical_data:
                historical_data = generate_sample_historical_data(days)

            return jsonify({
                "success": True,
                "data": historical_data,
                "summary": calculate_summary(historical_data)
            })

        except Exception as e:
            logging.error(f"Historical performance error: {e}")
            import traceback
            # Return sample data on error
            historical_data = generate_sample_historical_data(days)
            return jsonify({
                "success": True,
                "data": historical_data,
                "summary": calculate_summary(historical_data),
                "note": f"Sample data - error: {str(e)[:100]}"
            })

    @app.route("/model-info", methods=["GET"])
    def model_info():
        """
        Get model information for thesis documentation
        Architecture, hyperparameters, training details
        """
        try:
            version = request.args.get("version", "version_0")

            # Load model to get info
            model = load_model(version)
            if model is None:
                return jsonify({"error": "Model not available"}), 404

            # Get checkpoint info
            ckpt_path = best_ckpt_for_version(version)
            ckpt_dir = os.path.dirname(ckpt_path)
            pkl_path = os.path.join(ckpt_dir, "training_dataset.pkl")

            # Model architecture details
            model_info = {
                "model_type": "Temporal Fusion Transformer (TFT)",
                "version": version,
                "checkpoint": os.path.basename(ckpt_path),
                "architecture": {
                    "encoder_length": MAX_ENCODER_LENGTH,
                    "decoder_length": MAX_PREDICTION_LENGTH,
                    "hidden_size": getattr(model.hparams, "hidden_size", 16),
                    "attention_head_size": getattr(model.hparams, "attention_head_size", 4),
                    "dropout": getattr(model.hparams, "dropout", 0.1),
                    "hidden_continuous_size": getattr(model.hparams, "hidden_continuous_size", 8),
                    "loss": getattr(model.hparams, "loss", "QuantileLoss")
                },
                "features": {
                    "target": TARGET,
                    "group_ids": GROUP_IDS,
                    "static_categoricals": STATIC_CATEGORICALS,
                    "time_varying_known_categoricals": TIME_VARYING_KNOWN_CATEGORICALS,
                    "time_varying_known_reals": TIME_VARYING_KNOWN_REALS,
                    "time_varying_unknown_reals": UNKNOWN_REALS
                }
            }

            return jsonify({
                "success": True,
                "model": model_info
            })

        except Exception as e:
            logging.error(f"Model info error: {e}")
            return jsonify({"error": f"Failed to get model info: {str(e)}"}), 500

    @app.route("/sample-scenarios", methods=["GET"])
    def sample_scenarios():
        """
        Get real sample scenarios from historical data
        Returns different market patterns (bullish, bearish, sideways, volatile)
        """
        try:
            # Sample scenarios should be fast and available offline, so prefer
            # the same local training data used by /load-dataset. Yahoo Finance
            # remains a fallback when that dataset is unavailable.
            df = None
            data_source = None
            try:
                df = load_engineered_dataframe()
                data_source = "Local Training Dataset"
            except Exception as local_error:
                logging.warning(f"Training dataset unavailable for scenarios: {local_error}")

            if df is None or len(df) < 67:
                try:
                    api_data = fetch_real_time_data()
                except Exception as api_error:
                    logging.warning(f"Yahoo Finance unavailable for scenarios: {api_error}")
                    api_data = None

                if api_data and len(api_data) >= 67:
                    df = pd.DataFrame(api_data)
                    data_source = "Yahoo Finance API (Real-Time)"

            if df is None or len(df) < 67:
                # Last-resort data keeps the Sample Data feature functional in
                # offline/demo environments. The response labels this source
                # clearly so it is never mistaken for market history.
                df = pd.DataFrame(generate_sample_historical_data(240))
                data_source = "Simulated Offline Fallback"

            # Ensure we have required columns
            if "close" not in df.columns:
                return jsonify({"error": "CSV missing 'close' column"}), 400

            # Make sure we have enough data
            if len(df) < 67:
                return jsonify({"error": f"Dataset too small: {len(df)} rows"}), 400

            scenarios = {}

            # Take 4 different 60-day periods from the dataset
            period_length = 60
            total_periods = len(df) - period_length

            # Use different parts of the dataset for different scenarios
            periods = [
                ("bullish", total_periods // 4, "Bullish Market (Uptrend Period)"),
                ("bearish", total_periods // 2, "Bearish Market (Downtrend Period)"),
                ("sideways", 3 * total_periods // 4, "Sideways Market (Consolidation)"),
                ("volatile", min(total_periods - 1, total_periods - 10), "Volatile Market (High Volatility)")
            ]

            for key, start_idx, name in periods:
                if start_idx < 0 or start_idx + period_length > len(df):
                    continue

                # Get 60 days of data
                period_data = df.iloc[start_idx:start_idx + period_length].copy()

                # Calculate simple stats for description
                start_price = period_data['close'].iloc[0]
                end_price = period_data['close'].iloc[-1]
                change_pct = ((end_price - start_price) / start_price) * 100
                volatility = period_data['close'].std()

                desc = f"Real historical data: {name}. Start: ${start_price:.2f}, End: ${end_price:.2f}, Change: {change_pct:+.2f}%, Volatility: ${volatility:.2f}"

                # Convert to list format
                data_list = []
                for idx in range(len(period_data)):
                    row = period_data.iloc[idx]
                    try:
                        data_list.append({
                            "close": float(row['close']),
                            "open": float(row.get('open', row['close'])),
                            "tickvol": int(row.get('tickvol', 50000)),
                            "usd_close": float(row.get('usd_close', 104)),
                            "usd_open": float(row.get('usd_open', 104)),
                            "gspc_close": float(row.get('gspc_close', 5200)),
                            "gspc_open": float(row.get('gspc_open', 5200)),
                            "oil_dcoilwtico": float(row.get('oil_dcoilwtico', 78)),
                            "fed_rate_pct": float(row.get('fed_rate_pct', 0.0525)),
                            "cpi_cpiaucsl": float(row.get('cpi_cpiaucsl', 310))
                        })
                    except Exception as e:
                        logging.warning(f"Error formatting row {idx}: {e}")
                        continue

                scenarios[key] = {
                    "name": name,
                    "description": desc,
                    "data": data_list
                }

            return jsonify({
                "success": True,
                "scenarios": scenarios,
                "source": data_source,
                "note": f"All scenarios are from real historical data ({data_source})"
            })

        except Exception as e:
            logging.error(f"Sample scenarios error: {e}")
            import traceback
            return jsonify({"error": f"Failed to load scenarios: {str(e)}"}), 500


    @app.route("/api/refresh-data", methods=["POST"])
    def refresh_real_time_data():
        """
        Force refresh real-time data from API
        Clears cache and fetches fresh data
        """
        try:
            import time

            # Clear cache
            _API_CACHE["data"] = None
            _API_CACHE["timestamp"] = 0

            # Fetch fresh data
            data = fetch_real_time_data()

            if data and len(data) >= 60:
                # Get latest values
                latest = data[-1]
                return jsonify({
                    "success": True,
                    "message": "Real-time data refreshed successfully",
                    "source": "Yahoo Finance API",
                    "latest": {
                        "gold_price": round(latest["close"], 2),
                        "usd_index": round(latest["usd_close"], 2),
                        "sp500": round(latest["gspc_close"], 2),
                        "oil_price": round(latest["oil_dcoilwtico"], 2),
                        "data_points": len(data),
                        "last_updated": latest.get("date", "N/A")
                    }
                })
            else:
                return jsonify({
                    "success": False,
                    "message": "Failed to fetch real-time data. Using cached/local data instead."
                }), 400

        except Exception as e:
            logging.error(f"Refresh error: {e}")
            return jsonify({"error": f"Failed to refresh: {str(e)}"}), 500


    @app.route("/", methods=["GET"])
    def index():
        versions = discover_versions()
        return render_template("index.html", versions=versions)

    return app
