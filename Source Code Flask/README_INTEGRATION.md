# TFT Flask App Integration - Summary

## Fixed Issues

### 1. Categorical Encoding Mismatch
**Problem:** `IndexError: index out of range in self` during TFT inference
**Root Cause:** Creating new `TimeSeriesDataSet()` creates different categorical encodings than the training data
**Solution:**
- Save training dataset as pickle during training (`train_tft_cpu_complete.py`)
- Load training dataset pickle during model loading (`load_model()`)
- Use `training_dataset.from_dataset()` for inference to preserve encodings

### 2. Weekend Date Mapping
**Problem:** Training data only has weekdays (stock market), but inference generated all 7 days
**Root Cause:** `add_date_features()` generated Saturday/Sunday which weren't in training data
**Solution:** Map weekend days to Friday in `add_date_features()`:
```python
weekend_map = {"Saturday": "Friday", "Sunday": "Friday"}
day_name = weekend_map.get(day_name, day_name)
```

## Files Modified

1. **`train_tft_cpu_complete.py`**
   - Added training dataset pickle saving (line 422-433)

2. **`save_training_dataset.py`** (NEW)
   - Script to create training dataset pickle from existing checkpoint

3. **`flask_tft_app/app/__init__.py`**
   - Updated `load_model()` to load training dataset pickle (line 90-125)
   - Updated `infer_tft_model()` to use `from_dataset()` (line 388-425)
   - Updated `add_date_features()` to map weekends (line 214-236)

## Usage

### Run Flask App
```bash
cd flask_tft_app
python -m app.flask_app
```

### Test API
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "version": "version_0",
    "encoder_last_60": [...],
    "future_7": [...]
  }'
```

## Model Status
- **Checkpoint:** `Source Code Model/models/version_0/checkpoints/tft_cpu_model-epoch=0-val_loss=94.9073.ckpt`
- **Training Dataset:** `Source Code Model/models/version_0/checkpoints/training_dataset.pkl`
- **Status:** Working TFT inference
- **Mode:** TFT-model (not fallback)

## Next Steps for Better Performance
1. Train for more epochs (currently only epoch 0)
2. Run with `train_tft_cpu_complete.py` for full training with Optuna tuning
