# Gold Price Predictor - TFT Model

Production-ready web application for gold price forecasting using Temporal Fusion Transformer (TFT) deep learning model.

## Features

### 🎯 Three Input Methods

1. **CSV Upload**
   - Upload historical gold price data in CSV format
   - Automatic validation and preview
   - Drag & drop support
   - Minimum 60 rows required

2. **Manual Input**
   - Enter current gold price manually
   - Input market indicators (USD Index, S&P 500, Oil Price, Fed Rate, CPI)
   - Select market trend scenario
   - Automatically generates historical patterns

3. **Sample Scenarios**
   - Pre-configured scenarios: Bullish, Bearish, Sideways, Volatile
   - One-click scenario testing
   - Perfect for demonstration

### 📊 Output Features

- **Interactive Chart**: Plotly-powered visualization
- **7-Day Forecast Table**: Daily predictions with change percentages
- **Forecast Summary**: Key metrics including price range and total change
- **CSV Export**: Download predictions for further analysis
- **Model Status Indicator**: Shows whether TFT model or fallback mode is used

## Installation

```bash
cd flask_tft_app
pip install -r requirements.txt
```

## Running the Application

```bash
python -m app.flask_app
```

The app will be available at: `http://localhost:5000`

## API Endpoints

### `GET /`
Web interface

### `POST /predict`
Submit prediction request

**Request Body:**
```json
{
  "version": "version_0",
  "encoder_last_60": [
    {
      "close": 2350.50,
      "open": 2348.20,
      "tickvol": 55000,
      "usd_close": 103.50,
      "usd_open": 103.45,
      "gspc_close": 4780.20,
      "gspc_open": 4775.30,
      "oil_dcoilwtico": 72.50,
      "fed_rate_pct": 0.0525,
      "cpi_cpiaucsl": 310.20
    }
    // ... 60 rows total
  ],
  "future_7": [
    {
      "usd_close": 103.50,
      "usd_open": 103.45,
      "gspc_close": 4780.20,
      "gspc_open": 4775.30,
      "oil_dcoilwtico": 72.50,
      "fed_rate_pct": 0.0525,
      "cpi_cpiaucsl": 310.20
    }
    // ... 7 rows total
  ]
}
```

**Response:**
```json
{
  "version": "version_0",
  "predictions": [
    {"time_idx": 61, "prediction": 2355.30},
    {"time_idx": 62, "prediction": 2358.70}
    // ... 7 predictions total
  ],
  "mode": "TFT-model"
}
```

### `GET /status`
Check model status

### `GET /health`
Health check endpoint

### `POST /upload-csv`
Upload and validate CSV file

## CSV Format

### Required Columns

| Column | Description | Example |
|--------|-------------|---------|
| `close` | Gold closing price (USD/oz) | 2350.50 |
| `open` | Gold opening price (USD/oz) | 2348.20 |
| `usd_close` | USD Index closing value | 103.50 |
| `gspc_close` | S&P 500 closing value | 4780.20 |
| `oil_dcoilwtico` | WTI Oil price | 72.50 |
| `fed_rate_pct` | Federal funds rate (decimal) | 0.0525 |
| `cpi_cpiaucsl` | CPI Index value | 310.20 |

### Optional Columns

- `date`: Date column (format: YYYY-MM-DD)
- `tickvol`: Trading volume
- `usd_open`, `usd_adj_close`: USD Index additional values
- `gspc_open`: S&P 500 opening value
- `fed_target_rate_from_pct`, `fed_target_rate_to_pct`: Fed target rates

### Sample CSV

```csv
date,close,open,usd_close,gspc_close,oil_dcoilwtico,fed_rate_pct,cpi_cpiaucsl
2024-01-01,2350.50,2348.20,103.50,4780.20,72.50,0.0525,310.20
2024-01-02,2355.30,2351.80,103.45,4795.40,73.20,0.0525,310.25
...
```

A sample CSV is available at: `/Dataset/sample_gold_data.csv`

## Model Information

- **Model Type**: Temporal Fusion Transformer (TFT)
- **Input Window**: 60 days historical data
- **Forecast Horizon**: 7 days
- **Features**: 32 engineered features including lag prices, moving averages, returns, volatility, and external market indicators

## Production Deployment

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app.flask_app:app
```

### Using Docker

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app.flask_app:app"]
```

### Environment Variables

- `FLASK_ENV`: Set to `production` for production mode
- `FLASK_DEBUG`: Set to `False` in production

## Troubleshooting

### Model Not Loading

If the model badge shows "Fallback Mode":

1. Check that model files exist in `../Source Code Model/models/version_0/checkpoints/`
2. Ensure `training_dataset.pkl` exists in the same directory
3. Check logs for specific error messages

### CSV Upload Issues

- Ensure CSV has at least 60 data rows
- Verify required columns are present
- Check that numeric values don't contain text
- Remove any special characters from headers

### Prediction Errors

- Verify all numeric inputs are valid numbers
- Check that encoder data has exactly 60 rows
- Check that future data has exactly 7 rows
- Ensure all required columns are present

## License

This project is part of the TFT Gold Price Prediction system.

## Support

For issues or questions, please check the logs in the Flask application console.
