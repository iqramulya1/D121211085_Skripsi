#!/bin/bash
echo "=== Flask TFT App - Test Prediction ==="
echo ""

echo "1. Health Check:"
curl -s http://localhost:5000/health | python -m json.tool
echo ""

echo "2. Predict:"
curl -s -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "version": "version_0",
    "encoder_last_60": [
      {"close": 1950.0, "open": 1949.0, "tickvol": 50000, "usd_close": 104.0, "usd_open": 103.9, "gspc_close": 5200.0, "gspc_open": 5199.0, "oil_dcoilwtico": 78.0, "fed_rate_pct": 0.0525, "cpi_cpiaucsl": 310.0}
    ] * 60,
    "future_7": [
      {"usd_close": 104.0, "usd_open": 103.9, "gspc_close": 5200.0, "gspc_open": 5199.0, "oil_dcoilwtico": 78.0, "fed_rate_pct": 0.0525, "cpi_cpiaucsl": 310.0}
    ] * 7
  }' | python -m json.tool
