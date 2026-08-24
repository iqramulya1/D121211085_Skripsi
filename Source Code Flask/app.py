#!/usr/bin/env python
"""Flask TFT App - Main Entry Point"""
from app import create_app

app = create_app()

if __name__ == "__main__":
    print("=" * 50)
    print("🪙 Gold Price Predictor - TFT Model")
    print("=" * 50)
    print("Starting server at http://localhost:5000")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
