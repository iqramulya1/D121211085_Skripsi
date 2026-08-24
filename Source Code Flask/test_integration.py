# -*- coding: utf-8 -*-
"""
Test script untuk memverifikasi integrasi TFT dengan Flask app.
"""
import sys
import os
import io

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, load_model, discover_versions, build_input_dataframe, infer_from_payload, best_ckpt_for_version
import pandas as pd
import numpy as np

def test_config():
    """Test config import"""
    print("=" * 60)
    print("Test 1: Config Import")
    print("=" * 60)
    try:
        from app.config import (
            UNKNOWN_REALS, MAX_ENCODER_LENGTH, MAX_PREDICTION_LENGTH,
            MIN_ENCODER_LENGTH, MIN_PREDICTION_LENGTH, TARGET,
            GROUP_IDS, STATIC_CATEGORICALS, TIME_VARYING_KNOWN_CATEGORICALS,
            TIME_VARYING_KNOWN_REALS, DEFAULT_VALUES
        )
        print("[OK] Config import successful")
        print(f"   - UNKNOWN_REALS: {len(UNKNOWN_REALS)} columns")
        print(f"   - MAX_ENCODER_LENGTH: {MAX_ENCODER_LENGTH}")
        print(f"   - MAX_PREDICTION_LENGTH: {MAX_PREDICTION_LENGTH}")
        print(f"   - NO IDR in features: {'idr_close' not in UNKNOWN_REALS}")
        return True
    except Exception as e:
        print(f"[FAIL] Config import failed: {e}")
        return False


def test_model_discovery():
    """Test model discovery"""
    print("\n" + "=" * 60)
    print("Test 2: Model Discovery")
    print("=" * 60)
    try:
        versions = discover_versions()
        print(f"[OK] Found {len(versions)} version(s): {versions}")
        if versions:
            for v in versions:
                ckpt = best_ckpt_for_version(v)
                print(f"   - {v}: {ckpt}")
        return True
    except Exception as e:
        print(f"[FAIL] Model discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_loading():
    """Test model loading"""
    print("\n" + "=" * 60)
    print("Test 3: Model Loading")
    print("=" * 60)
    try:
        versions = discover_versions()
        if not versions:
            print("[WARN] No versions found, skipping model loading test")
            return True

        version = versions[0]
        model = load_model(version)
        if model:
            print(f"[OK] Model loaded successfully for {version}")
            print(f"   - Model type: {type(model).__name__}")
            return True
        else:
            print(f"[FAIL] Failed to load model for {version}")
            return False
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_build_input_dataframe():
    """Test build_input_dataframe function"""
    print("\n" + "=" * 60)
    print("Test 4: Build Input DataFrame")
    print("=" * 60)
    try:
        # Create sample encoder_last_60 data
        encoder_last_60 = []
        for i in range(60):
            encoder_last_60.append({
                "close": 1900.0 + i * 10,
                "open": 1895.0 + i * 10,
                "tickvol": 1000 + i * 10,
                "usd_close": 90.0,
                "usd_open": 89.9,
                "gspc_close": 3800.0,
                "gspc_open": 3795.0,
                "oil_dcoilwtico": 55.0,
                "fed_rate_pct": 0.09,
                "cpi_cpiaucsl": 260.0,
            })

        # Create sample future_7 data
        future_7 = []
        for i in range(7):
            future_7.append({
                "usd_close": 90.0,
                "usd_open": 89.9,
                "gspc_close": 3800.0,
                "gspc_open": 3795.0,
                "oil_dcoilwtico": 55.0,
                "fed_rate_pct": 0.09,
                "cpi_cpiaucsl": 260.0,
            })

        df = build_input_dataframe(encoder_last_60, future_7)
        print(f"[OK] DataFrame built successfully")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {list(df.columns[:10])}...")
        print(f"   - Has 'close': {'close' in df.columns}")
        print(f"   - Has 'idr_close': {'idr_close' in df.columns} (should be False)")
        print(f"   - time_idx range: {df['time_idx'].min()} to {df['time_idx'].max()}")
        return True
    except Exception as e:
        print(f"[FAIL] Build DataFrame failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference():
    """Test inference with model"""
    print("\n" + "=" * 60)
    print("Test 5: Inference")
    print("=" * 60)
    try:
        # Load model
        versions = discover_versions()
        if not versions:
            print("[WARN] No versions found, skipping inference test")
            return True

        version = versions[0]
        model = load_model(version)
        if not model:
            print("[WARN] Could not load model, testing fallback inference")
            model = None

        # Create sample payload
        encoder_last_60 = []
        for i in range(60):
            encoder_last_60.append({
                "close": 1900.0 + i * 10,
                "open": 1895.0 + i * 10,
                "tickvol": 1000 + i * 10,
                "usd_close": 90.0,
                "usd_open": 89.9,
                "gspc_close": 3800.0,
                "gspc_open": 3795.0,
                "oil_dcoilwtico": 55.0,
                "fed_rate_pct": 0.09,
                "cpi_cpiaucsl": 260.0,
            })

        future_7 = []
        for i in range(7):
            future_7.append({
                "usd_close": 90.0,
                "usd_open": 89.9,
                "gspc_close": 3800.0,
                "gspc_open": 3795.0,
                "oil_dcoilwtico": 55.0,
                "fed_rate_pct": 0.09,
                "cpi_cpiaucsl": 260.0,
            })

        payload = {
            "version": version,
            "encoder_last_60": encoder_last_60,
            "future_7": future_7
        }

        preds, mode = infer_from_payload(payload, model)
        print(f"[OK] Inference successful")
        print(f"   - Mode: {mode}")
        print(f"   - Predictions: {len(preds)} values")
        if preds:
            print(f"   - First prediction: {preds[0]}")
        return True
    except Exception as e:
        print(f"[FAIL] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print(" TFT Flask App Integration Test")
    print("=" * 60)

    results = []
    results.append(("Config Import", test_config()))
    results.append(("Model Discovery", test_model_discovery()))
    results.append(("Model Loading", test_model_loading()))
    results.append(("Build DataFrame", test_build_input_dataframe()))
    results.append(("Inference", test_inference()))

    print("\n" + "=" * 60)
    print(" Test Results Summary")
    print("=" * 60)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"   {name:<30} {status}")

    total = len(results)
    passed = sum(1 for _, r in results if r)
    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n[SUCCESS] All tests passed!")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
