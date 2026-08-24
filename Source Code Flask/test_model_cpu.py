#!/usr/bin/env python
"""Test script untuk verifikasi model TFT bisa load dan inference tanpa GPU"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_1_discover_versions():
    """Test 1: Temukan semua version di lightning_logs"""
    from app import discover_versions

    versions = discover_versions()
    print(f"\n[TEST 1] Versions found: {versions}")
    assert len(versions) > 0, "No versions found"
    print("[PASS] Versions discovered")
    return versions[0]  # Return first version for next test


def test_2_find_checkpoint(version):
    """Test 2: Temukan checkpoint file"""
    from app import best_ckpt_for_version

    ckpt = best_ckpt_for_version(version)
    print(f"\n[TEST 2] Checkpoint: {ckpt}")
    assert ckpt is not None, f"No checkpoint for {version}"
    assert os.path.isfile(ckpt), f"Checkpoint not a file: {ckpt}"
    size_mb = os.path.getsize(ckpt) / 1024 / 1024
    print(f"[PASS] Checkpoint found ({size_mb:.2f} MB)")
    return ckpt


def test_3_load_model(version):
    """Test 3: Load model dari checkpoint (CPU MODE)"""
    from app import load_model

    print(f"\n[TEST 3] Loading model {version} on CPU...")
    model = load_model(version)

    assert model is not None, "Model is None"
    print(f"[PASS] Model loaded successfully")

    # Check model is on CPU
    import torch
    device = next(model.parameters()).device
    print(f"[INFO] Model device: {device}")
    assert device.type == "cpu", f"Model not on CPU: {device}"
    print("[PASS] Model is on CPU")

    return model


def test_4_inference(model):
    """Test 4: Jalankan inference dengan dummy data"""
    import pandas as pd
    import numpy as np
    from app import build_input_dataframe, infer_from_payload

    print(f"\n[TEST 4] Testing inference...")

    # Buat dummy data - 60 hari historical + 7 hari future
    encoder_60 = []
    for i in range(60):
        encoder_60.append({
            "close": 1900.0 + i * 0.5,
            "open": 1899.0 + i * 0.5,
            "tickvol": 50000,
            "usd_close": 104.0,
            "usd_open": 103.9,
            "gspc_close": 5200.0,
            "gspc_open": 5199.0,
            "oil_dcoilwtico": 78.0,
            "fed_rate_pct": 0.0525,
            "cpi_cpiaucsl": 310.0,
        })

    future_7 = []
    for i in range(7):
        future_7.append({
            "usd_close": 104.0,
            "usd_open": 103.9,
            "gspc_close": 5200.0,
            "gspc_open": 5199.0,
            "oil_dcoilwtico": 78.0,
            "fed_rate_pct": 0.0525,
            "cpi_cpiaucsl": 310.0,
        })

    # Test build dataframe
    df = build_input_dataframe(encoder_60, future_7)
    print(f"[INFO] Input DataFrame shape: {df.shape}")
    assert df.shape[0] == 67, f"Expected 67 rows, got {df.shape[0]}"
    print("[PASS] Input DataFrame built")

    # Test inference
    payload = {
        "version": "version_0",
        "encoder_last_60": encoder_60,
        "future_7": future_7
    }

    preds, mode = infer_from_payload(payload, model)
    print(f"[INFO] Inference mode: {mode}")
    print(f"[INFO] Predictions: {preds[:3]}...")

    assert preds is not None, "Predictions is None"
    assert len(preds) == 7, f"Expected 7 predictions, got {len(preds)}"
    print("[PASS] Inference successful")

    return preds


def main():
    print("=" * 60)
    print("TFT Model CPU Inference Test Suite")
    print("=" * 60)

    try:
        version = test_1_discover_versions()
        test_2_find_checkpoint(version)
        model = test_3_load_model(version)
        test_4_inference(model)

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nModel TFT siap digunakan di Flask app tanpa GPU.")
        return 0

    except Exception as e:
        print(f"\n[FAILED] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
