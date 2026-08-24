#!/usr/bin/env python
"""Test GPU checkpoint (full training) for CPU inference"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_gpu_checkpoint():
    from app import discover_versions, load_model, infer_from_payload, build_input_dataframe
    import torch

    # Use GPU checkpoint
    version = "version_0"
    ckpt_path = r"C:\Users\rh638\Downloads\materi penjelasan TFT\lightning_logs\version_0\checkpoints\epoch=11-step=192.ckpt"

    print(f"\n[TEST] Loading GPU checkpoint: {ckpt_path}")
    print(f"Size: {os.path.getsize(ckpt_path) / 1024 / 1024:.2f} MB")

    # Load model manually
    import pickle
    from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
    from pytorch_forecasting import TimeSeriesDataSet

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Try loading original training dataset
    dataset_pkl = r"C:\Users\rh638\Downloads\materi penjelasan TFT\lightning_logs\version_0\checkpoints\training_dataset.pkl"
    training_dataset = None

    try:
        with open(dataset_pkl, 'rb') as f:
            training_dataset = pickle.load(f)
        print(f"[SUCCESS] Loaded training dataset from PKL")

        # Check encodings
        print(f"[INFO] Categorical encodings:")
        for name, encoder in training_dataset.categorical_encoders.items():
            if hasattr(encoder, 'classes_'):
                print(f"  {name}: {len(encoder.classes_)} categories - {list(encoder.classes_[:5])}...")
    except Exception as e:
        print(f"[WARNING] Failed to load training dataset: {e}")

    # Get hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    print(f"[INFO] Hyperparameters keys: {list(hparams.keys())[:10]}")

    # Create model from training dataset
    if training_dataset:
        model = TemporalFusionTransformer.from_dataset(
            training_dataset,
            **hparams
        )

        # Load state dict
        state_dict = checkpoint['state_dict']
        model_state_dict = {}
        for key, value in state_dict.items():
            if not key.startswith('logging_metrics'):
                model_state_dict[key] = value

        model.load_state_dict(model_state_dict, strict=False)
        model.logging_metrics = torch.nn.ModuleList()
        model = model.cpu()
        model.eval()

        print(f"[SUCCESS] Model loaded on CPU")
        print(f"[INFO] Model device: {next(model.parameters()).device}")

        # Test prediction
        payload = {
            'version': 'version_0',
            'encoder_last_60': [{'close': 1900.0, 'open': 1899.0, 'tickvol': 50000, 'usd_close': 104.0, 'usd_open': 103.9, 'gspc_close': 5200.0, 'gspc_open': 5199.0, 'oil_dcoilwtico': 78.0, 'fed_rate_pct': 0.0525, 'cpi_cpiaucsl': 310.0}] * 60,
            'future_7': [{'usd_close': 104.0, 'usd_open': 103.9, 'gspc_close': 5200.0, 'gspc_open': 5199.0, 'oil_dcoilwtico': 78.0, 'fed_rate_pct': 0.0525, 'cpi_cpiaucsl': 310.0}] * 7
        }

        df = build_input_dataframe(payload['encoder_last_60'], payload['future_7'])
        print(f"[INFO] Input shape: {df.shape}")
        print(f"[INFO] Sample day_of_week: {df['day_of_week'].unique()[:5]}")
        print(f"[INFO] Sample month: {df['month'].unique()[:5]}")

        # Try inference
        from app import infer_tft_model
        preds = infer_tft_model(model, df)

        if preds:
            print(f"[SUCCESS] TFT Inference worked!")
            print(f"[INFO] Predictions: {preds[:3]}")
        else:
            print(f"[FAILED] TFT Inference failed, using fallback")
            preds, mode = infer_from_payload(payload, model)
            print(f"[INFO] Fallback predictions: {preds[:3]}")

if __name__ == "__main__":
    test_gpu_checkpoint()
