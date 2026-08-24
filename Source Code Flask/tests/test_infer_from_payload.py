import types
import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.tft_inference import infer_from_payload


class DummyModel:
    def predict(self, df):
        # return simple increasing values as predictions
        return [i for i in range(7)]


def test_infer_from_payload_with_mock_model():
    # Create a minimal payload
    encoder_last_60 = [{"USD": 1.0, "GSPC": 100.0, "IDR": 14000.0, "Oil": 70.0, "FedRate": 0.25, "CPI": 2.5} for _ in range(60)]
    future_7 = [{"USD": 1.0, "GSPC": 98.0, "IDR": 14050.0, "Oil": 71.0, "FedRate": 0.25, "CPI": 2.5} for _ in range(7)]
    payload = {"version": "version_0", "encoder_last_60": encoder_last_60, "future_7": future_7}
    preds = infer_from_payload(payload, DummyModel())
    # Expect 7 predictions with time_idx 61..67
    assert len(preds) == 7
    assert preds[0]["time_idx"] == 61
    assert isinstance(preds[0]["prediction"], float) or isinstance(preds[0]["prediction"], (int, float))
