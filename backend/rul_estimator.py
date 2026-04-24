import logging
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUL_MODEL_DIR = PROJECT_ROOT / "RUL_final"

RUL_MODEL_CONFIG = {
    "outer": {
        "model_path": RUL_MODEL_DIR / "outer_race_model.pkl",
        "scaler_path": RUL_MODEL_DIR / "outer_race_scaler.pkl",
        "note": "Confidence=High (R2=0.97,MAE=7.18 minutes)",
    },
    "inner": {
        "model_path": RUL_MODEL_DIR / "inner_race_model.pkl",
        "scaler_path": RUL_MODEL_DIR / "inner_race_scaler.pkl",
        "note": "Confidence=Medium-High(R2=0.76,MAE=51.27 minutes)",
    },
    "ball": {
        "model_path": RUL_MODEL_DIR / "ball_model.pkl",
        "scaler_path": RUL_MODEL_DIR / "ball_scaler.pkl",
        "note": "Confidence=Medium-High (R2=0.79,MAE=54.89 minutes)",
    },
}

FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "max",
    "min",
    "peak_to_peak",
    "skewness",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "entropy",
    "fft_mean",
    "fft_std",
    "fft_max",
    "energy",
]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator not in (0, 0.0) else 0.0


def extract_features_from_raw(signal: np.ndarray) -> pd.DataFrame:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    if signal.size == 0:
        raise ValueError("Signal is empty.")

    f_mean = np.mean(signal)
    f_std = np.std(signal)
    f_rms = np.sqrt(np.mean(signal**2))
    f_max = np.max(signal)
    f_min = np.min(signal)
    f_p2p = np.ptp(signal)
    f_skew = skew(signal)
    f_kurt = kurtosis(signal)

    abs_mean = np.mean(np.abs(signal))
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(signal)))

    f_crest = _safe_ratio(f_max, f_rms)
    f_shape = _safe_ratio(f_rms, abs_mean)
    f_impulse = _safe_ratio(f_max, abs_mean)
    f_clearance = _safe_ratio(f_max, sqrt_abs_mean**2)

    hist, _ = np.histogram(signal, bins=100, density=True)
    f_entropy = entropy(hist + 1e-9)

    fft_vals = np.abs(np.fft.rfft(signal))
    f_fft_mean = np.mean(fft_vals)
    f_fft_std = np.std(fft_vals)
    f_fft_max = np.max(fft_vals)

    f_energy = np.sum(signal**2)

    return pd.DataFrame(
        [[
            f_mean,
            f_std,
            f_rms,
            f_max,
            f_min,
            f_p2p,
            f_skew,
            f_kurt,
            f_crest,
            f_shape,
            f_impulse,
            f_clearance,
            f_entropy,
            f_fft_mean,
            f_fft_std,
            f_fft_max,
            f_energy,
        ]],
        columns=FEATURE_NAMES,
    )


def fault_group_from_class(fault_class: str) -> str:
    if fault_class == "Normal":
        return "normal"
    if fault_class.startswith("Ball"):
        return "ball"
    if fault_class.startswith("IR"):
        return "inner"
    if fault_class.startswith("OR"):
        return "outer"
    return "normal"


class RULPredictor:
    def __init__(self):
        self._loaded = False
        self._models: dict[str, dict[str, Any]] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        self._models.clear()

        for group, config in RUL_MODEL_CONFIG.items():
            model_path = config["model_path"]
            scaler_path = config["scaler_path"]

            if not model_path.exists():
                raise FileNotFoundError(f"RUL model file not found: {model_path}")
            if not scaler_path.exists():
                raise FileNotFoundError(f"RUL scaler file not found: {scaler_path}")

            self._models[group] = {
                "model": joblib.load(model_path),
                "scaler": joblib.load(scaler_path),
                "note": config["note"],
            }

        self._loaded = True
        logger.info("RUL predictor loaded: outer, inner, and ball models are ready.")

    def predict(self, signal: np.ndarray, fault_class: str) -> dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("RUL predictor is not loaded. Call load() first.")

        group = fault_group_from_class(fault_class)
        if group == "normal":
            return {
                "rul_estimate": None,
                "rul_units": None,
                "rul_note": "System Healthy — RUL is designer-defined for the Normal class.",
                "rul_model": None,
            }

        bundle = self._models[group]
        features = extract_features_from_raw(signal)
        scaled_features = bundle["scaler"].transform(features)
        prediction = float(bundle["model"].predict(scaled_features)[0])
        prediction = max(0.0, prediction)

        if prediction >= 2000.0:
            return {
                "rul_estimate": 2000.0,
                "rul_units": "minutes",
                "rul_note": f"Actual RUL >= 2000 (Bearing in healthy period). {bundle['note']}",
                "rul_model": group,
            }

        return {
            "rul_estimate": round(prediction, 1),
            "rul_units": "minutes",
            "rul_note": bundle["note"],
            "rul_model": group,
        }


rul_predictor = RULPredictor()