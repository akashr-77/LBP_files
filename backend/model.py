# model.py
# Singleton pattern: model is loaded ONCE when the process starts.
# Every request reuses the same in-memory model → no 2-3s reload latency.
# Thread-safe: Keras predict() releases the GIL; concurrent requests are fine.

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "Models" / "best_cwru_cnn.keras"
MODEL_PATH = os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))

class BearingModel:
    """
    Wrapper around the loaded Keras model.
    Holds the model object + metadata extracted at load time.
    """
    def __init__(self):
        self._model: Optional[tf.keras.Model] = None
        # self.model_name: Optional[str] = None
        self._loaded_at: Optional[str] = None
        self._input_shape: Optional[list] = None
        self._num_params: Optional[int] = None

    def load(self, model_path: str = MODEL_PATH):
        """
        Called once at app startup (from main.py lifespan).
        Logs model summary so you can verify architecture in the terminal.
        """
        model_path = Path(model_path)
        if not model_path.is_absolute():
            model_path = (PROJECT_ROOT / model_path).resolve()

        logger.info(f"Loading model from: {model_path}")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: '{model_path}'. "
                f"Expected the default model under '{DEFAULT_MODEL_PATH}' or a valid MODEL_PATH value."
            )
        try:
            self._model = tf.keras.models.load_model(str(model_path))
            self._loaded_at = datetime.utcnow().isoformat() + "Z"
            self._input_shape = list(self._model.input_shape)
            self._num_params = self._model.count_params()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError(f"Failed to load model from '{model_path}'. See logs for details.")

        logger.info(f"Model loaded. Input shape: {self._input_shape}")
        logger.info(f"Parameters: {self._num_params:,}")
        logger.info(f"Output units: {self._model.output_shape}")

        # Warm-up inference: first call compiles the TF graph (takes ~0.5s)
        # Subsequent calls are fast. Do this at startup, not on first request.
        dummy = np.zeros((1, *self._input_shape[1:]), dtype=np.float32)
        _ = self._model.predict(dummy, verbose=0)
        logger.info("Warm-up inference done. Model ready.")

    @property
    #can use function like a varible
    def is_loaded(self) -> bool:
        return self._model is not None
    
    def predict_proba(self, windows: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the input windows.
        Args:
            windows: A 3D numpy array of shape (num_windows, window_size, 1).
        Returns:
            A 2D numpy array of shape (num_windows, num_classes) with class probabilities.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() before predict_proba().")
        return self._model.predict(windows, verbose=0)

    def get_model_status(self) -> dict:
        """
        Returns a dictionary with model status information.
        """
        if not self.is_loaded:
            return {
                "status": "degraded",
                "model_loaded": False,
            }
        return{
            "status": "ok",
            "model_loaded": True,
            "timestamp": self._loaded_at,
        }
    
bearing_model = BearingModel()