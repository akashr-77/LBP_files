# features.py
# Responsible for everything between raw signal and model-ready tensor.
# The model was trained on specific windows — we must match that exactly.

import numpy as np
from typing import Tuple

#Constant parameters for feature extraction
WINDOW_SIZE = 2048 #samples per window
OVERLAP = WINDOW_SIZE // 2 #50% overlap
MODEL_CLASSES = [
    "Normal",
    "Ball-007",
    "Ball-014",
    "Ball-021",
    "IR-007",
    "IR-014",
    "IR-021",
    "OR-007",
    "OR-014",
    "OR-021",
]
NUM_CLASSES = len(MODEL_CLASSES)

def extract_windows(signal: list[float], window_size: int = WINDOW_SIZE, overlap: int = OVERLAP) -> np.ndarray:
    """
    Extracts overlapping many windows from the input signal.
    Args:
        signal: The input signal as a list of floats.
        window_size: The number of samples in each window.
        overlap: The number of samples to overlap between windows.
    """
    signal_np = np.array(signal, dtype=np.float32)
    windows = []

    if len(signal_np) < window_size:
        raise ValueError(f"Signal length {len(signal_np)} is shorter than the window size {window_size}.")
    
    step = window_size - overlap
    if(step<=0):
        raise ValueError(f"Overlap {overlap} must be less than the window size {window_size}.")
    
    n_segments = (len(signal_np) - window_size) // step + 1
    segments = np.zeros((n_segments, window_size, 1), dtype=np.float32)
    for i in range(n_segments):
        start = i * step
        end   = start + window_size
        segments[i, :, 0] = signal_np[start:end]
    
    return segments

def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """
    Normalizes each window to have zero mean and unit variance.
    Args:
        windows: A 3D numpy array of shape (num_windows, window_size, 1).
    """
    mean = np.mean(windows, axis=1, keepdims=True)
    std = np.std(windows, axis=1, keepdims=True) + 1e-8 # avoid division by zero
    normalized = (windows - mean) / std
    return normalized

def reshape(windows: np.ndarray) -> np.ndarray:
    """
    Reshapes the windows to match the model's expected input shape.
    Args:
        windows: A 3D numpy array of shape (num_windows, window_size, 1).
    """
    return windows.reshape(-1, WINDOW_SIZE, 1)

