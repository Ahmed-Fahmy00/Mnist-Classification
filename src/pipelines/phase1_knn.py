"""Compatibility wrappers for older imports and tests.

The project now uses notebook orchestration and reusable modules in:
- src/features/mnist_features.py
- src/models/knn_model.py
"""

import os
from typing import Tuple

import numpy as np
from dotenv import load_dotenv

from src.features.mnist_features import (
    load_mnist as feature_load_mnist,
    normalize_pixels as feature_normalize_pixels,
    select_binary_classes as feature_select_binary_classes,
    split_data as feature_split_data,
)

load_dotenv()
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))


def load_mnist(npz_path) -> Tuple[np.ndarray, np.ndarray]:
    return feature_load_mnist(str(npz_path))


def select_binary_classes(
    x: np.ndarray, y: np.ndarray, class_a: int, class_b: int
) -> Tuple[np.ndarray, np.ndarray]:
    return feature_select_binary_classes(x, y, class_a, class_b)


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return feature_split_data(x, y, test_size, val_size, RANDOM_STATE)


def normalize_pixels(x: np.ndarray) -> np.ndarray:
    return feature_normalize_pixels(x)

