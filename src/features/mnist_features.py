from typing import Dict, Tuple

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_mnist(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load MNIST arrays from an .npz file and merge train/test into one pool."""
    data = np.load(npz_path)

    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    x_all = np.concatenate([x_train, x_test], axis=0)
    y_all = np.concatenate([y_train, y_test], axis=0)

    return x_all, y_all


def select_binary_classes(
    x: np.ndarray, y: np.ndarray, class_a: int, class_b: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter dataset to only two selected classes."""
    mask = (y == class_a) | (y == class_b)
    return x[mask], y[mask]


def split_data(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float,
    val_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create stratified train/validation/test splits."""
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    val_relative = val_size / (1.0 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_relative,
        random_state=random_state,
        stratify=y_train_val,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def normalize_pixels(x: np.ndarray) -> np.ndarray:
    """Normalize grayscale pixels from [0, 255] to [0, 1]."""
    return x.astype(np.float32) / 255.0


def extract_flatten_features(x: np.ndarray) -> np.ndarray:
    return x.reshape(x.shape[0], -1)


def extract_hog_features(
    x: np.ndarray,
    pixels_per_cell: Tuple[int, int] = (4, 4),
    cells_per_block: Tuple[int, int] = (2, 2),
    orientations: int = 9,
) -> np.ndarray:
    features = []
    for image in x:
        features.append(
            hog(
                image,
                orientations=orientations,
                pixels_per_cell=pixels_per_cell,
                cells_per_block=cells_per_block,
                block_norm="L2-Hys",
                feature_vector=True,
            )
        )
    return np.asarray(features)


def build_features(
    mode: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    pca_components: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Build selected feature set and apply train-fitted transforms only."""
    info: Dict[str, float] = {}

    if mode == "flatten":
        train_features = extract_flatten_features(x_train)
        val_features = extract_flatten_features(x_val)
        test_features = extract_flatten_features(x_test)
    elif mode == "pca":
        train_flat = extract_flatten_features(x_train)
        val_flat = extract_flatten_features(x_val)
        test_flat = extract_flatten_features(x_test)

        pca = PCA(n_components=pca_components, random_state=random_state)
        train_features = pca.fit_transform(train_flat)
        val_features = pca.transform(val_flat)
        test_features = pca.transform(test_flat)
        info["pca_components_retained"] = float(pca.n_components_)
        info["pca_explained_variance"] = float(np.sum(pca.explained_variance_ratio_))
    elif mode == "hog":
        train_features = extract_hog_features(x_train)
        val_features = extract_hog_features(x_val)
        test_features = extract_hog_features(x_test)
    else:
        raise ValueError(f"Unknown feature mode: {mode}")

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    return train_features, val_features, test_features, info


def class_distribution(y: np.ndarray) -> Dict[int, int]:
    classes, counts = np.unique(y, return_counts=True)
    return {int(c): int(n) for c, n in zip(classes, counts)}
