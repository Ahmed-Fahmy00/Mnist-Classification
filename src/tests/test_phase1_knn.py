import numpy as np

from src.pipelines.phase1_knn import (
    normalize_pixels,
    select_binary_classes,
    split_data,
)


def test_select_binary_classes_filters_only_two_labels():
    x = np.random.randint(0, 255, size=(10, 28, 28), dtype=np.uint8)
    y = np.array([0, 1, 2, 3, 0, 1, 4, 5, 0, 1])

    x_bin, y_bin = select_binary_classes(x, y, 0, 1)

    assert x_bin.shape[0] == 6
    assert set(np.unique(y_bin)) == {0, 1}


def test_normalize_pixels_range():
    x = np.array([[[0, 255]]], dtype=np.uint8)
    normalized = normalize_pixels(x)

    assert normalized.dtype == np.float32
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)


def test_split_data_returns_expected_sizes():
    x = np.random.rand(100, 28, 28).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        x, y, test_size=0.2, val_size=0.2
    )

    assert x_test.shape[0] == 20
    assert x_val.shape[0] == 20
    assert x_train.shape[0] == 60
    assert len(y_train) + len(y_val) + len(y_test) == 100
