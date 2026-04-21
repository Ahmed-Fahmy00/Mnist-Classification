import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA

def load_mnist(npz_path):
    """Load MNIST dataset from a .npz file."""
    data = np.load(npz_path)

    X_train = data['x_train']
    y_train = data['y_train']
    X_test = data['x_test']
    y_test = data['y_test']

    x_all = np.concatenate((X_train, X_test), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)

    return x_all, y_all

def select_binary_classes(x, y, class_a, class_b):
    """Select a binary classification subset from the dataset."""
    mask = (y == class_a) | (y == class_b)
    x_binary = x[mask]
    y_binary = y[mask]

    return x_binary, y_binary

def class_distribution(y):
    """Calculate the class distribution in the dataset."""
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    return distribution

def split_data(x, y, test_size, val_size, random_state):
    """Split the dataset into training, validation, and test sets."""
    np.random.seed(random_state)
    indices = np.random.permutation(len(x))

    test_split = int(len(x) * test_size)
    val_split = int(len(x) * val_size)

    x_test, y_test = [], []
    x_val, y_val = [], []
    x_train, y_train = [], []

    # Fill test set
    for i in range(test_split):
        x_test.append(x[indices[i]])
        y_test.append(y[indices[i]])

    # Fill validation set
    for i in range(test_split, test_split + val_split):
        x_val.append(x[indices[i]])
        y_val.append(y[indices[i]])

    # Fill training set
    for i in range(test_split + val_split, len(x)):
        x_train.append(x[indices[i]])
        y_train.append(y[indices[i]])

    return np.array(x_train), np.array(y_train),np.array(x_val), np.array(y_val), np.array(x_test), np.array(y_test)

def normalize_data(x):
    """Normalize the pixel values to the range [0, 1]."""
    return x.astype(np.float32) / 255.0

def balance_binary_classes(x, y, method='undersample', random_state=42):
    """Balance binary classes by random under/over sampling on the provided split."""
    labels, counts = np.unique(y, return_counts=True)
    if labels.shape[0] != 2:
        raise ValueError("balance_binary_classes expects exactly 2 classes")

    if method not in {'undersample', 'oversample'}:
        raise ValueError("method must be 'undersample' or 'oversample'")

    idx_a = np.where(y == labels[0])[0]
    idx_b = np.where(y == labels[1])[0]
    rng = np.random.default_rng(random_state)

    if method == 'undersample':
        target = min(len(idx_a), len(idx_b))
        idx_a = rng.choice(idx_a, size=target, replace=False)
        idx_b = rng.choice(idx_b, size=target, replace=False)
    else:
        target = max(len(idx_a), len(idx_b))
        idx_a = rng.choice(idx_a, size=target, replace=True)
        idx_b = rng.choice(idx_b, size=target, replace=True)

    balanced_idx = np.concatenate([idx_a, idx_b])
    rng.shuffle(balanced_idx)
    return x[balanced_idx], y[balanced_idx]

def standardize_by_train(train_features, val_features, test_features):
    """Standardize features using the mean and std of the training set."""
    mean = np.mean(train_features, axis=0)
    std = np.std(train_features, axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # Avoid division by zero

    train_standardized = (train_features - mean) / std
    val_standardized = (val_features - mean) / std
    test_standardized = (test_features - mean) / std

    return train_standardized, val_standardized, test_standardized

def build_features(mode, x_train, x_val, x_test, pca_components=0.95, random_state=42):
    """Build features based on the specified mode."""
    if mode == 'flatten':
        train_features = extract_flatten_features(x_train)
        val_features = extract_flatten_features(x_val)
        test_features = extract_flatten_features(x_test)
    elif mode == 'hog':
        train_features = extract_hog_features(x_train)
        val_features = extract_hog_features(x_val)
        test_features = extract_hog_features(x_test)
    elif mode == 'pca':
        x_train_flat = extract_flatten_features(x_train)
        x_val_flat = extract_flatten_features(x_val)
        x_test_flat = extract_flatten_features(x_test)

        pca = PCA(n_components=pca_components, random_state=random_state)
        train_features = pca.fit_transform(x_train_flat)
        val_features = pca.transform(x_val_flat)
        test_features = pca.transform(x_test_flat)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Use 'flatten', 'hog', or 'pca'.")

    # Standardize features
    train_standardized, val_standardized, test_standardized = standardize_by_train(train_features, val_features, test_features)

    return train_standardized, val_standardized, test_standardized

def extract_flatten_features(x):
    """Flatten the 28x28 images into 784-dimensional vectors."""
    return x.reshape(x.shape[0], -1)

def extract_hog_features(x, pixels_per_cell=(4,4), cells_per_block=(2,2), orientations=9):
    """Extract Histogram of Oriented Gradients (HOG) features from the images."""
    hog_features = []
    for img in x:
        features = hog(img, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, orientations=orientations)
        hog_features.append(features)

    return np.array(hog_features)

def extract_pca_features(x, n_components=0.95, random_state=42):
    """Extract PCA features from one image array by flattening then applying PCA."""
    x_flat = extract_flatten_features(x)
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca.fit_transform(x_flat)