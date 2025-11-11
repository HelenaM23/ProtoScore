from collections import namedtuple

import numpy as np
from sklearn.preprocessing import OneHotEncoder


def preprocess_map(dataset):
    """
    Merges data and splits it again, since map needs idx.
    """
    Data = namedtuple("Data", ["X", "y", "idx"])

    # 1) Concatenate all splits to form the full dataset
    X_full = np.concatenate((dataset.x_train, dataset.x_val, dataset.x_test), axis=0)
    y_full = np.concatenate((dataset.y_train, dataset.y_val, dataset.y_test), axis=0)

    # 2) Build a global index array for the entire dataset
    idx_full = np.arange(len(X_full))

    # 3) Figure out how many samples are in each split
    n_train = len(dataset.x_train)
    n_val = len(dataset.x_val)
    n_test = len(dataset.x_test)

    # 4) Slice idx_full for train, val, test
    idx_train = idx_full[0:n_train]
    idx_val = idx_full[n_train : n_train + n_val]
    idx_test = idx_full[n_train + n_val : n_train + n_val + n_test]

    # 5) One-hot encode the *entire* label array, then slice
    #    - By default, OneHotEncoder returns a sparse matrix.
    #    - Use `sparse=False` (or `sparse_output=False` in newer sklearn)
    #      to get a dense numpy array.
    encoder = OneHotEncoder(
        categories="auto", sparse_output=False
    )  # or OneHotEncoder(sparse_output=False)
    y_full_1h = encoder.fit_transform(y_full.reshape(-1, 1))

    # 6) Create the three namedtuples
    train = Data(X=dataset.x_train, y=y_full_1h[idx_train], idx=idx_train)

    val = Data(X=dataset.x_val, y=y_full_1h[idx_val], idx=idx_val)

    test = Data(X=dataset.x_test, y=y_full_1h[idx_test], idx=idx_test)

    return train, val, test
