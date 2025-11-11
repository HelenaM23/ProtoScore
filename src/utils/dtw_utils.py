import numba
import numpy as np
from joblib import Parallel, delayed

class DTWUtils:
    """Utilities for DTW calculations."""

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def fast_dtw_core_1d(x: np.ndarray, y: np.ndarray, window_size: int) -> float:
        """DTW für 1D-Zeitreihen."""
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            j_start = max(1, i - window_size)
            j_end = min(m + 1, i + window_size + 1)
            for j in range(j_start, j_end):
                cost = abs(x[i - 1] - y[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )
        return dtw_matrix[n, m]

    @staticmethod
    @numba.jit(nopython=True, fastmath=True)
    def fast_dtw_core_multivariate(x: np.ndarray, y: np.ndarray, window_size: int) -> float:
        """DTW für multivariate Zeitreihen."""
        n, m = len(x), len(y)
        dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
        dtw_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            j_start = max(1, i - window_size)
            j_end = min(m + 1, i + window_size + 1)
            for j in range(j_start, j_end):
                cost = np.sqrt(np.sum((x[i - 1] - y[j - 1]) ** 2))
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1]
                )
        return dtw_matrix[n, m]

    @staticmethod
    def dtw_cdist_with_progress(X, Y, window_fraction: float = 0.1, n_jobs: int = -1) -> np.ndarray:
        """Compute DTW distance matrix with progress tracking."""
        
        X = [np.asarray(seq, dtype=np.float32) for seq in X]
        Y = [np.asarray(seq, dtype=np.float32) for seq in Y]
        
        is_univariate = len(X[0].shape) == 1 if len(X) > 0 else True
        
        avg_length = np.mean([len(seq) for seq in X + Y])
        window_size = max(1, int(avg_length * window_fraction))
        
        min_seq_length = min(min(len(seq) for seq in X), min(len(seq) for seq in Y))
        if window_size >= min_seq_length:
            window_size = max(1, min_seq_length - 1)
            print(f"Warning: Window size adjusted to {window_size}")

        def compute_row(i):
            row = np.zeros(len(Y), dtype=np.float32)
            for j, y_seq in enumerate(Y):
                try:
                    if is_univariate:
                        row[j] = DTWUtils.fast_dtw_core_1d(X[i], y_seq, window_size)
                    else:
                        row[j] = DTWUtils.fast_dtw_core_multivariate(X[i], y_seq, window_size)
                except Exception as e:
                    print(f"DTW error at ({i},{j}): {e}")
                    row[j] = np.inf
            return row

        if n_jobs == 1:
            rows = [compute_row(i) for i in range(len(X))]
        else:
            rows = Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(compute_row)(i) for i in range(len(X))
            )

        return np.array(rows, dtype=np.float32)