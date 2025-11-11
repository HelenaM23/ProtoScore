"""
Dataset classes for prototype-based explainable AI methods.

This module implements the Template Method pattern and Factory pattern
to provide a clean, extensible dataset framework.
"""

import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Tuple
from types import SimpleNamespace

import keras
import numpy as np
import tensorflow as tf
from scipy.io import arff
from scipy.io.arff import loadarff
from scipy.stats import genextreme
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sktime.datasets._readers_writers.ts import load_from_tsfile_to_dataframe

logger = logging.getLogger(__name__)


# =====================================================
# EXCEPTIONS
# =====================================================

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


# =====================================================
# UTILITIES
# =====================================================

class DataSplitter:
    """Utility class for dataset splitting operations."""
    
    @staticmethod
    def split(
        x: np.ndarray, 
        y: np.ndarray, 
        test_size: float,
        random_state: int,
        validation_size: Optional[float] = None,
        stratify: bool = True,
        min_samples_per_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split dataset into training and validation sets.
        
        Args:
            x: Input features with shape (n_samples, ...)
            y: Target labels with shape (n_samples,)
            test_size: Proportion for validation split
            random_state: Random seed for reproducibility
            stratify: Whether to stratify the split
            
        Returns:
            Tuple of (x_train, x_val, y_train, y_val)
            
        Raises:
            DataProcessingError: If splitting fails
        """
        try:
            if min_samples_per_class is not None:
                return DataSplitter._stratified_split_with_all_labels(
                    x, y, test_size, validation_size, random_state, min_samples_per_class
                )
            
            stratify_param = y if stratify else None
            if validation_size is None:
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=stratify_param
                )
                return x_train, x_test, y_train, y_test
            
            x_train, x_temp, y_train, y_temp = train_test_split(
                x, y, 
                test_size=validation_size + test_size, 
                random_state=random_state,
                stratify=stratify_param
            )
            
            x_val, x_test, y_val, y_test = train_test_split(
                x_temp, y_temp, 
                test_size=test_size / (validation_size + test_size), 
                random_state=random_state,
                stratify=y_temp if stratify else None
            )
            
            return x_train, x_val, x_test, y_train, y_val, y_test
            
        except Exception as e:
            raise DataProcessingError(f"Failed to split dataset: {e}") from e

    @staticmethod
    def _stratified_split_with_all_labels(
        x: np.ndarray,
        y: np.ndarray,
        test_size: float,
        validation_size: Optional[float] = None,
        random_state: int = 42,
        min_samples_per_class: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform stratified split ensuring all labels appear in both splits.
        
        Args:
            x: Input features
            y: Target labels
            test_size: Proportion for validation split
            random_state: Random seed
            min_samples_per_class: Minimum samples per class in each split
            
        Returns:
            Tuple of (x_train, x_val, y_train, y_val)
        """
        np.random.seed(random_state)
        unique_labels = np.unique(y)
        
        train_indices = []
        val_indices = []
        test_indices = []
        
        for label in unique_labels:
            # Get all indices for this label
            label_indices = np.where(y == label)[0]
            n_samples = len(label_indices)
            
            # Calculate split size
            n_val = max(min_samples_per_class, int(n_samples * validation_size)) if validation_size else 0
            n_test = max(min_samples_per_class, int(n_samples * test_size))
            n_train = n_samples - n_val - n_test

            # Ensure minimum samples in train split
            if n_train < min_samples_per_class:
                n_train = min_samples_per_class
                n_test = n_samples - n_train - n_val
            
            # Shuffle indices for this label
            np.random.shuffle(label_indices)
            
            # Split indices
            train_indices.extend(label_indices[:n_train])
            if validation_size:
                val_indices.extend(label_indices[n_train:n_train + n_val])
            test_indices.extend(label_indices[n_train + n_val:n_train + n_val + n_test])
        
        # Convert to arrays and shuffle
        train_indices = np.array(train_indices)
        if validation_size:
            val_indices = np.array(val_indices)
        test_indices = np.array(test_indices)

        np.random.shuffle(train_indices)
        if validation_size:
            np.random.shuffle(val_indices)
        np.random.shuffle(test_indices)

        # Create splits
        x_train, y_train = x[train_indices], y[train_indices]
        if validation_size:
            x_val, y_val = x[val_indices], y[val_indices]
        x_test, y_test = x[test_indices], y[test_indices]

        # Validate that all labels are present
        DataSplitter._validate_all_labels_present(y, y_train, y_test, y_val if validation_size else None)
        
        if validation_size:
            return x_train, x_val, x_test, y_train, y_val, y_test
        else:
            return x_train, x_test, y_train, y_test

    @staticmethod
    def _validate_all_labels_present(
        y_original: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        y_val: Optional[np.ndarray] = None
    ) -> None:
        """
        Validate that all labels from original dataset appear in both splits.
        
        Args:
            y_original: Original labels
            y_train: Training labels
            y_val: Validation labels
            
        Raises:
            DataValidationError: If labels are missing in any split
        """
        unique_original = set(np.unique(y_original))
        unique_train = set(np.unique(y_train))
        unique_test = set(np.unique(y_test))
        if y_val is not None:
            unique_val = set(np.unique(y_val))
        
        missing_in_train = unique_original - unique_train
        missing_in_val = unique_original - unique_val if y_val is not None else None
        missing_in_test = unique_original - unique_test

        if missing_in_train:
            raise DataValidationError(
                f"Labels missing in training set: {missing_in_train}"
            )
        
        if missing_in_val:
            raise DataValidationError(
                f"Labels missing in validation set: {missing_in_val}"
            )
        if missing_in_test:
            raise DataValidationError(
                f"Labels missing in test set: {missing_in_test}"
            )
        logger.info(f"✓ All {len(unique_original)} labels present in train and validation splits")
    
# =====================================================
# BASE DATASET CLASS
# =====================================================

class BaseDataset(ABC):
    """
    Abstract base class for datasets using Template Method pattern.
    
    This class defines the common dataset workflow while allowing
    subclasses to implement dataset-specific details.
    """
    
    def __init__(self, **kwargs):
        """Initialize dataset with common parameters."""
        # Outlier noise parameters
        self._outlier_fraction = kwargs.get('outlier_fraction', 0.1)
        self._noise_seed = kwargs.get('noise_seed', 42)
        self._use_gev = kwargs.get('use_gev', True)
        self._gev_params = kwargs.get('gev_params', (0.1, 0.0, 1.0))
        self._apply_outlier_noise = kwargs.get('apply_outlier_noise', False)
        
        # split configurations
        self._min_samples_per_class = kwargs.get('min_samples_per_class', None)
        self._validation_size = kwargs.get('validation_size', None)
        self._test_size = kwargs.get('test_size', 0.2)
        self._stratified = kwargs.get('stratified', False)
        self._random_state = kwargs.get('random_state', 42)

        # Initialize utilities
        self._data_splitter = DataSplitter()
        
        # Initialize dataset attributes
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.n_classes = None
        self.num_classes = None
        
        # Load and preprocess data using template method
        self._execute_loading_pipeline()
    
    def _execute_loading_pipeline(self) -> None:
        """Template method for complete data loading pipeline."""
        try:
            logger.info(f"Loading dataset: {self.__class__.__name__}")
            
            # Template method steps
            self._load_data()
            self._validate_data()
            self._set_class_attributes()
            
            # Apply outlier noise if requested
            if self._apply_outlier_noise:
                self._apply_outlier_noise_to_splits()
            
            val_info = f"{len(self.x_val)} val, " if self.x_val is not None else ""
            logger.info(f"Dataset loaded successfully: {len(self.x_train)} train, "
                    f"{val_info}{len(self.x_test)} test samples")
            
        except Exception as e:
            logger.error(f"Failed to load dataset {self.__class__.__name__}: {e}")
            raise DataProcessingError(f"Dataset loading failed: {e}") from e
    
    @abstractmethod
    def _load_data(self) -> None:
        """Load and preprocess data. Must be implemented by subclasses."""
        pass
    
    def _validate_data(self) -> None:
        """Validate loaded data integrity."""
        if self.x_train is None or self.y_train is None:
            raise DataValidationError("Missing training data")
        if self.x_test is None or self.y_test is None:
            raise DataValidationError("Missing test data")
        
        if self.x_val is not None and self.y_val is not None:
            if len(self.x_val) != len(self.y_val):
                raise DataValidationError("Validation data shape mismatch")
        
        if len(self.x_train) != len(self.y_train):
            raise DataValidationError("Training data shape mismatch")
        if len(self.x_test) != len(self.y_test):
            raise DataValidationError("Test data shape mismatch")
    
    def _set_class_attributes(self) -> None:
        """Set class-related attributes."""
        if hasattr(self, 'n_classes') and self.n_classes is not None:
            self.num_classes = self.n_classes
        elif hasattr(self, 'y_train') and self.y_train is not None:
            self.n_classes = len(np.unique(self.y_train))
            self.num_classes = self.n_classes

    
    # =============================================================
    # PREPROCESSING METHODS
    # =============================================================
    
    def convert_to_namespace(self) -> Tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
        """Convert data to SimpleNamespace format."""
        train = SimpleNamespace(X=self.x_train, y=self.y_train)
        val = SimpleNamespace(X=self.x_val, y=self.y_val)
        test = SimpleNamespace(X=self.x_test, y=self.y_test)
        return train, val, test
    
    def ensure_one_hot_labels(
        self, 
        dataset: SimpleNamespace, 
        num_classes: Optional[int] = None
    ) -> SimpleNamespace:
        """
        Ensure labels are one-hot encoded.
        
        Args:
            dataset: Dataset with labels
            num_classes: Number of classes
            
        Returns:
            Dataset with one-hot encoded labels
            
        Raises:
            ValueError: For inconsistent labels
        """
        if not hasattr(dataset, "y") or dataset.y is None:
            return dataset
        
        # Check if already one-hot encoded
        if self._is_one_hot_encoded(dataset.y):
            logger.debug("Labels are already one-hot encoded")
            return dataset
        
        return self._convert_to_one_hot(dataset, num_classes)
    
    def _is_one_hot_encoded(self, labels: np.ndarray) -> bool:
        """Check if labels are already one-hot encoded."""
        return (len(labels.shape) == 2 and 
                labels.shape[1] > 1 and 
                np.allclose(np.sum(labels, axis=1), 1.0))
    
    def _convert_to_one_hot(
        self, 
        dataset: SimpleNamespace, 
        num_classes: Optional[int]
    ) -> SimpleNamespace:
        """Convert labels to one-hot format."""
        labels = self._extract_labels(dataset)
        
        if num_classes is None:
            num_classes = len(np.unique(labels))
        
        # Normalize labels to 0-based indices
        labels = self._normalize_labels(labels, num_classes)
        
        # Convert to one-hot
        dataset.y = keras.utils.to_categorical(labels, num_classes=num_classes)
        
        logger.debug(f"Labels converted to one-hot: {num_classes} classes")
        return dataset
    
    @staticmethod
    def _extract_labels(dataset: SimpleNamespace) -> np.ndarray:
        """Extract labels as 1D array."""
        if len(dataset.y.shape) == 2 and dataset.y.shape[1] == 1:
            return dataset.y.flatten()
        return dataset.y
    
    @staticmethod
    def _normalize_labels(labels: np.ndarray, num_classes: int) -> np.ndarray:
        """Normalize labels to 0-based indices."""
        min_label = np.min(labels)
        max_label = np.max(labels)
        unique_labels = np.unique(labels)
        
        # Automatic detection and correction
        if min_label == 1 and max_label == num_classes:
            # Convert 1-based to 0-based labels
            return labels - 1
        elif min_label == 0 and max_label == num_classes - 1:
            # Already 0-based
            return labels
        elif len(unique_labels) == num_classes:
            # General solution for arbitrary label values
            label_to_index = {
                label: idx for idx, label in enumerate(sorted(unique_labels))
            }
            return np.array([label_to_index[label] for label in labels])
        else:
            raise ValueError(
                f"Inconsistent labels: {len(unique_labels)} unique labels, "
                f"but {num_classes} classes expected"
            )

    
    # =============================================================
    # OUTLIER NOISE METHODS
    # =============================================================
    
    def _apply_outlier_noise_to_splits(self) -> None:
        """Apply outlier noise to all data splits."""
        splits = [('train', 0), ('val', 1), ('test', 2)]
        
        for split_name, offset in splits:
            data = getattr(self, f'x_{split_name}')
            if data is not None:
                num_outliers = int(len(data) * self._outlier_fraction)
                if num_outliers > 0:
                    rng = np.random.RandomState(self._noise_seed + offset)
                    outlier_indices = rng.choice(
                        len(data), size=num_outliers, replace=False
                    )
                    setattr(self, f'x_{split_name}', 
                           self._apply_outlier_noise(data, outlier_indices, offset))
    
    def _apply_outlier_noise(
        self, 
        data: np.ndarray, 
        indices: np.ndarray, 
        offset: int
    ) -> np.ndarray:
        """Apply outlier noise to selected samples."""
        rng = np.random.RandomState(self._noise_seed + offset)
        num_samples = len(indices)
        noise_shape = (num_samples,) + data.shape[1:]
        
        if self._use_gev:
            c, loc, scale = self._gev_params
            noise = genextreme.rvs(
                c, loc=loc, scale=scale, size=noise_shape, random_state=rng
            )
        else:
            noise = rng.randn(*noise_shape)
        
        data_copy = data.copy()
        data_copy[indices] = data_copy[indices] + noise
        return data_copy


# =====================================================
# DATASET IMPLEMENTATIONS
# =====================================================
class SAWSINE(BaseDataset):
    """SAWSINE dataset implementation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _load_data(self) -> None:
        """Load SAWSINE artificial data."""
        X, y = self._generate_sawsine_data()
        
        # Split data using configured parameters
        result = self._data_splitter.split(
            x=X, 
            y=y,
            random_state=self._random_state,
            test_size=self._test_size,
            validation_size=self._validation_size,
            stratify=self._stratified,
            min_samples_per_class=self._min_samples_per_class
        )
        
        if self._validation_size is not None:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = result
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = result
            self.x_val = None
            self.y_val = None
        
        self.n_classes = len(np.unique(self.y_train))
    
    def _generate_sawsine_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate artificial SAWSINE data."""
        len_ds = 8000
        noise_level = 1.1
        X = np.ones((len_ds, 110, 1))
        y = np.hstack((np.zeros(int(len_ds / 2)), np.ones(int(len_ds / 2))))

        for i in range(int(len_ds / 4)):
            X[i, :, 0] = self._add_noise(np.sin(np.arange(0, 11, 0.1)), noise_level)
            X[i + int(len_ds / 4), :, 0] = self._add_noise(np.abs(np.sin(np.arange(0, 11, 0.1))), noise_level)

            X[i + int(len_ds / 2), :, 0] = self._add_noise(
                np.hstack((np.zeros(30), np.ones(50) * np.arange(50) / -50, np.zeros(30))), noise_level)
            X[i + int(len_ds / 4 * 3), :, 0] = self._add_noise(
                np.hstack((np.zeros(20), np.ones(30), np.zeros(60))), noise_level)
        
        return X, y
    
    @staticmethod
    def _add_noise(signal: np.ndarray, noise_level: float) -> np.ndarray:
        """Add noise to signal."""
        return signal * (1 - np.random.rand(110) * noise_level) + np.random.rand(110) * noise_level


class ECG200(BaseDataset):
    """ECG200 dataset implementation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def _load_data(self) -> None:
        """Load ECG200 data."""
        filepath = Path("datasets/ECG200")
        X, y = self._get_ecg_data(filepath)
        y = self._convert_negatives_to_zero(y)

        result = self._data_splitter.split(
            x=X, 
            y=y,
            random_state=self._random_state,
            test_size=self._test_size,
            validation_size=self._validation_size,
            stratify=self._stratified,
            min_samples_per_class=self._min_samples_per_class
        )
        
        if self._validation_size is not None:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = result
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = result
            self.x_val = None
            self.y_val = None

        self.n_classes = len(np.unique(self.y_train))
        
    
    @staticmethod
    def _convert_negatives_to_zero(array: np.ndarray) -> np.ndarray:
        """Convert -1 labels to 0."""
        array[array == -1.] = 0.
        return array

    def _get_ecg_data(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load ECG200 data from files."""
        data_train = self._read_arff_file(filepath / "ECG200_TRAIN.arff")
        data_test = self._read_arff_file(filepath / "ECG200_TEST.arff")

        data_train_array = self._convert_arff_to_numpy(data_train[0])
        data_test_array = self._convert_arff_to_numpy(data_test[0])
        
        data_combined = np.concatenate([data_train_array, data_test_array])
        
        y = data_combined[:, -1]
        X = data_combined[:, :-1, np.newaxis]
        
        return X, y
    
    @staticmethod
    def _read_arff_file(file_path: Path) -> Tuple:
        """Read ARFF file."""
        with open(file_path, 'rt', encoding='utf-8') as f:
            data_read = f.read()
        stream = StringIO(data_read)
        return arff.loadarff(stream)
    
    @staticmethod
    def _convert_arff_to_numpy(data_list) -> np.ndarray:
        """Convert ARFF data list to numpy array."""
        data_array = np.empty(shape=(len(data_list), len(data_list[0])))
        for index, signal in enumerate(data_list):
            data_array[index, :] = list(signal)
        return data_array


class FordA_Dataset(BaseDataset):
    """Ford A dataset implementation."""
    
    def _load_data(self) -> None:
        """Load Ford A data from URL."""
        root_url = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
        x_train_raw, y_train_raw = self._read_ucr_data(root_url + "FordA_TRAIN.tsv")
        x_test_raw, y_test_raw = self._read_ucr_data(root_url + "FordA_TEST.tsv")

        # Combine all data for proper splitting
        X = np.concatenate([x_train_raw, x_test_raw], axis=0)
        y = np.concatenate([y_train_raw, y_test_raw], axis=0)
        
        # Preprocess: change labels from -1 to 0
        y = self._convert_negatives_to_zero(y)
        
        # Add third dimension for model
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split data using configured parameters
        result = self._data_splitter.split(
            x=X, 
            y=y,
            random_state=self._random_state,
            test_size=self._test_size,
            validation_size=self._validation_size,
            stratify=self._stratified,
            min_samples_per_class=self._min_samples_per_class
        )
        
        if self._validation_size is not None:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = result
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = result
            self.x_val = None
            self.y_val = None
        
        self.n_classes = len(np.unique(self.y_train))
    
    @staticmethod
    def _read_ucr_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read UCR format data."""
        data = np.loadtxt(filename, delimiter="\t")
        y = data[:, 0]
        x = data[:, 1:]
        return x, y.astype(int)
    
    @staticmethod
    def _convert_negatives_to_zero(y: np.ndarray) -> np.ndarray:
        """Convert -1 labels to 0."""
        y[y == -1] = 0
        return y


class FordB_Dataset(BaseDataset):
    """Ford B dataset implementation."""
    
    def _load_data(self) -> None:
        """Load Ford B data."""
        train_path = "datasets/FordB/FordB_TRAIN.arff"
        test_path = "datasets/FordB/FordB_TEST.arff"

        x_train_raw, y_train_raw = self._read_arff_data(train_path)
        x_test_raw, y_test_raw = self._read_arff_data(test_path)

        # Combine all data
        X = np.concatenate([x_train_raw, x_test_raw], axis=0)
        y = np.concatenate([y_train_raw, y_test_raw], axis=0)
        
        # Preprocess: change labels from -1 to 0
        y = self._convert_negatives_to_zero(y)
        
        # Reshape to 3D
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Scale data per sample
        X = self._scale_data_per_sample(X)
        
        # Split data using configured parameters
        result = self._data_splitter.split(
            x=X, 
            y=y,
            random_state=self._random_state,
            test_size=self._test_size,
            validation_size=self._validation_size,
            stratify=self._stratified,
            min_samples_per_class=self._min_samples_per_class
        )
        
        if self._validation_size is not None:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = result
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = result
            self.x_val = None
            self.y_val = None
        
        self.n_classes = len(np.unique(self.y_train))
    
    @staticmethod
    def _read_arff_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read data from ARFF file and return features and labels."""
        raw_data, meta = loadarff(path)
        data = np.zeros((raw_data.shape[0], len(meta.names())))
        for i, name in enumerate(meta.names()):
            data[:, i] = raw_data[name]
        x = data[:, :-1]
        y = data[:, -1].astype(int)
        return x, y

    @staticmethod
    def _convert_negatives_to_zero(y: np.ndarray) -> np.ndarray:
        """Convert -1 labels to 0."""
        y[y == -1] = 0
        return y

    @staticmethod
    def _scale_data_per_sample(x: np.ndarray) -> np.ndarray:
        """Scale data per sample using StandardScaler."""
        x_scaled = np.zeros_like(x)
        for i in range(x.shape[0]):
            scaler = StandardScaler()
            x_scaled[i, :, 0] = scaler.fit_transform(x[i, :, 0].reshape(-1, 1)).flatten()
        return x_scaled


class Wafer_Dataset(BaseDataset):
    """Wafer dataset implementation."""
    
    def _load_data(self) -> None:
        """Load Wafer data."""
        # Note: TEST and TRAIN files are swapped because test set is larger
        train_path = "datasets/Wafer/Wafer_TEST.arff"
        test_path = "datasets/Wafer/Wafer_TRAIN.arff"
        
        data_train = self._read_arff_data(train_path)
        data_test = self._read_arff_data(test_path)
        
        # Combine all data
        data_combined = np.concatenate([data_train, data_test], axis=0)
        
        # Convert to 3D and scale
        data3d = self._convert_to_3d(data_combined)
        data_scaled = self._scale_wafer_data(data3d)
        
        # Extract features and labels
        X = np.expand_dims(data_scaled[:, :, 0], axis=2)
        y = data_scaled[:, :, 1][:, 0]
        y = np.array([1 if x == 1. else 0 for x in y], dtype=int)
        
        # Split data using configured parameters
        result = self._data_splitter.split(
            x=X, 
            y=y,
            random_state=self._random_state,
            test_size=self._test_size,
            validation_size=self._validation_size,
            stratify=self._stratified,
            min_samples_per_class=self._min_samples_per_class
        )
        
        if self._validation_size is not None:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = result
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = result
            self.x_val = None
            self.y_val = None
        
        self.n_classes = len(np.unique(self.y_train))

    @staticmethod
    def _read_arff_data(path: str) -> np.ndarray:
        """Read ARFF file and return as numpy array."""
        raw_data, meta = loadarff(path)
        cols = [x for x in meta]
        data2d = np.zeros([raw_data.shape[0], len(cols)])
        for i, col in zip(range(len(cols)), cols):
            data2d[:, i] = raw_data[col]
        return data2d
    
    @staticmethod
    def _convert_to_3d(data: np.ndarray) -> np.ndarray:
        """Convert 2D data to 3D format."""
        x, y = data.shape
        data3d = np.zeros([x, y - 1, 2])
        for i in range(x):
            data3d[i, :, 0] = data[i][:-1].T
            data3d[i, :, 1] = np.full((y - 1), data[i][-1])
        return data3d
    
    @staticmethod
    def _scale_wafer_data(data: np.ndarray) -> np.ndarray:
        """Scale wafer data using StandardScaler."""
        df_scaled = np.zeros(data.shape)
        stder = StandardScaler()
        
        for i in range(data.shape[0]):
            df_scaled[i, :, 0] = stder.fit_transform(
                data[i, :, 0].reshape((data.shape[1], 1))
            ).reshape((data.shape[1]))
            df_scaled[i, :, 1] = data[i, :, 1]
        
        return df_scaled


class StarLightCurve(BaseDataset):
    """StarLight Curve dataset implementation."""
    
    def _load_data(self) -> None:
        """Load StarLight Curve data."""
        # Note: TEST and TRAIN files are swapped for better training data size
        train_path = "datasets/StarLightCurves/StarLightCurves_TEST.arff"
        test_path = "datasets/StarLightCurves/StarLightCurves_TRAIN.arff"

        x_train_raw, y_train_raw = self._read_arff_data(train_path)
        x_test_raw, y_test_raw = self._read_arff_data(test_path)

        # Combine all data
        X = np.concatenate([x_train_raw, x_test_raw], axis=0)
        y = np.concatenate([y_train_raw, y_test_raw], axis=0)
        
        # Preprocess: change labels from -1 to 0
        y = self._convert_negatives_to_zero(y)
        
        # Reshape to 3D
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Scale data per sample
        X = self._scale_data_per_sample(X)
        
        # Split data using configured parameters
        result = self._data_splitter.split(
            x=X, 
            y=y,
            random_state=self._random_state,
            test_size=self._test_size,
            validation_size=self._validation_size,
            stratify=self._stratified,
            min_samples_per_class=self._min_samples_per_class
        )
        
        if self._validation_size is not None:
            self.x_train, self.x_val, self.x_test, self.y_train, self.y_val, self.y_test = result
        else:
            self.x_train, self.x_test, self.y_train, self.y_test = result
            self.x_val = None
            self.y_val = None
        
        self.n_classes = len(np.unique(self.y_train))
    
    @staticmethod
    def _read_arff_data(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read data from ARFF file and return features and labels."""
        raw_data, meta = loadarff(path)
        data = np.zeros((raw_data.shape[0], len(meta.names())))
        for i, name in enumerate(meta.names()):
            data[:, i] = raw_data[name]
        x = data[:, :-1]
        y = data[:, -1].astype(int)
        return x, y

    @staticmethod
    def _convert_negatives_to_zero(y: np.ndarray) -> np.ndarray:
        """Convert -1 labels to 0."""
        y[y == -1] = 0
        return y

    @staticmethod
    def _scale_data_per_sample(x: np.ndarray) -> np.ndarray:
        """Scale data per sample using StandardScaler."""
        x_scaled = np.zeros_like(x)
        for i in range(x.shape[0]):
            scaler = StandardScaler()
            x_scaled[i, :, 0] = scaler.fit_transform(x[i, :, 0].reshape(-1, 1)).flatten()
        return x_scaled
    
    
# =====================================================
# FACTORY
# =====================================================

class DatasetFactory:
    """Factory for creating dataset instances."""
    
    _datasets = {
        "FordA": FordA_Dataset,
        "FordB": FordB_Dataset,
        "Wafer": Wafer_Dataset,
        "ECG200": ECG200,
        "Sawsine": SAWSINE,
        "StarLightCurve": StarLightCurve,
    }
    
    @classmethod
    def create_dataset(cls, name: str, **kwargs) -> BaseDataset:
        """Create dataset instance by name."""
        if name not in cls._datasets:
            available = list(cls._datasets.keys())
            raise ValueError(f"Unknown dataset: {name}. Available: {available}")
        
        return cls._datasets[name](**kwargs)
    
    @classmethod
    def get_available_datasets(cls) -> list:
        """Get list of available dataset names."""
        return list(cls._datasets.keys())
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class: type) -> None:
        """Register a new dataset class."""
        if not issubclass(dataset_class, BaseDataset):
            raise ValueError("Dataset class must inherit from BaseDataset")
        cls._datasets[name] = dataset_class