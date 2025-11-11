"""
Utility functions and classes for prototype-based explainable AI methods.

This module provides helper functionality following the Single Responsibility
Principle with proper separation of concerns.
"""

import argparse
import json
import pickle
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
import pandas as pd
from ..core.data import DataProcessingError

class FileOperationError(Exception):
    """Custom exception for file operation errors."""
    pass



class FileHandler(ABC):
    """Abstract base class for file handlers using Strategy pattern."""
    
    @abstractmethod
    def read(self, filepath: str) -> Any:
        """Read data from file."""
        pass
    
    @abstractmethod
    def write(self, data: Any, filepath: str) -> None:
        """Write data to file."""
        pass


class HDF5Handler(FileHandler):
    """Handler for HDF5 files."""
    
    def read(self, filepath: str) -> Dict[str, Any]:
        """Read HDF5 file and return keys."""
        try:
            with h5py.File(filepath, "r") as f:
                return {"keys": list(f.keys())}
        except Exception as e:
            raise FileOperationError(f"Failed to read HDF5 file {filepath}: {e}") from e
    
    def write(self, data: Any, filepath: str) -> None:
        """Write data to HDF5 file."""
        # Implementation would depend on data structure
        raise NotImplementedError("HDF5 writing not implemented")
    
    def inspect_file(self, filepath: str) -> None:
        """Inspect HDF5 file and print keys."""
        try:
            result = self.read(filepath)
            print(f"HDF5 file keys: {result['keys']}")
        except FileOperationError as e:
            print(f"Error: {e}")


class PickleHandler(FileHandler):
    """Handler for pickle files."""
    
    def read(self, filepath: str) -> Any:
        """Read pickle file."""
        try:
            with open(filepath, "rb") as file:
                return pickle.load(file)
        except FileNotFoundError:
            raise FileOperationError(f"Pickle file not found: {filepath}")
        except pickle.UnpicklingError:
            raise FileOperationError(f"Failed to unpickle file: {filepath}")
        except Exception as e:
            raise FileOperationError(f"Error reading pickle file {filepath}: {e}") from e
    
    def write(self, data: Any, filepath: str) -> None:
        """Write data to pickle file."""
        try:
            with open(filepath, "wb") as file:
                pickle.dump(data, file)
        except Exception as e:
            raise FileOperationError(f"Error writing pickle file {filepath}: {e}") from e


class CSVHandler(FileHandler):
    """Handler for CSV files."""
    
    def read(self, filepath: str) -> pd.DataFrame:
        """Read CSV file."""
        try:
            return pd.read_csv(filepath)
        except Exception as e:
            raise FileOperationError(f"Error reading CSV file {filepath}: {e}") from e
    
    def write(self, data: pd.DataFrame, filepath: str) -> None:
        """Write DataFrame to CSV file."""
        try:
            data.to_csv(filepath, index=False)
        except Exception as e:
            raise FileOperationError(f"Error writing CSV file {filepath}: {e}") from e


class JSONHandler(FileHandler):
    """Handler for JSON files."""
    
    def read(self, filepath: str) -> Dict[str, Any]:
        """Read JSON file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileOperationError(f"JSON file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise FileOperationError(f"Invalid JSON format in {filepath}: {e}")
        except Exception as e:
            raise FileOperationError(f"Error reading JSON file {filepath}: {e}") from e
    
    def write(self, data: Dict[str, Any], filepath: str) -> None:
        """Write data to JSON file."""
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise FileOperationError(f"Error writing JSON file {filepath}: {e}") from e


class FileHandlerFactory:
    """Factory for creating file handlers."""
    
    _handlers = {
        '.h5': HDF5Handler,
        '.hdf5': HDF5Handler,
        '.pkl': PickleHandler,
        '.pickle': PickleHandler,
        '.csv': CSVHandler,
        '.json': JSONHandler,
    }
    
    @classmethod
    def create_handler(cls, filepath: str) -> FileHandler:
        """Create appropriate file handler based on file extension."""
        extension = Path(filepath).suffix.lower()
        
        if extension not in cls._handlers:
            raise ValueError(f"Unsupported file type: {extension}")
        
        return cls._handlers[extension]()


class PrototypeExtractor:
    """
    Utility class for extracting prototypes from different sources.
    
    Uses the Strategy pattern for different extraction methods.
    """
    
    def __init__(self, file_handler_factory: FileHandlerFactory = None):
        """Initialize with optional file handler factory."""
        self._factory = file_handler_factory or FileHandlerFactory()
    
    def extract_from_csv_map(self, filepath: str) -> np.ndarray:
        """
        Extract latent centers from a CSV file (MAP method format).
        
        Args:
            filepath: Path to the CSV file containing latent centers
            
        Returns:
            NumPy array of latent centers
            
        Raises:
            DataProcessingError: If extraction fails
        """
        try:
            handler = self._factory.create_handler(filepath)
            df = handler.read(filepath)
            
            if "latent_centers" not in df.columns:
                raise DataProcessingError("CSV file must contain 'latent_centers' column")
            
            string_data = df["latent_centers"][0]
            return self._parse_latent_centers_string(string_data)
            
        except FileOperationError as e:
            raise DataProcessingError(f"Failed to extract from CSV: {e}") from e
    
    def _parse_latent_centers_string(self, string_data: str) -> np.ndarray:
        """Parse latent centers from string representation."""
        try:
            string_content = string_data.strip()[1:-1]
            rows = re.split(r"\]\s*\n\s*\[", string_content)
            
            data = []
            for row in rows:
                row = row.strip("[] \n")
                numbers = re.findall(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", row)
                numbers = [float(num) for num in numbers]
                data.append(numbers)
            
            return np.array(data)
            
        except Exception as e:
            raise DataProcessingError(f"Failed to parse latent centers: {e}") from e


class ConfigurationManager:
    """Manager for handling configuration files and arguments."""
    
    def __init__(self, json_handler: JSONHandler = None):
        """Initialize with optional JSON handler."""
        self._json_handler = json_handler or JSONHandler()
    
    def load_from_json(self, json_path: str) -> argparse.Namespace:
        """
        Load arguments from JSON configuration file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            Namespace object with loaded arguments
            
        Raises:
            FileOperationError: If loading fails
        """
        try:
            config = self._json_handler.read(json_path)
            parser = self._create_parser_from_config(config)
            
            args = parser.parse_args([])  # Empty args list for defaults
            args.sparse = True  # Add default sparse attribute
            
            return args
            
        except FileOperationError:
            raise
        except Exception as e:
            raise FileOperationError(f"Failed to load configuration: {e}") from e
    
    def _create_parser_from_config(self, config: Dict[str, Any]) -> argparse.ArgumentParser:
        """Create argument parser from configuration dictionary."""
        parser = argparse.ArgumentParser()
        
        for key, value in config.items():
            if value is not None:
                parser.add_argument(f"--{key}", default=value, type=type(value))
            else:
                parser.add_argument(f"--{key}", default=None)
        
        return parser