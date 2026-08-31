"""
Utility Functions Module
General utilities for the project.
"""

import yaml
import pickle
import joblib
import json
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to config file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_model(model: Any, filepath: str, format: str = 'pkl') -> bool:
    """
    Save trained model to disk.
    
    Args:
        model: Model to save
        filepath (str): Path to save model
        format (str): Format to use ('pkl' or 'joblib')
        
    Returns:
        bool: Success status
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if format == 'pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        elif format == 'joblib':
            joblib.dump(model, filepath)
        else:
            logger.error(f"Unknown format: {format}")
            return False
        
        logger.info(f"Model saved to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(filepath: str, format: str = 'pkl') -> Optional[Any]:
    """
    Load trained model from disk.
    
    Args:
        filepath (str): Path to model file
        format (str): Format of model ('pkl' or 'joblib')
        
    Returns:
        Any: Loaded model or None
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"Model file not found: {filepath}")
        return None
    
    try:
        if format == 'pkl':
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
        elif format == 'joblib':
            model = joblib.load(filepath)
        else:
            logger.error(f"Unknown format: {format}")
            return None
        
        logger.info(f"Model loaded from {filepath}")
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def save_config(config: Dict[str, Any], filepath: str) -> bool:
    """
    Save configuration to YAML file.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        filepath (str): Path to save config
        
    Returns:
        bool: Success status
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False


def save_results(results: Dict[str, Any], filepath: str) -> bool:
    """
    Save results to JSON file.
    
    Args:
        results (Dict[str, Any]): Results dictionary
        filepath (str): Path to save results
        
    Returns:
        bool: Success status
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return False


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level (str): Logging level
        log_file (str): Path to log file (optional)
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers
    )


class MetricsTracker:
    """Track and store model metrics."""
    
    def __init__(self):
        """Initialize metrics tracker."""
        self.metrics = []
    
    def add_metrics(self, metrics: Dict[str, Any], model_name: str = None):
        """
        Add metrics to tracker.
        
        Args:
            metrics (Dict[str, Any]): Metrics dictionary
            model_name (str): Name of model
        """
        entry = {'model': model_name or 'unknown', **metrics}
        self.metrics.append(entry)
    
    def get_best_model(self, metric: str = 'r2') -> Optional[Dict[str, Any]]:
        """
        Get best performing model.
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            Dict[str, Any]: Best model metrics
        """
        if not self.metrics:
            return None
        
        return max(self.metrics, key=lambda x: x.get(metric, float('-inf')))
    
    def to_dataframe(self):
        """
        Convert metrics to pandas DataFrame.
        
        Returns:
            pd.DataFrame: Metrics dataframe
        """
        import pandas as pd
        return pd.DataFrame(self.metrics)
    
    def save(self, filepath: str):
        """
        Save metrics to file.
        
        Args:
            filepath (str): Path to save metrics
        """
        self.to_dataframe().to_csv(filepath, index=False)
        logger.info(f"Metrics saved to {filepath}")


def print_model_summary(model: Any, model_name: str = 'Model'):
    """
    Print model summary.
    
    Args:
        model: Model object
        model_name (str): Name of model
    """
    print(f"\n{'='*50}")
    print(f"Model Summary - {model_name}")
    print(f"{'='*50}")
    
    if hasattr(model, 'model_type'):
        print(f"Type: {model.model_type}")
    
    if hasattr(model, 'model'):
        print(f"Details: {model.model}")
    
    print(f"{'='*50}\n")
