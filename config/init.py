"""
Configuration utilities for crop disease detection project
Author: Ali SU - Hacettepe University
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def get_model_config(model_type: str, model_name: str) -> Dict[str, Any]:
    """
    Get specific model configuration
    
    Args:
        model_type: 'cnn' or 'vit'
        model_name: Specific model name
        
    Returns:
        Model configuration dictionary
    """
    config = load_config()
    
    if model_type not in config['models']:
        raise ValueError(f"Model type '{model_type}' not found in config")
    
    if model_name not in config['models'][model_type]:
        raise ValueError(f"Model '{model_name}' not found in {model_type} models")
    
    return config['models'][model_type][model_name]

def get_training_config(phase: str = None) -> Dict[str, Any]:
    """
    Get training configuration
    
    Args:
        phase: Specific training phase ('phase1' or 'phase2')
        
    Returns:
        Training configuration dictionary
    """
    config = load_config()
    
    if phase:
        return config['training'][phase]
    else:
        return config['training']

# Quick access functions
def get_dataset_config() -> Dict[str, Any]:
    """Get dataset configuration"""
    return load_config()['dataset']

def get_results_config() -> Dict[str, Any]:
    """Get results from previous runs"""
    return load_config()['results']

__all__ = [
    'load_config',
    'get_model_config', 
    'get_training_config',
    'get_dataset_config',
    'get_results_config'
]
