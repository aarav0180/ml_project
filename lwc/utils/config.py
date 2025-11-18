"""Configuration loading utilities."""

import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = 'configs/default_config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file (relative to project root)
        
    Returns:
        Configuration dictionary with resolved paths
    """
    # Get the project root (parent of lwc package)
    project_root = Path(__file__).parent.parent.parent
    
    # Resolve config path relative to project root
    if Path(config_path).is_absolute():
        config_file = Path(config_path)
    else:
        config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    logger.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Resolve all paths relative to project root
    config = _resolve_paths(config, project_root)
    
    return config


def _resolve_paths(config: Dict[str, Any], base_path: Path) -> Dict[str, Any]:
    """
    Resolve all paths in config relative to base_path.
    
    Args:
        config: Configuration dictionary
        base_path: Base path for resolution
        
    Returns:
        Configuration with resolved paths
    """
    if isinstance(config, dict):
        resolved = {}
        for key, value in config.items():
            if key.endswith('_path') or key.endswith('_paths'):
                if isinstance(value, str):
                    resolved[key] = str(base_path / value)
                elif isinstance(value, dict):
                    resolved[key] = _resolve_paths(value, base_path)
                else:
                    resolved[key] = value
            elif isinstance(value, (dict, list)):
                resolved[key] = _resolve_paths(value, base_path)
            else:
                resolved[key] = value
        return resolved
    elif isinstance(config, list):
        return [_resolve_paths(item, base_path) if isinstance(item, (dict, list)) else item for item in config]
    else:
        return config

