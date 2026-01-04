"""
Utility functions for the system
"""

import logging
import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: str = "./config/settings.yaml") -> Dict:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file

    Returns:
        dict: Configuration data
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML config: {e}")


def setup_logging(config: Dict) -> logging.Logger:
    """
    Setup logging configuration

    Args:
        config: Configuration dict

    Returns:
        Logger instance
    """
    log_config = config.get("logging", {})

    # Create logs directory
    log_file = Path(log_config.get("file", "./output/logs/system.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    level_str = log_config.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Format
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_enabled_channels(config: Dict) -> list:
    """
    Get list of enabled channels from config

    Args:
        config: Configuration dict

    Returns:
        list: Enabled channel configurations
    """
    channels = config.get("channels", [])
    enabled = [ch for ch in channels if ch.get("enabled", False)]
    return enabled
