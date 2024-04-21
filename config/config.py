import logging as log

import yaml


def load_config(file_path):
    """
    Load configuration from a YAML file.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        log.warning(f"Error: Configuration file not found at {file_path}.")
    except yaml.YAMLError as e:
        log.warning(f"Error loading configuration from {file_path}: {e}")
