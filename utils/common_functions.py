import os
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException
import yaml

logger = get_logger(__name__)

def read_yaml(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"YAML file not found at: {file_path}")
        
        with open(file_path, 'r') as yaml_file:
            config =  yaml.safe_load(yaml_file)
            logger.info(f"Successfully read the YAML file from: {file_path}")
            return config
        
    except Exception as e:
        logger.error(f"Error reading YAML file: {e}")
        raise CustomException("Failed to read YAML file", e)