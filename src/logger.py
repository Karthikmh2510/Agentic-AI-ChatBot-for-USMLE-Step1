import logging
import os
from datetime import datetime

LOGS_DIR = "logs" # Directory where logs will be stored
os.makedirs(LOGS_DIR, exist_ok=True)

# To create a file like this log_2025-06-01
LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d')}")

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s - %(levelname)s - %(message)s", # Format of the log message
    level=logging.INFO, # Only levels INFO and above will be logged (warning, error, critical)
)

# Function to get a logger instance (this func is to initialize logger in different files)
def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)  # Set the logging level for this logger
    return logger