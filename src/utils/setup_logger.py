"""Process-wide logger writing to ``logs/`` and to stderr."""

import logging
import os
from datetime import datetime


def setup_logger():
    """Configure and return a timestamped file + console logger."""
    # check if the logger is already set up, if so return it, else set it up
    if len(logging.getLogger().handlers) > 0:
        return logging.getLogger(__name__)

    # Generate a timestamped log filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # create ./logs folder if needed (exist_ok avoids races between parallel jobs)
    os.makedirs("./logs", exist_ok=True)
    log_file = os.path.join("./logs", f"run_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),  # Log to a new file for each run
            logging.StreamHandler(),  # Also log to console
        ],
    )

    # Create a global logger instance
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")

    return logger
