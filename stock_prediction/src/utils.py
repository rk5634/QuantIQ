
import logging
from pathlib import Path
from datetime import datetime
from stock_prediction.config import settings

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """Function to setup as many loggers as you want"""
    
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = settings.LOG_DIR / f"{name}_{timestamp}.log"
    else:
        log_file = settings.LOG_DIR / log_file

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s: %(message)s')

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger
