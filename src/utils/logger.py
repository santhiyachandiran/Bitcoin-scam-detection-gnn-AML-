"""
Centralized logging helper for consistent formatting across modules.
"""

import logging
import sys

def get_logger(name: str = "elliptic_gnn", level: int = logging.INFO) -> logging.Logger:
    """
    Returns a configured logger instance.
    
    Args:
        name: Name of the logger.
        level: Logging level (default INFO).
        
    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
