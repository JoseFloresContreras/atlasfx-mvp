#!/usr/bin/env python3
"""
Custom Logger Module
Provides logging functionality with Unicode support and timestamped log files.
"""

import logging
import os
from datetime import datetime
from typing import Optional


class CustomLogger:
    """
    Custom logger class that supports Unicode characters and writes to timestamped log files.
    """
    
    def __init__(self, log_directory: str = "logs"):
        """
        Initialize the custom logger.
        
        Args:
            log_directory (str): Directory to store log files
        """
        self.log_directory = log_directory
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup the logger with file and console handlers."""
        # Create logs directory if it doesn't exist
        os.makedirs(self.log_directory, exist_ok=True)
        
        # Generate timestamp for log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"log-{timestamp}.log"
        log_filepath = os.path.join(self.log_directory, log_filename)
        
        # Create logger
        self.logger = logging.getLogger('atlasfx_pipeline')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatter with Unicode support
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with Unicode support
        file_handler = logging.FileHandler(log_filepath, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler (disabled by default)
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        # self.logger.addHandler(console_handler)
        
        # Store the log filepath for reference
        self.log_filepath = log_filepath
        
        # Log initial setup
        self.info(f"Logger initialized. Log file: {log_filepath}")
    
    def debug(self, message: str, also_print: bool = False):
        """
        Log a debug message.
        
        Args:
            message (str): Debug message to log
            also_print (bool): Whether to also print to console (default: False)
        """
        self.logger.debug(message)
        if also_print:
            print(f"[DEBUG] {message}")
    
    def info(self, message: str, also_print: bool = False):
        """
        Log an info message.
        
        Args:
            message (str): Info message to log
            also_print (bool): Whether to also print to console (default: False)
        """
        self.logger.info(message)
        if also_print:
            print(f"[INFO] {message}")
    
    def warning(self, message: str, also_print: bool = False):
        """
        Log a warning message.
        
        Args:
            message (str): Warning message to log
            also_print (bool): Whether to also print to console (default: False)
        """
        self.logger.warning(message)
        if also_print:
            print(f"[WARNING] {message}")
    
    def error(self, message: str, also_print: bool = False):
        """
        Log an error message.
        
        Args:
            message (str): Error message to log
            also_print (bool): Whether to also print to console (default: False)
        """
        self.logger.error(message)
        if also_print:
            print(f"[ERROR] {message}")
    
    def critical(self, message: str, also_print: bool = False):
        """
        Log a critical message.
        
        Args:
            message (str): Critical message to log
            also_print (bool): Whether to also print to console (default: False)
        """
        self.logger.critical(message)
        if also_print:
            print(f"[CRITICAL] {message}")
    
    def get_log_filepath(self) -> str:
        """
        Get the current log file path.
        
        Returns:
            str: Path to the current log file
        """
        return self.log_filepath


# Create a global logger instance
log = CustomLogger() 