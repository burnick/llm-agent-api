"""Logging configuration and setup."""

import logging
import logging.config
import sys
from typing import Dict, Any, Optional

from app.config.models import Environment, LogLevel
from .formatters import StructuredFormatter, JSONFormatter


def get_logging_config(
    environment: Environment,
    log_level: LogLevel,
    enable_json: bool = False
) -> Dict[str, Any]:
    """Get logging configuration based on environment and settings."""
    
    # Choose formatter based on environment and preference
    if environment == Environment.PRODUCTION or enable_json:
        formatter_class = "app.logging.formatters.JSONFormatter"
    else:
        formatter_class = "app.logging.formatters.StructuredFormatter"
    
    config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": formatter_class,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level.value,
                "formatter": "structured",
                "stream": sys.stdout,
            },
        },
        "loggers": {
            # Application loggers
            "app": {
                "level": log_level.value,
                "handlers": ["console"],
                "propagate": False,
            },
            # FastAPI and Uvicorn loggers
            "uvicorn": {
                "level": log_level.value,
                "handlers": ["console"],
                "propagate": False,
            },
            "uvicorn.access": {
                "level": log_level.value,
                "handlers": ["console"],
                "propagate": False,
            },
            "fastapi": {
                "level": log_level.value,
                "handlers": ["console"],
                "propagate": False,
            },
        },
        "root": {
            "level": log_level.value,
            "handlers": ["console"],
        },
    }
    
    # Add file logging for production
    if environment == Environment.PRODUCTION:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": log_level.value,
            "formatter": "structured",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        }
        
        # Add file handler to all loggers
        for logger_config in config["loggers"].values():
            logger_config["handlers"].append("file")
        config["root"]["handlers"].append("file")
    
    return config


def setup_logging(
    environment: Environment = Environment.DEVELOPMENT,
    log_level: LogLevel = LogLevel.INFO,
    enable_json: bool = False
) -> None:
    """Setup logging configuration for the application."""
    
    # Create logs directory for production
    if environment == Environment.PRODUCTION:
        import os
        os.makedirs("logs", exist_ok=True)
    
    # Get and apply logging configuration
    config = get_logging_config(environment, log_level, enable_json)
    logging.config.dictConfig(config)
    
    # Log startup information
    logger = logging.getLogger("app.logging")
    logger.info(
        "Logging configured",
        extra={
            "environment": environment.value,
            "log_level": log_level.value,
            "json_format": enable_json,
        }
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name."""
    return logging.getLogger(f"app.{name}")


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger instance for this class."""
        return get_logger(self.__class__.__name__.lower())


def log_exception(
    logger: logging.Logger,
    message: str,
    exc_info: Optional[Exception] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Log an exception with detailed context."""
    log_extra = extra or {}
    
    if exc_info:
        log_extra.update({
            "exception_type": type(exc_info).__name__,
            "exception_message": str(exc_info),
        })
    
    logger.error(message, extra=log_extra, exc_info=exc_info)


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration: float,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """Log performance metrics for an operation."""
    log_extra = extra or {}
    log_extra.update({
        "operation": operation,
        "duration_ms": round(duration * 1000, 2),
        "performance": True,
    })
    
    logger.info(f"Operation completed: {operation}", extra=log_extra)