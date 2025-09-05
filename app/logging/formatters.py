"""Custom logging formatters for structured and JSON logging."""

import json
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional


class StructuredFormatter(logging.Formatter):
    """Structured formatter for human-readable logs in development."""
    
    def __init__(self):
        super().__init__()
        self.base_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured information."""
        # Format the base message
        formatted = super().format(record)
        
        # Add extra fields if present
        extra_fields = self._get_extra_fields(record)
        if extra_fields:
            extra_str = " | ".join([f"{k}={v}" for k, v in extra_fields.items()])
            formatted += f" | {extra_str}"
        
        # Add exception information if present
        if record.exc_info:
            formatted += f"\n{self.formatException(record.exc_info)}"
        
        return formatted
    
    def _get_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract extra fields from log record."""
        # Standard fields to exclude
        standard_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'message', 'exc_info',
            'exc_text', 'stack_info', 'asctime'
        }
        
        extra = {}
        for key, value in record.__dict__.items():
            if key not in standard_fields:
                extra[key] = value
        
        return extra


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging in production."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields
        extra_fields = self._get_extra_fields(record)
        if extra_fields:
            log_entry.update(extra_fields)
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_entry, default=str, ensure_ascii=False)
    
    def _get_extra_fields(self, record: logging.LogRecord) -> Dict[str, Any]:
        """Extract extra fields from log record."""
        # Standard fields to exclude
        standard_fields = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 'filename',
            'module', 'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'message', 'exc_info',
            'exc_text', 'stack_info'
        }
        
        extra = {}
        for key, value in record.__dict__.items():
            if key not in standard_fields:
                extra[key] = value
        
        return extra


class RequestFormatter(logging.Formatter):
    """Specialized formatter for HTTP request/response logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format HTTP request/response log records."""
        if hasattr(record, 'request_id'):
            # This is an HTTP request log
            return self._format_request_log(record)
        else:
            # Fall back to standard formatting
            return super().format(record)
    
    def _format_request_log(self, record: logging.LogRecord) -> str:
        """Format HTTP request log with relevant details."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "request_id": getattr(record, 'request_id', None),
            "method": getattr(record, 'method', None),
            "path": getattr(record, 'path', None),
            "status_code": getattr(record, 'status_code', None),
            "duration_ms": getattr(record, 'duration_ms', None),
            "user_agent": getattr(record, 'user_agent', None),
            "client_ip": getattr(record, 'client_ip', None),
            "message": record.getMessage(),
        }
        
        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        return json.dumps(log_data, default=str, ensure_ascii=False)