"""Logging utilities and helper functions."""

import functools
import time
from typing import Any, Callable, Dict, Optional, TypeVar, Union

from .config import get_logger, log_exception, log_performance

F = TypeVar('F', bound=Callable[..., Any])


def log_function_call(
    logger_name: Optional[str] = None,
    log_args: bool = False,
    log_result: bool = False,
    log_performance: bool = True
) -> Callable[[F], F]:
    """Decorator to log function calls with optional arguments and results."""
    
    def decorator(func: F) -> F:
        logger = get_logger(logger_name or func.__module__.split('.')[-1])
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Prepare log context
            log_extra = {
                "function": func.__name__,
                "module": func.__module__,
            }
            
            if log_args:
                log_extra.update({
                    "args": str(args) if args else None,
                    "kwargs": kwargs if kwargs else None,
                })
            
            logger.debug(f"Calling function: {func.__name__}", extra=log_extra)
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful completion
                duration = time.time() - start_time
                completion_extra = log_extra.copy()
                
                if log_performance:
                    completion_extra["duration_ms"] = round(duration * 1000, 2)
                
                if log_result:
                    completion_extra["result"] = str(result)
                
                logger.debug(
                    f"Function completed: {func.__name__}",
                    extra=completion_extra
                )
                
                return result
                
            except Exception as exc:
                duration = time.time() - start_time
                error_extra = log_extra.copy()
                error_extra.update({
                    "duration_ms": round(duration * 1000, 2),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                })
                
                logger.error(
                    f"Function failed: {func.__name__}",
                    extra=error_extra,
                    exc_info=exc
                )
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Prepare log context
            log_extra = {
                "function": func.__name__,
                "module": func.__module__,
            }
            
            if log_args:
                log_extra.update({
                    "args": str(args) if args else None,
                    "kwargs": kwargs if kwargs else None,
                })
            
            logger.debug(f"Calling function: {func.__name__}", extra=log_extra)
            
            try:
                result = func(*args, **kwargs)
                
                # Log successful completion
                duration = time.time() - start_time
                completion_extra = log_extra.copy()
                
                if log_performance:
                    completion_extra["duration_ms"] = round(duration * 1000, 2)
                
                if log_result:
                    completion_extra["result"] = str(result)
                
                logger.debug(
                    f"Function completed: {func.__name__}",
                    extra=completion_extra
                )
                
                return result
                
            except Exception as exc:
                duration = time.time() - start_time
                error_extra = log_extra.copy()
                error_extra.update({
                    "duration_ms": round(duration * 1000, 2),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                })
                
                logger.error(
                    f"Function failed: {func.__name__}",
                    extra=error_extra,
                    exc_info=exc
                )
                raise
        
        # Return appropriate wrapper based on function type
        if hasattr(func, '__code__') and func.__code__.co_flags & 0x80:  # CO_COROUTINE
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger_name: str, **context):
        self.logger = get_logger(logger_name)
        self.context = context
        self.original_extra = {}
    
    def __enter__(self):
        # Store original extra data if any
        if hasattr(self.logger, '_extra_context'):
            self.original_extra = self.logger._extra_context.copy()
        else:
            self.logger._extra_context = {}
        
        # Add new context
        self.logger._extra_context.update(self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original context
        if self.original_extra:
            self.logger._extra_context = self.original_extra
        else:
            delattr(self.logger, '_extra_context')


def create_audit_logger(name: str) -> 'AuditLogger':
    """Create an audit logger for tracking important system events."""
    return AuditLogger(name)


class AuditLogger:
    """Specialized logger for audit events and security-related logging."""
    
    def __init__(self, name: str):
        self.logger = get_logger(f"audit.{name}")
    
    def log_user_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        result: str = "success",
        **extra
    ):
        """Log user actions for audit purposes."""
        audit_data = {
            "audit_type": "user_action",
            "action": action,
            "user_id": user_id,
            "resource": resource,
            "result": result,
            **extra
        }
        
        self.logger.info(f"User action: {action}", extra=audit_data)
    
    def log_system_event(
        self,
        event: str,
        component: str,
        severity: str = "info",
        **extra
    ):
        """Log system events for monitoring and debugging."""
        audit_data = {
            "audit_type": "system_event",
            "event": event,
            "component": component,
            "severity": severity,
            **extra
        }
        
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(f"System event: {event}", extra=audit_data)
    
    def log_security_event(
        self,
        event: str,
        threat_level: str = "low",
        source_ip: Optional[str] = None,
        **extra
    ):
        """Log security-related events."""
        audit_data = {
            "audit_type": "security_event",
            "event": event,
            "threat_level": threat_level,
            "source_ip": source_ip,
            **extra
        }
        
        # Security events are always logged as warnings or errors
        log_method = self.logger.error if threat_level in ["high", "critical"] else self.logger.warning
        log_method(f"Security event: {event}", extra=audit_data)