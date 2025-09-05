"""Logging middleware for FastAPI request/response logging."""

import time
import uuid
from typing import Callable, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_logger


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    def __init__(self, app, logger_name: str = "middleware"):
        super().__init__(app)
        self.logger = get_logger(logger_name)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add request ID to request state for use in other parts of the app
        request.state.request_id = request_id
        
        # Extract request details
        start_time = time.time()
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else None
        user_agent = request.headers.get("user-agent")
        client_ip = self._get_client_ip(request)
        
        # Log incoming request
        self.logger.info(
            f"Incoming request: {method} {path}",
            extra={
                "request_id": request_id,
                "method": method,
                "path": path,
                "query_params": query_params,
                "user_agent": user_agent,
                "client_ip": client_ip,
                "request_type": "incoming",
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            duration_ms = round(duration * 1000, 2)
            
            # Log successful response
            self.logger.info(
                f"Request completed: {method} {path} - {response.status_code}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                    "user_agent": user_agent,
                    "client_ip": client_ip,
                    "request_type": "completed",
                }
            )
            
            # Add request ID to response headers for tracing
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as exc:
            # Calculate duration for failed requests
            duration = time.time() - start_time
            duration_ms = round(duration * 1000, 2)
            
            # Log error
            self.logger.error(
                f"Request failed: {method} {path}",
                extra={
                    "request_id": request_id,
                    "method": method,
                    "path": path,
                    "duration_ms": duration_ms,
                    "user_agent": user_agent,
                    "client_ip": client_ip,
                    "request_type": "failed",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
                exc_info=exc
            )
            
            # Re-raise the exception
            raise exc
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP address from request headers."""
        # Check for forwarded headers first (for reverse proxies)
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()
        
        # Check for real IP header
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct client IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return None


class ErrorLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware specifically for error logging and handling."""
    
    def __init__(self, app, logger_name: str = "errors"):
        super().__init__(app)
        self.logger = get_logger(logger_name)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and handle errors with detailed logging."""
        try:
            response = await call_next(request)
            
            # Log 4xx and 5xx responses as warnings/errors
            if response.status_code >= 400:
                log_level = "warning" if response.status_code < 500 else "error"
                getattr(self.logger, log_level)(
                    f"HTTP error response: {response.status_code}",
                    extra={
                        "request_id": getattr(request.state, "request_id", None),
                        "method": request.method,
                        "path": request.url.path,
                        "status_code": response.status_code,
                        "client_ip": self._get_client_ip(request),
                    }
                )
            
            return response
            
        except Exception as exc:
            # Log unhandled exceptions with full context
            self.logger.error(
                f"Unhandled exception in request processing",
                extra={
                    "request_id": getattr(request.state, "request_id", None),
                    "method": request.method,
                    "path": request.url.path,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "client_ip": self._get_client_ip(request),
                },
                exc_info=exc
            )
            
            # Re-raise to let FastAPI handle the response
            raise exc
    
    def _get_client_ip(self, request: Request) -> Optional[str]:
        """Extract client IP address from request headers."""
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return None