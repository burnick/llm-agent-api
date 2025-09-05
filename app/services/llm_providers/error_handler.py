"""Error handling utilities for LLM providers."""

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Type, TypeVar
from functools import wraps
from datetime import datetime, timedelta

from app.models.errors import LLMProviderError, ValidationError


logger = logging.getLogger(__name__)

T = TypeVar('T')


class ExponentialBackoff:
    """Exponential backoff utility for retrying operations."""
    
    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
    
    def get_delay(self, attempt: int) -> float:
        """Get delay for the given attempt number."""
        delay = self.initial_delay * (self.multiplier ** attempt)
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            # Add jitter to prevent thundering herd
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
    
    async def sleep(self, attempt: int) -> None:
        """Sleep for the calculated delay."""
        delay = self.get_delay(attempt)
        logger.debug(f"Backing off for {delay:.2f}s (attempt {attempt})")
        await asyncio.sleep(delay)


class CircuitBreaker:
    """Circuit breaker pattern implementation for provider calls."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if execution is allowed."""
        if self.state == "closed":
            return True
        
        if self.state == "open":
            if self.last_failure_time:
                time_since_failure = datetime.utcnow() - self.last_failure_time
                if time_since_failure.total_seconds() >= self.recovery_timeout:
                    self.state = "half-open"
                    logger.info("Circuit breaker moved to half-open state")
                    return True
            return False
        
        # half-open state
        return True
    
    def record_success(self) -> None:
        """Record a successful execution."""
        self.failure_count = 0
        self.last_failure_time = None
        if self.state != "closed":
            self.state = "closed"
            logger.info("Circuit breaker closed after successful execution")
    
    def record_failure(self, exception: Exception) -> None:
        """Record a failed execution."""
        if isinstance(exception, self.expected_exception):
            self.failure_count += 1
            self.last_failure_time = datetime.utcnow()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


def with_retry(
    max_retries: int = 3,
    backoff: Optional[ExponentialBackoff] = None,
    retry_on: tuple = (Exception,),
    dont_retry_on: tuple = (ValidationError,)
):
    """Decorator for adding retry logic to async functions."""
    if backoff is None:
        backoff = ExponentialBackoff()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                    
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on certain exceptions
                    if any(isinstance(e, exc_type) for exc_type in dont_retry_on):
                        logger.debug(f"Not retrying {func.__name__} due to {type(e).__name__}")
                        raise
                    
                    # Only retry on specified exceptions
                    if not any(isinstance(e, exc_type) for exc_type in retry_on):
                        logger.debug(f"Not retrying {func.__name__} due to unexpected exception {type(e).__name__}")
                        raise
                    
                    if attempt < max_retries:
                        logger.warning(f"Function {func.__name__} failed on attempt {attempt + 1}: {e}")
                        await backoff.sleep(attempt)
                    else:
                        logger.error(f"Function {func.__name__} failed after {max_retries + 1} attempts")
            
            # If we get here, all retries failed
            if last_exception:
                raise last_exception
            else:
                raise LLMProviderError("All retries failed with no exception recorded")
        
        return wrapper
    return decorator


def with_circuit_breaker(circuit_breaker: CircuitBreaker):
    """Decorator for adding circuit breaker pattern to async functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            if not circuit_breaker.can_execute():
                raise LLMProviderError(
                    f"Circuit breaker is open for {func.__name__}",
                    error_type="circuit_breaker_open"
                )
            
            try:
                result = await func(*args, **kwargs)
                circuit_breaker.record_success()
                return result
                
            except Exception as e:
                circuit_breaker.record_failure(e)
                raise
        
        return wrapper
    return decorator


def with_timeout(timeout_seconds: float):
    """Decorator for adding timeout to async functions."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            try:
                return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                raise LLMProviderError(
                    f"Function {func.__name__} timed out after {timeout_seconds}s",
                    error_type="timeout"
                )
        
        return wrapper
    return decorator


class ErrorContext:
    """Context manager for error handling and logging."""
    
    def __init__(
        self,
        operation: str,
        provider: str,
        context: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.provider = provider
        self.context = context or {}
        self.start_time: Optional[datetime] = None
    
    async def __aenter__(self):
        self.start_time = datetime.utcnow()
        logger.debug(f"Starting {self.operation} for provider {self.provider}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.utcnow() - self.start_time).total_seconds() if self.start_time else 0
        
        if exc_type is None:
            logger.debug(f"Completed {self.operation} for provider {self.provider} in {duration:.2f}s")
        else:
            logger.error(
                f"Failed {self.operation} for provider {self.provider} after {duration:.2f}s: {exc_val}",
                extra={
                    "operation": self.operation,
                    "provider": self.provider,
                    "duration": duration,
                    "error_type": exc_type.__name__,
                    "context": self.context
                }
            )
        
        return False  # Don't suppress exceptions


def handle_provider_errors(provider_name: str):
    """Decorator for standardizing provider error handling."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            async with ErrorContext(func.__name__, provider_name):
                try:
                    return await func(*args, **kwargs)
                except LLMProviderError:
                    # Re-raise our own errors as-is
                    raise
                except ValidationError:
                    # Re-raise validation errors as-is
                    raise
                except Exception as e:
                    # Wrap unexpected errors
                    raise LLMProviderError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        provider=provider_name,
                        error_type="unexpected"
                    ) from e
        
        return wrapper
    return decorator