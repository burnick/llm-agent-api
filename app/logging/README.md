# Logging System Documentation

This module provides a comprehensive logging system for the LLM Agent API with structured logging, middleware for request/response logging, and utilities for different environments.

## Features

- **Structured Logging**: Human-readable logs in development, JSON logs in production
- **Request/Response Middleware**: Automatic logging of HTTP requests with unique request IDs
- **Error Handling**: Detailed error logging with context and stack traces
- **Performance Monitoring**: Built-in performance logging and metrics
- **Environment-Specific Configuration**: Different logging configurations for dev/staging/production
- **Audit Logging**: Specialized logging for security and audit events

## Quick Start

### Basic Setup

```python
from app.logging import setup_logging, get_logger
from app.config.models import Environment, LogLevel

# Setup logging (usually done in main.py)
setup_logging(Environment.DEVELOPMENT, LogLevel.INFO)

# Get a logger
logger = get_logger("my_module")

# Log messages
logger.info("Application started")
logger.error("Something went wrong", extra={"user_id": "123"})
```

### Using with FastAPI

```python
from fastapi import FastAPI
from app.logging import LoggingMiddleware, ErrorLoggingMiddleware

app = FastAPI()

# Add logging middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(ErrorLoggingMiddleware)
```

## Components

### 1. Configuration (`config.py`)

- `setup_logging()`: Main function to configure logging
- `get_logger()`: Get logger instances with consistent naming
- `LoggerMixin`: Mixin class for adding logging to any class

### 2. Formatters (`formatters.py`)

- `StructuredFormatter`: Human-readable structured logs for development
- `JSONFormatter`: JSON-formatted logs for production
- `RequestFormatter`: Specialized formatter for HTTP requests

### 3. Middleware (`middleware.py`)

- `LoggingMiddleware`: Logs all HTTP requests/responses with timing
- `ErrorLoggingMiddleware`: Specialized error handling and logging

### 4. Utilities (`utils.py`)

- `@log_function_call`: Decorator for automatic function logging
- `LogContext`: Context manager for adding structured context
- `AuditLogger`: Specialized logger for audit and security events

## Usage Examples

### Structured Logging

```python
logger = get_logger("user_service")

# Basic logging
logger.info("User logged in")

# Structured logging with context
logger.info(
    "User action performed",
    extra={
        "user_id": "user123",
        "action": "create_document",
        "duration_ms": 245.7,
        "success": True
    }
)
```

### Class-based Logging

```python
from app.logging import LoggerMixin

class UserService(LoggerMixin):
    def create_user(self, user_data):
        self.logger.info("Creating new user", extra={"email": user_data.email})
        # ... implementation
        self.logger.info("User created successfully", extra={"user_id": user.id})
```

### Function Decorators

```python
from app.logging import log_function_call

@log_function_call(log_performance=True)
async def process_llm_request(prompt: str):
    # Function implementation
    return response
```

### Audit Logging

```python
from app.logging import create_audit_logger

audit = create_audit_logger("user_management")

# Log user actions
audit.log_user_action(
    action="login",
    user_id="user123",
    result="success",
    ip_address="192.168.1.1"
)

# Log security events
audit.log_security_event(
    event="failed_login_attempt",
    threat_level="medium",
    source_ip="192.168.1.100",
    attempts=3
)
```

### Error Logging

```python
from app.logging import log_exception

logger = get_logger("service")

try:
    # Some operation
    pass
except Exception as e:
    log_exception(
        logger,
        "Failed to process request",
        exc_info=e,
        extra={"user_id": "123", "operation": "create_document"}
    )
```

## Environment Configuration

### Development
- Human-readable structured logs
- Console output only
- Debug information included

### Production
- JSON-formatted logs
- File rotation (10MB files, 5 backups)
- Structured for log aggregation systems

### Configuration Options

```python
# Environment variables
LOG_LEVEL=INFO
ENABLE_JSON_LOGGING=false  # Override JSON logging in production

# Programmatic configuration
setup_logging(
    environment=Environment.PRODUCTION,
    log_level=LogLevel.INFO,
    enable_json=True  # Force JSON logging
)
```

## Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about application flow
- **WARNING**: Something unexpected happened but application continues
- **ERROR**: Serious problem occurred
- **CRITICAL**: Very serious error occurred

## Request Logging

The middleware automatically logs:
- Request method and path
- Response status code
- Request duration
- Client IP address
- User agent
- Unique request ID (added to response headers)

Example request log:
```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "INFO",
  "logger": "app.middleware",
  "message": "Request completed: GET /api/users - 200",
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "method": "GET",
  "path": "/api/users",
  "status_code": 200,
  "duration_ms": 45.67,
  "client_ip": "192.168.1.1",
  "user_agent": "Mozilla/5.0..."
}
```

## Best Practices

1. **Use structured logging**: Always include relevant context in `extra` fields
2. **Don't log sensitive data**: Sanitize passwords, API keys, etc.
3. **Use appropriate log levels**: Don't use ERROR for expected conditions
4. **Include request IDs**: Use `request.state.request_id` in endpoints
5. **Log performance metrics**: Include timing information for operations
6. **Use audit logging**: For security-sensitive operations

## Integration with Monitoring

The JSON logs are designed to work with log aggregation systems like:
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Splunk
- DataDog
- CloudWatch Logs

Key fields for monitoring:
- `request_id`: Trace requests across services
- `duration_ms`: Performance monitoring
- `status_code`: Error rate monitoring
- `user_id`: User-specific analysis
- `component`: Service/module identification