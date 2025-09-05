"""Error models for the LLM Agent API."""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error_code: str = Field(..., description="Specific error code for programmatic handling")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the error occurred")
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "AGENT_EXECUTION_FAILED",
                "message": "Agent execution failed due to tool timeout",
                "details": {
                    "tool_name": "web_search",
                    "timeout_duration": 30.0,
                    "step": "tool_execution"
                },
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456"
            }
        }


class ValidationErrorDetail(BaseModel):
    """Details for validation errors."""
    
    field: str = Field(..., description="Field that failed validation")
    message: str = Field(..., description="Validation error message")
    invalid_value: Optional[Any] = Field(default=None, description="The invalid value that was provided")


class ValidationErrorResponse(ErrorResponse):
    """Validation error response with field-specific details."""
    
    validation_errors: list[ValidationErrorDetail] = Field(..., description="List of validation errors")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error_code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": {"total_errors": 2},
                "timestamp": "2024-01-01T12:00:00Z",
                "request_id": "req_123456",
                "validation_errors": [
                    {
                        "field": "temperature",
                        "message": "Temperature must be between 0.0 and 2.0",
                        "invalid_value": 3.5
                    },
                    {
                        "field": "message",
                        "message": "Message cannot be empty",
                        "invalid_value": ""
                    }
                ]
            }
        }


# Common error codes as constants
class ErrorCodes:
    """Standard error codes used throughout the application."""
    
    # General errors
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NOT_FOUND = "NOT_FOUND"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    
    # LLM Provider errors
    LLM_PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
    LLM_PROVIDER_UNAVAILABLE = "LLM_PROVIDER_UNAVAILABLE"
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    
    # Agent errors
    AGENT_EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    AGENT_TIMEOUT = "AGENT_TIMEOUT"
    AGENT_TOOL_ERROR = "AGENT_TOOL_ERROR"
    
    # Tool errors
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_EXECUTION_FAILED = "TOOL_EXECUTION_FAILED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_PERMISSION_DENIED = "TOOL_PERMISSION_DENIED"
    
    # Configuration errors
    CONFIG_ERROR = "CONFIG_ERROR"
    CONFIG_MISSING = "CONFIG_MISSING"
    CONFIG_INVALID = "CONFIG_INVALID"


# Custom exception classes
class LLMAgentAPIError(Exception):
    """Base exception for LLM Agent API errors."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or ErrorCodes.INTERNAL_SERVER_ERROR
        self.details = details or {}


class ValidationError(LLMAgentAPIError):
    """Exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, invalid_value: Any = None):
        super().__init__(message, ErrorCodes.VALIDATION_ERROR)
        if field:
            self.details["field"] = field
        if invalid_value is not None:
            self.details["invalid_value"] = invalid_value


class ConfigurationError(LLMAgentAPIError):
    """Exception for configuration errors."""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, ErrorCodes.CONFIG_ERROR)
        if config_key:
            self.details["config_key"] = config_key


class LLMProviderError(LLMAgentAPIError):
    """Exception for LLM provider errors."""
    
    def __init__(self, message: str, provider: str = None, error_type: str = None):
        error_code = ErrorCodes.LLM_PROVIDER_ERROR
        if error_type == "rate_limit":
            error_code = ErrorCodes.LLM_RATE_LIMIT
        elif error_type == "timeout":
            error_code = ErrorCodes.LLM_TIMEOUT
        elif error_type == "unavailable":
            error_code = ErrorCodes.LLM_PROVIDER_UNAVAILABLE
        
        super().__init__(message, error_code)
        if provider:
            self.details["provider"] = provider
        if error_type:
            self.details["error_type"] = error_type


class AgentError(LLMAgentAPIError):
    """Exception for agent execution errors."""
    
    def __init__(self, message: str, agent_type: str = None, step: str = None):
        super().__init__(message, ErrorCodes.AGENT_EXECUTION_FAILED)
        if agent_type:
            self.details["agent_type"] = agent_type
        if step:
            self.details["step"] = step


class AgentExecutionError(LLMAgentAPIError):
    """Exception for agent execution errors with execution steps."""
    
    def __init__(self, message: str, agent_type: str = None, execution_steps: list = None):
        super().__init__(message, ErrorCodes.AGENT_EXECUTION_FAILED)
        if agent_type:
            self.details["agent_type"] = agent_type
        if execution_steps:
            self.details["execution_steps"] = execution_steps


class ToolError(LLMAgentAPIError):
    """Exception for tool execution errors."""
    
    def __init__(self, message: str, tool_name: str = None, error_type: str = None):
        error_code = ErrorCodes.TOOL_EXECUTION_FAILED
        if error_type == "timeout":
            error_code = ErrorCodes.TOOL_TIMEOUT
        elif error_type == "permission_denied":
            error_code = ErrorCodes.TOOL_PERMISSION_DENIED
        elif error_type == "not_found":
            error_code = ErrorCodes.TOOL_NOT_FOUND
        
        super().__init__(message, error_code)
        if tool_name:
            self.details["tool_name"] = tool_name
        if error_type:
            self.details["error_type"] = error_type