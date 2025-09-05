"""Models package for the LLM Agent API."""

# Request models
from .requests import (
    AgentRequest,
    LLMRequest,
    HealthCheckRequest,
)

# Response models
from .responses import (
    TokenUsage,
    ExecutionStep,
    AgentResponse,
    LLMResponse,
    AgentCapability,
    AgentInfo,
    HealthStatus,
)

# Error models
from .errors import (
    ErrorResponse,
    ValidationErrorDetail,
    ValidationErrorResponse,
    ErrorCodes,
)

# Context and data structures
from .context import (
    MessageRole,
    Message,
    ToolDefinition,
    ExecutionContext,
    ModelInfo,
)

# Abstract interfaces
from .interfaces import (
    ILLMProvider,
    IAgentService,
    IToolService,
    IMemoryService,
    IHealthService,
)

__all__ = [
    # Request models
    "AgentRequest",
    "LLMRequest", 
    "HealthCheckRequest",
    
    # Response models
    "TokenUsage",
    "ExecutionStep",
    "AgentResponse",
    "LLMResponse",
    "AgentCapability",
    "AgentInfo",
    "HealthStatus",
    
    # Error models
    "ErrorResponse",
    "ValidationErrorDetail",
    "ValidationErrorResponse",
    "ErrorCodes",
    
    # Context and data structures
    "MessageRole",
    "Message",
    "ToolDefinition",
    "ExecutionContext",
    "ModelInfo",
    
    # Abstract interfaces
    "ILLMProvider",
    "IAgentService",
    "IToolService",
    "IMemoryService",
    "IHealthService",
]