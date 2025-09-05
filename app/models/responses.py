"""Response models for the LLM Agent API."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage information."""
    
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the completion")
    total_tokens: int = Field(..., description="Total number of tokens used")


class ExecutionStep(BaseModel):
    """Individual step in agent execution."""
    
    step_type: str = Field(..., description="Type of execution step (tool_call, reasoning, etc.)")
    description: str = Field(..., description="Description of what happened in this step")
    input_data: Optional[Dict[str, Any]] = Field(default=None, description="Input data for this step")
    output_data: Optional[Dict[str, Any]] = Field(default=None, description="Output data from this step")
    duration: float = Field(..., description="Duration of this step in seconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When this step occurred")


class AgentResponse(BaseModel):
    """Response model for agent execution."""
    
    response: str = Field(..., description="The agent's response message")
    execution_steps: List[ExecutionStep] = Field(default_factory=list, description="Steps taken during execution")
    tokens_used: TokenUsage = Field(..., description="Token usage information")
    execution_time: float = Field(..., description="Total execution time in seconds")
    session_id: Optional[str] = Field(default=None, description="Session ID for this interaction")
    agent_type: str = Field(..., description="Type of agent that processed the request")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "The weather in San Francisco is currently 72°F and sunny.",
                "execution_steps": [
                    {
                        "step_type": "tool_call",
                        "description": "Called web search tool",
                        "input_data": {"query": "San Francisco weather"},
                        "output_data": {"result": "72°F sunny"},
                        "duration": 1.2,
                        "timestamp": "2024-01-01T12:00:00Z"
                    }
                ],
                "tokens_used": {
                    "prompt_tokens": 50,
                    "completion_tokens": 25,
                    "total_tokens": 75
                },
                "execution_time": 2.5,
                "session_id": "session_123",
                "agent_type": "default"
            }
        }


class LLMResponse(BaseModel):
    """Response model for direct LLM interaction."""
    
    content: str = Field(..., description="The generated content")
    model: str = Field(..., description="Model used for generation")
    tokens_used: TokenUsage = Field(..., description="Token usage information")
    finish_reason: str = Field(..., description="Reason why generation finished")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Quantum computing uses quantum mechanics principles...",
                "model": "gpt-4",
                "tokens_used": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120
                },
                "finish_reason": "stop",
                "metadata": {"temperature": 0.7}
            }
        }


class AgentCapability(BaseModel):
    """Information about an agent's capabilities."""
    
    name: str = Field(..., description="Name of the capability")
    description: str = Field(..., description="Description of what this capability does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters this capability accepts")


class AgentInfo(BaseModel):
    """Information about an available agent."""
    
    agent_type: str = Field(..., description="Type identifier for the agent")
    name: str = Field(..., description="Human-readable name of the agent")
    description: str = Field(..., description="Description of the agent's purpose")
    capabilities: List[AgentCapability] = Field(default_factory=list, description="List of agent capabilities")
    available_tools: List[str] = Field(default_factory=list, description="Tools available to this agent")


class HealthStatus(BaseModel):
    """Health status information."""
    
    status: str = Field(..., description="Overall health status (healthy, unhealthy, degraded)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the health check was performed")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="Application uptime in seconds")
    dependencies: Optional[Dict[str, Dict[str, Any]]] = Field(default=None, description="Dependency health information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-01-01T12:00:00Z",
                "version": "1.0.0",
                "uptime": 3600.0,
                "dependencies": {
                    "openai": {"status": "healthy", "response_time": 0.5},
                    "database": {"status": "healthy", "response_time": 0.1}
                }
            }
        }