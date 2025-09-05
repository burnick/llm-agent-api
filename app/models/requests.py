"""Request models for the LLM Agent API."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    """Request model for agent execution."""
    
    message: str = Field(..., description="The message to send to the agent")
    agent_type: str = Field(default="default", description="Type of agent to use")
    tools: Optional[List[str]] = Field(default=None, description="List of tools to make available to the agent")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context for the agent")
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation continuity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What's the weather like today?",
                "agent_type": "default",
                "tools": ["web_search", "calculator"],
                "context": {"location": "San Francisco"},
                "session_id": "session_123"
            }
        }


class LLMRequest(BaseModel):
    """Request model for direct LLM interaction."""
    
    prompt: str = Field(..., description="The prompt to send to the LLM")
    model: Optional[str] = Field(default=None, description="Specific model to use")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for response generation")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="Maximum tokens in response")
    stream: bool = Field(default=False, description="Whether to stream the response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Explain quantum computing in simple terms",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False
            }
        }


class HealthCheckRequest(BaseModel):
    """Request model for health check with optional detailed checks."""
    
    include_dependencies: bool = Field(default=False, description="Include dependency health checks")
    timeout: Optional[float] = Field(default=5.0, gt=0, description="Timeout for health checks in seconds")