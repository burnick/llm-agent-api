"""Core data structures for agent execution context."""

from datetime import datetime
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    """Roles for conversation messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """Individual message in a conversation."""
    
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Content of the message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the message was created")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional message metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What's the weather like?",
                "timestamp": "2024-01-01T12:00:00Z",
                "metadata": {"source": "web_interface"}
            }
        }


class ToolDefinition(BaseModel):
    """Definition of a tool that can be used by agents."""
    
    name: str = Field(..., description="Unique name of the tool")
    description: str = Field(..., description="Description of what the tool does")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="JSON schema for tool parameters")
    required_permissions: List[str] = Field(default_factory=list, description="Permissions required to use this tool")
    timeout: Optional[float] = Field(default=30.0, description="Timeout for tool execution in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                },
                "required_permissions": ["web_access"],
                "timeout": 30.0
            }
        }


class ExecutionContext(BaseModel):
    """Context for agent execution including conversation history and available tools."""
    
    session_id: str = Field(..., description="Unique session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier if available")
    conversation_history: List[Message] = Field(default_factory=list, description="History of messages in this session")
    available_tools: List[str] = Field(default_factory=list, description="Names of tools available to the agent")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional execution metadata")
    max_history_length: int = Field(default=50, description="Maximum number of messages to keep in history")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the context was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the context was last updated")
    
    def add_message(self, message: Message) -> None:
        """Add a message to the conversation history."""
        self.conversation_history.append(message)
        self.updated_at = datetime.utcnow()
        
        # Trim history if it exceeds max length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def get_recent_messages(self, count: int = 10) -> List[Message]:
        """Get the most recent messages from the conversation history."""
        return self.conversation_history[-count:] if count > 0 else self.conversation_history
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "user_id": "user_456",
                "conversation_history": [
                    {
                        "role": "user",
                        "content": "Hello",
                        "timestamp": "2024-01-01T12:00:00Z",
                        "metadata": {}
                    }
                ],
                "available_tools": ["web_search", "calculator"],
                "execution_metadata": {"agent_type": "default"},
                "max_history_length": 50,
                "created_at": "2024-01-01T12:00:00Z",
                "updated_at": "2024-01-01T12:00:00Z"
            }
        }


class ModelInfo(BaseModel):
    """Information about an LLM model."""
    
    name: str = Field(..., description="Model name")
    provider: str = Field(..., description="Provider of the model")
    max_tokens: int = Field(..., description="Maximum tokens the model can handle")
    supports_streaming: bool = Field(default=False, description="Whether the model supports streaming")
    supports_tools: bool = Field(default=False, description="Whether the model supports tool calling")
    cost_per_token: Optional[float] = Field(default=None, description="Cost per token if available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "gpt-4",
                "provider": "openai",
                "max_tokens": 8192,
                "supports_streaming": True,
                "supports_tools": True,
                "cost_per_token": 0.00003
            }
        }