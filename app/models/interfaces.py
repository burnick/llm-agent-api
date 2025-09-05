"""Abstract interfaces for LLM providers and agent services."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional
from .requests import AgentRequest, LLMRequest
from .responses import AgentResponse, LLMResponse, AgentInfo, TokenUsage
from .context import ExecutionContext, ModelInfo, ToolDefinition


class ILLMProvider(ABC):
    """Abstract interface for LLM providers."""
    
    @abstractmethod
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion response."""
        pass
    
    @abstractmethod
    async def stream_completion(
        self, 
        messages: List[Dict[str, str]], 
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion response."""
        pass
    
    @abstractmethod
    async def embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings for the given text."""
        pass
    
    @abstractmethod
    def get_model_info(self, model: Optional[str] = None) -> ModelInfo:
        """Get information about the specified model."""
        pass
    
    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validate that the provider connection is working."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the name of this provider."""
        pass
    
    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """Get list of available models for this provider."""
        pass


class IAgentService(ABC):
    """Abstract interface for agent services."""
    
    @abstractmethod
    async def execute_agent(
        self, 
        request: AgentRequest, 
        context: Optional[ExecutionContext] = None
    ) -> AgentResponse:
        """Execute an agent with the given request and context."""
        pass
    
    @abstractmethod
    async def list_available_agents(self) -> List[str]:
        """Get a list of available agent types."""
        pass
    
    @abstractmethod
    async def get_agent_info(self, agent_type: str) -> AgentInfo:
        """Get detailed information about a specific agent type."""
        pass
    
    @abstractmethod
    async def get_agent_capabilities(self, agent_type: str) -> Dict[str, Any]:
        """Get the capabilities of a specific agent type."""
        pass
    
    @abstractmethod
    async def create_execution_context(
        self, 
        session_id: str, 
        user_id: Optional[str] = None
    ) -> ExecutionContext:
        """Create a new execution context for an agent session."""
        pass
    
    @abstractmethod
    async def update_execution_context(self, context: ExecutionContext) -> ExecutionContext:
        """Update an existing execution context."""
        pass


class IToolService(ABC):
    """Abstract interface for tool services."""
    
    @abstractmethod
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        context: Optional[ExecutionContext] = None
    ) -> Dict[str, Any]:
        """Execute a tool with the given parameters."""
        pass
    
    @abstractmethod
    async def list_available_tools(self) -> List[str]:
        """Get a list of available tool names."""
        pass
    
    @abstractmethod
    async def get_tool_definition(self, tool_name: str) -> ToolDefinition:
        """Get the definition of a specific tool."""
        pass
    
    @abstractmethod
    async def register_tool(self, tool_definition: ToolDefinition) -> bool:
        """Register a new tool with the service."""
        pass
    
    @abstractmethod
    async def validate_tool_permissions(
        self, 
        tool_name: str, 
        context: Optional[ExecutionContext] = None
    ) -> bool:
        """Validate that the current context has permission to use the tool."""
        pass


class IMemoryService(ABC):
    """Abstract interface for memory services."""
    
    @abstractmethod
    async def store_context(self, context: ExecutionContext) -> bool:
        """Store an execution context for later retrieval."""
        pass
    
    @abstractmethod
    async def retrieve_context(self, session_id: str) -> Optional[ExecutionContext]:
        """Retrieve a stored execution context by session ID."""
        pass
    
    @abstractmethod
    async def update_context(self, context: ExecutionContext) -> bool:
        """Update a stored execution context."""
        pass
    
    @abstractmethod
    async def delete_context(self, session_id: str) -> bool:
        """Delete a stored execution context."""
        pass
    
    @abstractmethod
    async def cleanup_expired_contexts(self, max_age_hours: int = 24) -> int:
        """Clean up expired contexts and return the number of contexts removed."""
        pass


class IHealthService(ABC):
    """Abstract interface for health check services."""
    
    @abstractmethod
    async def check_health(self, include_dependencies: bool = False) -> Dict[str, Any]:
        """Perform a health check and return status information."""
        pass
    
    @abstractmethod
    async def check_dependency_health(self, dependency_name: str) -> Dict[str, Any]:
        """Check the health of a specific dependency."""
        pass
    
    @abstractmethod
    async def get_system_info(self) -> Dict[str, Any]:
        """Get general system information."""
        pass