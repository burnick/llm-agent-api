"""Tool service implementation."""

from typing import Any, Dict, List, Optional
from ..models.interfaces import IToolService
from ..models.context import ExecutionContext, ToolDefinition
from ..tools import ToolRegistry, ToolExecutionContext, ToolFactory


class ToolService(IToolService):
    """Service for managing and executing tools."""
    
    def __init__(self, tool_registry: Optional[ToolRegistry] = None):
        """Initialize the tool service.
        
        Args:
            tool_registry: Tool registry to use, creates default if None
        """
        self.registry = tool_registry or ToolFactory.create_default_registry()
    
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        context: Optional[ExecutionContext] = None
    ) -> Dict[str, Any]:
        """Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for tool execution
            context: Execution context
            
        Returns:
            Dictionary containing execution result
        """
        # Convert ExecutionContext to ToolExecutionContext
        tool_context = None
        if context:
            tool_context = ToolExecutionContext(
                session_id=context.session_id,
                user_id=context.user_id,
                user_permissions=context.available_tools,  # Use available tools as permissions
                execution_metadata=context.execution_metadata,
                allow_file_operations=True,  # Default to allowing file ops
                allow_network_access=True,   # Default to allowing network access
            )
        
        # Execute the tool
        result = await self.registry.execute_tool(tool_name, parameters, tool_context)
        
        # Convert ToolResult to dictionary
        return {
            "status": result.status.value,
            "result": result.result,
            "error_message": result.error_message,
            "execution_time": result.execution_time,
            "metadata": result.metadata,
            "timestamp": result.timestamp.isoformat()
        }
    
    async def list_available_tools(self) -> List[str]:
        """Get a list of available tool names.
        
        Returns:
            List of tool names
        """
        return self.registry.list_tools()
    
    async def get_tool_definition(self, tool_name: str) -> ToolDefinition:
        """Get the definition of a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolDefinition for the tool
            
        Raises:
            ValueError: If tool is not found
        """
        definition = self.registry.get_tool_definition(tool_name)
        if not definition:
            raise ValueError(f"Tool '{tool_name}' not found")
        return definition
    
    async def register_tool(self, tool_definition: ToolDefinition) -> bool:
        """Register a new tool with the service.
        
        Args:
            tool_definition: Definition of the tool to register
            
        Returns:
            True if registration was successful, False otherwise
            
        Note:
            This method currently doesn't support registering tools from definitions.
            Tools must be registered directly with the registry using BaseTool instances.
        """
        # This would require creating a tool instance from the definition
        # For now, return False as this is not implemented
        return False
    
    async def validate_tool_permissions(
        self, 
        tool_name: str, 
        context: Optional[ExecutionContext] = None
    ) -> bool:
        """Validate that the current context has permission to use the tool.
        
        Args:
            tool_name: Name of the tool to check
            context: Execution context to validate against
            
        Returns:
            True if permissions are valid, False otherwise
        """
        # Convert ExecutionContext to ToolExecutionContext
        tool_context = None
        if context:
            tool_context = ToolExecutionContext(
                session_id=context.session_id,
                user_id=context.user_id,
                user_permissions=context.available_tools,
                execution_metadata=context.execution_metadata
            )
        
        return await self.registry.validate_tool_permissions(tool_name, tool_context)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry.
        
        Returns:
            Dictionary containing registry statistics
        """
        return self.registry.get_registry_stats()