"""Tool registry for managing available tools."""

import asyncio
import time
from typing import Any, Dict, List, Optional
from ..models.context import ToolDefinition
from .base import BaseTool, ToolResult, ToolExecutionError, ToolExecutionStatus
from .context import ToolExecutionContext


class ToolRegistry:
    """Registry for managing and executing tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_definitions: Dict[str, ToolDefinition] = {}
    
    def register_tool(self, tool: BaseTool) -> bool:
        """Register a tool with the registry.
        
        Args:
            tool: Tool instance to register
            
        Returns:
            True if registration was successful, False if tool already exists
        """
        if tool.name in self._tools:
            return False
        
        self._tools[tool.name] = tool
        self._tool_definitions[tool.name] = ToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters_schema,
            required_permissions=tool.required_permissions,
            timeout=tool.timeout
        )
        return True
    
    def unregister_tool(self, tool_name: str) -> bool:
        """Unregister a tool from the registry.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if tool didn't exist
        """
        if tool_name not in self._tools:
            return False
        
        del self._tools[tool_name]
        del self._tool_definitions[tool_name]
        return True
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name.
        
        Args:
            tool_name: Name of the tool to retrieve
            
        Returns:
            Tool instance if found, None otherwise
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """Get a list of all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tool_definition(self, tool_name: str) -> Optional[ToolDefinition]:
        """Get the definition of a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            ToolDefinition if found, None otherwise
        """
        return self._tool_definitions.get(tool_name)
    
    def get_all_tool_definitions(self) -> Dict[str, ToolDefinition]:
        """Get all tool definitions.
        
        Returns:
            Dictionary mapping tool names to their definitions
        """
        return self._tool_definitions.copy()
    
    def get_tools_for_permissions(self, permissions: List[str]) -> List[str]:
        """Get tools that can be used with the given permissions.
        
        Args:
            permissions: List of available permissions
            
        Returns:
            List of tool names that can be used
        """
        available_tools = []
        for tool_name, tool in self._tools.items():
            if not tool.required_permissions or all(perm in permissions for perm in tool.required_permissions):
                available_tools.append(tool_name)
        return available_tools
    
    async def execute_tool(
        self, 
        tool_name: str, 
        parameters: Dict[str, Any],
        context: Optional[ToolExecutionContext] = None
    ) -> ToolResult:
        """Execute a tool with the given parameters and context.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Parameters for tool execution
            context: Execution context with permissions and session info
            
        Returns:
            ToolResult containing execution result and metadata
            
        Raises:
            ToolExecutionError: If tool is not found or execution fails
        """
        tool = self.get_tool(tool_name)
        if not tool:
            raise ToolExecutionError(
                f"Tool '{tool_name}' not found",
                error_code="TOOL_NOT_FOUND",
                details={"tool_name": tool_name, "available_tools": self.list_tools()}
            )
        
        # Check permissions
        if not tool.check_permissions(context):
            return ToolResult(
                status=ToolExecutionStatus.PERMISSION_DENIED,
                error_message=f"Insufficient permissions to execute tool '{tool_name}'",
                execution_time=0.0,
                metadata={
                    "required_permissions": tool.required_permissions,
                    "user_permissions": context.user_permissions if context else []
                }
            )
        
        # Validate parameters
        try:
            validated_parameters = tool.validate_parameters(parameters)
        except ToolExecutionError as e:
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                error_message=f"Parameter validation failed: {e.message}",
                execution_time=0.0,
                metadata={"validation_error": e.details}
            )
        
        # Execute tool with timeout
        start_time = time.time()
        timeout = context.max_execution_time if context else tool.timeout
        
        try:
            result = await asyncio.wait_for(
                tool.execute(validated_parameters, context),
                timeout=timeout
            )
            return result
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            return ToolResult(
                status=ToolExecutionStatus.TIMEOUT,
                error_message=f"Tool execution timed out after {timeout} seconds",
                execution_time=execution_time,
                metadata={"timeout": timeout}
            )
        except ToolExecutionError as e:
            execution_time = time.time() - start_time
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                error_message=e.message,
                execution_time=execution_time,
                metadata={"error_code": e.error_code, "error_details": e.details}
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                error_message=f"Unexpected error during tool execution: {str(e)}",
                execution_time=execution_time,
                metadata={"exception_type": type(e).__name__}
            )
    
    async def validate_tool_permissions(
        self, 
        tool_name: str, 
        context: Optional[ToolExecutionContext] = None
    ) -> bool:
        """Validate that the current context has permission to use the tool.
        
        Args:
            tool_name: Name of the tool to check
            context: Execution context to validate against
            
        Returns:
            True if permissions are valid, False otherwise
        """
        tool = self.get_tool(tool_name)
        if not tool:
            return False
        
        return tool.check_permissions(context)
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool registry.
        
        Returns:
            Dictionary containing registry statistics
        """
        return {
            "total_tools": len(self._tools),
            "tool_names": list(self._tools.keys()),
            "tools_by_permissions": {
                tool_name: tool.required_permissions 
                for tool_name, tool in self._tools.items()
            }
        }