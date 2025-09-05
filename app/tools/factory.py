"""Factory for creating and configuring tools."""

from typing import Dict, List, Optional
from .registry import ToolRegistry
from .implementations.calculator import CalculatorTool
from .implementations.web_search import WebSearchTool
from .implementations.file_operations import FileOperationsTool


class ToolFactory:
    """Factory for creating and configuring tools."""
    
    @staticmethod
    def create_default_registry(
        web_search_api_key: Optional[str] = None,
        web_search_engine: str = "duckduckgo",
        allowed_file_directories: Optional[List[str]] = None
    ) -> ToolRegistry:
        """Create a tool registry with default tools.
        
        Args:
            web_search_api_key: API key for web search service
            web_search_engine: Search engine to use for web search
            allowed_file_directories: Directories allowed for file operations
            
        Returns:
            Configured ToolRegistry with default tools
        """
        registry = ToolRegistry()
        
        # Register calculator tool
        calculator = CalculatorTool()
        registry.register_tool(calculator)
        
        # Register web search tool
        web_search = WebSearchTool(
            api_key=web_search_api_key,
            search_engine=web_search_engine
        )
        registry.register_tool(web_search)
        
        # Register file operations tool
        file_ops = FileOperationsTool(
            allowed_directories=allowed_file_directories
        )
        registry.register_tool(file_ops)
        
        return registry
    
    @staticmethod
    def create_calculator_tool() -> CalculatorTool:
        """Create a calculator tool instance.
        
        Returns:
            Configured CalculatorTool
        """
        return CalculatorTool()
    
    @staticmethod
    def create_web_search_tool(
        api_key: Optional[str] = None,
        search_engine: str = "duckduckgo"
    ) -> WebSearchTool:
        """Create a web search tool instance.
        
        Args:
            api_key: API key for search service
            search_engine: Search engine to use
            
        Returns:
            Configured WebSearchTool
        """
        return WebSearchTool(api_key=api_key, search_engine=search_engine)
    
    @staticmethod
    def create_file_operations_tool(
        allowed_directories: Optional[List[str]] = None
    ) -> FileOperationsTool:
        """Create a file operations tool instance.
        
        Args:
            allowed_directories: Directories allowed for file operations
            
        Returns:
            Configured FileOperationsTool
        """
        return FileOperationsTool(allowed_directories=allowed_directories)
    
    @staticmethod
    def get_available_tools() -> Dict[str, str]:
        """Get information about available tools.
        
        Returns:
            Dictionary mapping tool names to descriptions
        """
        return {
            "calculator": "Perform mathematical calculations safely",
            "web_search": "Search the web for information",
            "file_operations": "Perform safe file operations with security constraints"
        }