"""Base tool interface and execution structures."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ToolExecutionStatus(str, Enum):
    """Status of tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"


class ToolResult(BaseModel):
    """Result of tool execution."""
    
    status: ToolExecutionStatus = Field(..., description="Execution status")
    result: Optional[Any] = Field(default=None, description="Tool execution result")
    error_message: Optional[str] = Field(default=None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Time taken to execute the tool in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional execution metadata")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When the execution completed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "result": {"answer": 42},
                "error_message": None,
                "execution_time": 0.15,
                "metadata": {"tool_version": "1.0"},
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }


class ToolExecutionError(Exception):
    """Exception raised during tool execution."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "TOOL_EXECUTION_ERROR"
        self.details = details or {}


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(self, name: str, description: str, parameters_schema: Dict[str, Any], 
                 timeout: float = 30.0, required_permissions: Optional[list] = None):
        self.name = name
        self.description = description
        self.parameters_schema = parameters_schema
        self.timeout = timeout
        self.required_permissions = required_permissions or []
    
    @abstractmethod
    async def execute(self, parameters: Dict[str, Any], context: Optional['ToolExecutionContext'] = None) -> ToolResult:
        """Execute the tool with given parameters and context.
        
        Args:
            parameters: Tool parameters validated against the schema
            context: Execution context containing session info and permissions
            
        Returns:
            ToolResult containing the execution result and metadata
            
        Raises:
            ToolExecutionError: If execution fails
        """
        pass
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate parameters against the tool's schema.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Validated parameters
            
        Raises:
            ToolExecutionError: If validation fails
        """
        # Basic validation - in a real implementation, you'd use jsonschema
        required_params = self.parameters_schema.get("required", [])
        properties = self.parameters_schema.get("properties", {})
        
        # Check required parameters
        for param in required_params:
            if param not in parameters:
                raise ToolExecutionError(
                    f"Missing required parameter: {param}",
                    error_code="MISSING_PARAMETER",
                    details={"parameter": param, "required": required_params}
                )
        
        # Check parameter types (basic validation)
        validated = {}
        for param_name, param_value in parameters.items():
            if param_name in properties:
                param_schema = properties[param_name]
                param_type = param_schema.get("type")
                
                # Basic type checking
                if param_type == "string" and not isinstance(param_value, str):
                    raise ToolExecutionError(
                        f"Parameter {param_name} must be a string",
                        error_code="INVALID_PARAMETER_TYPE",
                        details={"parameter": param_name, "expected_type": "string", "actual_type": type(param_value).__name__}
                    )
                elif param_type == "number" and not isinstance(param_value, (int, float)):
                    raise ToolExecutionError(
                        f"Parameter {param_name} must be a number",
                        error_code="INVALID_PARAMETER_TYPE",
                        details={"parameter": param_name, "expected_type": "number", "actual_type": type(param_value).__name__}
                    )
                elif param_type == "boolean" and not isinstance(param_value, bool):
                    raise ToolExecutionError(
                        f"Parameter {param_name} must be a boolean",
                        error_code="INVALID_PARAMETER_TYPE",
                        details={"parameter": param_name, "expected_type": "boolean", "actual_type": type(param_value).__name__}
                    )
                
                validated[param_name] = param_value
            else:
                # Allow extra parameters but warn
                validated[param_name] = param_value
        
        return validated
    
    def check_permissions(self, context: Optional['ToolExecutionContext'] = None) -> bool:
        """Check if the current context has required permissions.
        
        Args:
            context: Execution context to check permissions against
            
        Returns:
            True if permissions are satisfied, False otherwise
        """
        if not self.required_permissions:
            return True
        
        if not context or not context.user_permissions:
            return False
        
        return all(perm in context.user_permissions for perm in self.required_permissions)
    
    def get_definition(self) -> Dict[str, Any]:
        """Get the tool definition for registration.
        
        Returns:
            Tool definition dictionary
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters_schema,
            "required_permissions": self.required_permissions,
            "timeout": self.timeout
        }