"""File operations tool with security constraints."""

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional
from ..base import BaseTool, ToolResult, ToolExecutionError, ToolExecutionStatus
from ..context import ToolExecutionContext


class FileOperationsTool(BaseTool):
    """Tool for safe file operations with security constraints."""
    
    def __init__(self, allowed_directories: Optional[list] = None):
        """Initialize the file operations tool.
        
        Args:
            allowed_directories: List of directories where file operations are allowed
        """
        self.allowed_directories = allowed_directories or ["/tmp", "./workspace"]
        
        parameters_schema = {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "exists", "size", "create_dir"],
                    "description": "File operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for write operation)"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8"
                },
                "max_size": {
                    "type": "integer",
                    "description": "Maximum file size to read in bytes (default: 1MB)",
                    "default": 1048576
                }
            },
            "required": ["operation", "path"]
        }
        
        super().__init__(
            name="file_operations",
            description="Perform safe file operations including read, write, list, exists, size, and create_dir. Operations are restricted to allowed directories for security.",
            parameters_schema=parameters_schema,
            timeout=30.0,
            required_permissions=["file_access"]
        )
    
    def _is_path_allowed(self, path: str) -> bool:
        """Check if a path is within allowed directories.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is allowed, False otherwise
        """
        try:
            abs_path = os.path.abspath(path)
            for allowed_dir in self.allowed_directories:
                allowed_abs = os.path.abspath(allowed_dir)
                if abs_path.startswith(allowed_abs):
                    return True
            return False
        except (OSError, ValueError):
            return False
    
    def _check_file_size(self, path: str, max_size: int) -> bool:
        """Check if file size is within limits.
        
        Args:
            path: File path to check
            max_size: Maximum allowed size in bytes
            
        Returns:
            True if file size is acceptable, False otherwise
        """
        try:
            return os.path.getsize(path) <= max_size
        except OSError:
            return False
    
    async def execute(self, parameters: Dict[str, Any], context: Optional[ToolExecutionContext] = None) -> ToolResult:
        """Execute the file operations tool.
        
        Args:
            parameters: Tool parameters containing operation details
            context: Execution context for permission and size checking
            
        Returns:
            ToolResult with operation result
        """
        start_time = time.time()
        
        try:
            operation = parameters["operation"]
            path = parameters["path"]
            content = parameters.get("content", "")
            encoding = parameters.get("encoding", "utf-8")
            max_size = parameters.get("max_size", 1048576)  # 1MB default
            
            # Check file operations permission
            if context and not context.allow_file_operations:
                return ToolResult(
                    status=ToolExecutionStatus.PERMISSION_DENIED,
                    error_message="File operations are not allowed in this context",
                    execution_time=time.time() - start_time,
                    metadata={"required_permission": "file_operations"}
                )
            
            # Override max_size from context if specified
            if context and context.max_file_size:
                max_size = min(max_size, context.max_file_size)
            
            # Check if path is allowed
            if not self._is_path_allowed(path):
                return ToolResult(
                    status=ToolExecutionStatus.PERMISSION_DENIED,
                    error_message=f"Path '{path}' is not in allowed directories",
                    execution_time=time.time() - start_time,
                    metadata={
                        "path": path,
                        "allowed_directories": self.allowed_directories
                    }
                )
            
            # Execute the requested operation
            result_data = {}
            
            if operation == "read":
                if not os.path.exists(path):
                    raise ToolExecutionError(
                        f"File does not exist: {path}",
                        error_code="FILE_NOT_FOUND"
                    )
                
                if not os.path.isfile(path):
                    raise ToolExecutionError(
                        f"Path is not a file: {path}",
                        error_code="NOT_A_FILE"
                    )
                
                if not self._check_file_size(path, max_size):
                    raise ToolExecutionError(
                        f"File too large (max {max_size} bytes): {path}",
                        error_code="FILE_TOO_LARGE"
                    )
                
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    result_data = {
                        "operation": "read",
                        "path": path,
                        "content": content,
                        "size": len(content.encode(encoding)),
                        "encoding": encoding
                    }
                except UnicodeDecodeError:
                    raise ToolExecutionError(
                        f"Cannot decode file with encoding {encoding}: {path}",
                        error_code="ENCODING_ERROR"
                    )
            
            elif operation == "write":
                if not content and operation == "write":
                    raise ToolExecutionError(
                        "Content is required for write operation",
                        error_code="MISSING_CONTENT"
                    )
                
                content_size = len(content.encode(encoding))
                if content_size > max_size:
                    raise ToolExecutionError(
                        f"Content too large (max {max_size} bytes): {content_size} bytes",
                        error_code="CONTENT_TOO_LARGE"
                    )
                
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(path), exist_ok=True)
                
                with open(path, 'w', encoding=encoding) as f:
                    f.write(content)
                
                result_data = {
                    "operation": "write",
                    "path": path,
                    "bytes_written": content_size,
                    "encoding": encoding
                }
            
            elif operation == "list":
                if not os.path.exists(path):
                    raise ToolExecutionError(
                        f"Directory does not exist: {path}",
                        error_code="DIRECTORY_NOT_FOUND"
                    )
                
                if not os.path.isdir(path):
                    raise ToolExecutionError(
                        f"Path is not a directory: {path}",
                        error_code="NOT_A_DIRECTORY"
                    )
                
                items = []
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    items.append({
                        "name": item,
                        "type": "directory" if os.path.isdir(item_path) else "file",
                        "size": os.path.getsize(item_path) if os.path.isfile(item_path) else None
                    })
                
                result_data = {
                    "operation": "list",
                    "path": path,
                    "items": items,
                    "count": len(items)
                }
            
            elif operation == "exists":
                result_data = {
                    "operation": "exists",
                    "path": path,
                    "exists": os.path.exists(path),
                    "type": "directory" if os.path.isdir(path) else "file" if os.path.isfile(path) else "other"
                }
            
            elif operation == "size":
                if not os.path.exists(path):
                    raise ToolExecutionError(
                        f"Path does not exist: {path}",
                        error_code="PATH_NOT_FOUND"
                    )
                
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                elif os.path.isdir(path):
                    size = sum(
                        os.path.getsize(os.path.join(dirpath, filename))
                        for dirpath, dirnames, filenames in os.walk(path)
                        for filename in filenames
                    )
                else:
                    size = 0
                
                result_data = {
                    "operation": "size",
                    "path": path,
                    "size": size,
                    "type": "directory" if os.path.isdir(path) else "file"
                }
            
            elif operation == "create_dir":
                os.makedirs(path, exist_ok=True)
                result_data = {
                    "operation": "create_dir",
                    "path": path,
                    "created": True
                }
            
            else:
                raise ToolExecutionError(
                    f"Unsupported operation: {operation}",
                    error_code="UNSUPPORTED_OPERATION"
                )
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                status=ToolExecutionStatus.SUCCESS,
                result=result_data,
                execution_time=execution_time,
                metadata={
                    "tool_version": "1.0",
                    "allowed_directories": self.allowed_directories
                }
            )
            
        except ToolExecutionError:
            # Re-raise tool execution errors
            raise
        except (OSError, IOError) as e:
            execution_time = time.time() - start_time
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                error_message=f"File system error: {str(e)}",
                execution_time=execution_time,
                metadata={"error_type": type(e).__name__}
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                status=ToolExecutionStatus.ERROR,
                error_message=f"Unexpected error: {str(e)}",
                execution_time=execution_time,
                metadata={"exception_type": type(e).__name__}
            )