"""Tool execution context for managing permissions and session data."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolExecutionContext(BaseModel):
    """Context for tool execution containing session info and permissions."""
    
    session_id: str = Field(..., description="Session identifier")
    user_id: Optional[str] = Field(default=None, description="User identifier if available")
    user_permissions: List[str] = Field(default_factory=list, description="Permissions granted to the user")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional execution metadata")
    max_execution_time: float = Field(default=300.0, description="Maximum allowed execution time in seconds")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the context was created")
    
    # Security constraints
    allow_file_operations: bool = Field(default=False, description="Whether file operations are allowed")
    allow_network_access: bool = Field(default=True, description="Whether network access is allowed")
    allowed_domains: List[str] = Field(default_factory=list, description="Domains allowed for network access")
    max_file_size: int = Field(default=10_000_000, description="Maximum file size in bytes for file operations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session_123",
                "user_id": "user_456",
                "user_permissions": ["web_access", "calculator"],
                "execution_metadata": {"agent_type": "default"},
                "max_execution_time": 300.0,
                "created_at": "2024-01-01T12:00:00Z",
                "allow_file_operations": False,
                "allow_network_access": True,
                "allowed_domains": ["api.example.com"],
                "max_file_size": 10000000
            }
        }
    
    def has_permission(self, permission: str) -> bool:
        """Check if the context has a specific permission.
        
        Args:
            permission: Permission to check
            
        Returns:
            True if permission is granted, False otherwise
        """
        return permission in self.user_permissions
    
    def has_all_permissions(self, permissions: List[str]) -> bool:
        """Check if the context has all specified permissions.
        
        Args:
            permissions: List of permissions to check
            
        Returns:
            True if all permissions are granted, False otherwise
        """
        return all(perm in self.user_permissions for perm in permissions)
    
    def is_domain_allowed(self, domain: str) -> bool:
        """Check if a domain is allowed for network access.
        
        Args:
            domain: Domain to check
            
        Returns:
            True if domain is allowed, False otherwise
        """
        if not self.allow_network_access:
            return False
        
        if not self.allowed_domains:
            return True  # No restrictions if list is empty
        
        return domain in self.allowed_domains or any(
            domain.endswith(f".{allowed}") for allowed in self.allowed_domains
        )