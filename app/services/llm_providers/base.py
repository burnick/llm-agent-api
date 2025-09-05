"""Base LLM provider implementation with common functionality."""

import asyncio
import logging
from abc import ABC
from typing import Any, AsyncIterator, Dict, List, Optional
from datetime import datetime

from app.models.interfaces import ILLMProvider
from app.models.responses import LLMResponse, TokenUsage
from app.models.context import ModelInfo
from app.models.errors import LLMProviderError, ValidationError


logger = logging.getLogger(__name__)


class BaseLLMProvider(ILLMProvider, ABC):
    """Base implementation for LLM providers with common functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration."""
        self.config = config
        self._validate_config()
        
    def _validate_config(self) -> None:
        """Validate provider configuration. Override in subclasses."""
        required_fields = self._get_required_config_fields()
        missing_fields = [field for field in required_fields if field not in self.config]
        
        if missing_fields:
            raise ValidationError(
                f"Missing required configuration fields for {self.provider_name}: {missing_fields}"
            )
    
    def _get_required_config_fields(self) -> List[str]:
        """Get list of required configuration fields. Override in subclasses."""
        return []
    
    async def _handle_rate_limit(self, retry_count: int = 0, max_retries: int = 3) -> None:
        """Handle rate limiting with exponential backoff."""
        if retry_count >= max_retries:
            raise LLMProviderError(
                f"Max retries ({max_retries}) exceeded for {self.provider_name}",
                provider=self.provider_name,
                error_type="rate_limit_exceeded"
            )
        
        # Exponential backoff: 1s, 2s, 4s, 8s...
        delay = 2 ** retry_count
        logger.warning(
            f"Rate limited by {self.provider_name}, retrying in {delay}s (attempt {retry_count + 1}/{max_retries})"
        )
        await asyncio.sleep(delay)
    
    def _create_llm_response(
        self,
        content: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        finish_reason: str = "stop",
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """Create a standardized LLM response."""
        return LLMResponse(
            content=content,
            model=model,
            tokens_used=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            ),
            finish_reason=finish_reason,
            metadata=metadata or {}
        )
    
    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Validate message format."""
        if not messages:
            raise ValidationError("Messages list cannot be empty")
        
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValidationError(f"Message {i} must be a dictionary")
            
            if "role" not in message:
                raise ValidationError(f"Message {i} missing 'role' field")
            
            if "content" not in message:
                raise ValidationError(f"Message {i} missing 'content' field")
            
            if message["role"] not in ["system", "user", "assistant", "tool"]:
                raise ValidationError(f"Message {i} has invalid role: {message['role']}")
    
    def _count_tokens_estimate(self, text: str) -> int:
        """Rough token count estimation. Override with provider-specific logic."""
        # Very rough approximation: ~4 characters per token
        return len(text) // 4
    
    async def _log_request(
        self, 
        messages: List[Dict[str, str]], 
        model: str, 
        **kwargs
    ) -> None:
        """Log request details for monitoring."""
        logger.info(
            f"LLM request to {self.provider_name}",
            extra={
                "provider": self.provider_name,
                "model": model,
                "message_count": len(messages),
                "estimated_tokens": sum(self._count_tokens_estimate(msg.get("content", "")) for msg in messages),
                "parameters": {k: v for k, v in kwargs.items() if k not in ["api_key", "authorization"]}
            }
        )
    
    async def _log_response(
        self, 
        response: LLMResponse, 
        duration: float
    ) -> None:
        """Log response details for monitoring."""
        logger.info(
            f"LLM response from {self.provider_name}",
            extra={
                "provider": self.provider_name,
                "model": response.model,
                "tokens_used": response.tokens_used.total_tokens,
                "duration": duration,
                "finish_reason": response.finish_reason
            }
        )
    
    async def _log_error(
        self, 
        error: Exception, 
        context: Dict[str, Any]
    ) -> None:
        """Log error details for debugging."""
        logger.error(
            f"LLM provider error in {self.provider_name}",
            extra={
                "provider": self.provider_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context
            },
            exc_info=True
        )