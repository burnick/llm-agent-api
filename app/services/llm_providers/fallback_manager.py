"""Fallback manager for handling provider failures and routing."""

import asyncio
import logging
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

from app.models.interfaces import ILLMProvider
from app.models.responses import LLMResponse
from app.models.context import ModelInfo
from app.models.errors import LLMProviderError, ConfigurationError
from .factory import LLMProviderFactory


logger = logging.getLogger(__name__)


class ProviderStatus(str, Enum):
    """Status of a provider."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"


class ProviderHealth:
    """Health tracking for a provider."""
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.status = ProviderStatus.HEALTHY
        self.consecutive_failures = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_success_time: Optional[datetime] = None
        self.total_requests = 0
        self.total_failures = 0
        self.average_response_time = 0.0
        
    def record_success(self, response_time: float) -> None:
        """Record a successful request."""
        self.consecutive_failures = 0
        self.last_success_time = datetime.utcnow()
        self.total_requests += 1
        
        # Update average response time
        if self.total_requests == 1:
            self.average_response_time = response_time
        else:
            self.average_response_time = (
                (self.average_response_time * (self.total_requests - 1) + response_time) 
                / self.total_requests
            )
        
        # Update status based on consecutive failures
        if self.status == ProviderStatus.UNHEALTHY and self.consecutive_failures == 0:
            self.status = ProviderStatus.HEALTHY
            logger.info(f"Provider {self.provider_name} recovered to healthy status")
    
    def record_failure(self, error_type: str) -> None:
        """Record a failed request."""
        self.consecutive_failures += 1
        self.last_failure_time = datetime.utcnow()
        self.total_requests += 1
        self.total_failures += 1
        
        # Update status based on consecutive failures
        if self.consecutive_failures >= 5:
            self.status = ProviderStatus.UNHEALTHY
            logger.warning(f"Provider {self.provider_name} marked as unhealthy after {self.consecutive_failures} failures")
        elif self.consecutive_failures >= 3:
            self.status = ProviderStatus.DEGRADED
            logger.warning(f"Provider {self.provider_name} marked as degraded after {self.consecutive_failures} failures")
    
    def should_circuit_break(self) -> bool:
        """Check if circuit breaker should be triggered."""
        if self.status == ProviderStatus.UNHEALTHY:
            # Check if enough time has passed to try again
            if self.last_failure_time:
                time_since_failure = datetime.utcnow() - self.last_failure_time
                return time_since_failure < timedelta(minutes=5)  # 5-minute circuit breaker
        return False
    
    def get_health_info(self) -> Dict[str, Any]:
        """Get health information as a dictionary."""
        return {
            "provider_name": self.provider_name,
            "status": self.status,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "last_success_time": self.last_success_time.isoformat() if self.last_success_time else None,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "failure_rate": self.total_failures / max(self.total_requests, 1),
            "average_response_time": self.average_response_time
        }


class FallbackManager:
    """Manages fallback logic between multiple LLM providers."""
    
    def __init__(self, provider_configs: List[Dict[str, Any]]):
        """Initialize fallback manager with provider configurations."""
        self.providers: Dict[str, ILLMProvider] = {}
        self.provider_health: Dict[str, ProviderHealth] = {}
        self.provider_priority: List[str] = []
        
        # Initialize providers
        for config in provider_configs:
            self._add_provider(config)
        
        if not self.providers:
            raise ConfigurationError("No providers configured for fallback manager")
        
        logger.info(f"Initialized fallback manager with {len(self.providers)} providers: {list(self.providers.keys())}")
    
    def _add_provider(self, config: Dict[str, Any]) -> None:
        """Add a provider to the fallback manager."""
        try:
            provider = LLMProviderFactory.create_from_config(config)
            provider_name = provider.provider_name
            
            self.providers[provider_name] = provider
            self.provider_health[provider_name] = ProviderHealth(provider_name)
            self.provider_priority.append(provider_name)
            
            logger.info(f"Added provider {provider_name} to fallback manager")
            
        except Exception as e:
            logger.error(f"Failed to add provider from config {config}: {e}")
            # Don't raise here - we want to continue with other providers
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [
            name for name, health in self.provider_health.items()
            if health.status != ProviderStatus.DISABLED and not health.should_circuit_break()
        ]
    
    def get_best_provider(self, exclude: Optional[List[str]] = None) -> Optional[ILLMProvider]:
        """Get the best available provider based on health and priority."""
        exclude = exclude or []
        available_providers = [
            name for name in self.get_available_providers()
            if name not in exclude
        ]
        
        if not available_providers:
            return None
        
        # Sort by priority and health status
        def provider_score(name: str) -> Tuple[int, int, float]:
            health = self.provider_health[name]
            priority = self.provider_priority.index(name) if name in self.provider_priority else 999
            status_score = {
                ProviderStatus.HEALTHY: 0,
                ProviderStatus.DEGRADED: 1,
                ProviderStatus.UNHEALTHY: 2,
                ProviderStatus.DISABLED: 3
            }[health.status]
            
            return (status_score, priority, health.consecutive_failures)
        
        best_provider_name = min(available_providers, key=provider_score)
        return self.providers[best_provider_name]
    
    async def chat_completion_with_fallback(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Attempt chat completion with fallback to other providers."""
        attempted_providers = []
        last_error = None
        
        while True:
            provider = self.get_best_provider(exclude=attempted_providers)
            if not provider:
                # No more providers to try
                if last_error:
                    raise last_error
                else:
                    raise LLMProviderError(
                        "No healthy providers available for chat completion",
                        error_type="no_providers_available"
                    )
            
            provider_name = provider.provider_name
            attempted_providers.append(provider_name)
            health = self.provider_health[provider_name]
            
            try:
                logger.info(f"Attempting chat completion with provider: {provider_name}")
                start_time = asyncio.get_event_loop().time()
                
                response = await provider.chat_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                # Record success
                response_time = asyncio.get_event_loop().time() - start_time
                health.record_success(response_time)
                
                logger.info(f"Chat completion successful with provider: {provider_name}")
                return response
                
            except Exception as e:
                last_error = e
                health.record_failure(type(e).__name__)
                
                logger.warning(f"Chat completion failed with provider {provider_name}: {e}")
                
                # If this was the last provider, re-raise the error
                if len(attempted_providers) >= len(self.providers):
                    break
                
                # Continue to next provider
                continue
        
        # If we get here, all providers failed
        raise LLMProviderError(
            f"All providers failed for chat completion. Last error: {last_error}",
            error_type="all_providers_failed"
        )
    
    async def stream_completion_with_fallback(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Attempt streaming completion with fallback to other providers."""
        attempted_providers = []
        last_error = None
        
        while True:
            provider = self.get_best_provider(exclude=attempted_providers)
            if not provider:
                # No more providers to try
                if last_error:
                    raise last_error
                else:
                    raise LLMProviderError(
                        "No healthy providers available for streaming completion",
                        error_type="no_providers_available"
                    )
            
            provider_name = provider.provider_name
            attempted_providers.append(provider_name)
            health = self.provider_health[provider_name]
            
            try:
                logger.info(f"Attempting streaming completion with provider: {provider_name}")
                start_time = asyncio.get_event_loop().time()
                
                async for chunk in provider.stream_completion(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                ):
                    yield chunk
                
                # Record success
                response_time = asyncio.get_event_loop().time() - start_time
                health.record_success(response_time)
                
                logger.info(f"Streaming completion successful with provider: {provider_name}")
                return
                
            except Exception as e:
                last_error = e
                health.record_failure(type(e).__name__)
                
                logger.warning(f"Streaming completion failed with provider {provider_name}: {e}")
                
                # If this was the last provider, re-raise the error
                if len(attempted_providers) >= len(self.providers):
                    break
                
                # Continue to next provider
                continue
        
        # If we get here, all providers failed
        raise LLMProviderError(
            f"All providers failed for streaming completion. Last error: {last_error}",
            error_type="all_providers_failed"
        )
    
    async def validate_all_providers(self) -> Dict[str, bool]:
        """Validate connections for all providers."""
        results = {}
        
        for provider_name, provider in self.providers.items():
            try:
                is_valid = await provider.validate_connection()
                results[provider_name] = is_valid
                
                if is_valid:
                    self.provider_health[provider_name].record_success(0.0)
                else:
                    self.provider_health[provider_name].record_failure("validation_failed")
                    
            except Exception as e:
                logger.error(f"Provider validation failed for {provider_name}: {e}")
                results[provider_name] = False
                self.provider_health[provider_name].record_failure("validation_error")
        
        return results
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for all providers."""
        return {
            "total_providers": len(self.providers),
            "healthy_providers": len([
                h for h in self.provider_health.values() 
                if h.status == ProviderStatus.HEALTHY
            ]),
            "provider_details": {
                name: health.get_health_info()
                for name, health in self.provider_health.items()
            }
        }
    
    def disable_provider(self, provider_name: str) -> bool:
        """Manually disable a provider."""
        if provider_name in self.provider_health:
            self.provider_health[provider_name].status = ProviderStatus.DISABLED
            logger.info(f"Provider {provider_name} manually disabled")
            return True
        return False
    
    def enable_provider(self, provider_name: str) -> bool:
        """Manually enable a provider."""
        if provider_name in self.provider_health:
            self.provider_health[provider_name].status = ProviderStatus.HEALTHY
            self.provider_health[provider_name].consecutive_failures = 0
            logger.info(f"Provider {provider_name} manually enabled")
            return True
        return False