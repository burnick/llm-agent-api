"""Factory for creating LLM provider instances."""

import logging
from typing import Any, Dict, Optional, Type
from enum import Enum

from app.models.interfaces import ILLMProvider
from app.models.errors import ConfigurationError, ValidationError
from .openai_provider import OpenAIProvider


logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    _providers: Dict[str, Type[ILLMProvider]] = {
        ProviderType.OPENAI: OpenAIProvider,
        # Additional providers will be registered here
    }
    
    @classmethod
    def register_provider(cls, provider_type: str, provider_class: Type[ILLMProvider]) -> None:
        """Register a new provider type."""
        if not issubclass(provider_class, ILLMProvider):
            raise ValidationError(f"Provider class must implement ILLMProvider interface")
        
        cls._providers[provider_type] = provider_class
        logger.info(f"Registered LLM provider: {provider_type}")
    
    @classmethod
    def create_provider(
        cls, 
        provider_type: str, 
        config: Dict[str, Any]
    ) -> ILLMProvider:
        """Create a provider instance of the specified type."""
        if provider_type not in cls._providers:
            available_providers = list(cls._providers.keys())
            raise ConfigurationError(
                f"Unknown provider type: {provider_type}. "
                f"Available providers: {available_providers}"
            )
        
        provider_class = cls._providers[provider_type]
        
        try:
            logger.info(f"Creating {provider_type} provider instance")
            provider = provider_class(config)
            logger.info(f"Successfully created {provider_type} provider")
            return provider
            
        except Exception as e:
            logger.error(f"Failed to create {provider_type} provider: {e}")
            raise ConfigurationError(
                f"Failed to initialize {provider_type} provider: {str(e)}"
            ) from e
    
    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider types."""
        return list(cls._providers.keys())
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> ILLMProvider:
        """Create a provider from configuration dictionary."""
        if "provider_type" not in config:
            raise ConfigurationError("Configuration must include 'provider_type'")
        
        provider_type = config["provider_type"]
        
        # Extract provider-specific config
        provider_config = {k: v for k, v in config.items() if k != "provider_type"}
        
        return cls.create_provider(provider_type, provider_config)
    
    @classmethod
    def create_default_provider(
        cls, 
        default_type: Optional[str] = None,
        **config_overrides
    ) -> ILLMProvider:
        """Create a default provider instance."""
        provider_type = default_type or ProviderType.OPENAI
        
        # Default configuration that can be overridden
        default_config = {
            "timeout": 30.0,
            "max_retries": 3,
            "temperature": 0.7,
        }
        
        # Merge with overrides
        config = {**default_config, **config_overrides}
        
        return cls.create_provider(provider_type, config)
    
    @classmethod
    def validate_provider_config(
        cls, 
        provider_type: str, 
        config: Dict[str, Any]
    ) -> bool:
        """Validate configuration for a specific provider type."""
        if provider_type not in cls._providers:
            return False
        
        try:
            # Create a temporary instance to validate config
            provider_class = cls._providers[provider_type]
            temp_provider = provider_class(config)
            return True
        except Exception as e:
            logger.warning(f"Config validation failed for {provider_type}: {e}")
            return False