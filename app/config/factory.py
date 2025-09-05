"""Configuration factory for different environments."""

import os
from typing import Dict, Any, Optional

from .models import AppConfig, Environment, LogLevel, LLMProvider
from .settings import get_config, Settings
from .validation import validate_config


class ConfigurationError(Exception):
    """Configuration error exception."""
    pass


class ConfigFactory:
    """Factory for creating environment-specific configurations."""
    
    @staticmethod
    def create_config(environment: Optional[Environment] = None) -> AppConfig:
        """Create configuration for the specified environment.
        
        Args:
            environment: Target environment. If None, uses ENVIRONMENT env var.
            
        Returns:
            AppConfig: Validated application configuration.
            
        Raises:
            ConfigurationError: If configuration is invalid.
        """
        if environment:
            # Override environment variable temporarily
            original_env = os.environ.get("ENVIRONMENT")
            os.environ["ENVIRONMENT"] = environment.value
            
        try:
            config = get_config()
            
            # Apply environment-specific overrides
            if config.environment == Environment.DEVELOPMENT:
                config = ConfigFactory._apply_development_overrides(config)
            elif config.environment == Environment.TESTING:
                config = ConfigFactory._apply_testing_overrides(config)
            elif config.environment == Environment.STAGING:
                config = ConfigFactory._apply_staging_overrides(config)
            elif config.environment == Environment.PRODUCTION:
                config = ConfigFactory._apply_production_overrides(config)
            
            # Validate the final configuration
            validation_errors = validate_config(config)
            if validation_errors:
                error_msg = "Configuration validation failed:\n" + "\n".join(validation_errors)
                raise ConfigurationError(error_msg)
            
            return config
            
        finally:
            # Restore original environment variable
            if environment and original_env is not None:
                os.environ["ENVIRONMENT"] = original_env
            elif environment and original_env is None:
                os.environ.pop("ENVIRONMENT", None)
    
    @staticmethod
    def _apply_development_overrides(config: AppConfig) -> AppConfig:
        """Apply development environment overrides."""
        # Enable debug mode in development
        config.debug = True
        
        # Use more verbose logging in development
        if config.api.log_level == LogLevel.INFO:
            config.api.log_level = LogLevel.DEBUG
        
        # Allow all CORS origins in development
        if config.api.cors_origins == ["*"]:
            config.api.cors_origins = ["*"]
        
        return config
    
    @staticmethod
    def _apply_testing_overrides(config: AppConfig) -> AppConfig:
        """Apply testing environment overrides."""
        # Disable debug mode in testing
        config.debug = False
        
        # Use a different port for testing to avoid conflicts
        config.api.port = 8001
        
        # Reduce timeouts for faster tests
        config.agent.timeout = 30
        
        # Use buffer memory for predictable testing
        config.agent.memory_type = "buffer"
        
        return config
    
    @staticmethod
    def _apply_staging_overrides(config: AppConfig) -> AppConfig:
        """Apply staging environment overrides."""
        # Disable debug mode in staging
        config.debug = False
        
        # Use production-like settings but with more logging
        config.api.log_level = LogLevel.INFO
        
        # Restrict CORS origins in staging
        if config.api.cors_origins == ["*"]:
            config.api.cors_origins = ["https://staging.example.com"]
        
        return config
    
    @staticmethod
    def _apply_production_overrides(config: AppConfig) -> AppConfig:
        """Apply production environment overrides."""
        # Disable debug mode in production
        config.debug = False
        
        # Use appropriate logging level for production
        if config.api.log_level == LogLevel.DEBUG:
            config.api.log_level = LogLevel.INFO
        
        # Ensure CORS origins are properly configured for production
        if config.api.cors_origins == ["*"]:
            # In production, we should have specific origins
            # This is a warning case that should be addressed
            pass
        
        return config


def create_config_for_environment(env: Environment) -> AppConfig:
    """Create configuration for a specific environment.
    
    Args:
        env: Target environment.
        
    Returns:
        AppConfig: Environment-specific configuration.
        
    Raises:
        ConfigurationError: If configuration is invalid.
    """
    return ConfigFactory.create_config(env)


def get_current_config() -> AppConfig:
    """Get configuration for the current environment.
    
    Returns:
        AppConfig: Current environment configuration.
        
    Raises:
        ConfigurationError: If configuration is invalid.
    """
    return ConfigFactory.create_config()