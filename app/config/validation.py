"""Configuration validation with clear error messages."""

from typing import List

from .models import AppConfig, Environment, LLMProvider


def validate_config(config: AppConfig) -> List[str]:
    """Validate application configuration and return list of error messages.
    
    Args:
        config: Application configuration to validate.
        
    Returns:
        List of validation error messages. Empty list if valid.
    """
    errors = []
    
    # Validate LLM configuration
    errors.extend(_validate_llm_config(config))
    
    # Validate agent configuration
    errors.extend(_validate_agent_config(config))
    
    # Validate API configuration
    errors.extend(_validate_api_config(config))
    
    # Validate environment-specific requirements
    errors.extend(_validate_environment_config(config))
    
    return errors


def _validate_llm_config(config: AppConfig) -> List[str]:
    """Validate LLM configuration."""
    errors = []
    
    # Check API keys based on default provider
    if config.llm.default_provider == LLMProvider.OPENAI:
        if not config.llm.openai_api_key:
            errors.append(
                "OpenAI API key is required when using OpenAI as the default provider. "
                "Set OPENAI_API_KEY environment variable."
            )
        elif len(config.llm.openai_api_key) < 10:
            errors.append(
                "OpenAI API key appears to be too short. "
                "Please check your API key."
            )
    
    elif config.llm.default_provider == LLMProvider.ANTHROPIC:
        if not config.llm.anthropic_api_key:
            errors.append(
                "Anthropic API key is required when using Anthropic as the default provider. "
                "Set ANTHROPIC_API_KEY environment variable."
            )
        elif len(config.llm.anthropic_api_key) < 10:
            errors.append(
                "Anthropic API key appears to be too short. "
                "Please check your API key."
            )
    
    # Validate model name
    if not config.llm.default_model or not config.llm.default_model.strip():
        errors.append("Default model name cannot be empty. Set DEFAULT_MODEL environment variable.")
    
    return errors


def _validate_agent_config(config: AppConfig) -> List[str]:
    """Validate agent configuration."""
    errors = []
    
    # Validate timeout
    if config.agent.timeout < 1:
        errors.append("Agent timeout must be at least 1 second.")
    elif config.agent.timeout > 3600:
        errors.append("Agent timeout cannot exceed 3600 seconds (1 hour).")
    
    # Validate max tool calls
    if config.agent.max_tool_calls < 1:
        errors.append("Maximum tool calls must be at least 1.")
    elif config.agent.max_tool_calls > 100:
        errors.append("Maximum tool calls cannot exceed 100.")
    
    return errors


def _validate_api_config(config: AppConfig) -> List[str]:
    """Validate API configuration."""
    errors = []
    
    # Validate port
    if config.api.port < 1 or config.api.port > 65535:
        errors.append("API port must be between 1 and 65535.")
    
    # Validate host
    if not config.api.host or not config.api.host.strip():
        errors.append("API host cannot be empty.")
    
    # Validate CORS origins
    if not config.api.cors_origins:
        errors.append("CORS origins cannot be empty.")
    
    return errors


def _validate_environment_config(config: AppConfig) -> List[str]:
    """Validate environment-specific configuration requirements."""
    errors = []
    
    if config.environment == Environment.PRODUCTION:
        # Production-specific validations
        if config.debug:
            errors.append("Debug mode should be disabled in production.")
        
        if config.api.cors_origins == ["*"]:
            errors.append(
                "CORS origins should be restricted in production. "
                "Avoid using '*' and specify allowed origins explicitly."
            )
        
        if config.api.log_level.value == "DEBUG":
            errors.append(
                "Debug logging should be avoided in production. "
                "Use INFO or higher log level."
            )
    
    elif config.environment == Environment.TESTING:
        # Testing-specific validations
        if config.agent.timeout > 60:
            errors.append(
                "Agent timeout should be reduced in testing environment "
                "for faster test execution."
            )
    
    return errors


def validate_startup_config() -> None:
    """Validate configuration at startup and raise exception if invalid.
    
    Raises:
        ValueError: If configuration is invalid with detailed error message.
    """
    from .factory import get_current_config
    
    try:
        config = get_current_config()
        errors = validate_config(config)
        
        if errors:
            error_message = (
                "Configuration validation failed. Please fix the following issues:\n\n" +
                "\n".join(f"• {error}" for error in errors) +
                "\n\nCheck your environment variables and .env file."
            )
            raise ValueError(error_message)
            
    except Exception as e:
        if isinstance(e, ValueError) and "Configuration validation failed" in str(e):
            raise
        
        # Re-raise with more context
        raise ValueError(
            f"Failed to load or validate configuration: {str(e)}\n"
            "Please check your environment variables and .env file."
        ) from e