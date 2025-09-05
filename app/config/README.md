# Configuration Management System

This module provides a comprehensive configuration management system for the LLM Agent API using Pydantic for type validation and environment variable loading.

## Features

- **Type-safe configuration** using Pydantic models
- **Environment variable loading** with `.env` file support
- **Environment-specific overrides** (development, testing, staging, production)
- **Startup validation** with clear error messages
- **Configuration factory** for different environments

## Usage

### Basic Usage

```python
from app.config import get_current_config, validate_startup_config

# Validate configuration at startup
validate_startup_config()

# Get current configuration
config = get_current_config()

print(f"Environment: {config.environment}")
print(f"LLM Provider: {config.llm.default_provider}")
print(f"API Port: {config.api.port}")
```

### Environment-Specific Configuration

```python
from app.config import create_config_for_environment, Environment

# Create configuration for specific environment
dev_config = create_config_for_environment(Environment.DEVELOPMENT)
prod_config = create_config_for_environment(Environment.PRODUCTION)
```

## Configuration Structure

The configuration is organized into several sections:

### LLM Configuration (`config.llm`)
- `openai_api_key`: OpenAI API key
- `anthropic_api_key`: Anthropic API key  
- `default_provider`: Default LLM provider (openai, anthropic, local)
- `default_model`: Default model to use

### Agent Configuration (`config.agent`)
- `timeout`: Agent execution timeout in seconds (1-3600)
- `max_tool_calls`: Maximum tool calls per execution (1-100)
- `enable_memory`: Enable conversation memory
- `memory_type`: Type of memory (buffer, summary, vector)

### API Configuration (`config.api`)
- `host`: API server host
- `port`: API server port (1-65535)
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `cors_origins`: CORS allowed origins

## Environment Variables

Set these environment variables or add them to your `.env` file:

```bash
# LLM Provider Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4

# Agent Configuration
AGENT_TIMEOUT=300
MAX_TOOL_CALLS=10
ENABLE_MEMORY=true
MEMORY_TYPE=buffer

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
CORS_ORIGINS=*

# Environment
ENVIRONMENT=development
DEBUG=false
```

## Environment-Specific Behavior

### Development
- Debug mode enabled
- Verbose logging (DEBUG level)
- All CORS origins allowed

### Testing  
- Debug mode disabled
- Different port (8001) to avoid conflicts
- Reduced timeouts for faster tests
- Buffer memory for predictable testing

### Staging
- Debug mode disabled
- Production-like settings with more logging
- Restricted CORS origins

### Production
- Debug mode disabled
- Appropriate logging levels
- CORS origins validation
- Additional security validations

## Validation

The system performs comprehensive validation:

- **API Key Validation**: Ensures required API keys are present based on provider
- **Range Validation**: Validates numeric ranges (ports, timeouts, etc.)
- **Environment-Specific Validation**: Additional checks for production environments
- **Startup Validation**: Validates configuration at application startup

## Error Handling

Configuration errors provide clear, actionable error messages:

```
Configuration validation failed. Please fix the following issues:

• OpenAI API key is required when using OpenAI as the default provider. Set OPENAI_API_KEY environment variable.
• API port must be between 1 and 65535.
• CORS origins should be restricted in production. Avoid using '*' and specify allowed origins explicitly.

Check your environment variables and .env file.
```