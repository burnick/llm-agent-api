"""Configuration models using Pydantic for type validation."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class Environment(str, Enum):
    """Application environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class MemoryType(str, Enum):
    """Agent memory types."""
    BUFFER = "buffer"
    SUMMARY = "summary"
    VECTOR = "vector"


class LLMConfig(BaseModel):
    """LLM provider configuration."""
    openai_api_key: Optional[str] = Field(None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(None, description="Anthropic API key")
    default_provider: LLMProvider = Field(LLMProvider.OPENAI, description="Default LLM provider")
    default_model: str = Field("gpt-4", description="Default model to use")
    
    def model_post_init(self, __context):
        """Post-initialization validation."""
        # Validate API keys based on default provider
        if self.default_provider == LLMProvider.OPENAI and not self.openai_api_key:
            raise ValueError("OpenAI API key is required when using OpenAI as default provider")
        elif self.default_provider == LLMProvider.ANTHROPIC and not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required when using Anthropic as default provider")


class AgentConfig(BaseModel):
    """Agent configuration."""
    timeout: int = Field(300, ge=1, le=3600, description="Agent execution timeout in seconds")
    max_tool_calls: int = Field(10, ge=1, le=100, description="Maximum number of tool calls per execution")
    enable_memory: bool = Field(True, description="Enable conversation memory")
    memory_type: MemoryType = Field(MemoryType.BUFFER, description="Type of memory to use")


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = Field("0.0.0.0", description="API server host")
    port: int = Field(8000, ge=1, le=65535, description="API server port")
    log_level: LogLevel = Field(LogLevel.INFO, description="Logging level")
    cors_origins: List[str] = Field(["*"], description="CORS allowed origins")
    
    @validator('cors_origins', pre=True)
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            if v == "*":
                return ["*"]
            return [origin.strip() for origin in v.split(",")]
        return v


class AppConfig(BaseModel):
    """Main application configuration."""
    environment: Environment = Field(Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(False, description="Enable debug mode")
    llm: LLMConfig = Field(default_factory=LLMConfig, description="LLM configuration")
    agent: AgentConfig = Field(default_factory=AgentConfig, description="Agent configuration")
    api: APIConfig = Field(default_factory=APIConfig, description="API configuration")
    
    @validator('debug', pre=True, always=True)
    def set_debug_from_environment(cls, v, values):
        """Set debug mode based on environment."""
        env = values.get('environment', Environment.DEVELOPMENT)
        if env == Environment.DEVELOPMENT:
            return True
        return v

    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True