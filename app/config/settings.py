"""Environment variable loading and configuration settings."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .models import (
    AgentConfig,
    APIConfig,
    AppConfig,
    Environment,
    LLMConfig,
    LogLevel,
    LLMProvider,
    MemoryType,
)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    environment: Environment = Field(Environment.DEVELOPMENT, alias="ENVIRONMENT")
    debug: bool = Field(False, alias="DEBUG")
    
    # LLM Configuration
    openai_api_key: Optional[str] = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: Optional[str] = Field(None, alias="ANTHROPIC_API_KEY")
    default_llm_provider: LLMProvider = Field(LLMProvider.OPENAI, alias="DEFAULT_LLM_PROVIDER")
    default_model: str = Field("gpt-4", alias="DEFAULT_MODEL")
    
    # Agent Configuration
    agent_timeout: int = Field(300, alias="AGENT_TIMEOUT")
    max_tool_calls: int = Field(10, alias="MAX_TOOL_CALLS")
    enable_memory: bool = Field(True, alias="ENABLE_MEMORY")
    memory_type: MemoryType = Field(MemoryType.BUFFER, alias="MEMORY_TYPE")
    
    # API Configuration
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")
    log_level: LogLevel = Field(LogLevel.INFO, alias="LOG_LEVEL")
    cors_origins: str = Field("*", alias="CORS_ORIGINS")
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    def to_app_config(self) -> AppConfig:
        """Convert settings to AppConfig model."""
        return AppConfig(
            environment=self.environment,
            debug=self.debug,
            llm=LLMConfig(
                openai_api_key=self.openai_api_key,
                anthropic_api_key=self.anthropic_api_key,
                default_provider=self.default_llm_provider,
                default_model=self.default_model,
            ),
            agent=AgentConfig(
                timeout=self.agent_timeout,
                max_tool_calls=self.max_tool_calls,
                enable_memory=self.enable_memory,
                memory_type=self.memory_type,
            ),
            api=APIConfig(
                host=self.api_host,
                port=self.api_port,
                log_level=self.log_level,
                cors_origins=self.cors_origins,
            ),
        )


def load_settings() -> Settings:
    """Load settings from environment variables and .env file."""
    return Settings()


def get_config() -> AppConfig:
    """Get application configuration."""
    settings = load_settings()
    return settings.to_app_config()