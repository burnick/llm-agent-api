"""OpenAI provider implementation."""

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import openai
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from app.models.responses import LLMResponse
from app.models.context import ModelInfo
from app.models.errors import LLMProviderError, ValidationError, ConfigurationError
from .base import BaseLLMProvider


logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation with async support and error handling."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI provider with configuration."""
        super().__init__(config)
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(
            api_key=self.config["api_key"],
            timeout=self.config.get("timeout", 30.0),
            max_retries=0  # We handle retries ourselves
        )
        
        # Provider configuration
        self.default_model = self.config.get("default_model", "gpt-3.5-turbo")
        self.max_retries = self.config.get("max_retries", 3)
        self.temperature = self.config.get("temperature", 0.7)
        self.max_tokens = self.config.get("max_tokens", None)
        
        # Model information cache
        self._model_info_cache: Dict[str, ModelInfo] = {}
        
        logger.info(f"Initialized OpenAI provider with model: {self.default_model}")
    
    def _get_required_config_fields(self) -> List[str]:
        """Get required configuration fields for OpenAI."""
        return ["api_key"]
    
    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openai"
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return [
            "gpt-4",
            "gpt-4-turbo-preview",
            "gpt-4-0125-preview",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0125",
            "gpt-3.5-turbo-16k"
        ]
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a chat completion response."""
        self._validate_messages(messages)
        
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        # Log the request
        await self._log_request(messages, model, temperature=temperature, max_tokens=max_tokens)
        
        start_time = time.time()
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Make the API call
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                
                # Process the response
                llm_response = self._process_chat_response(response, model)
                
                # Log successful response
                duration = time.time() - start_time
                await self._log_response(llm_response, duration)
                
                return llm_response
                
            except openai.RateLimitError as e:
                await self._handle_rate_limit_error(e, retry_count)
                retry_count += 1
                
            except openai.APITimeoutError as e:
                raise LLMProviderError(
                    f"OpenAI API timeout: {str(e)}",
                    provider=self.provider_name,
                    error_type="timeout"
                )
                
            except openai.APIConnectionError as e:
                raise LLMProviderError(
                    f"OpenAI API connection error: {str(e)}",
                    provider=self.provider_name,
                    error_type="connection"
                )
                
            except openai.AuthenticationError as e:
                raise LLMProviderError(
                    f"OpenAI authentication error: {str(e)}",
                    provider=self.provider_name,
                    error_type="authentication"
                )
                
            except openai.BadRequestError as e:
                raise ValidationError(f"Invalid request to OpenAI: {str(e)}")
                
            except Exception as e:
                await self._log_error(e, {
                    "model": model,
                    "message_count": len(messages),
                    "retry_count": retry_count
                })
                raise LLMProviderError(
                    f"Unexpected OpenAI error: {str(e)}",
                    provider=self.provider_name,
                    error_type="unexpected"
                )
        
        # If we get here, we've exhausted all retries
        raise LLMProviderError(
            f"Max retries ({self.max_retries}) exceeded for OpenAI",
            provider=self.provider_name,
            error_type="rate_limit_exceeded"
        )
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate a streaming chat completion response."""
        self._validate_messages(messages)
        
        model = model or self.default_model
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        # Log the request
        await self._log_request(messages, model, temperature=temperature, max_tokens=max_tokens, stream=True)
        
        retry_count = 0
        
        while retry_count <= self.max_retries:
            try:
                # Make the streaming API call
                stream = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                )
                
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                
                return  # Successfully completed streaming
                
            except openai.RateLimitError as e:
                await self._handle_rate_limit_error(e, retry_count)
                retry_count += 1
                
            except Exception as e:
                await self._log_error(e, {
                    "model": model,
                    "message_count": len(messages),
                    "retry_count": retry_count,
                    "streaming": True
                })
                raise LLMProviderError(
                    f"OpenAI streaming error: {str(e)}",
                    provider=self.provider_name,
                    error_type="streaming"
                )
        
        # If we get here, we've exhausted all retries
        raise LLMProviderError(
            f"Max retries ({self.max_retries}) exceeded for OpenAI streaming",
            provider=self.provider_name,
            error_type="rate_limit_exceeded"
        )
    
    async def embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings for the given text."""
        if not text.strip():
            raise ValidationError("Text for embedding cannot be empty")
        
        model = model or "text-embedding-ada-002"
        
        try:
            response = await self.client.embeddings.create(
                model=model,
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            await self._log_error(e, {"model": model, "text_length": len(text)})
            raise LLMProviderError(
                f"OpenAI embedding error: {str(e)}",
                provider=self.provider_name,
                error_type="embedding"
            )
    
    def get_model_info(self, model: Optional[str] = None) -> ModelInfo:
        """Get information about the specified model."""
        model = model or self.default_model
        
        # Return cached info if available
        if model in self._model_info_cache:
            return self._model_info_cache[model]
        
        # Model information mapping
        model_info_map = {
            "gpt-4": ModelInfo(
                name="gpt-4",
                provider="openai",
                max_tokens=8192,
                supports_streaming=True,
                supports_tools=True,
                cost_per_token=0.00003
            ),
            "gpt-4-turbo-preview": ModelInfo(
                name="gpt-4-turbo-preview",
                provider="openai",
                max_tokens=128000,
                supports_streaming=True,
                supports_tools=True,
                cost_per_token=0.00001
            ),
            "gpt-3.5-turbo": ModelInfo(
                name="gpt-3.5-turbo",
                provider="openai",
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=True,
                cost_per_token=0.0000015
            ),
            "gpt-3.5-turbo-16k": ModelInfo(
                name="gpt-3.5-turbo-16k",
                provider="openai",
                max_tokens=16384,
                supports_streaming=True,
                supports_tools=True,
                cost_per_token=0.000003
            )
        }
        
        if model not in model_info_map:
            # Default info for unknown models
            info = ModelInfo(
                name=model,
                provider="openai",
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=False
            )
        else:
            info = model_info_map[model]
        
        # Cache the info
        self._model_info_cache[model] = info
        return info
    
    async def validate_connection(self) -> bool:
        """Validate that the OpenAI connection is working."""
        try:
            # Make a simple API call to test the connection
            await self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
            
        except Exception as e:
            logger.warning(f"OpenAI connection validation failed: {e}")
            return False
    
    def _process_chat_response(self, response: ChatCompletion, model: str) -> LLMResponse:
        """Process OpenAI chat completion response into our standard format."""
        if not response.choices:
            raise LLMProviderError(
                "OpenAI returned empty response",
                provider=self.provider_name,
                error_type="invalid_response"
            )
        
        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason or "unknown"
        
        # Extract token usage
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        
        # Create metadata
        metadata = {
            "model": model,
            "finish_reason": finish_reason,
            "response_id": response.id,
            "created": response.created
        }
        
        return self._create_llm_response(
            content=content,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
            metadata=metadata
        )
    
    async def _handle_rate_limit_error(self, error: openai.RateLimitError, retry_count: int) -> None:
        """Handle rate limit errors with exponential backoff."""
        logger.warning(f"OpenAI rate limit hit: {error}")
        
        # Extract retry-after header if available
        retry_after = getattr(error.response, 'headers', {}).get('retry-after')
        if retry_after:
            try:
                delay = int(retry_after)
            except (ValueError, TypeError):
                delay = 2 ** retry_count  # Fallback to exponential backoff
        else:
            delay = 2 ** retry_count
        
        # Cap the delay at 60 seconds
        delay = min(delay, 60)
        
        logger.info(f"Waiting {delay}s before retry (attempt {retry_count + 1})")
        await asyncio.sleep(delay)
    
    def _count_tokens_estimate(self, text: str) -> int:
        """More accurate token count estimation for OpenAI models."""
        # Rough approximation based on OpenAI's tokenization
        # This is a simplified version - for production, consider using tiktoken
        return len(text.split()) + len(text) // 4