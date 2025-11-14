from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import openai
import anthropic
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from LLM clients."""
    content: str
    usage: Dict[str, int] = None
    model: str = None
    finish_reason: str = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         **kwargs) -> LLMResponse:
        """Generate response from messages."""
        pass
    
    @abstractmethod
    def generate_structured_response(self, 
                                   messages: List[Dict[str, str]], 
                                   schema: Dict[str, Any],
                                   **kwargs) -> Dict[str, Any]:
        """Generate structured response following a schema."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API client."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", base_url: str = None):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (e.g., gpt-3.5-turbo, gpt-4)
            base_url: Optional base URL for API (for compatible services)
        """
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        logger.info(f"Initialized OpenAI client with model: {model}")
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                model=response.model,
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    def generate_structured_response(self, 
                                   messages: List[Dict[str, str]], 
                                   schema: Dict[str, Any],
                                   temperature: float = 0.3,
                                   **kwargs) -> Dict[str, Any]:
        """Generate structured response using JSON mode."""
        try:
            # Add schema instruction to the last message
            system_message = {
                "role": "system",
                "content": f"You must respond with valid JSON that follows this schema: {json.dumps(schema)}"
            }
            
            structured_messages = [system_message] + messages
            
            # Remove response_format for DeepSeek compatibility - it causes JSON parsing issues
            response = self.client.chat.completions.create(
                model=self.model,
                messages=structured_messages,
                temperature=temperature,
                # response_format={"type": "json_object"}, # Removed for DeepSeek compatibility
                **kwargs
            )
            
            content = response.choices[0].message.content
            
            # DEBUG: Log raw response before parsing
            logger.info(f"🔍 RAW LLM RESPONSE DEBUG:")
            logger.info(f"Response type: {type(content)}")
            logger.info(f"Response length: {len(content) if content else 0}")
            logger.info(f"Raw content: {repr(content)}")
            logger.info(f"First 500 chars: {content[:500] if content else 'None'}")
            
            
            # Clean content before JSON parsing to handle whitespace issues
            if content:
                content = content.strip()
                # Remove any markdown code block formatting if present
                if content.startswith('```json'):
                    content = content[7:]  # Remove ```json
                if content.endswith('```'):
                    content = content[:-3]  # Remove trailing ```
                content = content.strip()
            
            return json.loads(content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI structured response error: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude API client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Anthropic client.
        
        Args:
            api_key: Anthropic API key
            model: Model name (e.g., claude-3-sonnet-20240229, claude-3-opus-20240229)
        """
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        logger.info(f"Initialized Anthropic client with model: {model}")
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         temperature: float = 0.7,
                         max_tokens: int = 1000,
                         **kwargs) -> LLMResponse:
        """Generate response using Anthropic API."""
        try:
            # Convert messages format (Anthropic expects slightly different format)
            anthropic_messages = []
            system_content = ""
            
            for message in messages:
                if message["role"] == "system":
                    system_content += message["content"] + "\n"
                else:
                    anthropic_messages.append(message)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_content.strip() if system_content else None,
                messages=anthropic_messages,
                **kwargs
            )
            
            return LLMResponse(
                content=response.content[0].text,
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens
                },
                model=response.model,
                finish_reason=response.stop_reason
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    def generate_structured_response(self, 
                                   messages: List[Dict[str, str]], 
                                   schema: Dict[str, Any],
                                   temperature: float = 0.3,
                                   **kwargs) -> Dict[str, Any]:
        """Generate structured response by adding schema instructions."""
        try:
            # Add schema instruction to system message
            schema_instruction = f"You must respond with valid JSON that follows this exact schema: {json.dumps(schema)}"
            
            # Find or create system message
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            other_messages = [msg for msg in messages if msg["role"] != "system"]
            
            if system_messages:
                system_content = system_messages[0]["content"] + "\n\n" + schema_instruction
            else:
                system_content = schema_instruction
            
            # Create structured messages
            structured_messages = [{"role": "system", "content": system_content}] + other_messages
            
            response = self.generate_response(
                messages=structured_messages,
                temperature=temperature,
                **kwargs
            )
            
            return json.loads(response.content)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise
        except Exception as e:
            logger.error(f"Anthropic structured response error: {e}")
            raise


class LLMManager:
    """Manages multiple LLM clients and provides unified interface."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM manager with configuration.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        self.clients = {}
        self.default_client = None
        
        # Initialize OpenAI client if configured
        if config.get("openai", {}).get("api_key"):
            openai_config = config["openai"]
            self.clients["openai"] = OpenAIClient(
                api_key=openai_config["api_key"],
                model=openai_config.get("model", "gpt-3.5-turbo"),
                base_url=openai_config.get("base_url")
            )
            if not self.default_client:
                self.default_client = "openai"
        
        # Initialize Anthropic client if configured
        if config.get("anthropic", {}).get("api_key"):
            anthropic_config = config["anthropic"]
            self.clients["anthropic"] = AnthropicClient(
                api_key=anthropic_config["api_key"],
                model=anthropic_config.get("model", "claude-3-sonnet-20240229")
            )
            if not self.default_client:
                self.default_client = "anthropic"
        
        # Initialize DeepSeek (OpenAI-compatible) client if configured
        if config.get("deepseek", {}).get("api_key"):
            deepseek_config = config["deepseek"]
            self.clients["deepseek"] = OpenAIClient(
                api_key=deepseek_config["api_key"],
                model=deepseek_config.get("model", "deepseek-chat"),
                base_url="https://api.deepseek.com"
            )
            if not self.default_client:
                self.default_client = "deepseek"

        if not self.clients:
            raise ValueError("No LLM clients configured. Please provide API keys.")

        # Override default client if explicitly specified in config
        if config.get("default_provider") and config["default_provider"] in self.clients:
            self.default_client = config["default_provider"]
            logger.info(f"Using configured default provider: {self.default_client}")

        logger.info(f"Initialized LLM manager with clients: {list(self.clients.keys())}")
        logger.info(f"Default client: {self.default_client}")
    
    def get_client(self, provider: str = None) -> BaseLLMClient:
        """Get LLM client by provider name."""
        provider = provider or self.default_client
        if provider not in self.clients:
            raise ValueError(f"Provider '{provider}' not configured. Available: {list(self.clients.keys())}")
        return self.clients[provider]
    
    def generate_response(self, 
                         messages: List[Dict[str, str]], 
                         provider: str = None,
                         **kwargs) -> LLMResponse:
        """Generate response using specified or default provider."""
        client = self.get_client(provider)
        return client.generate_response(messages, **kwargs)
    
    def generate_structured_response(self, 
                                   messages: List[Dict[str, str]], 
                                   schema: Dict[str, Any],
                                   provider: str = None,
                                   **kwargs) -> Dict[str, Any]:
        """Generate structured response using specified or default provider."""
        client = self.get_client(provider)
        return client.generate_structured_response(messages, schema, **kwargs)
    
    def list_providers(self) -> List[str]:
        """List available LLM providers."""
        return list(self.clients.keys())
    
    def set_default_provider(self, provider: str):
        """Set the default LLM provider."""
        if provider not in self.clients:
            raise ValueError(f"Provider '{provider}' not configured.")
        self.default_client = provider
        logger.info(f"Set default provider to: {provider}")