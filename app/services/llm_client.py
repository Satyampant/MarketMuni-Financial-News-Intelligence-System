"""
LLM Client Service for MarketMuni
Provides Groq API integration with structured output support using LangChain.
"""

import os
import time
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, ValidationError
import json

try:
    from langchain_groq import ChatGroq
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠ Warning: langchain-groq not installed. Install with: pip install langchain-groq")


class LLMServiceError(Exception):
    """Custom exception for LLM service failures."""
    pass


class GroqLLMClient:
    """
    LLM client for Groq API with structured output support.
    Uses LangChain for API calls and Pydantic for schema validation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize Groq LLM client.
        
        Args:
            api_key: Groq API key (defaults to GROQ_API_KEY env var)
            model: Model name (default: llama-3.3-70b-versatile)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens in response
            timeout: Request timeout in seconds
            max_retries: Number of retry attempts on failure
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-groq is required. Install with: pip install langchain-groq"
            )
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set via environment variable or pass to constructor."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Initialize LangChain ChatGroq
        self.llm = ChatGroq(
            api_key=self.api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        print(f"✓ GroqLLMClient initialized with model: {model}")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """
        Execute function with exponential backoff retry logic.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
            
        Raises:
            LLMServiceError: If all retries fail
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    error_msg = f"LLM service failed after {self.max_retries} attempts: {str(e)}"
                    print(f"✗ {error_msg}")
                    raise LLMServiceError(error_msg)
                
                # Check for rate limit errors
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    wait_time = (2 ** attempt) * 2  # 4s, 8s, 16s
                    print(f"⚠ Rate limit hit. Retrying in {wait_time}s... (attempt {attempt}/{self.max_retries})")
                else:
                    wait_time = 2 ** attempt  # 2s, 4s, 8s
                    print(f"⚠ Request failed: {e}. Retrying in {wait_time}s... (attempt {attempt}/{self.max_retries})")
                
                time.sleep(wait_time)
    
    def generate_structured_output(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        
        def _generate():
            # This handles schema conversion automatically.
            structured_llm = self.llm.with_structured_output(schema)
            
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            
            messages.append(HumanMessage(content=prompt))
            
            return structured_llm.invoke(messages)

        try:
            result = self._retry_with_backoff(_generate)
            if isinstance(result, BaseModel):
                return result.model_dump()
            return result
        except Exception as e:
            # Fallback logic could go here
            raise LLMServiceError(f"Structured generation failed: {e}")
    
    def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> str:
        """
        Generate unstructured text response.
        
        Args:
            prompt: User prompt/query
            system_message: Optional system message for context
            
        Returns:
            Generated text string
            
        Raises:
            LLMServiceError: If generation fails
        """
        def _generate():
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = self.llm.invoke(messages)
            return response.content
        
        return self._retry_with_backoff(_generate)
    
    def generate_with_json_mode(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON output using JSON mode (fallback method).
        
        Args:
            prompt: User prompt/query (should request JSON format)
            system_message: Optional system message
            
        Returns:
            Parsed JSON dict
            
        Raises:
            LLMServiceError: If generation or JSON parsing fails
        """
        def _generate():
            # Use model with JSON mode
            json_llm = self.llm.bind(model_kwargs={"response_format": {"type": "json_object"}})
            
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = json_llm.invoke(messages)
            
            # Parse JSON from response
            try:
                return json.loads(response.content)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Failed to parse JSON: {e}\nResponse: {response.content}")
        
        return self._retry_with_backoff(_generate)
    
    def validate_connection(self) -> bool:
        """
        Test API connection with simple prompt.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.generate_text(
                prompt="Respond with 'OK' if you can read this.",
                system_message="You are a helpful assistant."
            )
            return "ok" in response.lower()
        except Exception as e:
            print(f"✗ Connection validation failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get current model configuration.
        
        Returns:
            Dict with model settings
        """
        return {
            "provider": "groq",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }