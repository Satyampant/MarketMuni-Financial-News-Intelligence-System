"""
LLM Client Service for MarketMuni - Groq API integration via LangChain.
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
    pass


class GroqLLMClient:
    """Groq API client wrapper using LangChain with structured output support."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 30,
        max_retries: int = 3
    ):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("langchain-groq is required.")
        
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found.")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.llm = ChatGroq(
            api_key=self.api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        
        print(f"✓ GroqLLMClient initialized with model: {model}")
    
    def _retry_with_backoff(self, func, *args, **kwargs):
        """Executes function with exponential backoff retry logic."""
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    error_msg = f"LLM service failed after {self.max_retries} attempts: {str(e)}"
                    print(f"✗ {error_msg}")
                    raise LLMServiceError(error_msg)
                
                # Handle rate limits (429) specifically with longer wait
                error_str = str(e).lower()
                if "rate limit" in error_str or "429" in error_str:
                    wait_time = (2 ** attempt) * 2
                    print(f"⚠ Rate limit hit. Retrying in {wait_time}s... (attempt {attempt}/{self.max_retries})")
                else:
                    wait_time = 2 ** attempt
                    print(f"⚠ Request failed: {e}. Retrying in {wait_time}s... (attempt {attempt}/{self.max_retries})")
                
                time.sleep(wait_time)
    
    def generate_structured_output(
        self,
        prompt: str,
        schema: Type[BaseModel],
        system_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generates response matching a Pydantic schema."""
        
        def _generate():
            # Automatically handles schema conversion/tool calling
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
            raise LLMServiceError(f"Structured generation failed: {e}")
    
    def generate_text(
        self,
        prompt: str,
        system_message: Optional[str] = None
    ) -> str:
        """Generates standard unstructured text response."""
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
        """Fallback generation using JSON mode instead of tool binding."""
        def _generate():
            json_llm = self.llm.bind(model_kwargs={"response_format": {"type": "json_object"}})
            
            messages = []
            if system_message:
                messages.append(SystemMessage(content=system_message))
            messages.append(HumanMessage(content=prompt))
            
            response = json_llm.invoke(messages)
            
            try:
                return json.loads(response.content)
            except json.JSONDecodeError as e:
                raise ValidationError(f"Failed to parse JSON: {e}\nResponse: {response.content}")
        
        return self._retry_with_backoff(_generate)
    
    def validate_connection(self) -> bool:
        """Simple connectivity test."""
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
        return {
            "provider": "groq",
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }