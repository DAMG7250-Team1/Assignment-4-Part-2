"""
Generator for RAG pipeline.

This module provides functionality to generate responses to user queries
based on retrieved documents using a language model.
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable

logger = logging.getLogger(__name__)

class Generator:
    """
    Generator for RAG pipeline.
    
    This class handles generating responses to user queries based on 
    retrieved documents using a language model.
    """
    
    def __init__(self, 
                model_provider: str = "google",
                model_name: str = "gemini-flash",
                temperature: float = 0.7,
                max_tokens: int = 1000,
                context_window: int = 16000,
                system_prompt: Optional[str] = None,
                client: Optional[Any] = None,
                client_args: Optional[Dict[str, Any]] = None):
        """
        Initialize the generator.
        
        Args:
            model_provider: Provider of the language model (openai, anthropic, google, mistral)
            model_name: Name of the language model to use
            temperature: Temperature parameter for the model
            max_tokens: Maximum number of tokens to generate
            context_window: Maximum context window size for the model
            system_prompt: Optional system prompt to use
            client: Optional pre-configured client
            client_args: Arguments for client initialization
        """
        self.model_provider = model_provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        
        # Set default system prompt if not provided
        if system_prompt is None:
            self.system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            If the answer is not in the context, say that you don't know based on the provided information. 
            Always cite your sources when they are from specific documents."""
        else:
            self.system_prompt = system_prompt
        
        # Initialize client
        self.client = client
        if client is None:
            client_args = client_args or {}
            self._initialize_client(client_args)
        
        logger.info(f"Initialized Generator with model_provider='{model_provider}', model_name='{model_name}'")
    
    def _initialize_client(self, client_args: Dict[str, Any]) -> None:
        """Initialize the appropriate client based on model provider."""
        if self.model_provider == "openai":
            self._initialize_openai_client(client_args)
        elif self.model_provider == "anthropic":
            self._initialize_anthropic_client(client_args)
        elif self.model_provider == "mistral":
            self._initialize_mistral_client(client_args)
        elif self.model_provider == "google":
            self._initialize_google_client(client_args)
        else:
            raise ValueError(f"Unsupported model provider: {self.model_provider}")
    
    def _initialize_openai_client(self, client_args: Dict[str, Any]) -> None:
        """Initialize the OpenAI client."""
        try:
            import openai
            
            # Get API key from environment or client args
            api_key = client_args.get("api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key not provided and not found in environment")
            
            # Initialize client
            self.client = openai.OpenAI(api_key=api_key)
            logger.info(f"Initialized OpenAI client with model {self.model_name}")
            
        except ImportError:
            logger.error("Failed to import openai. Install with 'pip install openai'")
            raise
    
    def _initialize_anthropic_client(self, client_args: Dict[str, Any]) -> None:
        """Initialize the Anthropic client."""
        try:
            import anthropic
            
            # Get API key from environment or client args
            api_key = client_args.get("api_key") or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key not provided and not found in environment")
            
            # Initialize client
            self.client = anthropic.Anthropic(api_key=api_key)
            logger.info(f"Initialized Anthropic client with model {self.model_name}")
            
        except ImportError:
            logger.error("Failed to import anthropic. Install with 'pip install anthropic'")
            raise
    
    def _initialize_mistral_client(self, client_args: Dict[str, Any]) -> None:
        """Initialize the Mistral client."""
        try:
            from mistralai.client import MistralClient
            
            # Get API key from environment or client args
            api_key = client_args.get("api_key") or os.getenv("MISTRAL_API_KEY")
            if not api_key:
                raise ValueError("Mistral API key not provided and not found in environment")
            
            # Initialize client
            self.client = MistralClient(api_key=api_key)
            logger.info(f"Initialized Mistral client with model {self.model_name}")
            
        except ImportError:
            logger.error("Failed to import mistralai. Install with 'pip install mistralai'")
            raise
    
    def _initialize_google_client(self, client_args: Dict[str, Any]) -> None:
        """Initialize the Google Gemini client."""
        import google.generativeai as genai
        
        # Get API key from environment or client args
        api_key = client_args.get("api_key") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key not provided and not found in environment variables (GEMINI_API_KEY or GOOGLE_API_KEY)")
        
        # Initialize client
        genai.configure(api_key=api_key)
        self.client = genai
        logger.info(f"Initialized Google Gemini client with model {self.model_name}")
    
    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string for the model.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context = ""
        
        for i, doc in enumerate(documents):
            text = doc["text"]
            metadata = doc.get("metadata", {})
            source = metadata.get("file_name", f"Source {i+1}")
            
            context += f"Document {i+1} (Source: {source}):\n{text}\n\n"
        
        return context.strip()
    
    def _format_messages_openai(self, query: str, context: str) -> List[Dict[str, str]]:
        """Format messages for OpenAI API."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    
    def _format_prompt_anthropic(self, query: str, context: str) -> str:
        """Format prompt for Anthropic API."""
        from anthropic import HUMAN_PROMPT, AI_PROMPT
        
        return f"{HUMAN_PROMPT} {self.system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}{AI_PROMPT}"
    
    def _format_messages_mistral(self, query: str, context: str) -> List[Dict[str, str]]:
        """Format messages for Mistral API."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    
    def _format_prompt_google(self, query: str, context: str) -> str:
        """Format prompt for Google Gemini API."""
        return f"""
{self.system_prompt}

Below is the context information to help answer the question:

{context}

The user has asked the following question: {query}

Please provide a direct and comprehensive answer to the question using only the information from the context above. If the information is not in the context, please say so.
"""
    
    def generate(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Dictionary containing the generated response and metadata
        """
        if not documents:
            # No documents provided, generate response without context
            logger.warning("No documents provided for generation")
            return self.generate_without_context(query)
        
        # Format context from documents
        context = self._format_context(documents)
        
        # Generate response based on model provider
        try:
            if self.model_provider == "openai":
                return self._generate_openai(query, context, documents)
            elif self.model_provider == "anthropic":
                return self._generate_anthropic(query, context, documents)
            elif self.model_provider == "mistral":
                return self._generate_mistral(query, context, documents)
            elif self.model_provider == "google":
                return self._generate_google(query, context, documents)
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"Error generating response: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "model": self.model_name,
                    "provider": self.model_provider
                }
            }
    
    def _generate_openai(self, query: str, context: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using OpenAI API."""
        messages = self._format_messages_openai(query, context)
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            "response": response.choices[0].message.content,
            "metadata": {
                "model": self.model_name,
                "provider": self.model_provider,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens,
                    "total": response.usage.total_tokens
                },
                "sources": [doc.get("metadata", {}).get("file_name", f"Source {i+1}") 
                           for i, doc in enumerate(documents)]
            }
        }
    
    def _generate_anthropic(self, query: str, context: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Anthropic API."""
        prompt = self._format_prompt_anthropic(query, context)
        
        response = self.client.completions.create(
            prompt=prompt,
            model=self.model_name,
            max_tokens_to_sample=self.max_tokens,
            temperature=self.temperature
        )
        
        return {
            "response": response.completion,
            "metadata": {
                "model": self.model_name,
                "provider": self.model_provider,
                "sources": [doc.get("metadata", {}).get("file_name", f"Source {i+1}") 
                           for i, doc in enumerate(documents)]
            }
        }
    
    def _generate_mistral(self, query: str, context: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Mistral API."""
        messages = self._format_messages_mistral(query, context)
        
        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        return {
            "response": response.choices[0].message.content,
            "metadata": {
                "model": self.model_name,
                "provider": self.model_provider,
                "sources": [doc.get("metadata", {}).get("file_name", f"Source {i+1}") 
                           for i, doc in enumerate(documents)]
            }
        }
    
    def _generate_google(self, query: str, context: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate response using Google Gemini API."""
        try:
            # Format prompt
            prompt = self._format_prompt_google(query, context)
            
            # Generate response
            model = self.client.GenerativeModel(model_name=self.model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            
            # Extract text from response
            response_text = response.text
            
            return {
                "response": response_text,
                "metadata": {
                    "model": self.model_name,
                    "provider": self.model_provider,
                    "success": True,
                    "sources": [doc.get("metadata", {}).get("file_name", f"Source {i+1}") 
                               for i, doc in enumerate(documents)]
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating response with Google Gemini: {str(e)}")
            raise
    
    def generate_response(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """
        Generate a response to the query based on the retrieved documents.
        
        Args:
            query: User query
            documents: List of retrieved documents
            
        Returns:
            Generated response text
        """
        result = self.generate(query, documents)
        return result.get("response", "")
        
    def generate_without_context(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to the query without any context.
        
        Args:
            query: User query
            
        Returns:
            Dictionary containing response and metadata
        """
        try:
            if self.model_provider == "openai":
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ]
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "metadata": {
                        "model": self.model_name,
                        "provider": self.model_provider,
                        "no_context": True
                    }
                }
                
            elif self.model_provider == "anthropic":
                prompt = f"{self.system_prompt}\n\nHuman: {query}\n\nAssistant:"
                
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens_to_sample=self.max_tokens
                )
                
                return {
                    "response": response.completion,
                    "metadata": {
                        "model": self.model_name,
                        "provider": self.model_provider,
                        "no_context": True
                    }
                }
                
            elif self.model_provider == "mistral":
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": query}
                ]
                
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
                return {
                    "response": response.choices[0].message.content,
                    "metadata": {
                        "model": self.model_name,
                        "provider": self.model_provider,
                        "no_context": True
                    }
                }
                
            elif self.model_provider == "google":
                model = self.client.GenerativeModel(model_name=self.model_name)
                prompt = f"""
{self.system_prompt}

The user has asked the following question: {query}

Please provide a direct and comprehensive answer.
"""
                
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }
                )
                
                return {
                    "response": response.text,
                    "metadata": {
                        "model": self.model_name,
                        "provider": self.model_provider,
                        "no_context": True,
                        "success": True
                    }
                }
                
            else:
                raise ValueError(f"Unsupported model provider: {self.model_provider}")
                
        except Exception as e:
            logger.error(f"Error generating response without context: {str(e)}")
            return {
                "response": f"Error generating response: {str(e)}",
                "error": str(e),
                "success": False
            } 