"""
LiteLLM Custom Handlers for RAG API Systems

This module provides a custom LiteLLM handler that integrates with RAG (Retrieval-Augmented
Generation) API systems. The RAGChatbotLLM handler wraps OpenAI-compatible RAG endpoints,
preserving response fields like 'context' with retrieval citations that standard LiteLLM
processing would filter out. This enables ASQI to validate and test RAG systems while
maintaining access to citation and context information for evaluation.
"""

from litellm.llms.custom_llm import CustomLLM
from litellm.files.main import ModelResponse
from typing import Any, Union
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class RAGChatbotLLM(CustomLLM):
    """
    Custom LiteLLM handler for RAG chatbot that preserves all response fields.

    This handler uses the OpenAI client to call the RAG server and returns
    the raw response without filtering any custom fields like 'context' with citations.
    """

    def __init__(self, timeout: float = 120.0):
        """
        Initialize the RAG chatbot LLM handler.

        Args:
            timeout: Request timeout in seconds (default: 120.0)
        """
        super().__init__()
        self.timeout = timeout

    def completion(self, *args, **kwargs) -> Union[ModelResponse, Any]:
        """Synchronous completion method."""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        api_base = kwargs.get("api_base")
        api_key = kwargs.get("api_key")
        temperature = kwargs.get("temperature")
        extra_body = kwargs.get("extra_body", {})

        if not api_base:
            raise ValueError("api_base is required for RAG chatbot")

        try:
            # Initialize OpenAI client with RAG server endpoint
            client = OpenAI(base_url=api_base, api_key=api_key)

            # Make the request using OpenAI client
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )

            # Return raw response dict with all fields preserved
            return (
                response.model_dump()
                if hasattr(response, "model_dump")
                else response.__dict__
            )

        except Exception as e:
            logger.error(f"RAG Chatbot API Error: {str(e)}")
            raise

    async def acompletion(self, *args, **kwargs) -> Union[ModelResponse, Any]:
        """Asynchronous completion method."""
        model = kwargs.get("model", "")
        messages = kwargs.get("messages", [])
        api_base = kwargs.get("api_base")
        api_key = kwargs.get("api_key")
        temperature = kwargs.get("temperature")
        extra_body = kwargs.get("extra_body", {})

        if not api_base:
            raise ValueError("api_base is required for RAG chatbot")

        try:
            # Initialize OpenAI client with RAG server endpoint
            client = OpenAI(base_url=api_base, api_key=api_key)

            # Make the request using OpenAI client
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                extra_body=extra_body if extra_body else None,
            )

            # Return raw response dict with all fields preserved
            return (
                response.model_dump()
                if hasattr(response, "model_dump")
                else response.__dict__
            )

        except Exception as e:
            logger.error(f"RAG Chatbot API Error: {str(e)}")
            raise


# Create handler instances
rag_chatbot_llm = RAGChatbotLLM()
