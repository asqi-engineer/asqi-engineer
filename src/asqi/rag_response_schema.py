from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

__all__ = [
    "RAGCitation",
    "RAGContext",
    "validate_rag_response",
]


class RAGCitation(BaseModel):
    """A single citation from RAG retrieval.

    Represents one retrieved document/chunk that was used to generate the response.
    Citations provide traceability from generated content back to source documents.

    Attributes:
        retrieved_context: The retrieved text content that was used in generation
        document_id: Stable identifier for the originating document (e.g., filename, UUID)
        score: Optional retrieval ranking or confidence score, normalized to 0.0-1.0 range
        source_id: Optional collection/index/knowledge-base identifier for document origin

    Example:
        ```python
        citation = RAGCitation(
            retrieved_context="All customers are eligible for 30-day refunds",
            document_id="return_policy.pdf",
            score=0.95,
            source_id="company_policies"
        )
        ```
    """

    retrieved_context: str = Field(
        ...,
        description="The retrieved text content that was used in generation",
        min_length=1,
    )
    document_id: str = Field(
        ...,
        description="Stable identifier for the originating document",
        min_length=1,
    )
    score: Optional[float] = Field(
        None,
        description="Retrieval ranking or confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    source_id: Optional[str] = Field(
        None, description="Collection/index/knowledge-base identifier"
    )


class RAGContext(BaseModel):
    """Context object containing retrieval citations.

    This is the minimal structure expected in message.context for RAG API responses.
    Validates that the citations field exists and contains properly formatted citations.

    Attributes:
        citations: List of citations from retrieval. May be empty if no relevant
                  context was found, but the field itself is required.

    Example:
        ```python
        context = RAGContext(citations=[
            RAGCitation(
                retrieved_context="Policy states 30-day returns...",
                document_id="policy.pdf",
                score=0.92
            )
        ])
        ```
    """

    citations: List[RAGCitation] = Field(
        default_factory=list,
        description="List of citations from retrieval (may be empty if no relevant context found)",
    )


def validate_rag_response(response_dict: Dict[str, Any]) -> List[RAGCitation]:
    """Validate a RAG API response and extract citations.

    This function validates that a RAG system's response contains the required
    context.citations structure. It focuses on validating only the RAG-specific
    parts (message.context.citations) without recreating the entire OpenAI response.

    Args:
        response_dict: Raw response from RAG API as a dictionary.
                      Typically obtained via response.model_dump() from OpenAI client.

    Returns:
        List of validated RAGCitation objects from the first choice

    Raises:
        pydantic.ValidationError: If response doesn't contain valid context.citations
        KeyError: If required fields are missing from the response structure
        IndexError: If response has no choices

    Example:
        ```python
        from asqi.response_schemas import validate_rag_response
        from openai import OpenAI
        from pydantic import ValidationError

        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.chat.completions.create(
            model="rag-model",
            messages=[{"role": "user", "content": "What is the policy?"}]
        )

        try:
            # Validate and extract citations
            citations = validate_rag_response(response.model_dump())

            result = {
                "success": True,
                "num_citations": len(citations),
                "documents": [c.document_id for c in citations]
            }

        except (ValidationError, KeyError, IndexError) as e:
            result = {
                "success": False,
                "error": f"Invalid RAG response: {str(e)}"
            }
        ```
    """
    # Navigate to the context field in the first choice's message
    # Expected structure: response_dict["choices"][0]["message"]["context"]
    try:
        message = response_dict["choices"][0]["message"]
        context_dict = message["context"]
    except (KeyError, IndexError, TypeError) as e:
        raise KeyError(
            f"Response missing required structure 'choices[0].message.context': {e}"
        )

    # Validate the context structure using our Pydantic model
    context = RAGContext(**context_dict)

    return context.citations
