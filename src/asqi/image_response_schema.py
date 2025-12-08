from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

__all__ = [
    "ImageObject",
    "UsageInfo",
    "ImageResponse",
    "validate_image_response",
]


class UsageInfo(BaseModel):
    """Usage statistics for image generation/editing requests.

    Tracks token consumption across different modalities.

    Attributes:
        input_tokens: Total tokens in the input (images + text)
        input_tokens_details: Breakdown of input tokens by type
        output_tokens: Number of tokens in the output image(s)
        total_tokens: Total tokens used for the request

    Example:
        ```python
        usage = UsageInfo(
            input_tokens=100,
            input_tokens_details={"image_tokens": 85, "text_tokens": 15},
            output_tokens=170,
            total_tokens=270
        )
        ```
    """

    input_tokens: Optional[int] = Field(
        None,
        description="Total tokens in the input (images + text)",
        ge=0,
    )
    input_tokens_details: Optional[Dict[str, int]] = Field(
        None,
        description="Breakdown of input tokens (e.g., {'image_tokens': 85, 'text_tokens': 15})",
    )
    output_tokens: Optional[int] = Field(
        None,
        description="Number of tokens in the output image(s)",
        ge=0,
    )
    total_tokens: Optional[int] = Field(
        None,
        description="Total tokens used for the request",
        ge=0,
    )


class ImageObject(BaseModel):
    """An individual generated or edited image.

    Represents one image in the response data array.

    Attributes:
        url: The URL of the generated/edited image (when response_format is "url")
        b64_json: Base64-encoded JSON of the image (when response_format is "b64_json")
        revised_prompt: Optional revised prompt returned by some providers

    Example:
        ```python
        image = ImageObject(
            url="https://example.com/generated_image.png",
            revised_prompt="A beautiful sunset over mountains..."
        )
        ```
    """

    url: Optional[str] = Field(
        None,
        description="The URL of the generated/edited image (when response_format is 'url')",
    )
    b64_json: Optional[str] = Field(
        None,
        description="Base64-encoded JSON of the image (when response_format is 'b64_json')",
    )
    revised_prompt: Optional[str] = Field(
        None,
        description="Optional revised prompt returned by some providers (like DALL-E)",
    )


class ImageResponse(BaseModel):
    """Standardized response format for image generation and editing APIs.

    Compatible with OpenAI Images API format. Used by LiteLLM to ensure
    consistent responses across different image providers.

    Attributes:
        created: Integer timestamp when the request was created
        data: Array of generated/edited images
        usage: Optional usage statistics

    Example:
        ```python
        response = ImageResponse(
            created=1703658209,
            data=[
                ImageObject(
                    url="https://example.com/image.png",
                    revised_prompt="A beautiful landscape..."
                )
            ],
            usage=UsageInfo(
                input_tokens=85,
                output_tokens=170,
                total_tokens=255
            )
        )
        ```
    """

    created: int = Field(
        ...,
        description="Integer timestamp when the request was created",
        gt=0,
    )
    data: List[ImageObject] = Field(
        ...,
        description="Array of generated/edited images",
        min_items=1,
    )
    usage: Optional[UsageInfo] = Field(
        None,
        description="Optional usage statistics (token counts)",
    )


def validate_image_response(response_dict: Dict[str, Any]) -> ImageResponse:
    """Validate an image generation/editing API response.

    This function validates that an image system's response follows the
    OpenAI Images API format. It ensures the response contains properly
    formatted image data and validates all required fields.

    Args:
        response_dict: Raw response from image API as a dictionary.
                      Typically obtained via response.model_dump() from OpenAI client.

    Returns:
        Validated ImageResponse object

    Raises:
        pydantic.ValidationError: If response doesn't contain valid image data
        KeyError: If required fields are missing from the response structure

    Example:
        ```python
        from asqi.image_response_schema import validate_image_response
        from openai import OpenAI
        from pydantic import ValidationError

        client = OpenAI(base_url=base_url, api_key=api_key)
        response = client.images.generate(
            model="dall-e-3",
            prompt="A cute baby sea otter"
        )

        try:
            # Validate and extract image data
            validated_response = validate_image_response(response.model_dump())

            result = {
                "success": True,
                "num_images": len(validated_response.data),
                "image_urls": [img.url for img in validated_response.data if img.url],
                "total_tokens": validated_response.usage.total_tokens if validated_response.usage else None
            }

        except ValidationError as e:
            result = {
                "success": False,
                "error": f"Invalid image response: {str(e)}"
            }
        ```
    """
    # Validate the entire response structure using our Pydantic model
    response = ImageResponse(**response_dict)

    return response
