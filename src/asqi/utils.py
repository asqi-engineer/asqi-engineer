"""Utility functions for ASQI library and test containers."""

from __future__ import annotations

from typing import Any, Dict, Optional, Union

from asqi.schemas import ExecutionMetadata


def get_openai_tracking_kwargs(
    metadata: Optional[Union[Dict[str, Any], ExecutionMetadata]] = None,
) -> Dict[str, Any]:
    """
    Convert ASQI metadata into kwargs that can be splatted into OpenAI/LiteLLM calls.

    This function is designed to be used in test containers to convert the metadata
    structure passed from the workflow into OpenAI/LiteLLM client parameters.

    Accepts either a dict or ExecutionMetadata Pydantic model for type safety.

    Expected ASQI metadata format (from workflow):
    {
      "user_id": "<optional>",          # Top-level from metadata_config
      "custom_field": "<optional>",     # Other top-level from metadata_config
      "tags": {                          # Workflow tracking + metadata_config["tags"]
        "job_id": "...",
        "job_type": "...",
        "parent_id": "...",
        "experiment_id": "...",          # Example custom tag
      }
    }

    Output (OpenAI/LiteLLM client kwargs):
    {
      "user": "<user_id>",                          # From metadata["user_id"]
      "extra_body": {
        "metadata": {
          "tags": ["k:v", ...],                     # From metadata["tags"]
          "custom_field": "..."                      # Other metadata keys
        }
      }
    }

    Args:
        metadata: Optional metadata dictionary from ASQI workflow

    Returns:
        Dictionary of kwargs to pass to OpenAI/LiteLLM client methods

    Example:
        >>> metadata = {
        ...     "user_id": "user123",
        ...     "custom_field": "experiment_A",
        ...     "tags": {"job_id": "test-001", "job_type": "test"}
        ... }
        >>> kwargs = get_openai_tracking_kwargs(metadata)
        >>> client.chat.completions.create(
        ...     model="gpt-4",
        ...     messages=[...],
        ...     **kwargs
        ... )
    """
    # Convert Pydantic model to dict if needed
    if isinstance(metadata, ExecutionMetadata):
        metadata = metadata.model_dump()

    metadata = metadata or {}

    user_id = metadata.get("user_id", "") or ""
    tags_dict = metadata.get("tags", {}) or {}

    # If tags is not a dict, fail safe (avoid crashing container)
    if not isinstance(tags_dict, dict):
        tags_dict = {"tags": str(tags_dict)}

    # Convert tags dict into ["key:value", ...]
    tags_list = []
    for k, v in tags_dict.items():
        if v is None:
            continue
        # Flatten values safely
        if isinstance(v, (dict, list, tuple)):
            v_str = str(v)
        else:
            v_str = str(v)
        tags_list.append(f"{k}:{v_str}")

    # Build extra_body.metadata with tags and any other metadata fields
    extra_metadata = {"tags": tags_list}

    # Add other top-level metadata fields to extra_body.metadata
    reserved_keys = {"user_id", "tags"}
    for key, value in metadata.items():
        if key not in reserved_keys:
            extra_metadata[key] = value

    kwargs: Dict[str, Any] = {
        "extra_body": {
            "metadata": extra_metadata,
        }
    }
    if user_id:
        kwargs["user"] = user_id

    return kwargs
