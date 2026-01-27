from __future__ import annotations

from typing import Any, Dict, Optional


def get_litellm_tracking_kwargs(
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Convert ASQI metadata into kwargs that can be splatted into OpenAI/LiteLLM calls.

    Expected ASQI metadata format (injected by workflow):
    {
      "user_id": "<optional>",
      "tags": {
        "job_id": "...",
        "job_type": "...",
        "parent_id": "...",
        # plus any arbitrary extra keys injected by metadata_config
      }
    }

    Output:
    {
      "user": "<user_id>",
      "extra_body": {"metadata": {"tags": ["k:v", ...]}}
    }
    """
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

    kwargs: Dict[str, Any] = {
        "extra_body": {
            "metadata": {
                "tags": tags_list,
            }
        }
    }
    if user_id:
        kwargs["user"] = user_id

    return kwargs
