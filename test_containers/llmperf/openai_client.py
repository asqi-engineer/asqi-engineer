"""Custom LLM client using OpenAI SDK for proper metadata handling."""

from __future__ import annotations

import os
import time
from typing import Any, Dict, Tuple

import ray
from llmperf import common_metrics
from llmperf.models import RequestConfig
from llmperf.ray_llm_client import LLMClient
from openai import OpenAI
from utils import get_openai_tracking_kwargs


@ray.remote
class OpenAISDKClient(LLMClient):
    """Client for OpenAI Chat Completions API using official SDK with metadata support."""

    def __init__(self, metadata: Dict[str, Any] | None = None):
        """Initialize with optional metadata for tracking.

        Args:
            metadata: ASQI metadata dict with 'tags' and optional 'user_id'
        """
        self.metadata = metadata or {}

    def llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[Dict[str, Any], str, RequestConfig]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
        model = request_config.model
        sampling_params = request_config.sampling_params or {}

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = -1
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0

        metrics = {}
        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        address = os.environ.get("OPENAI_API_BASE")
        if not address:
            raise ValueError("the environment variable OPENAI_API_BASE must be set.")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError("the environment variable OPENAI_API_KEY must be set.")

        # Build request kwargs - filter out metadata/user from sampling_params
        # since we handle them separately via tracking_kwargs
        filtered_sampling = {
            k: v for k, v in sampling_params.items() if k not in ("metadata", "user")
        }

        # Get standardized tracking kwargs from metadata
        tracking_kwargs = get_openai_tracking_kwargs(self.metadata)

        try:
            client = OpenAI(base_url=address, api_key=key)

            response = client.chat.completions.create(
                model=model,
                messages=message,
                stream=True,
                **filtered_sampling,
                **tracking_kwargs,
            )

            for chunk in response:
                tokens_received += 1

                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if delta.content:
                        if not ttft:
                            ttft = time.monotonic() - start_time
                            time_to_next_token.append(ttft)
                        else:
                            time_to_next_token.append(
                                time.monotonic() - most_recent_received_token_time
                            )
                        most_recent_received_token_time = time.monotonic()
                        generated_text += delta.content

            total_request_time = time.monotonic() - start_time
            output_throughput = (
                tokens_received / total_request_time if total_request_time > 0 else 0
            )

        except Exception as e:
            error_msg = str(e)
            error_response_code = getattr(e, "status_code", -1)
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code
            print(f"Warning Or Error: {e}")
            print(error_response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token)
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
