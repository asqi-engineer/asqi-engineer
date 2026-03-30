# System API Reference

This document defines the exact request and response formats for the various system types supported by ASQI Engineer.

## Quick Reference Table

| System Type | Key Request Fields | Key Response Fields | Protocol / Spec |
| :--- | :--- | :--- | :--- |
| `llm_api` | `model`, `messages`, `temperature`, `thinking`, `reasoning_effort` | `id`, `choices`, `usage` | OpenAI Chat Completion |
| `rag_api` | `model`, `messages` | `choices[0].message.context.citations` | OpenAI + RAG Extension |
| `vlm_api` | `model`, `messages` (multi-modal), `supports_vision` | `id`, `choices` | OpenAI Multi-modal |
| `image_generation_api` | `prompt`, `model`, `size`, `n` | `data` (URLs or base64) | OpenAI Image Gen |
| `image_editing_api` | `image`, `mask`, `prompt` | `data` (URLs or base64) | OpenAI Image Edit |
| `embedding_api` | `model`, `input` | `data` (vectors), `usage` | OpenAI Embeddings |
| `agent_cli` | `base_url`, `model`, `provider` | `results.success`, `results.score` | ASQI Agent Protocol |
| `rest_api` | `params` (Custom JSON) | `results.success`, `results.score` | Custom REST / ASQI |
| `Object Detection` | `image`, `model`, `provider`, `api_key`, `base_url` | `detections` (list of boxes) | Internal Multipart/Form |

---

## 1. LLM API (`llm_api`)

The `llm_api` type is used for standard text-based Large Language Models.

### Request Format
Follows the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat/create).

**Key Fields:**
- `model` (string, required): The ID of the model to use.
- `messages` (array, required): A list of message objects (`role`, `content`).
- `temperature` (float, optional): Sampling temperature.
- `max_tokens` (integer, optional): Maximum tokens to generate.
- `thinking` (object, optional): Configuration for models with extended reasoning (e.g., Claude 3.7).
    - `type` (string): "enabled" or "adaptive".
    - `budget_tokens` (integer): Token budget for thinking (required if type="enabled").
- `reasoning_effort` (string, optional): Effort level for reasoning models (e.g., "low", "medium", "high").
- `extra_body` (object, optional): Any provider-specific parameters.

### Response Format
Standard OpenAI Chat Completion response.

**Example Request:**
```json
{
  "model": "gpt-4",
  "messages": [
    { "role": "system", "content": "You are a helpful assistant." },
    { "role": "user", "content": "Hello!" }
  ],
  "temperature": 0.7
}
```

**Example Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-4",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 18,
    "completion_tokens": 9,
    "total_tokens": 27
  }
}
```

---

## 2. RAG API (`rag_api`)

The `rag_api` type is used for Retrieval-Augmented Generation systems. It extends the standard LLM response to include source citations.

### Request Format
Identical to the **LLM API** request.

### Response Format
Extends the OpenAI response structure. The first choice's message object **must** contain a `context` field with `citations`.

**Exact Fields in `choices[0].message`:**
- `content` (string): The generated answer.
- `context` (object):
    - `citations` (array): A list of citation objects:
        - `retrieved_context` (string, required): The text chunk used for the answer.
        - `document_id` (string, optional): Filename or stable ID of the source.
        - `score` (float, optional): Retrieval confidence score (0.0 to 1.0).
        - `source_id` (string, optional): Identifier for the knowledge base.

**Example Request:**
```json
{
  "model": "rag-model",
  "messages": [{ "role": "user", "content": "What is the return policy?" }]
}
```

**Example Response:**
```json
{
  "id": "chatcmpl-rag-123",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Our return policy allows for 30-day refunds.",
      "context": {
        "citations": [
          {
            "retrieved_context": "All customers are eligible for 30-day refunds",
            "document_id": "return_policy.pdf",
            "score": 0.95
          }
        ]
      }
    }
  }]
}
```

---

## 3. VLM API (`vlm_api`)

The `vlm_api` type is used for Vision Language Models that accept both text and images.

### Request Format
Follows the OpenAI multi-modal message format.

**Key Fields:**
- `model` (string, required): The ID of the model to use.
- `messages` (array, required): A list of message objects (`role`, `content`).
- `supports_vision` (boolean, constant): Must be `true` for `vlm_api`.

**Example Message Content:**
```json
{
  "role": "user",
  "content": [
    { "type": "text", "text": "What is in this image?" },
    { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
  ]
}
```

### Response Format
Standard OpenAI Chat Completion response.

**Example Request:**
```json
{
  "model": "gpt-4-vision",
  "messages": [
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "What is in this image?" },
        { "type": "image_url", "image_url": { "url": "data:image/jpeg;base64,..." } }
      ]
    }
  ],
  "supports_vision": true
}
```

**Example Response:**
```json
{
  "id": "chatcmpl-vlm-123",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The image shows a blue ocean with a small island."
    }
  }]
}
```

---

## 4. Image Generation API (`image_generation_api`)

Used for generating images from text prompts.

### Request Format
Follows the [OpenAI Image Generation API](https://platform.openai.com/docs/api-reference/images/create).

**Key Fields:**
- `prompt` (string, required): A text description of the desired image(s).
- `model` (string, optional): The model to use (e.g., `dall-e-3`).
- `n` (integer, optional): The number of images to generate.
- `size` (string, optional): The dimensions of the generated images.

### Response Format
Standard OpenAI Image Generation response (containing URLs or base64 data).

**Example Request:**
```json
{
  "prompt": "A futuristic city in the style of cyberpunk",
  "model": "dall-e-3",
  "n": 1,
  "size": "1024x1024"
}
```

**Example Response:**
```json
{
  "created": 1677652288,
  "data": [
    { "url": "https://example.com/generated_image.png" }
  ]
}
```

---

## 5. Image Editing API (`image_editing_api`)

Used for editing or extending existing images based on a prompt.

### Request Format
Follows the [OpenAI Image Edit API](https://platform.openai.com/docs/api-reference/images/createEdit).

**Key Fields:**
- `image` (file, required): The image to edit.
- `mask` (file, optional): An additional image whose fully transparent areas indicate where `image` should be edited.
- `prompt` (string, required): A text description of the desired edit.

### Response Format
Standard OpenAI Image response.

**Example Request (Multipart/Form-Data):**
- `image`: [binary file]
- `mask`: [binary file]
- `prompt`: "Add a hat to the person"

**Example Response:**
```json
{
  "created": 1677652288,
  "data": [
    { "url": "https://example.com/edited_image.png" }
  ]
}
```

---

## 6. Embedding API (`embedding_api`)

Used for converting text into numerical vector representations.

### Request Format
Follows the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings/create).

**Key Fields:**
- `model` (string, required): ID of the model to use.
- `input` (string or array, required): Input text to embed.

### Response Format
Standard OpenAI Embedding response.

**Example Request:**
```json
{
  "model": "text-embedding-3-small",
  "input": "This is a test sentence."
}
```

**Example Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0123, -0.0456, 0.0789, ...]
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

---

## 7. Agent CLI (`agent_cli`)


Used for agentic systems that operate via a command-line interface or a specialized API (e.g., Aider, Harbor).

### Request Parameters
The parameters for `agent_cli` are an extension of the `llm_api` parameters, defined in `AgentCLIParams`. Key fields include:
- `provider` (string, required): The agent framework name (e.g., `aider`, `goose`).
- `base_url` (string): Endpoint for the agent service.
- `model` (string): Underlying model ID.
- Other parameters from `llm_api` like `api_key` and `env_file` are also supported.


**Example Request (Internal ASQI Protocol):**
```json
{
  "base_url": "http://aider-service:8080",
  "model": "gpt-4",
  "provider": "aider"
}
```

**Example Response (ContainerOutput):**
```json
{
  "results": {
    "success": true,
    "score": 0.85,
    "output": "Successfully refactored the module."
  }
}
```

---

## 8. Generic REST API (`rest_api`)

A flexible type for custom integrations that do not follow the OpenAI spec.

### Request Format
Custom JSON payload defined by the specific integration.

### Response Format
Expected to return a `ContainerOutput` structure (see below).

**Key Fields in `results`:**
- `success` (boolean, required): Whether the operation completed without errors.
- `score` (float, optional): The primary metric score.
- `error` (string, optional): Error message if `success` is false.

**Example Request:**
```json
{
  "params": {
    "custom_field": "value",
    "threshold": 0.5
  }
}
```

**Example Response:**
```json
{
  "results": {
    "success": true,
    "score": 0.92,
    "custom_metrics": { "accuracy": 0.95 }
  }
}
```

---

## 9. Object Detection API (Internal)

Used by `object_detection_standard` and `hf_vision_tester`. This API acts as a wrapper for various providers, including Hugging Face and Roboflow.

### Request (`POST /detect`)
- **Body**: `multipart/form-data`
- **Fields**:
    - `image`: The binary image file.
    - `model`: Model ID string.
    - `provider` (string, optional): Provider name (e.g., `huggingface`, `roboflow`).
    - `api_key` (string, optional): API key for the provider.
    - `base_url` (string, optional): Base URL for the provider API.

### Response
```json
{
  "detections": [
    {
      "xyxy": [100, 150, 300, 400],
      "confidence": 0.92,
      "class_id": 1,
      "class_name": "person"
    }
  ]
}
```

**Example Request (Multipart/Form-Data):**
- `image`: [binary file]
- `model`: "facebook/detr-resnet-50"
- `provider`: "huggingface"
- `api_key`: "hf_..."
- `base_url`: "https://api-inference.huggingface.co"

---

## 10. Hugging Face Inference API (Object Detection)

When using the `huggingface` provider with the Internal Object Detection API, the underlying request to Hugging Face follows this format.

### Request (`POST /models/{model_id}`)
- **Headers**: `Authorization: Bearer {hf_token}`
- **Body**: Binary image data or JSON with base64.
- **Parameters** (optional):
    - `threshold` (float): Minimum confidence score for detections.

### Response Format
```json
[
  {
    "score": 0.944,
    "label": "remote",
    "box": {
      "xmin": 40,
      "ymin": 70,
      "xmax": 175,
      "ymax": 150
    }
  }
]
```

**Example Response:**
```json
[
  {
    "score": 0.998,
    "label": "couch",
    "box": { "xmin": 0, "ymin": 1, "xmax": 638, "ymax": 473 }
  }
]
```




