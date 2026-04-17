import uuid
from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    StringConstraints,
    model_validator,
)

# This is necessary because pydantic prefers Annotated types outside classes
IDsStringPattern = Annotated[str, StringConstraints(pattern="^[0-9a-z_]{1,32}$")]

# ----------------------------------------------------------------------------
# HuggingFace Feature Types
# ----------------------------------------------------------------------------

# HuggingFace dataset dtypes as a Literal for better JSON schema support
# Complete list from: https://huggingface.co/docs/datasets/v4.4.2/en/package_reference/main_classes#datasets.Value
HFDtype = Literal[
    "null",
    "bool",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float",  # alias for float32
    "float64",
    "double",  # alias for float64
    "time32[s]",
    "time32[ms]",
    "time64[us]",
    "time64[ns]",
    "timestamp[s]",
    "timestamp[ms]",
    "timestamp[us]",
    "timestamp[ns]",
    "date32",
    "date64",
    "duration[s]",
    "duration[ms]",
    "duration[us]",
    "duration[ns]",
    "binary",
    "large_binary",
    "binary_view",
    "string",
    "large_string",
    "string_view",
]


class ValueFeature(BaseModel):
    """
    Corresponds to HuggingFace's Value feature type.
    Represents a scalar value feature (string, int64, float32, bool, etc.)
    """

    feature_type: Literal["Value"] = Field(default="Value", description="Feature type discriminator")
    name: str = Field(..., description="The name of the feature")
    dtype: HFDtype = Field(
        ...,
        description="The data type of the feature. "
        "Common types: 'string', 'int64', 'int32', 'float64', 'float32', 'bool'. "
        "See: https://huggingface.co/docs/datasets/about_dataset_features",
    )
    required: bool = Field(
        default=False,
        description="Whether this feature is required in the dataset. "
        "If False, the feature may be absent or contain null values.",
    )
    description: str | None = Field(default=None, description="Description of the feature - data type, purpose etc.")


class ListFeature(BaseModel):
    """Corresponds to HuggingFace's List/Sequence feature type."""

    feature_type: Literal["List"] = Field(default="List", description="Feature type discriminator")
    name: str = Field(..., description="The name of the feature")
    feature: HFDtype | Literal["Image", "Audio", "Video", "ClassLabel", "Dict", "List"] = Field(
        ...,
        description=(
            "List element type. Can be: (1) scalar dtype string ('string', 'int32', etc.) for "
            "List(Value(dtype)), or (2) simple feature type name ('Image', 'Audio', 'Video', "
            "'ClassLabel', 'Dict', 'List') for List(FeatureType()). For complex nested structures "
            "with custom parameters, define nested ListFeature or DictFeature objects."
        ),
    )
    length: int = Field(
        default=-1,
        description="List length constraint: -1 for variable-length lists, >=0 for fixed-length sequences.",
    )
    required: bool = Field(
        default=False,
        description="Whether this feature is required in the dataset. "
        "If False, the feature may be absent or contain null values.",
    )
    description: str | None = Field(default=None, description="Description of the list feature")

    @model_validator(mode="after")
    def _validate_length(self) -> "ListFeature":
        """Ensure length is >= -1."""
        if self.length < -1:
            raise ValueError(f"List length must be >= -1 (got {self.length}). Use -1 for variable-length lists.")
        return self


class DictFeature(BaseModel):
    """Corresponds to HuggingFace's "Python dict" feature type."""

    feature_type: Literal["Dict"] = Field(
        default="Dict",
        description="Feature type discriminator",
    )
    name: str = Field(..., description="The name of the feature")
    fields: list["HFFeature"] = Field(
        ...,
        min_length=1,
        description="Named fields within the dict. Each field can be any HFFeature type (Value, List, Dict, etc.)",
    )
    required: bool = Field(
        default=False,
        description="Whether this feature is required in the dataset. "
        "If False, the feature may be absent or contain null values.",
    )
    description: str | None = Field(default=None, description="Description of the dict feature")


class ClassLabelFeature(BaseModel):
    """Corresponds to HuggingFace's ClassLabel feature type. Represents categorical data with named categories."""

    feature_type: Literal["ClassLabel"] = Field(default="ClassLabel", description="Feature type discriminator")
    name: str = Field(..., description="The name of the feature")
    names: list[str] = Field(
        ...,
        min_length=1,
        description="Category names (e.g., ['positive', 'negative', 'neutral'])",
    )
    required: bool = Field(
        default=False,
        description="Whether this feature is required in the dataset. "
        "If False, the feature may be absent or contain null values.",
    )
    description: str | None = Field(default=None, description="Description of the classification categories")


class ImageFeature(BaseModel):
    """Corresponds to HuggingFace's Image feature type."""

    feature_type: Literal["Image"] = Field(default="Image", description="Feature type discriminator")
    name: str = Field(..., description="The name of the feature")
    required: bool = Field(
        default=False,
        description="Whether this feature is required in the dataset. "
        "If False, the feature may be absent or contain null values.",
    )
    description: str | None = Field(default=None, description="Description of the image feature")


class AudioFeature(BaseModel):
    """Corresponds to HuggingFace's Audio feature type."""

    feature_type: Literal["Audio"] = Field(default="Audio", description="Feature type discriminator")
    name: str = Field(..., description="The name of the feature")
    required: bool = Field(
        default=False,
        description="Whether this feature is required in the dataset. "
        "If False, the feature may be absent or contain null values.",
    )
    description: str | None = Field(default=None, description="Description of the audio feature")


class VideoFeature(BaseModel):
    """Corresponds to HuggingFace's Video feature type."""

    feature_type: Literal["Video"] = Field(default="Video", description="Feature type discriminator")
    name: str = Field(..., description="The name of the feature")
    required: bool = Field(
        default=False,
        description="Whether this feature is required in the dataset. "
        "If False, the feature may be absent or contain null values.",
    )
    description: str | None = Field(default=None, description="Description of the video feature")


class DatasetFeature(BaseModel):
    """Defines a feature/column within a dataset.

    The dtype field uses HuggingFace datasets dtype values.
    Common types: 'string', 'int64', 'float32', 'bool'.
    """

    name: str = Field(
        ...,
        description="The name of the feature.",
    )
    dtype: HFDtype = Field(
        default=...,
        description="The data type of the feature. "
        "Common types: 'string', 'int64', 'int32', 'float64', 'float32', 'bool'. "
        "See: https://huggingface.co/docs/datasets/about_dataset_features",
    )
    required: bool = Field(
        default=False,
        description="Whether this feature is required in the dataset. "
        "If False, the feature may be absent or contain null values.",
    )
    description: str | None = Field(default=None, description="Description of the feature - data type, purpose etc.")


# Union type for all HuggingFace feature types
HFFeature = Annotated[
    ValueFeature | ListFeature | DictFeature | ClassLabelFeature | ImageFeature | AudioFeature | VideoFeature,
    Field(discriminator="feature_type"),
]

# Rebuild models with forward references to resolve recursive HFFeature references
ListFeature.model_rebuild()
DictFeature.model_rebuild()


# ----------------------------------------------------------------------------
# Schemas for manifest.yaml (Embedded in Test Containers)
# ----------------------------------------------------------------------------


class SystemInput(BaseModel):
    """Defines a system input that the container requires."""

    name: str = Field(
        ...,
        description="The system input name, e.g., 'system_under_test', 'simulator_system', 'evaluator_system'.",
    )
    type: str | list[str] = Field(
        ...,
        description=(
            "The system type(s) accepted. Can be a single string (e.g., 'llm_api') "
            "or a list of strings (e.g., ['llm_api', 'vlm_api']) for containers "
            "that support multiple system types. Valid types: 'llm_api', 'rest_api', "
            "'rag_api', 'image_generation_api', 'image_editing_api', 'vlm_api', "
            "'agent_cli'."
        ),
    )
    required: bool = Field(True, description="Whether this system input is required.")
    description: str | None = Field(None, description="Description of the system's role in the test.")


class InputParameter(BaseModel):
    """
    Defines a parameter that can be passed to the test container.
    Supports simple types (string, integer, float, boolean) as well as list, object, enum.
    """

    name: str = Field(..., description="Parameter name")
    type: Literal["string", "integer", "float", "boolean", "list", "object", "enum"] = Field(
        ..., description="Parameter type"
    )
    required: bool = Field(default=False, description="Whether this parameter is required")
    description: str | None = Field(default=None, description="Human-readable description of the parameter")

    items: Union[Literal["string", "integer", "float", "boolean", "object", "enum"], "InputParameter"] | None = Field(
        default=None,
        description="For type='list': defines the schema for list items. "
        "Can be a simple type name string (e.g., 'string', 'integer') for basic typed lists, "
        "or a full InputParameter object for complex items (enums, objects, nested lists). ",
    )

    properties: list["InputParameter"] | None = Field(
        default=None,
        description="For type='object': list of nested parameter definitions. "
        "Each property is a full InputParameter with its own name, type, and constraints. ",
    )

    choices: list[str | int | float] | None = Field(
        default=None,
        description="For type='enum': list of allowed values. Required when type='enum'.",
    )

    default: str | int | float | bool | list | dict | None = Field(
        default=None,
        description="Default value for this parameter when not provided by user",
    )

    ui_config: dict[str, Any] | None = Field(
        default=None,
        description="Optional UI configuration hints as arbitrary key-value pairs.",
    )

    @model_validator(mode="after")
    def validate_rich_fields(self) -> "InputParameter":
        """Validate that rich-type fields are only used with appropriate types."""

        if self.items is not None and self.type != "list":
            raise ValueError(f"'items' field can only be specified when type='list' (got type='{self.type}')")

        # properties only valid for type="object"
        if self.properties is not None and self.type != "object":
            raise ValueError(f"'properties' field can only be specified when type='object' (got type='{self.type}')")

        # choices only valid for type="enum"
        if self.choices is not None and self.type != "enum":
            raise ValueError(f"'choices' field can only be specified when type='enum' (got type='{self.type}')")

        # type="enum" requires choices
        if self.type == "enum" and self.choices is None:
            raise ValueError("type='enum' requires 'choices' field to be specified")

        # Validate default value is in choices for enums
        if self.type == "enum" and self.default is not None and self.choices is not None:
            if self.default not in self.choices:
                raise ValueError(f"Default value '{self.default}' must be one of the allowed choices: {self.choices}")

        return self


# Rebuild model to resolve forward references for self-referential fields
InputParameter.model_rebuild()


class OutputMetric(BaseModel):
    """Defines a metric that will be present in the test container's output."""

    name: str
    type: Literal["string", "integer", "float", "boolean", "list", "object"]
    description: str | None = None


class OutputArtifact(BaseModel):
    """Defines a file artifact generated by the test container."""

    name: str
    path: str
    description: str | None = None


class EnvironmentVariable(BaseModel):
    """Defines an environment variable required by the test container."""

    name: str = Field(
        ...,
        description="Environment variable name (e.g., 'OPENAI_API_KEY', 'HF_TOKEN').",
    )
    required: bool = Field(
        True,
        description="Whether this environment variable is mandatory for execution.",
    )
    description: str | None = Field(None, description="Explanation of what this variable is used for.")


class DatasetType(StrEnum):
    """Supported dataset types for InputDataset."""

    HUGGINGFACE = "huggingface"
    PDF = "pdf"
    TXT = "txt"


class InputDataset(BaseModel):
    """Defines a dataset input that the container requires.

    Supported dataset types:
    - 'huggingface': HuggingFace datasets (requires features to be defined)
    - 'pdf': PDF documents
    - 'txt': Plain text files

    Supports "either/or" relationships by accepting multiple types:
    - type: "pdf" - Single type
    - type: ["pdf", "txt"] - Either PDF or TXT
    - type: ["huggingface", "pdf", "txt"] - Any of these formats
    """

    name: str = Field(
        ...,
        description="The dataset name, e.g., 'evaluation_data', 'test_prompts'.",
    )
    required: bool = Field(default=True, description="Whether this dataset is mandatory for execution.")
    type: DatasetType | list[DatasetType] = Field(
        default=...,
        description="The dataset type(s): single type or list of accepted types. "
        "Examples: 'huggingface', ['pdf', 'txt'], or ['huggingface', 'pdf', 'txt'].",
    )
    description: str | None = Field(default=None, description="Description of the dataset's role in the test.")
    features: list[DatasetFeature | HFFeature] | None = Field(
        default=None,
        description="List of required features within a HuggingFace dataset. "
        "Supports both simple scalar features and complex feature types.",
    )

    @model_validator(mode="after")
    def _validate_features_for_huggingface(self) -> "InputDataset":
        """Ensure HuggingFace datasets have features defined whenever huggingface is an accepted type."""
        types = self.type if isinstance(self.type, list) else [self.type]
        if DatasetType.HUGGINGFACE in types and not self.features:
            raise ValueError(
                "Features must be defined when 'huggingface' is an accepted dataset type. "
                "Specify the expected feature names and dtypes in the features list."
            )
        return self


class OutputReports(BaseModel):
    """Defines a report that will be generated by the test container."""

    name: str = Field(
        ...,
        description="The name of the report ('detailed_report', 'summary_report', ...).",
    )
    type: Literal["pdf", "html"] = Field(..., description="The report file format ('pdf' or 'html').")
    description: str | None = Field(None, description="Short description of the report content.")


# This is a slightly relaxed version of input dataset, if provided could be used for validation
class OutputDataset(BaseModel):
    """Defines a dataset output that will be generated by the test container."""

    name: str = Field(
        ...,
        description="The name of this output dataset (e.g., 'augmented_rag_data')",
    )
    type: DatasetType = Field(..., description="Type of dataset: huggingface, pdf, or txt")
    description: str | None = Field(
        default=None,
        description="Description of the output dataset's purpose and contents",
    )
    features: list[DatasetFeature | HFFeature] | None = Field(
        default=None,
        description="List of required features within a HuggingFace dataset. "
        "Supports both simple scalar features and complex feature types.",
    )


class Manifest(BaseModel):
    """Schema for the manifest.yaml file inside a test container."""

    name: str = Field(..., description="The canonical name for the test framework.")
    version: str
    description: str | None = None
    host_access: bool = Field(
        False,
        description="Whether the container requires host access (e.g., for Docker-in-Docker).",
    )
    input_systems: list[SystemInput] = Field(
        [],
        description=(
            "Systems required as input. Can be empty for containers that don't require systems "
            "(e.g., pure data transformation)."
        ),
    )
    input_schema: list[InputParameter] = Field(
        [], description="Defines the schema for the user-provided 'params' object."
    )
    input_datasets: list[InputDataset] = Field(
        [],
        description="Defines the schema for user-provided input datasets.",
    )
    output_metrics: list[str] | list[OutputMetric] = Field(
        [],
        description=(
            "Defines expected high-level metrics in the final JSON output. Can be a simple list "
            "of strings or detailed metric definitions."
        ),
    )
    output_artifacts: list[OutputArtifact] | None = None
    environment_variables: list[EnvironmentVariable] = Field(
        [],
        description="Environment variables required by this test container. Used for validation and documentation.",
    )
    output_reports: list[OutputReports] = Field(
        default_factory=list,
        description="Defines the reports generated by the test container.",
    )
    output_datasets: list[OutputDataset] = Field(
        default_factory=list,
        description="Defines the datasets generated by the container.",
    )


# ----------------------------------------------------------------------------
# Schema for systems.yaml (User-provided)
# ----------------------------------------------------------------------------


class SystemDefinition(BaseModel):
    """Base system definition."""

    description: str | None = Field(
        None,
        description="Description of the system being evaluated.",
    )

    provider: str | None = Field(
        None,
        description=(
            "Name of the provider of the system, either 'custom' for internal systems or "
            "'openai, aws-bedrock...' for external systems."
        ),
    )


# LLM API system


class ThinkingParams(BaseModel):
    """Extended reasoning configuration for models that support it (e.g., Claude Opus 4.6 etc.)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["enabled", "adaptive"] = Field(
        ...,
        description=(
            "Thinking mode: 'enabled' activates extended reasoning (requires budget_tokens); "
            "'adaptive' lets the model decide when to think."
        ),
    )
    budget_tokens: int | None = Field(
        None,
        ge=1,
        description="Maximum token budget for the thinking process. Required when type='enabled'.",
    )

    @model_validator(mode="after")
    def _validate_budget_tokens(self) -> "ThinkingParams":
        if self.type == "enabled" and self.budget_tokens is None:
            raise ValueError("budget_tokens is required when type='enabled'")
        if self.type == "adaptive" and self.budget_tokens is not None:
            raise ValueError("budget_tokens must not be set when type='adaptive'")
        return self


class LLMAPIParams(BaseModel):
    """Parameters for the LLM API systems."""

    base_url: str = Field(
        ...,
        description="Base URL for the OpenAI-compatible API (e.g., 'http://localhost:4000/v1', 'https://api.openai.com/v1')",
    )
    model: str = Field(
        ...,
        description="Model name to use with the API",
    )
    env_file: str | None = Field(
        None,
        description="Path to .env file containing environment variables for authentication",
    )
    api_key: str | None = Field(
        None,
        description="Direct API key for authentication (alternative to env_file)",
    )
    thinking: ThinkingParams | None = Field(
        None,
        description=(
            "Optional thinking/extended reasoning configuration. Only applicable to models that support this feature."
        ),
    )
    reasoning_effort: str | None = Field(
        None,
        description=(
            "Reasoning effort level for models supporting it (e.g. 'low', 'medium', 'high', "
            "'none'); allowed values depend on the model and provider."
        ),
    )


class VLMAPIParams(LLMAPIParams):
    """Parameters for Vision Language Model API systems."""

    supports_vision: Literal[True] = Field(
        True,
        description="Whether the VLM system supports vision. Forced to True for vlm_api type.",
    )


class AgentCLIParams(LLMAPIParams):
    """Parameters for Agent CLI systems."""

    provider: str = Field(
        ...,
        description="The agent CLI provider name (e.g., 'aider', 'cline', 'goose', 'codex')",
    )


class LLMAPIConfig(SystemDefinition):
    """Configuration for LLM API systems."""

    type: Literal["llm_api"] = Field(
        ...,
        description="LLM API system: llm_api",
    )
    params: LLMAPIParams = Field(
        ...,
        description=("Parameters for the LLM API system (base URL, model, API key, env file)."),
    )


# Embedding API system


class EmbeddingAPIConfig(SystemDefinition):
    """Configuration for Embedding API systems."""

    type: Literal["embedding_api"] = Field(
        ...,
        description="Embedding API system: embedding_api",
    )
    params: LLMAPIParams = Field(
        ...,
        description=("Parameters for the Embedding API system (base URL, model, API key, env file)."),
    )


# Custom system


class CustomParams(BaseModel):
    """Parameters for custom systems (e.g., SAP analytics, proprietary APIs)."""

    base_url: str = Field(
        ...,
        description="Base URL of the custom system.",
    )
    api_key: str | None = Field(
        default=None,
        description="Optional API key for authentication.",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific parameters for the custom system. "
        "Test containers are responsible for validating and interpreting these fields.",
    )


class CustomConfig(SystemDefinition):
    """Configuration for custom systems (e.g., SAP analytics, proprietary APIs)."""

    type: Literal["custom"] = Field(
        "custom",
        description="Custom system type",
    )
    params: CustomParams = Field(
        ...,
        description="Parameters specific to the custom system (base_url, api_key, and domain-specific extras).",
    )


# RAG API system


class RAGAPIConfig(SystemDefinition):
    """Configuration for RAG API systems."""

    type: Literal["rag_api"] = Field(
        ...,
        description="RAG API system: rag_api",
    )
    params: LLMAPIParams = Field(
        ...,
        description=("Parameters for the RAG API system (base URL, model, API key, env file)."),
    )


# Image Generation API system


class ImageGenerationAPIConfig(SystemDefinition):
    """Configuration for Image Generation API systems."""

    type: Literal["image_generation_api"] = Field(
        ...,
        description="Image Generation API system: image_generation_api",
    )
    params: LLMAPIParams = Field(
        ...,
        description=("Parameters for the Image Generation API system (base URL, model, API key, env file)."),
    )


# Image Editing API system


class ImageEditingAPIConfig(SystemDefinition):
    """Configuration for Image Editing API systems."""

    type: Literal["image_editing_api"] = Field(
        ...,
        description="Image Editing API system: image_editing_api",
    )
    params: LLMAPIParams = Field(
        ...,
        description=("Parameters for the Image Editing API system (base URL, model, API key, env file)."),
    )


# VLM API system


class VLMAPIConfig(SystemDefinition):
    """Configuration for Vision Language Model API systems."""

    type: Literal["vlm_api"] = Field(
        ...,
        description="Vision Language Model API system: vlm_api",
    )
    params: VLMAPIParams = Field(
        ...,
        description=("Parameters for the VLM API system (base URL, model, API key, env file, vision support)."),
    )


# Agentic CLI system


class AgentCLIConfig(SystemDefinition):
    """Configuration for Agent CLI systems.

    Agent CLI systems are autonomous agents and coding frameworks
    that can be invoked via CLI or API (e.g., Codex, Qwen Coder,
    OpenCode, Goose, Aider, Cline).
    """

    type: Literal["agent_cli"] = Field(
        "agent_cli",
        description="Agent CLI system type",
    )
    params: AgentCLIParams = Field(
        ...,
        description="Parameters specific to the Agent CLI system (provider, model, api_key, base_url)",
    )


# Generic system


class GenericSystemConfig(SystemDefinition):
    """Generic system configuration for system types without specific validation.

    This allows backward compatibility and support for system types that don't have dedicated config classes yet.
    """

    type: str = Field(
        ...,
        description="System type, e.g., 'rest_api', 'custom_api', etc.",
    )
    params: dict[str, Any] = Field(
        ...,
        description="Parameters specific to the system type.",
    )


SystemConfig = (
    LLMAPIConfig
    | RAGAPIConfig
    | ImageGenerationAPIConfig
    | ImageEditingAPIConfig
    | VLMAPIConfig
    | AgentCLIConfig
    | EmbeddingAPIConfig
    | CustomConfig
    | GenericSystemConfig
)


class SystemsConfig(BaseModel):
    """Schema for the top-level systems configuration file.

    Extension Guide:
        1. Create a new XXXConfig class inheriting from SystemDefinition. e.g, RESTAPIConfig
        2. Create a new XXXParam class for the parameters of the system. e.g. RESTAPIParams
        2. Add the new system definition (XXXConfig) to the SystemConfig union type
            e.g. SystemConfig = LLMAPIConfig | XXXConfig | ... | GenericSystemConfig

    """

    systems: dict[str, SystemConfig] = Field(..., description="Dictionary of system definitions.")


# ----------------------------------------------------------------------------
# Schema for Dataset Registry (User-provided)
# ----------------------------------------------------------------------------


class DatasetLoaderParams(BaseModel):
    """Parameters for loading a HuggingFace dataset.

    Supports two mutually exclusive modes:
    - Hub mode: Load from HuggingFace Hub using `hub_path`
    - Local mode: Load from local files using `builder_name` with `data_dir` or `data_files`
    """

    hub_path: str | None = Field(
        default=None,
        description="HuggingFace Hub dataset path (e.g., 'detection-datasets/coco'). "
        "Mutually exclusive with builder_name.",
    )
    name: str | None = Field(
        default=None,
        description="Configuration name for multi-config Hub datasets (e.g., 'default', '2017').",
    )
    split: str | None = Field(
        default=None,
        description="Dataset split to load (e.g., 'train', 'validation', 'test'). "
        "Defaults to 'train' if not specified.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Allow execution of remote code from HuggingFace Hub datasets. "
        "Only set to True for trusted datasets.",
    )
    builder_name: (
        Literal[
            "json", "csv", "parquet", "arrow", "text", "xml", "webdataset", "imagefolder", "audiofolder", "videofolder"
        ]
        | None
    ) = Field(
        default=None,
        description=(
            "The dataset builder name for local files. Passed to datasets.load_dataset() as the "
            "path argument. Mutually exclusive with hub_path."
        ),
    )
    data_dir: str | None = Field(
        default=None,
        description="Directory containing dataset files, relative to the input mount.",
    )
    data_files: str | list[str] | None = Field(
        default=None,
        description="Single file path or list of file paths, relative to the input mount.",
    )
    revision: str | None = Field(
        default=None,
        description="Git revision (commit hash, tag, or branch) for HuggingFace Hub datasets. "
        "Recommended for reproducibility when using hub_path.",
    )
    streaming: bool = Field(
        default=False,
        description="Enable streaming mode to avoid loading entire dataset into memory. "
        "Returns IterableDataset instead of Dataset. Recommended for large datasets.",
    )
    token: str | None = Field(
        default=None,
        description="HuggingFace token for accessing private datasets. "
        "If not provided, falls back to HF_TOKEN environment variable.",
    )

    @model_validator(mode="after")
    def _validate_loader_params(self) -> "DatasetLoaderParams":
        is_hub = self.hub_path is not None
        is_local = self.builder_name is not None

        if is_hub and is_local:
            raise ValueError(
                "Cannot specify both 'hub_path' and 'builder_name'. "
                "Use 'hub_path' for HuggingFace Hub datasets or 'builder_name' for local files."
            )
        if not is_hub and not is_local:
            raise ValueError("Must specify either 'hub_path' (for Hub datasets) or 'builder_name' (for local files).")
        if is_hub and (self.data_dir or self.data_files):
            raise ValueError(
                "'data_dir' and 'data_files' are not used with 'hub_path'. "
                "These options are only for local file loading with 'builder_name'."
            )
        if is_local and not (self.data_dir or self.data_files):
            raise ValueError("Local mode requires either 'data_dir' or 'data_files' to specify the data location.")
        if is_local and self.data_dir and self.data_files:
            raise ValueError("Cannot specify both 'data_dir' and 'data_files'. Use one or the other.")
        return self


class LabelFieldDefinition(BaseModel):
    """Defines a rich field definition for classification tasks within a label map."""

    field_description: str
    field_data_type: str
    field_enum_values: list[str] | None = None
    field_multi_label: bool = False


LabelMap = dict[int, str] | dict[str, str | LabelFieldDefinition]


class HFDatasetDefinition(BaseModel):
    """Defines a reusable HuggingFace dataset that can be referenced by name in test suites and generation jobs."""

    type: Literal["huggingface"] = Field(
        ...,
        description="Dataset type identifier for HuggingFace datasets.",
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the dataset's purpose and contents.",
    )
    loader_params: DatasetLoaderParams = Field(
        ...,
        description="Configuration for loading the dataset from files or directories.",
    )
    mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional mapping from existing dataset column names to their required names in the "
            "manifest (current_name: manifest_name)."
        ),
    )
    label_map: LabelMap | None = Field(
        default=None,
        description="Optional label mapping for datasets. Supports simple ID-to-name mappings "
        "(e.g., {0: 'person', 1: 'car'}) and rich field definitions using LabelFieldDefinition "
        "for classification tasks (e.g., {'sentiment': LabelFieldDefinition(field_description='...', "
        "field_data_type='string', field_enum_values=['positive', 'negative'])}).",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for categorization (e.g., ['evaluation', 'en']).",
    )


class FileDatasetBase(BaseModel):
    """Base class for file-based dataset definitions."""

    description: str | None = Field(
        None,
        description="Human-readable description of the dataset's purpose and contents.",
    )
    file_path: str = Field(
        ...,
        description=(
            "Path to the input file, relative to the input mount. Can be a local path or remote URL "
            "depending on container support."
        ),
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Optional tags for categorization (e.g., ['evaluation', 'en']).",
    )


class PDFDatasetDefinition(FileDatasetBase):
    """Defines a PDF document dataset that can be referenced by name in test suites and generation jobs."""

    type: Literal["pdf"] = Field(
        ...,
        description="Dataset type identifier for PDF documents.",
    )


class TXTDatasetDefinition(FileDatasetBase):
    """Defines a plain text file dataset that can be referenced by name in test suites and generation jobs."""

    type: Literal["txt"] = Field(
        ...,
        description="Dataset type identifier for plain text files.",
    )


DatasetDefinition = Annotated[
    HFDatasetDefinition | PDFDatasetDefinition | TXTDatasetDefinition,
    Field(discriminator="type"),
]


class DatasetsConfig(BaseModel):
    """Schema for datasets configuration file.

    Datasets are defined in a dictionary where keys are dataset names.
    Each dataset must specify a 'type' field that determines its schema:
    - 'huggingface': HuggingFace datasets with loader_params
    - 'pdf': PDF document datasets with file_path
    - 'txt': Plain text file datasets with file_path
    """

    datasets: dict[str, DatasetDefinition] = Field(
        ...,
        description=(
            "Dataset definitions keyed by name. Each entry is HFDatasetDefinition, "
            "PDFDatasetDefinition, or TXTDatasetDefinition (discriminated by 'type')."
        ),
    )


# ----------------------------------------------------------------------------
# Schema for Test Suite (User-provided)
# ----------------------------------------------------------------------------


class TestDefinitionBase(BaseModel):
    """Base class for test configuration fields shared between TestDefinition and TestSuiteDefault."""

    systems_under_test: list[str] | None = Field(
        None,
        description=(
            "System names from systems.yaml to run this test against. Can be inherited from test_suite_default."
        ),
    )
    systems: dict[str, str] | None = Field(
        None,
        description="Optional additional systems for the test (e.g., simulator_system, evaluator_system).",
    )
    tags: list[str] | None = Field(None, description="Optional tags for filtering and reporting.")
    params: dict[str, Any] | None = Field(
        None, description="Parameters to be passed to the test container's entrypoint."
    )
    input_datasets: dict[str, str] | None = Field(
        None,
        description=(
            "Input dataset names mapped to registry references. Values must be names from the "
            "datasets registry config (--datasets-config)."
        ),
    )
    volumes: dict[str, Any] | None = Field(None, description="Optional input/output mounts.")
    env_file: str | None = Field(
        None,
        description=("Path to .env file with variables for this test's container (e.g. '.env', 'test.env')."),
    )
    environment: dict[str, str] | None = Field(
        None,
        description=(
            "Environment variables for the test container. Supports interpolation "
            "(e.g. {'OPENAI_API_KEY': '${OPENAI_API_KEY}'})."
        ),
    )


class TestDefinition(TestDefinitionBase):
    """A single test to be executed."""

    id: IDsStringPattern = Field(
        ...,
        description=(
            "A unique, human-readable ID (up to 32 characters) for this test instance. Can "
            "include lowercase letters (a-z), digits (0-9) and underscore (_)."
        ),
    )
    name: str | None = Field(
        None,
        description="A descriptive, human-friendly name for this test instance.",
    )
    description: str | None = Field(
        None,
        description="A short summary of the purpose of the test and what it aims to validate.",
    )
    image: str = Field(
        ...,
        description="The Docker image to run for this test, e.g., 'my-registry/garak:latest'.",
    )


class TestSuiteDefault(TestDefinitionBase):
    """Default values that apply to all tests in the suite unless overridden."""

    pass


class SuiteConfig(BaseModel):
    """Schema for the top-level Test Suite configuration file."""

    suite_name: str = Field(..., description="Name of this test suite.")
    test_suite_default: TestSuiteDefault | None = Field(
        None,
        description="Default values that apply to all tests in the suite unless overridden",
    )
    description: str | None = Field(
        None,
        description="A short summary of the test suite and what it aims to evaluate.",
    )
    test_suite: list[TestDefinition] = Field(..., description="List of individual focused tests.")


# ----------------------------------------------------------------------------
# Schema for Score Card (User-provided)
# ----------------------------------------------------------------------------


class ScoreCardFilter(BaseModel):
    """Defines which test results an indicator applies to."""

    test_id: str = Field(
        ...,
        description="Test id to filter by, e.g., 'run_mock_on_compatible_sut'",
    )
    target_system_type: str | list[str] | None = Field(
        None,
        description=(
            "Optional: filter by system type(s). Single type (e.g. 'llm_api') or list "
            "(e.g. ['llm_api', 'vlm_api']). If omitted, applies to all system types."
        ),
    )


class AssessmentRule(BaseModel):
    """Individual assessment outcome with condition."""

    outcome: str = Field(..., description="Assessment outcome, e.g., 'PASS', 'FAIL', 'A', 'F'")
    condition: Literal[
        "equal_to",
        "greater_than",
        "less_than",
        "greater_equal",
        "less_equal",
        "all_true",
        "any_false",
        "count_equals",
    ] = Field(..., description="Condition to evaluate against the metric value")
    threshold: int | float | bool | None = Field(None, description="Threshold value for comparison conditions")
    description: str | None = Field(None, description="Human-readable description for this assessment outcome")


class MetricExpression(BaseModel):
    """Expression-based metric evaluation with explicit value declarations."""

    expression: str = Field(
        ...,
        description=("Mathematical formula combining metrics. Variable names must match keys in 'values'."),
    )
    values: dict[str, str] = Field(
        ...,
        description=(
            "Maps expression variables to metric paths. Keys appear in the expression; values are "
            "paths into test results."
        ),
    )


class ScoreCardIndicator(BaseModel):
    """Individual score card indicator with filtering and assessment."""

    id: IDsStringPattern = Field(
        ...,
        description=(
            "A unique, human-readable ID (up to 32 characters) for this score card indicator. "
            "Can include lowercase letters (a-z), digits (0-9) and underscore (_)."
        ),
    )
    name: str | None = Field(None, description="Human-readable name for this score card indicator")
    apply_to: ScoreCardFilter = Field(
        ...,
        description="Filter criteria for which test results this indicator applies to",
    )
    metric: str | MetricExpression = Field(
        ...,
        description=(
            "Metric to evaluate. Can be either:\n"
            "1. Simple path (string): 'average_answer_relevance' or 'stats.pass_rate'\n"
            "2. Expression object with 'expression' and 'values' fields:\n"
            "   metric:\n"
            "     expression: '0.7 * accuracy + 0.3 * relevance'\n"
            "     values:\n"
            "       accuracy: 'average_answer_accuracy'\n"
            "       relevance: 'average_answer_relevance'\n"
            "   Supports arithmetic operators (+, -, *, /), functions (min, max, avg), and parentheses.\n"
            "   Variable names in expression are mapped to metric paths via the 'values' dict."
        ),
    )
    assessment: list[AssessmentRule] = Field(..., description="List of assessment rules to evaluate against the metric")
    display_reports: list[str] = Field(
        default_factory=list,
        description=("List of report names to include from the test container manifest."),
    )


# ----------------------------------------------------------------------------
# Schema for Score Card with Audit Indicators (User-provided)
# ----------------------------------------------------------------------------


class AuditAssessmentRule(BaseModel):
    """Assessment outcome for audit indicators."""

    outcome: str = Field(..., description="Assessment outcome, e.g., 'A', 'B', 'C', 'PASS', 'FAIL'.")
    description: str | None = Field(
        None,
        description="Human-readable description for this audit outcome",
    )


class AuditScoreCardIndicator(BaseModel):
    """Manual audit indicator.

    Outcome is provided via an external audit responses file.
    """

    id: IDsStringPattern = Field(
        ...,
        description=(
            "A unique, human-readable ID (up to 32 characters) for this audit "
            "indicator. Can include lowercase letters (a-z), digits (0-9) and underscore (_)."
        ),
    )
    type: Literal["audit"] = Field(
        "audit",
        description="Indicator type. Must be 'audit' for manual audit indicators.",
    )
    name: str | None = Field(
        None,
        description="Human-readable name for this audit indicator",
    )
    assessment: list[AuditAssessmentRule] = Field(
        ...,
        description="List of possible audit outcomes (A-E, PASS/FAIL, etc.).",
    )


class ScoreCard(BaseModel):
    """Complete grading score card configuration."""

    score_card_name: str = Field(..., description="Name of the grading score card")
    indicators: list[ScoreCardIndicator | AuditScoreCardIndicator] = Field(
        ...,
        description="List of score card indicators to evaluate (non-audit and audit).",
    )


# ----------------------------------------------------------------------------
# Schema for Audit Response (User-provided)
# ----------------------------------------------------------------------------
class AuditResponse(BaseModel):
    indicator_id: str = Field(..., description="ID of the audit indicator")
    sut_name: str | None = Field(
        None,
        description="Name of the system under test this response applies to. If omitted, applies globally.",
    )
    selected_outcome: str = Field(..., description="Letter grade or label (A-E, PASS/FAIL, etc.).")
    notes: str | None = Field(None, description="Optional free text notes")


class AuditResponses(BaseModel):
    responses: list[AuditResponse]


# ----------------------------------------------------------------------------
# Schema for Data Generation Jobs
# ----------------------------------------------------------------------------
class GenerationJobConfig(BaseModel):
    id: str = Field(..., description="Unique identifier for the generation job")
    systems: dict[str, str] | None = Field(None, description="Mapping of system alias to system identifier")
    name: str = Field(..., description="Human-readable data generation job name")
    image: str = Field(..., description="Container image to run the data generation job")
    tags: list[str] | None = Field(None, description="Optional tags for filtering and reporting.")
    input_datasets: dict[str, str] | None = Field(
        None,
        description=(
            "Input dataset names mapped to registry references. Values must be names from the "
            "datasets registry config (--datasets-config)."
        ),
    )
    params: dict[str, Any] | None = Field(
        None, description="Parameters to be passed to the test container's entrypoint."
    )
    volumes: dict[str, Any] | None = Field(None, description="Optional input/output mounts.")
    env_file: str | None = Field(
        None,
        description=("Path to .env file with variables for this job's container (e.g. '.env', 'test.env')."),
    )
    environment: dict[str, str] | None = Field(
        None,
        description=(
            "Environment variables for the job container. Supports interpolation "
            "(e.g. {'OPENAI_API_KEY': '${OPENAI_API_KEY}'})."
        ),
    )


class DataGenerationConfig(BaseModel):
    """Schema for the data generation configuration manifest."""

    job_name: str = Field(..., description="Name of the data generation job")
    generation_jobs: list[GenerationJobConfig] = Field(..., description="List of generation jobs to execute")


# ----------------------------------------------------------------------------
# Execution Metadata Schemas
# ----------------------------------------------------------------------------


class ExecutionTags(BaseModel):
    """
    Tags for workflow execution tracking.
    """

    parent_id: str = Field(..., description="Parent workflow ID for tracking execution hierarchy")
    job_type: str = Field(..., description="Type of job (e.g., 'test', 'generation')")
    job_id: str = Field(..., description="Unique identifier for this specific job")
    model_config = {"extra": "allow"}


class ExecutionMetadata(BaseModel):
    """
    Metadata structure passed from workflow to test containers.
    """

    tags: ExecutionTags = Field(..., description="Workflow tracking tags for LiteLLM attribution")
    user_id: str | None = Field(None, description="Optional user identifier (maps to OpenAI 'user' parameter)")
    model_config = {"extra": "allow"}


# ----------------------------------------------------------------------------
# Test Case Schemas (One per SUT type)
# ----------------------------------------------------------------------------
#
# Each TestCase class:
#   - Stores data in human-readable, semantic fields (the dataset representation).
#   - Computes API-native request / expected_response as @property accessors — never stored.
#   - Uses lineage_id and scenario for cross-stage traceability.
#   - Keeps SDG-generated or debug _metadata as a private attribute (not serialised).
#   - Provides a from_api_data() factory to ingest from existing API-format data.
#
# Models are standalone (no subclassing between test case types). Semantic families:
#   - LLM / UnansweredLLM / AnsweredLLM — text llm_api (Q / Q+metrics / QA).
#   - RAG / UnansweredRAG / AnsweredRAG / ContextualizedRAG — rag_api message shape.
#   - VLM / UnansweredVLM / AnsweredVLM — vlm_api multimodal messages.
#   - Image generation / editing / OD-aware editing; embedding; object detection (+ answered).
#
# Extension guide:
#   1. Create a new XXXTestCase BaseModel following the pattern below.
#   2. Add it to the TestCase union at the bottom of this section.


def _test_case_pop_metadata_and_lineage(data: dict[str, Any]) -> dict[str, Any]:
    """Strip ``metadata`` and drop explicit ``lineage_id`` when None (so default_factory runs)."""
    metadata = data.pop("metadata", {})
    if data.get("lineage_id") is None:
        data.pop("lineage_id", None)
    return metadata


# ---------------------------------------------------------------------------
# LLM test cases
# ---------------------------------------------------------------------------


class LLMTestCase(BaseModel):
    """Query-only base for text LLM API (`llm_api`) test cases (Q)."""

    model_config = ConfigDict(extra="ignore")

    # --- Identification & Lineage ---
    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'instruction-following', 'summarisation'")

    # --- Human-Readable Request Fields ---
    query: str = Field(..., description="The user turn prompt")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")

    # --- Private Attributes (Internal/Debugging Only) ---
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    # --- Extra Parameters of the System Request ---
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        """Computes the OpenAI-compatible request for llm_api / rag_api."""
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.query})
        return {"messages": messages, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class UnansweredLLMTestCase(BaseModel):
    """LLM test case without a ground-truth reference answer.

    Use with metrics that do not require a reference output, e.g. judge-based
    metrics for bias, toxicity, hallucination, policy compliance.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'instruction-following', 'summarisation'")
    query: str = Field(..., description="The user turn prompt")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.query})
        return {"messages": messages, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class AnsweredLLMTestCase(BaseModel):
    """LLM test case with a required ground-truth reference answer.

    Use with reference-based metrics such as BLEU, ROUGE, BERTScore,
    exact_match, and correctness.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'instruction-following', 'summarisation'")
    query: str = Field(..., description="The user turn prompt")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )
    answer: str = Field(..., description="Reference answer or expected completion")

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.query})
        return {"messages": messages, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def expected_response(self) -> dict[str, Any]:
        """Computes the expected OpenAI response structure."""
        return {"choices": [{"message": {"role": "assistant", "content": self.answer}}]}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        response: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AnsweredLLMTestCase":
        messages = request.get("messages", [])
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        answer = response["choices"][0]["message"]["content"]
        extra_params = {k: v for k, v in request.items() if k != "messages"}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            query=query,
            system_prompt=system_prompt,
            answer=answer,
            metadata=metadata or {},
            extra_params=extra_params,
        )


# ---------------------------------------------------------------------------
# RAG test cases
# ---------------------------------------------------------------------------


class RAGTestCase(BaseModel):
    """Query-only base for RAG API (`rag_api`) test cases (Qc).

    Request shape matches text LLM messages; use this type (or semantic variants)
    for RAG-specific metrics and routing.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'accuracy', 'robustness'")
    query: str = Field(..., description="The user turn prompt")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.query})
        return {"messages": messages, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class UnansweredRAGTestCase(BaseModel):
    """RAG test case without ground-truth references.

    Use with metrics that score without a reference answer or retrieved context,
    e.g. faithfulness, groundedness, helpfulness, retrieval_relevance.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'accuracy', 'robustness'")
    query: str = Field(..., description="The user turn prompt")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.query})
        return {"messages": messages, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class AnsweredRAGTestCase(BaseModel):
    """RAG test case with a ground-truth reference answer.

    Use with reference-based metrics that compare the model answer to
    `answer` (e.g. answer correctness, Ragas context_recall when keyed
    on the reference answer). For ground-truth retrieved document chunks, use
    `ContextualizedRAGTestCase`.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'accuracy', 'robustness'")
    query: str = Field(..., description="The user turn prompt")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )
    answer: str = Field(
        ...,
        description="The ground truth reference answer (use empty string when no reference text)",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.query})
        return {"messages": messages, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def expected_response(self) -> dict[str, Any] | None:
        """Computes the expected RAG API response structure."""
        if not self.answer:
            return None
        return {
            "choices": [
                {
                    "message": {
                        "content": self.answer,
                        "context": {"citations": []},
                    }
                }
            ]
        }

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        response: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AnsweredRAGTestCase":
        messages = request.get("messages", [])
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        msg = response["choices"][0]["message"]
        answer = msg.get("content") or ""
        extra_params = {k: v for k, v in request.items() if k != "messages"}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            query=query,
            system_prompt=system_prompt,
            answer=answer,
            metadata=metadata or {},
            extra_params=extra_params,
        )


class ContextualizedRAGTestCase(BaseModel):
    """RAG test case with ground-truth retrieved document chunks.

    Use with metrics that need reference contexts (e.g. context_precision,
    retrieval correctness, hit@k) without requiring a reference answer on the
    same object.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'accuracy', 'robustness'")
    query: str = Field(..., description="The user turn prompt")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )
    context: list[str] = Field(
        ...,
        description="Ground truth document chunks expected to be retrieved (empty list if none)",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self.query})
        return {"messages": messages, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def expected_response(self) -> dict[str, Any] | None:
        """Computes the expected RAG API response structure."""
        if not self.context:
            return None
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "context": {"citations": [{"retrieved_context": ctx} for ctx in self.context]},
                    }
                }
            ]
        }

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        response: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ContextualizedRAGTestCase":
        messages = request.get("messages", [])
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
        query = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        msg = response["choices"][0]["message"]
        citations = msg.get("context", {}).get("citations", [])
        context = [c["retrieved_context"] for c in citations]
        extra_params = {k: v for k, v in request.items() if k != "messages"}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            query=query,
            system_prompt=system_prompt,
            context=context,
            metadata=metadata or {},
            extra_params=extra_params,
        )


# ---------------------------------------------------------------------------
# VLM test cases
# ---------------------------------------------------------------------------


class VLMTestCase(BaseModel):
    """Query + image(s) base for VLM API (`vlm_api`) test cases (QI)."""

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'scene-description', 'visual-qa'")
    query: str = Field(..., description="The text question about the image(s)")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    images: list[str] = Field(
        ...,
        min_length=1,
        description="List of images as base64-encoded data URIs (e.g. 'data:image/jpeg;base64,...')",
    )
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        """Computes the OpenAI multi-modal request for vlm_api."""
        content: list[dict[str, Any]] = [{"type": "text", "text": self.query}]
        for image_uri in self.images:
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return {"messages": messages, "supports_vision": True, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class UnansweredVLMTestCase(BaseModel):
    """VLM test case without a ground-truth reference answer.

    Use with metrics that do not require a reference output, e.g. judge-based
    metrics for hallucination and policy compliance.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'scene-description', 'visual-qa'")
    query: str = Field(..., description="The text question about the image(s)")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    images: list[str] = Field(
        ...,
        min_length=1,
        description="List of images as base64-encoded data URIs (e.g. 'data:image/jpeg;base64,...')",
    )
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": self.query}]
        for image_uri in self.images:
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return {"messages": messages, "supports_vision": True, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class AnsweredVLMTestCase(BaseModel):
    """VLM test case with a required ground-truth reference answer.

    Use with reference-based metrics such as correctness.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'scene-description', 'visual-qa'")
    query: str = Field(..., description="The text question about the image(s)")
    system_prompt: str | None = Field(None, description="The system instruction (if any)")
    images: list[str] = Field(
        ...,
        min_length=1,
        description="List of images as base64-encoded data URIs (e.g. 'data:image/jpeg;base64,...')",
    )
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra body params, e.g. temperature, max_tokens",
    )
    answer: str = Field(..., description="Reference answer or description")

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": self.query}]
        for image_uri in self.images:
            content.append({"type": "image_url", "image_url": {"url": image_uri}})
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": content})
        return {"messages": messages, "supports_vision": True, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def expected_response(self) -> dict[str, Any]:
        """Computes the expected OpenAI response structure."""
        return {"choices": [{"message": {"role": "assistant", "content": self.answer}}]}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        response: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AnsweredVLMTestCase":
        messages = request.get("messages", [])
        system_prompt = next((m["content"] for m in messages if m["role"] == "system"), None)
        user_msg = next((m for m in reversed(messages) if m["role"] == "user"), {})
        content = user_msg.get("content", [])
        query = next((c["text"] for c in content if c.get("type") == "text"), "")
        images = [c["image_url"]["url"] for c in content if c.get("type") == "image_url"]
        answer = response["choices"][0]["message"]["content"]
        extra_params = {k: v for k, v in request.items() if k not in ("messages", "supports_vision")}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            query=query,
            images=images,
            system_prompt=system_prompt,
            answer=answer,
            metadata=metadata or {},
            extra_params=extra_params,
        )


# ---------------------------------------------------------------------------
# Image generation test cases
# ---------------------------------------------------------------------------


class ImageGenerationTestCase(BaseModel):
    """Request-only base for Image Generation API (`image_generation_api`) test cases.

    Use this type (or `UnansweredImageGenerationTestCase`) when no reference output
    image is provided. For a reference image for metrics (e.g. FID-style comparisons),
    use `AnsweredImageGenerationTestCase`.
    """

    model_config = ConfigDict(extra="ignore")

    # --- Identification & Lineage ---
    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'text-to-image', 'style-transfer'")

    # --- Human-Readable Request Fields ---
    prompt: str = Field(..., description="The text description of the desired image")
    size: str | None = Field(None, description="Image dimensions, e.g. '1024x1024'")
    n: int = Field(default=1, description="Number of images to generate")

    # --- Private Attributes (Internal/Debugging Only) ---
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    # --- Extra Parameters of the System Request ---
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Extra body params, e.g. model, style")

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        """Computes the OpenAI Image Generation API request."""
        req: dict[str, Any] = {"prompt": self.prompt, "n": self.n}
        if self.size:
            req["size"] = self.size
        return {**req, **self.extra_params}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ImageGenerationTestCase":
        extra_params = {k: v for k, v in request.items() if k not in ("prompt", "n", "size")}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            prompt=request["prompt"],
            size=request.get("size") or "",
            n=request["n"] if "n" in request else 1,
            metadata=metadata or {},
            extra_params=extra_params,
        )

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class UnansweredImageGenerationTestCase(BaseModel):
    """Image generation test case without a reference output image.

    Use with metrics that do not require a reference image, e.g. judge-based or
    policy checks on generated content.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'text-to-image', 'style-transfer'")
    prompt: str = Field(..., description="The text description of the desired image")
    size: str | None = Field(None, description="Image dimensions, e.g. '1024x1024'")
    n: int = Field(default=1, description="Number of images to generate")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Extra body params, e.g. model, style")

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        req: dict[str, Any] = {"prompt": self.prompt, "n": self.n}
        if self.size:
            req["size"] = self.size
        return {**req, **self.extra_params}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "UnansweredImageGenerationTestCase":
        extra_params = {k: v for k, v in request.items() if k not in ("prompt", "n", "size")}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            prompt=request["prompt"],
            size=request.get("size") or "",
            n=request["n"] if "n" in request else 1,
            metadata=metadata or {},
            extra_params=extra_params,
        )

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class AnsweredImageGenerationTestCase(BaseModel):
    """Image generation test case with a reference / expected output image."""

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'text-to-image', 'style-transfer'")
    prompt: str = Field(..., description="The text description of the desired image")
    size: str | None = Field(None, description="Image dimensions, e.g. '1024x1024'")
    n: int = Field(default=1, description="Number of images to generate")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Extra body params, e.g. model, style")
    generation: str = Field(
        ...,
        description=("Reference / expected output image as a base64-encoded data URI (for metrics vs model output)."),
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        req: dict[str, Any] = {"prompt": self.prompt, "n": self.n}
        if self.size:
            req["size"] = self.size
        return {**req, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def expected_response(self) -> dict[str, Any]:
        """Expected image output shape (reference image as data URI)."""
        return {"image": self.generation}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        generation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "AnsweredImageGenerationTestCase":
        extra_params = {k: v for k, v in request.items() if k not in ("prompt", "n", "size")}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            prompt=request["prompt"],
            size=request.get("size") or "",
            n=request["n"] if "n" in request else 1,
            generation=generation,
            metadata=metadata or {},
            extra_params=extra_params,
        )


# ---------------------------------------------------------------------------
# Image editing test cases
# ---------------------------------------------------------------------------


class ImageEditingTestCase(BaseModel):
    """Request-only base for Image Editing API (`image_editing_api`) test cases.

    Use this type (or `UnansweredImageEditingTestCase`) when no reference edited
    image is provided. For ground-truth edited output, use `AnsweredImageEditingTestCase`.
    """

    model_config = ConfigDict(extra="ignore")

    # --- Identification & Lineage ---
    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'inpainting', 'object-removal'")

    # --- Human-Readable Request Fields ---
    # Images stored as base64-encoded strings; multipart conversion happens in .request.
    image: str = Field(..., description="The source image as a base64-encoded data URI")
    edit_prompt: str = Field(..., description="Natural-language description of the desired edit")
    mask: str | None = Field(
        None,
        description="Optional mask image as a base64-encoded data URI; transparent areas indicate edit regions",
    )

    # --- Private Attributes (Internal/Debugging Only) ---
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    # --- Extra Parameters of the System Request ---
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Extra body params")

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        """
        Computes the OpenAI Image Edit API request fields.
        The image and mask are base64 data URIs here; the test container is responsible
        for converting them to binary file objects when constructing the multipart/form-data call.
        """
        req: dict[str, Any] = {"image": self.image, "prompt": self.edit_prompt}
        if self.mask:
            req["mask"] = self.mask
        return {**req, **self.extra_params}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ImageEditingTestCase":
        extra_params = {k: v for k, v in request.items() if k not in ("image", "mask", "prompt")}
        mask = request.get("mask")
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            image=request["image"],
            edit_prompt=request["prompt"],
            mask=mask if mask else None,
            metadata=metadata or {},
            extra_params=extra_params,
        )

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class UnansweredImageEditingTestCase(BaseModel):
    """Image editing test case without a reference edited output image.

    Use with metrics that do not require a reference image, e.g. judge-based checks.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'inpainting', 'object-removal'")
    image: str = Field(..., description="The source image as a base64-encoded data URI")
    edit_prompt: str = Field(..., description="Natural-language description of the desired edit")
    mask: str | None = Field(
        None,
        description="Optional mask image as a base64-encoded data URI; transparent areas indicate edit regions",
    )
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Extra body params")

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        req: dict[str, Any] = {"image": self.image, "prompt": self.edit_prompt}
        if self.mask:
            req["mask"] = self.mask
        return {**req, **self.extra_params}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "UnansweredImageEditingTestCase":
        extra_params = {k: v for k, v in request.items() if k not in ("image", "mask", "prompt")}
        mask = request.get("mask")
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            image=request["image"],
            edit_prompt=request["prompt"],
            mask=mask if mask else None,
            metadata=metadata or {},
            extra_params=extra_params,
        )

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class AnsweredImageEditingTestCase(BaseModel):
    """Image editing test case with a reference / expected edited output image."""

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'inpainting', 'object-removal'")
    image: str = Field(..., description="The source image as a base64-encoded data URI")
    edit_prompt: str = Field(..., description="Natural-language description of the desired edit")
    mask: str | None = Field(
        None,
        description="Optional mask image as a base64-encoded data URI; transparent areas indicate edit regions",
    )
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Extra body params")
    generation: str = Field(
        ...,
        description="Reference / expected edited output image as a base64-encoded data URI",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        req: dict[str, Any] = {"image": self.image, "prompt": self.edit_prompt}
        if self.mask:
            req["mask"] = self.mask
        return {**req, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def expected_response(self) -> dict[str, Any]:
        """Expected edited image as data URI."""
        return {"image": self.generation}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        generation: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> "AnsweredImageEditingTestCase":
        extra_params = {k: v for k, v in request.items() if k not in ("image", "mask", "prompt")}
        mask = request.get("mask")
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            image=request["image"],
            edit_prompt=request["prompt"],
            mask=mask if mask else None,
            generation=generation,
            metadata=metadata or {},
            extra_params=extra_params,
        )


class EmbeddingTestCase(BaseModel):
    """Test case for Embedding API (`embedding_api`) systems."""

    model_config = ConfigDict(extra="ignore")

    # --- Identification & Lineage ---
    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'semantic-similarity', 'retrieval-quality'")

    # --- Human-Readable Request Fields ---
    text: str = Field(..., description="The input text to embed")

    # --- Human-Readable Response Fields (Ground Truth) ---
    # Raw embedding vectors are not specified as ground truth; evaluation uses similarity constraints.
    expected_similar_texts: list[str] = Field(
        default_factory=list,
        description="Texts that should be semantically close to `text` (cosine similarity above threshold)",
    )
    expected_dissimilar_texts: list[str] = Field(
        default_factory=list,
        description="Texts that should be semantically distant from `text`",
    )

    # --- Private Attributes (Internal/Debugging Only) ---
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    # --- Extra Parameters of the System Request ---
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Extra body params, e.g. model")

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        """Computes the OpenAI Embeddings API request."""
        return {"input": self.text, **self.extra_params}

    @property
    def expected_response(self) -> None:
        """
        No structured expected response — expected_similar_texts / expected_dissimilar_texts
        are evaluated by the metric after computing cosine similarity against returned vectors.
        """
        return None

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        expected_similar_texts: list[str] | None = None,
        expected_dissimilar_texts: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "EmbeddingTestCase":
        extra_params = {k: v for k, v in request.items() if k != "input"}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            text=request["input"],
            expected_similar_texts=expected_similar_texts or [],
            expected_dissimilar_texts=expected_dissimilar_texts or [],
            metadata=metadata or {},
            extra_params=extra_params,
        )

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class BoundingBox(BaseModel):
    """Ground truth bounding box in xyxy format."""

    xyxy: tuple[float, float, float, float] = Field(..., description="Box coordinates as (x_min, y_min, x_max, y_max)")
    class_name: str = Field(..., description="Object class label, e.g. 'person', 'car'")
    confidence: float | None = Field(None, description="Ground truth confidence (if sourced from a prior model run)")


class ODImageEditingTestCase(BaseModel):
    """Image editing test case with reference object locations on the seed image (SDG / detection-aware metrics).

    The synthetic or edited result still comes from the system response; ``expected_detections`` supplies
    the same structured box list as legacy ``detected_objects['bbox']`` rows in SDG datasets.
    An empty list means no boxes (many per-object metrics degenerate or error — see metric docs).
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'inpainting', 'object-removal'")
    image: str = Field(..., description="The source image as a base64-encoded data URI")
    edit_prompt: str = Field(..., description="Natural-language description of the desired edit")
    mask: str | None = Field(
        None,
        description="Optional mask image as a base64-encoded data URI; transparent areas indicate edit regions",
    )
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict, description="Extra body params")
    expected_detections: list[BoundingBox] = Field(
        default_factory=list,
        description="Reference boxes on the seed image (xyxy), aligned to synthetic crops by index order",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        req: dict[str, Any] = {"image": self.image, "prompt": self.edit_prompt}
        if self.mask:
            req["mask"] = self.mask
        return {**req, **self.extra_params}

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ODImageEditingTestCase":
        extra_params = {k: v for k, v in request.items() if k not in ("image", "mask", "prompt")}
        mask = request.get("mask")
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            image=request["image"],
            edit_prompt=request["prompt"],
            mask=mask if mask else None,
            metadata=metadata or {},
            extra_params=extra_params,
        )

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


# ---------------------------------------------------------------------------
# Object detection test cases
# ---------------------------------------------------------------------------


class ObjectDetectionTestCase(BaseModel):
    """Image-only base for Object Detection API (`object_detection_api`) test cases.

    Use this type (or `UnansweredObjectDetectionTestCase`) when no reference bounding
    boxes are provided — e.g. judge-only or exploratory metrics. For ground-truth
    boxes (including an empty list meaning zero objects expected), use
    `AnsweredObjectDetectionTestCase`.
    """

    model_config = ConfigDict(extra="ignore")

    # --- Identification & Lineage ---
    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'crowded-scene', 'low-light'")

    # --- Human-Readable Request Fields ---
    # Image stored as a base64-encoded data URI; the test container converts to binary for multipart upload.
    image: str = Field(..., description="The image as a base64-encoded data URI")

    # --- Private Attributes (Internal/Debugging Only) ---
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)

    # --- Extra Parameters of the System Request ---
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider overrides, e.g. {'model': 'facebook/detr-resnet-50', 'provider': 'huggingface'}",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        """
        Computes the Object Detection API request fields.
        `image` is a base64 data URI here; the test container converts it to a binary file
        when constructing the multipart/form-data POST to /detect.
        """
        return {"image": self.image, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class UnansweredObjectDetectionTestCase(BaseModel):
    """Object detection test case without ground-truth bounding boxes.

    Use with metrics that do not require reference boxes, e.g. judge-based checks.
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'crowded-scene', 'low-light'")
    image: str = Field(..., description="The image as a base64-encoded data URI")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider overrides, e.g. {'model': 'facebook/detr-resnet-50', 'provider': 'huggingface'}",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        return {"image": self.image, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata


class AnsweredObjectDetectionTestCase(BaseModel):
    """Object detection test case with ground-truth bounding boxes.

    `expected_detections` uses a nested list[BoundingBox] structure — an explicit
    exception to the flat-field rule, agreed under PROG-32, because bounding-box ground
    truth is inherently structured. An empty list means no objects are expected in the
    image (distinct from omitting ground truth on the base / unanswered type).
    """

    model_config = ConfigDict(extra="ignore")

    lineage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Stable ID linking variants to their seed; auto-generated if not provided",
    )
    scenario: str | None = Field(None, description="e.g., 'crowded-scene', 'low-light'")
    image: str = Field(..., description="The image as a base64-encoded data URI")
    _metadata: dict[str, Any] = PrivateAttr(default_factory=dict)
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider overrides, e.g. {'model': 'facebook/detr-resnet-50', 'provider': 'huggingface'}",
    )
    expected_detections: list[BoundingBox] = Field(
        default_factory=list,
        description="Ground truth detections; empty list means no objects expected",
    )
    confidence_threshold: float | None = Field(
        None,
        description="Minimum confidence for a predicted detection to be matched against ground truth",
    )

    def __init__(self, **data: Any) -> None:
        metadata = _test_case_pop_metadata_and_lineage(data)
        super().__init__(**data)
        self._metadata = metadata

    @property
    def request(self) -> dict[str, Any]:
        return {"image": self.image, **self.extra_params}

    def get_debug_metadata(self) -> dict[str, Any]:
        return self._metadata

    @property
    def expected_response(self) -> dict[str, Any]:
        """Computes the expected /detect response structure."""
        return {
            "detections": [
                {
                    "xyxy": list(det.xyxy),
                    "class_name": det.class_name,
                    **({"confidence": det.confidence} if det.confidence is not None else {}),
                }
                for det in self.expected_detections
            ]
        }

    @classmethod
    def from_api_data(
        cls,
        request: dict[str, Any],
        response: dict[str, Any],
        lineage_id: str | None = None,
        scenario: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "AnsweredObjectDetectionTestCase":
        detections = [
            BoundingBox(
                xyxy=tuple(d["xyxy"]),
                class_name=d["class_name"],
                confidence=d.get("confidence"),
            )
            for d in response.get("detections", [])
        ]
        extra_params = {k: v for k, v in request.items() if k != "image"}
        return cls(
            lineage_id=lineage_id,
            scenario=scenario,
            image=request["image"],
            expected_detections=detections,
            metadata=metadata or {},
            extra_params=extra_params,
        )


TestCase = (
    LLMTestCase
    | AnsweredLLMTestCase
    | RAGTestCase
    | AnsweredRAGTestCase
    | ContextualizedRAGTestCase
    | VLMTestCase
    | AnsweredVLMTestCase
    | ImageGenerationTestCase
    | AnsweredImageGenerationTestCase
    | ImageEditingTestCase
    | AnsweredImageEditingTestCase
    | ODImageEditingTestCase
    | EmbeddingTestCase
    | ObjectDetectionTestCase
    | AnsweredObjectDetectionTestCase
)
