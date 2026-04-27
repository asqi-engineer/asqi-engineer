from enum import StrEnum
from typing import Annotated, Any, Literal, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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

    items: (
        Union[
            Literal["string", "integer", "float", "boolean", "object", "enum"],
            "InputParameter",
        ]
        | None
    ) = Field(
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
            "json",
            "csv",
            "parquet",
            "arrow",
            "text",
            "xml",
            "webdataset",
            "imagefolder",
            "audiofolder",
            "videofolder",
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
