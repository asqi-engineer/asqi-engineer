from typing import get_args

import pytest
from datasets.features import Value
from pydantic import ValidationError

from asqi.schemas import (
    AudioFeature,
    ClassLabelFeature,
    DatasetFeature,
    DatasetType,
    DictFeature,
    HFDtype,
    ImageFeature,
    InputDataset,
    InputParameter,
    ListFeature,
    OutputDataset,
    ValueFeature,
    VideoFeature,
)


class TestHFDtypeLiteral:
    """Test that our HFDtype Literal matches actual HuggingFace Value types."""

    def test_all_literal_values_are_valid_hf_types(self):
        """Test that every value in our HFDtype Literal can create a valid HuggingFace Value."""
        literal_dtypes = get_args(HFDtype)

        failed_dtypes = []
        for dtype in literal_dtypes:
            try:
                # Try to create a Value with this dtype
                Value(dtype=dtype)  # type: ignore[arg-type]
            except Exception as e:
                failed_dtypes.append((dtype, str(e)))

        if failed_dtypes:
            error_msg = "The following dtypes in HFDtype Literal are not valid HuggingFace Value types:\n"
            for dtype, error in failed_dtypes:
                error_msg += f"  - '{dtype}': {error}\n"
            pytest.fail(error_msg)

    def test_common_hf_types_are_in_literal(self):
        """Test that common HuggingFace types are included in our Literal."""
        # Test the most common types that users will need
        common_types = [
            "string",
            "int64",
            "int32",
            "float64",
            "float32",
            "bool",
            "null",
            "binary",
            "timestamp[s]",
            "date32",
        ]

        literal_dtypes = get_args(HFDtype)

        missing_types = []
        for dtype in common_types:
            if dtype not in literal_dtypes:
                missing_types.append(dtype)

        if missing_types:
            pytest.fail(
                f"Common HuggingFace types missing from HFDtype Literal: {missing_types}"
            )


class TestDatasetFeature:
    """Test the DatasetFeature model with dtype validation."""

    def test_create_with_string_dtype(self):
        """Test creating DatasetFeature with valid string dtype."""
        feature = DatasetFeature(
            name="user_id", dtype="int64", description="Unique user identifier"
        )
        assert feature.name == "user_id"
        assert feature.dtype == "int64"
        assert feature.description == "Unique user identifier"

    def test_create_multiple_features(self):
        """Test creating multiple features with different dtypes."""
        features = [
            DatasetFeature(name="id", dtype="int64"),
            DatasetFeature(name="name", dtype="string"),
            DatasetFeature(name="score", dtype="float32"),
            DatasetFeature(name="active", dtype="bool"),
            DatasetFeature(name="created_at", dtype="timestamp[s]"),
        ]

        assert len(features) == 5
        assert features[0].dtype == "int64"
        assert features[1].dtype == "string"
        assert features[4].dtype == "timestamp[s]"

    def test_invalid_dtype_raises_validation_error(self):
        """Test that invalid dtypes raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            DatasetFeature(name="bad", dtype="invalid_type")  # type: ignore[arg-type]

        error = exc_info.value
        assert "dtype" in str(error)

    def test_missing_required_fields(self):
        """Test that missing required fields raise ValidationError."""
        # Missing name
        with pytest.raises(ValidationError) as exc_info:
            DatasetFeature(dtype="string")  # type: ignore[call-arg]
        assert "name" in str(exc_info.value)

        # Missing dtype
        with pytest.raises(ValidationError) as exc_info:
            DatasetFeature(name="test")  # type: ignore[call-arg]
        assert "dtype" in str(exc_info.value)

    def test_optional_description(self):
        """Test that description is optional."""
        feature = DatasetFeature(name="test", dtype="string")
        assert feature.description is None

        feature_with_desc = DatasetFeature(
            name="test", dtype="string", description="Test feature"
        )
        assert feature_with_desc.description == "Test feature"

    def test_all_common_dtypes(self):
        """Test all common HuggingFace dtypes are accepted."""
        common_dtypes = [
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
            "float64",
            "string",
            "binary",
        ]

        for dtype in common_dtypes:
            feature = DatasetFeature(name=f"test_{dtype}", dtype=dtype)  # type: ignore[arg-type]
            assert feature.dtype == dtype

    def test_dtype_aliases(self):
        """Test that dtype aliases are accepted."""
        # 'float' alias for 'float32'
        feature = DatasetFeature(name="value", dtype="float")
        assert feature.dtype == "float"

        # 'double' alias for 'float64'
        feature = DatasetFeature(name="value", dtype="double")
        assert feature.dtype == "double"

    def test_parametric_dtypes(self):
        """Test parametric dtypes (timestamp, time, duration)."""
        parametric_dtypes = [
            "timestamp[s]",
            "timestamp[ms]",
            "timestamp[us]",
            "timestamp[ns]",
            "time32[s]",
            "time32[ms]",
            "time64[us]",
            "time64[ns]",
            "duration[s]",
            "duration[ms]",
            "duration[us]",
            "duration[ns]",
        ]

        for dtype in parametric_dtypes:
            feature = DatasetFeature(name=f"test_{dtype}", dtype=dtype)  # type: ignore[arg-type]
            assert feature.dtype == dtype


class TestInputDatasetWithFeatures:
    """Test InputDataset with DatasetFeature validation."""

    def test_input_dataset_with_valid_features(self):
        """Test creating InputDataset with validated features."""
        dataset = InputDataset(
            name="user_activity",
            required=True,
            type=DatasetType.HUGGINGFACE,
            description="User activity dataset",
            features=[
                DatasetFeature(name="user_id", dtype="int64"),
                DatasetFeature(name="activity", dtype="string"),
                DatasetFeature(name="timestamp", dtype="timestamp[s]"),
            ],
        )

        assert dataset.name == "user_activity"
        assert dataset.required is True
        assert dataset.type == DatasetType.HUGGINGFACE
        assert dataset.features is not None
        assert len(dataset.features) == 3
        assert isinstance(dataset.features[0], DatasetFeature)
        assert dataset.features[0].dtype == "int64"
        assert isinstance(dataset.features[1], DatasetFeature)
        assert dataset.features[1].dtype == "string"
        assert isinstance(dataset.features[2], DatasetFeature)
        assert dataset.features[2].dtype == "timestamp[s]"

    def test_input_dataset_features_must_be_defined_for_huggingface(self):
        """Test that HuggingFace datasets require features to be defined."""
        with pytest.raises(ValidationError) as exc_info:
            InputDataset(
                name="test_dataset",
                required=True,
                type=DatasetType.HUGGINGFACE,
                features=None,  # Missing features
            )
        assert (
            "Features must be defined when 'huggingface' is an accepted dataset type"
            in str(exc_info.value)
        )

    def test_input_dataset_with_invalid_feature_dtype(self):
        """Test that invalid feature dtypes are caught."""
        with pytest.raises(ValidationError) as exc_info:
            InputDataset(
                name="test_dataset",
                required=True,
                type=DatasetType.HUGGINGFACE,
                features=[
                    DatasetFeature(name="valid", dtype="string"),
                    DatasetFeature(name="invalid", dtype="not_a_type"),  # type: ignore[arg-type]
                ],
            )
        error_str = str(exc_info.value)
        assert "dtype" in error_str

    def test_input_dataset_pdf_type_no_features(self):
        """Test that PDF datasets don't require features."""
        dataset = InputDataset(
            name="document",
            required=True,
            type=DatasetType.PDF,
            description="PDF document",
            features=None,
        )
        assert dataset.name == "document"
        assert dataset.type == DatasetType.PDF
        assert dataset.features is None

    def test_input_dataset_multiple_types(self):
        """Test that InputDataset can accept multiple types."""
        dataset = InputDataset(
            name="knowledge_base",
            required=True,
            type=[DatasetType.PDF, DatasetType.TXT],
            description="Knowledge base - accepts PDF or TXT",
            features=None,
        )
        assert dataset.name == "knowledge_base"
        assert dataset.type == [DatasetType.PDF, DatasetType.TXT]
        assert dataset.features is None

    def test_input_dataset_multiple_types_with_huggingface(self):
        """Test that HuggingFace in multi-type list DOES require features."""
        with pytest.raises(ValidationError) as exc_info:
            InputDataset(
                name="flexible_data",
                required=True,
                type=[DatasetType.HUGGINGFACE, DatasetType.PDF, DatasetType.TXT],
                description="Flexible input - accepts multiple formats",
                features=None,  # Features ARE required when huggingface is accepted
            )
        assert (
            "Features must be defined when 'huggingface' is an accepted dataset type"
            in str(exc_info.value)
        )

    def test_input_dataset_multiple_types_with_features(self):
        """Test that multi-type datasets can still have features defined."""
        dataset = InputDataset(
            name="flexible_data",
            required=True,
            type=[DatasetType.HUGGINGFACE, DatasetType.PDF, DatasetType.TXT],
            description="Flexible input with optional features",
            features=[
                DatasetFeature(name="text", dtype="string"),
            ],
        )
        assert dataset.name == "flexible_data"
        assert dataset.type == [
            DatasetType.HUGGINGFACE,
            DatasetType.PDF,
            DatasetType.TXT,
        ]
        assert dataset.features is not None
        assert len(dataset.features) == 1

    def test_input_dataset_single_huggingface_still_requires_features(self):
        """Test that single HuggingFace type still requires features (backward compatibility)."""
        with pytest.raises(ValidationError) as exc_info:
            InputDataset(
                name="hf_only",
                required=True,
                type=DatasetType.HUGGINGFACE,  # Single type
                features=None,  # Should fail
            )
        assert (
            "Features must be defined when 'huggingface' is an accepted dataset type"
            in str(exc_info.value)
        )


class TestDatasetFeatureDtypeExamples:
    """Test real-world usage examples of DatasetFeature dtypes."""

    def test_nlp_dataset_features(self):
        """Test typical NLP dataset feature types."""
        features = [
            DatasetFeature(name="text", dtype="string"),
            DatasetFeature(name="label", dtype="int64"),
            DatasetFeature(name="tokens", dtype="string"),
        ]
        assert all(f.dtype in ["string", "int64"] for f in features)

    def test_numeric_dataset_features(self):
        """Test typical numeric dataset feature types."""
        features = [
            DatasetFeature(name="id", dtype="int64"),
            DatasetFeature(name="value", dtype="float64"),
            DatasetFeature(name="count", dtype="int32"),
            DatasetFeature(name="percentage", dtype="float32"),
        ]
        assert len(features) == 4
        assert features[1].dtype == "float64"

    def test_temporal_dataset_features(self):
        """Test temporal data types."""
        features = [
            DatasetFeature(name="event_timestamp", dtype="timestamp[s]"),
            DatasetFeature(name="created_date", dtype="date32"),
            DatasetFeature(name="duration", dtype="duration[ms]"),
            DatasetFeature(name="event_time", dtype="time64[us]"),
        ]
        assert len(features) == 4
        assert "timestamp" in features[0].dtype


class TestValueFeature:
    """Test the new ValueFeature model."""

    def test_create_value_feature(self):
        """Test creating a simple ValueFeature."""

        vf = ValueFeature(name="text", dtype="string", description="Text field")
        assert vf.name == "text"
        assert vf.dtype == "string"
        assert vf.feature_type == "Value"
        assert vf.description == "Text field"

    def test_value_feature_default_feature_type(self):
        """Test that feature_type defaults to 'Value'."""

        vf = ValueFeature(name="score", dtype="float32")
        assert vf.feature_type == "Value"

    def test_value_feature_all_dtypes(self):
        """Test ValueFeature with various dtypes."""

        test_cases = [
            ("string", "string"),
            ("int64", "int64"),
            ("float32", "float32"),
            ("bool", "bool"),
            ("timestamp[s]", "timestamp[s]"),
        ]

        for name_suffix, dtype in test_cases:
            vf = ValueFeature(name=f"field_{name_suffix}", dtype=dtype)  # type: ignore[arg-type]
            assert vf.dtype == dtype


class TestListFeature:
    """Test the new ListFeature model."""

    def test_create_simple_list_feature(self):
        """Test creating a simple ListFeature with feature."""

        lf = ListFeature(name="tags", feature="string")
        assert lf.name == "tags"
        assert lf.feature == "string"
        assert lf.feature_type == "List"

    def test_list_feature_all_element_types(self):
        """Test ListFeature with various element dtypes."""

        element_types = ["string", "int32", "int64", "float32", "float64", "bool"]

        for dtype in element_types:
            lf = ListFeature(name=f"list_{dtype}", feature=dtype)  # type: ignore[arg-type]
            assert lf.feature == dtype


class TestDictFeature:
    """Test the new DictFeature model."""

    def test_create_simple_dict_feature(self):
        """Test creating a DictFeature with multiple fields."""

        df = DictFeature(
            name="answers",
            fields=[
                ListFeature(name="text", feature="string"),
                ListFeature(name="answer_start", feature="int32"),
            ],
        )
        assert df.name == "answers"
        assert df.feature_type == "Dict"
        assert len(df.fields) == 2

    def test_dict_feature_nested_types(self):
        """Test DictFeature with various nested field types."""

        df = DictFeature(
            name="complex_record",
            fields=[
                ValueFeature(name="id", dtype="string"),
                ListFeature(name="tags", feature="string"),
                ClassLabelFeature(name="category", names=["a", "b", "c"]),
            ],
        )
        assert len(df.fields) == 3
        assert df.fields[0].feature_type == "Value"
        assert df.fields[1].feature_type == "List"
        assert df.fields[2].feature_type == "ClassLabel"

    def test_dict_feature_requires_fields(self):
        """Test that DictFeature requires at least one field."""

        with pytest.raises(ValidationError) as exc_info:
            DictFeature(name="bad", fields=[])  # type: ignore[arg-type]
        assert "at least 1" in str(exc_info.value).lower()

    def test_dict_feature_nested_dicts(self):
        """Test DictFeature containing other DictFeatures (nested dicts)."""

        df = DictFeature(
            name="metadata",
            fields=[
                ValueFeature(name="id", dtype="string"),
                DictFeature(
                    name="author",
                    fields=[
                        ValueFeature(name="name", dtype="string"),
                        ValueFeature(name="email", dtype="string"),
                    ],
                ),
            ],
        )
        assert df.fields[1].feature_type == "Dict"
        assert len(df.fields[1].fields) == 2


class TestClassLabelFeature:
    """Test the new ClassLabelFeature model."""

    def test_create_class_label_feature(self):
        """Test creating a ClassLabelFeature."""

        clf = ClassLabelFeature(
            name="sentiment", names=["positive", "negative", "neutral"]
        )
        assert clf.name == "sentiment"
        assert clf.names == ["positive", "negative", "neutral"]
        assert clf.feature_type == "ClassLabel"

    def test_class_label_feature_with_description(self):
        """Test ClassLabelFeature with description."""

        clf = ClassLabelFeature(
            name="category",
            names=["A", "B", "C"],
            description="Category classification",
        )
        assert clf.description == "Category classification"

    def test_class_label_feature_empty_names_fails(self):
        """Test that empty names list raises validation error."""

        with pytest.raises(ValidationError):
            ClassLabelFeature(name="bad", names=[])  # type: ignore[arg-type]


class TestImageFeature:
    """Test the ImageFeature model."""

    def test_create_image_feature(self):
        """Test creating an ImageFeature."""
        img = ImageFeature(name="image")
        assert img.name == "image"
        assert img.feature_type == "Image"
        assert img.required is False

    def test_image_feature_optional(self):
        """Test ImageFeature can be marked as optional."""
        img = ImageFeature(name="thumbnail", required=False)
        assert img.required is False

    def test_image_feature_with_description(self):
        """Test ImageFeature with description."""
        img = ImageFeature(name="photo", description="Product image")
        assert img.description == "Product image"


class TestAudioFeature:
    """Test the AudioFeature model."""

    def test_create_audio_feature(self):
        """Test creating an AudioFeature."""
        aud = AudioFeature(name="audio")
        assert aud.name == "audio"
        assert aud.feature_type == "Audio"
        assert aud.required is False

    def test_audio_feature_optional(self):
        """Test AudioFeature can be marked as optional."""
        aud = AudioFeature(name="voice_note", required=False)
        assert aud.required is False

    def test_audio_feature_with_description(self):
        """Test AudioFeature with description."""
        aud = AudioFeature(
            name="recording", description="Audio recording of conversation"
        )
        assert aud.description == "Audio recording of conversation"


class TestVideoFeature:
    """Test the VideoFeature model."""

    def test_create_video_feature(self):
        """Test creating a VideoFeature."""
        vid = VideoFeature(name="video")
        assert vid.name == "video"
        assert vid.feature_type == "Video"
        assert vid.required is False

    def test_video_feature_optional(self):
        """Test VideoFeature can be marked as optional."""
        vid = VideoFeature(name="clip", required=False)
        assert vid.required is False

    def test_video_feature_with_description(self):
        """Test VideoFeature with description."""
        vid = VideoFeature(name="demo_video", description="Product demonstration video")
        assert vid.description == "Product demonstration video"


class TestMixedFeatureTypes:
    """Test using mixed old and new feature types together."""

    def test_input_dataset_with_legacy_features(self):
        """Test InputDataset works with legacy DatasetFeature."""

        ds = InputDataset(
            name="legacy_ds",
            type=DatasetType.HUGGINGFACE,
            features=[
                DatasetFeature(name="text", dtype="string"),
                DatasetFeature(name="score", dtype="float32"),
            ],
        )
        assert ds.features is not None
        assert len(ds.features) == 2
        assert isinstance(ds.features[0], DatasetFeature)

    def test_input_dataset_with_new_features(self):
        """Test InputDataset works with new HFFeature types."""

        ds = InputDataset(
            name="modern_ds",
            type=DatasetType.HUGGINGFACE,
            features=[
                ValueFeature(name="id", dtype="string"),
                ListFeature(name="tags", feature="string"),
                ClassLabelFeature(name="category", names=["A", "B"]),
                ImageFeature(name="image"),
            ],
        )
        assert ds.features is not None
        assert len(ds.features) == 4

    def test_input_dataset_mixed_features(self):
        """Test InputDataset with both old and new feature types."""

        ds = InputDataset(
            name="mixed_ds",
            type=DatasetType.HUGGINGFACE,
            features=[
                DatasetFeature(name="old_field", dtype="int64"),
                ValueFeature(name="new_field", dtype="string"),
                ListFeature(name="items", feature="float32"),
            ],
        )
        assert ds.features is not None
        assert len(ds.features) == 3

    def test_output_dataset_with_nested_structure(self):
        """Test OutputDataset with DictFeature (SQuAD-style)."""

        ds = OutputDataset(
            name="qa_dataset",
            type=DatasetType.HUGGINGFACE,
            features=[
                ValueFeature(name="id", dtype="string"),
                ValueFeature(name="question", dtype="string"),
                DictFeature(
                    name="answers",
                    fields=[
                        ListFeature(name="text", feature="string"),
                        ListFeature(name="answer_start", feature="int32"),
                    ],
                ),
            ],
        )
        assert ds.features is not None
        assert len(ds.features) == 3
        # Check nested structure
        answers_feature = ds.features[2]
        assert answers_feature.name == "answers"
        assert isinstance(answers_feature, DictFeature)
        assert answers_feature.feature_type == "Dict"
        assert len(answers_feature.fields) == 2
        assert answers_feature.fields[0].name == "text"
        assert answers_feature.fields[1].name == "answer_start"


class TestDiscriminatedUnion:
    """Test that discriminated union works correctly with feature_type."""

    def test_discriminator_present_in_all_types(self):
        """Test that all feature types have feature_type field."""

        features = [
            ValueFeature(name="v", dtype="string"),
            ListFeature(name="l", feature="int32"),
            ClassLabelFeature(name="c", names=["a"]),
            ImageFeature(name="i"),
            AudioFeature(name="a"),
            VideoFeature(name="vid"),
        ]

        expected_types = ["Value", "List", "ClassLabel", "Image", "Audio", "Video"]

        for feat, expected in zip(features, expected_types):
            assert feat.feature_type == expected

    def test_json_serialization_includes_discriminator(self):
        """Test that JSON serialization includes feature_type for discrimination."""

        vf = ValueFeature(name="test", dtype="string")
        vf_dict = vf.model_dump()
        assert "feature_type" in vf_dict
        assert vf_dict["feature_type"] == "Value"

        lf = ListFeature(name="items", feature="int32")
        lf_dict = lf.model_dump()
        assert "feature_type" in lf_dict
        assert lf_dict["feature_type"] == "List"


class TestOptionalFeatures:
    """Test optional/required field in all feature types."""

    def test_value_feature_required_defaults_to_false(self):
        """Test that required defaults to False."""

        vf = ValueFeature(name="test", dtype="string")
        assert vf.required is False

    def test_value_feature_optional(self):
        """Test creating optional ValueFeature."""

        vf = ValueFeature(name="optional_field", dtype="string", required=False)
        assert vf.required is False

    def test_list_feature_optional(self):
        """Test creating optional ListFeature."""

        lf = ListFeature(name="optional_tags", feature="string", required=False)
        assert lf.required is False

    def test_class_label_optional(self):
        """Test creating optional ClassLabelFeature."""

        clf = ClassLabelFeature(
            name="optional_category", names=["a", "b"], required=False
        )
        assert clf.required is False

    def test_image_feature_optional(self):
        """Test creating optional ImageFeature."""

        imf = ImageFeature(name="optional_thumbnail", required=False)
        assert imf.required is False

    def test_audio_feature_optional(self):
        """Test creating optional AudioFeature."""

        auf = AudioFeature(name="optional_narration", required=False)
        assert auf.required is False

    def test_video_feature_optional(self):
        """Test creating optional VideoFeature."""

        vdf = VideoFeature(name="optional_video_clip", required=False)
        assert vdf.required is False

    def test_output_dataset_with_optional_features(self):
        """Test OutputDataset with mix of required and optional features."""

        ds = OutputDataset(
            name="product_catalog",
            type=DatasetType.HUGGINGFACE,
            features=[
                ValueFeature(name="product_id", dtype="string", required=True),
                ValueFeature(name="name", dtype="string", required=True),
                ValueFeature(
                    name="description", dtype="string", required=False
                ),  # Optional
                ImageFeature(
                    name="thumbnail", required=False
                ),  # Optional - not all products have images
                ValueFeature(name="price", dtype="float32", required=True),  # Required
            ],
        )

        assert ds.features is not None
        assert len(ds.features) == 5
        # Check required/optional status (all are HFFeature types with required field)
        assert isinstance(
            ds.features[0],
            (
                ValueFeature,
                ListFeature,
                DictFeature,
                ClassLabelFeature,
                ImageFeature,
                AudioFeature,
            ),
        )
        assert ds.features[0].required is True  # product_id
        assert isinstance(
            ds.features[1],
            (
                ValueFeature,
                ListFeature,
                DictFeature,
                ClassLabelFeature,
                ImageFeature,
                AudioFeature,
            ),
        )
        assert ds.features[1].required is True  # name
        assert isinstance(
            ds.features[2],
            (
                ValueFeature,
                ListFeature,
                DictFeature,
                ClassLabelFeature,
                ImageFeature,
                AudioFeature,
            ),
        )
        assert ds.features[2].required is False  # description
        assert isinstance(
            ds.features[3],
            (
                ValueFeature,
                ListFeature,
                DictFeature,
                ClassLabelFeature,
                ImageFeature,
                AudioFeature,
            ),
        )
        assert ds.features[3].required is False  # thumbnail
        assert isinstance(
            ds.features[4],
            (
                ValueFeature,
                ListFeature,
                DictFeature,
                ClassLabelFeature,
                ImageFeature,
                AudioFeature,
            ),
        )
        assert ds.features[4].required is True  # price


class TestInputParameter:
    """Test InputParameter with basic types."""

    def test_create_simple_string_parameter(self):
        """Test creating a simple string parameter (backward compatible)."""
        param = InputParameter(name="test_name", type="string", required=True)
        assert param.name == "test_name"
        assert param.type == "string"
        assert param.required is True

    def test_create_all_simple_types(self):
        """Test creating parameters with all simple types."""
        types = ["string", "integer", "float", "boolean", "list", "object"]
        for param_type in types:
            param = InputParameter(name=f"test_{param_type}", type=param_type)  # type: ignore[arg-type]
            assert param.type == param_type

    def test_required_defaults_to_false(self):
        """Test that required defaults to False."""
        param = InputParameter(name="optional", type="string")
        assert param.required is False

    def test_description_is_optional(self):
        """Test that description is optional."""
        param = InputParameter(name="test", type="string")
        assert param.description is None

        param_with_desc = InputParameter(
            name="test", type="string", description="Test parameter"
        )
        assert param_with_desc.description == "Test parameter"


class TestInputParameterEnum:
    """Test InputParameter with enum type."""

    def test_create_enum_parameter(self):
        """Test creating an enum parameter with choices."""
        param = InputParameter(
            name="mode",
            type="enum",
            choices=["fast", "balanced", "thorough"],
            default="balanced",
        )
        assert param.type == "enum"
        assert param.choices == ["fast", "balanced", "thorough"]
        assert param.default == "balanced"

    def test_enum_requires_choices(self):
        """Test that enum type requires choices field."""
        with pytest.raises(ValidationError) as exc_info:
            InputParameter(name="bad_enum", type="enum")
        assert "requires 'choices'" in str(exc_info.value)

    def test_enum_default_must_be_in_choices(self):
        """Test that default value must be in choices for enums."""
        with pytest.raises(ValidationError) as exc_info:
            InputParameter(
                name="mode",
                type="enum",
                choices=["fast", "slow"],
                default="invalid",
            )
        assert "must be one of the allowed choices" in str(exc_info.value)

    def test_enum_with_integer_choices(self):
        """Test enum with integer choices."""
        param = InputParameter(
            name="priority", type="enum", choices=[1, 2, 3], default=2
        )
        assert param.choices == [1, 2, 3]
        assert param.default == 2

    def test_enum_with_mixed_types_in_choices(self):
        """Test enum with mixed string/int/float choices."""
        param = InputParameter(
            name="mixed", type="enum", choices=["auto", 1, 2.5], default="auto"
        )
        assert param.choices == ["auto", 1, 2.5]

    def test_choices_only_valid_for_enum(self):
        """Test that choices field can only be used with enum type."""
        with pytest.raises(ValidationError) as exc_info:
            InputParameter(
                name="bad",
                type="string",
                choices=["a", "b"],
            )
        assert "'choices' field can only be specified when type='enum'" in str(
            exc_info.value
        )


class TestInputParameterTypedLists:
    """Test InputParameter with typed lists using items field."""

    def test_simple_typed_list(self):
        """Test creating a list with simple type string."""
        param = InputParameter(
            name="tags", type="list", items="string", default=["test", "demo"]
        )
        assert param.type == "list"
        assert param.items == "string"
        assert param.default == ["test", "demo"]

    def test_list_of_integers(self):
        """Test list with integer items."""
        param = InputParameter(
            name="retry_delays", type="list", items="integer", default=[1, 2, 5, 10]
        )
        assert param.items == "integer"
        assert param.default == [1, 2, 5, 10]

    def test_list_without_items_is_valid(self):
        """Test that untyped lists (legacy) still work."""
        param = InputParameter(name="untyped_list", type="list")
        assert param.type == "list"
        assert param.items is None

    def test_items_only_valid_for_list(self):
        """Test that items field can only be used with list type."""
        with pytest.raises(ValidationError) as exc_info:
            InputParameter(name="bad", type="string", items="integer")
        assert "'items' field can only be specified when type='list'" in str(
            exc_info.value
        )


class TestInputParameterListOfEnums:
    """Test InputParameter with lists of enums."""

    def test_list_of_enums(self):
        """Test creating a list where each element is an enum."""
        param = InputParameter(
            name="severity_levels",
            type="list",
            items=InputParameter(
                type="enum",
                choices=["info", "warning", "error", "critical"],
                name="severity_item",
            ),
            default=["warning", "error"],
        )
        assert param.type == "list"
        assert isinstance(param.items, InputParameter)
        assert param.items.type == "enum"
        assert param.items.choices == ["info", "warning", "error", "critical"]


class TestInputParameterNestedObjects:
    """Test InputParameter with nested object properties."""

    def test_simple_nested_object(self):
        """Test creating an object with properties."""
        param = InputParameter(
            name="api_config",
            type="object",
            properties=[
                InputParameter(name="max_retries", type="integer", default=3),
                InputParameter(name="timeout", type="integer", default=30),
                InputParameter(name="enable_cache", type="boolean", default=True),
            ],
        )
        assert param.type == "object"
        assert param.properties is not None
        assert len(param.properties) == 3
        assert param.properties[0].name == "max_retries"
        assert param.properties[0].default == 3

    def test_object_without_properties_is_valid(self):
        """Test that untyped objects (legacy) still work."""
        param = InputParameter(name="untyped_object", type="object")
        assert param.type == "object"
        assert param.properties is None

    def test_properties_only_valid_for_object(self):
        """Test that properties field can only be used with object type."""
        with pytest.raises(ValidationError) as exc_info:
            InputParameter(
                name="bad",
                type="string",
                properties=[InputParameter(name="field", type="string")],
            )
        assert "'properties' field can only be specified when type='object'" in str(
            exc_info.value
        )

    def test_nested_object_with_enum_property(self):
        """Test object containing an enum property."""
        param = InputParameter(
            name="config",
            type="object",
            properties=[
                InputParameter(
                    name="strategy",
                    type="enum",
                    choices=["linear", "exponential"],
                    default="exponential",
                ),
                InputParameter(name="max_attempts", type="integer", default=5),
            ],
        )
        assert param.properties is not None
        assert param.properties[0].type == "enum"
        assert param.properties[0].choices == ["linear", "exponential"]

    def test_nested_object_with_list_property(self):
        """Test object containing a typed list property."""
        param = InputParameter(
            name="config",
            type="object",
            properties=[
                InputParameter(
                    name="endpoints",
                    type="list",
                    items="string",
                    default=["/api/v1/chat"],
                ),
            ],
        )
        assert param.properties is not None
        assert param.properties[0].type == "list"
        assert param.properties[0].items == "string"


class TestInputParameterListOfObjects:
    """Test InputParameter with lists of typed objects."""

    def test_list_of_objects(self):
        """Test creating a list where each element is an object."""
        param = InputParameter(
            name="test_scenarios",
            type="list",
            items=InputParameter(
                name="scenario",
                type="object",
                properties=[
                    InputParameter(name="scenario_name", type="string", required=True),
                    InputParameter(name="num_iterations", type="integer", default=1),
                    InputParameter(
                        name="severity",
                        type="enum",
                        choices=["low", "medium", "high"],
                        default="medium",
                    ),
                ],
            ),
        )
        assert param.type == "list"
        assert isinstance(param.items, InputParameter)
        assert param.items.type == "object"
        assert param.items.properties is not None
        assert len(param.items.properties) == 3
        assert param.items.properties[0].name == "scenario_name"
        assert param.items.properties[2].type == "enum"


class TestInputParameterNestedLists:
    """Test InputParameter with nested lists (matrices)."""

    def test_nested_list_matrix(self):
        """Test creating a 2D matrix (list of lists)."""
        param = InputParameter(
            name="test_matrix",
            type="list",
            items=InputParameter(name="row", type="list", items="integer"),
        )
        assert param.type == "list"
        assert isinstance(param.items, InputParameter)
        assert param.items.type == "list"
        assert param.items.items == "integer"

    def test_triple_nested_list(self):
        """Test creating a 3D structure (list of list of enums)."""
        param = InputParameter(
            name="color_palettes",
            type="list",
            items=InputParameter(
                name="palette",
                type="list",
                items=InputParameter(
                    name="color",
                    type="enum",
                    choices=["red", "green", "blue", "yellow"],
                ),
            ),
        )
        assert param.type == "list"
        assert isinstance(param.items, InputParameter)
        assert param.items.type == "list"
        assert isinstance(param.items.items, InputParameter)
        assert param.items.items.type == "enum"


class TestInputParameterComplexNested:
    """Test complex nested structures combining all features."""

    def test_complex_nested_structure(self):
        """Test complex nested object with multiple levels."""
        param = InputParameter(
            name="evaluation_suite",
            type="object",
            properties=[
                InputParameter(name="suite_name", type="string", required=True),
                InputParameter(
                    name="parallel_execution", type="boolean", default=False
                ),
                InputParameter(
                    name="test_groups",
                    type="list",
                    items=InputParameter(
                        name="group",
                        type="object",
                        properties=[
                            InputParameter(
                                name="group_name", type="string", required=True
                            ),
                            InputParameter(
                                name="enabled", type="boolean", default=True
                            ),
                            InputParameter(
                                name="test_cases",
                                type="list",
                                items=InputParameter(
                                    name="test_case",
                                    type="object",
                                    properties=[
                                        InputParameter(
                                            name="test_id", type="string", required=True
                                        ),
                                        InputParameter(
                                            name="expected_outcome",
                                            type="enum",
                                            choices=["pass", "fail", "skip"],
                                            default="pass",
                                        ),
                                        InputParameter(
                                            name="tags", type="list", items="string"
                                        ),
                                    ],
                                ),
                            ),
                        ],
                    ),
                ),
            ],
        )
        # Verify top-level structure
        assert param.type == "object"
        assert param.properties is not None
        assert len(param.properties) == 3

        # Verify 2 levels deep: object -> list -> object
        test_groups = param.properties[2]
        assert test_groups.type == "list"
        assert isinstance(test_groups.items, InputParameter)
        assert test_groups.items.type == "object"

        # Verify 3 levels deep: object -> list -> object -> list -> object
        assert test_groups.items.properties is not None
        test_cases = test_groups.items.properties[2]
        assert test_cases.type == "list"
        assert isinstance(test_cases.items, InputParameter)
        assert test_cases.items.type == "object"

        # Verify deepest level has enum (4 levels deep)
        assert test_cases.items.properties is not None
        assert test_cases.items.properties[1].type == "enum"


class TestInputParameterDefaultValues:
    """Test default values for all parameter types."""

    def test_default_for_simple_types(self):
        """Test default values for simple scalar types."""
        params = [
            InputParameter(name="name", type="string", default="test"),
            InputParameter(name="count", type="integer", default=10),
            InputParameter(name="ratio", type="float", default=0.5),
            InputParameter(name="enabled", type="boolean", default=True),
        ]
        assert params[0].default == "test"
        assert params[1].default == 10
        assert params[2].default == 0.5
        assert params[3].default is True

    def test_default_for_list(self):
        """Test default value for list parameter."""
        param = InputParameter(
            name="items", type="list", items="string", default=["a", "b", "c"]
        )
        assert param.default == ["a", "b", "c"]

    def test_default_for_object(self):
        """Test default value for object parameter."""
        param = InputParameter(
            name="config",
            type="object",
            default={"timeout": 30, "retries": 3},
        )
        assert param.default == {"timeout": 30, "retries": 3}


class TestInputParameterBackwardCompatibility:
    """Test backward compatibility with legacy manifests."""

    def test_legacy_simple_parameter_still_works(self):
        """Test that old-style simple parameters work unchanged."""
        param = InputParameter(
            name="delay_seconds", type="integer", required=False, description="Delay"
        )
        assert param.name == "delay_seconds"
        assert param.type == "integer"
        assert param.required is False
        assert param.description == "Delay"
        # New fields should be None
        assert param.items is None
        assert param.properties is None
        assert param.choices is None
        assert param.default is None

    def test_legacy_untyped_list_still_works(self):
        """Test that untyped lists (without items) still work."""
        param = InputParameter(name="datasets", type="list", required=False)
        assert param.type == "list"
        assert param.items is None  # No type specified

    def test_legacy_untyped_object_still_works(self):
        """Test that untyped objects (without properties) still work."""
        param = InputParameter(name="config", type="object", required=False)
        assert param.type == "object"
        assert param.properties is None  # No schema specified


class TestInputParameterJSONSchema:
    """Test that InputParameter generates correct JSON schema."""

    def test_parameter_serialization(self):
        """Test that parameters serialize correctly."""
        param = InputParameter(
            name="mode",
            type="enum",
            choices=["fast", "slow"],
            default="fast",
            description="Execution mode",
        )
        param_dict = param.model_dump()
        assert param_dict["name"] == "mode"
        assert param_dict["type"] == "enum"
        assert param_dict["choices"] == ["fast", "slow"]
        assert param_dict["default"] == "fast"

    def test_nested_parameter_serialization(self):
        """Test that nested parameters serialize correctly."""
        param = InputParameter(
            name="config",
            type="object",
            properties=[
                InputParameter(name="timeout", type="integer", default=30),
            ],
        )
        param_dict = param.model_dump()
        assert param_dict["type"] == "object"
        assert len(param_dict["properties"]) == 1
        assert param_dict["properties"][0]["name"] == "timeout"
