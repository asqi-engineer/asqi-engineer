import uuid

import pytest
from asqi.schemas import (
    AnsweredImageEditingTestCase,
    AnsweredImageGenerationTestCase,
    AnsweredLLMTestCase,
    AnsweredObjectDetectionTestCase,
    AnsweredRAGTestCase,
    AnsweredVLMTestCase,
    BoundingBox,
    ContextualizedRAGTestCase,
    EmbeddingTestCase,
    ImageEditingTestCase,
    ImageGenerationTestCase,
    LLMTestCase,
    ObjectDetectionTestCase,
    ODImageEditingTestCase,
    RAGTestCase,
    TestCase,
    UnansweredImageEditingTestCase,
    UnansweredImageGenerationTestCase,
    UnansweredLLMTestCase,
    UnansweredObjectDetectionTestCase,
    UnansweredRAGTestCase,
    UnansweredVLMTestCase,
    VLMTestCase,
)
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_URI = "data:image/jpeg;base64,/9j/abc123"
IMAGE_URI_2 = "data:image/png;base64,iVBORw0KGgo="


# ---------------------------------------------------------------------------
# UnansweredLLMTestCase
# ---------------------------------------------------------------------------


class TestUnansweredLLMTestCase:
    def test_minimal_construction(self):
        tc = UnansweredLLMTestCase(query="Hello")
        assert tc.query == "Hello"
        assert tc.system_prompt is None
        assert getattr(tc, "answer", None) is None
        assert tc.scenario is None
        assert tc.extra_params == {}

    def test_lineage_id_auto_generated(self):
        tc = UnansweredLLMTestCase(query="x")
        uuid.UUID(tc.lineage_id)  # raises if not a valid UUID

    def test_lineage_id_explicit(self):
        fixed = str(uuid.uuid4())
        tc = UnansweredLLMTestCase(query="x", lineage_id=fixed)
        assert tc.lineage_id == fixed

    def test_request_with_system_prompt(self):
        tc = UnansweredLLMTestCase(
            query="Hello",
            system_prompt="Be concise",
            extra_params={"max_tokens": 100},
        )
        req = tc.request
        assert req["messages"][0] == {"role": "system", "content": "Be concise"}
        assert req["messages"][1] == {"role": "user", "content": "Hello"}
        assert req["max_tokens"] == 100

    def test_request_without_system_prompt(self):
        tc = UnansweredLLMTestCase(query="Hello")
        req = tc.request
        assert len(req["messages"]) == 1
        assert req["messages"][0]["role"] == "user"

    def test_no_expected_response_accessor(self):
        tc = UnansweredLLMTestCase(query="x")
        assert not hasattr(tc, "expected_response")

    def test_metadata_not_serialised(self):
        tc = UnansweredLLMTestCase(query="x", metadata={"debug": "info"})
        assert tc.get_debug_metadata() == {"debug": "info"}
        dumped = tc.model_dump()
        assert "metadata" not in dumped
        assert "_metadata" not in dumped

    def test_extra_fields_ignored(self):
        tc = UnansweredLLMTestCase(query="x", unknown_field="ignored")
        assert not hasattr(tc, "unknown_field")

    def test_missing_required_field_raises(self):
        with pytest.raises(ValidationError):
            UnansweredLLMTestCase()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# AnsweredLLMTestCase
# ---------------------------------------------------------------------------


class TestAnsweredLLMTestCase:
    def test_full_construction(self):
        tc = AnsweredLLMTestCase(
            query="Summarise this",
            system_prompt="You are a helpful assistant",
            answer="Here is a summary",
            scenario="summarisation",
            extra_params={"temperature": 0.7},
        )
        assert tc.system_prompt == "You are a helpful assistant"
        assert tc.answer == "Here is a summary"
        assert tc.scenario == "summarisation"
        assert tc.extra_params == {"temperature": 0.7}

    def test_expected_response_with_output(self):
        tc = AnsweredLLMTestCase(query="x", answer="The answer")
        resp = tc.expected_response
        assert resp is not None
        assert resp["choices"][0]["message"]["content"] == "The answer"
        assert resp["choices"][0]["message"]["role"] == "assistant"

    def test_from_api_data(self):
        request = {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
            ],
            "temperature": 0.5,
        }
        response = {"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}
        tc = AnsweredLLMTestCase.from_api_data(request, response, scenario="greeting")
        assert tc.query == "Hi"
        assert tc.system_prompt == "Be helpful"
        assert tc.answer == "Hello!"
        assert tc.scenario == "greeting"
        assert tc.extra_params == {"temperature": 0.5}

    def test_from_api_data_lineage_id_propagated(self):
        fixed = str(uuid.uuid4())
        request = {"messages": [{"role": "user", "content": "x"}]}
        response = {"choices": [{"message": {"role": "assistant", "content": "y"}}]}
        tc = AnsweredLLMTestCase.from_api_data(request, response, lineage_id=fixed)
        assert tc.lineage_id == fixed


# ---------------------------------------------------------------------------
# UnansweredRAGTestCase
# ---------------------------------------------------------------------------


class TestUnansweredRAGTestCase:
    def test_minimal_construction(self):
        tc = UnansweredRAGTestCase(query="What is X?")
        assert tc.query == "What is X?"
        assert getattr(tc, "answer", None) is None
        assert getattr(tc, "context", None) is None

    def test_request_with_system_prompt(self):
        tc = UnansweredRAGTestCase(query="What is X?", system_prompt="Answer briefly")
        req = tc.request
        assert req["messages"][0] == {"role": "system", "content": "Answer briefly"}
        assert req["messages"][1] == {"role": "user", "content": "What is X?"}

    def test_request_without_system_prompt(self):
        tc = UnansweredRAGTestCase(query="What is X?")
        req = tc.request
        assert len(req["messages"]) == 1
        assert req["messages"][0]["content"] == "What is X?"

    def test_no_expected_response_accessor(self):
        tc = UnansweredRAGTestCase(query="Q?")
        assert not hasattr(tc, "expected_response")

    def test_metadata_not_serialised(self):
        tc = UnansweredRAGTestCase(query="Q?", metadata={"source": "sdg"})
        assert tc.get_debug_metadata() == {"source": "sdg"}
        assert "metadata" not in tc.model_dump()

    def test_extra_fields_ignored(self):
        tc = UnansweredRAGTestCase(query="Q?", unknown_field="ignored")
        assert not hasattr(tc, "unknown_field")

    def test_lineage_id_auto_generated(self):
        tc = UnansweredRAGTestCase(query="Q?")
        uuid.UUID(tc.lineage_id)


# ---------------------------------------------------------------------------
# AnsweredRAGTestCase
# ---------------------------------------------------------------------------


class TestAnsweredRAGTestCase:
    def test_expected_response_with_answer_only(self):
        tc = AnsweredRAGTestCase(query="Q?", answer="A")
        resp = tc.expected_response
        assert resp is not None
        msg = resp["choices"][0]["message"]
        assert msg["content"] == "A"
        assert msg["context"]["citations"] == []

    def test_answer_required(self):
        with pytest.raises(ValidationError):
            AnsweredRAGTestCase(query="Q?")  # type: ignore[call-arg]

    def test_expected_response_none_when_empty_answer(self):
        tc = AnsweredRAGTestCase(query="Q?", answer="")
        assert tc.expected_response is None

    def test_from_api_data(self):
        request = {"messages": [{"role": "user", "content": "What is X?"}]}
        response = {
            "choices": [
                {
                    "message": {
                        "content": "X is Y",
                        "context": {"citations": [{"retrieved_context": "chunk1"}]},
                    }
                }
            ]
        }
        tc = AnsweredRAGTestCase.from_api_data(request, response)
        assert tc.query == "What is X?"
        assert tc.answer == "X is Y"


# ---------------------------------------------------------------------------
# ContextualizedRAGTestCase
# ---------------------------------------------------------------------------


class TestContextualizedRAGTestCase:
    def test_expected_response_with_contexts(self):
        tc = ContextualizedRAGTestCase(
            query="What is X?",
            context=["chunk1", "chunk2"],
        )
        resp = tc.expected_response
        assert resp is not None
        msg = resp["choices"][0]["message"]
        assert msg["content"] is None
        citations = msg["context"]["citations"]
        assert citations[0]["retrieved_context"] == "chunk1"
        assert citations[1]["retrieved_context"] == "chunk2"

    def test_expected_response_none_when_no_contexts(self):
        tc = ContextualizedRAGTestCase(query="Q?", context=[])
        assert tc.expected_response is None

    def test_from_api_data(self):
        request = {"messages": [{"role": "user", "content": "What is X?"}]}
        response = {
            "choices": [
                {
                    "message": {
                        "content": "ignored for this schema",
                        "context": {"citations": [{"retrieved_context": "chunk1"}]},
                    }
                }
            ]
        }
        tc = ContextualizedRAGTestCase.from_api_data(request, response)
        assert tc.query == "What is X?"
        assert tc.context == ["chunk1"]


# ---------------------------------------------------------------------------
# UnansweredVLMTestCase
# ---------------------------------------------------------------------------


class TestUnansweredVLMTestCase:
    def test_minimal_construction(self):
        tc = UnansweredVLMTestCase(query="What is in this image?", images=[IMAGE_URI])
        assert tc.query == "What is in this image?"
        assert tc.images == [IMAGE_URI]
        assert getattr(tc, "answer", None) is None

    def test_multiple_images(self):
        tc = UnansweredVLMTestCase(query="Compare these images", images=[IMAGE_URI, IMAGE_URI_2])
        assert len(tc.images) == 2

    def test_images_required_non_empty(self):
        with pytest.raises(ValidationError):
            UnansweredVLMTestCase(query="Q?", images=[])

    def test_request_structure(self):
        tc = UnansweredVLMTestCase(
            query="Describe this",
            images=[IMAGE_URI],
            system_prompt="Be concise",
            extra_params={"max_tokens": 50},
        )
        req = tc.request
        assert req["supports_vision"] is True
        assert req["max_tokens"] == 50
        messages = req["messages"]
        assert messages[0] == {"role": "system", "content": "Be concise"}
        user_content = messages[1]["content"]
        assert user_content[0] == {"type": "text", "text": "Describe this"}
        assert user_content[1] == {"type": "image_url", "image_url": {"url": IMAGE_URI}}

    def test_request_multiple_images(self):
        tc = UnansweredVLMTestCase(query="Q?", images=[IMAGE_URI, IMAGE_URI_2])
        content = tc.request["messages"][-1]["content"]
        image_items = [c for c in content if c["type"] == "image_url"]
        assert len(image_items) == 2

    def test_no_expected_response_accessor(self):
        tc = UnansweredVLMTestCase(query="Q?", images=[IMAGE_URI])
        assert not hasattr(tc, "expected_response")

    def test_extra_fields_ignored(self):
        tc = UnansweredVLMTestCase(query="Q?", images=[IMAGE_URI], extra_junk="ignored")
        assert not hasattr(tc, "extra_junk")

    def test_lineage_id_auto_generated(self):
        tc = UnansweredVLMTestCase(query="Q?", images=[IMAGE_URI])
        uuid.UUID(tc.lineage_id)


# ---------------------------------------------------------------------------
# AnsweredVLMTestCase
# ---------------------------------------------------------------------------


class TestAnsweredVLMTestCase:
    def test_expected_response_with_output(self):
        tc = AnsweredVLMTestCase(query="Q?", images=[IMAGE_URI], answer="A cat")
        resp = tc.expected_response
        assert resp is not None
        assert resp["choices"][0]["message"]["content"] == "A cat"

    def test_from_api_data(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": IMAGE_URI}},
                    ],
                }
            ],
            "supports_vision": True,
        }
        response = {"choices": [{"message": {"role": "assistant", "content": "A cat"}}]}
        tc = AnsweredVLMTestCase.from_api_data(request, response)
        assert tc.query == "What is this?"
        assert tc.images == [IMAGE_URI]
        assert tc.answer == "A cat"


# ---------------------------------------------------------------------------
# ImageGenerationTestCase / AnsweredImageGenerationTestCase
# ---------------------------------------------------------------------------


class TestImageGenerationTestCase:
    def test_minimal_base_construction(self):
        tc = ImageGenerationTestCase(prompt="A sunset over the ocean", size="", n=1)
        assert tc.prompt == "A sunset over the ocean"
        assert tc.n == 1
        assert tc.size == ""

    def test_unanswered_alias(self):
        tc = UnansweredImageGenerationTestCase(prompt="x", size="", n=1)
        assert tc.prompt == "x"

    def test_full_construction(self):
        tc = ImageGenerationTestCase(prompt="A cat", size="1024x1024", n=2, scenario="text-to-image")
        assert tc.size == "1024x1024"
        assert tc.n == 2

    def test_request_with_size(self):
        tc = ImageGenerationTestCase(prompt="A cat", size="512x512", n=3)
        req = tc.request
        assert req == {"prompt": "A cat", "n": 3, "size": "512x512"}

    def test_request_without_size(self):
        tc = ImageGenerationTestCase(prompt="A cat", size="", n=1)
        req = tc.request
        assert "size" not in req
        assert req == {"prompt": "A cat", "n": 1}

    def test_request_includes_extra_params(self):
        tc = ImageGenerationTestCase(
            prompt="A cat",
            size="",
            n=1,
            extra_params={"style": "vivid"},
        )
        assert tc.request["style"] == "vivid"

    def test_from_api_data(self):
        request = {"prompt": "A cat", "n": 2, "size": "512x512", "style": "natural"}
        tc = ImageGenerationTestCase.from_api_data(request)
        assert tc.prompt == "A cat"
        assert tc.n == 2
        assert tc.size == "512x512"
        assert tc.extra_params == {"style": "natural"}

    def test_metadata_not_serialised(self):
        tc = ImageGenerationTestCase(prompt="x", size="", n=1, metadata={"gen": "sdg"})
        assert tc.get_debug_metadata() == {"gen": "sdg"}
        assert "metadata" not in tc.model_dump()

    def test_extra_fields_ignored(self):
        tc = ImageGenerationTestCase(prompt="x", size="", n=1, unknown_field="ignored")
        assert not hasattr(tc, "unknown_field")

    def test_lineage_id_auto_generated(self):
        tc = ImageGenerationTestCase(prompt="x", size="", n=1)
        uuid.UUID(tc.lineage_id)


class TestAnsweredImageGenerationTestCase:
    def test_expected_response_includes_generation_image(self):
        tc = AnsweredImageGenerationTestCase(prompt="A cat", size="", n=1, generation=IMAGE_URI_2)
        assert tc.expected_response == {"image": IMAGE_URI_2}

    def test_from_api_data(self):
        request = {"prompt": "A cat", "n": 2, "size": "512x512", "style": "natural"}
        tc = AnsweredImageGenerationTestCase.from_api_data(request, generation=IMAGE_URI_2)
        assert tc.prompt == "A cat"
        assert tc.n == 2
        assert tc.size == "512x512"
        assert tc.generation == IMAGE_URI_2
        assert tc.extra_params == {"style": "natural"}

    def test_metadata_not_serialised(self):
        tc = AnsweredImageGenerationTestCase(prompt="x", size="", n=1, generation=IMAGE_URI_2, metadata={"gen": "sdg"})
        assert tc.get_debug_metadata() == {"gen": "sdg"}
        assert "metadata" not in tc.model_dump()


# ---------------------------------------------------------------------------
# ImageEditingTestCase / AnsweredImageEditingTestCase
# ---------------------------------------------------------------------------


class TestImageEditingTestCase:
    def test_minimal_construction(self):
        tc = ImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="Remove the background",
        )
        assert tc.image == IMAGE_URI
        assert tc.edit_prompt == "Remove the background"
        assert tc.mask is None

    def test_unanswered_alias(self):
        tc = UnansweredImageEditingTestCase(image=IMAGE_URI, edit_prompt="x")
        assert tc.mask is None

    def test_with_mask(self):
        tc = ImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="Make it blue",
            mask=IMAGE_URI_2,
        )
        assert tc.mask == IMAGE_URI_2

    def test_request_without_mask(self):
        tc = ImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="Make it blue",
        )
        req = tc.request
        assert req == {"image": IMAGE_URI, "prompt": "Make it blue"}
        assert "mask" not in req

    def test_request_with_mask(self):
        tc = ImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="Make it blue",
            mask=IMAGE_URI_2,
        )
        req = tc.request
        assert req["mask"] == IMAGE_URI_2

    def test_request_includes_extra_params(self):
        tc = ImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="x",
            extra_params={"model": "dall-e-2"},
        )
        assert tc.request["model"] == "dall-e-2"

    def test_from_api_data_without_mask(self):
        request = {"image": IMAGE_URI, "prompt": "Make it blue"}
        tc = ImageEditingTestCase.from_api_data(request)
        assert tc.image == IMAGE_URI
        assert tc.edit_prompt == "Make it blue"
        assert tc.mask is None

    def test_from_api_data_with_mask(self):
        request = {"image": IMAGE_URI, "prompt": "Fill in", "mask": IMAGE_URI_2}
        tc = ImageEditingTestCase.from_api_data(request)
        assert tc.mask == IMAGE_URI_2
        assert tc.extra_params == {}

    def test_metadata_not_serialised(self):
        tc = ImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="x",
            metadata={"k": "v"},
        )
        assert tc.get_debug_metadata() == {"k": "v"}
        assert "metadata" not in tc.model_dump()

    def test_extra_fields_ignored(self):
        tc = ImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="x",
            extra_junk="ignored",
        )
        assert not hasattr(tc, "extra_junk")

    def test_lineage_id_auto_generated(self):
        tc = ImageEditingTestCase(image=IMAGE_URI, edit_prompt="x")
        uuid.UUID(tc.lineage_id)


class TestODImageEditingTestCase:
    def test_minimal_with_detections(self):
        tc = ODImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="add object",
            expected_detections=[
                BoundingBox(xyxy=(10.0, 20.0, 100.0, 200.0), class_name="person"),
            ],
        )
        assert len(tc.expected_detections) == 1
        assert tc.expected_detections[0].class_name == "person"
        assert tc.edit_prompt == "add object"

    def test_empty_detections_allowed(self):
        tc = ODImageEditingTestCase(image=IMAGE_URI, edit_prompt="x", expected_detections=[])
        assert tc.expected_detections == []


class TestAnsweredImageEditingTestCase:
    def test_expected_response_with_generation(self):
        tc = AnsweredImageEditingTestCase(
            image=IMAGE_URI,
            edit_prompt="x",
            generation=IMAGE_URI_2,
        )
        assert tc.expected_response == {"image": IMAGE_URI_2}

    def test_from_api_data_without_mask(self):
        request = {"image": IMAGE_URI, "prompt": "Make it blue"}
        tc = AnsweredImageEditingTestCase.from_api_data(request, generation=IMAGE_URI_2)
        assert tc.image == IMAGE_URI
        assert tc.edit_prompt == "Make it blue"
        assert tc.mask is None
        assert tc.generation == IMAGE_URI_2

    def test_from_api_data_with_mask(self):
        request = {"image": IMAGE_URI, "prompt": "Fill in", "mask": IMAGE_URI_2}
        tc = AnsweredImageEditingTestCase.from_api_data(request, generation="")
        assert tc.mask == IMAGE_URI_2
        assert tc.extra_params == {}


# ---------------------------------------------------------------------------
# EmbeddingTestCase
# ---------------------------------------------------------------------------


class TestEmbeddingTestCase:
    def test_minimal_construction(self):
        tc = EmbeddingTestCase(
            text="Hello world",
            expected_similar_texts=[],
            expected_dissimilar_texts=[],
        )
        assert tc.text == "Hello world"
        assert tc.expected_similar_texts == []
        assert tc.expected_dissimilar_texts == []

    def test_full_construction(self):
        tc = EmbeddingTestCase(
            text="Machine learning",
            expected_similar_texts=["deep learning", "neural networks"],
            expected_dissimilar_texts=["cooking recipes"],
            scenario="semantic-similarity",
        )
        assert tc.expected_similar_texts == ["deep learning", "neural networks"]
        assert tc.expected_dissimilar_texts == ["cooking recipes"]

    def test_request(self):
        tc = EmbeddingTestCase(
            text="Hello world",
            expected_similar_texts=[],
            expected_dissimilar_texts=[],
            extra_params={"model": "text-ada-002"},
        )
        assert tc.request == {"input": "Hello world", "model": "text-ada-002"}

    def test_expected_response_always_none(self):
        tc = EmbeddingTestCase(
            text="x",
            expected_similar_texts=["y"],
            expected_dissimilar_texts=["z"],
        )
        assert tc.expected_response is None

    def test_from_api_data(self):
        request = {"input": "Hello world", "model": "text-ada-002"}
        tc = EmbeddingTestCase.from_api_data(
            request,
            expected_similar_texts=["Hi there"],
            expected_dissimilar_texts=["Quantum physics"],
        )
        assert tc.text == "Hello world"
        assert tc.expected_similar_texts == ["Hi there"]
        assert tc.expected_dissimilar_texts == ["Quantum physics"]
        assert tc.extra_params == {"model": "text-ada-002"}

    def test_from_api_data_defaults_empty_lists(self):
        tc = EmbeddingTestCase.from_api_data({"input": "x"})
        assert tc.expected_similar_texts == []
        assert tc.expected_dissimilar_texts == []

    def test_metadata_not_serialised(self):
        tc = EmbeddingTestCase(
            text="x",
            expected_similar_texts=[],
            expected_dissimilar_texts=[],
            metadata={"src": "sdg"},
        )
        assert tc.get_debug_metadata() == {"src": "sdg"}
        assert "metadata" not in tc.model_dump()

    def test_extra_fields_ignored(self):
        tc = EmbeddingTestCase(
            text="x",
            expected_similar_texts=[],
            expected_dissimilar_texts=[],
            unknown_field="ignored",
        )
        assert not hasattr(tc, "unknown_field")

    def test_lineage_id_auto_generated(self):
        tc = EmbeddingTestCase(text="x", expected_similar_texts=[], expected_dissimilar_texts=[])
        uuid.UUID(tc.lineage_id)


# ---------------------------------------------------------------------------
# BoundingBox
# ---------------------------------------------------------------------------


class TestBoundingBox:
    def test_construction(self):
        bb = BoundingBox(xyxy=(10.0, 20.0, 100.0, 200.0), class_name="person")
        assert bb.xyxy == (10.0, 20.0, 100.0, 200.0)
        assert bb.class_name == "person"
        assert bb.confidence is None

    def test_with_confidence(self):
        bb = BoundingBox(xyxy=(0.0, 0.0, 1.0, 1.0), class_name="car", confidence=0.95)
        assert bb.confidence == 0.95

    def test_missing_xyxy_raises(self):
        with pytest.raises(ValidationError):
            BoundingBox(class_name="person")  # type: ignore[call-arg]

    def test_missing_class_name_raises(self):
        with pytest.raises(ValidationError):
            BoundingBox(xyxy=(0.0, 0.0, 1.0, 1.0))  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ObjectDetectionTestCase / AnsweredObjectDetectionTestCase
# ---------------------------------------------------------------------------


class TestObjectDetectionTestCase:
    def test_minimal_base_construction(self):
        tc = ObjectDetectionTestCase(image=IMAGE_URI)
        assert tc.image == IMAGE_URI

    def test_unanswered_alias(self):
        tc = UnansweredObjectDetectionTestCase(image=IMAGE_URI)
        assert tc.image == IMAGE_URI

    def test_request(self):
        tc = ObjectDetectionTestCase(
            image=IMAGE_URI,
            extra_params={"model": "facebook/detr-resnet-50"},
        )
        assert tc.request == {"image": IMAGE_URI, "model": "facebook/detr-resnet-50"}

    def test_metadata_not_serialised(self):
        tc = ObjectDetectionTestCase(image=IMAGE_URI, metadata={"k": "v"})
        assert tc.get_debug_metadata() == {"k": "v"}
        assert "metadata" not in tc.model_dump()

    def test_extra_fields_ignored(self):
        tc = ObjectDetectionTestCase(image=IMAGE_URI, unknown_field="ignored")
        assert not hasattr(tc, "unknown_field")

    def test_lineage_id_auto_generated(self):
        tc = ObjectDetectionTestCase(image=IMAGE_URI)
        uuid.UUID(tc.lineage_id)


class TestAnsweredObjectDetectionTestCase:
    def test_minimal_construction(self):
        tc = AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[])
        assert tc.image == IMAGE_URI
        assert tc.expected_detections == []
        assert tc.confidence_threshold is None

    def test_with_detections(self):
        bb = BoundingBox(xyxy=(0.0, 0.0, 50.0, 50.0), class_name="cat")
        tc = AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[bb])
        assert len(tc.expected_detections) == 1
        assert tc.expected_detections[0].class_name == "cat"

    def test_request(self):
        tc = AnsweredObjectDetectionTestCase(
            image=IMAGE_URI,
            expected_detections=[],
            extra_params={"model": "facebook/detr-resnet-50"},
        )
        assert tc.request == {"image": IMAGE_URI, "model": "facebook/detr-resnet-50"}

    def test_expected_response_empty_detections(self):
        tc = AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[])
        assert tc.expected_response == {"detections": []}

    def test_expected_response_with_detections(self):
        bb1 = BoundingBox(xyxy=(0.0, 0.0, 10.0, 10.0), class_name="cat", confidence=0.9)
        bb2 = BoundingBox(xyxy=(20.0, 20.0, 50.0, 50.0), class_name="dog")
        tc = AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[bb1, bb2])
        resp = tc.expected_response
        dets = resp["detections"]
        assert dets[0] == {"xyxy": [0.0, 0.0, 10.0, 10.0], "class_name": "cat", "confidence": 0.9}
        assert dets[1] == {"xyxy": [20.0, 20.0, 50.0, 50.0], "class_name": "dog"}
        assert "confidence" not in dets[1]

    def test_from_api_data(self):
        request = {"image": IMAGE_URI, "provider": "huggingface"}
        response = {
            "detections": [
                {"xyxy": [0.0, 0.0, 10.0, 10.0], "class_name": "cat", "confidence": 0.9},
                {"xyxy": [20.0, 20.0, 50.0, 50.0], "class_name": "dog"},
            ]
        }
        tc = AnsweredObjectDetectionTestCase.from_api_data(request, response, scenario="low-light")
        assert tc.image == IMAGE_URI
        assert tc.scenario == "low-light"
        assert len(tc.expected_detections) == 2
        assert tc.expected_detections[0].class_name == "cat"
        assert tc.expected_detections[0].confidence == 0.9
        assert tc.expected_detections[1].confidence is None
        assert tc.extra_params == {"provider": "huggingface"}

    def test_metadata_not_serialised(self):
        tc = AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[], metadata={"k": "v"})
        assert tc.get_debug_metadata() == {"k": "v"}
        assert "metadata" not in tc.model_dump()

    def test_extra_fields_ignored(self):
        tc = AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[], unknown_field="ignored")
        assert not hasattr(tc, "unknown_field")

    def test_lineage_id_auto_generated(self):
        tc = AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[])
        uuid.UUID(tc.lineage_id)


# ---------------------------------------------------------------------------
# TestCase union
# ---------------------------------------------------------------------------


class TestTestCaseUnion:
    def test_all_types_are_in_union(self):
        from typing import get_args

        union_types = get_args(TestCase)
        assert LLMTestCase in union_types
        assert AnsweredLLMTestCase in union_types
        assert RAGTestCase in union_types
        assert AnsweredRAGTestCase in union_types
        assert ContextualizedRAGTestCase in union_types
        assert VLMTestCase in union_types
        assert AnsweredVLMTestCase in union_types
        assert ImageGenerationTestCase in union_types
        assert AnsweredImageGenerationTestCase in union_types
        assert ImageEditingTestCase in union_types
        assert AnsweredImageEditingTestCase in union_types
        assert ODImageEditingTestCase in union_types
        assert EmbeddingTestCase in union_types
        assert ObjectDetectionTestCase in union_types
        assert AnsweredObjectDetectionTestCase in union_types

    def test_inheritance_chain(self):
        assert issubclass(UnansweredLLMTestCase, LLMTestCase)
        assert issubclass(AnsweredLLMTestCase, LLMTestCase)
        assert issubclass(RAGTestCase, LLMTestCase)
        assert issubclass(UnansweredRAGTestCase, RAGTestCase)
        assert issubclass(AnsweredRAGTestCase, RAGTestCase)
        assert issubclass(ContextualizedRAGTestCase, RAGTestCase)
        assert issubclass(VLMTestCase, LLMTestCase)
        assert issubclass(UnansweredVLMTestCase, VLMTestCase)
        assert issubclass(AnsweredVLMTestCase, VLMTestCase)
        assert issubclass(UnansweredObjectDetectionTestCase, ObjectDetectionTestCase)
        assert issubclass(AnsweredObjectDetectionTestCase, ObjectDetectionTestCase)
        assert issubclass(UnansweredImageGenerationTestCase, ImageGenerationTestCase)
        assert issubclass(AnsweredImageGenerationTestCase, ImageGenerationTestCase)
        assert issubclass(UnansweredImageEditingTestCase, ImageEditingTestCase)
        assert issubclass(AnsweredImageEditingTestCase, ImageEditingTestCase)
        assert issubclass(ODImageEditingTestCase, ImageEditingTestCase)

    def test_each_concrete_type_satisfies_union(self):
        instances: list[TestCase] = [  # type: ignore[assignment]
            UnansweredLLMTestCase(query="x"),
            AnsweredLLMTestCase(query="x", answer="y"),
            UnansweredRAGTestCase(query="x"),
            AnsweredRAGTestCase(query="x", answer="y"),
            ContextualizedRAGTestCase(query="x", context=["c"]),
            UnansweredVLMTestCase(query="x", images=[IMAGE_URI]),
            AnsweredVLMTestCase(query="x", images=[IMAGE_URI], answer="y"),
            UnansweredImageGenerationTestCase(prompt="x", size="", n=1),
            AnsweredImageGenerationTestCase(prompt="x", size="", n=1, generation=IMAGE_URI_2),
            UnansweredImageEditingTestCase(image=IMAGE_URI, edit_prompt="x"),
            AnsweredImageEditingTestCase(image=IMAGE_URI, edit_prompt="x", generation=IMAGE_URI_2),
            ODImageEditingTestCase(
                image=IMAGE_URI,
                edit_prompt="x",
                expected_detections=[BoundingBox(xyxy=(0.0, 0.0, 1.0, 1.0), class_name="thing")],
            ),
            EmbeddingTestCase(text="x", expected_similar_texts=[], expected_dissimilar_texts=[]),
            UnansweredObjectDetectionTestCase(image=IMAGE_URI),
            AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[]),
        ]
        assert len(instances) == 15


# ---------------------------------------------------------------------------
# Cross-cutting: lineage propagation
# ---------------------------------------------------------------------------


class TestLineagePropagation:
    """Verify that lineage_id is stable and can be explicitly propagated."""

    def test_each_type_generates_unique_lineage_id(self):
        ids = {
            UnansweredLLMTestCase(query="x").lineage_id,
            AnsweredLLMTestCase(query="x", answer="y").lineage_id,
            UnansweredRAGTestCase(query="x").lineage_id,
            AnsweredRAGTestCase(query="x", answer="y").lineage_id,
            ContextualizedRAGTestCase(query="x", context=["c"]).lineage_id,
            UnansweredVLMTestCase(query="x", images=[IMAGE_URI]).lineage_id,
            AnsweredVLMTestCase(query="x", images=[IMAGE_URI], answer="y").lineage_id,
            UnansweredImageGenerationTestCase(prompt="x", size="", n=1).lineage_id,
            AnsweredImageGenerationTestCase(prompt="x", size="", n=1, generation=IMAGE_URI_2).lineage_id,
            UnansweredImageEditingTestCase(image=IMAGE_URI, edit_prompt="x").lineage_id,
            AnsweredImageEditingTestCase(image=IMAGE_URI, edit_prompt="x", generation=IMAGE_URI_2).lineage_id,
            ODImageEditingTestCase(
                image=IMAGE_URI,
                edit_prompt="x",
                expected_detections=[BoundingBox(xyxy=(0.0, 0.0, 1.0, 1.0), class_name="thing")],
            ).lineage_id,
            EmbeddingTestCase(text="x", expected_similar_texts=[], expected_dissimilar_texts=[]).lineage_id,
            UnansweredObjectDetectionTestCase(image=IMAGE_URI).lineage_id,
            AnsweredObjectDetectionTestCase(image=IMAGE_URI, expected_detections=[]).lineage_id,
        }
        assert len(ids) == 15  # all distinct

    def test_explicit_lineage_id_preserved_across_types(self):
        shared_id = str(uuid.uuid4())
        cases = [
            UnansweredLLMTestCase(query="x", lineage_id=shared_id),
            UnansweredRAGTestCase(query="x", lineage_id=shared_id),
            UnansweredVLMTestCase(query="x", images=[IMAGE_URI], lineage_id=shared_id),
        ]
        for tc in cases:
            assert tc.lineage_id == shared_id
