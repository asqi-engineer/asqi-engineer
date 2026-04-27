import json
from pathlib import Path
from typing import Any

import pytest
from asqi.loaders import load_test_cases
from pydantic import BaseModel, ConfigDict


class LLMTestCase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str
    system_prompt: str | None = None
    scenario: str | None = None
    extra_params: dict[str, Any] = {}


class AnsweredLLMTestCase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str
    answer: str
    system_prompt: str | None = None
    scenario: str | None = None
    extra_params: dict[str, Any] = {}


class AnsweredRAGTestCase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str
    answer: str
    system_prompt: str | None = None
    scenario: str | None = None
    extra_params: dict[str, Any] = {}


class ContextualizedRAGTestCase(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str
    context: list[str]
    system_prompt: str | None = None
    scenario: str | None = None
    extra_params: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# LLM test cases — JSONL, JSON, CSV formats
# ---------------------------------------------------------------------------


class TestLoadTestCasesLLMJsonl:
    def test_valid_answered_llm(self, tmp_path: Path):
        rows = [
            {"query": "What is 2+2?", "answer": "4"},
            {"query": "Capital of France?", "answer": "Paris"},
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))

        cases = list(load_test_cases(path, AnsweredLLMTestCase))
        assert len(cases) == 2
        assert cases[0].query == "What is 2+2?"
        assert cases[0].answer == "4"
        assert cases[1].query == "Capital of France?"
        assert cases[1].answer == "Paris"

    def test_valid_query_only_llm(self, tmp_path: Path):
        rows = [{"query": "Simple question"}]
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(rows[0]))

        cases = list(load_test_cases(path, LLMTestCase))
        assert len(cases) == 1
        assert cases[0].query == "Simple question"
        assert cases[0].system_prompt is None

    def test_optional_fields_preserved(self, tmp_path: Path):
        row = {
            "query": "Q",
            "answer": "A",
            "system_prompt": "Be helpful",
            "scenario": "factual",
        }
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(row))

        cases = list(load_test_cases(path, AnsweredLLMTestCase))
        assert cases[0].system_prompt == "Be helpful"
        assert cases[0].scenario == "factual"

    def test_extra_fields_ignored(self, tmp_path: Path):
        row = {"query": "Q", "answer": "A", "unknown_field": "ignored"}
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(row))

        cases = list(load_test_cases(path, AnsweredLLMTestCase))
        assert len(cases) == 1
        assert cases[0].query == "Q"

    def test_yields_instances_of_correct_type(self, tmp_path: Path):
        row = {"query": "Q", "answer": "A"}
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(row))

        cases = list(load_test_cases(path, AnsweredLLMTestCase))
        assert isinstance(cases[0], AnsweredLLMTestCase)


class TestLoadTestCasesLLMJson:
    def test_valid_json_array(self, tmp_path: Path):
        rows = [
            {"query": "Hello", "answer": "World"},
            {"query": "Foo", "answer": "Bar"},
        ]
        path = tmp_path / "data.json"
        path.write_text(json.dumps(rows))

        cases = list(load_test_cases(path, AnsweredLLMTestCase))
        assert len(cases) == 2
        assert cases[0].query == "Hello"
        assert cases[1].query == "Foo"


class TestLoadTestCasesLLMCsv:
    def test_valid_csv(self, tmp_path: Path):
        path = tmp_path / "data.csv"
        path.write_text("query,answer\nHello,World\nFoo,Bar")

        cases = list(load_test_cases(path, AnsweredLLMTestCase))
        assert len(cases) == 2
        assert cases[0].query == "Hello"
        assert cases[0].answer == "World"
        assert cases[1].query == "Foo"
        assert cases[1].answer == "Bar"


# ---------------------------------------------------------------------------
# RAG test cases
# ---------------------------------------------------------------------------


class TestLoadTestCasesRAG:
    def test_valid_answered_rag_jsonl(self, tmp_path: Path):
        rows = [
            {"query": "What is RAG?", "answer": "Retrieval Augmented Generation"},
            {"query": "How does it work?", "answer": "It retrieves documents"},
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))

        cases = list(load_test_cases(path, AnsweredRAGTestCase))
        assert len(cases) == 2
        assert cases[0].query == "What is RAG?"
        assert cases[0].answer == "Retrieval Augmented Generation"
        assert isinstance(cases[0], AnsweredRAGTestCase)

    def test_rag_optional_system_prompt(self, tmp_path: Path):
        row = {"query": "Q", "answer": "A", "system_prompt": "You are a RAG assistant"}
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(row))

        cases = list(load_test_cases(path, AnsweredRAGTestCase))
        assert cases[0].system_prompt == "You are a RAG assistant"

    def test_contextualized_rag(self, tmp_path: Path):
        row = {"query": "Q", "context": ["chunk 1", "chunk 2"]}
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(row))

        cases = list(load_test_cases(path, ContextualizedRAGTestCase))
        assert len(cases) == 1
        assert cases[0].context == ["chunk 1", "chunk 2"]


# ---------------------------------------------------------------------------
# Validation errors — row number and field name in message
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_missing_required_field_row_0(self, tmp_path: Path):
        row = {"answer": "no query here"}  # missing required 'query'
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(row))

        with pytest.raises(ValueError, match="Row 0"):
            list(load_test_cases(path, AnsweredLLMTestCase))

    def test_missing_required_field_includes_field_name(self, tmp_path: Path):
        row = {"answer": "no query here"}  # 'query' is required
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(row))

        with pytest.raises(ValueError, match="'query'"):
            list(load_test_cases(path, AnsweredLLMTestCase))

    def test_invalid_row_second_row_reports_row_1(self, tmp_path: Path):
        rows = [
            {"query": "Valid", "answer": "OK"},
            {"answer": "Missing query field"},  # row 1 is invalid
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))

        with pytest.raises(ValueError, match="Row 1"):
            list(load_test_cases(path, AnsweredLLMTestCase))

    def test_rag_missing_answer_raises_with_row_number(self, tmp_path: Path):
        rows = [
            {"query": "Q1", "answer": "A1"},
            {"query": "Q2"},  # missing required 'answer' — row 1
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))

        with pytest.raises(ValueError, match="Row 1"):
            list(load_test_cases(path, AnsweredRAGTestCase))

    def test_first_valid_rows_yielded_before_error(self, tmp_path: Path):
        rows = [
            {"query": "OK1", "answer": "A"},
            {"query": "OK2", "answer": "B"},
            {"answer": "bad — no query"},  # row 2 invalid
        ]
        path = tmp_path / "data.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows))

        yielded = []
        with pytest.raises(ValueError, match="Row 2"):
            for tc in load_test_cases(path, AnsweredLLMTestCase):
                yielded.append(tc)

        assert len(yielded) == 2
        assert yielded[0].query == "OK1"
        assert yielded[1].query == "OK2"


# ---------------------------------------------------------------------------
# Parquet format
# ---------------------------------------------------------------------------


class TestLoadTestCasesParquet:
    def test_valid_parquet(self, tmp_path: Path):
        from datasets import Dataset as HFDataset

        data = {"query": ["Q1", "Q2"], "answer": ["A1", "A2"]}
        ds = HFDataset.from_dict(data)
        path = tmp_path / "data.parquet"
        ds.to_parquet(str(path))

        cases = list(load_test_cases(path, AnsweredLLMTestCase))
        assert len(cases) == 2
        assert cases[0].query == "Q1"
        assert cases[1].answer == "A2"


# ---------------------------------------------------------------------------
# File / format errors
# ---------------------------------------------------------------------------


class TestLoadTestCasesErrors:
    def test_file_not_found_raises(self, tmp_path: Path):
        path = tmp_path / "nonexistent.jsonl"

        with pytest.raises(FileNotFoundError):
            list(load_test_cases(path, LLMTestCase))


# ---------------------------------------------------------------------------
# input_mount_path resolution
# ---------------------------------------------------------------------------


class TestInputMountPath:
    def test_relative_path_resolved_against_mount(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test.jsonl").write_text(json.dumps({"query": "Q", "answer": "A"}))

        cases = list(
            load_test_cases(
                "data/test.jsonl",
                AnsweredLLMTestCase,
                input_mount_path=tmp_path,
            )
        )
        assert len(cases) == 1
        assert cases[0].query == "Q"

    def test_absolute_path_ignores_mount(self, tmp_path: Path):
        path = tmp_path / "test.jsonl"
        path.write_text(json.dumps({"query": "Q", "answer": "A"}))

        cases = list(
            load_test_cases(
                str(path),  # absolute
                AnsweredLLMTestCase,
                input_mount_path=Path("/some/other/dir"),  # should be ignored
            )
        )
        assert len(cases) == 1


# ---------------------------------------------------------------------------
# HFDatasetDefinition config dict — mirrors rag_datasets.yaml format
# ---------------------------------------------------------------------------


class TestLoadTestCasesFromConfig:
    """load_test_cases accepts HFDatasetDefinition dicts so rag_datasets.yaml
    can be passed directly, identical to how load_hf_dataset is called today."""

    def _parquet_config(self, tmp_path: Path, filename: str) -> dict:
        """Build a minimal HFDatasetDefinition dict pointing at a local parquet file."""
        return {
            "type": "huggingface",
            "loader_params": {
                "builder_name": "parquet",
                "data_files": filename,
            },
        }

    def _write_parquet(self, tmp_path: Path, data: dict, filename: str) -> Path:
        from datasets import Dataset as HFDataset

        ds = HFDataset.from_dict(data)
        path = tmp_path / filename
        ds.to_parquet(str(path))
        return path

    def test_config_dict_loads_parquet(self, tmp_path: Path):
        self._write_parquet(tmp_path, {"query": ["Q1", "Q2"], "answer": ["A1", "A2"]}, "data.parquet")
        config = self._parquet_config(tmp_path, "data.parquet")

        cases = list(load_test_cases(config, AnsweredLLMTestCase, input_mount_path=tmp_path))
        assert len(cases) == 2
        assert cases[0].query == "Q1"
        assert cases[1].answer == "A2"

    def test_config_dict_applies_column_mapping(self, tmp_path: Path):
        # Dataset has 'question' column; schema expects 'query'
        self._write_parquet(tmp_path, {"question": ["What is RAG?"], "answer": ["RAG"]}, "data.parquet")
        config = {
            "type": "huggingface",
            "loader_params": {"builder_name": "parquet", "data_files": "data.parquet"},
            "mapping": {"question": "query"},  # rename question → query
        }

        cases = list(load_test_cases(config, AnsweredLLMTestCase, input_mount_path=tmp_path))
        assert cases[0].query == "What is RAG?"
        assert cases[0].answer == "RAG"

    def test_config_dict_validates_rows(self, tmp_path: Path):
        # Row is missing required 'answer' — should raise with row number
        self._write_parquet(tmp_path, {"query": ["Q1", "Q2"]}, "data.parquet")
        config = self._parquet_config(tmp_path, "data.parquet")

        with pytest.raises(ValueError, match="Row 0"):
            list(load_test_cases(config, AnsweredLLMTestCase, input_mount_path=tmp_path))

    def test_config_dict_rag_with_context(self, tmp_path: Path):
        self._write_parquet(
            tmp_path,
            {"query": ["Q"], "context": [["ctx1", "ctx2"]]},
            "data.parquet",
        )
        config = self._parquet_config(tmp_path, "data.parquet")

        cases = list(load_test_cases(config, ContextualizedRAGTestCase, input_mount_path=tmp_path))
        assert cases[0].context == ["ctx1", "ctx2"]

    def test_hfdatasetdefinition_object_accepted(self, tmp_path: Path):
        from asqi.schemas import DatasetLoaderParams, HFDatasetDefinition

        self._write_parquet(tmp_path, {"query": ["Q"], "answer": ["A"]}, "data.parquet")
        config = HFDatasetDefinition(
            type="huggingface",
            loader_params=DatasetLoaderParams(builder_name="parquet", data_files="data.parquet"),
        )

        cases = list(load_test_cases(config, AnsweredLLMTestCase, input_mount_path=tmp_path))
        assert len(cases) == 1
        assert cases[0].query == "Q"


# ---------------------------------------------------------------------------
# Multiple schemas from the same dataset row
# ---------------------------------------------------------------------------


class TestMultipleSchemasSameDataset:
    """One dataset can be loaded with different schemas for different metrics.
    Each call validates only what its schema requires; invalid rows raise immediately."""

    def test_same_dataset_two_schemas(self, tmp_path: Path):
        # Dataset has query + answer + context — valid for both schemas
        rows = [{"query": "Q", "answer": "A", "context": ["ctx"]}]
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(rows[0]))

        answered = list(load_test_cases(path, AnsweredRAGTestCase))
        contextual = list(load_test_cases(path, ContextualizedRAGTestCase))

        assert answered[0].answer == "A"
        assert contextual[0].context == ["ctx"]
        # Both come from the same row
        assert answered[0].query == contextual[0].query == "Q"

    def test_schema_missing_required_field_raises(self, tmp_path: Path):
        # Dataset only has query + context — not valid for AnsweredRAGTestCase (needs answer)
        rows = [{"query": "Q", "context": ["ctx"]}]
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(rows[0]))

        # ContextualizedRAGTestCase succeeds
        cases = list(load_test_cases(path, ContextualizedRAGTestCase))
        assert len(cases) == 1

        # AnsweredRAGTestCase fails — 'answer' is required
        with pytest.raises(ValueError, match="Row 0"):
            list(load_test_cases(path, AnsweredRAGTestCase))
