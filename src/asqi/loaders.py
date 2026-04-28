import logging
from collections.abc import Generator
from pathlib import Path

from asqi.schemas import HFDatasetDefinition
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def _resolve_path(path: str | Path, input_mount_path: Path | None) -> Path:
    resolved = Path(path)
    if input_mount_path and not resolved.is_absolute():
        resolved = input_mount_path / resolved
    return resolved


def _format_validation_error(
    exc: ValidationError, row_idx: int, class_name: str
) -> str:
    errors = exc.errors()
    field_msgs = []
    for error in errors:
        loc_parts = [str(loc) for loc in error["loc"]] if error["loc"] else []
        field = " -> ".join(loc_parts) if loc_parts else "root"
        field_msgs.append(f"field '{field}': {error['msg']}")
    return f"Row {row_idx} failed validation against {class_name}: " + "; ".join(
        field_msgs
    )


def load_test_cases[T: BaseModel](
    path: str | Path | dict | HFDatasetDefinition,
    test_case_class: type[T],
    *,
    input_mount_path: Path | None = None,
) -> Generator[T, None, None]:
    """Load and validate test cases from a dataset, yielding one typed instance per row.

    Accepts either a plain file path **or** an ``HFDatasetDefinition`` config dict
    (the same format produced by ``rag_datasets.yaml``).  Dataset configs require
    **no changes** when migrating containers to use this function.

    Supported file formats (when ``path`` is a file path):
    JSONL (.jsonl), JSON (.json), CSV (.csv), Parquet (.parquet).

    When ``path`` is an ``HFDatasetDefinition`` / dict, all formats and sources
    are available (HuggingFace Hub, local files, column mapping via the config's
    ``mapping`` field).

    **Multiple schemas from one dataset**

    A single dataset row can be valid for more than one TestCase schema.  For
    example, a RAG dataset with ``query``, ``answer``, and ``context`` columns can
    validate against both ``AnsweredRAGTestCase`` (needs answer only) and
    ``ContextualizedRAGTestCase`` (needs context only).  Call ``load_test_cases``
    once per schema — each call is independent and validates only the fields its
    schema requires::

        p = Path("/input")
        answered = list(load_test_cases(config, AnsweredRAGTestCase, input_mount_path=p))
        contextual = list(load_test_cases(config, ContextualizedRAGTestCase, input_mount_path=p))

    If a row is missing a field required by the schema, validation fails for that
    row and a ``ValueError`` is raised with the row index and the failing field name.

    Args:
        path: One of:

            * ``str | Path`` — direct path to a local dataset file.
            * ``dict | HFDatasetDefinition`` — dataset config from
              ``rag_datasets.yaml``.  Column ``mapping`` and Hub/local loading
              are handled automatically.

        test_case_class: Pydantic model to validate each row against, e.g.
            ``AnsweredRAGTestCase``, ``LLMTestCase``, or a per-container schema.
        input_mount_path: Optional base directory prepended to relative file
            paths.

    Yields:
        Validated instances of ``test_case_class``, one per dataset row.

    Raises:
        FileNotFoundError: If the resolved file path does not exist.
        ValueError: If a row fails schema validation (includes row index and
            field names).

    Examples::

        # Plain file path
        from asqi.loaders import load_test_cases
        from asqi.schemas import AnsweredRAGTestCase

        for tc in load_test_cases("dataset.jsonl", AnsweredRAGTestCase):
            print(tc.query, tc.answer)

        # HFDatasetDefinition config dict — rag_datasets.yaml stays unchanged
        accuracy_rows = list(
            load_test_cases(
                dataset_config["accuracy_dataset"],
                AnsweredRAGTestCase,
                input_mount_path=input_mount_path,
            )
        )
    """
    if isinstance(path, (dict, HFDatasetDefinition)):
        from asqi.datasets import _load_from_hub, _load_from_local

        config = HFDatasetDefinition(**path) if isinstance(path, dict) else path
        loader_params = config.loader_params

        if loader_params.hub_path is not None:
            dataset = _load_from_hub(loader_params)
        else:
            dataset = _load_from_local(loader_params, input_mount_path)

        if config.mapping:
            dataset = dataset.rename_columns(config.mapping)
    else:
        from datasets import load_dataset

        resolved = _resolve_path(path, input_mount_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset file not found: {resolved}")

        _ext_to_builder = {".jsonl": "json", ".json": "json"}
        builder = _ext_to_builder.get(
            resolved.suffix.lower(), resolved.suffix.lower().lstrip(".")
        )

        dataset = load_dataset(path=builder, data_files=str(resolved), split="train")

    for row_idx, row in enumerate(dataset):
        try:
            yield test_case_class(**row)
        except ValidationError as exc:
            raise ValueError(
                _format_validation_error(exc, row_idx, test_case_class.__name__)
            ) from exc
