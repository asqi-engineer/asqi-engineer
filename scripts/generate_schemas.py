import json
import sys
from pathlib import Path
from typing import Dict, Type

# Add src to path to import asqi modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pydantic import BaseModel

from asqi.schemas import Manifest, ScoreCard, SuiteConfig, SUTsConfig


def generate_schemas():
    """Generate JSON Schema files from Pydantic models. Intended to be used as part of the build process"""

    schemas_dir = Path(__file__).parent.parent / "src" / "asqi" / "schemas"
    schemas_dir.mkdir(parents=True, exist_ok=True)
    schema_mappings: Dict[Type[BaseModel], str] = {
        SUTsConfig: "asqi_suts_config.schema.json",
        SuiteConfig: "asqi_suite_config.schema.json",
        ScoreCard: "asqi_score_card.schema.json",
        Manifest: "asqi_manifest.schema.json",
    }

    for model_class, filename in schema_mappings.items():
        schema = model_class.model_json_schema(mode="serialization", by_alias=True)
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema["title"] = f"ASQI {model_class.__name__}"

        # Write schema file
        output_path = schemas_dir / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, ensure_ascii=False)

        print(f"{model_class.__name__} -> Generated {output_path}")

    print(f"\nSchemas generated in: {schemas_dir}")


if __name__ == "__main__":
    generate_schemas()
