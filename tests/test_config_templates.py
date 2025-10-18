from asqi.config_templates import TemplateType, generate_template, resolve_manifest_path
from asqi.schemas import InputParameter, Manifest, SystemInput


def _build_sample_manifest() -> Manifest:
    return Manifest(
        name="mock_tester",
        version="1.0.0",
        description="A mock tester manifest",
        input_systems=[
            SystemInput(name="system_under_test", type="llm_api", required=True),
            SystemInput(name="simulator_system", type="llm_api", required=False),
        ],
        input_schema=[
            InputParameter(name="generations", type="integer", required=False),
            InputParameter(name="temperature", type="float", required=False),
        ],
        output_metrics=[],
        output_artifacts=None,
    )


def test_generate_systems_template_with_manifest():
    manifest = _build_sample_manifest()
    template = generate_template(TemplateType.SYSTEMS, manifest=manifest)

    assert "system_under_test" in template.payload["systems"]
    assert "simulator_system" in template.payload["systems"]

    sut_entry = template.payload["systems"]["system_under_test"]
    assert sut_entry["type"] == "llm_api"
    assert template.optional_systems == ["simulator_system"]


def test_generate_suite_template_uses_manifest_defaults():
    manifest = _build_sample_manifest()
    template = generate_template(
        TemplateType.SUITE, manifest=manifest, image="registry/mock:latest"
    )

    test_definition = template.payload["test_suite"][0]
    assert test_definition["image"] == "registry/mock:latest"
    assert test_definition["systems_under_test"] == ["system_under_test"]
    assert test_definition["systems"]["simulator_system"] == "simulator_system"
    assert test_definition["params"]["generations"] == 1
    assert test_definition["params"]["temperature"] == 0.0
    assert template.optional_systems == ["simulator_system"]


def test_generate_score_card_template_defaults():
    template = generate_template(TemplateType.SCORE_CARD)

    assert template.payload["score_card_name"] == "My Score Card"
    indicator = template.payload["indicators"][0]
    assert indicator["metric"] == "metrics.overall_score"


def test_resolve_manifest_path_accepts_directory(tmp_path):
    manifest_dir = tmp_path / "test_container"
    manifest_dir.mkdir()
    manifest_file = manifest_dir / "manifest.yaml"
    manifest_file.write_text(
        "name: example\nversion: 1.0\ninput_systems: []\ninput_schema: []\noutput_metrics: []\n"
    )

    resolved = resolve_manifest_path(manifest_dir)
    assert resolved == manifest_file
