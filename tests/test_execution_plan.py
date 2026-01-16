from asqi.schemas import (
    GenericSystemConfig,
    SuiteConfig,
    SystemsConfig,
    TestDefinition as SuiteTestDefinition,
)
from asqi.validation import create_test_execution_plan


def test_create_test_execution_plan_multi_system_uses_each_system_individually():
    systems = SystemsConfig(
        systems={
            "system_a": GenericSystemConfig(
                type="llm_api",
                description="System A",
                provider="openai",
                params={"model": "a-model"},
            ),
            "system_b": GenericSystemConfig(
                type="llm_api",
                description="System B",
                provider="openai",
                params={"model": "b-model"},
            ),
        }
    )

    suite = SuiteConfig(
        suite_name="multi system suite",
        test_suite=[
            SuiteTestDefinition(
                id="multi_system_test",
                name="multi system test",
                image="img:latest",
                systems_under_test=["system_a", "system_b"],
                params={},
            )
        ],
    )

    plan = create_test_execution_plan(suite, systems, {"img:latest": True})

    assert len(plan) == 2
    assert plan[0]["sut_name"] == "system_a"
    assert plan[1]["sut_name"] == "system_b"

    # Regression: each plan entry must have its own independent systems_params dict.
    assert plan[0]["systems_params"] is not plan[1]["systems_params"]

    assert plan[0]["systems_params"]["system_under_test"]["model"] == "a-model"
    assert plan[1]["systems_params"]["system_under_test"]["model"] == "b-model"


def test_create_test_execution_plan_preserves_additional_systems_per_entry():
    systems = SystemsConfig(
        systems={
            "system_a": GenericSystemConfig(
                type="llm_api",
                description="System A",
                provider="openai",
                params={"model": "a-model"},
            ),
            "system_b": GenericSystemConfig(
                type="llm_api",
                description="System B",
                provider="openai",
                params={"model": "b-model"},
            ),
            "support": GenericSystemConfig(
                type="llm_api",
                description="Support system",
                provider="openai",
                params={"model": "support-model"},
            ),
        }
    )

    suite = SuiteConfig(
        suite_name="multi system suite with support",
        test_suite=[
            SuiteTestDefinition(
                id="multi_system_test_with_support",
                name="multi system test with support",
                image="img:latest",
                systems_under_test=["system_a", "system_b"],
                systems={"support_system": "support"},
                params={},
            )
        ],
    )

    plan = create_test_execution_plan(suite, systems, {"img:latest": True})

    assert len(plan) == 2
    for entry in plan:
        assert entry["systems_params"]["support_system"]["model"] == "support-model"

    assert plan[0]["systems_params"]["system_under_test"]["model"] == "a-model"
    assert plan[1]["systems_params"]["system_under_test"]["model"] == "b-model"
