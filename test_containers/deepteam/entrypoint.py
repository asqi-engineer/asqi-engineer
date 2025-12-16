import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union

import openai

from deepeval.models import DeepEvalBaseLLM

from deepeval.confident.api import log_retry_error
from deepteam.attacks.multi_turn import (
    BadLikertJudge,
    CrescendoJailbreaking,
    LinearJailbreaking,
    SequentialJailbreak,
    TreeJailbreaking,
)
from deepteam.attacks.single_turn import (
    AdversarialPoetry,
    Base64,
    GrayBox,
    Leetspeak,
    MathProblem,
    Multilingual,
    PromptInjection,
    PromptProbing,
    Roleplay,
    ROT13,
    SystemOverride,
    PermissionEscalation,
    LinguisticConfusion,
    InputBypass,
    ContextPoisoning,
    GoalRedirection,
)
from deepteam.red_teamer import RedTeamer
from deepteam.vulnerabilities import (
    BFLA,
    BOLA,
    RBAC,
    SSRF,
    Bias,
    Ethics,
    Fairness,
    ChildProtection,
    Misinformation,
    Toxicity,
    Competition,
    DebugAccess,
    ExcessiveAgency,
    GoalTheft,
    GraphicContent,
    IllegalActivity,
    IntellectualProperty,
    PersonalSafety,
    PIILeakage,
    PromptLeakage,
    RecursiveHijacking,
    Robustness,
    ShellInjection,
    SQLInjection,
)

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    wait_exponential_jitter,
)

# Define retryable exceptions for OpenAI
try:
    from openai import (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
    )

    retryable_exceptions = (
        RateLimitError,
        APIConnectionError,
        APITimeoutError,
        APIStatusError,
    )
except ImportError:
    retryable_exceptions = (Exception,)


class CustomOpenAIModel(DeepEvalBaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0,
        **kwargs,
    ):
        self.model_name = model
        self.api_key = api_key
        self.base_url = base_url
        self.temperature = temperature
        self.client_kwargs = kwargs
        super().__init__(model)

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    def generate(  # type: ignore
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[str, BaseModel]:
        client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, **self.client_kwargs
        )

        completion_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if schema:
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            }

        response = client.chat.completions.create(**completion_params)
        content = response.choices[0].message.content

        if schema:
            data = json.loads(content)
            return schema.model_validate(data)
        return content

    @retry(
        wait=wait_exponential_jitter(initial=1, exp_base=2, jitter=2, max=10),
        retry=retry_if_exception_type(retryable_exceptions),
        after=log_retry_error,
    )
    async def a_generate(  # type: ignore
        self, prompt: str, schema: Optional[BaseModel] = None
    ) -> Union[str, BaseModel]:
        client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, **self.client_kwargs
        )

        completion_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }

        if schema:
            completion_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                },
            }

        response = await client.chat.completions.create(**completion_params)
        content = response.choices[0].message.content

        if schema:
            data = json.loads(content)
            return schema.model_validate(data)
        return content

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self):
        return None


class DeepTeamRedTeamTester:
    """Red team testing using DeepTeam library"""

    # Map vulnerability names to classes for easy configuration
    VULNERABILITY_MAP = {
        # Data Privacy
        "pii_leakage": PIILeakage,
        "prompt_leakage": PromptLeakage,
        # Responsible AI
        "bias": Bias,
        "toxicity": Toxicity,
        "ethics": Ethics,
        "fairness": Fairness,
        "child_protection": ChildProtection,
        # Security
        "bfla": BFLA,
        "bola": BOLA,
        "rbac": RBAC,
        "debug_access": DebugAccess,
        "shell_injection": ShellInjection,
        "sql_injection": SQLInjection,
        "ssrf": SSRF,
        # Safety
        "illegal_activity": IllegalActivity,
        "graphic_content": GraphicContent,
        "personal_safety": PersonalSafety,
        # Business
        "misinformation": Misinformation,
        "intellectual_property": IntellectualProperty,
        "competition": Competition,
        # Agentic
        "goal_theft": GoalTheft,
        "recursive_hijacking": RecursiveHijacking,
        "excessive_agency": ExcessiveAgency,
        "robustness": Robustness,
    }

    # Single-turn attack methods
    SINGLE_TURN_ATTACKS = {
        "adversarial_poetry": AdversarialPoetry,
        "context_poisoning": ContextPoisoning,
        "base64": Base64,
        "goal_redirection": GoalRedirection,
        "graybox": GrayBox,
        "input_bypass": InputBypass,
        "leetspeak": Leetspeak,
        "linguistic_confusion": LinguisticConfusion,
        "math_problem": MathProblem,
        "multilingual": Multilingual,
        "permission_escalation": PermissionEscalation,
        "prompt_injection": PromptInjection,
        "prompt_probing": PromptProbing,
        "roleplay": Roleplay,
        "rot13": ROT13,
        "system_override": SystemOverride,
    }

    # Multi-turn attack methods
    MULTI_TURN_ATTACKS = {
        "crescendo_jailbreaking": CrescendoJailbreaking,
        "linear_jailbreaking": LinearJailbreaking,
        "tree_jailbreaking": TreeJailbreaking,
        "sequential_jailbreak": SequentialJailbreak,
        "bad_likert_judge": BadLikertJudge,
    }

    def __init__(self, systems_params: Dict[str, Any], test_params: Dict[str, Any]):
        """Initialize with systems and test parameters"""
        self.systems_params = systems_params
        self.test_params = test_params
        self.sut_params = systems_params.get("system_under_test", {})
        self.client = self._setup_client()
        self.simulator_model = self._setup_system_model("simulator_system")
        self.evaluation_model = self._setup_system_model("evaluator_system")

    def _setup_client(self) -> openai.OpenAI:
        """Setup OpenAI client using provided SUT parameters"""
        api_key = os.environ.get("API_KEY")
        if not api_key:
            # Fallback to provider-specific keys
            api_key = (
                os.environ.get("OPENAI_API_KEY")
                or os.environ.get("ANTHROPIC_API_KEY")
                or os.environ.get("HUGGINGFACE_API_KEY")
            )

        if not api_key:
            raise ValueError("No API key found. Set API_KEY environment variable.")

        return openai.OpenAI(base_url=self.sut_params["base_url"], api_key=api_key)

    def _setup_system_model(self, system_role: str) -> Optional[CustomOpenAIModel]:
        """Setup model from systems configuration"""
        system_config = self.systems_params.get(system_role)

        if not system_config or not isinstance(system_config, dict):
            return None

        # Use the API key from the system config, or fallback to environment
        api_key = system_config.get("api_key") or os.environ.get("API_KEY", "")

        return CustomOpenAIModel(
            model=system_config.get("model", "gpt-4o-mini"),
            api_key=api_key,
            base_url=system_config.get("base_url", self.sut_params["base_url"]),
            temperature=system_config.get("temperature", 0),
        )

    async def _model_callback(self, input_text: str, turns: list = None) -> str:
        """Model callback function for DeepTeam red teaming"""
        # Build message history from turns if provided (for multi-turn attacks)
        messages = []
        if turns:
            for turn in turns:
                messages.append({"role": "user", "content": turn.user_input})
                if turn.model_output:
                    messages.append({"role": "assistant", "content": turn.model_output})

        # Add current input
        messages.append({"role": "user", "content": input_text})

        response = self.client.chat.completions.create(
            model=self.sut_params["model"],
            messages=messages,
        )
        return response.choices[0].message.content or ""

    def _create_vulnerabilities(self, vuln_configs: List[Dict[str, Any]]) -> List[Any]:
        """Create vulnerability objects from configuration"""
        vulnerabilities = []

        for config in vuln_configs:
            vuln_name = config.get("name", "").lower()
            vuln_types = config.get("types", [])

            if vuln_name in self.VULNERABILITY_MAP:
                vuln_class = self.VULNERABILITY_MAP[vuln_name]
                if vuln_types:
                    vulnerability = vuln_class(types=vuln_types)
                else:
                    vulnerability = vuln_class()
                vulnerabilities.append(vulnerability)
            else:
                print(f"Warning: Unknown vulnerability '{vuln_name}'. Skipping.")

        return vulnerabilities

    def _create_attacks(self, attack_list: List[str]) -> List[Any]:
        """Create attack objects from simple list of attack names"""
        attacks = []

        # Combine all available attacks
        all_attacks = {**self.SINGLE_TURN_ATTACKS, **self.MULTI_TURN_ATTACKS}

        for attack_name in attack_list:
            attack_name_lower = attack_name.lower()
            if attack_name_lower in all_attacks:
                attack_class = all_attacks[attack_name_lower]
                attacks.append(attack_class())
            else:
                print(f"Warning: Unknown attack '{attack_name}'. Skipping.")

        return attacks

    def _extract_detailed_test_cases(
        self, risk_assessment: Any, output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract detailed test case information from risk_assessment and save to JSON file

        Args:
            risk_assessment: The risk assessment object from DeepTeam red_team() call
            output_dir: Directory to save the detailed test cases JSON file (optional)

        Returns:
            List of detailed test case dictionaries
        """
        detailed_test_cases = []

        for test_case in risk_assessment.test_cases:
            case_detail = {
                "vulnerability": test_case.vulnerability,
                "vulnerability_type": test_case.vulnerability_type.value
                if test_case.vulnerability_type
                else None,
                "attack_method": test_case.attack_method,
                "risk_category": test_case.risk_category,
                "input": test_case.input,
                "actual_output": test_case.actual_output,
                "score": test_case.score,
                "reason": test_case.reason,
                "error": test_case.error,
                "status": "passed"
                if (test_case.score and test_case.score > 0)
                else ("errored" if test_case.error else "failed"),
            }

            # Include turns for multi-turn conversations if present
            if test_case.turns:
                case_detail["turns"] = [
                    {
                        "role": turn.role,
                        "content": turn.content,
                    }
                    for turn in test_case.turns
                ]

            detailed_test_cases.append(case_detail)

        # Save to JSON file if output directory is provided
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, "results.json")
                with open(output_file, "w") as f:
                    json.dump(detailed_test_cases, f, indent=2)
                print(f"Detailed test cases saved to {output_file}", file=sys.stderr)
            except Exception as e:
                print(
                    f"Warning: Failed to save detailed test cases to file: {e}",
                    file=sys.stderr,
                )

        return detailed_test_cases

    async def run_red_team_test(self, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute red team testing using DeepTeam"""
        try:
            # Extract test parameters with defaults
            vulnerability_configs = test_params.get(
                "vulnerabilities",
                [
                    {"name": "bias", "types": ["race", "gender"]},
                    {"name": "toxicity", "types": ["profanity", "insults"]},
                    {"name": "pii_leakage", "types": ["direct disclosure"]},
                ],
            )

            attack_list = test_params.get("attacks", ["prompt_injection", "leetspeak"])

            # Create vulnerability and attack objects
            vulnerabilities = self._create_vulnerabilities(vulnerability_configs)
            attacks = self._create_attacks(attack_list)

            if not vulnerabilities:
                return {
                    "success": False,
                    "error": "No valid vulnerabilities specified",
                    "score": 0.0,
                }

            if not attacks:
                return {
                    "success": False,
                    "error": "No valid attacks specified",
                    "score": 0.0,
                }

            # Extract configuration parameters
            max_concurrent = test_params.get("max_concurrent", 10)
            attacks_per_vulnerability_type = test_params.get(
                "attacks_per_vulnerability_type", 3
            )
            target_purpose = test_params.get(
                "target_purpose",
                "AI assistant being tested for security vulnerabilities",
            )

            red_teamer_kwargs = {
                "target_purpose": target_purpose,
                "async_mode": True,
                "max_concurrent": max_concurrent,
            }

            # Add custom models if configured
            if self.simulator_model:
                red_teamer_kwargs["simulator_model"] = self.simulator_model
            if self.evaluation_model:
                red_teamer_kwargs["evaluation_model"] = self.evaluation_model

            red_teamer = RedTeamer(**red_teamer_kwargs)

            # Run red team assessment
            print("Starting red team assessment...")
            try:
                risk_assessment = red_teamer.red_team(
                    model_callback=self._model_callback,  # type: ignore
                    vulnerabilities=vulnerabilities,
                    attacks=attacks,
                    attacks_per_vulnerability_type=attacks_per_vulnerability_type,
                    ignore_errors=False,
                    _upload_to_confident=False,
                )
            except Exception as e:
                # Silently skip Confident AI upload errors
                if "Invalid API key" in str(e) or "ConfidentApiError" in str(
                    type(e).__name__
                ):
                    if hasattr(red_teamer, "risk_assessment"):
                        risk_assessment = red_teamer.risk_assessment
                    else:
                        raise
                else:
                    raise

            # Extract statistics from risk assessment overview
            vulnerability_stats = {}
            attack_stats = {}
            total_tests = 0
            total_passing = 0
            total_failing = 0
            total_errored = 0

            if hasattr(risk_assessment, "overview"):
                overview = risk_assessment.overview

                # Extract vulnerability type results
                if hasattr(overview, "vulnerability_type_results"):
                    for vuln_result in overview.vulnerability_type_results:
                        vuln_name = vuln_result.vulnerability
                        vuln_type = (
                            str(vuln_result.vulnerability_type).split(".")[-1].lower()
                            if hasattr(vuln_result.vulnerability_type, "value")
                            else str(vuln_result.vulnerability_type)
                        )

                        if vuln_name not in vulnerability_stats:
                            vulnerability_stats[vuln_name] = {
                                "types": {},
                                "total_passing": 0,
                                "total_failing": 0,
                                "total_errored": 0,
                                "overall_pass_rate": 0.0,
                            }

                        vulnerability_stats[vuln_name]["types"][vuln_type] = {
                            "pass_rate": vuln_result.pass_rate,
                            "passing": vuln_result.passing,
                            "failing": vuln_result.failing,
                            "errored": vuln_result.errored,
                        }

                        # Aggregate totals for this vulnerability
                        vulnerability_stats[vuln_name]["total_passing"] += (
                            vuln_result.passing
                        )
                        vulnerability_stats[vuln_name]["total_failing"] += (
                            vuln_result.failing
                        )
                        vulnerability_stats[vuln_name]["total_errored"] += (
                            vuln_result.errored
                        )

                        # Update overall totals
                        total_passing += vuln_result.passing
                        total_failing += vuln_result.failing
                        total_errored += vuln_result.errored

                # Calculate overall pass rates for each vulnerability
                for vuln_name, stats in vulnerability_stats.items():
                    total_vuln_tests = (
                        stats["total_passing"]
                        + stats["total_failing"]
                        + stats["total_errored"]
                    )
                    if total_vuln_tests > 0:
                        stats["overall_pass_rate"] = (
                            stats["total_passing"] / total_vuln_tests
                        )

                # Extract attack method results
                if hasattr(overview, "attack_method_results"):
                    for attack_result in overview.attack_method_results:
                        attack_name = attack_result.attack_method
                        attack_stats[attack_name] = {
                            "pass_rate": attack_result.pass_rate,
                            "passing": attack_result.passing,
                            "failing": attack_result.failing,
                            "errored": attack_result.errored,
                        }

                # Get total errored from overview if available
                if hasattr(overview, "errored"):
                    total_errored = overview.errored

            total_tests = total_passing + total_failing + total_errored

            if total_tests == 0:
                return {
                    "success": False,
                    "error": "No test cases generated or executed",
                    "score": 0.0,
                }

            # Calculate overall metrics
            pass_rate = total_passing / total_tests if total_tests > 0 else 0.0
            failure_rate = total_failing / total_tests if total_tests > 0 else 0.0

            # Determine output directory for detailed results
            persistent_log_dir = None
            volumes = test_params.get("volumes", {})
            if "output" in volumes:
                # Construct test name from vulnerabilities
                vuln_names = "_".join(
                    list(vulnerability_stats.keys())[:3]
                ).lower()  # Use up to 3 vulnerability names
                if not vuln_names:
                    vuln_names = "unknown"

                # Make model name safe for file paths
                safe_model_name = (
                    self.sut_params["model"]
                    .replace("/", "_")
                    .replace(":", "_")
                    .replace(".", "_")
                )
                persistent_log_dir = f"/output/deepteam_{vuln_names}_{safe_model_name}"

            # Extract and save detailed test case information
            self._extract_detailed_test_cases(
                risk_assessment, output_dir=persistent_log_dir
            )

            return {
                "success": True,
                "pass_rate": pass_rate,
                "failure_rate": failure_rate,
                "total_tests": total_tests,
                "total_passing": total_passing,
                "total_failing": total_failing,
                "total_errored": total_errored,
                "vulnerability_stats": vulnerability_stats,
                "attack_stats": attack_stats,
                "vulnerabilities_tested": list(vulnerability_stats.keys()),
                "attacks_used": list(attack_stats.keys()),
                "model_tested": self.sut_params["model"],
            }

        except Exception as e:
            import traceback

            error_traceback = traceback.format_exc()
            print(f"Error during red team testing: {error_traceback}", file=sys.stderr)
            return {
                "success": False,
                "error": f"Red team testing failed: {str(e)}",
                "score": 0.0,
                "pass_rate": 0.0,
                "failure_rate": 1.0,
                "total_tests": 0,
                "total_passing": 0,
                "total_failing": 0,
                "total_errored": 0,
            }


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="DeepTeam Red Team Test Container")
    parser.add_argument(
        "--systems-params", required=True, help="Systems parameters as JSON"
    )
    parser.add_argument("--test-params", required=True, help="Test parameters as JSON")
    args = parser.parse_args()

    try:
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)
        return systems_params, test_params
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in arguments: {e}")


def validate_inputs(systems_params: Dict[str, Any], test_params: Dict[str, Any]):
    """Validate input parameters"""
    # Validate system_under_test params
    sut_params = systems_params.get("system_under_test", {})
    required_sut_fields = ["type", "base_url", "model"]
    for field in required_sut_fields:
        if field not in sut_params:
            raise ValueError(f"Missing required system_under_test parameter: {field}")

    if sut_params["type"] != "llm_api":
        raise ValueError(f"Unsupported system_under_test type: {sut_params['type']}")

    # Validate vulnerabilities
    valid_vulnerabilities = set(DeepTeamRedTeamTester.VULNERABILITY_MAP.keys())
    vuln_configs = test_params.get("vulnerabilities", [])

    if vuln_configs:
        invalid_vulnerabilities = []
        for vuln_config in vuln_configs:
            vuln_name = vuln_config.get("name", "").lower()
            if vuln_name not in valid_vulnerabilities and vuln_name != "custom":
                invalid_vulnerabilities.append(vuln_name)

        if invalid_vulnerabilities:
            valid_vuln_list = sorted(valid_vulnerabilities)
            raise ValueError(
                f"Invalid vulnerability names: {invalid_vulnerabilities}. "
                f"Valid vulnerabilities are: {valid_vuln_list}"
            )

    # Validate attacks
    valid_single_turn = set(DeepTeamRedTeamTester.SINGLE_TURN_ATTACKS.keys())
    valid_multi_turn = set(DeepTeamRedTeamTester.MULTI_TURN_ATTACKS.keys())
    valid_attacks = valid_single_turn | valid_multi_turn

    attack_list = test_params.get("attacks", [])
    if attack_list:
        invalid_attacks = []
        for attack_name in attack_list:
            attack_name_lower = attack_name.lower()
            if attack_name_lower not in valid_attacks:
                invalid_attacks.append(attack_name)

        if invalid_attacks:
            valid_attack_list = sorted(valid_attacks)
            raise ValueError(
                f"Invalid attack names: {invalid_attacks}. "
                f"Valid attacks are: {valid_attack_list}"
            )

    # Validate numeric parameters
    numeric_params = ["max_concurrent", "attacks_per_vulnerability_type"]
    for param in numeric_params:
        if param in test_params:
            if (
                not isinstance(test_params[param], (int, float))
                or test_params[param] <= 0
            ):
                raise ValueError(f"Parameter '{param}' must be a positive number")


async def main_async():
    """Async main execution function"""
    try:
        # Parse and validate inputs
        systems_params, test_params = parse_arguments()
        validate_inputs(systems_params, test_params)

        # Initialize tester
        tester = DeepTeamRedTeamTester(systems_params, test_params)

        # Run red team test
        result = await tester.run_red_team_test(test_params)

        # Output results
        print(json.dumps(result, indent=2))
        sys.exit(0 if result["success"] else 1)

    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "score": 0.0,
            "pass_rate": 0.0,
            "failure_rate": 1.0,
            "total_tests": 0,
            "total_passing": 0,
            "total_failing": 0,
            "total_errored": 0,
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


def main():
    """Main execution function"""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
