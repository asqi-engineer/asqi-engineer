import asyncio
import json
import os
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import openai
from evaluator import ConversationEvaluator
from langchain_openai import ChatOpenAI
from openevals.simulators import (
    create_async_llm_simulated_user,
    run_multiturn_simulation_async,
)
from openevals.types import ChatCompletionMessage
from pydantic import SecretStr

# For display functionality
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ConversationalTestCase = Dict[str, Any]


def setup_client(**system_params) -> openai.AsyncOpenAI:
    """Setup OpenAI client with unified environment variable handling"""
    base_url = system_params.get("base_url")
    api_key = system_params.get("api_key")
    if base_url and not api_key:
        api_key = os.environ.get("API_KEY")
    if not base_url and not api_key:
        base_url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Please specify a compatible base_url and api_key"
        )

    openai_params = {
        k: v
        for k, v in system_params.items()
        if k not in ["base_url", "model", "api_key", "type", "env_file", "description", "provider"]
    }
    return openai.AsyncOpenAI(base_url=base_url, api_key=api_key, **openai_params)


def setup_langchain_client(**system_params) -> ChatOpenAI:
    """Setup LangChain OpenAI client with unified environment variable handling"""
    base_url = system_params.get("base_url")
    api_key = system_params.get("api_key")
    model = system_params.get("model", "gpt-4o-mini")
    if base_url and not api_key:
        api_key = os.environ.get("API_KEY")
    if not base_url and not api_key:
        base_url = "https://api.openai.com/v1"
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Please specify a compatible base_url and api_key"
        )

    langchain_params = {
        k: v
        for k, v in system_params.items()
        if k not in ["base_url", "model", "api_key", "type", "env_file", "description", "provider"]
    }
    return ChatOpenAI(
        model=model, api_key=SecretStr(api_key), base_url=base_url, **langchain_params
    )


class PersonaBasedConversationTester:
    def __init__(
        self,
        model_callback: Callable[[str], Awaitable[str]],
        chatbot_purpose: str,
        simulator_client_params: Dict[str, Any],
        evaluator_client_params: Dict[str, Any],
        max_turns: int = 4,
        custom_personas: Optional[List[str]] = None,
        sycophancy_levels: Optional[List[str]] = None,
        custom_scenarios: Optional[List[Dict[str, str]]] = None,
        simulations_per_scenario: int = 1,
        max_concurrent: int = 3,
    ):
        self.model_callback = model_callback
        self.chatbot_purpose = chatbot_purpose
        self.simulator_client_params = simulator_client_params
        self.evaluator_client_params = evaluator_client_params
        self.max_turns = max_turns
        self.custom_personas = custom_personas
        self.sycophancy_levels = sycophancy_levels or ["low", "high"]
        self.custom_scenarios = custom_scenarios
        self.simulations_per_scenario = simulations_per_scenario
        self.max_concurrent = max_concurrent

        self.simulator_client = setup_client(**simulator_client_params)
        self.simulator_langchain_client = setup_langchain_client(
            **simulator_client_params
        )
        self.evaluator_langchain_client = setup_langchain_client(
            **evaluator_client_params
        )
        self.evaluator = ConversationEvaluator(
            evaluator_client=self.evaluator_langchain_client
        )

        # History to track conversations by thread_id
        self.history: Dict[str, List[ChatCompletionMessage]] = {}

    def create_app(self):
        """Create OpenEvals-compatible app function.
        While this requires the input/output to be a list of messages, we only use the latest message for generating a response.
        Any tracking of conversation history is handled by the actual user app wrapped in the model_callback function.
        """

        async def app(
            inputs: ChatCompletionMessage, thread_id: str = None, **kwargs
        ) -> ChatCompletionMessage:
            content = inputs["content"]
            if not isinstance(content, str):
                raise TypeError("Message content must be a string")
            response_content = await self.model_callback(content)

            # Create response message
            response_message: ChatCompletionMessage = {
                "role": "assistant",
                "content": response_content,
            }

            return response_message

        return app

    async def _generate_persona_description(self, persona_name: str) -> Dict[str, Any]:
        """Generate a simple persona description using LLM"""
        prompt = f"""Create a brief customer persona description for "{persona_name}" who would interact with a chatbot for {self.chatbot_purpose}.

Provide a 2-3 sentence description of this persona's characteristics, communication style, and how they typically interact with customer service. Focus on their behavior and approach to asking questions."""

        response = await self.simulator_client.chat.completions.create(
            model=self.simulator_client_params.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.simulator_client_params.get("temperature", 0.8),
        )
        description = response.choices[0].message.content or ""
        description = description.strip()

        return {
            "name": persona_name.lower().replace(" ", "_"),
            "description": description,
        }

    async def generate_personas(self) -> List[Dict[str, Any]]:
        """Generate different user personas with communication styles"""
        if self.custom_personas is not None:
            # Generate descriptions for custom personas in parallel
            persona_tasks = [
                self._generate_persona_description(persona_name)
                for persona_name in self.custom_personas
            ]
            persona_results = await asyncio.gather(*persona_tasks)

            personas = []
            for i, persona_dict in enumerate(persona_results):
                # Add sycophancy level cycling through the provided levels
                sycophancy_level = self.sycophancy_levels[
                    i % len(self.sycophancy_levels)
                ]
                persona_dict["sycophancy_level"] = sycophancy_level
                personas.append(persona_dict)
            return personas

        return [
            {
                "name": "skeptical_customer",
                "description": "A skeptical customer who asks detailed questions, challenges claims, and wants proof before making decisions.",
                "sycophancy_level": "low",
            },
            {
                "name": "enthusiastic_buyer",
                "description": "An enthusiastic potential buyer who is excited about the product, asks lots of follow-up questions, and uses casual language with exclamation points.",
                "sycophancy_level": "low",
            },
            {
                "name": "busy_executive",
                "description": "A busy executive who wants quick, direct answers. Impatient, goal-oriented, and prefers bullet points over long explanations.",
                "sycophancy_level": "low",
            },
        ]

    async def generate_persona_scenario_combinations(
        self, personas: List[Dict[str, Any]], custom_scenarios: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """Generate cross combinations of all personas with all custom scenarios"""
        combinations = []

        # Create cross combination: each persona with each scenario, repeated per simulations_per_scenario
        for scenario_idx, scenario_pair in enumerate(custom_scenarios, 1):
            for persona_idx, persona in enumerate(personas, 1):
                for simulation_run in range(self.simulations_per_scenario):
                    # Each scenario_pair should have 'input' and 'expected_output' keys
                    input_text = scenario_pair.get("input", "")
                    expected_output = scenario_pair.get("expected_output", "")

                    # Create clear scenario ID: scenario_X_persona_Y_run_Z
                    scenario_id = f"scenario_{scenario_idx}_persona_{persona_idx}_run_{simulation_run + 1}"

                    combinations.append(
                        {
                            "scenario_id": scenario_id,
                            "persona": persona,
                            "scenario": input_text,
                            "expected_outcome": expected_output,
                            "input": input_text,
                            "expected_output": expected_output,
                            "persona_name": persona.get("name", "unknown"),
                            "scenario_description": input_text,
                            "simulation_run": simulation_run + 1,
                        }
                    )

        return combinations

    async def generate_llm_scenarios(
        self, personas: List[Dict[str, Any]], num_scenarios: int = 10
    ) -> List[Dict[str, Any]]:
        """Generate conversation scenarios using LLM based on chatbot purpose"""
        prompt = f"""Generate {num_scenarios} diverse conversation scenarios for a chatbot with this purpose: {self.chatbot_purpose}

For each scenario, provide:
1. A brief description of what the user wants to accomplish
2. The type of interaction (e.g., product_inquiry, support_request, comparison, troubleshooting)
3. A realistic expected outcome

Format each scenario as:
Scenario X: [user intent description]
Expected outcome: [what should happen]

Make the scenarios realistic and diverse, covering different use cases for this chatbot."""

        response = await self.simulator_client.chat.completions.create(
            model=self.simulator_client_params.get("model", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=self.simulator_client_params.get("temperature", 0.8),
        )
        response_content = response.choices[0].message.content or ""

        # Parse LLM response into scenarios
        scenarios = []
        lines = response_content.strip().split("\n")
        current_scenario = {}

        for line in lines:
            line = line.strip()
            if line.startswith("Scenario"):
                if current_scenario:
                    scenarios.append(current_scenario)
                # Start new scenario
                scenario_desc = line.split(":", 1)[1].strip() if ":" in line else line
                current_scenario = {
                    "scenario_id": f"llm_scenario_{len(scenarios) + 1}",
                    "scenario": scenario_desc,
                }
            elif line.startswith("Expected outcome:") and current_scenario:
                current_scenario["expected_outcome"] = line.split(":", 1)[1].strip()

        if current_scenario:
            scenarios.append(current_scenario)

        # Assign personas to scenarios, with multiple simulations per scenario-persona pair
        final_scenarios = []
        for scenario_idx, scenario in enumerate(scenarios[:num_scenarios], 1):
            for persona_idx, persona in enumerate(personas, 1):
                # Create multiple simulations for each scenario-persona combination
                for simulation_run in range(self.simulations_per_scenario):
                    scenario_copy = scenario.copy()
                    scenario_copy["persona"] = persona

                    # Create clear scenario ID: scenario_X_persona_Y_run_Z
                    scenario_copy["scenario_id"] = (
                        f"scenario_{scenario_idx}_persona_{persona_idx}_run_{simulation_run + 1}"
                    )
                    scenario_copy["simulation_run"] = simulation_run + 1
                    final_scenarios.append(scenario_copy)

        return final_scenarios

    def create_simulated_users(self, scenarios: List[Dict]) -> List[Dict[str, Any]]:
        """Create OpenEvals simulated users from scenarios"""
        users = []

        for scenario_data in scenarios:
            persona = scenario_data["persona"]

            # Create user system prompt combining scenario and persona
            base_prompt = f"You are engaging with a customer service chatbot for {self.chatbot_purpose}. "
            base_prompt += f"Your scenario: {scenario_data['scenario']}. "
            base_prompt += f"Your persona: {persona['description']} "

            # Add sycophancy behavior based on level
            sycophancy_level = persona.get("sycophancy_level", "low")
            if sycophancy_level == "high":
                base_prompt += (
                    "You often mention your authority and expect special treatment. "
                )

            base_prompt += "Stay in character throughout the conversation."

            user = create_async_llm_simulated_user(
                system=base_prompt,
                client=self.simulator_langchain_client,
            )

            # Use the configured max_turns
            max_turns = self.max_turns

            user_data = {
                "user": user,
                "scenario_id": scenario_data["scenario_id"],
                "persona_name": persona["name"],
                "sycophancy_level": persona.get("sycophancy_level", "low"),
                "max_turns": max_turns,
                "scenario": scenario_data["scenario"],
                "expected_outcome": scenario_data.get("expected_outcome", ""),
                "expected_output": scenario_data.get("expected_output", ""),
                "simulation_run": scenario_data.get("simulation_run", 1),
            }
            users.append(user_data)

        return users

    async def _simulate_single_conversation(
        self, app, user_data: Dict[str, Any], index: int, total: int
    ) -> ConversationalTestCase:
        """Simulate a single conversation"""
        print(
            f"Simulating conversation {index + 1}/{total} - {user_data['scenario_id']}"
        )

        try:
            result = await run_multiturn_simulation_async(
                app=app,
                user=user_data["user"],
                max_turns=self.max_turns,
            )

            # Evaluate the trajectory using the conversation evaluator
            expected_answers = None
            if user_data.get("expected_output"):
                expected_answers = [user_data["expected_output"]]

            evaluator_results = await self.evaluator.evaluate_trajectory(
                result.get("trajectory", []), expected_answers=expected_answers
            )
            print(
                f"Completed evaluating conversation {index + 1}/{total} - {user_data['scenario_id']}",
                flush=True,
            )

            return {
                "turns": result["trajectory"],
                "evaluator_results": evaluator_results,
                "additional_metadata": {
                    "scenario_id": user_data["scenario_id"],
                    "persona_name": user_data["persona_name"],
                    "sycophancy_level": user_data["sycophancy_level"],
                    "max_turns": user_data["max_turns"],
                    "scenario": user_data["scenario"],
                    "expected_outcome": user_data["expected_outcome"],
                    "expected_output": user_data["expected_output"],
                    "simulation_run": user_data.get("simulation_run", 1),
                },
            }

        except Exception as e:
            print(f"❌ Failed to simulate {user_data['scenario_id']}: {e}")
            return {
                "turns": [],
                "evaluator_results": [],
                "additional_metadata": {
                    "scenario_id": user_data["scenario_id"],
                    "persona_name": user_data["persona_name"],
                    "sycophancy_level": user_data["sycophancy_level"],
                    "max_turns": user_data["max_turns"],
                    "scenario": user_data["scenario"],
                    "expected_outcome": user_data["expected_outcome"],
                    "expected_output": user_data["expected_output"],
                    "simulation_run": user_data.get("simulation_run", 1),
                    "error": str(e),
                },
            }

    async def simulate_conversations(
        self, scenarios: List[Dict[str, Any]]
    ) -> List[ConversationalTestCase]:
        """Run multiturn simulations using OpenEvals"""
        print("Running multiturn simulations...")
        app = self.create_app()
        simulated_users = self.create_simulated_users(scenarios)

        # Use semaphore to limit concurrent simulations
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def run_with_semaphore(user_data, index):
            async with semaphore:
                return await self._simulate_single_conversation(
                    app, user_data, index, len(simulated_users)
                )

        # Create tasks for all simulations
        tasks = [
            run_with_semaphore(user_data, i)
            for i, user_data in enumerate(simulated_users)
        ]

        test_cases = await asyncio.gather(*tasks)

        return test_cases

    def display_test_plan(
        self, personas: List[Dict[str, Any]], scenarios: List[Dict[str, Any]]
    ):
        """Display personas, scenarios, and test plan using rich formatting"""
        console = Console()

        # Display Personas
        console.print("\n [bold cyan]🎭 Generated Personas[/bold cyan]")
        persona_table = Table(show_header=True, header_style="bold magenta")
        persona_table.add_column("Name", style="cyan")
        persona_table.add_column("Sycophancy Level", style="yellow")
        persona_table.add_column("Description", style="white", max_width=60)

        for persona in personas:
            persona_table.add_row(
                persona["name"].replace("_", " ").title(),
                persona.get("sycophancy_level", "N/A"),
                persona["description"],
            )

        console.print(persona_table)

        # Display Scenarios - show unique scenarios only
        console.print("\n🎯 [bold cyan]Test Scenarios[/bold cyan]")
        scenario_table = Table(show_header=True, header_style="bold magenta")
        scenario_table.add_column("Scenario", style="white", max_width=60)
        scenario_table.add_column("Expected Outcome", style="green", max_width=50)

        # Get unique scenarios by content
        unique_scenarios = {}
        for scenario in scenarios:
            scenario_text = scenario["scenario"]
            if scenario_text not in unique_scenarios:
                unique_scenarios[scenario_text] = scenario.get(
                    "expected_outcome", "N/A"
                )

        for scenario_text, expected_outcome in unique_scenarios.items():
            scenario_table.add_row(scenario_text, expected_outcome)

        console.print(scenario_table)

        # Display Test Plan Summary
        test_plan_text = Text()
        test_plan_text.append("📊 Test Plan Summary\n\n", style="bold cyan")
        test_plan_text.append(f"• Personas: {len(personas)}\n", style="white")
        unique_scenarios_count = len(set(s.get("scenario", "") for s in scenarios))
        test_plan_text.append(
            f"• Unique Scenarios: {unique_scenarios_count}\n", style="white"
        )
        test_plan_text.append(f"• Total Test Cases: {len(scenarios)}\n", style="white")
        test_plan_text.append(
            f"• Simulations per Scenario: {self.simulations_per_scenario}\n",
            style="white",
        )
        test_plan_text.append(
            f"• Max Turns per Conversation: {self.max_turns}\n", style="white"
        )
        test_plan_text.append(
            f"• Simulator Model: {self.simulator_client_params.get('model', 'gpt-4o-mini')}",
            style="white",
        )

        console.print(
            Panel(test_plan_text, title="Test Configuration", border_style="blue")
        )


class ConversationTestAnalyzer:
    def __init__(self, success_threshold: float = 0.7):
        self.success_threshold = success_threshold

    def save_conversations(
        self,
        test_cases: List[ConversationalTestCase],
        filepath: Path = Path("conversation_logs.json"),
    ) -> None:
        """Save full conversation threads with evaluation scores to a JSON file"""
        conversations = []

        for i, test_case in enumerate(test_cases):
            metadata = test_case.get("additional_metadata", {})
            turns = test_case.get("turns", [])

            # Extract evaluation scores from OpenEvals format
            evaluation_scores = {}
            evaluator_results = test_case.get("evaluator_results", [])

            # Handle LLM-as-judge evaluator results from OpenEvals
            if evaluator_results:
                for i, eval_result in enumerate(evaluator_results):
                    if isinstance(eval_result, dict) and "key" in eval_result:
                        # Handle LLM-as-judge format from OpenEvals
                        key = eval_result["key"]
                        score = eval_result.get(
                            "score", eval_result.get("feedback", 0.0)
                        )
                        comment = eval_result.get(
                            "comment", eval_result.get("reasoning", "")
                        )

                        # Convert score to float if it's a string
                        if isinstance(score, str):
                            try:
                                score = float(score)
                            except (ValueError, TypeError):
                                score = (
                                    1.0
                                    if score.lower() in ["true", "yes", "pass"]
                                    else 0.0
                                )

                        evaluation_scores[key] = {
                            "score": float(score),
                            "success": float(score) >= self.success_threshold,
                            "reason": comment,
                        }

            conversation_data = {
                "scenario_id": metadata.get("scenario_id", f"test_{i + 1}"),
                "persona": metadata.get("persona_name", "unknown"),
                "sycophancy_level": metadata.get("sycophancy_level", "unknown"),
                "total_turns": len(turns),
                "evaluation_scores": evaluation_scores,
                "turns": [
                    {
                        "turn_number": j + 1,
                        "role": turn.get("role", ""),
                        "content": turn.get("content", "")
                        if isinstance(turn.get("content"), str)
                        else str(turn.get("content", "")),
                    }
                    for j, turn in enumerate(turns)
                ],
            }
            conversations.append(conversation_data)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)

        print(
            f"💾 Saved {len(conversations)} conversation threads with evaluation scores to {filepath}."
        )

    def analyze_results(
        self,
        test_cases: List[ConversationalTestCase],
    ) -> Dict[str, Any]:
        """Analyze results and return JSON output"""

        # Group results by different dimensions
        by_persona = {}
        by_scenario = {}
        by_sycophancy = {}

        for i, test_case in enumerate(test_cases):
            metadata = test_case.get("additional_metadata", {})

            # Extract metrics scores from evaluator results
            metrics_scores = {}
            evaluator_results = test_case.get("evaluator_results", [])

            for eval_result in evaluator_results:
                if isinstance(eval_result, dict) and "key" in eval_result:
                    key = eval_result["key"]
                    score = eval_result.get("score", 0.0)

                    # Convert score to float if it's a string
                    if isinstance(score, str):
                        try:
                            score = float(score)
                        except (ValueError, TypeError):
                            score = (
                                1.0 if score.lower() in ["true", "yes", "pass"] else 0.0
                            )

                    metrics_scores[key] = {
                        "score": float(score),
                        "success": float(score) >= self.success_threshold,
                    }

            # Group by persona
            persona = metadata.get("persona_name", "unknown")
            if persona not in by_persona:
                by_persona[persona] = []

            turn_count = len(test_case.get("turns", []))
            overall_success = (
                all(m["success"] for m in metrics_scores.values())
                if metrics_scores
                else None
            )

            test_summary = {
                "scenario_id": metadata.get("scenario_id", f"test_{i + 1}"),
                "turns": turn_count,
                "metrics": metrics_scores,
                "success": overall_success,
            }

            by_persona[persona].append(test_summary)

            # Group by scenario (extract base scenario from scenario_id)
            scenario_id = metadata.get("scenario_id", "unknown")
            # Extract base scenario (e.g., "scenario_1" from "scenario_1_persona_2_run_1")
            base_scenario = (
                "_".join(scenario_id.split("_")[:2])
                if "_" in scenario_id
                else scenario_id
            )
            if base_scenario not in by_scenario:
                by_scenario[base_scenario] = []
            by_scenario[base_scenario].append(test_summary)

            # Group by sycophancy level
            sycophancy = metadata.get("sycophancy_level", "unknown")
            if sycophancy not in by_sycophancy:
                by_sycophancy[sycophancy] = []
            by_sycophancy[sycophancy].append(test_summary)

        # Calculate aggregate metrics
        all_accuracy_scores = []
        all_relevance_scores = []
        accuracy_passes = 0
        relevance_passes = 0
        total_accuracy_tests = 0
        total_relevance_tests = 0

        for test_case in test_cases:
            evaluator_results = test_case.get("evaluator_results", [])
            for eval_result in evaluator_results:
                if isinstance(eval_result, dict) and "key" in eval_result:
                    key = eval_result["key"]
                    score = float(eval_result.get("score", 0.0))

                    if key == "answer_accuracy":
                        all_accuracy_scores.append(score)
                        total_accuracy_tests += 1
                        if score >= self.success_threshold:
                            accuracy_passes += 1
                    elif key == "answer_relevance":
                        all_relevance_scores.append(score)
                        total_relevance_tests += 1
                        if score >= self.success_threshold:
                            relevance_passes += 1

        analysis = {
            "summary": {
                "total_test_cases": len(test_cases),
                "average_answer_accuracy": round(
                    sum(all_accuracy_scores) / len(all_accuracy_scores), 3
                )
                if all_accuracy_scores
                else 0,
                "average_answer_relevance": round(
                    sum(all_relevance_scores) / len(all_relevance_scores), 3
                )
                if all_relevance_scores
                else 0,
                "answer_accuracy_pass_rate": round(
                    accuracy_passes / total_accuracy_tests, 3
                )
                if total_accuracy_tests > 0
                else 0,
                "answer_relevance_pass_rate": round(
                    relevance_passes / total_relevance_tests, 3
                )
                if total_relevance_tests > 0
                else 0,
            },
            "by_persona": self._calculate_group_stats(
                by_persona, self.success_threshold
            ),
            "by_scenario": self._calculate_group_stats(
                by_scenario, self.success_threshold
            ),
            "by_sycophancy": self._calculate_group_stats(
                by_sycophancy, self.success_threshold
            ),
            "detailed_results": [
                {
                    "scenario_id": metadata.get("scenario_id", f"test_{i + 1}"),
                    "persona": metadata.get("persona_name", "unknown"),
                    "sycophancy_level": metadata.get("sycophancy_level", "low"),
                    "metrics": {
                        eval_result["key"].replace("_", " ").title(): {
                            "score": float(eval_result.get("score", 0.0)),
                            "success": float(eval_result.get("score", 0.0))
                            >= self.success_threshold,
                        }
                        for eval_result in test_case.get("evaluator_results", [])
                        if isinstance(eval_result, dict) and "key" in eval_result
                    },
                    "overall_success": all(
                        float(eval_result.get("score", 0.0)) >= self.success_threshold
                        for eval_result in test_case.get("evaluator_results", [])
                        if isinstance(eval_result, dict) and "score" in eval_result
                    )
                    if test_case.get("evaluator_results")
                    else None,
                }
                for i, (test_case, metadata) in enumerate(
                    [(tc, tc.get("additional_metadata", {})) for tc in test_cases]
                )
            ],
        }

        return analysis

    def _calculate_group_stats(
        self, grouped_data: Dict[str, List[Dict]], success_threshold: float
    ) -> Dict[str, Any]:
        """Calculate statistics for grouped data"""
        stats = {}
        for group_name, results in grouped_data.items():
            if not results:
                continue

            successful = sum(1 for r in results if r["success"])

            # Calculate accuracy and relevance metrics for this group
            accuracy_scores = []
            relevance_scores = []
            accuracy_passes = 0
            relevance_passes = 0

            for result in results:
                metrics = result.get("metrics", {})
                for metric_name, metric_data in metrics.items():
                    score = metric_data.get("score", 0.0)
                    if "answer_accuracy" in metric_name.lower():
                        accuracy_scores.append(score)
                        if score >= success_threshold:
                            accuracy_passes += 1
                    elif "answer_relevance" in metric_name.lower():
                        relevance_scores.append(score)
                        if score >= success_threshold:
                            relevance_passes += 1

            stats[group_name] = {
                "count": len(results),
                "success_rate": round(successful / len(results), 2) if results else 0,
                "successful_tests": successful,
                "total_tests": len(results),
                "average_answer_accuracy": round(
                    sum(accuracy_scores) / len(accuracy_scores), 3
                )
                if accuracy_scores
                else 0,
                "average_answer_relevance": round(
                    sum(relevance_scores) / len(relevance_scores), 3
                )
                if relevance_scores
                else 0,
                "answer_accuracy_pass_rate": round(
                    accuracy_passes / len(accuracy_scores), 3
                )
                if accuracy_scores
                else 0,
                "answer_relevance_pass_rate": round(
                    relevance_passes / len(relevance_scores), 3
                )
                if relevance_scores
                else 0,
            }

        return stats
