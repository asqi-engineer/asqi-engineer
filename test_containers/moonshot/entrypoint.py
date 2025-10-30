import argparse
import asyncio
import json
import sys
from pathlib import Path
from slugify import slugify

moonshot_imported = False
moonshot_error = None
try:
    from moonshot.api import (
        api_create_endpoint,
        api_get_all_endpoint_name,
        api_delete_endpoint,
        api_create_runner,
        api_get_all_runner_name,
        api_load_runner,
        api_get_all_run,
        api_set_environment_variables,
    )

    moonshot_imported = True
except Exception as e:
    moonshot_error = str(e)


def setup_moonshot_environment():
    """Setup Moonshot environment variables pointing to moonshot-data directories"""
    moonshot_data_path = Path("/app/moonshot-data")

    if not moonshot_data_path.exists():
        raise RuntimeError(f"moonshot-data directory not found at {moonshot_data_path}")

    env_vars = {
        "ATTACK_MODULES": str(moonshot_data_path / "attack-modules"),
        "BOOKMARKS": str(moonshot_data_path / "generated-outputs/bookmarks"),
        "CONNECTORS": str(moonshot_data_path / "connectors"),
        "CONNECTORS_ENDPOINTS": str(moonshot_data_path / "connectors-endpoints"),
        "CONTEXT_STRATEGY": str(moonshot_data_path / "context-strategy"),
        "COOKBOOKS": str(moonshot_data_path / "cookbooks"),
        "DATABASES": str(moonshot_data_path / "generated-outputs/databases"),
        "DATABASES_MODULES": str(moonshot_data_path / "databases-modules"),
        "DATASETS": str(moonshot_data_path / "datasets"),
        "IO_MODULES": str(moonshot_data_path / "io-modules"),
        "METRICS": str(moonshot_data_path / "metrics"),
        "PROMPT_TEMPLATES": str(moonshot_data_path / "prompt-templates"),
        "RECIPES": str(moonshot_data_path / "recipes"),
        "RESULTS": str(moonshot_data_path / "generated-outputs/results"),
        "RESULTS_MODULES": str(moonshot_data_path / "results-modules"),
        "RUNNERS": str(moonshot_data_path / "generated-outputs/runners"),
        "RUNNERS_MODULES": str(moonshot_data_path / "runners-modules"),
    }

    for key in ["BOOKMARKS", "DATABASES", "RESULTS", "RUNNERS"]:
        Path(env_vars[key]).mkdir(parents=True, exist_ok=True)

    api_set_environment_variables(env_vars)


def main() -> None:
    parser = argparse.ArgumentParser(description="Moonshot test container")
    parser.add_argument(
        "--systems-params", required=True, help="Systems parameters JSON"
    )
    parser.add_argument("--test-params", required=True, help="Test parameters JSON")
    args = parser.parse_args()

    if moonshot_imported:
        try:
            setup_moonshot_environment()
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "status": "Failed to setup Moonshot environment",
            }
            print(json.dumps(error_result, indent=2))
            sys.exit(1)

    try:
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)

        sut = systems_params.get("system_under_test", {})
        base_url = sut.get("base_url")
        api_key = sut.get("api_key")
        sut_model = sut.get("model")

        recipe_name = test_params.get("recipe")
        prompt_selection_percentage = test_params.get(
            "prompt_selection_percentage", 100
        )
        random_seed = test_params.get("random_seed", 1)
        system_prompt = ""

        if not all([moonshot_imported, base_url, api_key, sut_model, recipe_name]):
            raise ValueError("Missing required parameters")

        # Create endpoint
        endpoint_name = "system_under_test"
        if endpoint_name in api_get_all_endpoint_name():
            api_delete_endpoint(endpoint_name)

        endpoint_id = api_create_endpoint(
            name=endpoint_name,
            connector_type="openai-connector",
            uri=base_url,
            token=api_key,
            max_calls_per_second=10,
            max_concurrency=1,
            model=sut_model,
            params={"timeout": 300, "max_attempts": 3, "temperature": 0.5},
        )

        # Create runner
        runner_name = f"runner-{recipe_name}"
        slugify_id = slugify(runner_name, lowercase=True)
        if slugify_id in api_get_all_runner_name():
            runner = api_load_runner(slugify_id)
        else:
            runner = api_create_runner(runner_name, [endpoint_id])

        # Run recipe
        async def run_recipe():
            await runner.run_recipes(
                recipes=[recipe_name],
                prompt_selection_percentage=prompt_selection_percentage,
                random_seed=random_seed,
                system_prompt=system_prompt,
                runner_processing_module="benchmarking",
                result_processing_module="benchmarking-result",
            )
            await runner.close()

            # Get all runs for this runner
            runner_runs = api_get_all_run(runner.id)
            if not runner_runs:
                return None

            # Get the last run
            last_run = runner_runs[-1]

            # Extract results from the run object
            return {
                "results": last_run.get("results", {}),
                "metadata": {
                    "id": last_run.get("run_id"),
                    "start_time": last_run.get("start_time"),
                    "end_time": last_run.get("end_time"),
                    "duration": last_run.get("duration"),
                    "status": str(last_run.get("status", "")),
                    "endpoints": last_run.get("endpoints", []),
                },
            }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        run_result = loop.run_until_complete(run_recipe())
        loop.close()

        # Return the complete results
        if run_result:
            result = {
                "success": True,
                "recipe": recipe_name,
                "results": run_result,
            }
        else:
            result = {
                "success": False,
                "error": "No results generated",
                "status": "failed",
            }

        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)

    except Exception as exc:
        import traceback

        error_result = {
            "success": False,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "status": "failed",
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
