# Quick Start

### Requirements

- **Python 3.12+** is required
- Docker for running test containers

## Installation

Get started with ASQI Engineer in 3 simple steps:

**1. Install the package:**

```bash
pip install asqi-engineer
```

**2. Run the setup script:**

```bash
curl -sSL https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/setup.sh | bash
```

**3. Configure and run:**

```bash
# Start the services and run your first test:
docker compose up -d
asqi execute-tests -t config/suites/demo_test.yaml -s config/systems/demo_systems.yaml
```

This short flow should download a demo test container and generate the test results in `output.json`.

## Getting Started with AI System Testing

To configure the AI systems you want to test:

1. **Configure your systems (`config/systems/demo_systems.yaml`):**

   ```yaml
   systems:
     my_llm_service:
       type: "llm_api"
       params:
         base_url: "http://localhost:4000/v1" # LiteLLM proxy
         model: "gpt-4o-mini"
         api_key: "sk-1234"

     openai_gpt4o_mini:
       type: "llm_api"
       params:
         base_url: "https://api.openai.com/v1"
         model: "gpt-4o-mini"
         api_key: "${OPENAI_API_KEY}" # Uses environment variable
   ```

   You should also modify the `.env` file created by the setup script to add your actual API keys.

2. **Download the required test packages**

   ```bash
    # Core test containers
    docker pull asqiengineer/test-container:mock_tester-latest
    docker pull asqiengineer/test-container:garak-latest
    docker pull asqiengineer/test-container:inspect_evals-latest
    docker pull asqiengineer/test-container:chatbot_simulator-latest
    docker pull asqiengineer/test-container:trustllm-latest
    docker pull asqiengineer/test-container:deepteam-latest

    # Verify installation
    asqi --help
    ```

3. **Try Out the CLI commands**

    **Test execution only** (great for development and debugging):
    ```bash
    asqi execute-tests \
      -t config/suites/demo_test.yaml \
      -s config/systems/demo_systems.yaml \
      -o results.json
    ```

    **Score card evaluation only** (apply different evaluation criteria to existing results):
    ```bash
    asqi evaluate-score-cards \
      --input-file results.json \
      -r config/score_cards/example_score_card.yaml \
      -o results_with_grades.json
    ```

    **End-to-end execution** (tests + score card evaluation in one workflow):
    ```bash
    asqi execute \
      -t config/suites/demo_test.yaml \
      -s config/systems/demo_systems.yaml \
      -r config/score_cards/example_score_card.yaml \
      -o complete_results.json
    ```

4. **Explore Different Test Packages**

With all configurations downloaded, you can immediately try different testing scenarios:

  ```bash
  # View all available configurations
  ls config/suites/       # All test suites
  ls config/score_cards/  # All score cards  
  ls config/systems/      # System configurations

  # Security testing with Garak
  asqi execute-tests -t config/suites/garak_test.yaml -s config/systems/demo_systems.yaml

  # Conversational testing with personas  
  asqi execute-tests -t config/suites/chatbot_simulator_test.yaml -s config/systems/demo_systems.yaml

  # TrustLLM evaluation
  asqi execute-tests -t config/suites/trustllm_test.yaml -s config/systems/demo_systems.yaml

  # Inspect evaluation
  asqi execute-tests -t config/suites/inspect_evals/reasoning.yaml -s config/systems/demo_systems.yaml

  # Red team testing
  asqi execute-tests -t config/suites/deepteam_test.yaml -s config/systems/demo_systems.yaml

  # Image generation and VLM evaluation
  asqi execute-tests -t config/suites/image_vlm_test.yaml -s config/systems/image_systems.yaml
  ```

## Next Steps

- Review [Available Test Containers](test-containers.md) to see all testing frameworks and examples
- Explore [Configuration](configuration.md) to understand how to configure systems, test suites, and score cards
- Check out [Examples](examples.md) for advanced usage scenarios
- See [CLI Reference](cli.rst) for complete command documentation
