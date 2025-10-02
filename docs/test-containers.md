# Available Test Containers

ASQI provides several pre-built test containers for different testing scenarios. All containers are available on Docker Hub and can be pulled using the installation commands.

## Container Overview

- **Mock Tester** (`asqiengineer/test-container:mock_tester-latest`): Basic test container for development and validation
- **Garak Security Tester** (`asqiengineer/test-container:garak-latest`): LLM security vulnerability assessment with 40+ attack vectors  
- **Inspect Evals** (`asqiengineer/test-container:inspect_evals-latest`): Comprehensive evaluation suite with 80+ tasks across cybersecurity, mathematics, reasoning, knowledge, bias, and safety domains
- **Chatbot Simulator** (`asqiengineer/test-container:chatbot_simulator-latest`): Persona-based conversational testing with multi-turn dialogue
- **TrustLLM** (`asqiengineer/test-container:trustllm-latest`): Comprehensive trustworthiness evaluation framework
- **DeepTeam** (`asqiengineer/test-container:deepteam-latest`): Red teaming library for adversarial robustness testing
- **PyRIT** (`asqiengineer/test-container:pyrit_container-latest`): Microsoft's Python Risk Identification Tool for red teaming and security testing with single-turn and multi-turn attack strategies

All containers are available on Docker Hub and can be pulled using the commands shown in the installation section.

## Test Container Examples

**Note:** Certain tests include volume mounting to save detailed logs. You might need to configure the volume output mount accordingly.

ASQI provides ready-to-use example configurations for each test container. Download and run these examples to get started quickly:

### Mock Tester Example

Basic test container for development and validation:

```bash
# Download and run the basic demo
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/demo_test.yaml
asqi execute-tests -t demo_test.yaml -s demo_systems.yaml -o results.json
```

### Garak Security Testing Example

LLM security vulnerability assessment with multiple attack probes:

```bash
# Download security test configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/garak_test.yaml

# Run security tests (includes encoding attacks and prompt injection)
asqi execute-tests -t garak_test.yaml -s demo_systems.yaml -o security_results.json
```

**Note**: Certain tests requires a `OPENAI_API_KEY` so it is recommended to pass it in via the `env_file` field as part of the system config.

### Inspect Evals Example

Comprehensive evaluation suite with 80+ academic benchmarks:

```bash
# Download inspect evals configuration (cybersecurity focus)
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/inspect_evals/cybersecurity.yaml

# Run cybersecurity evaluation
asqi execute-tests -t cybersecurity.yaml -s demo_systems.yaml -o inspect_results.json

# Or run knowledge evaluation
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/inspect_evals/knowledge.yaml
asqi execute-tests -t knowledge.yaml -s demo_systems.yaml -o knowledge_results.json
```

### Chatbot Simulator Example

Persona-based conversational testing with multi-turn dialogue:

```bash
# Download chatbot simulation configuration  
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/chatbot_simulator_test.yaml

# Run conversational tests 
asqi execute-tests -t chatbot_simulator_test.yaml -s demo_systems.yaml -o chatbot_results.json
```

### TrustLLM Example

Comprehensive trustworthiness evaluation across multiple dimensions:

```bash
# Download trustworthiness evaluation configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/trustllm_test.yaml

# Run trustworthiness evaluation
asqi execute-tests -t trustllm_test.yaml -s demo_systems.yaml -o trustllm_results.json
```

### DeepTeam Red Teaming Example

Advanced adversarial robustness testing:

```bash
# Download red teaming configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/deepteam_test.yaml

# Run red teaming tests
asqi execute-tests -t deepteam_test.yaml -s demo_systems.yaml -o redteam_results.json
```

### PyRIT Security Testing Example

Microsoft's Python Risk Identification Tool with single-turn and multi-turn attack strategies:

```bash
# Download PyRIT configuration
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/pyrit_test.yaml

# Run PyRIT security tests with prompt obfuscation
asqi execute-tests -t pyrit_test.yaml -s demo_systems.yaml -o pyrit_results.json
```

PyRIT supports various attack strategies:
- **Single-turn attacks** (PromptSendingAttack): Direct prompt injection attempts
- **Multi-turn attacks** (RedTeamingAttack): Adaptive conversational red teaming
- **Prompt converters**: Base64, ROT13, Leetspeak, Unicode, ASCII smuggling

See the [PyRIT container documentation](../test_containers/pyrit_container/README.md) for detailed configuration options.

## Evaluating Score Cards

Score cards provide automated assessment of test results against business-relevant criteria. ASQI engineer includes a flexible grading engine that evaluates individual test executions and provides structured feedback.

### How Score Cards Work

Score cards consist of **indicators** that evaluate specific metrics from test results:
- **Apply to specific tests**: Target individual test names from your test suite
- **Extract metrics**: Pull any field from test container JSON output
- **Assessment criteria**: Define pass/fail thresholds with business-friendly outcomes
- **Individual evaluation**: Each test execution is assessed separately (no aggregation)

### Basic Score Card Example

Using the simple example score card for mock tester results:

```bash
# First run a test to generate results
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/suites/demo_test.yaml
asqi execute-tests -t demo_test.yaml -s demo_systems.yaml -o test_results.json

# Download and apply basic score card
curl -O https://raw.githubusercontent.com/asqi-engineer/asqi-engineer/main/config/score_cards/example_score_card.yaml
asqi evaluate-score-cards --input-file test_results.json -r example_score_card.yaml -o results_with_grades.json

# Or run end-to-end (tests + score card evaluation)
asqi execute -t demo_test.yaml -s demo_systems.yaml -r example_score_card.yaml -o complete_results.json
```

For detailed information about specific test containers, see [LLM Test Containers](llm-test-containers.md).