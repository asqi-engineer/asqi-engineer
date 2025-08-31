# LLM Test Containers

ASQI Engineer provides several pre-built test containers specifically designed for comprehensive LLM system evaluation. Each container implements industry-standard testing frameworks and provides structured evaluation metrics.

## Mock Tester

**Purpose**: Development and validation testing with configurable simulation.

**Framework**: Custom lightweight testing framework  
**Location**: `test_containers/mock_tester/`

### System Requirements
- **System Under Test**: `llm_api` (required) - The LLM system being tested

### Input Parameters
- `delay_seconds` (integer, optional): Seconds to sleep simulating processing work

### Output Metrics
- `success` (boolean): Whether test execution completed successfully
- `score` (float): Mock test score (0.0 to 1.0)  
- `delay_used` (integer): Actual delay in seconds used
- `base_url` (string): API endpoint that was accessed
- `model` (string): Model name that was tested

### Example Configuration
```yaml
test_suite:
  - name: "basic_compatibility_check"
    image: "my-registry/mock_tester:latest"
    systems_under_test: ["my_llm_service"]
    params:
      delay_seconds: 2
```

### Build Instructions
```bash
cd test_containers/mock_tester
docker build -t my-registry/mock_tester:latest .
```

---

## Garak Security Tester

**Purpose**: Comprehensive LLM security vulnerability assessment.

**Framework**: [Garak](https://github.com/NVIDIA/garak) - Industry-standard LLM security testing  
**Location**: `test_containers/garak/`

### System Requirements
- **System Under Test**: `llm_api` (required) - The LLM system being tested for security vulnerabilities

### Input Parameters
- `probes` (list, optional): List of specific garak probes to execute
  - **Available probe categories**: `encoding`, `promptinject`, `dan`, `malwaregen`, `misleading`, `xss`, `continuation`, `av_spam_scanning`, `donotanswer`, and 20+ more
  - **Example probes**: `["promptinject", "encoding.InjectHex", "dan.DAN_Jailbreak"]`
- `generations` (integer, optional): Number of generations per probe (default: 1)
- `parallel_attempts` (integer, optional): Parallel probe execution (default: 8)

### Output Metrics
- `success` (boolean): Whether garak execution completed successfully
- `score` (float): Overall pass rate across all probes (0.0 to 1.0)
- `vulnerabilities_found` (integer): Total number of failed attempts (potential vulnerabilities)
- `total_attempts` (integer): Total number of probe attempts made
- `probes_used` (list): List of probes that were executed
- `generations` (integer): Number of generations per probe used
- `sut_type` (string): Type of system under test
- `probe_results` (object): Detailed results by probe and detector with individual scores

### Key Security Probes

**Prompt Injection**:
- `promptinject.HijackHateHumans`: Tests for harmful instruction hijacking
- `promptinject.HijackKillHumans`: Tests for violent instruction injection
- `promptinject.HijackLongPrompt`: Long-form prompt injection attacks

**Encoding Attacks**:
- `encoding.InjectBase64`: Base64-encoded malicious instructions
- `encoding.InjectHex`: Hexadecimal-encoded attacks
- `encoding.InjectROT13`: ROT13-encoded instruction injection
- `encoding.InjectMorse`: Morse code-based encoding attacks

**Jailbreak Attempts**:
- `dan.DAN_Jailbreak`: Standard DAN (Do Anything Now) jailbreak
- `dan.AutoDAN`: Automated jailbreak generation
- `dan.ChatGPT_Developer_Mode_v2`: Developer mode exploitation

**Content Generation**:
- `malwaregen.Payload`: Malware code generation attempts
- `malwaregen.Evasion`: Evasion technique generation
- `misleading.FalseAssertion`: False information generation tests

### Example Configuration
```yaml
test_suite:
  - name: "comprehensive_security_scan"
    image: "my-registry/garak:latest"
    systems_under_test: ["production_model"]
    params:
      probes: [
        "promptinject",
        "encoding.InjectBase64",
        "encoding.InjectHex", 
        "dan.DAN_Jailbreak",
        "dan.AutoDAN",
        "malwaregen.Payload",
        "misleading.FalseAssertion"
      ]
      generations: 20
      parallel_attempts: 6
```

### Build Instructions
```bash
cd test_containers/garak
docker build -t my-registry/garak:latest .
```

### Environment Requirements
```bash
export OPENAI_API_KEY="your-api-key"
# Or other provider-specific keys based on your system configuration
```

---

## DeepTeam Red Team Tester

**Purpose**: Advanced adversarial robustness testing with multi-system orchestration.

**Framework**: [DeepEval](https://github.com/confident-ai/deepeval) DeepTeam - Advanced red teaming library  
**Location**: `test_containers/deepteam/`

### System Requirements
- **System Under Test**: `llm_api` (required) - Primary system being tested
- **Simulator System**: `llm_api` (optional) - System for generating adversarial attacks and scenarios
- **Evaluator System**: `llm_api` (optional) - System for evaluating target responses

### Input Parameters
- `vulnerabilities` (list, optional): Vulnerability configurations with types
  - Each item has `name` and optional `types`
  - **Available vulnerabilities**: `bias`, `pii_leakage`, `prompt_leakage`, `toxicity`, `misinformation`, `excessive_agency`, `robustness`, `competition`, `intellectual_property`, `illegal_activity`, `graphic_content`, `personal_safety`, `unauthorized_access`, `custom`
- `attacks` (list, optional): Attack method names
  - **Available attacks**: `base64`, `graybox`, `leetspeak`, `math_problem`, `multilingual`, `prompt_injection`, `prompt_probing`, `roleplay`, `rot13`, `crescendo_jailbreaking`, `linear_jailbreaking`, `tree_jailbreaking`, `sequential_jailbreak`, `bad_likert_judge`
- `max_concurrent` (integer, optional): Maximum concurrent operations (default: 10)
- `attacks_per_vulnerability_type` (integer, optional): Attacks per vulnerability type (default: 3)
- `target_purpose` (string, optional): Description of target system's purpose for context

### Output Metrics
- `success` (boolean): Whether system passed security threshold (80% pass rate)
- `pass_rate` (float): Proportion of tests that passed (secure responses)
- `failure_rate` (float): Proportion of tests that failed (vulnerable responses)  
- `total_tests` (integer): Total number of red team tests performed
- `total_passing` (integer): Number of secure responses
- `total_failing` (integer): Number of vulnerable responses
- `total_errored` (integer): Number of tests with errors
- `vulnerability_stats` (object): Detailed per-vulnerability statistics including pass rates
- `attack_stats` (object): Detailed per-attack method statistics
- `vulnerabilities_tested` (list): Vulnerability types that were tested
- `attacks_used` (list): Attack methods that were used
- `model_tested` (string): Model identifier that was tested

### Example Configuration
```yaml
test_suite:
  - name: "advanced_red_team_assessment"
    image: "my-registry/deepteam:latest"
    systems_under_test: ["target_chatbot"]
    systems:
      simulator_system: "gpt4o_attacker"
      evaluator_system: "claude_judge"
    params:
      vulnerabilities:
        - name: "bias"
          types: ["gender", "racial", "political"]
        - name: "toxicity"
        - name: "pii_leakage"
        - name: "prompt_leakage"
      attacks: [
        "prompt_injection",
        "roleplay", 
        "crescendo_jailbreaking",
        "linear_jailbreaking",
        "leetspeak"
      ]
      attacks_per_vulnerability_type: 8
      max_concurrent: 6
      target_purpose: "customer service chatbot for financial services"
```

### Build Instructions
```bash
cd test_containers/deepteam
docker build -t my-registry/deepteam:latest .
```

---

## Chatbot Simulator

**Purpose**: Multi-turn conversational testing with persona-based simulation and LLM-as-judge evaluation.

**Framework**: Custom conversation simulation with LLM evaluation  
**Location**: `test_containers/chatbot_simulator/`

### System Requirements
- **System Under Test**: `llm_api` (required) - The chatbot system being tested
- **Simulator System**: `llm_api` (optional) - LLM for generating personas and conversation scenarios
- **Evaluator System**: `llm_api` (optional) - LLM for evaluating conversation quality

### Input Parameters
- `chatbot_purpose` (string, required): Description of the chatbot's purpose and domain
- `custom_scenarios` (list, optional): List of scenario objects with `input` and `expected_output` keys
- `custom_personas` (list, optional): Custom persona names (e.g., `["busy executive", "enthusiastic buyer"]`)
- `num_scenarios` (integer, optional): Number of conversation scenarios to generate if custom scenarios not provided
- `max_turns` (integer, optional): Maximum turns per conversation (default: 4)
- `sycophancy_levels` (list, optional): Sycophancy levels to cycle through (default: `["low", "high"]`)
- `simulations_per_scenario` (integer, optional): Simulation runs per scenario-persona combination (default: 1)
- `success_threshold` (float, optional): Threshold for evaluation success (default: 0.7)
- `max_concurrent` (integer, optional): Maximum concurrent conversation simulations (default: 3)

### Output Metrics
- `success` (boolean): Whether test execution completed successfully
- `total_test_cases` (integer): Total number of conversation test cases generated
- `average_answer_accuracy` (float): Average accuracy score across all conversations (0.0 to 1.0)
- `average_answer_relevance` (float): Average relevance score across all conversations (0.0 to 1.0)
- `answer_accuracy_pass_rate` (float): Percentage of conversations passing accuracy threshold
- `answer_relevance_pass_rate` (float): Percentage of conversations passing relevance threshold
- `by_persona` (object): Performance metrics broken down by persona type
- `by_scenario` (object): Performance metrics broken down by test scenario
- `by_sycophancy` (object): Performance metrics broken down by sycophancy level

### Example Configuration
```yaml
test_suite:
  - name: "customer_service_conversation_test"
    image: "my-registry/chatbot_simulator:latest"
    systems_under_test: ["customer_service_bot"]
    systems:
      simulator_system: "gpt4o_customer_simulator"
      evaluator_system: "claude_conversation_judge"
    params:
      chatbot_purpose: "customer service for e-commerce platform specializing in electronics"
      custom_scenarios:
        - input: "I want to return a laptop I bought 2 months ago because it's defective"
          expected_output: "Helpful explanation of return policy and steps to process return"
        - input: "My order shipped but tracking shows it's been stuck for a week"
          expected_output: "Empathetic response with concrete steps to investigate and resolve"
      custom_personas: [
        "frustrated customer with urgent need",
        "polite customer seeking information", 
        "tech-savvy customer with detailed questions",
        "elderly customer needing extra guidance"
      ]
      num_scenarios: 12
      max_turns: 6
      sycophancy_levels: ["low", "medium", "high"]
      success_threshold: 0.8
      max_concurrent: 4
```

### Build Instructions
```bash
cd test_containers/chatbot_simulator
docker build -t my-registry/chatbot_simulator:latest .
```

---

## TrustLLM Tester

**Purpose**: Comprehensive trustworthiness evaluation across 6 dimensions using academic-grade benchmarks.

**Framework**: [TrustLLM](https://github.com/HowieHwong/TrustLLM) - Academic trustworthiness evaluation framework  
**Location**: `test_containers/trustllm/`

### System Requirements
- **System Under Test**: `llm_api` (required) - The LLM system being evaluated for trustworthiness

### Input Parameters
- `test_type` (string, required): Test dimension to evaluate
  - **Available dimensions**: `ethics`, `privacy`, `fairness`, `truthfulness`, `robustness`, `safety`
- `datasets` (list, optional): Specific datasets for the chosen test type (without .json extension)
  - **Ethics datasets**: `awareness`, `explicit_moralchoice`, `implicit_ETHICS`, `implicit_SocialChemistry101`
  - **Privacy datasets**: `privacy_awareness_confAIde`, `privacy_awareness_query`, `privacy_leakage`
  - **Fairness datasets**: `disparagement`, `preference`, `stereotype_agreement`, `stereotype_query_test`, `stereotype_recognition`
  - **Truthfulness datasets**: `external`, `hallucination`, `golden_advfactuality`, `internal`, `sycophancy`
  - **Robustness datasets**: `ood_detection`, `ood_generalization`, `AdvGLUE`, `AdvInstruction`
  - **Safety datasets**: `jailbreak`, `exaggerated_safety`, `misuse`
- `max_new_tokens` (integer, optional): Maximum tokens in LLM responses (default: 1024)
- `max_rows` (integer, optional): Maximum rows per dataset for faster testing (default: 20)

### Output Metrics
- `success` (boolean): Whether TrustLLM evaluation completed successfully
- `test_type` (string): The test dimension that was evaluated
- `datasets_tested` (list): List of dataset names that were actually tested
- `dataset_results` (object): Individual results for each dataset with generation and evaluation results

### Example Configuration
```yaml
test_suite:
  - name: "ethics_evaluation"
    image: "my-registry/trustllm:latest"
    systems_under_test: ["target_model"]
    params:
      test_type: "ethics"
      datasets: ["awareness", "explicit_moralchoice"]
      max_new_tokens: 512
      max_rows: 50

  - name: "safety_assessment"
    image: "my-registry/trustllm:latest"
    systems_under_test: ["target_model"]
    params:
      test_type: "safety"
      datasets: ["jailbreak", "misuse"]
      max_rows: 30

  - name: "fairness_evaluation"
    image: "my-registry/trustllm:latest"
    systems_under_test: ["target_model"]
    params:
      test_type: "fairness"
      # Uses all fairness datasets by default
      max_rows: 25
```

### Build Instructions
```bash
cd test_containers/trustllm
docker build -t my-registry/trustllm:latest .
```

---

## Computer Vision Test Containers

While ASQI's primary focus is LLM testing, it also includes specialized containers for computer vision evaluation:

### Computer Vision Tester

**Purpose**: General computer vision model testing and evaluation.

**Location**: `test_containers/computer_vision/`

### CV Tester

**Purpose**: Specialized computer vision testing framework with advanced detection capabilities.

**Location**: `test_containers/cv_tester/`

---

## Multi-Container Testing Strategies

### Security-Focused Assessment

Combine multiple security testing frameworks for comprehensive coverage:

```yaml
suite_name: "Complete Security Assessment"
test_suite:
  # Fast baseline security scan
  - name: "baseline_security"
    image: "my-registry/garak:latest"
    systems_under_test: ["target_model"]
    params:
      probes: ["promptinject", "encoding.InjectBase64", "dan.DAN_Jailbreak"]
      generations: 10
      parallel_attempts: 8

  # Comprehensive adversarial testing
  - name: "advanced_red_team"
    image: "my-registry/deepteam:latest"
    systems_under_test: ["target_model"]
    systems:
      simulator_system: "gpt4o_attacker"
      evaluator_system: "claude_security_judge"
    params:
      vulnerabilities:
        - name: "bias"
          types: ["gender", "racial"]
        - name: "toxicity"
        - name: "pii_leakage"
      attacks: ["prompt_injection", "jailbreaking", "roleplay"]
      attacks_per_vulnerability_type: 5

  # Trustworthiness evaluation
  - name: "trustworthiness_assessment"
    image: "my-registry/trustllm:latest"
    systems_under_test: ["target_model"]
    params:
      evaluation_dimensions: ["truthfulness", "safety", "fairness"]
```

### Quality and Performance Testing

Evaluate conversational quality and system performance:

```yaml
suite_name: "Chatbot Quality and Performance"
test_suite:
  # Conversation quality assessment
  - name: "conversation_quality"
    image: "my-registry/chatbot_simulator:latest"
    systems_under_test: ["customer_bot"]
    systems:
      simulator_system: "gpt4o_customer"
      evaluator_system: "claude_judge"
    params:
      chatbot_purpose: "customer support for financial services"
      num_scenarios: 20
      max_turns: 8
      sycophancy_levels: ["low", "high"]
      success_threshold: 0.8

  # Performance and reliability
  - name: "performance_baseline"
    image: "my-registry/mock_tester:latest"
    systems_under_test: ["customer_bot"]
    params:
      delay_seconds: 0  # Test response time
```

## Container Selection Guide

### Choose the Right Container for Your Use Case

**For Security Assessment**:
- **Garak**: Comprehensive vulnerability scanning with 40+ probes
- **DeepTeam**: Advanced red teaming with multi-system orchestration
- **Combined**: Use both for complete security coverage

**For Conversational Quality**:
- **Chatbot Simulator**: Multi-turn dialogue testing with persona-based evaluation
- **TrustLLM**: Academic-grade trustworthiness assessment

**For Development and Validation**:
- **Mock Tester**: Quick compatibility and configuration validation

**For Research and Benchmarking**:
- **TrustLLM**: Standardized academic benchmarks
- **DeepTeam**: Research-grade adversarial evaluation

### Performance Considerations

**Container Resource Requirements**:
- **Mock Tester**: Minimal resources, fast execution
- **Garak**: Medium resources, depends on probe selection and generations
- **Chatbot Simulator**: Medium-high resources, depends on conversation complexity
- **DeepTeam**: High resources, requires multiple LLM API calls
- **TrustLLM**: High resources, comprehensive benchmark evaluation

**Optimization Tips**:
- Start with smaller `generations` and `num_scenarios` for development
- Use `parallel_attempts` and `max_concurrent` to balance speed vs. resource usage
- Test with Mock Tester first to validate configuration before expensive tests
- Use `--concurrent-tests` CLI option to run multiple containers in parallel

## Environment and API Key Management

### Required Environment Variables by Container

**Garak**:
```bash
# Requires API key for target system
export OPENAI_API_KEY="sk-your-key"        # For OpenAI systems
export ANTHROPIC_API_KEY="sk-ant-your-key" # For Anthropic systems
```

**DeepTeam**:
```bash
# Requires API keys for all three systems (target, simulator, evaluator)
export OPENAI_API_KEY="sk-your-openai-key"
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
```

**Chatbot Simulator**:
```bash
# Requires API keys for target, simulator, and evaluator systems
export OPENAI_API_KEY="sk-your-openai-key"      # For GPT-based simulation
export ANTHROPIC_API_KEY="sk-ant-your-key"      # For Claude-based evaluation
```

### LiteLLM Proxy Integration

All containers work seamlessly with LiteLLM proxy for unified provider access:

**LiteLLM Configuration** (config.yaml):
```yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: env/OPENAI_API_KEY
      
  - model_name: claude-3-5-sonnet
    litellm_params:
      model: anthropic/claude-3-5-sonnet-20241022
      api_key: env/ANTHROPIC_API_KEY
```

**System Configuration**:
```yaml
systems:
  proxy_target:
    type: "llm_api"
    params:
      base_url: "http://localhost:4000/v1"
      model: "gpt-4o"
      api_key: "sk-1234"  # LiteLLM proxy key
      
  proxy_evaluator:
    type: "llm_api"
    params:
      base_url: "http://localhost:4000/v1"
      model: "claude-3-5-sonnet"
      api_key: "sk-1234"
```

This approach centralizes API key management and provides unified access to 100+ LLM providers through a single proxy endpoint.