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
    description: "System Basic Compatibility Test"
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
    description: "Scans the Model For Security Risk and Vulnerabilities"
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
    description: "Runs Attacks to Test Against Red Team Assessments"
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
    description: "Tests How the Chatbot Handles Real Customer Conversations"
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

## Inspect Evals Tester

**Purpose**: Comprehensive evaluation suite with 100+ tasks across multiple domains including cybersecurity, mathematics, reasoning, knowledge, bias, and safety.

**Framework**: [Inspect Evals](https://github.com/UKGovernmentBEIS/inspect_evals) - UK Government BEIS evaluation framework  
**Location**: `test_containers/inspect_evals/`

### System Requirements
- **System Under Test**: `llm_api` (required) - The LLM system being evaluated

### Input Parameters
- `evaluation` (string, required): Name of the Inspect Evals task to run
  - **Cybersecurity**: `cyse3_visual_prompt_injection`, `threecb`, `cybermetric_80/500/2000/10000`, `cyse2_*`, `sevenllm_*`, `sec_qa_*`, `gdm_intercode_ctf`
  - **Safeguards**: `abstention_bench`, `agentdojo`, `agentharm`, `lab_bench_*`, `mask`, `make_me_pay`, `stereoset`, `strong_reject`, `wmdp_*`
  - **Mathematics**: `aime2024`, `gsm8k`, `math`, `mgsm`, `mathvista`
  - **Reasoning**: `arc_challenge`, `arc_easy`, `bbh`, `bbeh`, `boolq`, `drop`, `hellaswag`, `ifeval`, `lingoly`, `mmmu_*`, `musr`, `niah`, `paws`, `piqa`, `race_h`, `winogrande`, `worldsense`, `infinite_bench_*`
  - **Knowledge**: `agie_*`, `air_bench`, `chembench`, `commonsense_qa`, `gpqa_diamond`, `healthbench`, `hle`, `livebench`, `mmlu_*`, `mmlu_pro`, `medqa`, `onet_m6`, `pre_flight`, `pubmedqa`, `sosbench`, `sciknoweval`, `simpleqa`, `truthfulqa`, `xstest`
  - **Scheming**: `agentic_misalignment`, `gdm_*`
  - **Multimodal**: `zerobench`, `zerobench_subquestions`
  - **Bias**: `bbq`, `bold`
  - **Personality**: `personality_BFI`, `personality_TRAIT`
  - **Writing**: `writingbench`
- `limit` (integer, optional): Maximum number of samples to evaluate (default: 10)
- `evaluation_params` (object, optional): Task-specific parameter map passed to the underlying evaluation function
  - **How to specify**: Provide as a JSON object, e.g., `{"fewshot": 5}` or `{"subjects": ["anatomy","astronomy"], "cot": true}`
  - **Available parameters**: Vary by task - see detailed documentation at https://ukgovernmentbeis.github.io/inspect_evals

### Output Metrics
- `success` (boolean): Whether test execution completed successfully
- `evaluation` (string): The evaluation task that was run
- `evaluation_params` (object): Parameters used for the evaluation
- `total_samples` (integer): Total number of samples evaluated
- `metrics` (object): Task-specific evaluation metrics and scores
- `log_dir` (string): Path to stored evaluation logs (when output volume configured)

### Example Configuration
```yaml
test_suite:
  - name: "mathematics_evaluation"
    description: "Check the Models Ability to Solve Math Problems"
    image: "my-registry/inspect_evals:latest"
    systems_under_test: ["math_tutor_model"]
    params:
      evaluation: "gsm8k"
      limit: 50
      evaluation_params:
        fewshot: 5
        fewshot_seed: 42

  - name: "cybersecurity_assessment"
    description: "Check the Models Ability to Handle Cybersecurity Problems"
    image: "my-registry/inspect_evals:latest"
    systems_under_test: ["secure_assistant"]
    params:
      evaluation: "cyse2_prompt_injection"
      limit: 100

  - name: "knowledge_benchmark"
    description: "Measures the Model Knowledge"
    image: "my-registry/inspect_evals:latest"
    systems_under_test: ["knowledge_bot"]
    params:
      evaluation: "mmlu_5_shot"
      limit: 200
      evaluation_params:
        subjects: ["anatomy", "astronomy", "business_ethics"]
        cot: true

  - name: "bias_detection"
    description: "Evaluates Bias in the Chatbot Response"
    image: "my-registry/inspect_evals:latest"
    systems_under_test: ["chatbot"]
    params:
      evaluation: "bbq"
      limit: 150
      evaluation_params:
        use_fast_sampling: true
```

### Build Instructions
```bash
cd test_containers/inspect_evals
docker build -t my-registry/inspect_evals:latest .
```

### Environment Requirements
```bash

# For gated datasets (required by specific evaluations)
export HF_TOKEN="your-huggingface-token"
```

**Gated Dataset Requirements**: Some evaluations require access to gated HuggingFace datasets and need a valid `HF_TOKEN`:

- **GAIA Benchmarks**: `gaia`, `gaia_level1`, `gaia_level2`, `gaia_level3`
  - Requires access to: `gaia-benchmark/GAIA`
- **Abstention Bench**: `abstention_bench` 
  - Requires access to: `Idavidrein/gpqa`
- **MASK**: `mask`
  - Requires access to: `cais/MASK`
- **Lingoly**: `lingoly`
  - Requires access to: `ambean/lingOly`  
- **HLE**: `hle`
  - Requires access to: `cais/hle`
- **XSTest**: `xstest`
  - Requires access to: `walledai/XSTest`
- **TRAIT Personality**: `personality_TRAIT`
  - Requires access to: `mirlab/TRAIT`

To use these evaluations, you must:
1. Request access to the respective gated datasets on HuggingFace
2. Set your HuggingFace token: `export HF_TOKEN="hf_your_token_here"`

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
    description: "Test How Ethical the Responses Are"
    image: "my-registry/trustllm:latest"
    systems_under_test: ["target_model"]
    params:
      test_type: "ethics"
      datasets: ["awareness", "explicit_moralchoice"]
      max_new_tokens: 512
      max_rows: 50

  - name: "safety_assessment"
    description: "Check if the Model Avoids Unsafe Content"
    image: "my-registry/trustllm:latest"
    systems_under_test: ["target_model"]
    params:
      test_type: "safety"
      datasets: ["jailbreak", "misuse"]
      max_rows: 30

  - name: "fairness_evaluation"
    description: "Assesses Fairness Across Different Sets"
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
description: "Evaluate Model Security, Reliability and Trustworthiness"
test_suite:
  # Fast baseline security scan
  - name: "baseline_security"
    description: "Scan for Common Vulnerabilities"
    image: "my-registry/garak:latest"
    systems_under_test: ["target_model"]
    params:
      probes: ["promptinject", "encoding.InjectBase64", "dan.DAN_Jailbreak"]
      generations: 10
      parallel_attempts: 8

  # Comprehensive adversarial testing
  - name: "advanced_red_team"
    description: "Runs Advanced Tests to Expose Weaknesses"
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

  # Cybersecurity benchmark evaluation
  - name: "cybersecurity_benchmark"
    description: "Benchmarks Security Performance"
    image: "my-registry/inspect_evals:latest"
    systems_under_test: ["target_model"]
    params:
      evaluation: "cyse2_prompt_injection"
      limit: 100

  # Trustworthiness evaluation
  - name: "trustworthiness_assessment"
    description: "Check Model Trustworthiness"
    image: "my-registry/trustllm:latest"
    systems_under_test: ["target_model"]
    params:
      evaluation_dimensions: ["truthfulness", "safety", "fairness"]
```

### Quality and Performance Testing

Evaluate conversational quality and system performance:

```yaml
suite_name: "Chatbot Quality and Performance"
description: "Evaluates how well the Chatbot does across Quality and Performance"
test_suite:
  # Conversation quality assessment
  - name: "conversation_quality"
    description: "Checks how Naturally the Chatbot Handles Conversation"
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

  # Knowledge and reasoning assessment
  - name: "knowledge_evaluation"
    description: "Measures how well the Chatbot Reasons"
    image: "my-registry/inspect_evals:latest"
    systems_under_test: ["customer_bot"]
    params:
      evaluation: "mmlu_5_shot"
      limit: 100
      evaluation_params:
        subjects: ["business_ethics", "professional_psychology"]
        cot: true

  # Performance and reliability
  - name: "performance_baseline"
    description: "Measures the Chatbot Response Speed"
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
- **Inspect Evals**: Cybersecurity benchmarks and safety evaluations
- **Combined**: Use multiple containers for complete security coverage

**For Knowledge and Reasoning**:
- **Inspect Evals**: 100+ academic benchmarks across multiple domains
- **TrustLLM**: Specialized trustworthiness evaluation

**For Conversational Quality**:
- **Chatbot Simulator**: Multi-turn dialogue testing with persona-based evaluation
- **Inspect Evals**: Bias and personality assessments

**For Development and Validation**:
- **Mock Tester**: Quick compatibility and configuration validation

**For Research and Benchmarking**:
- **Inspect Evals**: Industry-standard evaluation suite with 100+ tasks
- **TrustLLM**: Specialized trustworthiness benchmarks
- **DeepTeam**: Research-grade adversarial evaluation

### Performance Considerations

**Container Resource Requirements**:
- **Mock Tester**: Minimal resources, fast execution
- **Garak**: Medium resources, depends on probe selection and generations
- **Inspect Evals**: Medium resources, varies by evaluation task and sample limit
- **Chatbot Simulator**: Medium-high resources, depends on conversation complexity
- **DeepTeam**: High resources, requires multiple LLM API calls
- **TrustLLM**: High resources, comprehensive benchmark evaluation

**Optimization Tips**:
- Start with smaller `generations`, `num_scenarios`, and `limit` for development
- Use `parallel_attempts` and `max_concurrent` to balance speed vs. resource usage
- Test with Mock Tester first to validate configuration before expensive tests
- For Inspect Evals, start with `limit: 10` and increase gradually
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

**Inspect Evals**:
```bash
# Requires API key for target system
export OPENAI_API_KEY="sk-your-key"        # For OpenAI systems
export ANTHROPIC_API_KEY="sk-ant-your-key" # For Anthropic systems
# For gated datasets (optional, only needed for specific evaluations)
export HF_TOKEN="hf_your_token_here"       # Required for GAIA, MASK, HLE, XSTest, etc.
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