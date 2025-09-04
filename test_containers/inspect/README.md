# Inspect Test Container

This test container integrates [Inspect](https://inspect.aisi.org.uk/) evaluation framework with ASQI Engineer, providing a generic interface for running various LLM evaluations.

## Supported Evaluations

The container currently supports the following evaluations from inspect-evals:

### Basic Evaluations
- **xstest**: XSTest benchmark for identifying exaggerated safety behaviors
- **truthfulqa**: TruthfulQA benchmark for measuring truthfulness
- **boolq**: BoolQ benchmark for boolean question answering

### MMLU Variants
- **mmlu**: MMLU benchmark for massive multitask language understanding (0-shot)
- **mmlu_0_shot**: MMLU benchmark (0-shot)
- **mmlu_5_shot**: MMLU benchmark (5-shot)
- **mmlu_pro**: MMLU Pro benchmark

### Code Evaluations
- **gsm8k**: GSM8K benchmark for math reasoning
- **ifeval**: IFEval benchmark for instruction following

### Commonsense Reasoning
- **hellaswag**: HellaSwag benchmark for commonsense reasoning
- **winogrande**: Winogrande benchmark for commonsense reasoning
- **piqa**: PIQA benchmark for physical interaction QA
- **commonsense_qa**: CommonsenseQA benchmark

### Science and Knowledge
- **arc**: ARC Challenge benchmark
- **arc_easy**: ARC Easy benchmark
- **arc_challenge**: ARC Challenge benchmark
- **medqa**: MedQA benchmark for medical question answering
- **pubmedqa**: PubMedQA benchmark for biomedical literature
- **chembench**: ChemBench benchmark for chemistry
- **healthbench**: HealthBench benchmark for healthcare
- **gpqa_diamond**: GPQA Diamond benchmark for graduate-level questions

### Mathematics
- **math**: MATH benchmark for mathematical problem solving
- **mathvista**: MathVista benchmark for mathematical visual reasoning
- **aime2024**: AIME 2024 benchmark
- **mgsm**: MGSM benchmark for multilingual grade school math

### Reading Comprehension
- **drop**: DROP benchmark for discrete reasoning
- **race_h**: RACE-H benchmark for reading comprehension

### Safety and Alignment
- **bbq**: BBQ benchmark for bias evaluation
- **strong_reject**: Strong Reject benchmark
- **sycophancy**: Sycophancy benchmark

### Multilingual
- **paws**: PAWS benchmark for paraphrase adversaries

### Long Context and Reasoning
- **infinite_bench_math_calc**: Infinite Bench mathematical calculation
- **infinite_bench_passkey**: Infinite Bench passkey retrieval

## Planned Evaluations (Coming Soon)

The following evaluations are planned but not yet fully supported:

### Code Evaluations
- **humaneval**: HumanEval benchmark for code generation (requires Docker-in-Docker)
- **mbpp**: MBPP benchmark for basic Python programming (requires Docker-in-Docker)
- **ds1000**: DS1000 benchmark for data science tasks (requires Docker-in-Docker)
- **class_eval**: ClassEval benchmark for object-oriented programming (requires Docker-in-Docker)

### Agent and Tool Use
- **agent_bench_os**: Agent Bench OS benchmark (requires Docker-in-Docker)

### Specialized Domains
- **bold**: BOLD benchmark for bias evaluation (requires additional PyTorch dependencies)
- **swe_bench**: SWE-bench for software engineering (requires additional dependencies)
- **cybench**: CyBench for cybersecurity evaluation

### Multi-modal
- **mmiu**: MMIU benchmark for multi-modal understanding
- **mmmu_multiple_choice**: MMMU multiple choice benchmark
- **mmmu_open**: MMMU open-ended benchmark

### Long Context
- **infinite_bench_code_debug**: Infinite Bench code debugging tasks

### Science
- **sciq**: SciQ benchmark for science questions
- **lingoly**: Lingoly benchmark for linguistic reasoning

## Usage

The inspect test container can be used in ASQI test suites like any other test container:

```yaml
test_suite:
  - name: "xstest_safe_evaluation"
    image: "asqiengineer/test-container:inspect-latest"
    params:
      evaluation: "xstest"
      evaluation_params:
        subset: "safe"
        scorer_model: "openai/gpt-4o"
      limit: 10
```

## Parameters

- **evaluation**: String - the evaluation to run (default: "xstest")
- **evaluation_params**: Object - parameters specific to the chosen evaluation
- **limit**: Integer - maximum number of samples to evaluate (default: 10)

## Evaluation-Specific Parameters

### XSTest
- **subset**: `"safe"` or `"unsafe"` - which subset of prompts to test
- **scorer_model**: String - model to use for grading responses (default: "openai/gpt-4o")

### TruthfulQA
- **target**: `"mc1"` or `"mc2"` - whether to use single-answer or multi-answer targets

### MMLU (all variants)
- **language**: String - language code (default: "EN_US")
- **subjects**: String - comma-separated list of subjects to test (optional)

### GSM8K
- No additional parameters required (uses default configuration)

### IFEval
- **template**: String - template for instruction following (optional)

### Safety Evaluations (BBQ, Strong Reject, Sycophancy)
- **scorer_model**: String - model to use for grading responses
- **template**: String - custom template for evaluation (optional)

### Mathematics Evaluations (MATH, MathVista, AIME2024, MGSM)
- **scorer_model**: String - model to use for grading mathematical responses
- **subjects**: String - specific math subjects to test (where applicable)

### Science Evaluations (ARC, MedQA, PubMedQA, ChemBench, HealthBench, GPQA)
- **scorer_model**: String - model to use for grading responses
- **domain**: String - specific domain or subject area (where applicable)

### Other Evaluations
Most evaluations accept standard parameters like `scorer_model`, `temperature`, etc. Check the inspect-evals documentation for specific evaluation parameters.

## Output Metrics

- **success**: Boolean - whether the evaluation completed successfully
- **evaluation**: String - the evaluation that was run
- **evaluation_params**: Object - the parameters used
- **total_samples**: Integer - total number of samples evaluated
- **model**: String - the model that was evaluated
- **metrics**: Object - evaluation-specific metrics returned by the Inspect framework (e.g., accuracy, score, etc.)

Note: The specific metrics in the `metrics` object depend on the evaluation type. Common metrics include:
- **accuracy**: Overall accuracy score
- **score**: Aggregate evaluation score
- **mean**: Mean score across samples
- **stderr**: Standard error of the mean

## Example Test Configurations

```yaml
# XSTest safe subset
- name: "xstest_safe"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "xstest"
    evaluation_params:
      subset: "safe"
      scorer_model: "openai/gpt-4o"
    limit: 10

# TruthfulQA
- name: "truthfulqa_mc1"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "truthfulqa"
    evaluation_params:
      target: "mc1"
    limit: 20

# GSM8K Math Reasoning
- name: "gsm8k"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "gsm8k"
    evaluation_params: {}
    limit: 20

# MMLU 5-shot
- name: "mmlu_5_shot"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "mmlu_5_shot"
    evaluation_params:
      subjects: "high_school_mathematics,college_mathematics"
    limit: 50

# HellaSwag Commonsense Reasoning
- name: "hellaswag"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "hellaswag"
    evaluation_params: {}
    limit: 30

# ARC Challenge
- name: "arc_challenge"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "arc_challenge"
    evaluation_params: {}
    limit: 25

# BBQ Bias Evaluation
- name: "bbq_bias"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "bbq"
    evaluation_params:
      scorer_model: "openai/gpt-4o"
    limit: 15

# MATH Problem Solving
- name: "math_evaluation"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "math"
    evaluation_params:
      scorer_model: "openai/gpt-4o"
    limit: 10
```

## Example Test Run

```bash
# Run multiple inspect evaluations
asqi execute-tests \
  --test-suite-config config/suites/inspect_test.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --output-file inspect_results.json
```

## Requirements

- OpenAI API key (for the target model and scorer models)
- The target system must be an `llm_api` type system

## Adding New Evaluations

To add support for new evaluations:

1. Ensure the evaluation is available in inspect-evals
2. Add the evaluation name and import to the `EVALUATION_REGISTRY` dictionary in `entrypoint.py`
3. Update the README with evaluation-specific parameters
4. Add test configurations to the suite files
5. Test the evaluation to ensure it works correctly

The container uses dynamic imports for all evaluations. The registry format is:
```python
"evaluation_name": ("module_path", "function_name")
```

For example:
```python
"new_eval": ("inspect_evals.new_eval", "new_eval_function")
```

Some evaluations may require additional dependencies or special environments (like Docker-in-Docker for code execution evaluations).

## Troubleshooting

- **Import errors**: Ensure the evaluation name matches exactly with the module name in inspect-evals
- **Parameter errors**: Check the inspect-evals documentation for the correct parameter names and types
- **Model compatibility**: Some evaluations may require specific model formats or capabilities
