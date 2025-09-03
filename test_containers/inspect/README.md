# Inspect Test Container

This test container integrates [Inspect](https://inspect.aisi.org.uk/) evaluation framework with ASQI Engineer, providing a generic interface for running various LLM evaluations.

## Supported Evaluations

The container currently supports the following evaluations from inspect-evals:

- **xstest**: XSTest benchmark for identifying exaggerated safety behaviors
- **truthfulqa**: TruthfulQA benchmark for measuring truthfulness
- **gsm8k**: GSM8K benchmark for math reasoning
- **mmlu**: MMLU benchmark for massive multitask language understanding (0-shot and 5-shot)
- **humaneval**: HumanEval benchmark for code generation
- **hellaswag**: HellaSwag benchmark for commonsense reasoning
- **winogrande**: Winogrande benchmark for commonsense reasoning
- **boolq**: BoolQ benchmark for boolean question answering

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

### MMLU
- **language**: String - language code (default: "EN_US")
- **subjects**: String - comma-separated list of subjects to test (optional)

# GSM8K
- No additional parameters required (uses default configuration)

# Other Evaluations
Most evaluations accept standard parameters like `scorer_model`, `temperature`, etc. Check the inspect-evals documentation for specific evaluation parameters.

## Output Metrics

- **success**: Boolean - whether the evaluation completed successfully
- **evaluation**: String - the evaluation that was run
- **evaluation_params**: Object - the parameters used
- **total_samples**: Integer - total number of samples evaluated
- **compliant_count**: Integer - number of fully compliant responses
- **partial_refusal_count**: Integer - number of partial refusal responses
- **full_refusal_count**: Integer - number of full refusal responses
- **metrics**: Object - additional metrics returned by the evaluation

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

# GSM8K
- name: "gsm8k"
  image: "asqiengineer/test-container:inspect-latest"
  params:
    evaluation: "gsm8k"
    evaluation_params: {}
    limit: 20
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
2. Add the evaluation name and import to the entrypoint.py
3. Update the README with evaluation-specific parameters
4. Add test configurations to the suite files

The container uses dynamic imports where possible, but some evaluations may need explicit import statements due to how they're structured in inspect-evals.

## Troubleshooting

- **Import errors**: Ensure the evaluation name matches exactly with the module name in inspect-evals
- **Parameter errors**: Check the inspect-evals documentation for the correct parameter names and types
- **Model compatibility**: Some evaluations may require specific model formats or capabilities
