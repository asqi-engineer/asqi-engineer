# resaro_judge test container

Judges generated answers against gold answers. For each dataset item, the question is asked once, we record the generated answer, compute a judge decision (heuristic or LLM-as-judge), and compute the overall average accuracy.

Inputs (manifest.yaml):
- system_under_test (llm_api): the model used to generate answers
- evaluator_system (llm_api, optional): LLM-as-judge; if omitted, a simple heuristic judge is used
- test_params:
  - test_type: "accuracy"
  - dataset: list of {question: str, answer: str}
  # The judge returns a boolean 'correct' flag per item.

Output:
- success, test_type
- per_question: [{question, gold_answer, generated_answer, judge_reasoning, correct}]
- overall_average_accuracy (mean of 'correct')

Image tag when using build_all.sh (default repo):
- asqiengineer/test-container:resaro_judge-latest