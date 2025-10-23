# Creating Custom Test Containers

This guide walks you through creating your own test containers for ASQI Engineer, enabling you to implement domain-specific testing frameworks and evaluation logic.

## Container Structure Requirements

Every test container must follow this standardized structure:

```
test_containers/my_custom_tester/
├── Dockerfile                    # Container build instructions
├── entrypoint.py                # Main test execution script
├── manifest.yaml               # Container capabilities declaration
└── requirements.txt            # Python dependencies (optional)
```

## Step-by-Step Development Guide

### 1. Create Container Directory

```bash
mkdir test_containers/my_custom_tester
cd test_containers/my_custom_tester
```

### 2. Define Container Capabilities

Create `manifest.yaml` to declare what your container can do:

```yaml
name: "my_custom_tester"
version: "1.0.0"
description: "Custom testing framework for specific AI system evaluation"
aliases:
  - "registry.example.com/asqi/my_custom_tester:1.0.0"
  - "my-custom-tester:stable"

input_systems:
  - name: "system_under_test"
    type: "llm_api"
    required: true
    description: "The primary system being tested"
  - name: "evaluator_system"
    type: "llm_api"
    required: false
    description: "Optional system for result evaluation"

input_schema:
  - name: "test_iterations"
    type: "integer"
    required: false
    description: "Number of test iterations to run"
  - name: "evaluation_criteria"
    type: "list"
    required: true
    description: "List of evaluation criteria to assess"
  - name: "difficulty_level"
    type: "string"
    required: false
    description: "Test difficulty: easy, medium, hard"

output_metrics:
  - name: "success"
    type: "boolean"
    description: "Whether the test completed successfully"
  - name: "score"
    type: "float"
    description: "Overall test score (0.0 to 1.0)"
  - name: "test_count"
    type: "integer"
    description: "Number of individual tests executed"
  - name: "detailed_results"
    type: "object"
    description: "Comprehensive test results and analysis"
```

`aliases` is optional but recommended when your container is published to remote registries or must respond to multiple image references. The validator uses these aliases to resolve typos (for example, `garak` vs `garak-ai`), multi-level registries (`registry.example.com/org/team/image`), and tag variations without duplicating manifests.

### 3. Implement Test Logic

Create `entrypoint.py` with standardized argument handling:

```python
#!/usr/bin/env python3
import argparse
import json
import sys
import logging
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_llm_client(system_params: Dict[str, Any]):
    """Create LLM client from system parameters."""
    try:
        import openai
        return openai.OpenAI(
            base_url=system_params.get("base_url"),
            api_key=system_params.get("api_key")
        )
    except ImportError:
        logger.error("OpenAI package not available")
        raise

def run_custom_tests(sut_client, test_params: Dict[str, Any], evaluator_client=None) -> Dict[str, Any]:
    """
    Implement your custom test logic here.
    
    Args:
        sut_client: OpenAI client for system under test
        test_params: Test parameters from YAML configuration
        evaluator_client: Optional OpenAI client for evaluation
        
    Returns:
        Dictionary with test results
    """
    test_iterations = test_params.get("test_iterations", 10)
    evaluation_criteria = test_params.get("evaluation_criteria", [])
    difficulty_level = test_params.get("difficulty_level", "medium")
    
    results = {
        "success": True,
        "score": 0.0,
        "test_count": 0,
        "detailed_results": {
            "individual_tests": [],
            "criteria_scores": {},
            "difficulty_used": difficulty_level
        }
    }
    
    try:
        # Example test logic
        passed_tests = 0
        
        for i in range(test_iterations):
            # Generate test prompt based on difficulty and criteria
            test_prompt = generate_test_prompt(i, difficulty_level, evaluation_criteria)
            
            # Get response from system under test
            response = sut_client.chat.completions.create(
                model=sut_client.model,  # This should be set from system_params
                messages=[{"role": "user", "content": test_prompt}],
                max_tokens=512
            )
            
            # Evaluate response
            if evaluator_client:
                evaluation = evaluate_with_llm(response.choices[0].message.content, 
                                             test_prompt, evaluator_client)
            else:
                evaluation = evaluate_response(response.choices[0].message.content, 
                                             evaluation_criteria)
            
            # Record individual test result
            test_result = {
                "test_id": i,
                "prompt": test_prompt,
                "response": response.choices[0].message.content,
                "evaluation": evaluation,
                "passed": evaluation["score"] >= 0.7
            }
            
            results["detailed_results"]["individual_tests"].append(test_result)
            
            if test_result["passed"]:
                passed_tests += 1
                
        # Calculate final metrics
        results["test_count"] = test_iterations
        results["score"] = passed_tests / test_iterations if test_iterations > 0 else 0.0
        
        # Calculate scores by criteria
        for criterion in evaluation_criteria:
            criterion_scores = [t["evaluation"].get(criterion, 0.0) 
                              for t in results["detailed_results"]["individual_tests"]]
            results["detailed_results"]["criteria_scores"][criterion] = {
                "average": sum(criterion_scores) / len(criterion_scores) if criterion_scores else 0.0,
                "min": min(criterion_scores) if criterion_scores else 0.0,
                "max": max(criterion_scores) if criterion_scores else 0.0
            }
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        results["success"] = False
        results["error"] = str(e)
        
    return results

def generate_test_prompt(test_id: int, difficulty: str, criteria: List[str]) -> str:
    """Generate test prompts based on difficulty and criteria."""
    base_prompts = {
        "easy": f"Simple test question {test_id}: What is 2+2?",
        "medium": f"Medium test {test_id}: Explain the concept of machine learning in simple terms.",
        "hard": f"Complex test {test_id}: Analyze the ethical implications of AI decision-making in healthcare."
    }
    
    prompt = base_prompts.get(difficulty, base_prompts["medium"])
    
    if criteria:
        prompt += f" Please address these aspects: {', '.join(criteria)}"
        
    return prompt

def evaluate_response(response: str, criteria: List[str]) -> Dict[str, Any]:
    """Implement your custom evaluation logic."""
    # Simple rule-based evaluation example
    evaluation = {
        "score": 0.8,  # Base score
        "length_appropriate": len(response.split()) > 10,
        "contains_keywords": any(criterion.lower() in response.lower() for criterion in criteria)
    }
    
    # Adjust score based on evaluation
    if not evaluation["length_appropriate"]:
        evaluation["score"] -= 0.2
    if not evaluation["contains_keywords"] and criteria:
        evaluation["score"] -= 0.3
        
    evaluation["score"] = max(0.0, min(1.0, evaluation["score"]))
    return evaluation

def evaluate_with_llm(response: str, prompt: str, evaluator_client) -> Dict[str, Any]:
    """Use LLM for evaluation (if evaluator system provided)."""
    evaluation_prompt = f"""
    Evaluate this response on a scale of 0.0 to 1.0:
    
    Original Prompt: {prompt}
    Response: {response}
    
    Consider accuracy, helpfulness, and appropriateness.
    Return only a JSON object with 'score' and 'reasoning' fields.
    """
    
    try:
        eval_response = evaluator_client.chat.completions.create(
            model=evaluator_client.model,
            messages=[{"role": "user", "content": evaluation_prompt}],
            max_tokens=256
        )
        
        # Parse JSON response
        eval_result = json.loads(eval_response.choices[0].message.content)
        return {
            "score": eval_result.get("score", 0.5),
            "reasoning": eval_result.get("reasoning", "No reasoning provided"),
            "evaluated_by_llm": True
        }
    except Exception as e:
        logger.warning(f"LLM evaluation failed: {e}")
        return {"score": 0.5, "error": str(e), "evaluated_by_llm": False}

def main():
    """Main entrypoint following ASQI container interface."""
    parser = argparse.ArgumentParser(description="Custom ASQI test container")
    parser.add_argument("--systems-params", required=True, 
                       help="JSON string with system configurations")
    parser.add_argument("--test-params", required=True,
                       help="JSON string with test parameters")
    args = parser.parse_args()
    
    try:
        # Parse input parameters
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)
        
        # Extract systems
        sut_params = systems_params.get("system_under_test", {})
        evaluator_params = systems_params.get("evaluator_system")
        
        # Create clients
        sut_client = create_llm_client(sut_params)
        sut_client.model = sut_params.get("model")  # Store model for use in requests
        
        evaluator_client = None
        if evaluator_params:
            evaluator_client = create_llm_client(evaluator_params)
            evaluator_client.model = evaluator_params.get("model")
        
        # Run tests
        results = run_custom_tests(sut_client, test_params, evaluator_client)
        
        # Output results as JSON to stdout
        print(json.dumps(results, indent=2))
        
    except Exception as e:
        # Always output JSON, even on error
        error_result = {
            "success": False,
            "error": str(e),
            "score": 0.0,
            "test_count": 0
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
```

### 4. Create Container Dependencies

Create `requirements.txt` with your dependencies:

```txt
openai>=1.0.0
pydantic>=2.0.0
requests>=2.28.0
# Add other dependencies your test logic needs
```

### 5. Build Container

Create `Dockerfile`:

```dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY entrypoint.py .
COPY manifest.yaml .

# Make entrypoint executable
RUN chmod +x entrypoint.py

# Set entrypoint
ENTRYPOINT ["python", "entrypoint.py"]
```

### 6. Build and Test

```bash
# Build container
docker build -t my-registry/my_custom_tester:latest .

# Test manually
docker run --rm my-registry/my_custom_tester:latest \
  --systems-params '{
    "system_under_test": {
      "type": "llm_api",
      "base_url": "http://localhost:4000/v1",
      "model": "gpt-4o-mini",
      "api_key": "sk-1234"
    }
  }' \
  --test-params '{
    "test_iterations": 3,
    "evaluation_criteria": ["accuracy", "helpfulness"],
    "difficulty_level": "medium"
  }'
```
