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

## Advanced Container Patterns

### Multi-System Coordination

Implement containers that coordinate multiple systems:

```python
def run_multi_system_test(systems_params: Dict[str, Any], test_params: Dict[str, Any]):
    """Example of coordinating multiple systems."""
    
    # Extract all systems
    sut_params = systems_params.get("system_under_test", {})
    simulator_params = systems_params.get("simulator_system", sut_params)
    evaluator_params = systems_params.get("evaluator_system", sut_params)
    
    # Create clients
    sut_client = create_llm_client(sut_params)
    simulator_client = create_llm_client(simulator_params)
    evaluator_client = create_llm_client(evaluator_params)
    
    results = []
    
    # 1. Generate test scenarios with simulator
    scenarios = generate_scenarios_with_llm(simulator_client, test_params)
    
    # 2. Run scenarios against system under test
    for scenario in scenarios:
        response = sut_client.chat.completions.create(
            model=sut_params.get("model"),
            messages=scenario["messages"]
        )
        
        # 3. Evaluate with evaluator system
        evaluation = evaluate_response_with_llm(
            evaluator_client,
            scenario["expected_outcome"],
            response.choices[0].message.content
        )
        
        results.append({
            "scenario": scenario,
            "response": response.choices[0].message.content,
            "evaluation": evaluation
        })
    
    return aggregate_results(results)
```

### Async/Concurrent Testing

Implement high-performance concurrent testing:

```python
import asyncio
import aiohttp
from typing import List

async def run_concurrent_tests(sut_params: Dict[str, Any], test_params: Dict[str, Any]):
    """Run tests concurrently for improved performance."""
    max_concurrent = test_params.get("max_concurrent", 5)
    test_cases = generate_test_cases(test_params)
    
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_single_test(test_case):
        async with semaphore:
            return await execute_test_case(sut_params, test_case)
    
    # Run all tests concurrently
    tasks = [run_single_test(case) for case in test_cases]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    return aggregate_async_results(results)

async def execute_test_case(sut_params: Dict[str, Any], test_case: Dict[str, Any]):
    """Execute individual test case asynchronously."""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": sut_params.get("model"),
            "messages": test_case["messages"],
            "max_tokens": 512
        }
        
        headers = {
            "Authorization": f"Bearer {sut_params.get('api_key')}",
            "Content-Type": "application/json"
        }
        
        async with session.post(
            f"{sut_params.get('base_url')}/chat/completions",
            json=payload,
            headers=headers
        ) as response:
            result = await response.json()
            return evaluate_test_result(test_case, result)
```

### Domain-Specific Evaluation

Implement specialized evaluation logic for your domain:

```python
def evaluate_financial_advice(response: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Custom evaluation for financial advice systems."""
    
    # Rule-based checks
    compliance_score = check_regulatory_compliance(response)
    accuracy_score = verify_factual_accuracy(response, context)
    risk_score = assess_risk_appropriateness(response, context["user_profile"])
    
    # Combine scores
    overall_score = (compliance_score + accuracy_score + risk_score) / 3
    
    return {
        "overall_score": overall_score,
        "compliance_score": compliance_score,
        "accuracy_score": accuracy_score,
        "risk_assessment_score": risk_score,
        "regulatory_compliant": compliance_score >= 0.8,
        "factually_accurate": accuracy_score >= 0.8,
        "risk_appropriate": risk_score >= 0.7
    }

def check_regulatory_compliance(response: str) -> float:
    """Check if response follows financial regulations."""
    compliance_keywords = ["not financial advice", "consult professional", "past performance"]
    risk_keywords = ["guaranteed returns", "no risk", "insider information"]
    
    compliance_mentions = sum(1 for keyword in compliance_keywords if keyword in response.lower())
    risk_violations = sum(1 for keyword in risk_keywords if keyword in response.lower())
    
    # Score based on compliance mentions and risk violations
    score = min(1.0, compliance_mentions * 0.3) - (risk_violations * 0.5)
    return max(0.0, score)
```

### Error Handling Best Practices

Always handle errors gracefully and output valid JSON:

```python
def safe_test_execution(test_function, *args, **kwargs):
    """Wrapper for safe test execution with error handling."""
    try:
        return test_function(*args, **kwargs)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "score": 0.0,
            "test_count": 0
        }

def main():
    """Main entrypoint with comprehensive error handling."""
    try:
        # Parse arguments
        args = parse_arguments()
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)
        
        # Validate required parameters
        validate_input_parameters(systems_params, test_params)
        
        # Run tests with error handling
        results = safe_test_execution(run_custom_tests, systems_params, test_params)
        
    except json.JSONDecodeError as e:
        results = {
            "success": False,
            "error": f"Invalid JSON in parameters: {e}",
            "score": 0.0
        }
    except ValueError as e:
        results = {
            "success": False,
            "error": f"Parameter validation failed: {e}",
            "score": 0.0
        }
    except Exception as e:
        results = {
            "success": False,
            "error": f"Unexpected error: {e}",
            "score": 0.0
        }
    
    # Always output JSON
    print(json.dumps(results, indent=2))
```

## Testing Your Container

### Manual Testing

Test your container before integrating with ASQI:

```bash
# 1. Build container
docker build -t my-registry/my_custom_tester:latest .

# 2. Test with minimal parameters
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
    "test_iterations": 2,
    "evaluation_criteria": ["accuracy"],
    "difficulty_level": "easy"
  }'

# 3. Test with multi-system configuration
docker run --rm my-registry/my_custom_tester:latest \
  --systems-params '{
    "system_under_test": {
      "type": "llm_api",
      "base_url": "http://localhost:4000/v1", 
      "model": "gpt-4o-mini",
      "api_key": "sk-1234"
    },
    "evaluator_system": {
      "type": "llm_api",
      "base_url": "http://localhost:4000/v1",
      "model": "claude-3-5-sonnet",
      "api_key": "sk-1234"
    }
  }' \
  --test-params '{
    "test_iterations": 3,
    "evaluation_criteria": ["accuracy", "helpfulness", "safety"],
    "difficulty_level": "medium"
  }'
```

### ASQI Integration Testing

Create test configurations for your container:

**Test Suite** (`config/suites/custom_test_suite.yaml`):
```yaml
suite_name: "Custom Container Testing"
test_suite:
  - name: "custom_evaluation_basic"
    image: "my-registry/my_custom_tester:latest"
    systems_under_test: ["my_llm_service"]
    params:
      test_iterations: 5
      evaluation_criteria: ["accuracy", "helpfulness"]
      difficulty_level: "easy"

  - name: "custom_evaluation_advanced"
    image: "my-registry/my_custom_tester:latest"
    systems_under_test: ["my_llm_service"]
    systems:
      evaluator_system: "claude_evaluator"
    params:
      test_iterations: 10
      evaluation_criteria: ["accuracy", "helpfulness", "safety", "compliance"]
      difficulty_level: "hard"
```

**Validate with ASQI**:
```bash
asqi validate \
  --test-suite-config config/suites/custom_test_suite.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --manifests-dir test_containers/
```

**Run with ASQI**:
```bash
asqi execute-tests \
  --test-suite-config config/suites/custom_test_suite.yaml \
  --systems-config config/systems/demo_systems.yaml \
  --output-file custom_test_results.json
```

## Container Development Best Practices

### Performance Optimization

**Efficient API Usage**:
- Batch API requests when possible
- Use appropriate `max_tokens` limits
- Implement request retry logic with exponential backoff
- Monitor and respect rate limits

**Resource Management**:
- Use streaming for large responses
- Clean up temporary files and resources
- Implement proper timeout handling
- Monitor memory usage for large datasets

### Security Considerations

**API Key Handling**:
- Never log API keys or sensitive parameters
- Use environment variables securely
- Validate API key format before use
- Handle authentication errors gracefully

**Input Validation**:
- Sanitize all input parameters
- Validate JSON schema compliance
- Check for injection attacks in test parameters
- Implement safe string handling

### Logging and Debugging

**Structured Logging**:
```python
import logging
import json

logger = logging.getLogger(__name__)

def log_test_execution(test_id: int, success: bool, score: float):
    """Log test execution in structured format."""
    log_data = {
        "test_id": test_id,
        "success": success,
        "score": score,
        "timestamp": datetime.utcnow().isoformat()
    }
    logger.info(f"Test execution: {json.dumps(log_data)}")
```

**Debug Mode Support**:
```python
def main():
    # Check for debug environment variable
    debug_mode = os.getenv("ASQI_DEBUG", "false").lower() == "true"
    if debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled")
```

## Integration with Score Cards

Design your output metrics to work well with ASQI score cards:

### Score Card Compatible Output

```python
# Structure output for easy score card evaluation
results = {
    "success": True,
    "overall_score": 0.85,
    "test_count": 100,
    
    # Individual metric scores for fine-grained assessment
    "accuracy_score": 0.90,
    "helpfulness_score": 0.85,
    "safety_score": 0.80,
    
    # Business-relevant metrics
    "production_ready": True,
    "requires_review": False,
    "critical_issues_found": 0,
    
    # Detailed breakdown for analysis
    "detailed_metrics": {
        "passed_tests": 85,
        "failed_tests": 15,
        "error_tests": 0,
        "average_response_time": 1.2,
        "criteria_breakdown": {
            "accuracy": {"passed": 90, "total": 100},
            "helpfulness": {"passed": 85, "total": 100},
            "safety": {"passed": 80, "total": 100}
        }
    }
}
```

### Corresponding Score Card

```yaml
score_card_name: "Custom Container Assessment"
indicators:
  - name: "Overall Quality Gate"
    apply_to:
      test_name: "custom_evaluation_advanced"
    metric: "overall_score"
    assessment:
      - { outcome: "EXCELLENT", condition: "greater_equal", threshold: 0.9 }
      - { outcome: "GOOD", condition: "greater_equal", threshold: 0.8 }
      - { outcome: "NEEDS_IMPROVEMENT", condition: "less_than", threshold: 0.8 }

  - name: "Production Readiness"
    apply_to:
      test_name: "custom_evaluation_advanced"
    metric: "production_ready"
    assessment:
      - { outcome: "DEPLOY", condition: "equal_to", threshold: true }
      - { outcome: "BLOCK", condition: "equal_to", threshold: false }

  - name: "Safety Requirements"
    apply_to:
      test_name: "custom_evaluation_advanced"
    metric: "safety_score"
    assessment:
      - { outcome: "SAFE", condition: "greater_equal", threshold: 0.8 }
      - { outcome: "REVIEW_REQUIRED", condition: "less_than", threshold: 0.8 }
```

## Container Registry and Distribution

### Container Versioning

Use semantic versioning for your containers:

```bash
# Tag with version
docker tag my-registry/my_custom_tester:latest my-registry/my_custom_tester:v1.0.0

# Push to registry
docker push my-registry/my_custom_tester:v1.0.0
docker push my-registry/my_custom_tester:latest
```

### Manifest Versioning

Keep manifest versions synchronized with container versions:

```yaml
name: "my_custom_tester"
version: "1.0.0"  # Match container version
description: "Custom testing framework - initial release"
```

### Distribution Best Practices

**Registry Organization**:
```bash
# Organize by purpose
my-registry/security/garak:v2.1.0
my-registry/quality/chatbot_simulator:v1.5.0
my-registry/custom/financial_tester:v1.0.0

# Or by team/organization
my-registry/asqi-official/mock_tester:latest
my-registry/myteam/domain_tester:v2.0.0
```

**Documentation and Changelog**:
- Include detailed README in container directory
- Maintain changelog for manifest and container updates
- Document breaking changes between versions
- Provide migration guides for configuration updates

This comprehensive approach ensures your custom test containers integrate seamlessly with ASQI's ecosystem while providing powerful, domain-specific testing capabilities.