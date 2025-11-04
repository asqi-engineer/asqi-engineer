# Architecture

ASQI Engineer is built around a modular, fault-tolerant architecture that enables durable execution of AI system tests through containerized testing frameworks.

## Core Components

### Main Entry Point (`src/asqi/main.py`)
- **CLI Interface**: Typer-based command structure with four main execution modes
- **Configuration Loading**: Validates and loads YAML configurations for systems, test suites, and score cards
- **Workflow Delegation**: Routes validated configurations to the appropriate workflow system
- **Signal Handling**: Graceful shutdown with container cleanup on termination

### Workflow System (`src/asqi/workflow.py`)
- **DBOS Integration**: Durable, fault-tolerant execution with automatic retry and recovery
- **Concurrent Execution**: Parallel test execution with configurable concurrency limits and queues
- **Test Lifecycle Management**: Complete orchestration from validation to result aggregation
- **Modular Architecture**:
  - `run_test_suite_workflow()` - Test execution only
  - `evaluate_score_cards_workflow()` - Score card evaluation only  
  - `run_end_to_end_workflow()` - Orchestrates both workflows
  - Reusable DBOS steps following DRY principles

### Container Management (`src/asqi/container_manager.py`)
- **Docker Integration**: Seamless Docker API integration for running test containers
- **Manifest Extraction**: Automatic extraction of test capabilities from container images
- **Resource Management**: Resource-limited container execution with configurable timeouts
- **Environment Passing**: Secure environment variable passing for API keys and configuration
- **Lifecycle Management**: Container startup, monitoring, and cleanup

### Configuration System (`src/asqi/schemas.py`, `src/asqi/config.py`)
- **Type Safety**: Pydantic schemas provide compile-time type checking and runtime validation
- **JSON Schema Generation**: Automatic IDE integration with autocompletion and validation
- **Three Configuration Types**:
  - **Systems**: AI systems and their connection parameters
  - **Test Suites**: Collections of tests and their parameters
  - **Score Cards**: Assessment criteria and evaluation rules
- **Validation Pipeline**: Cross-validates compatibility between systems and test containers

### Validation Engine (`src/asqi/validation.py`)
- **Cross-Validation**: Ensures systems, test definitions, and container capabilities are compatible
- **Execution Planning**: Creates optimized execution plans matching tests to available systems
- **Centralized Functions**:
  - `validate_execution_inputs()` - Validates workflow parameters
  - `validate_score_card_inputs()` - Validates score card evaluation inputs
  - `validate_test_execution_inputs()` - Validates individual test execution
  - `validate_workflow_configurations()` - Comprehensive configuration validation

### Score Card Engine (`src/asqi/score_card_engine.py`)
- **Individual Test Assessment**: Evaluates each test execution separately (no aggregation)
- **Flexible Criteria**: Configurable assessment conditions and thresholds
- **Multiple Outcomes**: Support for complex grading schemes (PASS/FAIL, A/B/C, custom outcomes)
- **Metric Extraction**: Works with any field from test container JSON output

## Key Concepts

### Systems
Defined in `config/systems/` - these represent the AI systems used in testing:
- **Target Systems**: The systems being evaluated
- **Simulator Systems**: Models used to generate test scenarios
- **Evaluator Systems**: Models used to assess results
- **APIs and Services**: Backend services and endpoints

### Test Suites
Defined in `config/suites/` - collections of tests to run against systems:
- **Test Definitions**: Specify container images and parameters
- **System Mapping**: Map tests to compatible systems
- **Parameter Passing**: Configure test-specific parameters

### Test Containers
Docker images in `test_containers/` with embedded capabilities metadata:
- **Standardized Interface**: All containers accept `--systems-params` and `--test-params`
- **JSON Output**: Structured results output to stdout
- **Manifest Declaration**: `manifest.yaml` describes capabilities and schemas

### Score Cards
Assessment criteria defined in `config/score_cards/`:
- **Indicators**: Individual assessment criteria
- **Conditions**: Flexible evaluation logic (equal_to, greater_than, etc.)
- **Outcomes**: Custom result categories (PASS/FAIL, grades, etc.)

## Execution Modes

ASQI provides four distinct execution modes for different use cases:

### 1. Validation Mode (`asqi validate`)
- Validates configuration compatibility without execution
- No Docker containers required
- Fast feedback for configuration development
- Cross-validates systems, suites, and test container capabilities

### 2. Test Execution Mode (`asqi execute-tests`)
- Runs test containers against systems without score card evaluation
- Produces test results JSON for later analysis
- Faster execution for iterative development and debugging
- Maintains DBOS durability for fault tolerance

### 3. Score Card Evaluation Mode (`asqi evaluate-score-cards`)
- Evaluates existing test results against score card criteria
- No test execution - works with previously saved results
- Enables post-hoc analysis and different evaluation criteria
- Apply multiple score cards to the same test results

### 4. End-to-End Mode (`asqi execute`)
- Combines test execution and score card evaluation in one workflow
- Complete evaluation pipeline with unified durability
- Traditional single-command experience
- Maintains workflow correlation across both phases

## Multi-System Test Architecture

ASQI supports complex testing scenarios that require multiple AI systems working together:

### System Orchestration
Tests can coordinate multiple systems for sophisticated evaluation:
- **System Under Test**: The primary system being evaluated
- **Simulator System**: Generates test scenarios, user queries, or adversarial inputs
- **Evaluator System**: Assesses results, provides scoring, or validates outputs

### Configuration Flexibility
```yaml
test_suite:
  - id: "chatbot_simulation"
    name: "chatbot simulation"
    image: "my-registry/chatbot_simulator:latest" 
    systems_under_test: ["my_chatbot"]
    systems:
      simulator_system: "gpt4o_simulator"
      evaluator_system: "claude_evaluator"
```

### Container Interface
Test containers receive a unified `systems_params` structure:
```python
systems_params = {
    "system_under_test": {...},
    "simulator_system": {...},
    "evaluator_system": {...}
}
```

## Environment Variable System

### Three-Level Configuration Priority
1. **Explicit System Parameters** (highest priority)
2. **Environment File Fallbacks** (`.env` file or custom `env_file`)
3. **Validation Error** (if required fields missing)

### LLM Provider Support
- **Universal API Interface**: All LLM systems use OpenAI-compatible format
- **LiteLLM Integration**: Support for 100+ providers through unified proxy
- **Flexible Configuration**: Direct provider APIs or unified proxy endpoints
- **Environment Fallbacks**: Global configuration through `.env` files

## Data Flow

### Test Execution Flow
1. **Configuration Loading**: Load and validate YAML configurations
2. **Manifest Extraction**: Extract test capabilities from Docker images
3. **Cross-Validation**: Ensure systems match test container requirements
4. **DBOS Workflow**: Execute tests concurrently with durability guarantees
5. **Result Aggregation**: Collect and structure test results in JSON format

### Score Card Evaluation Flow
1. **Result Loading**: Load existing test results from JSON file
2. **Score Card Validation**: Validate score card configuration schemas
3. **Assessment Execution**: Apply evaluation criteria to individual test results
4. **Outcome Generation**: Produce structured assessment outcomes
5. **Result Enhancement**: Merge score card results with original test data

