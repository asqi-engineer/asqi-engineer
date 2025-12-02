# Test Containers

ASQI Engineer provides a mock test container to demonstrate how to configure and run a RAG system evaluation. 

## Mock RAG Tester

**Purpose**: Development and validation testing for RAG systems with configurable simulation.

**Framework**: Custom lightweight testing framework  
**Location**: `test_containers/mock_rag_tester/`

### System Requirements
- **System Under Test**: `rag_api` (required) - The RAG system being tested

### Input Parameters
- `delay_seconds` (integer, optional): Seconds to sleep simulating processing work

### Output Metrics
- `success` (boolean): Whether test execution completed successfully
- `score` (float): Mock test score (0.0 to 1.0)  
- `delay_used` (integer): Actual delay in seconds used
- `base_url` (string): API endpoint that was accessed
- `model` (string): Model name that was tested
- `user_group` (string, optional): User group used in the API if provided

### Example Configuration
```yaml
test_suite:
  - id: "rag_basic_compatibility_check"
    name: "RAG basic compatibility check"
    description: "RAG System Basic Compatibility Test"
    image: "my-registry/mock_rag_tester:latest"
    systems_under_test: ["my_rag_service"]
    params:
      delay_seconds: 2
```

### Build Instructions
```bash
cd test_containers/mock_rag_tester
docker build -t my-registry/mock_rag_tester:latest .
```

---