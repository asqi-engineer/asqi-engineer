# Document Q&A Generator

Generate question-answer pairs from PDF or TXT documents for RAG (Retrieval-Augmented Generation) evaluation and training.

## Overview

This container processes technical documentation and automatically generates question-answer datasets suitable for:
- RAG system evaluation
- Training data creation
- Knowledge base testing
- Documentation quality assessment

## Features

- **Document Support**: Process both PDF and TXT files
- **Intelligent Chunking**: Configurable text chunking with overlap for context preservation
- **LLM-Powered Generation**: Uses any OpenAI-compatible API for Q&A generation
- **HuggingFace Output**: Generates datasets in HuggingFace format for easy integration
- **Configurable Parameters**: Control chunk size, overlap, and number of questions

## Input Requirements

### Systems
- **generation_system** (required): LLM API for generating Q&A pairs
  - Type: `llm_api`
  - Used for processing document chunks and generating questions/answers

### Datasets
- **source_documents** (required): Documents to process
  - Type: `pdf` or `txt`
  - Path to documentation file(s)

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `num_questions` | integer | No | 3 | Number of questions to generate per chunk |
| `chunk_size` | integer | No | 1000 | Size of text chunks in characters |
| `chunk_overlap` | integer | No | 200 | Overlap between chunks in characters |
| `output_dataset_name` | string | No | "qa_dataset" | Name for the output dataset |

## Output

### Metrics
- `success`: Whether generation completed successfully
- `questions_generated`: Total number of questions created
- `chunks_processed`: Number of text chunks processed
- `processing_time_seconds`: Total execution time
- `output_dataset_path`: Path to the generated dataset

### Datasets
- **qa_dataset**: HuggingFace dataset with:
  - `question`: Generated question from the document
  - `answer`: Corresponding answer to the question
  - `context`: Source text from document used to generate Q&A

## Usage Example

### 1. Build the Container

```bash
cd test_containers/doc_qa_generator
docker build -t asqiengineer/test-container:doc_qa_generator-latest .
```

### 2. Configure Systems

Create `config/qa_generation/systems.yaml`:

```yaml
systems:
  openai_gpt4o:
    type: llm_api
    description: "OpenAI GPT-4o for Q&A generation"
    provider: openai
    params:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o-mini"
      env_file: ".env"
```

### 3. Configure Datasets

Create `config/qa_generation/datasets.yaml`:

```yaml
datasets:
  sample_documentation:
    type: "txt"
    description: "Sample technical documentation"
    file_path: "test_data/sample_documentation.txt"
```

### 4. Create Generation Suite

Create `config/qa_generation/suite.yaml`:

```yaml
suite_name: "Document Q&A Generation Suite"
description: "Generate question-answer pairs from documentation"

test_suite:
  - id: "doc_qa_gen"
    name: "Generate Q&A from Documentation"
    image: "asqiengineer/test-container:doc_qa_generator-latest"
    systems:
      generation_system: openai_gpt4o
    datasets:
      source_documents: sample_documentation
    volumes:
      input: input/
      output: output/
    params:
      num_questions: 3
      chunk_size: 1000
      chunk_overlap: 200
      output_dataset_name: "docs_qa_v1"
```

### 5. Run Generation

```bash
asqi execute-tests \
  -t config/qa_generation/suite.yaml \
  -s config/qa_generation/systems.yaml \
  -d config/qa_generation/datasets.yaml \
  -o output.json
```

## Example Output

```json
{
  "test_results": {
    "success": true,
    "questions_generated": 45,
    "chunks_processed": 15,
    "processing_time_seconds": 67.3,
    "output_dataset_path": "/output/docs_qa_v1"
  },
  "output_datasets": {
    "qa_dataset": {
      "path": "/output/docs_qa_v1",
      "type": "huggingface",
      "num_rows": 45
    }
  }
}
```

## Sample Generated Q&A

From a RAG documentation, the container might generate:

**Question**: "What are the core components of a RAG system?"
**Answer**: "The core components include the document retriever for fetching relevant information, the language model for processing and generation, and the context manager for integrating retrieved documents."
**Context**: "Retrieval-Augmented Generation (RAG) is an AI framework that combines... [excerpt from document]"

## Use Cases

### RAG System Evaluation
Generate test datasets from your documentation to evaluate RAG retrieval accuracy and answer quality.

### Training Data Creation
Create domain-specific Q&A pairs from technical manuals, guides, and documentation.

### Knowledge Base Testing
Validate that Q&A systems correctly understand and retrieve information from documentation.

### Documentation Quality
Assess documentation completeness by examining what questions can be generated and answered.

## Technical Details

### Document Processing
1. **Extraction**: Text is extracted from PDF (using PyPDF2) or TXT files
2. **Chunking**: Documents are split into overlapping chunks for context preservation
3. **Generation**: Each chunk is processed by the LLM to generate Q&A pairs
4. **Dataset Creation**: Results are compiled into a HuggingFace dataset

### Chunking Strategy
- Default chunk size: 1000 characters
- Default overlap: 200 characters
- Overlap ensures questions at chunk boundaries have full context
- Empty chunks are filtered out

### LLM Prompting
The container uses a structured prompt that:
- Instructs the LLM to generate multiple Q&A pairs
- Ensures answers are grounded in the provided text
- Returns JSON-formatted output for reliable parsing
- Maintains consistency across chunks

## Best Practices

1. **Chunk Size**: Adjust based on document structure
   - Technical docs: 800-1200 characters
   - Narrative content: 1500-2000 characters

2. **Overlap**: Maintain sufficient overlap
   - Minimum: 15-20% of chunk size
   - Recommended: 200-300 characters

3. **Questions per Chunk**: Balance coverage and quality
   - Short chunks: 2-3 questions
   - Long chunks: 4-5 questions

4. **Model Selection**: Use capable models
   - GPT-4o-mini: Good balance of cost and quality
   - GPT-4: Higher quality but more expensive
   - Claude 3: Excellent for technical content

## Troubleshooting

### Low Quality Questions
- Increase chunk size for more context
- Use a more capable LLM model
- Reduce questions per chunk for better focus

### Missing Context
- Increase chunk overlap
- Adjust chunk boundaries to respect document structure
- Review source document formatting

### Parsing Errors
- Check LLM output format in logs
- Verify model supports structured output
- Adjust temperature for more consistent responses

## Dependencies

- Python 3.12+
- openai >= 1.0.0
- PyPDF2 >= 3.0.0
- datasets >= 2.14.0

## License

Apache 2.0 - See repository LICENSE file
