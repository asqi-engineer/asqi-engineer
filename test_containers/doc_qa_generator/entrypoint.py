#!/usr/bin/env python3
"""
Document Q&A Generator Container

Processes PDF or TXT documents and generates question-answer pairs
for RAG evaluation and training.
"""
import argparse
import json
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from datasets import Dataset
except ImportError:
    Dataset = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
        
    Raises:
        Exception: If PDF reading fails
    """
    if PdfReader is None:
        raise ImportError("PyPDF2 is required for PDF processing")
    
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from PDF: {e}")
        raise


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a TXT file.
    
    Args:
        file_path: Path to the TXT file
        
    Returns:
        File contents as a string
        
    Raises:
        Exception: If file reading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to read text file: {e}")
        raise


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Only add non-empty chunks
        if chunk.strip():
            chunks.append(chunk.strip())
        
        # Move to next chunk with overlap
        start = end - overlap
        
    return chunks


def generate_qa_pairs(
    chunk: str,
    llm_client: Any,
    model: str,
    num_questions: int = 3
) -> List[Tuple[str, str]]:
    """Generate question-answer pairs from a text chunk using an LLM.
    
    Args:
        chunk: Text chunk to generate Q&A from
        llm_client: OpenAI-compatible client
        model: Model name to use
        num_questions: Number of Q&A pairs to generate
        
    Returns:
        List of (question, answer) tuples
    """
    prompt = f"""Based on the following text, generate {num_questions} question-answer pairs.
Each question should be answerable using only the information in the text.
Each answer should be clear, concise, and directly based on the text.

Text:
{chunk}

Please respond with a JSON array where each element has "question" and "answer" fields.
Example format:
[
  {{"question": "What is...", "answer": "..."}},
  {{"question": "How does...", "answer": "..."}}
]

Only return the JSON array, no additional text."""

    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates question-answer pairs from documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        content = response.choices[0].message.content
        
        # Parse the JSON response
        qa_pairs_json = json.loads(content)
        
        # Extract question-answer tuples
        qa_pairs = []
        for item in qa_pairs_json:
            if "question" in item and "answer" in item:
                qa_pairs.append((item["question"], item["answer"]))
        
        return qa_pairs
        
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        logger.debug(f"Response content: {content}")
        return []
    except Exception as e:
        logger.error(f"Error generating Q&A pairs: {e}")
        return []


def process_document(
    file_path: str,
    llm_client: Any,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    num_questions: int
) -> Tuple[List[Dict[str, str]], int]:
    """Process a document and generate Q&A pairs.
    
    Args:
        file_path: Path to the document
        llm_client: OpenAI-compatible client
        model: Model name to use
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        num_questions: Questions per chunk
        
    Returns:
        Tuple of (list of Q&A dictionaries, number of chunks processed)
    """
    # Extract text based on file type
    file_path_lower = file_path.lower()
    if file_path_lower.endswith('.pdf'):
        logger.info(f"Extracting text from PDF: {file_path}")
        text = extract_text_from_pdf(file_path)
    elif file_path_lower.endswith('.txt'):
        logger.info(f"Reading text file: {file_path}")
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Only PDF and TXT are supported.")
    
    logger.info(f"Extracted {len(text)} characters from document")
    
    # Chunk the text
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    logger.info(f"Split document into {len(chunks)} chunks")
    
    # Generate Q&A pairs for each chunk
    all_qa_data = []
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        qa_pairs = generate_qa_pairs(chunk, llm_client, model, num_questions)
        
        # Add to dataset with context
        for question, answer in qa_pairs:
            all_qa_data.append({
                "question": question,
                "answer": answer,
                "context": chunk
            })
    
    return all_qa_data, len(chunks)


def save_dataset(qa_data: List[Dict[str, str]], output_dir: str, dataset_name: str) -> str:
    """Save Q&A data as a HuggingFace dataset.
    
    Args:
        qa_data: List of Q&A dictionaries
        output_dir: Output directory path
        dataset_name: Name for the dataset
        
    Returns:
        Path to the saved dataset
    """
    if Dataset is None:
        raise ImportError("datasets library is required to save HuggingFace datasets")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_list(qa_data)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save dataset
    dataset_path = os.path.join(output_dir, dataset_name)
    dataset.save_to_disk(dataset_path)
    
    logger.info(f"Saved dataset to {dataset_path}")
    return dataset_path


def main():
    """Main entrypoint for the document Q&A generator container."""
    parser = argparse.ArgumentParser(
        description="Generate Q&A pairs from PDF/TXT documents"
    )
    parser.add_argument(
        "--systems-params",
        required=True,
        help="JSON string with system configurations"
    )
    parser.add_argument(
        "--test-params",
        required=True,
        help="JSON string with test parameters"
    )
    parser.add_argument(
        "--input-datasets",
        required=True,
        help="JSON string with input dataset configurations"
    )
    parser.add_argument(
        "--output-dir",
        default="/output",
        help="Output directory for generated datasets"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        # Parse inputs
        systems_params = json.loads(args.systems_params)
        test_params = json.loads(args.test_params)
        input_datasets = json.loads(args.input_datasets)
        
        # Extract generation system parameters
        gen_system = systems_params.get("generation_system", {})
        if not gen_system:
            raise ValueError("Missing generation_system in systems_params")
        
        base_url = gen_system.get("base_url")
        api_key = gen_system.get("api_key")
        model = gen_system.get("model")
        
        if not all([base_url, model]):
            raise ValueError("generation_system must have base_url and model")
        
        # Create LLM client
        if OpenAI is None:
            raise ImportError("openai library is required")
        
        llm_client = OpenAI(base_url=base_url, api_key=api_key or "dummy-key")
        
        # Extract test parameters
        num_questions = test_params.get("num_questions", 3)
        chunk_size = test_params.get("chunk_size", 1000)
        chunk_overlap = test_params.get("chunk_overlap", 200)
        output_dataset_name = test_params.get("output_dataset_name", "qa_dataset")
        
        # Extract input dataset path
        source_docs = input_datasets.get("source_documents", {})
        file_path = source_docs.get("file_path")
        
        if not file_path:
            raise ValueError("source_documents must have file_path")
        
        logger.info("Starting document Q&A generation")
        logger.info(f"Document: {file_path}")
        logger.info(f"Model: {model}")
        logger.info(f"Questions per chunk: {num_questions}")
        logger.info(f"Chunk size: {chunk_size}, overlap: {chunk_overlap}")
        
        # Process the document
        qa_data, chunks_processed = process_document(
            file_path,
            llm_client,
            model,
            chunk_size,
            chunk_overlap,
            num_questions
        )
        
        questions_generated = len(qa_data)
        logger.info(f"Generated {questions_generated} Q&A pairs from {chunks_processed} chunks")
        
        # Save the dataset
        dataset_path = save_dataset(qa_data, args.output_dir, output_dataset_name)
        
        processing_time = time.time() - start_time
        
        # Output results
        result = {
            "test_results": {
                "success": True,
                "questions_generated": questions_generated,
                "chunks_processed": chunks_processed,
                "processing_time_seconds": round(processing_time, 2),
                "output_dataset_path": dataset_path
            },
            "output_datasets": {
                "qa_dataset": {
                    "path": dataset_path,
                    "type": "huggingface",
                    "num_rows": questions_generated
                }
            }
        }
        
        print(json.dumps(result, indent=2))
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error during Q&A generation: {e}", exc_info=True)
        
        processing_time = time.time() - start_time
        
        error_result = {
            "test_results": {
                "success": False,
                "error": str(e),
                "questions_generated": 0,
                "chunks_processed": 0,
                "processing_time_seconds": round(processing_time, 2),
                "output_dataset_path": ""
            }
        }
        
        print(json.dumps(error_result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
