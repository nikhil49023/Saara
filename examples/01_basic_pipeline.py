"""
Example 1: Basic Document Processing Pipeline

Shows how to use SAARA as a Python library to process a PDF document
and generate a training dataset.
"""

from pathlib import Path
from saara import DataPipeline, PipelineConfig


def main():
    """Basic pipeline example."""

    # Create configuration
    config = PipelineConfig(
        output_directory="./datasets/example",
        model="granite",
        use_ocr=True,
        ocr_model="qwen",  # or "moondream" for lighter alternative
        chunk_size=1500,
        chunk_overlap=200,
        generate_synthetic=False,
    )

    # Initialize pipeline
    pipeline = DataPipeline(config)

    # Process a single PDF
    pdf_path = "example_document.pdf"  # Replace with your PDF
    dataset_name = "my_dataset"

    if not Path(pdf_path).exists():
        print(f"Error: {pdf_path} not found")
        print("Please provide a PDF file to process")
        return

    # Run the pipeline
    print(f"Processing {pdf_path}...")
    result = pipeline.process_file(pdf_path, dataset_name)

    # Check results
    if result.success:
        print(f"✓ Processing complete!")
        print(f"  Documents processed: {result.documents_processed}")
        print(f"  Total chunks: {result.total_chunks}")
        print(f"  Total samples: {result.total_samples}")
        print(f"  Duration: {result.duration_seconds:.2f} seconds")
        print(f"\nOutput files:")
        for file_type, path in result.output_files.items():
            print(f"  - {file_type}: {path}")
    else:
        print(f"✗ Processing failed")
        for error in result.errors:
            print(f"  Error: {error}")


if __name__ == "__main__":
    main()
