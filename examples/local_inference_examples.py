"""
Local-First Inference Quick Start
Demonstrates using vLLM and Ollama for both local and cloud notebooks.
"""

from saara.local_inference import LocalInferenceEngine, InferenceConfig


# ===========================================================================
# SCENARIO 1: Local Machine (Recommended - vLLM)
# ===========================================================================

def example_local_machine():
    """Local machine - vLLM takes priority."""
    print("=== Local Machine Setup ===\n")
    
    # Auto-select best backend (vLLM > Ollama)
    config = InferenceConfig(
        model="mistral:latest",  # or "mistral:7b-instruct-v0.2-q4_K_M"
        temperature=0.7,
        max_tokens=2048,
    )
    
    engine = LocalInferenceEngine(config)
    print(f"Using: {engine.get_info()}\n")
    
    # Simple generation
    response = engine.generate("Explain quantum computing briefly")
    print(f"Response: {response[:200]}...\n")
    
    # Streaming
    print("Streaming response:")
    for chunk in engine.generate_stream("What is machine learning?"):
        print(chunk, end="", flush=True)
    print("\n")


# ===========================================================================
# SCENARIO 2: Google Colab / Kaggle (vLLM Native)
# ===========================================================================

def example_cloud_notebook():
    """Cloud notebook (Colab/Kaggle) - install dependencies at runtime."""
    print("=== Cloud Notebook Setup ===\n")
    
    # Install vLLM in Colab/Kaggle
    # Note: Run this in a notebook cell:
    # !pip install -q vllm
    
    config = InferenceConfig(
        model="meta-llama/Llama-2-7b-chat-hf",  # HF model ID
        # Or for vLLM Quant:
        # model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
        temperature=0.7,
        max_tokens=1024,
    )
    
    engine = LocalInferenceEngine(config)
    print(f"Using: {engine.get_info()}\n")
    
    # Generate as normal
    response = engine.generate(
        "Write a Python function to read CSV",
        system_prompt="You are a helpful Python coding assistant"
    )
    print(response)


# ===========================================================================
# SCENARIO 3: Force Ollama (if vLLM unavailable)
# ===========================================================================

def example_force_ollama():
    """Force Ollama backend explicitly."""
    print("=== Force Ollama ===\n")
    
    # Ensure Ollama is running: ollama serve
    # Pull model: ollama pull mistral
    
    config = InferenceConfig(
        backend="ollama",  # Force Ollama
        model="mistral",
        temperature=0.5,
        base_url="http://localhost:11434",
    )
    
    engine = LocalInferenceEngine(config)
    response = engine.generate("Hello, what can you do?")
    print(response)


# ===========================================================================
# SCENARIO 4: Batch Processing with Fallback
# ===========================================================================

def example_batch_with_fallback():
    """Process multiple prompts with automatic fallback."""
    print("=== Batch Processing ===\n")
    
    config = InferenceConfig(
        model="mistral:latest",
        enable_auto_fallback=True,  # Auto-switchto Ollama if vLLM fails
    )
    
    engine = LocalInferenceEngine(config)
    
    prompts = [
        "What is AI?",
        "Explain machine learning",
        "How do neural networks work?"
    ]
    
    for prompt in prompts:
        try:
            response = engine.generate(prompt)
            print(f"Q: {prompt}")
            print(f"A: {response[:150]}...\n")
        except Exception as e:
            print(f"Failed: {e}\n")


# ===========================================================================
# SCENARIO 5: Health Check & Diagnostics
# ===========================================================================

def example_health_check():
    """Check which backends are available."""
    print("=== System Diagnostics ===\n")
    
    config = InferenceConfig(model="mistral:latest")
    engine = LocalInferenceEngine(config)
    
    print(f"Current Backend: {engine.backend_name}\n")
    
    health = engine.health_check()
    print("Available Backends:")
    for backend, status in health.items():
        print(f"  {backend}:")
        print(f"    Available: {status.get('available')}")
        print(f"    Healthy: {status.get('healthy')}")
        if 'error' in status:
            print(f"    Error: {status['error']}")


# ===========================================================================
# SCENARIO 6: Using in SAARA Pipeline
# ===========================================================================

def example_integration_with_saara():
    """Integrate local inference into SAARA pipeline."""
    print("=== Integration with SAARA ===\n")
    
    from saara import DataPipeline, PipelineConfig
    
    # Configure pipeline with local inference
    config = PipelineConfig(
        output_directory="./datasets",
        use_ocr=True,
        teacher_model="local",  # Use local model instead of Gemini
    )
    
    # In pipeline.py, it will use LocalInferenceEngine internally
    pipeline = DataPipeline(config)
    result = pipeline.process_file("document.pdf", "my_dataset")
    
    print(f"Processing complete: {result}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
        if scenario == "local":
            example_local_machine()
        elif scenario == "colab":
            example_cloud_notebook()
        elif scenario == "ollama":
            example_force_ollama()
        elif scenario == "batch":
            example_batch_with_fallback()
        elif scenario == "health":
            example_health_check()
        elif scenario == "saara":
            example_integration_with_saara()
        else:
            print(f"Unknown scenario: {scenario}")
    else:
        print("Run with: python local_inference_examples.py [local|colab|ollama|batch|health|saara]")
