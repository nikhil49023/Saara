# 🖥️ Saara CLI User Guide

This comprehensive guide covers the installation, usage, and workflows of the **Saara Data Engine** Command Line Interface (CLI).

## 1. Installation

### Standard Installation
The recommended way to install Saara is via pip:

```bash
pip install saara-ai
```

### Upgrade
To get the latest version (with new features like the Deployment Wizard):

```bash
pip install --upgrade saara-ai
```

### Troubleshooting Installation
If you encounter **Permission Errors** (e.g., `[WinError 32]` or `site-packages is not writeable`), use the `--user` flag:

```bash
pip install --user --upgrade saara-ai
```

---

## 2. Quick Start

The easiest way to use Saara is through its interactive wizard. Simply run:

```bash
saara run
```

This will launch the **Splash Screen** and the **Main Menu**, guiding you through all available workflows.

---

## 3. Command Reference

### `saara run` (or `saara wizard`)
Launches the interactive mode. This is the primary entry point for most users.
* **Usage**: `saara run`

### `saara version`
Displays the installed version of Saara, Python version, and license information.
* **Usage**: `saara version`

### `saara process`
Process a single PDF file directly from the command line without the wizard.
* **Usage**: `saara process path/to/document.pdf`
* **Options**:
    * `--name`, `-n`: Name of the output dataset (default: filename).
    * `--config`, `-c`: Path to custom config file.

### `saara batch`
Process an entire directory of PDFs.
* **Usage**: `saara batch path/to/folder`
* **Options**:
    * `--name`: Name of the combined output dataset.

### `saara train`
Interactively fine-tune a model on your datasets.
* **Usage**: `saara train`
* **Features**: Prompts for dataset selection (file or directory merge) and base model selection.

### `saara deploy`
Launch the deployment wizard to chat with, export, or deploy your model.
* **Usage**: `saara deploy`

### `saara evaluate`
Evaluate a fine-tuned model.
* **Usage**: `saara evaluate [BASE_MODEL_ID] [ADAPTER_PATH]`
* **Important**: You must provide the exact HuggingFace Hub ID for the base model (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0`), not just a local folder name, to avoid size mismatch errors.

---

## 4. Interactive Workflows

When you run `saara run`, you can choose from four powerful modes:

### 📄 Option 1: Dataset Creation
**Goal**: Transform raw PDFs into high-quality instruction-tuning datasets.

1.  **Select Input**: Choose a PDF file or a directory of PDFs.
2.  **Vision OCR**: The engine uses advanced Vision-LLMs (Moondream/Qwen) to read the document, preserving tables and distinct layouts.
3.  **Labeling**: Agents (like Granite 4.0) generate Q&A pairs, summaries, and instructions from the text.
4.  **Output**: A clean `.jsonl` file in **ShareGPT** format, ready for training.

### 🧠 Option 2: Model Training
**Goal**: Fine-tune an LLM on your created dataset.

1.  **Select Dataset**: Pick the `.jsonl` file generated in step 1.
2.  **Base Model**: Choose a base model (e.g., `TinyLlama`, `Llama-3`, `Mistral`).
3.  **Training**: Saara runs a **QLoRA** fine-tuning job interactively.
    *   *Note: Requires a GPU for reasonable speeds.*
4.  **Output**: A fine-tuned LoRA adapter saved in the `models/` directory.

### 🧪 Option 3: Model Evaluation
**Goal**: Test your model's quality and improve it autonomously.

1.  **Select Adapter**: Choose one of your trained models.
2.  **Judge**: Saara uses a larger model (the "Judge") to grade your model's answers.
3.  **Autonomous Learning**:
    *   Topic: You provide a topic (e.g., "Ayurveda").
    *   Loop: The model generates answers -> Judge grades them -> If score is low, the "Teacher" model provides the correct answer -> Dataset is updated.

### 🚀 Option 4: Model Deployment
**Goal**: Use your model in the real world.

1.  **Select Model**: Choose your fine-tuned adapter.
2.  **Deployment Options**:
    *   **Local Chat**: Chat with your model directly in the terminal to test it.
    *   **Export to Ollama**: Convert your model to GGUF format to run it locally with Ollama (`ollama run my-model`).
    *   **Push to HuggingFace**: Upload your adapter to the HuggingFace Hub.
    *   **Cloud Deployment**: Generate Dockerfiles and deploy to Google Cloud Run or a FastAPI server.

---

## 5. Configuration (`config.yaml`)

Saara uses a `config.yaml` file to control advanced parameters. You can override defaults here.

```yaml
# LLM Settings
llm:
  provider: "ollama"
  model: "granite-code:8b"     # Model used for labeling
  vision_model: "moondream"    # Model used for OCR

# Pipeline Steps
steps:
  ocr: true
  clean: true
  label: true
  distill: true

# Training Params
training:
  batch_size: 2
  epochs: 3
  learning_rate: 2.0e-4
  quantization: "4bit"         # 4bit, 8bit, or none
```

---

## 6. Troubleshooting

**"Failed to find CUDA" / PyTorch Warnings**
*   This means you are running on CPU. Saara will still work for OCR and Labeling (slowly), but Training will be very slow.
*   Ensure you have installed the CUDA version of PyTorch if you have an NVIDIA GPU.

**"Ollama Connection Error"**
*   Ensure the Ollama application is running in the background.
*   Verify the model you are using (e.g., `granite-code:8b`) is pulled: `ollama pull granite-code:8b`.

**"Permission Denied" during pip install**
*   Windows sometimes locks files. Close any running terminals using Saara and try `pip install --user --upgrade saara-ai`.
