"""
Configuration settings for Unsloth UI.
Contains environment variables, constants, and shared settings.
"""

import os
import torch
from typing import List, Dict, Any

# Check for environment variable for model storage
UNSLOTH_FINETUNED_MODELS = os.environ.get(
    "UNSLOTH_FINETUNED_MODELS", 
    os.path.join(os.path.expanduser("~"), "unsloth_models")
)

# Create directory structure if it doesn't exist
os.makedirs(UNSLOTH_FINETUNED_MODELS, exist_ok=True)
os.makedirs(os.path.join(UNSLOTH_FINETUNED_MODELS, "base_models"), exist_ok=True)
os.makedirs(os.path.join(UNSLOTH_FINETUNED_MODELS, "datasets"), exist_ok=True)
os.makedirs(os.path.join(UNSLOTH_FINETUNED_MODELS, "finetuned"), exist_ok=True)
os.makedirs(os.path.join(UNSLOTH_FINETUNED_MODELS, "merged"), exist_ok=True)
os.makedirs(os.path.join(UNSLOTH_FINETUNED_MODELS, "gguf"), exist_ok=True)
os.makedirs(os.path.join(UNSLOTH_FINETUNED_MODELS, "ollama"), exist_ok=True)

# Hardware detection
CUDA_AVAILABLE = torch.cuda.is_available()
def is_bfloat16_supported():
    """Check if bfloat16 is supported by the GPU."""
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# Brand color for UI
BRAND_COLOR = "#19b98b"

# Default paths
DEFAULT_OUTPUT_DIR = os.path.join(UNSLOTH_FINETUNED_MODELS, "finetuned")
DEFAULT_MERGED_DIR = os.path.join(UNSLOTH_FINETUNED_MODELS, "merged")
DEFAULT_GGUF_DIR = os.path.join(UNSLOTH_FINETUNED_MODELS, "gguf")
DEFAULT_OLLAMA_DIR = os.path.join(UNSLOTH_FINETUNED_MODELS, "ollama")

# List of Unsloth 4bit quantized models to choose from
MODELS: List[str] = [
    "unsloth/Llama-3.2-1B-bnb-4bit",
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Mistral-Small-Instruct-2409",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",
    "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"
]

# List of chat templates - exact strings that Unsloth accepts
CHAT_TEMPLATES: List[str] = [
    "unsloth", "zephyr", "chatml", "mistral", "llama", "vicuna", "vicuna_old", "vicuna old", 
    "alpaca", "gemma", "gemma_chatml", "gemma2", "gemma2_chatml", "llama-3", "llama3", 
    "phi-3", "phi-35", "phi-3.5", "llama-3.1", "llama-31", "llama-3.2", "llama-32", 
    "llama-3.3", "llama-33", "qwen-2.5", "qwen-25", "qwen25", "qwen2.5", "phi-4", 
    "gemma-3", "gemma3"
]

# List of common datasets
COMMON_DATASETS: List[str] = [
    "mlabonne/FineTome-100k", "hellaswag", "imdb", "jfleg", "eli5",
    "databricks/databricks-dolly-15k", "tatsu-lab/alpaca",
    "HuggingFaceH4/no_robots", "Open-Orca/OpenOrca",
    "argilla/distilabel-math-preference-dpo"
]

# GGUF quantization options
GGUF_QUANTIZATIONS: List[str] = ["q4_k_m", "q5_k_m", "q8_0", "f16"]

# Hardware info for display
def get_hardware_info() -> Dict[str, str]:
    """Get hardware information for display in the UI."""
    if CUDA_AVAILABLE:
        gpu_info = f"GPU: {torch.cuda.get_device_name(0)}"
        gpu_memory = f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        bfloat16_support = "BFloat16 Support: " + ("Yes" if is_bfloat16_supported() else "No")
        return {
            "gpu_info": gpu_info,
            "gpu_memory": gpu_memory,
            "bfloat16_support": bfloat16_support
        }
    else:
        return {"warning": "No GPU detected. Training will be slow on CPU."}
