"""
Entry point for the Unsloth UI application.
Run this file to start the application.
"""

import os
import gradio as gr
from main import create_ui
import config

if __name__ == "__main__":
    # Print environment information
    print(f"Unsloth Models Directory: {config.UNSLOTH_FINETUNED_MODELS}")
    
    # Print available directories
    print(f"Base Models Directory: {os.path.join(config.UNSLOTH_FINETUNED_MODELS, 'base_models')}")
    print(f"Datasets Directory: {os.path.join(config.UNSLOTH_FINETUNED_MODELS, 'datasets')}")
    print(f"Finetuned Models Directory: {os.path.join(config.UNSLOTH_FINETUNED_MODELS, 'finetuned')}")
    print(f"Merged Models Directory: {os.path.join(config.UNSLOTH_FINETUNED_MODELS, 'merged')}")
    print(f"GGUF Models Directory: {os.path.join(config.UNSLOTH_FINETUNED_MODELS, 'gguf')}")
    print(f"Ollama Directory: {os.path.join(config.UNSLOTH_FINETUNED_MODELS, 'ollama')}")
    
    # Print hardware info
    if config.CUDA_AVAILABLE:
        print(f"CUDA Available: Yes")
        print(f"GPU: {config.torch.cuda.get_device_name(0)}")
        print(f"Memory: {config.torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"BFloat16 Support: {config.is_bfloat16_supported()}")
    else:
        print("CUDA Available: No - Training will be slow on CPU.")
    
    # Create and launch the UI
    app = create_ui()
    app.launch(share=False)  # Set share=True to create a public link
