# Unsloth UI - Directory Structure and Project Organization

Here's a detailed explanation of the project structure and how it's organized for modularity and maintainability:

## Core Files

- **run.py**: The entry point for the application. Initializes environment, shows system info, and launches the UI.
- **main.py**: Contains the Gradio UI definition and layout. Connects UI elements to the underlying functions.
- **config.py**: Contains configuration settings, environment variables, model lists, and hardware detection.
- **utils.py**: Core utility functions for dataset processing, model fine-tuning, and inference.
- **gguf_utils.py**: Functions for GGUF conversion, finding GGUF files, and Ollama modelfile creation.
- **ollama_utils.py**: Utilities for Ollama integration, testing models, and running servers.
- **convert_to_gguf.bat**: Windows batch script for llama.cpp conversion.

## Organization of Model Files

The application creates a structured environment variable-based directory:

```
$UNSLOTH_FINETUNED_MODELS/
├── base_models/            # Downloaded base models
├── datasets/               # Training datasets
├── finetuned/              # Fine-tuned models with LoRA adapters
├── merged/                 # Merged models (base + adapter)
├── gguf/                   # GGUF exports
└── ollama/                 # Ollama modelfiles and configurations
    └── modelfiles_*        # Timestamp-based directories for Modelfiles
```

## Ollama Integration Structure

For Ollama integration, the application uses the approach from the original script:

```
$UNSLOTH_FINETUNED_MODELS/ollama/
├── modelfiles_TIMESTAMP/   # Directory for generated Modelfiles
│   └── model_name.Modelfile  # Ollama Modelfile
└── Ignored_Agents/
    └── model_name/         # Directory for each Ollama model
        └── Modelfile         # Ollama Modelfile
```

## Modular Design

The application follows a modular design with clear separation of concerns:

1. **Configuration (config.py)**:
   - Environment variables and path setup
   - Hardware detection
   - Predefined model lists
   - UI constants

2. **Core Utilities (utils.py)**:
   - Dataset loading and processing
   - Model fine-tuning
   - Model inference
   - Repository ID generation

3. **GGUF Utilities (gguf_utils.py)**:
   - GGUF export with Unsloth
   - GGUF conversion with llama.cpp
   - File finding functions
   - Ollama modelfile creation

4. **Ollama Utilities (ollama_utils.py)**:
   - Ollama server management
   - Custom model creation
   - Model testing
   - Batch-based llama.cpp conversion

5. **UI Layer (main.py)**:
   - Gradio interface definition
   - UI layout and styling
   - Event handlers and callbacks

This modular approach makes the codebase:
- Easier to maintain (fixing one area doesn't affect others)
- More extensible (adding new features is simpler)
- More robust (better error handling in specific modules)
- More reusable (utilities can be used in other projects)

## Data Flow

1. **Configuration ➝ Utilities**: Config settings are imported by utilities
2. **Utilities ➝ UI**: UI calls utility functions when actions are performed
3. **UI ➝ User**: User interacts with UI and sees results
4. **Utilities ➝ External Systems**: Utilities interact with Hugging Face, Ollama, filesystem

## Environment Variables

The application uses environment variables to customize behavior:

- **UNSLOTH_FINETUNED_MODELS**: Base directory for all model files
- **USER**: Used for repository ID generation

If not set, the application uses sensible defaults for all paths.

## Customization Points

The application can be extended by modifying:

1. **Model Lists**: Add new models in config.py
2. **Dataset Lists**: Add new datasets in config.py
3. **Templates**: Modify chat templates in gguf_utils.py
4. **Quantization Options**: Add new options in config.py
5. **UI Layout**: Modify the layout in main.py
