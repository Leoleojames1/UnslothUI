"""
GGUF conversion and Ollama integration utilities for Unsloth UI.
"""

import os
import time
import subprocess
from typing import List, Optional, Union, Tuple
from huggingface_hub import login

import config

def export_to_gguf(
    model_dir: str, 
    output_dir: str, 
    quantization_methods: Optional[Union[List[str], str]] = None, 
    push_to_hub: bool = False, 
    repo_id: str = "", 
    hf_token: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Export a fine-tuned model to GGUF format with various quantization options.
    
    Args:
        model_dir: Directory of the fine-tuned model
        output_dir: Directory to save GGUF files
        quantization_methods: Quantization method(s) to use
        push_to_hub: Whether to push to Hugging Face Hub
        repo_id: Hugging Face repository ID (username/repo-name)
        hf_token: Hugging Face token for authentication
        
    Returns:
        Tuple containing success status and message
    """
    try:
        # Clear CUDA cache before loading model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Create device map for CPU offloading
        device_map = {
            "model.embed_tokens": "cpu",
            "lm_head": "cpu",
            "model.norm": "cpu",
            "model.layers": "sequential"
        }
        
        # Load model with CPU offloading
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
            device_map=device_map,
            llm_int8_enable_fp32_cpu_offload=True,
            offload_folder="temp_offload"
        )
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the fine-tuned model
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_dir,
            max_seq_length=2048,  # Use same as training
            dtype=None,  # Auto detection
            load_in_4bit=True
        )
        
        # Process quantization methods
        if quantization_methods is None:
            # Default to Q8_0 if not specified
            quantization_methods = ["q8_0"]
        elif isinstance(quantization_methods, str):
            quantization_methods = [quantization_methods]
        
        # Generate a simple Modelfile instead of accessing tokenizer._ollama_modelfile
        model_name = os.path.basename(os.path.normpath(model_dir))
        
        # Create template with proper indentation
        template = '''{{- if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>

{{- end }}{{ range $i, $message := .Messages }}{{- if eq $message.Role "user" }}<|start_header_id|>user<|end_header_id|>

{{ $message.Content }}<|eot_id|>

{{- else if eq $message.Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>

{{ $message.Content }}<|eot_id|>

{{- end }}{{ end }}<|start_header_id|>assistant<|end_header_id|>

'''
        
        # Build Modelfile content with proper spacing
        base_modelfile_content = f'''FROM ./model-q8_0.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096

# Template for the model (assuming Llama3 template as default)
TEMPLATE """{template}"""
'''
        
        # Save the Modelfile
        modelfile_path = os.path.join(output_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(base_modelfile_content)
        
        # Save local GGUF files
        results = []
        for quant_method in quantization_methods:
            try:
                gguf_path = os.path.join(output_dir, f"model-{quant_method}.gguf")
                model.save_pretrained_gguf(
                    output_dir, 
                    tokenizer, 
                    quantization_method=quant_method
                )
                results.append(f"Saved {quant_method} GGUF to {gguf_path}")
            except Exception as e:
                results.append(f"Error saving {quant_method} GGUF: {str(e)}")
        
        # Push to Hugging Face if requested
        if push_to_hub and repo_id:
            try:
                if hf_token:
                    from huggingface_hub import login
                    login(token=hf_token)
                
                # Push all quantization methods
                model.push_to_hub_gguf(
                    repo_id,
                    tokenizer,
                    quantization_method=quantization_methods,
                    token=hf_token
                )
                results.append(f"Successfully pushed GGUF models to {repo_id}")
            except Exception as e:
                results.append(f"Error pushing to HF Hub: {str(e)}")
        
        # Try to create Ollama model if Ollama is installed
        try:
            ollama_name = repo_id.split("/")[-1] if repo_id else "unsloth_model"
            subprocess.run(["ollama", "create", ollama_name, "-f", modelfile_path], 
                          check=True, capture_output=True, timeout=60)
            results.append(f"Created Ollama model '{ollama_name}'")
        except Exception as e:
            results.append(f"Ollama model creation skipped or failed: {str(e)}")
        
        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True, "\n".join(results)
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return False, f"Error exporting to GGUF: {str(e)}\n\nDetails:\n{error_details}"

def convert_hf_to_gguf(
    model_dir: str,
    output_dir: str,
    quantization_methods: List[str] = ["q8_0"],
    llama_cpp_dir: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Convert a Hugging Face model to GGUF format using llama.cpp convert script.
    
    Args:
        model_dir: Path to Hugging Face model directory
        output_dir: Path to save GGUF files
        quantization_methods: List of quantization methods to use
        llama_cpp_dir: Path to llama.cpp directory (if None, will try to find it)
        
    Returns:
        Tuple containing success status and message
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # If llama_cpp_dir is not provided, try to find it
        if llama_cpp_dir is None:
            # Check common locations
            possible_locations = [
                "./llama.cpp",
                "../llama.cpp",
                os.path.expanduser("~/llama.cpp"),
                os.path.join(config.UNSLOTH_FINETUNED_MODELS, "llama.cpp")
            ]
            
            for location in possible_locations:
                if os.path.exists(location) and os.path.exists(os.path.join(location, "convert-hf-to-gguf.py")):
                    llama_cpp_dir = location
                    break
            
            if llama_cpp_dir is None:
                # If still not found, try to clone it
                try:
                    subprocess.run(
                        ["git", "clone", "https://github.com/ggerganov/llama.cpp.git"],
                        check=True,
                        cwd=config.UNSLOTH_FINETUNED_MODELS
                    )
                    llama_cpp_dir = os.path.join(config.UNSLOTH_FINETUNED_MODELS, "llama.cpp")
                except Exception as e:
                    return False, f"Error: Could not find or clone llama.cpp repository: {str(e)}"
        
        # Check if convert script exists
        convert_script = os.path.join(llama_cpp_dir, "convert-hf-to-gguf.py")
        if not os.path.exists(convert_script):
            return False, f"Error: convert-hf-to-gguf.py not found in {llama_cpp_dir}"
        
        # Get model name (basename)
        model_name = os.path.basename(os.path.normpath(model_dir))
        
        # Results array
        results = []
        
        # Convert for each requested quantization method
        for quant in quantization_methods:
            output_file = os.path.join(output_dir, f"{model_name}-{quant}.gguf")
            
            # Prepare command
            cmd = [
                "python",
                convert_script,
                "--outtype", quant,
                "--model-name", f"{model_name}-{quant}",
                "--outfile", output_file,
                model_dir
            ]
            
            # Execute command
            try:
                process = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                if os.path.exists(output_file):
                    size_mb = os.path.getsize(output_file) / (1024 * 1024)
                    results.append(f"Successfully created {quant} model: {output_file} ({size_mb:.1f} MB)")
                else:
                    results.append(f"Error: Output file not created for {quant}")
                    
            except subprocess.CalledProcessError as e:
                results.append(f"Error converting to {quant}: {e.stderr}")
        
        if any("Successfully" in result for result in results):
            return True, "\n".join(results)
        else:
            return False, "\n".join(results)
            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return False, f"Error during HF to GGUF conversion: {str(e)}\n\nDetails:\n{error_details}"

def find_gguf_files() -> List[Tuple[str, str]]:
    """Find GGUF models in common directories
    
    Returns:
        List of tuples containing (file_path, display_name)
    """
    gguf_files = []
    
    # Common directories where GGUF files might be found
    search_dirs = [
        config.DEFAULT_GGUF_DIR,
        os.getcwd(),
        os.path.join(os.getcwd(), "finetuned_model_gguf"),
        os.path.join(os.getcwd(), "gguf_exports"),
    ]
    
    # Look for any directory with _gguf suffix up to 2 levels deep
    for root, dirs, _ in os.walk(os.getcwd()):
        # Limit depth to avoid excessive searching
        if root.count(os.sep) - os.getcwd().count(os.sep) <= 2:
            for dir_name in dirs:
                if dir_name.endswith("_gguf") or "_gguf_" in dir_name:
                    search_dirs.append(os.path.join(root, dir_name))
    
    # Search for GGUF files in all the directories
    for directory in search_dirs:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith(".gguf"):
                    full_path = os.path.join(directory, file)
                    # Get file size for information
                    size_mb = os.path.getsize(full_path) / (1024 * 1024)
                    gguf_files.append((full_path, f"{file} ({size_mb:.1f} MB)"))
    
    return gguf_files

def check_ollama_installed() -> Tuple[bool, str]:
    """Check if Ollama is installed and available
    
    Returns:
        Tuple containing status and message
    """
    try:
        result = subprocess.run(["ollama", "version"], capture_output=True, text=True)
        return True, f"Ollama found: {result.stdout.strip()}"
    except:
        return False, "⚠️ Ollama not found. Please install Ollama first: https://ollama.com"

def create_ollama_modelfile(
    gguf_path: str,
    model_name: str,
    system_prompt: str,
    template_type: str,
    temperature: float,
    context_length: int
) -> Tuple[Optional[str], str, str]:
    """Creates a Modelfile for an existing GGUF model
    
    Args:
        gguf_path: Path to GGUF file
        model_name: Name for the Ollama model
        system_prompt: System prompt
        template_type: Template type (llama, phi, mistral, etc.)
        temperature: Temperature value
        context_length: Context length
        
    Returns:
        Tuple containing (modelfile_path, modelfile_content, message)
    """
    try:
        # Create timestamp-based output directory
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_dir = os.path.join(config.DEFAULT_OLLAMA_DIR, f"modelfiles_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Create Modelfile content
        modelfile_content = f"FROM {gguf_path}\n\n"
        
        # Add system prompt if provided
        if system_prompt:
            modelfile_content += f'SYSTEM """{system_prompt}"""\n\n'
        
        # Add parameters
        modelfile_content += f"PARAMETER temperature {temperature}\n"
        modelfile_content += f"PARAMETER num_ctx {context_length}\n"
        
        # Add appropriate template based on model type
        if template_type in ["llama", "llama3"]:
            modelfile_content += '\n# Using Llama template\n'
            modelfile_content += 'TEMPLATE """{{- if .System }}<|start_header_id|>system<|end_header_id|>\n\n{{ .System }}<|eot_id|>\n\n{{- end }}{{ range $i, $message := .Messages }}{{- if eq $message.Role "user" }}<|start_header_id|>user<|end_header_id|>\n\n{{ $message.Content }}<|eot_id|>\n\n{{- else if eq $message.Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>\n\n{{ $message.Content }}<|eot_id|>\n\n{{- end }}{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n"""\n'
        elif template_type == "phi":
            modelfile_content += '\n# Using Phi template\n'
            modelfile_content += 'TEMPLATE """{{- if .System }}<|system|>\n{{ .System }}\n<|user|>\n{{- else }}<|user|>\n{{- end }}{{ range $i, $message := .Messages }}{{- if eq $message.Role "user" }}{{ $message.Content }}\n<|assistant|>\n{{- else if eq $message.Role "assistant" }}{{ $message.Content }}\n<|user|>\n{{- end }}{{ end }}"""\n'
        elif template_type == "mistral":
            modelfile_content += '\n# Using Mistral template\n'
            modelfile_content += 'TEMPLATE """{{- if .System }}[INST] {{ .System }} [/INST]\n\n{{- end }}{{ range $i, $message := .Messages }}{{- if eq $message.Role "user" }}[INST] {{ $message.Content }} [/INST]{{- else if eq $message.Role "assistant" }}\n\n{{ $message.Content }}{{- end }}{{ end }}"""\n'
        
        # Save the Modelfile
        safe_model_name = model_name.replace("/", "-").lower()
        output_path = os.path.join(output_dir, f"{safe_model_name}.Modelfile")
        with open(output_path, "w") as f:
            f.write(modelfile_content)
        
        return output_path, modelfile_content, f"✅ Modelfile generated at: {output_path}"
        
    except Exception as e:
        return None, "", f"Error generating Modelfile: {str(e)}"

def import_to_ollama(modelfile_path: str, model_name: str) -> str:
    """Imports the model to Ollama
    
    Args:
        modelfile_path: Path to Modelfile
        model_name: Name for the Ollama model
        
    Returns:
        Status message
    """
    try:
        # Format the name for Ollama (lowercase, replace spaces with hyphens)
        ollama_name = model_name.lower().replace(" ", "-")
        
        # Run the ollama create command
        result = subprocess.run(
            ["ollama", "create", ollama_name, "-f", modelfile_path], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            return f"""✅ Ollama model '{ollama_name}' created successfully!

You can now use it with:
- CLI: ollama run {ollama_name}
- Python: ollama.chat(model='{ollama_name}', messages=[{{'role': 'user', 'content': 'Hello'}}])
"""
        else:
            return f"❌ Failed to create Ollama model: {result.stderr}"
            
    except Exception as e:
        return f"❌ Error creating Ollama model: {str(e)}"

def test_ollama_model(model_name: str, prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> Tuple[bool, str]:
    """
    Test a model exported to Ollama.
    
    Args:
        model_name: Ollama model name
        prompt: Prompt to send to the model
        temperature: Temperature for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Tuple containing success status and message
    """
    try:
        import json
        import requests
        
        # Check if Ollama is running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code != 200:
                return False, "Ollama server is not running. Start it with 'ollama serve'"
        except Exception:
            return False, "Ollama server is not running. Start it with 'ollama serve'"
        
        # Construct the API request
        data = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        # Make API request
        response = requests.post(
            "http://localhost:11434/api/chat",
            json=data
        )
        
        if response.status_code == 200:
            result = response.json()
            return True, f"Response from Ollama ({model_name}):\n\n{result.get('message', {}).get('content', 'No content')}"
        else:
            return False, f"Error: {response.status_code} - {response.text}"
    
    except Exception as e:
        import traceback
        return False, f"Error testing Ollama model: {str(e)}\n\n{traceback.format_exc()}"
