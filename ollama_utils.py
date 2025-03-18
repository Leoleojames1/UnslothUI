"""
Ollama integration utilities for Unsloth UI.
Provides functions for creating, managing, and running Ollama models.
"""

import os
import time
import subprocess
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import requests
import shutil

import config
from gguf_utils import (
    create_ollama_modelfile, 
    import_to_ollama, 
    test_ollama_model,
    check_ollama_installed
)

def get_model_template(model_name: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Extract template and parameters from an existing Ollama model.
    
    Args:
        model_name: Name of the Ollama model to extract info from
        
    Returns:
        Tuple containing success status and dictionary with model info
    """
    try:
        # Add error handling for missing ollama executable
        if not shutil.which("ollama"):
            return False, {"error": "Ollama not found in system PATH"}
            
        result = subprocess.run(
            ["ollama", "show", "--modelfile", model_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode != 0:
            return False, {"error": f"Failed to get model info: {result.stderr}"}
            
        modelfile = result.stdout
        model_info = {
            "template": "",
            "parameters": {},
            "system_prompt": ""
        }
        
        # Extract template if present
        import re
        template_match = re.search(r'TEMPLATE\s+"""(.*?)"""', modelfile, re.DOTALL)
        if template_match:
            model_info["template"] = template_match.group(1).strip()
            
        # Extract parameters
        param_matches = re.finditer(r'PARAMETER\s+(\w+)\s+(.*?)(?=\n|$)', modelfile)
        for match in param_matches:
            param_name = match.group(1)
            param_value = match.group(2).strip()
            # Convert numeric values
            try:
                if '.' in param_value:
                    param_value = float(param_value)
                else:
                    param_value = int(param_value)
            except ValueError:
                # Keep as string if not numeric
                pass
            model_info["parameters"][param_name] = param_value
            
        # Extract system prompt
        system_match = re.search(r'SYSTEM\s+"""(.*?)"""', modelfile, re.DOTALL)
        if system_match:
            model_info["system_prompt"] = system_match.group(1).strip()
            
        return True, model_info
        
    except Exception as e:
        return False, {"error": f"Error extracting model info: {str(e)}"}
    
def create_and_deploy_custom_model(
    model_name: str,
    base_model: str = "llama3.2:3b",
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    context_length: int = 4096,
    stop_tokens: Optional[List[str]] = None,
    test_prompt: str = "Tell me what you can do",
    ollama_export: bool = True,
) -> Optional[str]:
    """
    Creates a custom ModelFile based on an existing Ollama model's configuration.
    
    Args:
        model_name: Name for your custom model
        base_model: Base Ollama model to use as template
        system_prompt: Custom system prompt (overrides base model's prompt)
        temperature: Temperature override (if None, uses base model's value)
        context_length: Context length override (if None, uses base model's value)
        stop_tokens: Custom stop tokens (if None, uses base model's tokens)
        test_prompt: Test prompt for verification
        ollama_export: Whether to export to Ollama
        
    Returns:
        Path to the generated Modelfile or None if failed
    """
    # Generate timestamp for unique file naming
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(config.DEFAULT_OLLAMA_DIR, f"modelfiles_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base model configuration
    success, base_info = get_model_template(base_model)
    if not success:
        print(f"❌ Failed to get base model info: {base_info.get('error', 'Unknown error')}")
        return None
    
    # Build the new Modelfile content
    modelfile_content = f"FROM {base_model}\n\n"
    
    # Add system prompt
    if system_prompt is not None:
        modelfile_content += f'SYSTEM """\n{system_prompt}\n"""\n\n'
    elif base_info["system_prompt"]:
        modelfile_content += f'SYSTEM """\n{base_info["system_prompt"]}\n"""\n\n'
    
    # Add parameters
    parameters = base_info["parameters"].copy()
    if temperature is not None:
        parameters["temperature"] = temperature
    if context_length is not None:
        parameters["num_ctx"] = context_length
    if stop_tokens is not None:
        parameters["stop"] = stop_tokens
        
    for param_name, param_value in parameters.items():
        if isinstance(param_value, list):
            for value in param_value:
                modelfile_content += f'PARAMETER {param_name} {json.dumps(value)}\n'
        else:
            modelfile_content += f'PARAMETER {param_name} {param_value}\n'
    
    # Add template if present in base model
    if base_info["template"]:
        modelfile_content += f'\nTEMPLATE """\n{base_info["template"]}\n"""\n'
    
    # Save the Modelfile
    output_path = os.path.join(output_dir, f"{model_name}.Modelfile")
    with open(output_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"✅ Generated Modelfile at: {output_path}")
    
    # Create Ollama model if requested
    # Add more robust error handling for model creation
    if ollama_export:
        try:
            ollama_name = model_name.split("/")[-1].lower().replace(" ", "-")
            print(f"🔄 Creating Ollama model '{ollama_name}'...")
            
            # Add timeout to prevent hanging
            result = subprocess.run(
                ["ollama", "create", ollama_name, "-f", output_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # More descriptive success/error messages
            if result.returncode == 0:
                print(f"✅ Ollama model '{ollama_name}' created successfully!")
                print("\n📋 Usage Instructions:")
                print(f"1. Command Line: ollama run {ollama_name}")
                print("2. Python API:")
                print(f"""
                import ollama
                response = ollama.chat(
                model='{ollama_name}',
                messages=[{{'role': 'user', 'content': '{test_prompt}'}}]
                )
                print(response['message']['content'])
                """)
            else:
                print(f"❌ Failed to create Ollama model: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("❌ Model creation timed out")
            return None

def extract_modelfile(model_name: str) -> Tuple[bool, str]:
    """
    Extracts the Modelfile from an existing Ollama model using 'ollama show'.
    
    Args:
        model_name: Name of the existing Ollama model
        
    Returns:
        Tuple containing success status and Modelfile content or error message
    """
    try:
        # Run 'ollama show --modelfile model_name' command
        result = subprocess.run(
            ["ollama", "show", "--modelfile", model_name],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Error: {result.stderr}"
    except subprocess.CalledProcessError as e:
        return False, f"Failed to extract Modelfile: {e.stderr}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def list_ollama_models() -> List[str]:
    """
    Get a list of all available Ollama models.
    
    Returns:
        List of model names
    """
    try:
        # Run 'ollama list' command
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.returncode == 0:
            # Parse the output to extract model names
            lines = result.stdout.strip().split('\n')
            # Skip the header line and extract the first column (model name)
            models = []
            for line in lines[1:]:  # Skip header
                if line.strip():  # Skip empty lines
                    models.append(line.split()[0])  # First column is model name
            return models
        else:
            print(f"Error listing models: {result.stderr}")
            return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def start_ollama_server() -> Tuple[bool, str]:
    """
    Start the Ollama server if it's not already running.
    
    Returns:
        Tuple containing success status and message
    """
    try:
        # Check if Ollama server is already running
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return True, "Ollama server is already running."
        except:
            pass  # Server not running, we'll start it
        
        # Start the server based on platform
        import platform
        system = platform.system()
        
        if system == "Windows":
            # On Windows, we use start command to run in background
            process = subprocess.Popen(
                ["start", "cmd", "/c", "ollama", "serve"],
                shell=True
            )
            message = "Started Ollama server in background. Please wait a few seconds for it to initialize."
        elif system in ["Linux", "Darwin"]:  # Darwin is macOS
            # On Linux/macOS, use nohup to run in background
            process = subprocess.Popen(
                ["nohup", "ollama", "serve", "&"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                shell=True
            )
            message = "Started Ollama server in background. Please wait a few seconds for it to initialize."
        else:
            return False, f"Unsupported platform: {system}"
        
        # Wait a bit for the server to start
        time.sleep(3)
        
        # Check if server started successfully
        try:
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                return True, f"{message}\nServer is now running."
            else:
                return False, f"{message}\nServer may not have started properly."
        except:
            return False, f"{message}\nServer did not start properly. Please check Ollama installation."
    
    except Exception as e:
        return False, f"Error starting Ollama server: {str(e)}"

def run_llama_cpp_convert(
    model_dir: str,
    output_dir: str,
    model_name: str,
    quantization_type: str = "q8_0"
) -> Tuple[bool, str]:
    """
    Run the llama.cpp convert-hf-to-gguf.py script using command line batch processing.
    
    Args:
        model_dir: Path to the model directory
        output_dir: Path to save the converted GGUF file
        model_name: Name for the GGUF model
        quantization_type: Quantization type (q8_0, f16, etc.)
        
    Returns:
        Tuple containing success status and message
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if the batch file exists and where it is
        batch_path = None
        possible_paths = [
            os.path.join(os.getcwd(), "convert_to_gguf.bat"),
            os.path.join(os.path.dirname(os.getcwd()), "convert_to_gguf.bat"),
            os.path.join(config.UNSLOTH_FINETUNED_MODELS, "convert_to_gguf.bat")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                batch_path = path
                break
        
        if batch_path is None:
            # Create the batch file if it doesn't exist
            batch_path = os.path.join(config.UNSLOTH_FINETUNED_MODELS, "convert_to_gguf.bat")
            with open(batch_path, "w") as f:
                f.write("@echo on\n")
                f.write("cd %1\n")
                f.write("python llama.cpp\\convert-hf-to-gguf.py --outtype %3 --model-name %2-%3 --outfile %1\\converted\\%2-%3.gguf %2\n")
                f.write("@REM python llama.cpp\\convert-hf-to-gguf.py --outtype f16 --model-name %2-f16 --outfile %1\\converted\\%2-f16.gguf %2\n")
                f.write("@REM python llama.cpp\\convert-hf-to-gguf.py --outtype f32 --model-name %2-f32 --outfile %1\\converted\\%2-f32.gguf %2\n")
        
        # Create converted directory
        os.makedirs(os.path.join(output_dir, "converted"), exist_ok=True)
        
        # Run the batch file
        process = subprocess.run(
            [batch_path, output_dir, model_name, quantization_type],
            shell=True,
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            # Check if file was created
            expected_file = os.path.join(output_dir, "converted", f"{model_name}-{quantization_type}.gguf")
            if os.path.exists(expected_file):
                file_size = os.path.getsize(expected_file) / (1024 * 1024)  # Convert to MB
                return True, f"Successfully converted model to GGUF format: {expected_file} ({file_size:.1f} MB)"
            else:
                return False, f"Conversion appeared to succeed but output file not found: {expected_file}"
        else:
            return False, f"Error in conversion process: {process.stderr}"
    
    except Exception as e:
        import traceback
        return False, f"Error running llama.cpp convert: {str(e)}\n\n{traceback.format_exc()}"

def create_ollama_model_from_gguf(
    gguf_path: str,
    model_name: str,
    system_prompt: str,
    output_dir: Optional[str] = None
) -> Tuple[bool, str]:
    """
    Create an Ollama model from a GGUF file using the batch file approach.
    
    Args:
        gguf_path: Path to the GGUF file
        model_name: Name for the Ollama model
        system_prompt: System prompt for the model
        output_dir: Path to save the Modelfile (defaults to Ignored_Agents dir)
        
    Returns:
        Tuple containing success status and message
    """
    try:
        # If output_dir is not provided, create default path
        if output_dir is None:
            output_dir = os.path.join(config.UNSLOTH_FINETUNED_MODELS, "ollama", "Ignored_Agents", model_name)
        
        # Create the directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create the Modelfile
        modelfile_path = os.path.join(output_dir, "Modelfile")
        with open(modelfile_path, "w") as f:
            f.write(f"FROM {gguf_path}\n")
            f.write(f"#temperature higher -> creative, lower -> coherent\n")
            f.write(f"PARAMETER temperature 0.7\n")
            f.write(f"\n#Set the system prompt\n")
            f.write(f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n")
        
        # Create the ollama model
        process = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True
        )
        
        if process.returncode == 0:
            return True, f"""✅ Ollama model '{model_name}' created successfully from GGUF file!

You can now use it with:
- CLI: ollama run {model_name}
- Python: ollama.chat(model='{model_name}', messages=[{{'role': 'user', 'content': 'Hello'}}])
"""
        else:
            return False, f"Failed to create Ollama model: {process.stderr}"
    
    except Exception as e:
        import traceback
        return False, f"Error creating Ollama model from GGUF: {str(e)}\n\n{traceback.format_exc()}"
