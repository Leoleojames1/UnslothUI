"""
Main entry point for Unsloth UI application.
Contains the Gradio UI code and connects to utility functions.
"""

import os
import gradio as gr
import time
from typing import Dict, Any, List, Tuple, Optional, Union

import config
import utils
import gguf_utils
import ollama_utils

def create_ui() -> gr.Blocks:
    """Create and return the Gradio UI application"""
    
    with gr.Blocks(title="Unsloth Fine-Tuning App", css=CSS) as app:
        # Header with logo
        with gr.Row(elem_id="header-container"):
            with gr.Column(elem_id="logo-container"):
                # Get current directory of the script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                logo_path = os.path.join(script_dir, "unslothSticker.png")
                
                # Use the unslothSticker.png image if it exists, otherwise fallback to emoji
                if os.path.exists(logo_path):
                    # For Gradio to find the image, we need to load it properly
                    logo_container = gr.Image(value=logo_path, show_label=False, container=False, 
                                            height=65, width=65)
                else:
                    logo_html = '<div id="logo-container"><div style="width:65px;height:65px;background-color:#19b98b;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:32px;">🦥</div></div>'
                    logo_container = gr.HTML(logo_html)
            
            with gr.Column():
                gr.HTML("<h1 class='header-text'>Unsloth Model Fine-Tuning</h1>")
        
        # Main content
        with gr.Row(elem_id="main-container"):
            # Config section (3/5 width)
            with gr.Column(elem_id="configs-container"):
                with gr.Tab("Fine-Tuning"):
                    gr.Markdown("### Model Configuration")
                    
                    with gr.Row():
                        with gr.Column():
                            model_dropdown = gr.Dropdown(choices=config.MODELS, label="Select Base Model", value=config.MODELS[3])
                        
                        with gr.Column():
                            chat_template = gr.Dropdown(
                                choices=config.CHAT_TEMPLATES, 
                                label="Chat Template", 
                                value="llama-3.2"
                            )
                    
                    # Dataset selection
                    gr.Markdown("### Dataset Configuration")
                    with gr.Row():
                        with gr.Column():
                            dataset_dropdown = gr.Dropdown(choices=config.COMMON_DATASETS, label="Dataset Name", value=config.COMMON_DATASETS[0])
                        
                        with gr.Column():
                            dataset_custom = gr.Textbox(label="Or enter custom dataset name (e.g., 'your-username/dataset')")
                    
                    with gr.Row():
                        with gr.Column():
                            split = gr.Textbox(label="Dataset Split", value="train")
                        
                        with gr.Column():
                            subsample = gr.Slider(minimum=1, maximum=100, step=1, value=100, label="Subsample Percentage")
                    
                    # Training parameters
                    gr.Markdown("### Training Parameters")
                    with gr.Row():
                        with gr.Column():
                            max_seq_length = gr.Slider(minimum=128, maximum=4096, step=128, value=2048, label="Max Sequence Length")
                        
                        with gr.Column():
                            lora_r = gr.Slider(minimum=1, maximum=256, step=1, value=16, label="LoRA Rank (r)")
                    
                    with gr.Row():
                        with gr.Column():
                            learning_rate = gr.Number(value=2e-4, label="Learning Rate")
                        
                        with gr.Column():
                            batch_size = gr.Slider(minimum=1, maximum=32, step=1, value=2, label="Batch Size")
                        
                        with gr.Column():
                            grad_accumulation = gr.Slider(minimum=1, maximum=32, step=1, value=4, label="Gradient Accumulation Steps")
                    
                    # Training duration
                    with gr.Row():
                        with gr.Column():
                            epochs = gr.Number(value=1, label="Number of Epochs (set 0 if using max_steps)")
                        
                        with gr.Column():
                            max_steps = gr.Number(value=60, label="Max Steps (overrides epochs if > 0)")
                        
                        with gr.Column():
                            train_responses_only = gr.Checkbox(label="Train Only on Assistant Responses", value=True)

                    # Saving options
                    gr.Markdown("### Output Configuration")
                    with gr.Row():
                        with gr.Column():
                            output_dir = gr.Textbox(label="Output Directory", value=config.DEFAULT_OUTPUT_DIR)
                        
                        with gr.Column():
                            push_to_hub = gr.Checkbox(label="Push to Hugging Face Hub", value=False)
                    
                    # Model merging option
                    with gr.Row():
                        with gr.Column():
                            merge_adapter = gr.Checkbox(label="Merge LoRA with base model", value=True, 
                                                      info="Creates a standalone model with the adapter integrated")
                    
                    # GGUF auto-export options
                    with gr.Row():
                        with gr.Column():
                            auto_export_gguf = gr.Checkbox(
                                label="Auto-export to GGUF after training",
                                value=False,
                                info="Automatically export the model to GGUF format after training"
                            )

                    with gr.Row(visible=False) as gguf_options_row:
                        with gr.Column():
                            auto_gguf_quantization = gr.CheckboxGroup(
                                choices=config.GGUF_QUANTIZATIONS,
                                label="GGUF Quantization Methods",
                                value=["q8_0"],
                                info="q4_k_m is smallest, q8_0 is balanced, f16 is highest quality"
                            )

                    # Update the visibility of GGUF options based on checkbox
                    auto_export_gguf.change(
                        fn=lambda x: gr.update(visible=x),
                        inputs=[auto_export_gguf],
                        outputs=[gguf_options_row]
                    )

                    # Repository settings with dual repo support
                    with gr.Row():
                        with gr.Column():
                            use_dual_repos = gr.Checkbox(
                                label="Use Separate Repositories for LoRA and Merged Model", 
                                value=False, 
                                visible=False,  # Initially hidden until push_to_hub and merge_adapter are both True
                                info="Upload the LoRA adapter and merged model to different repositories"
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            repo_id = gr.Textbox(label="Repository ID", value="")
                        
                        with gr.Column():
                            merged_repo_id = gr.Textbox(
                                label="Repository ID for Merged Model", 
                                value="", 
                                visible=False  # Initially hidden
                            )
                    
                    with gr.Row():
                        with gr.Column():
                            hf_token = gr.Textbox(label="Hugging Face Token (optional, for gated models or Hub)", type="password")
                    
                    # Buttons
                    with gr.Row():
                        start_button = gr.Button("Start Fine-Tuning", variant="primary")
                        stop_button = gr.Button("Stop Fine-Tuning (not implemented)", variant="stop")
                
                with gr.Tab("Inference"):
                    gr.Markdown("### Test Your Fine-Tuned Model")
                    
                    inference_model_path = gr.Textbox(label="Path to Fine-tuned Model", value=config.DEFAULT_OUTPUT_DIR)
                    
                    # Input for prompting
                    test_prompt = gr.TextArea(label="Enter your prompt", value="Continue the fibonacci sequence: 1, 1, 2, 3, 5, 8,", lines=5)
                    
                    # Generation parameters
                    with gr.Row():
                        with gr.Column():
                            temperature = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=0.7, label="Temperature")
                        
                        with gr.Column():
                            max_new_tokens = gr.Slider(minimum=10, maximum=2048, step=10, value=512, label="Max New Tokens")
                    
                    # Buttons for inference
                    infer_button = gr.Button("Run Inference", variant="primary")
              
                with gr.Tab("Export to GGUF"):
                    gr.Markdown("### Export Fine-tuned Model to GGUF Format")
                    
                    with gr.Row():
                        with gr.Column():
                            gguf_model_path = gr.Textbox(label="Path to Fine-tuned Model", value=config.DEFAULT_OUTPUT_DIR)
                        
                        with gr.Column():
                            gguf_output_dir = gr.Textbox(label="GGUF Output Directory", value=config.DEFAULT_GGUF_DIR)
                    
                    gr.Markdown("### Quantization Options")
                    with gr.Row():
                        with gr.Column():
                            # Multiple selection for quantization methods
                            quant_methods = gr.CheckboxGroup(
                                choices=config.GGUF_QUANTIZATIONS,
                                label="Quantization Methods",
                                value=["q8_0"],
                                info="q4_k_m is smallest, q8_0 is balanced, f16 is highest quality"
                            )
                    
                    gr.Markdown("### Hugging Face Repository Options")
                    with gr.Row():
                        with gr.Column():
                            gguf_push_to_hub = gr.Checkbox(label="Push GGUF to Hugging Face Hub", value=False)
                        
                        with gr.Column():
                            gguf_repo_id = gr.Textbox(label="Repository ID for GGUF models", value="")
                    
                    with gr.Row():
                        with gr.Column():
                            gguf_hf_token = gr.Textbox(label="Hugging Face Token", type="password")
                    
                    # Buttons
                    with gr.Row():
                        export_button = gr.Button("Export to GGUF", variant="primary")
                    
                    # Output section
                    gr.Markdown("### Export Output")
                    gguf_status_text = gr.Markdown("")
                    gguf_output_message = gr.TextArea(label="Log", interactive=False, lines=15)
                    
                    # Connect the export button
                    export_button.click(
                        fn=gguf_utils.export_to_gguf,
                        inputs=[
                            gguf_model_path, gguf_output_dir, quant_methods,
                            gguf_push_to_hub, gguf_repo_id, gguf_hf_token
                        ],
                        outputs=[gguf_status_text, gguf_output_message]
                    )
                    
                    # Ollama Testing Section
                    gr.Markdown("### Test with Ollama")
                    with gr.Row():
                        with gr.Column():
                            ollama_model_name = gr.Textbox(label="Ollama Model Name", value="unsloth_model")
                            ollama_test_prompt = gr.TextArea(
                                label="Test Prompt", 
                                value="Continue the Fibonacci sequence: 1, 1, 2, 3, 5, 8,",
                                lines=3
                            )
                            
                            with gr.Row():
                                with gr.Column():
                                    ollama_temperature = gr.Slider(
                                        minimum=0.1, maximum=2.0, step=0.1, value=0.7, 
                                        label="Temperature"
                                    )
                                
                                with gr.Column():
                                    ollama_max_tokens = gr.Slider(
                                        minimum=10, maximum=2048, step=10, value=512, 
                                        label="Max Tokens"
                                    )
                            
                            # Start Ollama server button
                            with gr.Row():
                                start_ollama_button = gr.Button("Start Ollama Server", variant="secondary")
                                test_ollama_button = gr.Button("Test with Ollama", variant="primary")
                            
                            ollama_output = gr.TextArea(label="Ollama Response", interactive=False, lines=15)
              
                with gr.Tab("Ollama Integration"):
                    # First check Ollama installation
                    ollama_installed, ollama_status = gguf_utils.check_ollama_installed()
                    gr.Markdown(f"### Ollama Status: {ollama_status}")
                    
                    if ollama_installed:
                        gr.Markdown("### Create Ollama Model from GGUF File")
                        
                        # GGUF file selection
                        with gr.Row():
                            with gr.Column():
                                ollama_gguf_path = gr.Textbox(label="Path to GGUF File", value=os.path.join(config.DEFAULT_GGUF_DIR, "model-q8_0.gguf"))
                            
                            with gr.Column():
                                ollama_scan_button = gr.Button("Scan for GGUF Files", variant="secondary")
                        
                        # GGUF file dropdown (populated by scan)
                        gguf_files_dropdown = gr.Dropdown(
                            label="Available GGUF Files",
                            choices=[],
                            info="Select a GGUF file or enter a path manually above"
                        )
                        
                        # Model configuration
                        gr.Markdown("### Model Configuration")
                        with gr.Row():
                            with gr.Column():
                                ollama_model_name_input = gr.Textbox(label="Ollama Model Name", value="my-unsloth-model")
                            
                            with gr.Column():
                                ollama_template_dropdown = gr.Dropdown(
                                    label="Chat Template", 
                                    choices=["llama", "llama3", "phi", "mistral", "other"],
                                    value="llama3",
                                    info="Select the template style matching your model"
                                )
                        
                        # System prompt
                        ollama_system_prompt = gr.TextArea(
                            label="System Prompt", 
                            value="You are a helpful AI assistant based on a fine-tuned model. You provide clear, accurate, and engaging responses to user queries.",
                            lines=5
                        )
                        
                        # Parameter settings
                        gr.Markdown("### Generation Parameters")
                        with gr.Row():
                            with gr.Column():
                                ollama_temperature = gr.Slider(
                                    minimum=0.1, maximum=2.0, step=0.1, value=0.7, 
                                    label="Temperature"
                                )
                            
                            with gr.Column():
                                ollama_context_length = gr.Slider(
                                    minimum=1024, maximum=16384, step=1024, value=4096,
                                    label="Context Length"
                                )
                        
                        # Buttons for creating
                        with gr.Row():
                            create_ollama_modelfile_button = gr.Button("Create Modelfile", variant="secondary")
                            import_to_ollama_button = gr.Button("Create & Import to Ollama", variant="primary")
                        
                        # Output displays
                        gr.Markdown("### Output")
                        ollama_modelfile_path = gr.Textbox(label="Modelfile Path", interactive=False)
                        ollama_modelfile_content = gr.Code(
                            label="Modelfile Content", 
                            language="yaml",
                            interactive=False,
                            lines=15
                        )
                        ollama_output_message = gr.Textbox(label="Status", interactive=False)
                        
                        # Connect the components
                        def scan_for_gguf_files():
                            gguf_files = gguf_utils.find_gguf_files()
                            if not gguf_files:
                                return [], "No GGUF files found in common directories."
                            choices = [path for path, label in gguf_files]
                            return gr.Dropdown.update(choices=choices, value=choices[0] if choices else None), f"Found {len(gguf_files)} GGUF files."
                        
                        ollama_scan_button.click(
                            fn=scan_for_gguf_files,
                            inputs=[],
                            outputs=[gguf_files_dropdown, ollama_output_message]
                        )
                        
                        gguf_files_dropdown.change(
                            fn=lambda x: x,
                            inputs=[gguf_files_dropdown],
                            outputs=[ollama_gguf_path]
                        )
                        
                        create_ollama_modelfile_button.click(
                            fn=gguf_utils.create_ollama_modelfile,
                            inputs=[
                                ollama_gguf_path, 
                                ollama_model_name_input, 
                                ollama_system_prompt, 
                                ollama_template_dropdown,
                                ollama_temperature, 
                                ollama_context_length
                            ],
                            outputs=[ollama_modelfile_path, ollama_modelfile_content, ollama_output_message]
                        )
                        
                        import_to_ollama_button.click(
                            fn=lambda *args: (
                                *gguf_utils.create_ollama_modelfile(*args)[:2],
                                gguf_utils.import_to_ollama(gguf_utils.create_ollama_modelfile(*args)[0], args[1])
                            ),
                            inputs=[
                                ollama_gguf_path, 
                                ollama_model_name_input, 
                                ollama_system_prompt, 
                                ollama_template_dropdown,
                                ollama_temperature, 
                                ollama_context_length
                            ],
                            outputs=[ollama_modelfile_path, ollama_modelfile_content, ollama_output_message]
                        )
    
            # Output section (2/5 width)
            with gr.Column(elem_id="outputs-container"):
                gr.Markdown("### Training Output")
                status_text = gr.Markdown("")
                output_message = gr.TextArea(label="Log", elem_id="output-message", interactive=False, lines=20)
                
                gr.Markdown("### Hardware Information")
                hw_info = config.get_hardware_info()
                if "warning" in hw_info:
                    gr.Markdown(f"* {hw_info['warning']}")
                else:
                    gr.Markdown(f"""
                    * {hw_info.get('gpu_info', '')}
                    * {hw_info.get('gpu_memory', '')}
                    * {hw_info.get('bfloat16_support', '')}
                    """)
                
                with gr.Tab("Inference Results"):
                    infer_output = gr.TextArea(label="Model Output", interactive=False, lines=20)
    
        # For the spinning animation, we'll use a different approach
        # We'll use HTML and CSS for non-image logo, and for the image logo,
        # we'll use a loading message instead since direct image animation is tricky
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(script_dir, "unslothSticker.png")
        logo_exists = os.path.exists(logo_path)
        
        def start_spinning():
            if logo_exists:
                # For image logo, we'll show a "Training..." text next to it
                return """<div style="display: flex; align-items: center; gap: 10px;">
                          <div style="color: white; font-weight: bold; font-size: 18px; animation: pulse 1.5s infinite;">Training...</div>
                          </div>"""
            else:
                # For emoji logo, we can do the spinning animation
                return '<div id="logo-container" class="spinning"><div style="width:65px;height:65px;background-color:#19b98b;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:32px;">🦥</div></div>'
        
        def stop_spinning():
            if logo_exists:
                # Clear the training message
                return ""
            else:
                # Stop spinning for emoji logo
                return '<div id="logo-container"><div style="width:65px;height:65px;background-color:#19b98b;border-radius:50%;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;font-size:32px;">🦥</div></div>'
        
        # Update visibility of dual repo options based on checkbox states
        def update_repo_visibility(push_hub, merge):
            """Update the visibility of repository fields based on selections"""
            # Only show dual repo option if pushing to hub and merging
            dual_repos_visible = push_hub and merge
            return {
                use_dual_repos: gr.update(visible=dual_repos_visible),
                repo_id: gr.update(label="Repository ID for LoRA Adapter" if dual_repos_visible else "Repository ID")
            }

        def update_merged_repo_visibility(use_dual):
            """Update merged repository field visibility"""
            return gr.update(visible=use_dual)

        def handle_dual_repos_change(use_dual, model, push):
            """Update repo IDs when dual repos option changes"""
            if use_dual and push:
                lora_id, merged_id = utils.update_repo_ids(model, push, True)
                return lora_id, merged_id
            else:
                single_id, _ = utils.update_repo_ids(model, push, False)
                return single_id, ""

        # Connect UI components for interactivity
        push_to_hub.change(
            fn=update_repo_visibility,
            inputs=[push_to_hub, merge_adapter],
            outputs=[use_dual_repos, repo_id]
        )

        merge_adapter.change(
            fn=update_repo_visibility,
            inputs=[push_to_hub, merge_adapter],
            outputs=[use_dual_repos, repo_id]
        )

        # Update merged repo visibility when dual repos option changes
        use_dual_repos.change(
            fn=update_merged_repo_visibility,
            inputs=[use_dual_repos],
            outputs=merged_repo_id
        )

        # Add dataset selection logic
        def get_dataset_name(dropdown_value, custom_value):
            """Return custom dataset if provided, otherwise return dropdown selection"""
            return custom_value if custom_value else dropdown_value
        
        dataset_custom.change(
            fn=get_dataset_name,
            inputs=[dataset_dropdown, dataset_custom],
            outputs=[dataset_dropdown]
        )
        
        # Add a HTML element for the training indicator
        with gr.Column():
            training_indicator = gr.HTML("")
        
        # Connect the start button for training
        start_button.click(
            fn=start_spinning,
            inputs=[],
            outputs=[training_indicator]
        ).then(
            fn=utils.finetune_model,
            inputs=[
                model_dropdown, dataset_dropdown, split, subsample, max_seq_length, 
                lora_r, learning_rate, batch_size, grad_accumulation, epochs, 
                max_steps, chat_template, train_responses_only, hf_token,
                output_dir, push_to_hub, repo_id, merged_repo_id, merge_adapter, 
                use_dual_repos, auto_export_gguf, auto_gguf_quantization
            ],
            outputs=[status_text, output_message]
        ).then(
            fn=stop_spinning,
            inputs=[],
            outputs=[training_indicator]
        )
        
        # Run inference on the fine-tuned model
        infer_button.click(
            fn=utils.test_inference,
            inputs=[inference_model_path, test_prompt, temperature, max_new_tokens],
            outputs=[status_text, infer_output]
        )
        
        # Connect Ollama testing buttons
        start_ollama_button.click(
            fn=ollama_utils.start_ollama_server,
            inputs=[],
            outputs=[ollama_output]
        )
        
        test_ollama_button.click(
            fn=gguf_utils.test_ollama_model,
            inputs=[ollama_model_name, ollama_test_prompt, ollama_temperature, ollama_max_tokens],
            outputs=[ollama_output]
        )
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p>Powered by Unsloth | Fine-tune LLMs faster and more efficiently</p>
        </div>
        """)
        
        return app

# CSS for dark theme and improved UI (bigger, wider, better spacing)
CSS = f"""
:root {{
    --brand-color: {config.BRAND_COLOR};
    --background-color: #1e1e2f;
    --text-color: #f8f8f2;
    --secondary-background: #272a44;
    --border-color: #3c4180;
}}

.gr-checkbox {{
    color: var(--brand-color) !important;
    transform: scale(1.3) !important; /* Larger checkbox */
    margin-right: 10px !important;
}}

.options-container {{
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
}}

.option-card {{
    background-color: var(--secondary-background);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 20px;
    flex: 1;
    min-width: 250px;
}}

/* IMPROVED WIDE LAYOUT */
#main-container {{
    display: flex;
    width: 100%;
    gap: 25px; /* Add spacing between columns */
    padding: 0 25px;
}}

#configs-container {{
    flex: 3;  /* Makes the config area even wider */
    padding-right: 25px;
}}

#outputs-container {{
    flex: 2;  /* Makes the output area wider too */
    display: flex;
    flex-direction: column;
}}

/* Make input fields wider */
.gr-input-container {{
    width: 100% !important;
}}

/* Better column spacing */
.gr-form > .gr-form-row {{
    gap: 25px !important;
}}

/* Better padding for the overall interface */
.gradio-container .main {{
    padding: 0 !important;
}}

/* Better dropdown styles */
.gr-dropdown {{
    height: 50px !important;
}}

/* Responsive layout improvements */
@media (max-width: 1400px) {{
    #main-container {{
        flex-direction: column;
    }}
    #configs-container, #outputs-container {{
        width: 100%;
    }}
}}

/* Section spacing */
.gr-block.gr-box {{
    margin-bottom: 25px !important;
}}

/* Improve tab items */
.tabs .tabitem {{
    font-size: 18px !important;
    padding: 15px 25px !important;
}}

/* Make textarea taller */
.gr-textarea {{
    min-height: 120px !important;
}}

body {{
    background-color: var(--background-color);
    color: var(--text-color);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 16px; /* Increased base font size */
}}

.dark {{
    color-scheme: dark;
}}

@keyframes spin {{
    from {{transform: rotate(0deg);}}
    to {{transform: rotate(360deg);}}
}}

/* HEADER IMPROVEMENTS */
#header-container {{
    position: sticky;
    top: 0;
    z-index: 999;
    background-color: var(--brand-color);
    padding: 15px 30px; /* Increased padding */
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    display: flex;
    align-items: center;
    height: 90px; /* Taller header */
    width: 100%;
}}

/* Add pulse animation for training indicator */
@keyframes pulse {{
    0% {{ opacity: 0.6; }}
    50% {{ opacity: 1; }}
    100% {{ opacity: 0.6; }}
}}

#logo-container {{
    width: 65px; /* Bigger logo container */
    height: 65px;
    margin-right: 20px;
    display: flex;
    align-items: center;
    justify-content: center;
}}

#logo-container.spinning img {{
    animation: spin 2s linear infinite;
}}

.logo-image {{
    width: 65px; /* Bigger logo */
    height: 65px;
}}

.header-text {{
    color: white;
    font-size: 28px; /* Larger header text */
    font-weight: bold;
    margin: 0;
}}

/* CONTAINER IMPROVEMENTS */
.gradio-container {{
    max-width: 2000px !important; /* Much wider max-width */
    margin: 0 auto;
    padding: 0 !important;
}}

.tabs {{
    background-color: var(--secondary-background);
    border-radius: 12px;
    margin-top: 20px;
    border: 1px solid var(--border-color);
}}

.block {{
    background-color: var(--secondary-background) !important;
}}

/* UI ELEMENT IMPROVEMENTS */
.gr-button-primary {{
    background-color: var(--brand-color) !important;
    font-size: 18px !important; /* Bigger buttons */
    padding: 12px 24px !important;
    border-radius: 8px !important;
}}

.gr-button-secondary {{
    color: var(--text-color) !important;
    border-color: var(--border-color) !important;
    font-size: 18px !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
}}

.gr-input, .gr-select, .gr-textarea {{
    background-color: var(--background-color) !important;
    color: var(--text-color) !important;
    border-color: var(--border-color) !important;
    font-size: 16px !important; /* Larger fonts */
    padding: 12px !important;
    border-radius: 8px !important;
}}

.gr-form {{
    background-color: var(--secondary-background) !important;
    border-radius: 12px;
    padding: 25px !important; /* More padding */
    border: 1px solid var(--border-color) !important;
    margin-bottom: 25px !important;
}}

.gr-panel {{
    border-color: var(--border-color) !important;
}}

.gr-box {{
    background-color: var(--secondary-background) !important;
    border-radius: 12px;
    border: 1px solid var(--border-color) !important;
}}

.gr-padded {{
    padding: 25px !important;
}}

.tabs .tabitem[aria-selected="true"] {{
    background-color: var(--brand-color) !important;
    color: white !important;
    font-size: 18px !important;
}}

.footer {{
    margin-top: 30px;
    text-align: center;
    font-size: 16px;
    color: #a1a1a1;
    padding: 20px;
}}

#output-message {{
    min-height: 400px; /* Taller output area */
    overflow-y: auto;
    font-size: 16px !important;
}}

label {{
    color: var(--text-color) !important;
    font-size: 16px !important; /* Larger labels */
    margin-bottom: 8px !important;
    font-weight: bold !important;
}}

h1, h2, h3, h4 {{
    color: var(--brand-color) !important;
    margin-top: 25px !important;
    margin-bottom: 20px !important;
}}

h3 {{
    font-size: 22px !important;
}}

.gr-slider {{
    background-color: var(--border-color) !important;
    height: 8px !important; /* Thicker slider */
}}

.gr-slider-handle {{
    background-color: var(--brand-color) !important;
    width: 20px !important; /* Larger slider handle */
    height: 20px !important;
}}
"""