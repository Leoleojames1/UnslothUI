"""
Core utility functions for Unsloth UI.
Contains dataset loading, model fine-tuning, and other utilities.
"""

import os
import torch
import time
from typing import Tuple, List, Dict, Any, Optional, Union
import subprocess
from datasets import load_dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt, train_on_responses_only
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from trl import SFTTrainer
from huggingface_hub import login
from peft import AutoPeftModelForCausalLM
import getpass

import config

def load_and_process_dataset(dataset_name: str, split: str = "train", subsample: int = 100) -> Tuple[Any, str]:
    """
    Load dataset from HuggingFace and process it.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace
        split: Dataset split to use
        subsample: Percentage of dataset to use
        
    Returns:
        Tuple containing the processed dataset and a status message
    """
    try:
        # Load the dataset
        if "/" in dataset_name:  # Full path like "mlabonne/FineTome-100k"
            dataset = load_dataset(dataset_name, split=split)
        else:  # Just the name like "imdb"
            dataset = load_dataset(dataset_name, split=split)
        
        # Subsample if needed
        if subsample < 100 and subsample > 0:
            dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * subsample / 100)))
        
        # Check if the dataset has the right format
        if 'conversations' in dataset.column_names:
            # Already in the right format
            dataset = standardize_sharegpt(dataset)
        elif all(col in dataset.column_names for col in ['input', 'output']):
            # Convert from input/output format to conversations format
            dataset = dataset.map(
                lambda x: {
                    'conversations': [
                        {'role': 'user', 'content': x['input']},
                        {'role': 'assistant', 'content': x['output']}
                    ]
                }
            )
        elif 'text' in dataset.column_names:
            # Dataset has only text, convert to simple Q&A format
            dataset = dataset.map(
                lambda x: {
                    'conversations': [
                        {'role': 'user', 'content': 'Please continue the following text:\n\n' + x['text'][:100] + '...'},
                        {'role': 'assistant', 'content': x['text']}
                    ]
                }
            )
        else:
            # Try to adapt to common dataset formats
            cols = dataset.column_names
            if 'question' in cols and 'answer' in cols:
                dataset = dataset.map(
                    lambda x: {
                        'conversations': [
                            {'role': 'user', 'content': x['question']},
                            {'role': 'assistant', 'content': x['answer']}
                        ]
                    }
                )
            else:
                # If we can't figure out the format, just return a message
                return None, f"Could not determine dataset format. Available columns: {', '.join(cols)}"
        
        return dataset, f"Successfully loaded dataset '{dataset_name}' with {len(dataset)} examples."
    except Exception as e:
        return None, f"Error loading dataset: {str(e)}"

def finetune_model(
    model: str, 
    dataset: str,
    split: str,
    subsample: int,
    max_seq_length: int,
    lora_r: int,
    learning_rate: float, 
    batch_size: int,
    grad_accumulation: int,
    epochs: int,
    max_steps: int,
    chat_template_value: str, 
    train_responses_only: bool,
    hf_token: str,
    output_dir: str,
    push_to_hub: bool,
    repo_id: str,
    merged_repo_id: str,
    merge_adapter: bool,
    use_dual_repos: bool,
    auto_export_gguf: bool = False,
    gguf_quantization: Optional[Union[List[str], str]] = None
) -> Tuple[str, str]:
    """
    Fine-tune a model using Unsloth.
    
    Args:
        model: Model name
        dataset: Dataset name
        split: Dataset split
        subsample: Percentage of dataset to use
        max_seq_length: Maximum sequence length
        lora_r: LoRA rank
        learning_rate: Learning rate
        batch_size: Batch size
        grad_accumulation: Gradient accumulation steps
        epochs: Number of epochs
        max_steps: Maximum number of steps
        chat_template_value: Chat template
        train_responses_only: Whether to train only on assistant responses
        hf_token: Hugging Face token
        output_dir: Output directory
        push_to_hub: Whether to push to Hugging Face Hub
        repo_id: Repository ID
        merged_repo_id: Repository ID for merged model
        merge_adapter: Whether to merge adapter with base model
        use_dual_repos: Whether to use separate repositories for LoRA and merged model
        auto_export_gguf: Whether to export to GGUF after training
        gguf_quantization: GGUF quantization methods
        
    Returns:
        Tuple containing status and detailed message
    """
    success, message = False, ""
    try:
        # Step 1: Load the dataset
        dataset_obj, msg = load_and_process_dataset(dataset, split, subsample)
        if dataset_obj is None:
            return "Status: Failed", msg
        
        # Step 2: Initialize model and tokenizer
        load_in_4bit = True
        dtype = None  # Auto detection
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model,
            max_seq_length=max_seq_length,
            dtype=dtype,
            load_in_4bit=load_in_4bit,
            token=hf_token if hf_token else None
        )
        
        # Step 3: Set the chat template
        print(f"Using chat template: '{chat_template_value}'")
        
        tokenizer = get_chat_template(
            tokenizer,
            chat_template=chat_template_value,
        )
        
        # Step 4: Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=lora_r,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Step 5: Prepare the dataset
        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
            return {"text": texts}

        processed_dataset = dataset_obj.map(formatting_prompts_func, batched=True, num_proc=1)
        
        # Step 6: Set up the trainer
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accumulation,
            warmup_steps=5,
            learning_rate=learning_rate,
            fp16=not config.is_bfloat16_supported(),
            bf16=config.is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        )
        
        # FIX: Set num_train_epochs or max_steps based on input
        # Important: Handle the case where either value could be None or 0
        if max_steps and max_steps > 0:
            training_args.max_steps = max_steps
            training_args.num_train_epochs = 1  # Set a default value instead of None
        else:
            # If epochs is None or 0, set a default value
            training_args.num_train_epochs = epochs if epochs and epochs > 0 else 1
            training_args.max_steps = -1  # Disable max_steps
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=processed_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
            dataset_num_proc=1,
            packing=False,
            args=training_args,
        )
        
        # Step 7: Train only on assistant responses if selected
        if train_responses_only:
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        
        # Step 8: Start training
        trainer_stats = trainer.train()
        
        # Step 9: Save the model locally
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Step 10: Handle model merging (if requested)
        if merge_adapter:
            merged_model_dir = f"{output_dir}_merged"
            os.makedirs(merged_model_dir, exist_ok=True)
            
            try:
                # Clear CUDA cache before loading model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Create custom device map for CPU offloading if needed
                total_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # Convert to GB
                device_map = "auto" if total_gpu_memory > 24 else {
                    "model.embed_tokens": "cpu",
                    "lm_head": "cpu",
                    "model.norm": "cpu",
                    "model.layers": "sequential"  # Load layers sequentially
                }
                
                # Load the LoRA adapter with CPU offloading enabled
                model = AutoPeftModelForCausalLM.from_pretrained(
                    output_dir,
                    torch_dtype=torch.float16 if not config.is_bfloat16_supported() else torch.bfloat16,
                    low_cpu_mem_usage=True,
                    device_map=device_map,
                    offload_folder="offload_folder",
                    offload_state_dict=True
                )
                
                # Merge and save in chunks to reduce memory usage
                merged_model = model.merge_and_unload()
                
                # Save the merged model with aggressive memory optimization
                merged_model.save_pretrained(
                    merged_model_dir,
                    safe_serialization=True,
                    max_shard_size="1GB",  # Reduced shard size
                    save_function=torch.save
                )
                tokenizer.save_pretrained(merged_model_dir)
                
                merge_message = f"\nSuccessfully merged LoRA adapter with base model. Saved to: {merged_model_dir}"
                
                # Clear memory again
                del model
                del merged_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                merge_message = f"\nWarning: Failed to merge the model: {str(e)}"
                merged_model_dir = output_dir  # Fall back to adapter model
        else:
            merge_message = ""
            merged_model_dir = output_dir
        
        # Step 11: Push to Hugging Face if requested
        if push_to_hub:
            # Login if token provided
            if hf_token:
                login(token=hf_token)
            
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                # Test token validity
                try:
                    api.whoami()
                    print("Token is valid")
                except Exception as e:
                    print(f"Token error: {e}")
                
                # Handle repository uploads based on user selection
                if use_dual_repos and merge_adapter:
                    # Upload both the LoRA adapter and merged model to separate repos
                    
                    # 1. Upload LoRA adapter
                    if repo_id:
                        api.create_repo(repo_id=repo_id, exist_ok=True)
                        api.upload_folder(
                            folder_path=output_dir,  # LoRA adapter path
                            repo_id=repo_id,
                            ignore_patterns=[".*", "__pycache__"]
                        )
                    
                    # 2. Upload merged model
                    if merged_repo_id:
                        api.create_repo(repo_id=merged_repo_id, exist_ok=True)
                        api.upload_folder(
                            folder_path=merged_model_dir,  # Merged model path
                            repo_id=merged_repo_id,
                            ignore_patterns=[".*", "__pycache__"]
                        )
                        
                    result_message = f"Model trained and pushed to Hugging Face Hub:\n"
                    result_message += f"- LoRA adapter: {repo_id}\n"
                    result_message += f"- Merged model: {merged_repo_id}"
                
                else:
                    # Original functionality - upload either LoRA or merged model to a single repo
                    model_dir_to_push = merged_model_dir if merge_adapter else output_dir
                    repo_to_use = repo_id
                    
                    if repo_to_use:
                        api.create_repo(repo_id=repo_to_use, exist_ok=True)
                        api.upload_folder(
                            folder_path=model_dir_to_push,
                            repo_id=repo_to_use,
                            ignore_patterns=[".*", "__pycache__"]
                        )
                        
                        result_message = f"Model trained and pushed to Hugging Face Hub: {repo_to_use}"
                        if merge_adapter:
                            result_message += f"\nUploaded the MERGED model (not just the adapter)."
                    else:
                        result_message = "No repository ID provided. Models saved locally only."
                
            except Exception as e:
                result_message = f"Error uploading to Hugging Face Hub: {str(e)}"
        else:
            result_message = f"Model trained and saved locally to: {output_dir}"
            if merge_adapter:
                result_message += f"\nMerged model saved to: {merged_model_dir}"
        
        # Step 12: Export to GGUF if requested
        if auto_export_gguf:
            try:
                # Set up the GGUF export directory
                gguf_dir = f"{output_dir}_gguf"
                
                # Determine which model to export (merged or adapter)
                model_to_export = merged_model_dir if merge_adapter else output_dir
                
                # Set default quantization if none provided
                if not gguf_quantization:
                    gguf_quantization = ["q8_0"]  # Default to q8_0
                
                # Determine repo ID for GGUF
                gguf_repo_id = ""
                if push_to_hub:
                    # Use the merged repo ID if available, otherwise use the main repo ID
                    if use_dual_repos and merged_repo_id:
                        gguf_repo_id = f"{merged_repo_id}-gguf"
                    else:
                        gguf_repo_id = f"{repo_id}-gguf" if repo_id else ""
                
                # Export to GGUF
                from gguf_utils import export_to_gguf
                gguf_success, gguf_message = export_to_gguf(
                    model_dir=model_to_export,
                    output_dir=gguf_dir,
                    quantization_methods=gguf_quantization,
                    push_to_hub=push_to_hub,
                    repo_id=gguf_repo_id,
                    hf_token=hf_token
                )
                
                # Add GGUF export results to the message
                if gguf_success:
                    result_message += f"\n\n--- GGUF Export Results ---\n{gguf_message}"
                else:
                    result_message += f"\n\nGGUF Export Failed: {gguf_message}"
                    
            except Exception as e:
                result_message += f"\n\nGGUF Export Error: {str(e)}"
        
        # Step 13: Return summary stats
        training_time_minutes = round(trainer_stats.metrics['train_runtime'] / 60, 2)
        result_message += merge_message
        result_message += f"\nTraining completed in {training_time_minutes} minutes."
        result_message += f"\nFinal loss: {trainer_stats.metrics.get('train_loss', 'N/A')}"
        
        success, message = True, result_message
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        success, message = False, f"Error during fine-tuning: {str(e)}\n\nDetails:\n{error_details}"
    
    # Return status and message
    return f"Status: {'Success' if success else 'Failed'}", message

def test_inference(output_dir: str, prompt: str, temperature: float = 0.7, max_new_tokens: int = 512) -> Tuple[bool, str]:
    """
    Test the fine-tuned model with inference.
    
    Args:
        output_dir: Path to the fine-tuned model
        prompt: Prompt for inference
        temperature: Temperature for generation
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Tuple containing success status and response
    """
    try:
        # Load the fine-tuned model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=output_dir,
            max_seq_length=2048,  # This should match the training setting
            dtype=None,  # Auto detection
            load_in_4bit=True
        )
        
        # Enable faster inference
        FastLanguageModel.for_inference(model)
        
        # Format the input
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        
        # Generate the response
        outputs = model.generate(
            input_ids=inputs, 
            max_new_tokens=max_new_tokens, 
            use_cache=True,
            temperature=temperature,
            min_p=0.1
        )
        
        # Decode the output
        response = tokenizer.batch_decode(outputs)[0]
        
        # Extract just the assistant's response
        try:
            # Split the response to get only the assistant's part
            assistant_start = response.find("<|start_header_id|>assistant<|end_header_id|>")
            if assistant_start != -1:
                assistant_response = response[assistant_start:]
                # Clean up the tags
                assistant_response = assistant_response.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
                assistant_response = assistant_response.replace("<|eot_id|>", "")
                return True, assistant_response.strip()
            else:
                return True, response  # Return full response if we can't isolate assistant's part
        except:
            return True, response  # Return full response if parsing fails
        
    except Exception as e:
        return False, f"Error during inference: {str(e)}"

def update_repo_ids(model_name: str, push_to_hub: bool, use_dual_repos: bool) -> Tuple[str, str]:
    if push_to_hub:
        parts = model_name.split('/')
        base_name = parts[-1]
        username = os.getenv('USERNAME') or getpass.getuser()  # Better Windows compatibility
        
        lora_repo = f"lora-{base_name}-{username}"
        merged_repo = f"merged-{base_name}-{username}"
        
        if use_dual_repos:
            return lora_repo, merged_repo
        else:
            return f"finetuned-{base_name}-{username}", ""
    else:
        return "", ""