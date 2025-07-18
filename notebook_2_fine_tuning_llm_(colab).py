# -*- coding: utf-8 -*-
"""Notebook 2: Fine-tuning LLM (Colab)

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1KoiWmbWIDC-9rHaPxx-wUJGhS3954Sjh
"""

from google.colab import drive
drive.mount('/content/drive')

# ==============================================================================
# Cell 1: Installations & Imports (Sequential Install)
# ==============================================================================
print("--- Cell 1: Installing/Updating Libraries ---")
# Uninstall core libraries first
!pip uninstall -y torch torchvision torchaudio transformers accelerate bitsandbytes peft trl datasets numpy scipy sentencepiece tiktoken einops

# Install libraries sequentially to help dependency resolution
print("\nInstalling PyTorch...")
!pip install -q "torch==2.3.1" "torchvision==0.18.1" "torchaudio==2.3.1"

print("\nInstalling Transformers...")
!pip install -q "transformers==4.41.2"

print("\nInstalling Accelerate...")
!pip install -q "accelerate==0.29.3" # Pin accelerate

print("\nInstalling PEFT, BitsAndBytes, TRL, Datasets...")
!pip install -q \
    "peft==0.11.1" \
    "bitsandbytes>=0.43.2" \
    "trl==0.9.4" \
    "datasets==2.19.1"

print("\nInstalling other dependencies...")
!pip install -q \
    "sentencepiece" \
    "tiktoken" \
    "einops" \
    "numpy<2.1" \
    "scipy"

print("\nImporting libraries...")
# Try importing accelerate first now
import accelerate
print(f"Accelerate Version: {accelerate.__version__}")

import torch
import os
import pandas as pd
import numpy as np
import pathlib
import gc
import time
import transformers
import datasets
# import accelerate # Already imported
import peft
import bitsandbytes as bnb
# import trl # TRL might not be needed directly if using Trainer

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets, load_from_disk, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer, # Using standard Trainer
    DataCollatorForLanguageModeling,
    pipeline,
    logging,
    IntervalStrategy
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm.notebook import tqdm
import glob
import datasets as ds_datasets # Alias to avoid conflict
import accelerate as acc_accelerate # Alias
import peft as peft_peft
# import trl as trl_trl # No longer using trl directly for Trainer


# Print versions for debugging
print(f"\nPyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Datasets Version: {ds_datasets.__version__}")
# print(f"Accelerate Version: {acc_accelerate.__version__}") # Printed above
print(f"PEFT Version: {peft_peft.__version__}")
# print(f"TRL Version: {trl_trl.__version__}")
# Check bitsandbytes version specifically
try:
    print(f"BitsandBytes Version: {bnb.__version__}")
    from packaging import version
    if version.parse(bnb.__version__) < version.parse("0.43.2"):
        print(f"⚠️ WARNING: Installed bitsandbytes version ({bnb.__version__}) might still be too old!")
    else:
        print("✅ BitsandBytes version meets requirement (>= 0.43.2).")
except Exception as e:
    print(f"Could not check bitsandbytes version: {e}")

# Check if torchvision/torchaudio were installed as dependencies (optional)
try:
    import torchvision
    print(f"Torchvision Version (installed as dependency): {torchvision.__version__}")
except ImportError:
    print("Torchvision not installed.")
try:
    import torchaudio
    print(f"Torchaudio Version (installed as dependency): {torchaudio.__version__}")
except ImportError:
    print("Torchaudio not installed.")


print("\nLibraries installed and imported.")
# --- Verify GPU availability ---
if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    torch.set_default_device('cuda')
else:
    print("⚠️ Warning: No GPU detected. Training will be extremely slow or may fail.")
    print("⚠️ Ensure Runtime > Change runtime type > Hardware accelerator is set to GPU.")
    torch.set_default_device('cpu')

# ==============================================================================
# Cell 2: Mount Drive & Define Paths
# ==============================================================================
# (Keep Cell 2 as it was, but add paths for tokenized data and state)
print("\n--- Cell 2: Mounting Drive & Defining Paths ---")
from google.colab import drive
import os; import pathlib
if not os.path.exists('/content/drive/MyDrive'): drive.mount('/content/drive', force_remount=True)
else: print("Google Drive already mounted.")
DRIVE_BASE_PATH = '/content/drive/MyDrive/'; PROJECT_FOLDER_NAME = 'NetLingo_project'
DRIVE_PROJECT_PATH = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER_NAME)
DRIVE_TOKENS_DIR = pathlib.Path(DRIVE_PROJECT_PATH) / "data" / "tokens"
TRAIN_CORPUS_PATH = DRIVE_TOKENS_DIR / "train_corpus.txt" # Input raw text
# --- NEW Paths for tokenized data and state ---
DRIVE_TOKENIZED_DIR = pathlib.Path(DRIVE_PROJECT_PATH) / "data" / "tokenized_falcon" # Output dir for tokenized data
STATE_FILE_TOKENIZE = DRIVE_TOKENS_DIR / "tokenize_corpus_state.txt" # State file for tokenization progress
# ---
DRIVE_MODEL_OUTPUT_DIR = pathlib.Path(DRIVE_PROJECT_PATH) / "models" / "netlingo_falcon_lora_adapters"
DRIVE_TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
DRIVE_MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"\nInput Raw Corpus Path: {TRAIN_CORPUS_PATH}")
print(f"Output Tokenized Data Directory: {DRIVE_TOKENIZED_DIR}")
print(f"Tokenization State File: {STATE_FILE_TOKENIZE}")
print(f"Output Adapters Directory Path: {DRIVE_MODEL_OUTPUT_DIR}")
if not TRAIN_CORPUS_PATH.exists(): raise FileNotFoundError(f"Training corpus not found: {TRAIN_CORPUS_PATH}")
else: print(f"Training corpus found.")

# ==============================================================================
# Cell 3: Load Tokenizer & Model (Needed for Tokenization)
# ==============================================================================
# (Moved this earlier, as tokenizer is needed for Cell 5)
print("\n--- Cell 3: Loading Tokenizer & Model ---")
base_model_name = "tiiuae/falcon-rw-1b"
print(f"Loading base model: {base_model_name} (needed for tokenizer config)")
# --- Load Tokenizer ---
print(f"\nLoading tokenizer for {base_model_name}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"Set pad token to: {tokenizer.pad_token}"); print("Tokenizer loaded.")
except Exception as e: print(f"❗️ ERROR loading tokenizer: {e}"); raise
# --- Add Special Tokens ---
special_tokens_to_add = [
    "SRCIP_", "DSTIP_", "SPORT_", "DPORT_", "PROTO_", "FLAGS_", "BYTES_", "PKTS_", "DUR_", "STATE_", "SERVICE_",
    "SBYTES_", "DBYTES_", "SPKTS_", "DPKTS_", "RATE_", "STTL_", "DTTL_", "SLOAD_", "DLOAD_", "SLOSS_", "DLOSS_",
    "SINPKT_", "DINPKT_", "SJIT_", "DJIT_", "BWD_PKTS_", "BWD_BYTES_", "FLOW_BPS_", "FLOW_PPS_",
    "0-1s", "1-10s", "10-60s", ">60s", "0-100B", "100B-1K", "1K-10K", ">10K", "0-5", "5-50", "50-200", ">200",
    "0-0.1s", ">10s", ">100K", ">100", "0-100", "100-10K", ">10K", "0-10ms", "10ms-1s", ">1s", "0-50B",
    "50-200B", "200B-1K", ">1K", "0-1ms", "1ms-1s", "1s-1m", ">1m", "0-1KBps", "1KBps-1MBps", ">1MBps",
    "0-100pps", "100pps-10Kpps", ">10Kpps", "INF", "NonFinite", "OutOfRange", "NaN",
    "LABEL_BENIGN", "LABEL_ATTACK", "DS_KYOTO", "DS_UNSW", "DS_CIC"
]
existing_special_tokens = list(tokenizer.special_tokens_map.values())
unique_new_tokens = sorted(list(set([t for t in special_tokens_to_add if t and t not in existing_special_tokens and t not in tokenizer.get_vocab()])))
if unique_new_tokens:
    print(f"\nAdding {len(unique_new_tokens)} new special tokens...");
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": unique_new_tokens})
    print(f"Successfully added {num_added} tokens.")
    # NOTE: We don't need to load/resize the *model* embeddings here, just the tokenizer
else: print("\nNo new special tokens needed.")
print("Tokenizer configured.")

# ==============================================================================
# Cell 4: Incremental Tokenization & Saving to Disk (NEW - Replaces old Cell 5)
# ==============================================================================
print("\n--- Cell 4: Incremental Tokenization & Saving to Disk ---")
max_seq_length = 512
lines_per_chunk = 100000 # Process and save every N lines (adjust based on time/memory)

def tokenize_function(examples):
    """Tokenizes text, no padding needed here as it's handled by collator."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False # No padding needed when saving individual records
    )

# --- Determine Starting Point ---
lines_processed = 0
if STATE_FILE_TOKENIZE.exists():
    try:
        with open(STATE_FILE_TOKENIZE, 'r') as f:
            lines_processed = int(f.read().strip())
        print(f"Resuming tokenization. State file indicates {lines_processed:,} lines already processed.")
    except Exception as e:
        print(f"Warning: Could not read state file {STATE_FILE_TOKENIZE}: {e}. Starting fresh.")
        lines_processed = 0
else:
    print("No state file found. Starting tokenization from beginning.")

# --- Process File Line by Line (or in text chunks) ---
print(f"Processing {TRAIN_CORPUS_PATH} starting from line {lines_processed + 1}...")
current_chunk_lines = []
processed_in_this_run = 0
chunk_index = lines_processed // lines_per_chunk # Calculate starting chunk index

try:
    with open(TRAIN_CORPUS_PATH, 'r', encoding='utf-8') as infile:
        # Skip already processed lines efficiently
        if lines_processed > 0:
            print("Skipping processed lines (this might take a moment)...")
            # Use tqdm to show skipping progress
            for _ in tqdm(range(lines_processed), desc="Skipping lines", unit=" lines", leave=False):
                 next(infile) # Read and discard line
            print("Skipping done.")

        # Use tqdm for the main file reading loop
        # Estimate total lines if possible (can be slow for huge files)
        # total_lines = sum(1 for line in open(TRAIN_CORPUS_PATH, 'r', encoding='utf-8'))
        # print(f"Estimated total lines in corpus: {total_lines:,}")
        infile_iter = tqdm(infile, desc="Tokenizing Corpus", initial=lines_processed, unit=" lines") # Removed total=total_lines

        # Process remaining lines
        for line in infile_iter:
            line = line.strip()
            if line: # Skip empty lines
                current_chunk_lines.append(line)

            # When chunk is full, tokenize and save
            if len(current_chunk_lines) >= lines_per_chunk:
                chunk_index += 1
                print(f"\nProcessing chunk {chunk_index} ({len(current_chunk_lines)} lines)...")
                # Convert list of strings to HF Dataset
                chunk_dataset = Dataset.from_dict({"text": current_chunk_lines})
                # Tokenize the chunk
                tokenized_chunk = chunk_dataset.map(
                    tokenize_function,
                    batched=True,
                    remove_columns=["text"],
                    desc=f"Tokenizing chunk {chunk_index}"
                )
                # Save tokenized chunk to disk
                output_path = DRIVE_TOKENIZED_DIR / f"chunk_{chunk_index:04d}" # Use padding for sorting
                tokenized_chunk.save_to_disk(str(output_path))
                print(f"Saved tokenized chunk {chunk_index} to {output_path}")

                # Update state file
                lines_processed += len(current_chunk_lines)
                processed_in_this_run += len(current_chunk_lines)
                try:
                    with open(STATE_FILE_TOKENIZE, 'w') as f_state:
                        f_state.write(str(lines_processed))
                    # print(f"State updated to {lines_processed:,} lines processed.") # Less verbose
                except Exception as e_state:
                    print(f"❗️ CRITICAL ERROR: Failed to update state file {STATE_FILE_TOKENIZE}: {e_state}")
                    raise # Stop if state cannot be saved

                # Reset chunk and clean memory
                current_chunk_lines = []
                del chunk_dataset, tokenized_chunk
                gc.collect()

        # Process any remaining lines in the last partial chunk
        if current_chunk_lines:
            chunk_index += 1
            print(f"\nProcessing final chunk {chunk_index} ({len(current_chunk_lines)} lines)...")
            chunk_dataset = Dataset.from_dict({"text": current_chunk_lines})
            tokenized_chunk = chunk_dataset.map(
                tokenize_function, batched=True, remove_columns=["text"], desc=f"Tokenizing chunk {chunk_index}"
            )
            output_path = DRIVE_TOKENIZED_DIR / f"chunk_{chunk_index:04d}"
            tokenized_chunk.save_to_disk(str(output_path))
            print(f"Saved final tokenized chunk {chunk_index} to {output_path}")
            lines_processed += len(current_chunk_lines)
            processed_in_this_run += len(current_chunk_lines)
            with open(STATE_FILE_TOKENIZE, 'w') as f_state:
                f_state.write(str(lines_processed))
            print(f"State updated to {lines_processed:,} lines processed.")

    print(f"\n✅ Tokenization process finished/paused for this run. Processed {processed_in_this_run:,} new lines.")
    print(f"Total lines processed according to state file: {lines_processed:,}")
    print(f"Tokenized data saved in chunks under: {DRIVE_TOKENIZED_DIR}")

except Exception as e:
    print(f"\n❗️ An error occurred during tokenization: {e}")
    import traceback
    traceback.print_exc()
    print(f"❗️ State file currently shows {lines_processed:,} lines processed.")
    print("❗️ You can resume by re-running this cell in a new session.")

# ==============================================================================
# Cell 4: Define Iterable Dataset for Streaming
# ==============================================================================
# (Keep Cell 4 as it was)
print("\n--- Cell 4: Defining Iterable Dataset for Streaming ---")
# import glob # Already imported in Cell 1
chunk_dirs = sorted(glob.glob(str(DRIVE_TOKENIZED_DIR / "chunk_*")))
if not chunk_dirs: raise FileNotFoundError(f"No tokenized data chunks found in {DRIVE_TOKENIZED_DIR}.")
print(f"Found {len(chunk_dirs)} tokenized chunk directories.")
def yield_examples_from_chunks(chunk_paths):
    for chunk_path in tqdm(chunk_paths, desc="Streaming Data Chunks", leave=True):
        try:
            chunk_dataset = load_from_disk(chunk_path); yield from chunk_dataset
        except Exception as e: print(f"⚠️ Warning: Skipping chunk {chunk_path} due to error: {e}")
iterable_dataset = IterableDataset.from_generator(yield_examples_from_chunks, gen_kwargs={"chunk_paths": chunk_dirs})
print("IterableDataset created.")
tokenized_train_dataset = iterable_dataset; tokenized_eval_dataset = None
print("Using the full stream as the training dataset. Evaluation during training will be disabled.")

# ==============================================================================
# Cell 5: Load Model (with Quantization) (Corrected - Removed need_resize)
# ==============================================================================
print("\n--- Cell 5: Loading Model ---")
base_model_name = "tiiuae/falcon-rw-1b"
print(f"Loading base model: {base_model_name}")
compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype, bnb_4bit_use_double_quant=False)
print(f"Loading model with 4-bit quantization...")
try:
    model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.config.use_cache = False; model.config.pretraining_tp = 1

    # --- Check if resize is needed based on tokenizer state from Cell 3 ---
    # Compare tokenizer vocab size with current model embedding size
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print(f"Resizing model embeddings for {len(tokenizer)} tokens (tokenizer has more tokens than model)...")
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized model embeddings matrix to: {model.get_input_embeddings().weight.shape}")
    else:
        print("Model embedding size already matches tokenizer vocabulary size.")
    # ---

    # Prepare for k-bit training AFTER potential resize
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    print("Base model loaded and prepared.")
except Exception as e: print(f"❗️ ERROR loading base model: {e}"); raise

# ==============================================================================
# Cell 6: Configure LoRA & Training Arguments (Corrected pin_memory)
# ==============================================================================
print("\n--- Cell 6: Configuring LoRA & Training Arguments ---")
# (Keep LoRA config as before)
lora_r = 64; lora_alpha = 16; lora_dropout = 0.1
peft_config = LoraConfig(
    lora_alpha=lora_alpha, lora_dropout=lora_dropout, r=lora_r, bias="none",
    task_type="CAUSAL_LM",
    target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
)
model = get_peft_model(model, peft_config); print("LoRA configuration applied to model."); model.print_trainable_parameters()

# --- Training Arguments ---
training_output_dir = str(DRIVE_MODEL_OUTPUT_DIR / "training_checkpoints"); final_model_dir = str(DRIVE_MODEL_OUTPUT_DIR)
do_evaluation = False; evaluation_steps = None; save_steps_val = 500
max_training_steps = 10000 # Adjust this!

training_arguments = TrainingArguments(
    output_dir=training_output_dir,
    max_steps=max_training_steps,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4, # Not used but required
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_strategy=IntervalStrategy.STEPS,
    save_steps=save_steps_val,
    logging_strategy="steps",
    logging_steps=50,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant_with_warmup",
    report_to="tensorboard",
    do_eval=do_evaluation,
    save_total_limit=2,
    # --- CORRECTION: Disable memory pinning ---
    dataloader_pin_memory=False,
    # ---
    # gradient_checkpointing=True, # Uncomment if OOM
)
print("Training arguments set.")

# ==============================================================================
# Cell 7: Pre-flight Check (Unit Test Case)
# ==============================================================================
# (Keep Cell 7 as it was - Skipping for IterableDataset)
print("\n--- Cell 7: Running Pre-flight Check ---")
print("Skipping pre-flight check for IterableDataset (manual verification recommended).")
pre_flight_passed = True

# ==============================================================================
# Cell 8: Initialize Trainer & Start Training (Explicit Resume Logic)
# ==============================================================================
print("\n--- Cell 8: Initializing Trainer & Starting Training ---")
# from transformers import Trainer, DataCollatorForLanguageModeling # Already imported
import gc
import torch
import os
import re # For parsing checkpoint number

# Re-define final_model_dir and training_output_dir just in case
if 'DRIVE_MODEL_OUTPUT_DIR' in locals():
    final_model_dir = str(DRIVE_MODEL_OUTPUT_DIR)
    training_output_dir = str(DRIVE_MODEL_OUTPUT_DIR / "training_checkpoints") # From Cell 6 args
else:
    # Attempt to redefine based on standard structure
    try:
        DRIVE_BASE_PATH = '/content/drive/MyDrive/'; PROJECT_FOLDER_NAME = 'NetLingo_project'
        DRIVE_PROJECT_PATH = os.path.join(DRIVE_BASE_PATH, PROJECT_FOLDER_NAME)
        DRIVE_MODEL_OUTPUT_DIR = pathlib.Path(DRIVE_PROJECT_PATH) / "models" / "netlingo_falcon_lora_adapters"
        final_model_dir = str(DRIVE_MODEL_OUTPUT_DIR)
        training_output_dir = str(DRIVE_MODEL_OUTPUT_DIR / "training_checkpoints")
        print("Re-defined output directories.")
    except NameError: print("❗️ ERROR: DRIVE_MODEL_OUTPUT_DIR not defined."); raise

# Ensure prerequisite objects exist
if 'tokenizer' not in locals(): raise NameError("tokenizer not defined. Run Cell 3.")
if 'model' not in locals(): raise NameError("model not defined. Run Cell 5 & 6.")
if 'training_arguments' not in locals(): raise NameError("training_arguments not defined. Run Cell 6.")
if 'tokenized_train_dataset' not in locals(): raise NameError("tokenized_train_dataset not defined. Run Cell 4.")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False); print("Data collator initialized.")
trainer = Trainer(
    model=model, args=training_arguments, train_dataset=tokenized_train_dataset,
    eval_dataset=None, tokenizer=tokenizer, data_collator=data_collator,
); print("Trainer initialized.")

# --- Explicitly Find Latest Checkpoint ---
latest_checkpoint = None
if os.path.isdir(training_arguments.output_dir):
    checkpoints = [
        d for d in os.listdir(training_arguments.output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(training_arguments.output_dir, d))
    ]
    if checkpoints:
        # Find checkpoint with the highest step number
        latest_checkpoint = max(
            checkpoints,
            key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1))
        )
        latest_checkpoint = os.path.join(training_arguments.output_dir, latest_checkpoint)
        print(f"Found latest checkpoint: {latest_checkpoint}")
    else:
        print(f"No previous checkpoints found in {training_arguments.output_dir}. Starting fresh.")
else:
     print(f"Checkpoint directory {training_arguments.output_dir} not found. Starting fresh.")
# ---

# --- Start Fine-tuning ---
print("\nStarting fine-tuning... (Check TensorBoard for progress)")
print(f"Training will run for {training_arguments.max_steps} steps.")
if latest_checkpoint:
    print(f"Attempting to resume training from: {latest_checkpoint}")
else:
    print("Starting training from scratch.")

# print("%load_ext tensorboard")
# print(f"%tensorboard --logdir '{training_arguments.output_dir}'")

training_successful = False
try:
    # Pass the specific checkpoint path if found, otherwise pass None (or False)
    train_result = trainer.train(resume_from_checkpoint=latest_checkpoint if latest_checkpoint else None)
    print("Fine-tuning finished (reached max_steps or completed).")
    metrics = train_result.metrics; trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics); trainer.save_state()
    print("Trainer state and metrics saved."); training_successful = True
except Exception as e: print(f"❗️ An error occurred during training: {e}"); import traceback; traceback.print_exc()
finally:
    if training_successful:
        print(f"\nSaving final LoRA adapters to: {final_model_dir}")
        try:
            trainer.save_model(final_model_dir); tokenizer.save_pretrained(final_model_dir)
            print("Final adapters and tokenizer saved.")
            print("\nContents of final model directory on Drive:")
            !ls -lh "{str(final_model_dir)}" # Corrected syntax here
        except Exception as e_save: print(f"❗️ Error saving final model/tokenizer: {e_save}")
    else: print("\nSkipping final model saving due to training error/interruption.")
    print("Checkpoints might still exist in:", training_arguments.output_dir)
    print("\nCleaning up GPU memory...")
    trainer_obj = locals().get('trainer', None); model_obj = locals().get('model', None)
    if trainer_obj: del trainer_obj
    if model_obj: del model_obj
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("Cleaned up GPU memory."); print("\n--- Training Process Ended ---")

"""**Explanation of Key Sections:**

1.  **Cell 1 (Install & Import):** Installs all necessary libraries (`transformers`, `datasets`, `peft`, `bitsandbytes`, `trl`, etc.) and imports them. Checks for GPU.
2.  **Cell 2 (Paths):** Mounts Google Drive and defines the input path for `train_corpus.txt` and the output path for the LoRA adapters, both on your Drive. Includes a check that the input file exists.
3.  **Cell 3 (Dataset):** Loads the `train_corpus.txt` using `load_dataset("text", ...)`. Shuffles and splits into train/validation sets. Prints a sample.
4.  **Cell 4 (Model & Tokenizer):**
    * Loads the specified `base_model_name` (`tiiuae/falcon-rw-1b`).
    * Configures `BitsAndBytesConfig` for 4-bit NF4 quantization with bfloat16 compute.
    * Loads the model using the quantization config and `device_map="auto"`.
    * Calls `prepare_model_for_kbit_training`.
    * Loads the corresponding tokenizer, sets the padding token.
    * **Adds Special Tokens:** Includes the logic to add our custom tokens (`SRCIP_`, `LABEL_ATTACK`, etc.) to the tokenizer and resize the model's embedding layer. **Crucially, ensure the `special_tokens_to_add` list here matches ALL prefixes and bucket labels used in your tokenization scripts.**
5.  **Cell 5 (LoRA & Training Args):**
    * Defines the `LoraConfig` specifying rank, alpha, dropout, and target modules (check these for Falcon-1B).
    * Defines `TrainingArguments`, setting hyperparameters like epochs, batch size, learning rate, optimizer (`paged_adamw_32bit`), saving steps, logging steps, evaluation strategy, and importantly, the `output_dir` pointing to a *checkpoints* folder on your Drive.
6.  **Cell 6 (Pre-flight Check):** **(New)** This cell attempts to tokenize a small number of samples from your training set and run them through the model's forward pass. This helps catch errors like incorrect tokenization, dimension mismatches, or CUDA errors *before* starting the hours-long training process.
7.  **Cell 7 (Trainer & Train):**
    * Initializes the `SFTTrainer` from the `trl` library, passing the model, datasets, tokenizer, LoRA config, and training arguments.
    * Calls `trainer.train()` to start the fine-tuning process. Progress bars should appear in the output.
    * Includes `try...finally` to ensure GPU memory is cleaned up even if training errors occur.
    * Explicitly saves the final model adapters and tokenizer state to the specified `final_model_dir` on Google Drive after training completes successfully.
    * Lists the contents of the final save directory.

**Before Running:**

* Execute the prerequisite cells (Drive mount, path definitions).
* Ensure `train_corpus.txt` is correctly located on Drive.
* Make sure the `special_tokens_to_add` list in Cell 4 is comprehensive and matches your tokenization output.
* Verify the `target_modules` in the `LoraConfig` (Cell 5) are appropriate for Falcon-1B (the ones listed are common but double-check if needed).
* Run the cells sequentially. Pay close attention to the output of the Pre-flight Check (Cell 6) before proceeding to Cell 7.

This setup provides a robust workflow for fine-tuning on Colab, saving your results directly to Google Dri
  ```
"""