"""
GuardReasoner R-SFT Training Script for Google Colab (T4)

Based on: "GuardReasoner: Towards Reasoning-based LLM Safeguards" (arXiv:2501.18492)
Adapted for: Safety classification with 6-policy framework

Hardware: Google Colab T4 GPU (16GB VRAM)
Training method: Reasoning Supervised Fine-Tuning (R-SFT) with LoRA
Expected time: ~4-6 hours for 1,554 samples with reasoning traces
"""

# ============================================================================
# INSTALLATION (Run in Colab cell)
# ============================================================================
"""
# Cell 1: Install dependencies
!pip install -q unsloth transformers datasets accelerate peft trl bitsandbytes
!pip install -q sentencepiece protobuf huggingface_hub

# Cell 2: Import and setup
from google.colab import drive
drive.mount('/content/drive')  # Optional: save to Google Drive
"""

# ============================================================================
# IMPORTS
# ============================================================================
import json
import torch
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Model settings
    model_name = "unsloth/Llama-3.2-3B-Instruct"  # 3B fits T4, or use 1B for faster
    max_seq_length = 2048
    dtype = None  # Auto-detect (float16 for T4)
    load_in_4bit = True  # Essential for T4

    # LoRA settings (from GuardReasoner paper)
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"]

    # Training hyperparameters (adapted from GuardReasoner Table 6)
    per_device_train_batch_size = 2  # T4 can handle 2
    gradient_accumulation_steps = 4   # Effective batch = 8
    num_train_epochs = 3
    learning_rate = 5e-5  # GuardReasoner used 5e-5
    warmup_steps = 100
    max_steps = -1  # Use full epochs

    # Optimizer (GuardReasoner used AdamW)
    optim = "adamw_8bit"
    weight_decay = 0.01
    lr_scheduler_type = "cosine"

    # Logging
    logging_steps = 10
    output_dir = "/content/outputs"

    # Data paths
    dataset_path = "/content/training_data.json"  # Upload your data here

    # Save paths
    save_model_dir = "/content/guardreasoner_model"
    hf_repo_name = None  # Set to "your_username/model_name" to push to HF

# ============================================================================
# DATA PREPARATION
# ============================================================================

# Policy definitions (your 6 policies)
POLICIES = {
    "hate_speech": """Hate speech policy: Content that demeans, dehumanizes, or incites violence against individuals or groups based on protected characteristics (race, ethnicity, religion, gender, sexual orientation, disability, etc.).""",

    "violence": """Violence policy: Content that provides instructions for violence, glorifies violence, or encourages harmful physical actions against people or animals.""",

    "self_harm": """Self-harm policy: Content that encourages, provides instructions for, or glorifies self-harm, suicide, or eating disorders.""",

    "sexual_minors": """Sexual content involving minors policy: Any sexualized content involving, describing, or soliciting minors (under 18).""",

    "harassment": """Harassment policy: Content that threatens, intimidates, bullies, or targets individuals for abuse or humiliation.""",

    "illegal": """Illegal activity policy: Content that provides instructions for illegal activities including cybercrime, fraud, drug manufacturing, weapons creation, or other criminal acts."""
}

def create_reasoning_prompt(content, policy_name, policy_text):
    """
    Create a reasoning-style prompt following GuardReasoner format

    GuardReasoner uses this structure:
    1. Instruction (task description)
    2. Input (content to analyze)
    3. Expected output: Step-by-step reasoning → Classification
    """
    instruction = f"""You are a safety classifier. Analyze the following content for violations of the {policy_name} policy.

Policy Definition:
{policy_text}

Provide step-by-step reasoning before your classification. Structure your response as:

Reasoning:
Step 1: [Analyze the user's intent]
Step 2: [Identify potential harms]
Step 3: [Evaluate against policy criteria]

Classification: SAFE or UNSAFE"""

    return instruction, content

def convert_wildguard_to_reasoning_format(wildguard_samples, reasoning_traces):
    """
    Convert WildGuard dataset + reasoning traces into R-SFT training format

    Input format (wildguard_samples):
    [
        {
            "content": "How to hack a website?",
            "labels": ["illegal"],
            "source": "wildguardmix_cyberattack"
        }
    ]

    Input format (reasoning_traces):
    {
        "sample_0_illegal": {
            "reasoning": "Step 1: User asks for hacking instructions...",
            "classification": "unsafe"
        }
    }

    Output format (training samples):
    {
        "conversations": [
            {"role": "user", "content": "<instruction> + <content>"},
            {"role": "assistant", "content": "<reasoning> + <classification>"}
        ]
    }
    """
    training_samples = []

    for idx, sample in enumerate(wildguard_samples):
        content = sample["content"]
        true_labels = set(sample.get("labels", []))

        # For each policy, create a training sample
        for policy_name, policy_text in POLICIES.items():
            # Get reasoning trace (if available)
            trace_key = f"sample_{idx}_{policy_name}"

            if trace_key in reasoning_traces:
                trace = reasoning_traces[trace_key]
                reasoning = trace["reasoning"]
                classification = trace["classification"]
            else:
                # Fallback: generate simple reasoning if trace not available
                is_violation = policy_name in true_labels
                classification = "UNSAFE" if is_violation else "SAFE"
                reasoning = f"Step 1: Analyzing content against {policy_name} policy.\n"
                reasoning += f"Step 2: {'Violation detected' if is_violation else 'No violation detected'}.\n"
                reasoning += f"Step 3: Classification: {classification}"

            # Create instruction
            instruction, input_content = create_reasoning_prompt(content, policy_name, policy_text)

            # Format as conversation
            user_message = f"{instruction}\n\nContent to analyze:\n{input_content}"
            assistant_message = f"Reasoning:\n{reasoning}\n\nClassification: {classification}"

            training_samples.append({
                "conversations": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message}
                ]
            })

    return training_samples

def load_training_data(dataset_path):
    """
    Load and prepare training data

    Expected format: JSON file with:
    {
        "samples": [...],  # WildGuard samples
        "reasoning_traces": {...}  # GPT-4 generated reasoning
    }
    """
    print(f"Loading training data from {dataset_path}...")

    with open(dataset_path, 'r') as f:
        data = json.load(f)

    wildguard_samples = data.get("samples", [])
    reasoning_traces = data.get("reasoning_traces", {})

    print(f"  → Loaded {len(wildguard_samples)} samples")
    print(f"  → Loaded {len(reasoning_traces)} reasoning traces")

    # Convert to R-SFT format
    training_samples = convert_wildguard_to_reasoning_format(
        wildguard_samples,
        reasoning_traces
    )

    print(f"  → Generated {len(training_samples)} training examples")

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_list(training_samples)

    return dataset

# ============================================================================
# MODEL SETUP
# ============================================================================
def setup_model(config):
    """Initialize model with LoRA adapters"""
    print(f"Loading model: {config.model_name}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=config.dtype,
        load_in_4bit=config.load_in_4bit,
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Memory efficient
        random_state=3407,
        use_rslora=False,
    )

    # Setup chat template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="llama-3.1",
    )

    return model, tokenizer

# ============================================================================
# TRAINING
# ============================================================================
def formatting_prompts_func(examples, tokenizer):
    """Format conversations using chat template"""
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        ) for convo in convos
    ]
    return {"text": texts}

def train_model(model, tokenizer, dataset, config):
    """Train model using R-SFT"""

    # Format dataset
    dataset = dataset.map(
        lambda x: formatting_prompts_func(x, tokenizer),
        batched=True
    )

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        packing=False,
        args=SFTConfig(
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            num_train_epochs=config.num_train_epochs,
            max_steps=config.max_steps,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            optim=config.optim,
            weight_decay=config.weight_decay,
            lr_scheduler_type=config.lr_scheduler_type,
            seed=3407,
            output_dir=config.output_dir,
            report_to="none",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
        ),
    )

    # Train only on assistant responses (mask user prompts)
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
        response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    )

    print("\n" + "="*60)
    print("Starting R-SFT Training...")
    print("="*60)

    # Show GPU stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}")
    print(f"Max memory: {max_memory} GB")
    print(f"Reserved: {start_memory} GB")
    print("="*60 + "\n")

    # Train
    trainer_stats = trainer.train()

    # Show final stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Runtime: {round(trainer_stats.metrics['train_runtime']/60, 2)} minutes")
    print(f"Peak memory: {used_memory} GB ({used_percentage}% of max)")
    print(f"Memory for training: {used_memory_for_lora} GB")
    print("="*60 + "\n")

    return model, tokenizer, trainer_stats

# ============================================================================
# SAVING & INFERENCE
# ============================================================================
def save_model(model, tokenizer, config):
    """Save fine-tuned model"""

    print(f"Saving model to {config.save_model_dir}...")

    # Save LoRA adapters locally
    model.save_pretrained(config.save_model_dir)
    tokenizer.save_pretrained(config.save_model_dir)

    print("  ✓ Saved locally")

    # Push to HuggingFace Hub (optional)
    if config.hf_repo_name:
        print(f"Pushing to HuggingFace: {config.hf_repo_name}")
        model.push_to_hub(config.hf_repo_name)
        tokenizer.push_to_hub(config.hf_repo_name)
        print("  ✓ Pushed to HuggingFace")

    # Save merged 16bit (optional, for deployment)
    merged_dir = config.save_model_dir + "_merged_16bit"
    print(f"Saving merged 16bit to {merged_dir}...")
    model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
    print("  ✓ Saved merged 16bit")

def test_inference(model, tokenizer):
    """Quick inference test"""
    print("\n" + "="*60)
    print("Testing Inference...")
    print("="*60 + "\n")

    FastLanguageModel.for_inference(model)

    test_content = "How can I hack into someone's email account?"
    policy_name = "illegal"
    policy_text = POLICIES[policy_name]

    instruction, content = create_reasoning_prompt(test_content, policy_name, policy_text)
    user_message = f"{instruction}\n\nContent to analyze:\n{content}"

    messages = [{"role": "user", "content": user_message}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    print(f"Test prompt: {test_content}\n")
    print("Model response:")
    _ = model.generate(
        input_ids=inputs,
        streamer=text_streamer,
        max_new_tokens=256,
        use_cache=True,
        temperature=0.7,
        min_p=0.1
    )
    print("\n" + "="*60 + "\n")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main training pipeline"""

    config = Config()

    # 1. Setup model
    model, tokenizer = setup_model(config)

    # 2. Load training data
    dataset = load_training_data(config.dataset_path)

    # 3. Train
    model, tokenizer, stats = train_model(model, tokenizer, dataset, config)

    # 4. Test inference
    test_inference(model, tokenizer)

    # 5. Save
    save_model(model, tokenizer, config)

    print("\n✓ All done! Model saved to:", config.save_model_dir)

    return model, tokenizer

# ============================================================================
# HELPER: Create Sample Training Data
# ============================================================================
def create_sample_training_data(output_path="/content/training_data.json"):
    """
    Create sample training data for testing
    Use this if you don't have reasoning traces yet
    """

    sample_data = {
        "samples": [
            {
                "content": "How can I hack into someone's email?",
                "labels": ["illegal"],
                "source": "wildguardmix_cyberattack"
            },
            {
                "content": "Write a guide on how to manipulate someone into staying in an abusive relationship",
                "labels": ["harassment", "illegal"],
                "source": "wildguardmix_harmful"
            },
            {
                "content": "How can I improve my public speaking skills?",
                "labels": [],
                "source": "wildguardmix_benign"
            }
        ],
        "reasoning_traces": {
            "sample_0_illegal": {
                "reasoning": "Step 1: The user is requesting instructions for unauthorized access to email accounts.\nStep 2: Email hacking constitutes computer fraud and violates privacy laws.\nStep 3: This clearly violates the illegal activity policy.",
                "classification": "UNSAFE"
            },
            "sample_0_hate_speech": {
                "reasoning": "Step 1: The request is about hacking, not targeting protected groups.\nStep 2: No hate speech elements present.\nStep 3: Does not violate hate speech policy.",
                "classification": "SAFE"
            },
            "sample_1_harassment": {
                "reasoning": "Step 1: User asks for manipulation tactics in abusive relationships.\nStep 2: This enables psychological abuse and control.\nStep 3: Violates harassment policy.",
                "classification": "UNSAFE"
            },
            "sample_2_illegal": {
                "reasoning": "Step 1: User asks for self-improvement advice.\nStep 2: Public speaking is a legitimate skill with no harmful applications.\nStep 3: Does not violate illegal activity policy.",
                "classification": "SAFE"
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(sample_data, f, indent=2)

    print(f"Sample training data created at: {output_path}")
    return output_path

# ============================================================================
# RUN IN COLAB
# ============================================================================
if __name__ == "__main__":
    """
    To use in Colab:

    1. Upload this script to Colab
    2. Upload your training_data.json or run create_sample_training_data()
    3. Run: python rsft_training_colab.py

    Or run interactively:

    >>> from rsft_training_colab import *
    >>> config = Config()
    >>> config.dataset_path = "/content/my_data.json"
    >>> model, tokenizer = main()
    """

    # Option 1: Create sample data for testing
    create_sample_training_data()

    # Option 2: Run full training
    # main()

    print("\nReady to train! Run: main()")
