#!/usr/bin/env python3
"""
Experiment 21: L0 Bouncer - DeBERTa Distillation

Train a fast DeBERTa classifier to act as "bouncer" for the Safety Cascade.
Uses Exp 18 model as teacher to generate soft labels.

Architecture:
- L0 (This): DeBERTa-v3-xsmall, handles 70% traffic at 1ms
- L1: Our 3B R-SFT model, handles 25% at 50ms
- L2: 6-policy gauntlet, handles 4% at 200ms
- L3: GPT-4o async audit, handles 1%

Expected Results:
- Accuracy: 85-90% (vs teacher's 95%)
- Speed: <5ms (vs teacher's 50-100ms)
- Traffic filtered: 70% with high confidence

Usage:
    # Generate teacher labels first
    python train_l0_bouncer.py --generate-labels

    # Train bouncer
    python train_l0_bouncer.py --train

    # Full pipeline
    python train_l0_bouncer.py
"""

import os
import json
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import torch.nn as nn

# Configuration
TEACHER_MODEL = "./models/exp_18_batch8_lora"
TEACHER_BASE = "unsloth/Llama-3.2-3B-Instruct"
STUDENT_MODEL = "microsoft/deberta-v3-xsmall"  # 22M params, very fast
OUTPUT_DIR = "./models/l0_bouncer"
DATA_DIR = "./data"

# Training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
EPOCHS = 10  # Increased from 3 for small dataset
MAX_LENGTH = 256  # Short for speed


class SafetyDataset(Dataset):
    """Dataset for safety classification."""

    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Tokenize
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Add label
        encoding["labels"] = item["label"]

        # Note: soft_labels not used - DeBERTa doesn't support them directly
        # Future: could implement knowledge distillation with KL divergence loss

        return encoding


class WeightedTrainer(Trainer):
    """Trainer with class-weighted loss for imbalanced datasets."""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = torch.tensor(self.class_weights, dtype=torch.float32).to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_training_data():
    """Load training data from our dataset."""

    # Try to load from combined dataset
    data_path = Path(DATA_DIR) / "combined_train.json"

    if not data_path.exists():
        # Fall back to experiments directory
        data_path = Path("../combined_train.json")

    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found at {data_path}")

    with open(data_path) as f:
        raw_data = json.load(f)

    # Convert to format needed
    data = []
    for item in raw_data:
        text = item.get("prompt", item.get("text", ""))
        label_str = item.get("label", "safe").lower()
        label = 1 if label_str == "harmful" else 0

        data.append({
            "text": text,
            "label": label,
        })

    print(f"Loaded {len(data)} training samples")
    return data


def generate_teacher_labels(data, output_path="teacher_labels.json"):
    """Generate soft labels from teacher model (Exp 18)."""

    print("=" * 60)
    print("Generating Teacher Labels")
    print("=" * 60)

    # Load teacher model
    print(f"Loading teacher: {TEACHER_MODEL}")

    base_tokenizer = AutoTokenizer.from_pretrained(TEACHER_BASE)
    base_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_BASE,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )

    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, TEACHER_MODEL)
    model.eval()

    # System prompt
    system_prompt = """You are a content moderation expert. Analyze the user's request and determine if it is safe or harmful.

Provide step-by-step reasoning, then give your final classification."""

    # Generate labels
    labeled_data = []

    for i, item in enumerate(tqdm(data, desc="Generating labels")):
        # Format prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["text"]}
        ]

        input_text = base_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = base_tokenizer(input_text, return_tensors="pt").to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                pad_token_id=base_tokenizer.eos_token_id,
            )

        response = base_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Parse prediction
        response_lower = response.lower()

        # Look for conclusion
        if "therefore, the content is safe" in response_lower or \
           "the content is safe" in response_lower:
            pred_label = 0  # safe
            confidence = 0.9
        elif "therefore, the content is harmful" in response_lower or \
             "the content is harmful" in response_lower:
            pred_label = 1  # harmful
            confidence = 0.9
        else:
            # Count occurrences
            safe_count = response_lower.count("safe") + response_lower.count("benign")
            harmful_count = response_lower.count("harmful") + response_lower.count("unsafe")

            if harmful_count > safe_count:
                pred_label = 1
                confidence = 0.7
            else:
                pred_label = 0
                confidence = 0.7

        # Create soft label
        if pred_label == 1:
            soft_label = [1 - confidence, confidence]
        else:
            soft_label = [confidence, 1 - confidence]

        labeled_data.append({
            "text": item["text"],
            "label": pred_label,
            "soft_label": soft_label,
            "original_label": item["label"],
        })

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)}")

    # Save
    with open(output_path, "w") as f:
        json.dump(labeled_data, f, indent=2)

    print(f"\nSaved {len(labeled_data)} labeled samples to {output_path}")

    # Stats
    teacher_correct = sum(1 for d in labeled_data if d["label"] == d["original_label"])
    print(f"Teacher accuracy: {teacher_correct/len(labeled_data)*100:.1f}%")

    return labeled_data


def train_bouncer(data):
    """Train DeBERTa bouncer on teacher labels."""

    print("=" * 60)
    print("Training L0 Bouncer")
    print("=" * 60)

    # Load tokenizer and model
    print(f"Loading student model: {STUDENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL,
        num_labels=2,
        problem_type="single_label_classification",
    )

    # Prepare dataset
    dataset = SafetyDataset(data, tokenizer, MAX_LENGTH)

    # Split train/eval
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    print(f"Train: {train_size}, Eval: {eval_size}")

    # Calculate class weights for imbalanced data
    labels = np.array([d["label"] for d in data])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=labels
    )
    # Boost harmful weight for better recall (safety-first approach)
    class_weights[1] *= 1.5  # Increase harmful weight by 50%
    print(f"\nClass weights: safe={class_weights[0]:.3f}, harmful={class_weights[1]:.3f} (boosted)")

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        fp16=True,
    )

    # Metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", pos_label=1
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    # Trainer with class weights
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\nModel saved to {OUTPUT_DIR}")

    # Final evaluation
    results = trainer.evaluate()
    print("\nFinal Results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    return model, tokenizer


def benchmark_speed(model_path):
    """Benchmark inference speed of bouncer."""

    print("=" * 60)
    print("Benchmarking Speed")
    print("=" * 60)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Test prompts
    test_prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Write a story about a dragon",
        "How do I make a cake?",
        "Explain quantum physics",
    ]

    # Warmup
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model(**inputs)

    # Benchmark
    times = []

    for prompt in test_prompts * 20:  # 100 runs
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model(**inputs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()

        times.append((end - start) * 1000)  # ms

    print(f"\nInference Speed:")
    print(f"  Mean: {np.mean(times):.2f} ms")
    print(f"  Std:  {np.std(times):.2f} ms")
    print(f"  Min:  {np.min(times):.2f} ms")
    print(f"  Max:  {np.max(times):.2f} ms")
    print(f"  P99:  {np.percentile(times, 99):.2f} ms")

    # Compare to target
    target_ms = 5
    if np.mean(times) < target_ms:
        print(f"\n✅ Meets target (<{target_ms}ms)")
    else:
        print(f"\n⚠️  Above target ({target_ms}ms)")

    return np.mean(times)


def main():
    parser = argparse.ArgumentParser(description="L0 Bouncer Training")
    parser.add_argument("--generate-labels", action="store_true",
                       help="Generate teacher labels only")
    parser.add_argument("--train", action="store_true",
                       help="Train bouncer only (assumes labels exist)")
    parser.add_argument("--benchmark", action="store_true",
                       help="Benchmark speed only")
    parser.add_argument("--labels-file", type=str, default="teacher_labels.json",
                       help="Path to teacher labels file")
    args = parser.parse_args()

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    if args.benchmark:
        benchmark_speed(OUTPUT_DIR)
        return

    # Load data
    data = load_training_data()

    if args.generate_labels:
        generate_teacher_labels(data, args.labels_file)
        return

    if args.train:
        # Load existing labels
        with open(args.labels_file) as f:
            labeled_data = json.load(f)
        train_bouncer(labeled_data)
        benchmark_speed(OUTPUT_DIR)
        return

    # Full pipeline
    print("Running full pipeline...")

    # Step 1: Generate labels
    labels_path = Path(args.labels_file)
    if not labels_path.exists():
        labeled_data = generate_teacher_labels(data, args.labels_file)
    else:
        print(f"Loading existing labels from {args.labels_file}")
        with open(args.labels_file) as f:
            labeled_data = json.load(f)

    # Step 2: Train bouncer
    train_bouncer(labeled_data)

    # Step 3: Benchmark
    benchmark_speed(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("L0 Bouncer Training Complete!")
    print("=" * 60)
    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("  1. Test on held-out data")
    print("  2. Build cascade router")
    print("  3. Deploy with L1 (Exp 18 model)")


if __name__ == "__main__":
    main()
