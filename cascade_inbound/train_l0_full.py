#!/usr/bin/env python3
"""
Train L0 Bouncer on Full GuardReasoner Dataset (124k samples)

Uses DeBERTa-v3-xsmall for fast inference with good accuracy.

Usage:
    python train_l0_full.py
"""

import json
import random
from pathlib import Path
from collections import Counter

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def extract_label(output_text):
    """Extract label from GuardReasoner output."""
    text = output_text.lower()
    if "request: harmful" in text:
        return "harmful"
    elif "request: unharmful" in text:
        return "safe"
    elif "request: benign" in text:
        return "safe"
    return None


def extract_prompt(input_text):
    """Extract prompt from GuardReasoner input."""
    text = input_text.strip()
    if text.startswith("Human user:"):
        text = text[11:].strip()
    if "\nAI assistant:" in text:
        text = text.split("\nAI assistant:")[0].strip()
    return text


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', pos_label=1
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def main():
    print("=" * 60)
    print("L0 BOUNCER FULL TRAINING (124k samples)")
    print("=" * 60)

    # Load full GuardReasoner data
    data_path = Path("../guardreasoner/guardreasoner_data/all_combined.json")
    print(f"\nLoading data from {data_path}")

    with open(data_path) as f:
        all_data = json.load(f)

    print(f"Total samples: {len(all_data):,}")

    # Convert to training format
    samples = []
    for item in all_data:
        prompt = extract_prompt(item["input"])
        label = extract_label(item["output"])

        if not prompt or not label or len(prompt) < 10:
            continue

        samples.append({
            "text": prompt,
            "label": 1 if label == "harmful" else 0,
        })

    print(f"Valid samples: {len(samples):,}")

    # Check distribution
    labels = Counter(s["label"] for s in samples)
    print(f"  Safe (0): {labels[0]:,} ({labels[0]/len(samples)*100:.1f}%)")
    print(f"  Harmful (1): {labels[1]:,} ({labels[1]/len(samples)*100:.1f}%)")

    # Shuffle
    random.seed(42)
    random.shuffle(samples)

    # Split: 90% train, 10% validation
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    print(f"\nTrain: {len(train_samples):,}")
    print(f"Val: {len(val_samples):,}")

    # Create datasets
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    # Load model and tokenizer
    model_name = "microsoft/deberta-v3-xsmall"
    print(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "safe", 1: "harmful"},
        label2id={"safe": 0, "harmful": 1},
    )

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    print("Tokenizing...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)

    # Training arguments
    output_dir = "./models/l0_bouncer_full"

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=100,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Epochs: 3")
    print(f"Batch size: 32")
    print(f"Total steps: ~{len(train_samples) * 3 // 32:,}")
    print("=" * 60 + "\n")

    trainer.train()

    # Evaluate
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    results = trainer.evaluate()
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Save
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"\nModel saved to {output_dir}")
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
