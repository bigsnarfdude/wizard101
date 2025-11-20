#!/usr/bin/env python3
"""
Train L0 Bouncer with Mega Dataset

Uses the combined mega_train.json (2540 samples) for improved training.

Expected improvements:
- Better accuracy from 3.2x more data
- Better balance (55%/45% vs 76%/24%)
- Reduced need for class weighting

Usage:
    python train_l0_mega.py
"""

import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch.nn as nn

# Configuration
STUDENT_MODEL = "microsoft/deberta-v3-xsmall"
OUTPUT_DIR = "./models/l0_bouncer_mega"
DATA_FILE = "mega_train.json"

# Training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 8  # Reduced for 16GB GPU
EPOCHS = 5  # Fewer epochs with more data
MAX_LENGTH = 256


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

        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )

        # Convert label to int (0=safe, 1=harmful)
        label = 1 if item["label"] == "harmful" else 0
        encoding["labels"] = label

        return encoding


class WeightedTrainer(Trainer):
    """Trainer with optional class-weighted loss."""

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


def main():
    print("=" * 60)
    print("L0 BOUNCER MEGA TRAINING")
    print("=" * 60)

    # Load mega dataset
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Run build_mega_dataset.py first")
        return

    with open(data_path) as f:
        data = json.load(f)

    print(f"\nLoaded {len(data)} samples")

    # Convert to training format
    train_data = []
    for item in data:
        train_data.append({
            "text": item["text"],
            "label": item["label"],
        })

    # Stats
    labels = [d["label"] for d in train_data]
    safe_count = labels.count("safe")
    harmful_count = labels.count("harmful")
    print(f"  Safe: {safe_count} ({safe_count/len(labels)*100:.1f}%)")
    print(f"  Harmful: {harmful_count} ({harmful_count/len(labels)*100:.1f}%)")

    # Load model and tokenizer
    print(f"\nLoading model: {STUDENT_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL,
        num_labels=2,
        problem_type="single_label_classification",
    )

    # Prepare dataset
    dataset = SafetyDataset(train_data, tokenizer, MAX_LENGTH)

    # Split train/eval (90/10)
    train_size = int(0.9 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        dataset, [train_size, eval_size]
    )

    print(f"\nTrain: {train_size}, Eval: {eval_size}")

    # Class weights (light weighting since more balanced now)
    label_ints = np.array([1 if l == "harmful" else 0 for l in labels])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=label_ints
    )
    # Small boost for harmful detection
    class_weights[1] *= 1.1
    print(f"Class weights: safe={class_weights[0]:.3f}, harmful={class_weights[1]:.3f}")

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
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=False,  # Disable for memory stability
        gradient_accumulation_steps=4,  # Effective batch size 32
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

    # Trainer
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
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Compare to original
    print("\n" + "=" * 60)
    print("COMPARISON TO ORIGINAL")
    print("=" * 60)
    print("Original L0 (800 samples):")
    print("  Accuracy: ~70%")
    print("  F1: ~57%")
    print(f"\nNew L0 ({len(data)} samples):")
    print(f"  Accuracy: {results.get('eval_accuracy', 0)*100:.1f}%")
    print(f"  F1: {results.get('eval_f1', 0)*100:.1f}%")

    if results.get('eval_f1', 0) > 0.57:
        improvement = (results['eval_f1'] - 0.57) / 0.57 * 100
        print(f"\nF1 improved by {improvement:.1f}%")


if __name__ == "__main__":
    main()
