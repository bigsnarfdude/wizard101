#!/usr/bin/env python3
"""
Train L0 Bouncer with 12K GuardReasoner Dataset

Uses balanced 12k samples from GuardReasoner data.

Expected: Better accuracy from 15x more data.

Usage:
    python train_l0_12k.py
"""

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
import numpy as np

# Configuration
STUDENT_MODEL = "microsoft/deberta-v3-xsmall"
OUTPUT_DIR = "./models/l0_bouncer_12k"
DATA_FILE = "train_12k.json"

# Training hyperparameters
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
EPOCHS = 3  # Fewer epochs with more data
MAX_LENGTH = 256


class SafetyDataset(Dataset):
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
        label = 1 if item["label"] == "harmful" else 0
        encoding["labels"] = label
        return encoding


def main():
    print("=" * 60)
    print("L0 BOUNCER 12K TRAINING")
    print("=" * 60)

    # Load dataset
    with open(DATA_FILE) as f:
        data = json.load(f)

    print(f"\nLoaded {len(data):,} samples")

    # Stats
    safe = sum(1 for d in data if d["label"] == "safe")
    harmful = len(data) - safe
    print(f"  Safe: {safe:,} ({safe/len(data)*100:.1f}%)")
    print(f"  Harmful: {harmful:,} ({harmful/len(data)*100:.1f}%)")

    # Load model
    print(f"\nLoading {STUDENT_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL,
        num_labels=2,
    )

    # Prepare dataset - use ALL for training
    train_dataset = SafetyDataset(data, tokenizer, MAX_LENGTH)
    print(f"Train: {len(train_dataset):,} (all samples)")

    # Load separate test set for evaluation
    test_path = Path("../combined_test.json")
    if test_path.exists():
        with open(test_path) as f:
            test_data = json.load(f)
        # Convert format
        test_converted = []
        for item in test_data:
            text = item.get("prompt", item.get("text", ""))
            label = item.get("label", "").lower()
            if label in ["harmless", "safe"]:
                label = "safe"
            else:
                label = "harmful"
            test_converted.append({"text": text, "label": label})
        eval_dataset = SafetyDataset(test_converted, tokenizer, MAX_LENGTH)
        print(f"Eval: {len(eval_dataset):,} (separate test set)")

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        fp16=False,
        gradient_accumulation_steps=4,
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train
    print("\nTraining...")
    trainer.train()

    # Save
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved to {OUTPUT_DIR}")

    # Results
    results = trainer.evaluate()
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['eval_accuracy']*100:.1f}%")
    print(f"Precision: {results['eval_precision']*100:.1f}%")
    print(f"Recall: {results['eval_recall']*100:.1f}%")
    print(f"F1: {results['eval_f1']*100:.1f}%")

    print("\nComparison:")
    print("  Original (800): F1 ~57%")
    print("  Mega (2.5k): F1 85.6%")
    print(f"  12K ({len(data):,}): F1 {results['eval_f1']*100:.1f}%")


if __name__ == "__main__":
    main()
