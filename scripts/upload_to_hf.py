#!/usr/bin/env python3
"""
Upload GuardReasoner model to HuggingFace Hub
"""
from huggingface_hub import HfApi, create_repo
import os

# Configuration
MODEL_PATH = "experiments/guardreasoner/models/exp_19_hsdpo_toy_lora"
REPO_NAME = "guardreasoner-exp19-hsdpo-toy"
REPO_ID = f"vincentoh/{REPO_NAME}"

def main():
    print(f"Uploading model from: {MODEL_PATH}")
    print(f"Target repository: {REPO_ID}")

    # Initialize HF API
    api = HfApi()

    # Create repository (will skip if exists)
    try:
        print(f"\nCreating repository: {REPO_ID}")
        create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
        print("Repository created/verified!")
    except Exception as e:
        print(f"Repository creation: {e}")

    # Upload all files from the model directory
    print("\nUploading model files...")
    try:
        api.upload_folder(
            folder_path=MODEL_PATH,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message="Upload GuardReasoner Exp 19: HS-DPO Toy (10% dataset)",
            ignore_patterns=["*.pyc", "__pycache__", ".git"]
        )
        print("\n✅ Model uploaded successfully!")
        print(f"\n🔗 View your model at: https://huggingface.co/{REPO_ID}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise

if __name__ == "__main__":
    main()
