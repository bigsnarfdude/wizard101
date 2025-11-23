"""
Download benchmark datasets for DLP evaluation.

Datasets:
- ai4privacy/pii-masking-200k: PII detection benchmark
- Presidio test samples: For comparison with Presidio baseline
"""

import json
import os
from pathlib import Path

# Try to import datasets
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: 'datasets' package not installed. Run: pip install datasets")


def download_ai4privacy(output_dir: str, sample_size: int = 1000):
    """
    Download ai4privacy PII masking dataset from HuggingFace.

    Dataset contains text with PII annotations for:
    - Names, emails, phone numbers, SSN
    - Addresses, dates, organizations
    - Financial info (credit cards, bank accounts)
    """
    if not HF_AVAILABLE:
        print("Cannot download - datasets package not available")
        return None

    print(f"Downloading ai4privacy dataset (sample_size={sample_size})...")

    # Load dataset
    dataset = load_dataset("ai4privacy/pii-masking-200k", split="train")

    # Sample for evaluation
    if sample_size and sample_size < len(dataset):
        dataset = dataset.shuffle(seed=42).select(range(sample_size))

    # Convert to list of dicts
    samples = []
    for item in dataset:
        samples.append({
            "text": item.get("source_text", item.get("masked_text", "")),
            "masked_text": item.get("masked_text", ""),
            "privacy_mask": item.get("privacy_mask", []),
            "span_labels": item.get("span_labels", []),
        })

    # Save to JSON
    output_path = Path(output_dir) / "ai4privacy_sample.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} samples to {output_path}")
    return samples


def create_pii_test_set(output_dir: str):
    """
    Create a curated PII test set with known ground truth.

    Categories:
    - Names (PERSON)
    - Emails (EMAIL)
    - Phone numbers (PHONE)
    - SSN (US_SSN)
    - Credit cards (CREDIT_CARD)
    - Addresses (ADDRESS)
    """
    test_cases = [
        # True positives - should detect
        {
            "text": "Contact John Smith at john.smith@example.com or call 555-123-4567",
            "expected": ["PERSON", "EMAIL", "PHONE"],
            "label": True,
        },
        {
            "text": "My SSN is 123-45-6789 and credit card is 4532-1234-5678-9012",
            "expected": ["US_SSN", "CREDIT_CARD"],
            "label": True,
        },
        {
            "text": "Dr. Sarah Johnson lives at 123 Main Street, New York, NY 10001",
            "expected": ["PERSON", "ADDRESS"],
            "label": True,
        },
        {
            "text": "Bank account: 123456789, routing: 021000021",
            "expected": ["US_BANK_NUMBER"],
            "label": True,
        },
        {
            "text": "Patient: Jane Doe, DOB: 03/15/1985, MRN: 12345678",
            "expected": ["PERSON", "DATE", "MEDICAL_RECORD"],
            "label": True,
        },

        # True negatives - should NOT detect (false positive tests)
        {
            "text": "The weather today is sunny with a high of 75 degrees.",
            "expected": [],
            "label": False,
        },
        {
            "text": "For more information, visit example.com or call our main office.",
            "expected": [],
            "label": False,
        },
        {
            "text": "The function returns 123-456-789 as the error code.",
            "expected": [],  # Looks like SSN but is code
            "label": False,
        },
        {
            "text": "Use API key placeholder: YOUR_API_KEY_HERE in the config.",
            "expected": [],  # Placeholder, not real
            "label": False,
        },
        {
            "text": "Contact support@example.com for test purposes only.",
            "expected": [],  # Example domain
            "label": False,
        },
    ]

    output_path = Path(output_dir) / "pii_test_set.json"
    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"Created {len(test_cases)} PII test cases at {output_path}")
    return test_cases


def create_secret_test_set(output_dir: str):
    """
    Create a secret detection test set based on SecretBench categories.

    Categories from SecretBench:
    - Private Keys (RSA, EC, PGP)
    - API Keys (AWS, Stripe, Google)
    - Authentication Tokens (JWT, GitHub, Slack)
    - Generic Secrets (passwords, connection strings)
    """
    test_cases = [
        # True positives - should detect
        {
            "text": "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE",
            "expected": ["AWS_ACCESS_KEY"],
            "label": True,
            "category": "Authentication Key",
        },
        {
            "text": "export GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "expected": ["GITHUB_TOKEN"],
            "label": True,
            "category": "Authentication Token",
        },
        {
            "text": "stripe_api_key = 'sk_live_EXAMPLE_KEY_REPLACE'",
            "expected": ["STRIPE_KEY"],
            "label": True,
            "category": "API Key",
        },
        {
            "text": "postgres://admin:secretpassword@db.prod.com:5432/mydb",
            "expected": ["POSTGRES_URI", "PASSWORD_IN_URL"],
            "label": True,
            "category": "Database Credential",
        },
        {
            "text": "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----",
            "expected": ["RSA_PRIVATE_KEY"],
            "label": True,
            "category": "Private Key",
        },
        {
            "text": "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            "expected": ["JWT", "BEARER_TOKEN"],
            "label": True,
            "category": "Authentication Token",
        },
        {
            "text": "SLACK_TOKEN=xoxb-0000000000-0000000000000-EXAMPLETOKEN",
            "expected": ["SLACK_TOKEN"],
            "label": True,
            "category": "Authentication Token",
        },
        {
            "text": "password = 'MyS3cretP@ssw0rd!'",
            "expected": ["PASSWORD_ASSIGNMENT"],
            "label": True,
            "category": "Generic Secret",
        },

        # True negatives - should NOT detect
        {
            "text": "The API documentation is available at docs.example.com",
            "expected": [],
            "label": False,
            "category": "Clean",
        },
        {
            "text": "Use placeholder YOUR_API_KEY in the configuration file.",
            "expected": [],
            "label": False,
            "category": "Placeholder",
        },
        {
            "text": "The test uses fake credentials: user=test, pass=test123",
            "expected": [],  # Too short/obvious test values
            "label": False,
            "category": "Test Value",
        },
    ]

    output_path = Path(output_dir) / "secret_test_set.json"
    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"Created {len(test_cases)} secret test cases at {output_path}")
    return test_cases


def main():
    """Download all datasets for evaluation."""
    output_dir = Path(__file__).parent / "datasets"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("CASCADE DLP - DATASET DOWNLOAD")
    print("=" * 60)

    # Create curated test sets
    print("\n1. Creating PII test set...")
    create_pii_test_set(output_dir)

    print("\n2. Creating secret test set...")
    create_secret_test_set(output_dir)

    # Download ai4privacy if available
    print("\n3. Downloading ai4privacy dataset...")
    if HF_AVAILABLE:
        try:
            download_ai4privacy(output_dir, sample_size=1000)
        except Exception as e:
            print(f"Error downloading ai4privacy: {e}")
            print("Continuing with curated test sets...")
    else:
        print("Skipping ai4privacy (datasets package not installed)")

    print("\n" + "=" * 60)
    print("DATASET DOWNLOAD COMPLETE")
    print("=" * 60)

    # List created files
    print("\nCreated files:")
    for f in output_dir.glob("*.json"):
        size = f.stat().st_size / 1024
        print(f"  - {f.name} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
