"""
Download all benchmark datasets for scale validation.

Datasets:
1. ai4privacy full 200k
2. PIILO (if available)
3. Presidio test suite
"""

import json
import os
from pathlib import Path

try:
    from datasets import load_dataset
    import requests
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Install: pip install datasets requests")


def download_ai4privacy_full(output_dir: str):
    """Download full ai4privacy dataset (200k samples)."""
    if not HF_AVAILABLE:
        return None

    print("Downloading ai4privacy FULL dataset...")
    print("This may take a few minutes...")

    dataset = load_dataset("ai4privacy/pii-masking-200k", split="train")

    print(f"Total samples: {len(dataset)}")

    # Convert to list
    samples = []
    for i, item in enumerate(dataset):
        if i % 10000 == 0:
            print(f"  Processing {i}/{len(dataset)}...")

        samples.append({
            "text": item.get("source_text", item.get("masked_text", "")),
            "masked_text": item.get("masked_text", ""),
            "privacy_mask": item.get("privacy_mask", []),
            "span_labels": item.get("span_labels", []),
        })

    output_path = Path(output_dir) / "ai4privacy_full.json"
    with open(output_path, "w") as f:
        json.dump(samples, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(samples)} samples to {output_path} ({size_mb:.1f} MB)")
    return samples


def download_presidio_test_data(output_dir: str):
    """Download Presidio's test data from GitHub."""
    print("\nDownloading Presidio test data...")

    # Presidio has test data in their repo
    base_url = "https://raw.githubusercontent.com/microsoft/presidio/main/presidio-analyzer/tests/data"

    test_files = [
        "demo.txt",
    ]

    samples = []

    for filename in test_files:
        url = f"{base_url}/{filename}"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                text = response.text
                samples.append({
                    "text": text,
                    "source": f"presidio/{filename}",
                    "has_pii": True,  # Presidio test data has PII
                })
                print(f"  ✓ Downloaded {filename}")
            else:
                print(f"  ✗ Failed to download {filename}: {response.status_code}")
        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {e}")

    # Also create synthetic Presidio-style test cases
    presidio_test_cases = [
        {"text": "John Smith's phone number is 212-555-5555", "entities": ["PERSON", "PHONE_NUMBER"]},
        {"text": "My credit card is 4095-2609-9393-4932", "entities": ["CREDIT_CARD"]},
        {"text": "Her social security number is 078-05-1120", "entities": ["US_SSN"]},
        {"text": "My email is johnsmith@gmail.com", "entities": ["EMAIL_ADDRESS"]},
        {"text": "I live at 123 Main Street, New York, NY 10001", "entities": ["LOCATION"]},
        {"text": "My passport number is 912803456", "entities": ["US_PASSPORT"]},
        {"text": "Driver license: A1234567", "entities": ["US_DRIVER_LICENSE"]},
        {"text": "IBAN: DE89370400440532013000", "entities": ["IBAN_CODE"]},
        {"text": "IP address: 192.168.1.1", "entities": ["IP_ADDRESS"]},
        {"text": "Date of birth: January 15, 1990", "entities": ["DATE_TIME"]},
    ]

    for case in presidio_test_cases:
        samples.append({
            "text": case["text"],
            "expected_entities": case["entities"],
            "has_pii": True,
        })

    output_path = Path(output_dir) / "presidio_test.json"
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved {len(samples)} Presidio test cases to {output_path}")
    return samples


def download_synthetic_secrets(output_dir: str):
    """Create synthetic secret test set based on SecretBench patterns."""
    print("\nCreating synthetic secret test set...")

    # Based on SecretBench categories
    test_cases = [
        # AWS
        {"text": "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE", "type": "AWS_ACCESS_KEY", "category": "Authentication Key"},
        {"text": "aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY'", "type": "AWS_SECRET_KEY", "category": "Authentication Key"},

        # GitHub
        {"text": "GITHUB_TOKEN=ghp_aBcDeFgHiJkLmNoPqRsTuVwXyZ0123456789", "type": "GITHUB_TOKEN", "category": "Authentication Token"},
        {"text": "gho_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "type": "GITHUB_OAUTH", "category": "Authentication Token"},

        # Google
        {"text": "AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe", "type": "GOOGLE_API_KEY", "category": "API Key"},

        # Slack (pattern: xoxb-NUMBERS-NUMBERS-ALPHANUM)
        {"text": "SLACK_BOT_TOKEN=YOUR_SLACK_TOKEN_HERE", "type": "SLACK_TOKEN", "category": "Authentication Token"},

        # Database
        {"text": "postgres://admin:password123@db.example.com:5432/production", "type": "POSTGRES_URI", "category": "Database Credential"},
        {"text": "mysql://root:secret@localhost:3306/mydb", "type": "MYSQL_URI", "category": "Database Credential"},
        {"text": "mongodb+srv://user:pass@cluster.mongodb.net/db", "type": "MONGODB_URI", "category": "Database Credential"},

        # Private Keys
        {"text": "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----", "type": "RSA_PRIVATE_KEY", "category": "Private Key"},
        {"text": "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEA...\n-----END OPENSSH PRIVATE KEY-----", "type": "OPENSSH_PRIVATE_KEY", "category": "Private Key"},

        # JWT
        {"text": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", "type": "JWT", "category": "Authentication Token"},

        # Generic
        {"text": "api_key = 'sk_live_EXAMPLE_KEY_REPLACE_THIS'", "type": "GENERIC_API_KEY", "category": "API Key"},
        {"text": "password = 'SuperSecretPassword123!'", "type": "PASSWORD_ASSIGNMENT", "category": "Generic Secret"},

        # SendGrid
        {"text": "SG.xxxxxxxxxxxxxxxxxxxxxx.xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", "type": "SENDGRID_API_KEY", "category": "API Key"},

        # Clean (no secrets)
        {"text": "This is normal text without any secrets", "type": None, "category": "Clean"},
        {"text": "API documentation available at docs.example.com", "type": None, "category": "Clean"},
        {"text": "Use YOUR_API_KEY as a placeholder", "type": None, "category": "Placeholder"},
    ]

    output_path = Path(output_dir) / "synthetic_secrets.json"
    with open(output_path, "w") as f:
        json.dump(test_cases, f, indent=2)

    print(f"Saved {len(test_cases)} synthetic secret test cases to {output_path}")
    return test_cases


def main():
    output_dir = Path(__file__).parent / "datasets"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("DOWNLOADING ALL BENCHMARK DATASETS")
    print("=" * 60)

    # 1. Full ai4privacy (200k)
    print("\n1. ai4privacy (200k samples)")
    print("-" * 40)
    try:
        download_ai4privacy_full(output_dir)
    except Exception as e:
        print(f"Error: {e}")

    # 2. Presidio test data
    print("\n2. Presidio test data")
    print("-" * 40)
    try:
        download_presidio_test_data(output_dir)
    except Exception as e:
        print(f"Error: {e}")

    # 3. Synthetic secrets
    print("\n3. Synthetic secrets (SecretBench-based)")
    print("-" * 40)
    try:
        download_synthetic_secrets(output_dir)
    except Exception as e:
        print(f"Error: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)

    print("\nDatasets created:")
    for f in sorted(output_dir.glob("*.json")):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        else:
            size_str = f"{size / 1024:.1f} KB"
        print(f"  - {f.name} ({size_str})")

    print("\nNote: SecretBench requires BigQuery access with data protection agreement")
    print("      PIILO dataset requires academic access")


if __name__ == "__main__":
    main()
