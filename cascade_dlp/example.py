#!/usr/bin/env python3
"""
Example usage of Cascade DLP v2.
"""

from cascade_dlp import DLPCascade


def main():
    # Initialize the pipeline
    print("Initializing Cascade DLP v2...")
    dlp = DLPCascade()

    # Example texts to process
    examples = [
        # Password disclosure
        "Your temporary password is: SecretPass123!",

        # Personal info
        "Contact John Smith at john.smith@company.com or 555-123-4567",

        # Financial data
        "Credit card: 4111-1111-1111-1111, expires 12/25",

        # API credentials
        "Set your API key to sk_live_abc123xyz789 in the config",

        # Address
        "Ship to: 123 Main Street, New York, NY 10001",

        # Mixed PII
        "User John Doe (DOB: 01/15/1990, SSN: 123-45-6789) registered from IP 192.168.1.100",

        # Clean text
        "The weather today is sunny with a high of 72 degrees.",
    ]

    print("\n" + "=" * 70)
    print("PROCESSING EXAMPLES")
    print("=" * 70)

    for text in examples:
        result = dlp.process(text)

        print(f"\n{'─' * 70}")
        print(f"INPUT:  {text}")
        print(f"ACTION: {result.action}")
        print(f"OUTPUT: {result.redacted_text}")

        if result.detections:
            print(f"FOUND:  {len(result.detections)} entities")
            for d in result.detections:
                print(f"        • {d.entity_type}: '{d.text}' (conf: {d.confidence:.2f})")
        else:
            print("FOUND:  No PII detected")

        print(f"TIME:   {result.processing_time_ms:.1f}ms")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
