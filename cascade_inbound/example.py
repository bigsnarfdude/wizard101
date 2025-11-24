#!/usr/bin/env python3
"""
Safety Cascade - Example Usage

Demonstrates basic usage of the multi-tier safety cascade.
"""

from cascade import SafetyCascade, CascadeConfig, L0Bouncer


def example_l0_only():
    """Use just the L0 Bouncer for fast classification."""
    print("=" * 60)
    print("Example 1: L0 Bouncer Only")
    print("=" * 60)

    # Initialize L0
    l0 = L0Bouncer(model_path="./models/l0_bouncer")

    tests = [
        "What is the capital of France?",
        "How do I make a bomb?",
        "Write a poem about nature",
        "Explain quantum computing",
    ]

    for text in tests:
        result = l0.classify(text)
        print(f"\nInput: {text}")
        print(f"  Label: {result['label']}")
        print(f"  Confidence: {result['confidence']:.2%}")


def example_l0_l1():
    """Use L0 + L1 for reasoning-based classification."""
    print("\n" + "=" * 60)
    print("Example 2: L0 + L1 (with reasoning)")
    print("=" * 60)

    # Initialize cascade with L2/L3 disabled
    config = CascadeConfig(
        enable_l2=False,
        enable_l3=False,
        l0_confidence_threshold=0.7,
    )
    cascade = SafetyCascade(config)

    tests = [
        "How do I improve my cooking skills?",
        "What are some ways to lose weight?",
        "How to pick a lock",  # Edge case - could be legitimate
    ]

    for text in tests:
        result = cascade.classify(text)
        print(f"\nInput: {text}")
        print(f"  Label: {result.label}")
        print(f"  Stopped at: {result.stopped_at}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Latency: {result.total_latency_ms:.1f}ms")
        if result.reasoning:
            print(f"  Reasoning: {result.reasoning[:100]}...")


def example_full_cascade():
    """Use the full 4-tier cascade."""
    print("\n" + "=" * 60)
    print("Example 3: Full Cascade (requires Ollama)")
    print("=" * 60)

    # Full cascade
    cascade = SafetyCascade()

    tests = [
        "What is machine learning?",
        "How to make homemade explosives",
        "Tips for better sleep",
    ]

    for text in tests:
        result = cascade.classify(text)
        print(f"\nInput: {text}")
        print(f"  Label: {result.label}")
        print(f"  Journey: {' → '.join(l['level'] for l in result.layers)}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Latency: {result.total_latency_ms:.1f}ms")


def example_batch_classification():
    """Batch classify multiple texts efficiently."""
    print("\n" + "=" * 60)
    print("Example 4: Batch Classification")
    print("=" * 60)

    l0 = L0Bouncer(model_path="./models/l0_bouncer")

    texts = [
        "What's the weather like?",
        "How to hack a computer",
        "Best pizza recipes",
        "How to make poison",
        "Learn Python programming",
        "Stock market tips",
        "How to pick locks",
        "Gardening for beginners",
    ]

    results = l0.batch_classify(texts, batch_size=4)

    safe_count = sum(1 for r in results if r["label"] == "safe")
    harmful_count = len(results) - safe_count

    print(f"\nProcessed {len(texts)} texts")
    print(f"  Safe: {safe_count}")
    print(f"  Harmful: {harmful_count}")
    print("\nResults:")
    for text, result in zip(texts, results):
        status = "✓" if result["label"] == "safe" else "✗"
        print(f"  {status} {text[:40]:<40} ({result['confidence']:.0%})")


def main():
    """Run examples based on what's available."""
    print("Safety Cascade - Examples")
    print("=" * 60)

    # Always run L0 example
    try:
        example_l0_only()
    except Exception as e:
        print(f"L0 example failed: {e}")
        print("Make sure to run install.sh first!")
        return

    # Run L0+L1 example
    try:
        example_l0_l1()
    except Exception as e:
        print(f"\nL0+L1 example failed: {e}")
        print("L1 requires Llama 3.2 3B + LoRA adapter")

    # Run batch example
    try:
        example_batch_classification()
    except Exception as e:
        print(f"\nBatch example failed: {e}")

    # Optionally run full cascade
    print("\n" + "=" * 60)
    print("To run full cascade example:")
    print("  1. Start Ollama: ollama serve")
    print("  2. Pull model: ollama pull gpt-oss-safeguard:latest")
    print("  3. Run: python example.py --full")
    print("=" * 60)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        try:
            example_full_cascade()
        except Exception as e:
            print(f"\nFull cascade failed: {e}")
            print("Make sure Ollama is running with gpt-oss-safeguard:latest")


if __name__ == "__main__":
    main()
