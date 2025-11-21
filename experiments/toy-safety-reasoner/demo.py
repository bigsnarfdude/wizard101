#!/usr/bin/env python3
"""
Interactive Demo for Toy Safety Reasoner

Run this script to experiment with the safety reasoner interactively.
"""

import sys
from safety_reasoner import SafetyReasoner, ReasoningLevel, Classification, format_result


def print_header():
    """Print welcome header"""
    print("\n" + "=" * 80)
    print(" " * 20 + "TOY SAFETY REASONER - Interactive Demo")
    print("=" * 80)
    print("\nThis is an educational tool demonstrating safety reasoning concepts.")
    print("NOT for production use - learning purposes only!\n")
    print("Based on concepts from OpenAI's gpt-oss-safeguard technical report.")
    print("=" * 80 + "\n")


def print_menu():
    """Print main menu"""
    print("\nOptions:")
    print("  1. Evaluate custom content")
    print("  2. Run quick examples")
    print("  3. Compare reasoning levels")
    print("  4. Test specific policy")
    print("  5. View available policies")
    print("  6. Learn about safety reasoners")
    print("  q. Quit")
    print()


def evaluate_custom_content(reasoner: SafetyReasoner):
    """Let user input content to evaluate"""
    print("\n" + "-" * 80)
    print("CUSTOM CONTENT EVALUATION")
    print("-" * 80)

    content = input("\nEnter content to evaluate (or 'back' to return):\n> ")

    if content.lower() == 'back':
        return

    # Choose reasoning level
    print("\nReasoning level:")
    print("  1. Low (fast, basic checks)")
    print("  2. Medium (balanced)")
    print("  3. High (thorough, slow)")

    level_choice = input("\nChoose level [1-3] (default: 2): ").strip()

    level_map = {
        '1': ReasoningLevel.LOW,
        '2': ReasoningLevel.MEDIUM,
        '3': ReasoningLevel.HIGH,
        '': ReasoningLevel.MEDIUM
    }

    reasoning_level = level_map.get(level_choice, ReasoningLevel.MEDIUM)

    print(f"\nEvaluating with {reasoning_level.value} reasoning...\n")

    result = reasoner.evaluate(content, reasoning_level=reasoning_level)

    # Display results
    print(format_result(result))

    # Ask if user wants to see full reasoning
    show_full = input("\nShow full reasoning chain? [y/N]: ").strip().lower()
    if show_full == 'y':
        print("\n" + result.reasoning_summary)

    input("\nPress Enter to continue...")


def run_quick_examples(reasoner: SafetyReasoner):
    """Run a few pre-defined examples"""
    print("\n" + "-" * 80)
    print("QUICK EXAMPLES")
    print("-" * 80)

    examples = [
        ("Safe: Political disagreement", "I disagree with their tax policy proposals."),
        ("Unsafe: Hate speech", "People from that group are subhuman vermin."),
        ("Edge case: Academic", "This research analyzes hate speech patterns online."),
        ("Unsafe: Violence", "Here's how to build explosives from household items."),
    ]

    for label, content in examples:
        print(f"\n--- {label} ---")
        print(f"Content: \"{content}\"")

        result = reasoner.evaluate(content, reasoning_level=ReasoningLevel.MEDIUM)

        # Color code output
        if result.overall_classification == Classification.SAFE:
            status = "✓ SAFE"
        elif result.overall_classification == Classification.UNSAFE:
            status = "✗ UNSAFE"
        else:
            status = "? UNCLEAR"

        print(f"Result: {status} (confidence: {result.overall_confidence:.1%})")

        if result.overall_classification == Classification.UNSAFE:
            violated = [pe.policy_name for pe in result.policy_evaluations
                       if pe.classification == Classification.UNSAFE]
            print(f"Violated: {', '.join(violated)}")

    input("\nPress Enter to continue...")


def compare_reasoning_levels(reasoner: SafetyReasoner):
    """Show how different reasoning levels work"""
    print("\n" + "-" * 80)
    print("REASONING LEVEL COMPARISON")
    print("-" * 80)

    content = input("\nEnter content to compare (or press Enter for default):\n> ").strip()

    if not content:
        content = "That group of people often engage in criminal behavior according to statistics."

    print(f"\nContent: \"{content}\"\n")

    results = {}
    for level in [ReasoningLevel.LOW, ReasoningLevel.MEDIUM, ReasoningLevel.HIGH]:
        results[level] = reasoner.evaluate(content, reasoning_level=level)

    # Display comparison
    print(f"{'Level':<10} {'Classification':<15} {'Confidence':<12} {'Steps':<10}")
    print("-" * 50)

    for level, result in results.items():
        steps = sum(len(pe.reasoning_chain) for pe in result.policy_evaluations)
        print(f"{level.value:<10} {result.overall_classification.value:<15} "
              f"{result.overall_confidence:.1%}{'':>6} {steps:<10}")

    print("\nKey insights:")
    print("  - LOW: Fast but may miss nuance")
    print("  - MEDIUM: Balanced speed and accuracy")
    print("  - HIGH: Most thorough but slower")

    input("\nPress Enter to continue...")


def test_specific_policy(reasoner: SafetyReasoner):
    """Test against a specific policy"""
    print("\n" + "-" * 80)
    print("SPECIFIC POLICY TEST")
    print("-" * 80)

    policies = reasoner.policies
    print("\nAvailable policies:")
    for i, policy in enumerate(policies, 1):
        print(f"  {i}. {policy['name']} - {policy['description'][:60]}...")

    choice = input("\nSelect policy [1-{}]: ".format(len(policies))).strip()

    try:
        policy_idx = int(choice) - 1
        if policy_idx < 0 or policy_idx >= len(policies):
            print("Invalid choice")
            return
    except ValueError:
        print("Invalid input")
        return

    selected_policy = policies[policy_idx]
    print(f"\nTesting against: {selected_policy['name']}")

    content = input("\nEnter content to evaluate:\n> ")

    result = reasoner.evaluate(
        content,
        policy_ids=[selected_policy['id']],
        reasoning_level=ReasoningLevel.HIGH
    )

    print(f"\n{format_result(result)}")

    input("\nPress Enter to continue...")


def view_policies(reasoner: SafetyReasoner):
    """Display all available policies"""
    print("\n" + "-" * 80)
    print("AVAILABLE POLICIES")
    print("-" * 80)

    for i, policy in enumerate(reasoner.policies, 1):
        print(f"\n{i}. {policy['name']} (ID: {policy['id']})")
        print(f"   Severity: {policy['severity']}")
        print(f"   Description: {policy['description']}")
        print(f"   Indicators: {', '.join(policy['indicators'][:3])}...")

    input("\nPress Enter to continue...")


def learn_about_safety_reasoners():
    """Educational information"""
    print("\n" + "=" * 80)
    print("WHAT IS A SAFETY REASONER?")
    print("=" * 80)

    print("""
A safety reasoner is an AI system that evaluates content against safety policies.

KEY CONCEPTS:

1. POLICY-BASED EVALUATION
   - Takes explicit written policies (e.g., "no hate speech")
   - Reasons from the policy rather than implicit training
   - Can adapt to new policies without retraining

2. CHAIN-OF-THOUGHT REASONING
   - Shows transparent reasoning steps
   - Explains WHY content was classified a certain way
   - Helps humans understand and validate decisions

3. MULTI-POLICY CLASSIFICATION
   - Can check many policies at once
   - Reports which specific policies are violated
   - Handles complex cases with multiple concerns

4. REASONING EFFORT LEVELS
   - LOW: Fast basic checks (~1-2 steps)
   - MEDIUM: Balanced analysis (~3-4 steps)
   - HIGH: Thorough evaluation (~5+ steps)

REAL-WORLD SYSTEMS:

OpenAI's gpt-oss-safeguard models:
- 20B and 120B parameter models
- Support 14+ languages
- ~80% F1 score on safety benchmarks
- Used for content moderation at scale

This toy implementation demonstrates the concepts but is NOT suitable
for production use. Real systems use large language models with billions
of parameters and extensive training data.

LIMITATIONS OF THIS TOY:

- Uses simple keyword matching (not semantic understanding)
- Only supports English
- Much lower accuracy than production systems
- Cannot handle adversarial inputs well
- Educational only - not for actual content moderation

LEARN MORE:

- OpenAI gpt-oss-safeguard Technical Report
- Instruction Hierarchy paper (Wallace et al.)
- StrongReject jailbreak benchmark
    """)

    input("\nPress Enter to continue...")


def main():
    """Main interactive loop"""
    reasoner = SafetyReasoner()

    print_header()

    while True:
        print_menu()
        choice = input("Choose an option: ").strip().lower()

        if choice == '1':
            evaluate_custom_content(reasoner)
        elif choice == '2':
            run_quick_examples(reasoner)
        elif choice == '3':
            compare_reasoning_levels(reasoner)
        elif choice == '4':
            test_specific_policy(reasoner)
        elif choice == '5':
            view_policies(reasoner)
        elif choice == '6':
            learn_about_safety_reasoners()
        elif choice == 'q':
            print("\nThanks for exploring safety reasoners! Keep learning.\n")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nExiting... Goodbye!\n")
        sys.exit(0)
