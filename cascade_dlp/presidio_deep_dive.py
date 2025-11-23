#!/usr/bin/env python3
"""
Presidio Deep Dive - Understanding the techniques.

Explore what makes Presidio more than regex + NER.
"""

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpEngineProvider


def main():
    print("=" * 60)
    print("PRESIDIO DEEP DIVE")
    print("=" * 60)

    # Initialize
    analyzer = AnalyzerEngine()

    # 1. List all built-in recognizers
    print("\n" + "-" * 60)
    print("1. BUILT-IN RECOGNIZERS")
    print("-" * 60)

    recognizers = analyzer.get_recognizers()
    print(f"\nTotal recognizers: {len(recognizers)}")

    for r in sorted(recognizers, key=lambda x: x.name):
        entities = ", ".join(r.supported_entities)
        print(f"  • {r.name}: {entities}")

    # 2. Examine a specific recognizer
    print("\n" + "-" * 60)
    print("2. CREDIT CARD RECOGNIZER DETAILS")
    print("-" * 60)

    cc_recognizers = [r for r in recognizers if "CREDIT_CARD" in r.supported_entities]
    for r in cc_recognizers:
        print(f"\n{r.name}:")
        if hasattr(r, 'patterns'):
            print(f"  Patterns: {len(r.patterns)}")
            for p in r.patterns[:3]:
                print(f"    - {p.name}: {p.regex[:50]}...")
        if hasattr(r, 'context'):
            print(f"  Context words: {r.context[:5]}...")

    # 3. Context enhancement demo
    print("\n" + "-" * 60)
    print("3. CONTEXT ENHANCEMENT")
    print("-" * 60)

    # Same number, different context
    test_cases = [
        ("Call me at 4532015112830366", "No context"),
        ("Credit card: 4532015112830366", "With 'Credit card'"),
        ("My card number is 4532015112830366", "With 'card number'"),
    ]

    for text, description in test_cases:
        results = analyzer.analyze(text=text, language="en")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]
        if cc_results:
            score = cc_results[0].score
            print(f"\n{description}:")
            print(f"  Text: {text}")
            print(f"  Score: {score:.2f}")
        else:
            print(f"\n{description}:")
            print(f"  Text: {text}")
            print(f"  Not detected as credit card")

    # 4. Checksum validation
    print("\n" + "-" * 60)
    print("4. CHECKSUM VALIDATION (Luhn Algorithm)")
    print("-" * 60)

    test_numbers = [
        ("4532015112830366", "Valid Visa"),
        ("4532015112830367", "Invalid (wrong checksum)"),
        ("1234567890123456", "Invalid (fails Luhn)"),
    ]

    for number, description in test_numbers:
        text = f"Card: {number}"
        results = analyzer.analyze(text=text, language="en")
        cc_results = [r for r in results if r.entity_type == "CREDIT_CARD"]

        print(f"\n{description}:")
        print(f"  Number: {number}")
        if cc_results:
            print(f"  Detected: Yes (score: {cc_results[0].score:.2f})")
        else:
            print(f"  Detected: No (failed validation)")

    # 5. NER model entities
    print("\n" + "-" * 60)
    print("5. NER MODEL ENTITIES")
    print("-" * 60)

    text = "John Smith from Microsoft in Seattle emailed john@microsoft.com about the $50,000 deal."
    results = analyzer.analyze(text=text, language="en")

    print(f"\nText: {text}")
    print(f"\nEntities found:")
    for r in sorted(results, key=lambda x: x.start):
        entity_text = text[r.start:r.end]
        print(f"  • {r.entity_type}: '{entity_text}' (score: {r.score:.2f})")

    # 6. Supported entity types
    print("\n" + "-" * 60)
    print("6. ALL SUPPORTED ENTITY TYPES")
    print("-" * 60)

    all_entities = set()
    for r in recognizers:
        all_entities.update(r.supported_entities)

    print(f"\nTotal entity types: {len(all_entities)}")
    for entity in sorted(all_entities):
        print(f"  • {entity}")

    # 7. Custom recognizer example
    print("\n" + "-" * 60)
    print("7. CUSTOM RECOGNIZER EXAMPLE")
    print("-" * 60)

    # Add custom recognizer for employee IDs
    employee_pattern = Pattern(
        name="employee_id_pattern",
        regex=r"\bEMP-\d{6}\b",
        score=0.9
    )

    employee_recognizer = PatternRecognizer(
        supported_entity="EMPLOYEE_ID",
        patterns=[employee_pattern],
        context=["employee", "emp", "staff", "id"]
    )

    analyzer.registry.add_recognizer(employee_recognizer)

    test_text = "Contact employee EMP-123456 for details."
    results = analyzer.analyze(text=test_text, language="en")

    print(f"\nCustom pattern: EMP-\\d{{6}}")
    print(f"Text: {test_text}")
    print(f"Found: {[(r.entity_type, test_text[r.start:r.end]) for r in results]}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: PRESIDIO TECHNIQUES")
    print("=" * 60)

    print("""
1. PATTERN RECOGNIZERS
   - Regex with named groups
   - Multiple patterns per entity type
   - Configurable scores

2. NER MODELS
   - spaCy en_core_web_lg
   - PERSON, ORG, GPE, LOC, etc.
   - Transformer models available

3. CONTEXT ENHANCEMENT
   - Words near entity boost score
   - "credit card:" → higher confidence
   - Configurable context words

4. VALIDATORS
   - Luhn checksum for credit cards
   - Format validation for SSN
   - Custom validators supported

5. DENY LISTS
   - Exact match lists
   - Fast lookup

6. CUSTOM RECOGNIZERS
   - Add domain-specific patterns
   - Custom validators
   - Context words

7. SCORE AGGREGATION
   - Multiple recognizers can match
   - Scores combined/filtered
   - Threshold configurable
""")


if __name__ == "__main__":
    main()
