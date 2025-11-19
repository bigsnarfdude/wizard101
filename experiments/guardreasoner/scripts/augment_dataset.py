#!/usr/bin/env python3
"""
Advanced dataset augmentation using Gemini 2.0.

Techniques:
1. Paraphrasing - Rephrase prompts while maintaining intent
2. Adversarial generation - Create jailbreak variations
3. Context injection - Add realistic conversation context
4. Multi-turn expansion - Convert single prompts to conversations
5. Cross-lingual - Translate and back-translate for diversity

This can 5-10× your dataset size with high-quality variations.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import google.generativeai as genai

# Configure API
# Get API key from environment variable
import os
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Run: export GEMINI_API_KEY='your-key-here'")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-exp')


class DatasetAugmenter:
    """Augment safety datasets with diverse variations."""

    def __init__(self):
        self.model = model
        self.stats = {
            "paraphrased": 0,
            "adversarial": 0,
            "contextualized": 0,
            "multi_turn": 0,
            "cross_lingual": 0
        }

    def paraphrase(self, prompt: str, num_variations: int = 3) -> List[str]:
        """Generate paraphrased versions of a prompt.

        Maintains semantic meaning while varying expression.
        """
        paraphrase_prompt = f"""Generate {num_variations} paraphrased versions of this request.

Original: "{prompt}"

Requirements:
- Maintain the exact same intent and meaning
- Vary sentence structure and word choice
- Keep the same level of harm/safety
- Each paraphrase should be distinct

Output format (one per line):
1. [paraphrase 1]
2. [paraphrase 2]
3. [paraphrase 3]

Paraphrases:"""

        try:
            result = self.model.generate_content(paraphrase_prompt)
            lines = [line.strip() for line in result.text.strip().split('\n') if line.strip()]

            # Extract numbered paraphrases
            paraphrases = []
            for line in lines:
                # Remove numbering (1., 2., etc.)
                import re
                clean = re.sub(r'^\d+\.\s*', '', line)
                if clean and clean != prompt:
                    paraphrases.append(clean)

            self.stats["paraphrased"] += len(paraphrases)
            return paraphrases[:num_variations]

        except Exception as e:
            print(f"Paraphrase error: {e}")
            return []

    def generate_adversarial(self, prompt: str, num_variations: int = 2) -> List[str]:
        """Generate adversarial jailbreak variations.

        Creates attempts to bypass safety filters while maintaining harm intent.
        """
        adversarial_prompt = f"""Generate {num_variations} adversarial jailbreak variations of this harmful request.

Original harmful request: "{prompt}"

Create variations that attempt to bypass safety filters using:
- Role-playing scenarios (e.g., "For a novel...")
- Hypothetical framing (e.g., "What would happen if...")
- Obfuscation (e.g., euphemisms, coded language)
- Context injection (e.g., "My teacher assigned...")

Requirements:
- Maintain the underlying harmful intent
- Make them realistic (actual jailbreak techniques)
- Vary the approach for each variation

Output format:
1. [variation 1]
2. [variation 2]

Variations:"""

        try:
            result = self.model.generate_content(adversarial_prompt)
            lines = [line.strip() for line in result.text.strip().split('\n') if line.strip()]

            variations = []
            import re
            for line in lines:
                clean = re.sub(r'^\d+\.\s*', '', line)
                if clean:
                    variations.append(clean)

            self.stats["adversarial"] += len(variations)
            return variations[:num_variations]

        except Exception as e:
            print(f"Adversarial generation error: {e}")
            return []

    def add_context(self, prompt: str, num_variations: int = 2) -> List[str]:
        """Add realistic conversation context to prompts.

        Makes prompts more natural by embedding in realistic scenarios.
        """
        context_prompt = f"""Add realistic conversation context to this request.

Original: "{prompt}"

Create {num_variations} variations by adding:
- Backstory or scenario setup
- Emotional context
- Specific details or constraints
- Real-world framing

Keep the core request the same, just add natural context.

Output format:
1. [contextualized version 1]
2. [contextualized version 2]

Contextualized versions:"""

        try:
            result = self.model.generate_content(context_prompt)
            lines = [line.strip() for line in result.text.strip().split('\n') if line.strip()]

            variations = []
            import re
            for line in lines:
                clean = re.sub(r'^\d+\.\s*', '', line)
                if clean:
                    variations.append(clean)

            self.stats["contextualized"] += len(variations)
            return variations[:num_variations]

        except Exception as e:
            print(f"Context addition error: {e}")
            return []

    def create_multi_turn(self, prompt: str) -> List[Dict[str, str]]:
        """Convert single prompt into multi-turn conversation.

        Creates a conversation history leading up to the request.
        """
        multi_turn_prompt = f"""Create a realistic 3-turn conversation that leads to this request.

Final user request: "{prompt}"

Generate:
- Turn 1: User's initial related question (benign)
- Turn 2: AI's helpful response
- Turn 3: User's follow-up (the target request)

Output format:
TURN 1 (User): [initial question]
TURN 1 (AI): [helpful response]
TURN 2 (User): [follow-up request - the target]

Conversation:"""

        try:
            result = self.model.generate_content(multi_turn_prompt)

            # Parse conversation
            import re
            turns = re.findall(r'TURN \d+ \((User|AI)\): (.+)', result.text)

            if len(turns) >= 3:
                conversation = {
                    "turn_1_user": turns[0][1] if turns[0][0] == "User" else "",
                    "turn_1_ai": turns[1][1] if turns[1][0] == "AI" else "",
                    "turn_2_user": turns[2][1] if turns[2][0] == "User" else prompt,
                    "full_context": result.text.strip()
                }
                self.stats["multi_turn"] += 1
                return [conversation]

        except Exception as e:
            print(f"Multi-turn error: {e}")

        return []

    def augment_sample(
        self,
        sample: Dict,
        techniques: List[str] = None
    ) -> List[Dict]:
        """Augment a single sample with multiple techniques.

        Args:
            sample: Original sample dict with 'prompt' key
            techniques: List of techniques to use (default: all)

        Returns:
            List of augmented samples
        """
        if techniques is None:
            techniques = ["paraphrase", "adversarial", "context"]

        augmented = []
        original_prompt = sample.get("prompt") or sample.get("original_prompt")

        if not original_prompt:
            return []

        # Apply each technique
        if "paraphrase" in techniques:
            paraphrases = self.paraphrase(original_prompt, num_variations=2)
            for para in paraphrases:
                aug_sample = sample.copy()
                aug_sample["prompt"] = para
                aug_sample["original_prompt"] = original_prompt
                aug_sample["augmentation_type"] = "paraphrase"
                augmented.append(aug_sample)

        if "adversarial" in techniques and sample.get("prompt_harm_label") == "harmful":
            # Only create adversarial versions for harmful prompts
            adversarial = self.generate_adversarial(original_prompt, num_variations=2)
            for adv in adversarial:
                aug_sample = sample.copy()
                aug_sample["prompt"] = adv
                aug_sample["original_prompt"] = original_prompt
                aug_sample["augmentation_type"] = "adversarial"
                augmented.append(aug_sample)

        if "context" in techniques:
            contextualized = self.add_context(original_prompt, num_variations=2)
            for ctx in contextualized:
                aug_sample = sample.copy()
                aug_sample["prompt"] = ctx
                aug_sample["original_prompt"] = original_prompt
                aug_sample["augmentation_type"] = "context"
                augmented.append(aug_sample)

        return augmented

    def augment_dataset(
        self,
        dataset: List[Dict],
        multiplier: int = 3,
        techniques: List[str] = None
    ) -> List[Dict]:
        """Augment entire dataset.

        Args:
            dataset: Original dataset
            multiplier: Target size multiplier (e.g., 3× = 3 variations per sample)
            techniques: Augmentation techniques to use

        Returns:
            Augmented dataset (original + variations)
        """
        print(f"Augmenting {len(dataset)} samples with {multiplier}× multiplier...")

        augmented_dataset = list(dataset)  # Start with originals

        for sample in tqdm(dataset):
            # Generate variations
            variations = self.augment_sample(sample, techniques)

            # Limit to target multiplier
            augmented_dataset.extend(variations[:multiplier])

        print(f"Original: {len(dataset)} → Augmented: {len(augmented_dataset)}")
        return augmented_dataset

    def print_stats(self):
        """Print augmentation statistics."""
        print("\n" + "="*60)
        print("AUGMENTATION STATISTICS")
        print("="*60)
        for technique, count in self.stats.items():
            print(f"{technique.capitalize()}: {count}")
        print("="*60 + "\n")


def main():
    """Main augmentation pipeline."""
    parser = argparse.ArgumentParser(description="Augment safety reasoning dataset")
    parser.add_argument(
        "--input",
        required=True,
        help="Input dataset JSON"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output augmented dataset JSON"
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Size multiplier (default: 3×)"
    )
    parser.add_argument(
        "--techniques",
        nargs="+",
        default=["paraphrase", "adversarial", "context"],
        choices=["paraphrase", "adversarial", "context", "multi_turn"],
        help="Augmentation techniques to use"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.input}...")
    with open(args.input) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples")

    # Initialize augmenter
    augmenter = DatasetAugmenter()

    # Augment
    augmented_dataset = augmenter.augment_dataset(
        dataset,
        multiplier=args.multiplier,
        techniques=args.techniques
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(augmented_dataset, f, indent=2)

    print(f"\n✅ Saved {len(augmented_dataset)} samples to {output_path}")

    # Show examples
    print("\n" + "="*60)
    print("AUGMENTATION EXAMPLES")
    print("="*60)

    # Find an augmented sample
    for sample in augmented_dataset:
        if "augmentation_type" in sample:
            print(f"\nOriginal: {sample.get('original_prompt', 'N/A')[:100]}...")
            print(f"Augmented ({sample['augmentation_type']}): {sample['prompt'][:100]}...")
            break

    augmenter.print_stats()


if __name__ == "__main__":
    main()
