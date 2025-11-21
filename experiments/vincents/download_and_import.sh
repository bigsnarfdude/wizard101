#!/bin/bash
# Download GuardReasoner-8B from HuggingFace

echo "Installing huggingface_hub..."
pip install huggingface_hub

echo "Downloading GuardReasoner-8B..."
huggingface-cli download yueliu1999/GuardReasoner-8B --local-dir ./guardreasoner-8b

echo "Creating Ollama model..."
ollama create guardreasoner-8b -f Modelfile.guardreasoner

echo "Testing model..."
ollama run guardreasoner-8b "What is 2+2?"

echo "Done!"
