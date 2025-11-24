#!/bin/bash
# Cascade DLP v2 - Installation Script

set -e

echo "Installing Cascade DLP v2..."
echo "=============================="

# Check Python version
python3 --version || { echo "Python 3 required"; exit 1; }

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install spacy model for Presidio
echo "Installing spaCy model..."
python -m spacy download en_core_web_lg

echo ""
echo "=============================="
echo "Installation complete!"
echo ""
echo "To activate: source venv/bin/activate"
echo "To run example: python example.py"
echo "=============================="
