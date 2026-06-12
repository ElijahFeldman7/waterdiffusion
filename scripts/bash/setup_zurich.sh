#!/bin/bash
# Setup script for Zurich watermark evaluation

set -e

echo "Setting up Zurich watermark evaluation environment..."

# Check if we're in the right directory
if [ ! -d "zurich" ]; then
    echo "Error: zurich directory not found. Please run from /Users/eli/research/waterdiffusion"
    exit 1
fi

cd zurich

# Install Zurich dependencies
echo "Installing Zurich dependencies..."
pip install -q -r requirements.txt

# Install the Zurich package in development mode
echo "Installing Zurich package..."
pip install -e .

echo ""
echo "✓ Setup complete! You can now run the watermark evaluation."
echo ""
echo "Example usage:"
echo "python run_zurich_simple.py --num_samples 5 --delta_values 0.5 1.0 --output test_results.jsonl"
