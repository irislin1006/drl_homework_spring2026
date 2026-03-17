#!/usr/bin/env bash
set -e

echo "=== HW2 Environment Setup ==="

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Installing system dependencies (requires sudo)..."
    sudo apt-get update
    sudo apt-get install -y swig python3.10-dev
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing system dependencies via Homebrew..."
    brew install swig cmake
else
    echo "Unsupported OS: $OSTYPE. Please install swig and Python dev headers manually."
    exit 1
fi

echo "Installing Python dependencies with uv..."
uv sync

echo ""
echo "=== Setup complete! ==="
echo "Test with: uv run src/scripts/run.py --env_name CartPole-v0 -n 100 -b 1000 --exp_name cartpole"
