#!/bin/bash
# Quick Start Script for Champions League Predictor

echo "⚽ Champions League Predictor - Quick Start"
echo "=========================================="

# Change to project directory
cd "$(dirname "$0")"

echo "📍 Working directory: $(pwd)"

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3."
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Check if streamlit is installed
if ! $PYTHON_CMD -c "import streamlit" &> /dev/null; then
    echo "⚠️  Streamlit not found. Installing..."
    $PYTHON_CMD -m pip install streamlit
fi

# Generate sample data if not exists
if [ ! -f "data/raw/sample_matches.csv" ]; then
    echo "📊 Generating sample data..."
    $PYTHON_CMD scripts/generate_sample_data.py
fi

echo "🚀 Starting Streamlit app..."
echo "🌐 Opening http://localhost:8501 in your browser..."

# Start the app
$PYTHON_CMD -m streamlit run simple_app.py --server.port=8501

echo "✅ App stopped."
