#!/bin/bash

echo "🚀 Deploying UCL Predictor to GitHub..."

# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "🏆 Champions League Predictor 2025/26 - Real fixtures & teams by Arav Mathur 😎"

# Set main branch
git branch -M main

# Add your GitHub repository (replace YOUR_USERNAME with your actual GitHub username)
echo "📝 Replace YOUR_USERNAME in the next command with your GitHub username"
echo "git remote add origin https://github.com/YOUR_USERNAME/ucl-predictor.git"
echo "git push -u origin main"

echo ""
echo "✅ After creating the GitHub repo, run:"
echo "git remote add origin https://github.com/YOUR_USERNAME/ucl-predictor.git"
echo "git push -u origin main"
