#!/bin/bash
# Real Estate System - Quick Setup Script for Unix/Mac/Linux
# Run this script to quickly set up and test the Real Estate Pricing Intelligence System

echo "🏠 Real Estate System - Quick Setup"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found!"
    echo "Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python found:"
python3 --version

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found!"
    echo "Please install pip3 first"
    exit 1
fi

echo "✅ pip found"
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv real_estate_env

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source real_estate_env/bin/activate

# Install requirements
echo "📥 Installing Python packages..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"

# Run setup verification
echo "🔍 Running setup verification..."
python check_setup.py

echo ""
echo "🎯 To run the Real Estate System:"
echo "   python final_business_report.py"
echo ""
echo "📖 For more information, see README.md and SETUP_GUIDE.md"
