@echo off
REM Real Estate System - Quick Setup Script for Windows
REM Run this script to quickly set up and test the Real Estate Pricing Intelligence System

echo 🏠 Real Estate System - Quick Setup
echo ======================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found!
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found:
python --version

REM Check if pip is installed
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ pip not found!
    echo Please ensure pip is installed with Python
    pause
    exit /b 1
)

echo ✅ pip found
echo.

REM Create virtual environment
echo 📦 Creating virtual environment...
python -m venv real_estate_env

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call real_estate_env\Scripts\activate.bat

REM Install requirements
echo 📥 Installing Python packages...
pip install -r requirements.txt

echo.
echo ✅ Installation complete!

REM Run setup verification
echo 🔍 Running setup verification...
python check_setup.py

echo.
echo 🎯 To run the Real Estate System:
echo    python final_business_report.py
echo.
echo 📖 For more information, see README.md and SETUP_GUIDE.md

pause
