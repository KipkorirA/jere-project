#!/usr/bin/env python3
"""
Real Estate System Setup Verification Script
Run this script to verify your installation is ready to use.
"""

import sys
import os
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python Version...")
    version = sys.version_info
    print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("   ✅ Python version is compatible (3.8+)")
        return True
    else:
        print("   ❌ Python 3.8+ is required")
        return False

def check_packages():
    """Check if required packages are installed"""
    print("\n📦 Checking Required Packages...")
    
    required_packages = {
        'pandas': 'Data manipulation and analysis',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning (scikit-learn)',
        'xgboost': 'Gradient boosting framework',
        'matplotlib': 'Data visualization',
        'seaborn': 'Statistical data visualization',
        'joblib': 'Parallel computing and model serialization'
    }
    
    missing_packages = []
    
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
                print(f"   ✅ {package:<12} - {description}")
            else:
                __import__(package)
                print(f"   ✅ {package:<12} - {description}")
        except ImportError:
            missing_packages.append(package)
            print(f"   ❌ {package:<12} - {description}")
    
    return missing_packages

def check_data_files():
    """Check if required data files are present"""
    print("\n📁 Checking Data Files...")
    
    required_files = {
        'kenya_listings.csv': 'Original real estate dataset',
        'cleaned_real_estate_data.csv': 'Processed dataset (if available)',
        'best_price_model.pkl': 'Trained ML model (if available)',
        'feature_scaler.pkl': 'Feature scaling parameters (if available)'
    }
    
    missing_files = []
    
    for file, description in required_files.items():
        if os.path.exists(file):
            size = os.path.getsize(file)
            size_mb = size / (1024 * 1024)
            print(f"   ✅ {file:<25} - {description} ({size_mb:.1f} MB)")
        else:
            missing_files.append(file)
            print(f"   ⚠️  {file:<25} - {description} (MISSING)")
    
    return missing_files

def check_python_scripts():
    """Check if all Python scripts are present"""
    print("\n🐍 Checking Python Scripts...")
    
    required_scripts = [
        'real_estate_analysis.py',
        'price_prediction_models.py',
        'anomaly_detection.py',
        'final_business_report.py'
    ]
    
    missing_scripts = []
    
    for script in required_scripts:
        if os.path.exists(script):
            print(f"   ✅ {script:<30} - Analysis script")
        else:
            missing_scripts.append(script)
            print(f"   ❌ {script:<30} - Missing script")
    
    return missing_scripts

def check_optional_files():
    """Check for optional but helpful files"""
    print("\n📄 Checking Optional Files...")
    
    optional_files = {
        'requirements.txt': 'Python dependencies list',
        'README.md': 'Project documentation',
        'SETUP_GUIDE.md': 'Setup instructions'
    }
    
    for file, description in optional_files.items():
        if os.path.exists(file):
            print(f"   ✅ {file:<20} - {description}")
        else:
            print(f"   ⚠️  {file:<20} - {description} (MISSING)")

def install_missing_packages(missing_packages):
    """Attempt to install missing packages"""
    if not missing_packages:
        return True
    
    print(f"\n🔧 Attempting to install missing packages: {', '.join(missing_packages)}")
    
    for package in missing_packages:
        try:
            if package == 'sklearn':
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-learn'])
            else:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"   ✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {package}")
            return False
    
    return True

def main():
    """Main verification function"""
    print("="*60)
    print("🏠 REAL ESTATE SYSTEM SETUP VERIFICATION")
    print("="*60)
    
    # Check Python version
    python_ok = check_python_version()
    
    if not python_ok:
        print("\n❌ Python version check failed!")
        print("Please install Python 3.8 or higher from https://python.org")
        return False
    
    # Check packages
    missing_packages = check_packages()
    
    # Check data files
    missing_files = check_data_files()
    
    # Check scripts
    missing_scripts = check_python_scripts()
    
    # Check optional files
    check_optional_files()
    
    # Summary
    print("\n" + "="*60)
    print("📋 SETUP VERIFICATION SUMMARY")
    print("="*60)
    
    if not missing_packages and not missing_files and not missing_scripts:
        print("🎉 CONGRATULATIONS!")
        print("✅ All requirements are satisfied!")
        print("🚀 Your Real Estate Pricing Intelligence System is ready to run!")
        
        print("\nTo get started, run:")
        print("   python final_business_report.py")
        
        print("\nOr run individual components:")
        print("   python real_estate_analysis.py")
        print("   python price_prediction_models.py")
        print("   python anomaly_detection.py")
        
        return True
    else:
        print("⚠️  SETUP ISSUES FOUND:")
        
        if missing_packages:
            print(f"\n❌ Missing Packages ({len(missing_packages)}):")
            for package in missing_packages:
                print(f"   - {package}")
            print(f"\n💡 Fix: pip install -r requirements.txt")
        
        if missing_files:
            print(f"\n⚠️  Missing Data Files ({len(missing_files)}):")
            for file in missing_files:
                print(f"   - {file}")
            print(f"\n💡 Fix: Transfer data files from original project")
        
        if missing_scripts:
            print(f"\n❌ Missing Scripts ({len(missing_scripts)}):")
            for script in missing_scripts:
                print(f"   - {script}")
            print(f"\n💡 Fix: Transfer Python scripts from original project")
        
        print(f"\n📖 For detailed setup instructions, see SETUP_GUIDE.md")
        return False

def quick_fix():
    """Offer to automatically fix common issues"""
    print("\n🔧 QUICK FIX OPTIONS:")
    print("1. Install missing packages")
    print("2. Show setup guide")
    print("3. Exit")
    
    choice = input("\nSelect option (1-3): ").strip()
    
    if choice == "1":
        missing_packages = check_packages()
        if missing_packages:
            install_choice = input(f"Install {len(missing_packages)} missing packages? (y/n): ").strip().lower()
            if install_choice == 'y':
                install_missing_packages(missing_packages)
        else:
            print("All packages are already installed!")
    
    elif choice == "2":
        if os.path.exists("SETUP_GUIDE.md"):
            print("\nOpening SETUP_GUIDE.md...")
            try:
                if sys.platform.startswith('win'):
                    os.startfile("SETUP_GUIDE.md")
                elif sys.platform.startswith('darwin'):
                    subprocess.call(["open", "SETUP_GUIDE.md"])
                else:
                    subprocess.call(["xdg-open", "SETUP_GUIDE.md"])
            except:
                print("Could not open SETUP_GUIDE.md automatically")
                print("Please open it manually in your text editor")
        else:
            print("SETUP_GUIDE.md not found!")
    
    elif choice == "3":
        print("Goodbye!")
        return
    
    else:
        print("Invalid choice!")

if __name__ == "__main__":
    success = main()
    
    if not success:
        quick_fix_option = input("\nWould you like to see quick fix options? (y/n): ").strip().lower()
        if quick_fix_option == 'y':
            quick_fix()
