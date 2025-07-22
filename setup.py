#!/usr/bin/env python3
"""
Setup script for the Crypto Trading AI Agent.

This script helps users set up the trading agent with proper configuration.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def print_banner():
    """Print setup banner."""
    print("=" * 60)
    print("🚀 Crypto Trading AI Agent Setup")
    print("=" * 60)
    print()


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")


def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)


def setup_configuration():
    """Setup configuration files."""
    print("⚙️  Setting up configuration...")
    
    config_example = Path("config/config.example.yaml")
    config_file = Path("config/config.yaml")
    
    if not config_example.exists():
        print("❌ Configuration example file not found")
        sys.exit(1)
    
    if config_file.exists():
        print("⚠️  Configuration file already exists, skipping...")
    else:
        shutil.copy(config_example, config_file)
        print("✅ Configuration file created")
        print("📝 Please edit config/config.yaml with your API keys and settings")


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = ["logs", "data", "exports"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created {directory}/ directory")


def run_tests():
    """Run basic tests."""
    print("🧪 Running basic tests...")
    
    try:
        subprocess.check_call([sys.executable, "test_agent.py"])
        print("✅ Tests completed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Tests failed, but setup can continue")
        print("   You can run tests later with: python test_agent.py")


def print_next_steps():
    """Print next steps for the user."""
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Edit config/config.yaml with your API keys")
    print("2. Configure your trading strategy parameters")
    print("3. Set up Gaia API credentials (for AI inference)")
    print("4. Run the agent: python src/main.py")
    print("5. For testing: python test_agent.py")
    print()
    print("Documentation:")
    print("- README.md for detailed instructions")
    print("- config/config.example.yaml for configuration options")
    print()
    print("Competition Features:")
    print("- The agent is ready for competition deployment")
    print("- Gaia integration provides AI-powered trading decisions")
    print("- Performance tracking and risk management included")
    print()


def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Setup configuration
    setup_configuration()
    
    # Create directories
    create_directories()
    
    # Run tests
    run_tests()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 