#!/usr/bin/env python3
"""
Financial News Intelligence API - Startup Script
Cross-platform script to launch the LangGraph Multi-Agent System
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header():
    """Print startup header"""
    print("=" * 50)
    print("  Financial News Intelligence API")
    print("  LangGraph Multi-Agent System")
    print("=" * 50)
    print()


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        "fastapi",
        "uvicorn",
        "langgraph",
        "sentence_transformers",
        "chromadb",
        "spacy"
    ]
    
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Installing dependencies from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"])
    else:
        print("✓ All dependencies installed")


def check_spacy_model():
    """Check if spaCy model is downloaded"""
    print("Checking spaCy models...")
    
    try:
        import spacy
        spacy.load("en_core_web_sm")
        print("✓ spaCy model installed")
    except Exception:
        print("Downloading spaCy model...")
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def start_server():
    """Start the FastAPI server"""
    print()
    print("=" * 50)
    print("Starting API server on http://0.0.0.0:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 50)
    print()
    
    # Start uvicorn
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])


def main():
    """Main entry point"""
    print_header()
    
    # Check if we're in the right directory
    if not Path("api.py").exists():
        print("Error: api.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    try:
        check_dependencies()
        check_spacy_model()
        start_server()
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()