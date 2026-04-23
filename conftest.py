"""
conftest.py — pytest configuration
Adds src/ to sys.path so tests can import project modules directly.
"""
import sys
import os

# Make src/ importable from any test file
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
