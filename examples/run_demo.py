#!/usr/bin/env python3
"""Run the model-serving demo (offline). PYTHONPATH=src python examples/run_demo.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_serving.demo import run_demo
if __name__ == "__main__":
    run_demo()
