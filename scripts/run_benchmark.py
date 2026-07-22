#!/usr/bin/env python3

import sys, os, argparse
from dotenv import load_dotenv

# Load environment variables.
load_dotenv()

# Add the source folder to the Python path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from benchmark_runner import run_evaluation
from context_strategies import CONTEXT_STRATEGIES
from config import Provider

if __name__ == "__main__":
    # Define command-line arguments.
    parser = argparse.ArgumentParser(description="Run RAG benchmark for any BIRD db_id.")
    parser.add_argument("--db", type=str, required=True, help="db_id in dev.json (e.g. toxicology)")
    parser.add_argument("--arch", type=str, required=True, choices=list(CONTEXT_STRATEGIES.keys()))
    parser.add_argument("--provider", type=str, default=Provider.GOOGLE.value)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--force-thoughts", action="store_true")
    args = parser.parse_args()

    # Run the benchmark.
    run_evaluation(db_name=args.db, arch=args.arch, provider=args.provider, model=args.model, force_thoughts=args.force_thoughts)