#!/usr/bin/env python3
"""
Test script for natural language to SparkSQL conversion.

Usage:
    python test_nl_query.py
    python test_nl_query.py --force-thoughts
    python test_nl_query.py --provider openai --model o3-mini
    python test_nl_query.py --provider openai --model o3-mini --force-thoughts

Requirements:
    - GOOGLE_API_KEY environment variable (or .env file)
    - OPENAI_API_KEY environment variable (for OpenAI models)
    - Database files in db/bird-1/
"""

import sys
import os

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from config import Provider
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session, get_spark_sql, get_spark_agent,
    run_nl_query, process_result, print_results, AgentMonitoringCallback
)
from utils import ensure_sqlite_jdbc_driver, pretty_print_result
import argparse


# Configuration
DB_NAME = "superhero"
# NL_QUERY = "What are the names of all superheroes who have the power of flight?"
NL_QUERY = "How many superheroes have blue eyes? blue eyes refers to colour = 'Blue' and eye_colour_id = colour.id;"

def main(provider, force_thoughts=False, model=None):
    # Set config flag for thought generation
    if force_thoughts:
        config.FORCE_THOUGHT_GENERATION = True
        print("[Config] Force thought generation: ENABLED")

    print("=" * 60)
    print(" NATURAL LANGUAGE TO SPARKSQL TEST")
    print("=" * 60)
    print(f"\nDatabase: {DB_NAME}")
    print(f"Query: {NL_QUERY}")
    print(f"LLM Provider: {provider}")
    if model:
        print(f"Model: {model}")
    print("=" * 60)

    # Setup
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    spark = get_spark_session(extra_configs={
        "spark.jars": jdbc_jar_path,
        "spark.driver.extraClassPath": jdbc_jar_path,
    })
    load_tables(spark, DB_NAME)

    # Run query
    llm = get_llm(provider=provider, model=model)
    agent = get_spark_agent(get_spark_sql(), llm)
    run_nl_query(agent, NL_QUERY, llm)

    # Display results
    json_result = process_result()

    print("\nINFERRED SPARKSQL QUERY:")
    print("-" * 40)
    sql_query = json_result.get('sparksql_query')
    print(f"\033[94m{sql_query}\033[0m" if sql_query else "\033[91mNo SQL query generated.\033[0m")

    print("\nEXECUTION STATUS:")
    print("-" * 40)
    print_results(json_result, print_result=False)

    print(f"\nQUERY RESULTS: {NL_QUERY}")
    print("-" * 40)
    if json_result.get('execution_status') == "VALID":
        pretty_print_result(json_result.get('query_result'))
    elif json_result.get('spark_error'):
        print(f"\033[91mError: {json_result.get('spark_error')}\033[0m")
    else:
        print("No results available.")
    spark.stop()
    print("\nDone!")
    return json_result


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Test NL to SparkSQL agent.")
    parser.add_argument("--provider", type=str, default=Provider.GOOGLE.value, help="LLM provider (default: google)")
    parser.add_argument("--model", type=str, help="Specific model name (e.g., o1, o3-mini, gpt-4)")
    parser.add_argument("--force-thoughts", action="store_true", help="Force text thought generation before tool calls")
    args = parser.parse_args()
    provider = args.provider
    main(provider, force_thoughts=args.force_thoughts, model=args.model)
