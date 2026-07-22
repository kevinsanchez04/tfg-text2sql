#!/usr/bin/env python3
"""
Test script for natural language to SparkSQL conversion (Hybrid RAG V2).
Includes execution accuracy validation against the golden query.

Usage:
    python scripts/test_single_query.py --qid 210
    python scripts/test_single_query.py --qid 215 --provider openai --model o3-mini

Requirements:
    - GOOGLE_API_KEY environment variable (or .env file)
    - Database files in db/bird-1/
"""

import sys
import os
import json
import argparse
import ast
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import numpy as np
import pyspark.sql

from dotenv import load_dotenv
load_dotenv()

# Adjust path to import custom modules from 'src'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from config import Provider
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session, get_spark_sql, get_spark_agent,
    run_nl_query, process_result, print_results
)
from utils import ensure_sqlite_jdbc_driver, pretty_print_result


from hybrid_rag_v2.orchestrator import get_hybrid_context

#from hybrid_rag_v1.orchestrator import get_hybrid_context

# Configuration
BENCHMARK_FILE = "db/bird-1/dev.json"
NULL_TOKEN = "__NULL__"


def convert_to_dataframe(obj):
    """Converts various inputs into a Pandas DataFrame."""
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, pyspark.sql.DataFrame):
        return obj.toPandas()
    else:
        return pd.DataFrame(obj)

def _normalize_value(v):
    """Normalize a single value to a canonical string representation."""
    if pd.isna(v):
        return NULL_TOKEN

    try:
        if isinstance(v, str):
            v = v.strip()
            i = int(v)
            return str(i)
    except Exception:
        pass

    try:
        f = float(v)
        if float(f).is_integer():
            return str(int(f))
    except Exception:
        pass

    try:
        s = str(v).strip()
        s_low = s.lower()

        if s_low in {"none", "null", "nan", "na", "n/a"}:
            return NULL_TOKEN

        return s
    except Exception:
        pass

    return str(v)

def _get_column_values(df, col_idx):
    """Return a tuple of normalized values for the given column index."""
    return tuple(_normalize_value(v) for v in df.iloc[:, col_idx])

def execution_accuracy(df_gt, df_inf):
    """
    Compute execution accuracy based on the Spider 2.0-lite definition.
    Returns 1.0 if all gold columns are present in inferred, 0.0 otherwise.
    """
    df_gt = convert_to_dataframe(df_gt)
    df_inf = convert_to_dataframe(df_inf)

    if df_gt.empty and df_inf.empty:
        return 1.0

    if df_gt.empty or df_inf.empty:
        return 0.0

    # Build a list of inferred columns (as tuples of normalized values)
    inf_cols = [_get_column_values(df_inf, col_idx) for col_idx in range(df_inf.shape[1])]

    # For each gold column, find a matching inferred column (multiset semantics)
    used = [False] * len(inf_cols)
    for col_idx in range(df_gt.shape[1]):
        gt_col_values = _get_column_values(df_gt, col_idx)
        found = False
        for i in range(len(inf_cols)):
            if not used[i] and inf_cols[i] == gt_col_values:
                used[i] = True
                found = True
                break
        if not found:
            return 0.0

    return 1.0




def get_query_details(query_id, db_name):
    """Fetches the natural language question and golden SQL from the dataset."""
    if not os.path.exists(BENCHMARK_FILE):
        raise FileNotFoundError(f"Dataset not found at: {BENCHMARK_FILE}")
        
    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        if item.get('question_id') == query_id and item.get('db_id') == db_name:
            return item.get('question'), item.get('SQL')
            
    raise ValueError(f"QID {query_id} not found for db '{db_name}'.")

def main(provider, query_id, db_name, force_thoughts=False, model=None):
    if force_thoughts:
        config.FORCE_THOUGHT_GENERATION = True
        print("[Config] Force thought generation: ENABLED")

    # Fetch original question and expected SQL
    try:
        nl_query, golden_sql = get_query_details(query_id, db_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("=" * 60)
    print(" HYBRID RAG V2 - SINGLE QUERY TEST ")
    print("=" * 60)
    print(f"Database: {db_name}")
    print("=" * 60)

    # Setup Local Spark Session
    print("[*] Starting Local Spark Session...")
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    spark = get_spark_session(extra_configs={
        "spark.jars": jdbc_jar_path,
        "spark.driver.extraClassPath": jdbc_jar_path,
    })
    load_tables(spark, db_name)

    # Setup Agent & LLM
    print(f"[*] Initializing LLM ({provider}) and Spark Agent...")
    llm = get_llm(provider=provider, model=model)
    spark_sql = get_spark_sql()
    agent = get_spark_agent(spark_sql, llm)

    # Generate Hybrid Context V2
    print("[*] Generating Hybrid RAG V2 Context...")
    enriched_query = get_hybrid_context(
        nl_query=nl_query, 
        query_id=query_id, 
        llm=llm, 
        spark_sql=spark_sql,
        embedder=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
        db_name=db_name
    )


    # Execute Agent
    print("[*] Executing LLM Agent...")
    run_nl_query(agent, enriched_query, llm)
    json_result = process_result()

    # Display Generated SQL
    print("\n" + "="*60)
    print(f"Query ID: {query_id}")
    print(f"Question: {nl_query}")
    print(f"Golden SQL: \n\033[92m{golden_sql}\033[0m")
    print("INFERRED SPARKSQL QUERY:")
    print("-" * 40)
    sql_query = json_result.get('sparksql_query')
    print(f"\033[94m{sql_query}\033[0m" if sql_query else "\033[91mNo SQL query generated.\033[0m")

    # Evaluate Accuracy
    print("\n" + "="*60)
    print("ACCURACY EVALUATION:")
    print("-" * 40)
    accuracy_score = 0.0
    
    if sql_query and json_result.get('execution_status') == "VALID":
        try:
            # Execute Golden SQL
            df_golden = spark.sql(golden_sql).toPandas()
            # Execute Generated SQL
            df_generated = spark.sql(sql_query).toPandas()
            # Compute Match
            accuracy_score = execution_accuracy(df_golden, df_generated)
            print(f"Accuracy Score: \033[92m{accuracy_score * 100:.2f}%\033[0m")
        except Exception as e:
            print(f"\033[91m Error evaluating accuracy (likely malformed generated SQL): {e}\033[0m")
    else:
        print("\033[91mAccuracy Score: 0.00% (Execution Failed or No Query)\033[0m")

    # Display Data Result
    """print(f"\nDATA RESULTS FOR QID {query_id}:")
    print("-" * 40)
    if json_result.get('execution_status') == "VALID":
        pretty_print_result(json_result.get('query_result'))
    elif json_result.get('spark_error'):
        print(f"\033[91mSpark Error: {json_result.get('spark_error')}\033[0m")
    else:
        print("No results available.")"""
    
    # Prompt which the LLM will see
    print("\n" + "="*60)
    print("EXACT PROMPT SENT TO THE LLM:")
    print("="*60)
    print(enriched_query)
    print("="*60 + "\n")
        
    # Cleanup
    spark.stop()
    print("\nDone!")
    return json_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NL to SparkSQL agent using Hybrid V2.")
    parser.add_argument("--qid", type=int, required=True, help="Question ID to evaluate (e.g. 210)")
    parser.add_argument("--db", type=str, required=True, help="db_id (e.g. toxicology)")
    parser.add_argument("--provider", type=str, default=Provider.GOOGLE.value, help="LLM provider (default: google)")
    parser.add_argument("--model", type=str, help="Specific model name (e.g., o1, o3-mini, gpt-4)")
    parser.add_argument("--force-thoughts", action="store_true", help="Force text thought generation before tool calls")
    
    args = parser.parse_args()
    
    main(
        provider=args.provider, 
        query_id=args.qid,
        db_name=args.db,
        force_thoughts=args.force_thoughts, 
        model=args.model
    )