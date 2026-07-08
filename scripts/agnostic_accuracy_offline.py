#!/usr/bin/env python3

import os
import sys
import json
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import ensure_sqlite_jdbc_driver
from spark_nl import get_spark_session
from load_db import load_tables

NULL_TOKEN = "__NULL__"

def to_dataframe(obj):
    """Safely converts various input types to a Pandas DataFrame."""
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj
    else:
        return pd.DataFrame(obj)

def normalize_value(val):
    """Normalizes cell values to handle typing inconsistencies."""
    if pd.isna(val): 
        return NULL_TOKEN
    
    try:
        if isinstance(val, str): 
            return str(int(val.strip()))
    except ValueError:
        pass
        
    try:
        f_val = float(val)
        if f_val.is_integer(): 
            return str(int(f_val))
    except (ValueError, TypeError):
        pass
        
    try:
        str_val = str(val).strip().lower()
        if str_val in {"none", "null", "nan", "na", "n/a"}: 
            return NULL_TOKEN
        return str(val).strip()
    except Exception:
        pass
        
    return str(val)

def extract_relaxed_column(df, col_index):
    """
    Data normalization, removal of duplicates, and sorting.
    This prevents false negatives caused by missing ORDER BY or extra DISTINCT clauses.
    """
    raw_values = [normalize_value(v) for v in df.iloc[:, col_index]]
    # Lowercase strings to prevent case-sensitivity false negatives (e.g., 'C' vs 'c')
    lowered_values = [v.lower() if isinstance(v, str) else v for v in raw_values]
    return tuple(sorted(list(set(lowered_values))))

def check_relaxed_accuracy(truth_df, pred_df):
    """Compares two dataframes using multiset semantics (ignoring row order and duplicates)."""
    truth_df = to_dataframe(truth_df)
    pred_df = to_dataframe(pred_df)

    if truth_df.empty and pred_df.empty: 
        return 1.0
    if truth_df.empty or pred_df.empty: 
        return 0.0

    # Extract parsed columns from the generated output
    pred_columns = [extract_relaxed_column(pred_df, i) for i in range(pred_df.shape[1])]
    used_columns = [False] * len(pred_columns)

    # Cross-reference with the ground truth columns
    for col_index in range(truth_df.shape[1]):
        truth_col_data = extract_relaxed_column(truth_df, col_index)
        match_found = False
        
        for i in range(len(pred_columns)):
            if not used_columns[i] and pred_columns[i] == truth_col_data:
                used_columns[i] = True
                match_found = True
                break
                
        if not match_found:
            return 0.0
            
    return 1.0

def main():
    log_files = [
        "logs/simplified_analysis_baseline.json", 
        "logs/simplified_analysis_rag.json",
        "logs/simplified_graph_toxicology.json",
        "logs/simplified_hybrid_toxicology.json",
        "logs/simplified_hybrid_v2_toxicology.json"
    ]
    
    print("[*] Starting local Spark session (Offline Mode)...")
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    
    # Initialize Spark silently
    try:
        with open(os.devnull, "w") as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                spark = get_spark_session(extra_configs={
                    "spark.jars": jdbc_jar_path, 
                    "spark.driver.extraClassPath": jdbc_jar_path
                })
        load_tables(spark, "toxicology")
    except Exception as e:
        print(f"[!] Failed to initialize Spark: {e}")
        sys.exit(1)

    try:
        for file_path in log_files:
            if not os.path.exists(file_path):
                print(f"[-] Skipping {file_path} (File not found)")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                parsed_json = json.load(f)

            # Handle both full logs and simplified lists
            queries = parsed_json.get("detailed_queries", parsed_json) if isinstance(parsed_json, dict) else parsed_json

            print(f"\n[+] Analyzing: {os.path.basename(file_path)}")
            
            new_accuracies = []
            original_accuracies = []
            
            for item in queries:
                ground_truth_sql = item.get("golden_query", "")
                predicted_sql = item.get("generated_query", item.get("sparksql_query", ""))
                
                # Fetch original accuracy from the log, default to 0 if missing
                orig_acc = item.get("accuracy", 0.0)
                original_accuracies.append(orig_acc)
                
                if not ground_truth_sql or not predicted_sql:
                    new_accuracies.append(0.0)
                    continue

                try:
                    # Execute queries locally
                    truth_data = spark.sql(ground_truth_sql).toPandas()
                    pred_data = spark.sql(predicted_sql).toPandas()
                    
                    # Evaluate using relaxed metrics
                    acc_score = check_relaxed_accuracy(truth_data, pred_data)
                    new_accuracies.append(acc_score)
                except Exception:
                    # Catch syntax errors or runtime exceptions from bad generated SQL
                    new_accuracies.append(0.0)

            if not new_accuracies:
                print("    => No valid queries found to process.")
                continue

            # Calculate and format the averages
            avg_original = sum(original_accuracies) / len(original_accuracies)
            avg_new = sum(new_accuracies) / len(new_accuracies)
            diff = avg_new - avg_original
            
            sign = "+" if diff >= 0 else ""
            
            print(f"    => Original Strict Accuracy: {avg_original:.2%}")
            print(f"    => New Relaxed Accuracy:     {avg_new:.2%} ({sign}{diff:.2%})")

    except KeyboardInterrupt:
        print("\n\n[!] Execution interrupted by user (Ctrl+C). Cleaning up...")
    finally:
        print("[*] Shutting down Spark session...")
        spark.stop()
        print("[*] Done.")

if __name__ == "__main__":
    main()