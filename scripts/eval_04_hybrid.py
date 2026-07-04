#!/usr/bin/env python3

import sys
import os
import json
import chromadb
from contextlib import redirect_stdout, redirect_stderr

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from config import Provider
from evaluation import execution_accuracy
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session, get_spark_sql, get_spark_agent,
    run_nl_query, process_result, print_results, AgentMonitoringCallback,
    save_results
)
from utils import ensure_sqlite_jdbc_driver, pretty_print_result
import argparse
from hybrid_rag.orchestrator import get_hybrid_context


# Configuration
BENCHMARK_SPEC_FILE = "db/bird-1/dev.json"

def main(provider, force_thoughts=False, model=None):
    # Set config flag for thought generation
    if force_thoughts:
        config.FORCE_THOUGHT_GENERATION = True
        print("[Config] Force thought generation: ENABLED")

    print("=" * 60)
    print(" Hybrid Query benchmarking test")
    print("=" * 60)

    benchmark_spec = json.load(open(BENCHMARK_SPEC_FILE))
    db_name = "toxicology"
    toxicology_queries = [ item for item in benchmark_spec if 195 <= item['question_id'] <= 339]

    print(f"\nDatabase: {db_name}")
    print(f"Queries: {len(toxicology_queries)} queries of the range 195-339")
    print(f"LLM Provider: {provider}")
    if model:
        print(f"Model: {model}")
    print("=" * 60)

    # Setup Spark
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            spark = get_spark_session(extra_configs={
                "spark.jars": jdbc_jar_path,
                "spark.driver.extraClassPath": jdbc_jar_path,
            })
        print("[Setup] Spark session created with SQLite JDBC driver.")
    load_tables(spark, db_name)

    accuracies = []
    logs = []

    # Run query
    for item in toxicology_queries:
        query_id = item['question_id']
        nl_query = item['question']
        golden_query = item['SQL']
        
        llm = get_llm(provider=provider, model=model)
        spark_sql = get_spark_sql()

        # Get ground truth (expected result)
        try:
            ground_truth = spark.sql(golden_query).toPandas()
        except Exception as e:
            print(f"Error executing golden query: {e}")
            continue
        
        agent = get_spark_agent(spark_sql, llm)

        hybrid_context = get_hybrid_context(nl_query=nl_query, query_id=query_id, llm=llm, spark_sql=spark_sql)

        enriched_query = f"{hybrid_context}\n\nUse this context to solve the following question:\nCurrent Question: {nl_query}"

        run_nl_query(agent, enriched_query, llm)

        # Display results
        json_result = process_result()
        json_result['query_id'] = query_id

        print("\nINFERRED SPARKSQL QUERY:")
        print("-" * 40)
        sql_query = json_result.get('sparksql_query')
        print(f"\033[94m{sql_query}\033[0m" if sql_query else "\033[91mNo SQL query generated.\033[0m")

        print("\nEXECUTION STATUS:")
        print("-" * 40)
        print_results(json_result, print_result=False)

        print(f"\nQUERY RESULTS: {nl_query}")
        print("-" * 40)

        execution_acc = 0.0
        if json_result.get('execution_status') == "VALID":
            print("Got result:")
            result = json_result.get('query_result')
            pretty_print_result(result)
            execution_acc = execution_accuracy(ground_truth, result)
            print(f"Result accuracy: {execution_acc:.2%}")
        elif json_result.get('spark_error'):
            print(f"\033[91mError: {json_result.get('spark_error')}\033[0m")
        else:
            print("No results available.")

        accuracies.append(execution_acc)
        json_result['accuracy'] = execution_acc 
        logs.append(json_result)

    spark.stop()
    if accuracies:
        accuracy_avg = sum(accuracies) / len(accuracies)
        print("\n" + "=" * 60)
        print(f" EVALUATION FINISHED - RAG TOXICOLOGY")
        print(f" Evaluated queries: {len(accuracies)}")
        print(f" Average accuracy: {accuracy_avg:.2%}")
        print("=" * 60)

        total_in_tokens = sum(log.get('input_tokens', 0) for log in logs)
        total_out_tokens = sum(log.get('output_tokens',0) for log in logs)
        total_tokens = total_in_tokens + total_out_tokens

        final_export = {
            "summary_metrics": {
                "experiment": "hybrid_toxicology",
                "evaluated_queries": len(accuracies),
                "average_accuracy": accuracy_avg,
                "total_input_tokens": total_in_tokens,
                "total_ouput_tokens": total_out_tokens,
                "total_tokens": total_tokens,
                "avg_tokens_per_query": total_tokens/len(logs) if logs else 0,
                "provider": provider,
                "model": model
            },
            "detailed_queries": logs 
        }
        
        save_results(results=final_export, output_file="hybrid_toxicology.json", base_folder="logs")
        print("All results saved successfully in 'logs/hybrid_toxicology.json'")
    
    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NL to SparkSQL agent WITH RAG.")
    parser.add_argument("--provider", type=str, default=Provider.GOOGLE.value, help="LLM provider (default: google)")
    parser.add_argument("--model", type=str, help="Specific model name (e.g., o1, o3-mini, gpt-4)")
    parser.add_argument("--force-thoughts", action="store_true", help="Force text thought generation before tool calls")
    args = parser.parse_args()
    provider = args.provider
    main(provider, force_thoughts=args.force_thoughts, model=args.model)