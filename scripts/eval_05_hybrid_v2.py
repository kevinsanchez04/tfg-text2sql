#!/usr/bin/env python3

import sys
import os
import json
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
    run_nl_query, process_result, print_results, save_results
)
from utils import ensure_sqlite_jdbc_driver, pretty_print_result
import argparse

from hybrid_rag_v2.orchestrator import get_hybrid_context
from langchain_huggingface import HuggingFaceEmbeddings

BENCHMARK_SPEC_FILE = "db/bird-1/dev.json"

def main(provider, force_thoughts=False, model=None):
    if force_thoughts:
        config.FORCE_THOUGHT_GENERATION = True
        print("[Config] Force thought generation: ENABLED")

    print("=" * 60)
    print(" Hybrid RAG V2 (Pruned) benchmarking test")
    print("=" * 60)

    benchmark_spec = json.load(open(BENCHMARK_SPEC_FILE))
    db_name = "toxicology"
    toxicology_queries = [ item for item in benchmark_spec if 195 <= item['question_id'] <= 339]

    # Setup Spark
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            spark = get_spark_session(extra_configs={
                "spark.jars": jdbc_jar_path,
                "spark.driver.extraClassPath": jdbc_jar_path,
            })
    load_tables(spark, db_name)

    # ------------------------------------------------------------------
    # 🌟 OPTIMIZACIÓN CRÍTICA: Inicializamos una sola vez FUERA del bucle
    # ------------------------------------------------------------------
    print("[Setup] Inicializando infraestructura del Agente...")
    llm = get_llm(provider=provider, model=model)
    spark_sql = get_spark_sql()
    agent = get_spark_agent(spark_sql, llm)
    embedder=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    accuracies = []
    logs = []

    # Run query
    for item in toxicology_queries:
        query_id = item['question_id']
        nl_query = item['question']
        golden_query = item['SQL']
        
        # Obtenemos ground truth
        try:
            ground_truth = spark.sql(golden_query).toPandas()
        except Exception as e:
            print(f"Error executing golden query: {e}")
            continue

        enriched_query = get_hybrid_context(nl_query=nl_query, query_id=query_id, llm=llm, spark_sql=spark_sql, embedder=embedder)

        run_nl_query(agent, enriched_query, llm)

        json_result = process_result()
        json_result['query_id'] = query_id

        print(f"\n[QID {query_id}] INFERRED SPARKSQL QUERY:")
        sql_query = json_result.get('sparksql_query', '')
        print(f"\033[94m{sql_query}\033[0m" if sql_query else "\033[91mNo SQL query generated.\033[0m")

        execution_acc = 0.0
        if json_result.get('execution_status') == "VALID":
            result = json_result.get('query_result')
            execution_acc = execution_accuracy(ground_truth, result)
            print(f"Result accuracy: {execution_acc:.2%}")
        elif json_result.get('spark_error'):
            print(f"\033[91mError: {json_result.get('spark_error')}\033[0m")

        accuracies.append(execution_acc)
        json_result['accuracy'] = execution_acc 
        logs.append(json_result)

    spark.stop()
    
    if accuracies:
        accuracy_avg = sum(accuracies) / len(accuracies)
        print("\n" + "=" * 60)
        print(f" EVALUATION FINISHED - RAG TOXICOLOGY V2")
        print(f" Average accuracy: {accuracy_avg:.2%}")
        print("=" * 60)

        total_in_tokens = sum(log.get('input_tokens', 0) for log in logs)
        total_out_tokens = sum(log.get('output_tokens',0) for log in logs)
        total_tokens = total_in_tokens + total_out_tokens

        final_export = {
            "summary_metrics": {
                "experiment": "hybrid_toxicology_v2_pruned",
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
        
        save_results(results=final_export, output_file="hybrid_v2_toxicology.json", base_folder="logs")
        print("All results saved successfully in 'logs/hybrid_v2_toxicology.json'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test NL to SparkSQL agent WITH RAG V2.")
    parser.add_argument("--provider", type=str, default=Provider.GOOGLE.value, help="LLM provider")
    parser.add_argument("--model", type=str, help="Specific model name")
    parser.add_argument("--force-thoughts", action="store_true", help="Force text thoughts")
    args = parser.parse_args()
    main(args.provider, force_thoughts=args.force_thoughts, model=args.model)