import json
from evaluation import execution_accuracy
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session,
    get_spark_sql,
    get_spark_agent,
    run_nl_query,
    process_result,
    print_results,
    save_results,
)
from utils import ensure_sqlite_jdbc_driver, pretty_print_result
from paths import get_logs_folder
from context_strategies import CONTEXT_STRATEGIES
from contextlib import redirect_stdout, redirect_stderr
import os

BENCHMARK_SPEC_FILE = "db/bird-1/dev.json"


def run_evaluation(db_name, arch, provider, model=None, force_thoughts=False):
    import config

    # Enable reasoning generation if requested.
    if force_thoughts:
        config.FORCE_THOUGHT_GENERATION = True

    # Select the context generation strategy.
    context_fn = CONTEXT_STRATEGIES[arch]

    benchmark_spec = json.load(open(BENCHMARK_SPEC_FILE))

    # Keep only queries for the selected database.
    db_queries = [
        item for item in benchmark_spec
        if item.get("db_id") == db_name
    ]

    print(
        f"[Runner] db={db_name} arch={arch} "
        f"provider={provider} queries={len(db_queries)}"
    )

    # Initialize Spark.
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            spark = get_spark_session(
                extra_configs={
                    "spark.jars": jdbc_jar_path,
                    "spark.driver.extraClassPath": jdbc_jar_path,
                }
            )

    # Load database tables.
    load_tables(spark, db_name)

    embedder = None
    if arch == "hybrid_v2":
        from langchain_huggingface import HuggingFaceEmbeddings

        # Load embedding model.
        embedder = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

    accuracies = []
    logs = []

    for item in db_queries:
        query_id = item["question_id"]
        nl_query = item["question"]
        golden_query = item["SQL"]

        llm = get_llm(provider=provider, model=model)
        spark_sql = get_spark_sql()

        # Execute the reference SQL query.
        try:
            ground_truth = spark.sql(golden_query).toPandas()
        except Exception as e:
            print(f"Error executing golden query: {e}")
            continue

        # Build the NL-to-SQL agent.
        agent = get_spark_agent(spark_sql, llm)

        # Generate the RAG context.
        enriched_query = context_fn(
            nl_query,
            query_id,
            llm,
            spark_sql,
            db_name,
            embedder,
        )

        # Run the NL query.
        run_nl_query(agent, enriched_query, llm)

        json_result = process_result()
        json_result["query_id"] = query_id

        execution_acc = 0.0

        # Compare the generated result with the ground truth.
        if json_result.get("execution_status") == "VALID":
            result = json_result.get("query_result")
            execution_acc = execution_accuracy(
                ground_truth,
                result,
            )
            print(f"[QID {query_id}] accuracy: {execution_acc:.2%}")

        elif json_result.get("spark_error"):
            print(
                f"[QID {query_id}] "
                f"\033[91mError: {json_result.get('spark_error')}\033[0m"
            )

        accuracies.append(execution_acc)
        json_result["accuracy"] = execution_acc
        logs.append(json_result)

    # Stop Spark.
    spark.stop()

    if not accuracies:
        print("No queries evaluated.")
        return None

    # Compute summary metrics.
    accuracy_avg = sum(accuracies) / len(accuracies)
    total_in = sum(l.get("input_tokens", 0) for l in logs)
    total_out = sum(l.get("output_tokens", 0) for l in logs)

    final_export = {
        "summary_metrics": {
            "experiment": f"{arch}_{db_name}",
            "evaluated_queries": len(accuracies),
            "average_accuracy": accuracy_avg,
            "total_input_tokens": total_in,
            "total_ouput_tokens": total_out,
            "total_tokens": total_in + total_out,
            "avg_tokens_per_query": (
                (total_in + total_out) / len(logs)
                if logs else 0
            ),
            "provider": provider,
            "model": model,
        },
        "detailed_queries": logs,
    }

    # Save experiment results.
    output_file = f"{arch}_{db_name}.json"

    save_results(
        results=final_export,
        output_file=output_file,
        base_folder=get_logs_folder(db_name),
    )

    print(
        f"\nAverage accuracy: {accuracy_avg:.2%}"
        f"  -> saved to {get_logs_folder(db_name)}/{output_file}"
    )

    return final_export