#!/usr/bin/env python3

import sys
import os
import json
from contextlib import redirect_stdout, redirect_stderr

from dotenv import load_dotenv
load_dotenv()

# Ajustamos el path para que mire dentro de 'src'
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hybrid_rag_v2.orchestrator import get_hybrid_context
from config import Provider
from llm import get_llm
from load_db import load_tables
from spark_nl import (
    get_spark_session, get_spark_sql, get_spark_agent,
    run_nl_query, process_result
)
from utils import ensure_sqlite_jdbc_driver

def main():
    test_file = "16_failed_queries.json"
    output_file = "logs/test_16_queries_results.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if not os.path.exists(test_file):
        print(f"❌ Error: No se encontró '{test_file}'.")
        sys.exit(1)

    with open(test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    queries = data.get("detailed_queries", data) if isinstance(data, dict) else data

    print("Iniciando Spark para el test (puede tardar unos segundos)...")
    jdbc_jar_path = ensure_sqlite_jdbc_driver()
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            spark = get_spark_session(extra_configs={
                "spark.jars": jdbc_jar_path,
                "spark.driver.extraClassPath": jdbc_jar_path,
            })
    load_tables(spark, "toxicology")

    # ------------------------------------------------------------------
    # 🌟 OPTIMIZACIÓN CRÍTICA: Inicializamos una sola vez FUERA del bucle
    # ------------------------------------------------------------------
    print("[Setup] Inicializando el LLM y el Agente de Spark...")
    llm = get_llm(provider=Provider.GOOGLE.value)
    spark_sql = get_spark_sql()
    agent = get_spark_agent(spark_sql, llm)

    print(f"\n🚀 Iniciando Prueba V2 con {len(queries)} consultas...\n")
    print("=" * 60)

    results_log = []

    for i, item in enumerate(queries, 1):
        query_id = item.get('question_id', i)
        nl_query = item.get('question', item.get('nl_query', ''))
        golden_query = item.get('SQL', item.get('golden_query', ''))
        
        print(f"[{i}/{len(queries)}] Procesando QID: {query_id}...", end=" ")
        sys.stdout.flush()
        
        try:
            # Obtenemos el contexto híbrido V2 usando el agente persistente
            hybrid_context = get_hybrid_context(
                nl_query=nl_query, 
                query_id=query_id, 
                llm=llm, 
                spark_sql=spark_sql
            )
            
            enriched_query = f"{hybrid_context}\n\nUse this context to solve the following question:\nCurrent Question: {nl_query}"

            # Lanzamos la consulta al agente único
            run_nl_query(agent, enriched_query, llm)
            json_result = process_result()
            
            generated_sql = json_result.get('sparksql_query', '') or ""
            
            # Evaluación visual rápida
            status_msg = "UNKNOWN"
            if "molecule" in generated_sql.lower() and "molecule" not in golden_query.lower():
                status_msg = "ALERTA (Context Overload)"
                print("⚠️ Context Overload")
            elif "molecule" not in generated_sql.lower() and "molecule" not in golden_query.lower():
                status_msg = "ÉXITO (Pruner funcionó)"
                print("✅ Éxito")
            else:
                status_msg = "INFO (Revisar SQL)"
                print("ℹ️ Info")

            results_log.append({
                "question_id": query_id,
                "question": nl_query,
                "golden_query": golden_query,
                "generated_sql": generated_sql,
                "evaluation_status": status_msg,
                "error": None
            })
                
        except Exception as e:
            print(f"❌ Error: {e}")
            results_log.append({
                "question_id": query_id,
                "question": nl_query,
                "golden_query": golden_query,
                "generated_sql": None,
                "evaluation_status": "ERROR",
                "error": str(e)
            })

    spark.stop()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, indent=4, ensure_ascii=False)
        
    print("=" * 60)
    print(f"📁 ¡Test terminado con éxito! Resultados en: {output_file}")

if __name__ == "__main__":
    main()