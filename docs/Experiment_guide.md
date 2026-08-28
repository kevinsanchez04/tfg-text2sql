# Experiment Guide

This guide reproduces every log file and plot referenced in the thesis. All commands assume they are run from the repository root with the Python environment from `requirements.txt` active and a `.env` file configured (see the root `README.md`).

## 1. One-time setup per database

The five databases evaluated are `toxicology`, `financial`, `superhero`, `formula_1` and `codebase_community` (see [`dataset.md`](dataset.md) for how to obtain the BIRD files). For **each** database:

```bash
# Vector index (needed by: vector, hybrid_v1, hybrid_v2 and all its ablations)
python src/vector_rag/build_vector_db.py --db <db_name>

# Property graph (needed by: graph, hybrid_v1, hybrid_v2 and all its ablations)
python src/graph_rag/ast_parser.py --db <db_name>
python src/graph_rag/graph_builder.py --db <db_name>
```

`graph_builder.py` will attempt to start a local Spark session and load the database to perform entity resolution (sampling values for `sample_values`); if PySpark/the JDBC driver cannot be initialised it falls back to building the graph without entity resolution and prints a warning.

## 2. Running a single architecture on a single database

```bash
python scripts/run_benchmark.py --db <db_name> --arch <arch_key> \
    --provider google [--model <model_name>] [--force-thoughts]
```

`<arch_key>` is any key of `CONTEXT_STRATEGIES` (`src/context_strategies.py`):

| Purpose | Keys |
|---|---|
| Main comparison | `baseline`, `vector`, `graph`, `hybrid_v1`, `hybrid_v2` |
| Ablation (leave-one-out of a single V2 component) | `hybrid_v2_no_entity`, `hybrid_v2_no_alpha`, `hybrid_v2_no_negprompt` |
| ALPHA sensitivity sweep | `hybrid_v2_alpha_0.0`, `hybrid_v2_alpha_0.2`, `hybrid_v2_alpha_0.4`, `hybrid_v2_alpha_0.6`, `hybrid_v2_alpha_0.8`, `hybrid_v2_alpha_1.0` |

Results are written to `logs/<db_name>/<arch_key>_<db_name>.json`, containing a `summary_metrics` block (average accuracy, token totals, provider/model) and a `detailed_queries` list with one entry per question (generated SQL, execution status, per-tool/per-LLM-call timings, token usage, chain-of-thought log).

`--force-thoughts` switches the agent's prompt prefix from `SQL_PREFIX` to `SQL_PREFIX_WITH_THOUGHTS` (`spark_toolkit/prompt.py`), forcing the model to emit an explicit `Thought:` block before every tool call.

## 3. Full experiment matrix for the thesis

For each of the 5 databases, run the 5 main architectures plus the 3 single-component ablations plus the 6-point ALPHA sweep (14 runs per database, 70 runs in total):

```bash
for db in toxicology financial superhero formula_1 codebase_community; do
  for arch in baseline vector graph hybrid_v1 hybrid_v2 \
              hybrid_v2_no_entity hybrid_v2_no_alpha hybrid_v2_no_negprompt \
              hybrid_v2_alpha_0.0 hybrid_v2_alpha_0.2 hybrid_v2_alpha_0.4 \
              hybrid_v2_alpha_0.6 hybrid_v2_alpha_0.8 hybrid_v2_alpha_1.0; do
    python scripts/run_benchmark.py --db "$db" --arch "$arch" --provider google
  done
done
```

Runs use `temperature=0.0` by default (`config.DEFAULT_TEMPERATURE`); some residual non-determinism from the underlying LLM API should still be expected and is discussed in the thesis's threats-to-validity section.

## 4. Debugging a single question

To inspect the exact prompt, generated SQL, agent chain-of-thought and accuracy for one question under Hybrid RAG V2:

```bash
python scripts/test_single_query.py --qid <question_id> --db <db_name> \
    [--provider <provider>] [--model <model_name>] [--force-thoughts]
```

`scripts/test_single_query.py` is hard-wired to `hybrid_rag_v2.orchestrator.get_hybrid_context`; swap the import at the top of the script to debug a different architecture the same way.

## 5. Simplifying a log for manual review

```bash
python scripts/logs_summary.py --input logs/<db_name>/<arch>_<db_name>.json \
    --output <output_file>.json --db_id <db_name>
```

Produces a flat list merging each logged query with its natural-language question and golden SQL from `dev.json`, dropping the heavier per-tool/per-LLM-call telemetry — useful for manually skimming errors.

## 6. Generating the comparative plots

```bash
python scripts/generate_plots.py --db <db_name>
```

`generate_plots.py` expects the five main-comparison log files to already exist at `logs/<db_name>/{baseline,vector,graph,hybrid_v1,hybrid_v2}_<db_name>.json` (the ablation and ALPHA-sweep runs are **not** picked up by this script and must be analysed separately). It produces, under `data/plots/<db_name>/`:

- `<db_name>_overall_accuracy.png` — mean Execution Accuracy per architecture.
- `<db_name>_accuracy_by_difficulty.png` — accuracy broken down by the BIRD `difficulty` label (`simple`/`moderate`/`challenging`).
- `<db_name>_baseline_improvement.png` — absolute percentage-point improvement over the Baseline.
