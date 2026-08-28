# Architecture

This document describes how the five RAG architectures, the ablation variants and the SparkSQL agent fit together, and defines the single extension point for adding a new architecture.

## 1. High-level pipeline

For every benchmark question, `src/benchmark_runner.py` (or `scripts/test_single_query.py` for a single question) performs the same four steps regardless of architecture:

1. **Build a fresh `SparkSQL` wrapper and LLM client** for the question (`get_spark_sql`, `get_llm`).
2. **Build the enriched prompt** by calling the architecture's context function (see §2), registered in `CONTEXT_STRATEGIES` (`src/context_strategies.py`).
3. **Run the LangGraph ReAct agent** (`spark_toolkit/base.py::create_spark_sql_agent`) on the enriched prompt (`spark_nl.py::run_nl_query`).
4. **Score the result** against the golden query with `execution_accuracy` (`src/evaluation.py`).

```
question ──▶ context_fn(nl_query, query_id, llm, spark_sql, db_name, embedder)
                    │
                    ▼
           enriched prompt (string)
                    │
                    ▼
        LangGraph ReAct agent (spark_toolkit)
                    │
        query_sql_db / schema_sql_db / list_tables_sql_db / query_checker_sql_db
                    │
                    ▼
              Spark SQL result ──▶ execution_accuracy(golden, result)
```

## 2. The extension point: `context_strategies.py`

Every architecture is a plain function with the signature

```python
def context_fn(nl_query: str, query_id: int, llm, spark_sql, db_name: str, embedder=None) -> str
```

registered in the `CONTEXT_STRATEGIES` dict. This is the **only** seam `scripts/run_benchmark.py` and `src/benchmark_runner.py` depend on — adding a new architecture means writing one such function and one dict entry; nothing else in the pipeline needs to change.

## 3. The five architectures

### 3.1 Baseline (Zero-Shot) — `baseline_context`

Returns `nl_query` unmodified. No retrieval of any kind. This is the lower bound against which every other architecture is compared.

### 3.2 Vector RAG — `vector_context`

Calls `vector_rag/retriever.py::get_similar_queries_context`, which queries a per-database ChromaDB collection (`db/vector_store/<db_name>`, populated by `vector_rag/build_vector_db.py`) for the `top_k=3` questions most similar to `nl_query`, excluding the question currently being evaluated (`exclude_query_id`, i.e. leave-one-out) to prevent data leakage. Each result contributes its natural-language question and its golden SQL as a few-shot example.

### 3.3 Graph RAG — `graph_context`

Two steps:

1. **Schema linking** (`graph_rag/schema_linker.py::extract_tables_from_query`): an LLM call, prompted with the column names and sample values stored in the property graph, decides which tables are strictly needed to answer the question.
2. **Graph navigation** (`graph_rag/graph_navigator.py::graph_navigator`): for every pair of selected tables, enumerates simple paths (`nx.all_simple_paths`, `cutoff=3`) in the property graph and keeps the path with the highest cumulative historical edge weight. The winning paths' join conditions and per-table historical metadata (top columns, aggregations used, common filters) are formatted into the prompt.

The **property graph** itself is built offline in two stages:

- `graph_rag/ast_parser.py`: parses every golden query of a database with `sqlglot`, extracting, per table, the join edges (explicit `JOIN ... ON` and implicit `WHERE` equi-joins), the columns referenced, and structural patterns (`WHERE`/`GROUP BY`/`ORDER BY`/`LIMIT` usage, aggregation functions, filter predicates).
- `graph_rag/graph_builder.py`: aggregates those per-query ASTs into a single `networkx.DiGraph`, accumulating frequency counters on both nodes (`columns_freq`, `operations_freq`, `aggregations_freq`, `predicates_freq`, `sample_values`) and edges (`conditions`, `weight`). Entity resolution (`sample_values`) is populated by sampling up to 3 distinct values per column directly from Spark, when a Spark session is passed in.

### 3.4 Hybrid RAG V1 — `hybrid_v1_context` (`hybrid_rag_v1/orchestrator.py`)

Straightforward concatenation of the Vector RAG example block and the Graph RAG structural block (same schema linker and the same unweighted `graph_rag.graph_navigator` used by pure Graph RAG).

### 3.5 Hybrid RAG V2 — `hybrid_v2_context` (`hybrid_rag_v2/orchestrator.py`)

Builds on Hybrid V1 with three additional components, each independently toggled off in the ablation study (`hybrid_rag_v2/ablation.py`):

1. **Entity resolution block** — the sample values returned by the schema linker for the *selected* tables are injected verbatim into the prompt (`ENTITY RESOLUTION` section), so the agent can match the exact casing/format of categorical literals.
2. **ALPHA-weighted graph navigator** (`hybrid_rag_v2/graph_navigator.py`) — replaces the purely-frequential navigator with one that scores each candidate path as
   `score = ALPHA · historical_score + (1 − ALPHA) · semantic_score`, where `historical_score` is a `log1p`-normalised sum of edge weights and `semantic_score` is the cosine similarity between the embedding of `nl_query` and the embedding of the path's table names, further discounted by `0.95^(num_edges − 1)` to penalise longer paths. Unlike the pair-wise search of `graph_rag.graph_navigator`, this navigator grows a **single connected subgraph incrementally**: starting from one table, at each step it attaches whichever pending table is reachable via the best-scoring path from *any* table already in the subgraph, guaranteeing the final set of tables forms one connected component. It also explicitly casts the loaded graph to an **undirected** `nx.Graph` before pathfinding — see [`limitations.md`](limitations.md) for why this differs meaningfully from `graph_rag.graph_navigator`, which preserves the `DiGraph`'s directionality.
3. **Negative-prompting instructions** (`INSTRUCTIONS` constant) — explicit anti-hallucination and strict-casing rules prepended to the whole prompt.

## 4. The SparkSQL agent (`spark_toolkit/`)

A LangGraph `create_react_agent` equipped with four tools (`query_sql_db`, `schema_sql_db`, `list_tables_sql_db`, `query_checker_sql_db`). Two behaviours materially affect how results should be interpreted and are **not** obvious from the tool list alone:

- **Single execution attempt.** `spark_nl.py::get_spark_agent` monkey-patches `SparkSQL.run` so that, immediately after the *first* call to `query_sql_db` — whether it succeeds or raises a Spark error — an `AgentEarlyExit` exception is raised and the agent loop terminates. There is no automatic retry/repair loop driven by the agent seeing its own execution error; `query_checker_sql_db` is the only pre-execution safety net.
- **Schema-exploration cap.** `AgentMonitoringCallback` raises `AgentLoopException` if `schema_sql_db` is called more than `config.SCHEMA_LOOP_COUNT` (3) times in one run, to prevent runaway exploration loops.

## 5. Scope decision: the BIRD `evidence` field is never used

`context_strategies.py` and `benchmark_runner.py` read the question directly from `item["question"]` in `dev.json`. The `evidence` field (hand-written natural-language hints bundled with the BIRD benchmark) is deliberately never concatenated into the prompt in the benchmarking pipeline, even though `load_db.py::load_query_info` — used only by earlier/manual tooling — shows how such a concatenation could be done. This is a conscious scope constraint: the system is designed to be applicable to real-world natural-language questions, which will not come with manually curated hints.
