# Enhancing Text-to-SQL Systems in Complex Domains using a Hybrid RAG Architecture

A research framework for the comparative evaluation of five Text-to-SQL architectures (**Baseline Zero-Shot**, **Vector RAG**, **Graph RAG**, **Hybrid RAG V1**, and **Hybrid RAG V2**) over the [BIRD](https://bird-bench.github.io/) benchmark, using a ReAct agent built with LangChain/LangGraph and executed on Apache Spark.

The system orchestrates a comprehensive ablation study (*Leave-One-Out* methodology) to measure *Execution Accuracy*, token cost, and execution robustness of each architecture across five relational domains with varying topologies: `toxicology`, `financial`, `superhero`, `formula_1`, and `codebase_community`.

This repository is a **research and experimentation framework**, not an end-user application: there is no GUI, and its primary usage is through command-line scripts.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Credentials Configuration](#credentials-configuration)
- [Data Preparation (BIRD)](#data-preparation-bird)
- [Offline Pipeline: Vector Index and Property Graph Construction](#offline-pipeline-vector-index-and-property-graph-construction)
- [Running Evaluation Experiments](#running-evaluation-experiments)
- [Results Analysis and Plot Generation](#results-analysis-and-plot-generation)
- [Repository Structure](#repository-structure)
- [Available Architectures](#available-architectures)
- [Main Hyperparameters](#main-hyperparameters)

## Prerequisites

- **Python 3.10+**
- **Java JDK 8, 11, or 17** with `JAVA_HOME` configured (required for Apache Spark / PySpark)
- **Apache Spark** (installed via `pyspark`, no separate cluster installation needed)
- Internet access on first run (automatic download of the SQLite JDBC driver)
- At least one API key from a supported LLM provider:
  - `GOOGLE_API_KEY` (Google Gemini — default provider)
  - `ANTHROPIC_API_KEY` (Anthropic Claude)
  - `OPENAI_API_KEY` (OpenAI)
  - `CLOUDFLARE_ACCOUNT_ID` + `CLOUDFLARE_API_TOKEN` (Cloudflare Workers AI)
  - `NVIDIA_API_KEY` (NVIDIA NIM)

## Installation

```bash
# 1. Clone the repository
git clone [https://github.com/kevinsanchez04/tfg-text2sql.git](https://github.com/kevinsanchez04/tfg-text2sql.git)
cd tfg-text2sql

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

> The SQLite JDBC driver (`sqlite-jdbc-3.47.2.0.jar`) is downloaded automatically into `jars/` the first time any script initializes a Spark session (function `ensure_sqlite_jdbc_driver` in `src/utils.py`). If there is no Internet connection, it must be manually downloaded from Maven Central and placed in `jars/`.

## Credentials Configuration

Create a `.env` file at the project root (loaded automatically via `python-dotenv`):

```env
GOOGLE_API_KEY=xxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=xxxxxxxxxxxxxxxx
OPENAI_API_KEY=xxxxxxxxxxxxxxxx
CLOUDFLARE_ACCOUNT_ID=xxxxxxxxxxxxxxxx
CLOUDFLARE_API_TOKEN=xxxxxxxxxxxxxxxx
NVIDIA_API_KEY=xxxxxxxxxxxxxxxx
```

You only need to define the key for the provider you intend to use with `--provider`.

## Data Preparation (BIRD)

This repository **does not include** the BIRD dataset (due to size and licensing). You need to download the `dev` split from [BIRD-SQL](https://bird-bench.github.io/) and place it using the following structure:

```text
db/
└── bird-1/
    ├── dev.json
    ├── toxicology/
    │   └── toxicology.sqlite
    ├── financial/
    │   └── financial.sqlite
    ├── superhero/
    │   └── superhero.sqlite
    ├── formula_1/
    │   └── formula_1.sqlite
    └── codebase_community/
        └── codebase_community.sqlite
```

## Offline Pipeline: Vector Index and Property Graph Construction

Before evaluating any architecture with a retrieval component, you must generate the vector index (for Vector RAG and hybrid variants) and the property graph (for Graph RAG and hybrid variants) **for each database**. All scripts are executed from the root of the repository.

```bash
DB=toxicology   # replace with: financial | superhero | formula_1 | codebase_community

# 1. Build the vector store in ChromaDB
python src/vector_rag/build_vector_db.py --db $DB

# 2. Extract the AST of all golden queries for the domain
python src/graph_rag/ast_parser.py --db $DB

# 3. Consolidate the AST into the Property Graph (with entity resolution via Spark)
python src/graph_rag/graph_builder.py --db $DB
```

This pipeline generates:
- `db/vector_store/<db_name>/` — Persisted ChromaDB collection
- `data/ast/<db_name>/ast.json` — AST per query
- `data/graphs/<db_name>/property_graph.json` — Aggregated property graph

These three steps must be repeated for each of the five domains before running the corresponding experiments.

## Running Evaluation Experiments

### Full benchmark over a domain

```bash
python scripts/run_benchmark.py \
  --db toxicology \
  --arch hybrid_v2 \
  --provider google \
  --model gemini-2.5-flash
```

Parameters:
- `--db`: `toxicology` | `financial` | `superhero` | `formula_1` | `codebase_community`
- `--arch`: `baseline` | `vector` | `graph` | `hybrid_v1` | `hybrid_v2`
- `--provider`: `google` (default) | `claude` | `openai` | `cloudflare` | `nvidia`
- `--model`: Specific model name (optional; uses the provider's default model if omitted)
- `--force-thoughts`: Forces explicit verbalization of the reasoning before each tool call (for debugging only)

Results are saved in `logs/<db_name>/<arch>_<db_name>.json`.

### Test a single query (debugging, Hybrid RAG V2)

```bash
python scripts/test_single_query.py --qid 196 --db toxicology
```

Displays the generated SparkSQL query, compares it with the golden query, calculates the accuracy, and shows the exact prompt sent to the LLM. To debug with Hybrid RAG V1 instead of V2, you must toggle the `get_hybrid_context` import at the top of the file (see code comments).

### Log Simplification

```bash
python scripts/logs_summary.py \
  --input logs/toxicology/hybrid_v2_toxicology.json \
  --output data/simplified_hybrid_v2_toxicology.json \
  --db_id toxicology
```

### Generating Comparative Plots

```bash
python scripts/generate_plots.py --db toxicology
```

Requires the five files `logs/<db>/{baseline,vector,graph,hybrid_v1,hybrid_v2}_<db>.json` to already exist. Generates the plots in `data/plots/<db_name>/`.

## Repository Structure

```text
.
├── src/
│   ├── config.py                  # Global configuration and providers
│   ├── llm.py                     # LLM client factory (Gemini, Claude, OpenAI, Cloudflare, NVIDIA)
│   ├── load_db.py                 # Loading BIRD/SQLite tables into Spark
│   ├── paths.py                   # Centralized path resolution (graphs, vector stores, logs...)
│   ├── evaluation.py              # Execution Accuracy calculation
│   ├── spark_nl.py                # Agent orchestration, instrumentation, and metrics
│   ├── benchmark_runner.py        # Generic evaluation loop per architecture/domain
│   ├── context_strategies.py      # Common adapter for the 5 architectures
│   ├── spark_toolkit/             # ReAct agent tools over Spark SQL
│   ├── vector_rag/                # Semantic retrieval (ChromaDB + embeddings)
│   ├── graph_rag/                 # AST Parser, graph construction and navigation (V1)
│   ├── hybrid_rag_v1/             # Vanilla fusion orchestrator
│   └── hybrid_rag_v2/             # Weighted navigator, entity resolution, negative prompting
├── scripts/
│   ├── run_benchmark.py           # Main evaluation entry point
│   ├── test_single_query.py       # Single query debugging
│   ├── logs_summary.py            # Raw logs simplification
│   ├── generate_plots.py          # Comparative plots generation
│   └── old_scripts/               # Scripts prior to unification via context_strategies.py
├── db/
│   ├── bird-1/                    # BIRD Dataset (not included in the repository)
│   └── vector_store/              # Persisted ChromaDB collections (generated)
├── data/
│   ├── ast/                       # Extracted ASTs per domain (generated)
│   ├── graphs/                    # Persisted property graphs (generated)
│   └── plots/                     # Generated plots
├── docs/                          # Extra documentation (agent architecture, etc.)
├── logs/                          # Raw evaluation results per domain/architecture
└── jars/                          # SQLite JDBC Driver (downloaded automatically)
```

## Available Architectures

| `--arch` Key | Description |
|---|---|
| `baseline` | No retrieval context; pure Zero-Shot agent |
| `vector` | Semantic retrieval of few-shot examples (ChromaDB, Leave-One-Out) |
| `graph` | Pure structural navigation by frequency weights over the property graph |
| `hybrid_v1` | Vanilla concatenation of semantic + structural context |
| `hybrid_v2` | Weighted navigation (semantic + frequency), entity resolution, and negative prompting |

## Main Hyperparameters

| Parameter | Default Value | Location |
|---|---|---|
| Top-K (few-shot examples) | `K = 3` | `src/vector_rag/retriever.py` |
| α (historical vs. semantic weight) | `α = 0.4` | `src/hybrid_rag_v2/graph_navigator.py` |
| Attenuation factor per hop | `0.95` | `src/hybrid_rag_v2/graph_navigator.py` |
| Navigation hop limit | `cutoff = 3` | `src/graph_rag/graph_navigator.py`, `src/hybrid_rag_v2/graph_navigator.py` |
| Embeddings model | `all-MiniLM-L6-v2` (d=384) | `src/vector_rag/retriever.py`, `src/hybrid_rag_v2/graph_navigator.py` |
| LLM Temperature | `0.0` | `src/config.py` |

## License & Citation
This project was developed as a Bachelor's Thesis (TFG) in Computer Engineering at Universitat Rovira i Virgili (URV).
