# Text-to-SQL over Apache Spark: A Comparative Study of RAG Architectures

This repository contains the full implementation of a Bachelor's Final Degree Project (TFG) that designs, implements and benchmarks five Retrieval-Augmented Generation (RAG) architectures for the **Text-to-SQL** task.

The system is executed against **Apache Spark / SparkSQL** using a **LangChain/LangGraph ReAct agent** and evaluated on five relational databases with varying topologies from the [BIRD benchmark](https://bird-bench.github.io/):

- `toxicology`
- `financial`
- `superhero`
- `formula_1`
- `codebase_community`

The full technical write-up (methodology, algorithmic analysis, complexity study, and experimental results) is available in the accompanying thesis document.

This README focuses on setting up the environment, running the code, and reproducing the experiments.

---

## Architectures

| Key | Name | Description |
|---|---|---|
| `baseline` | **Baseline (Zero-Shot)** | The natural-language question is sent to the agent with no retrieved context. |
| `vector` | **Vector RAG** | Retrieves the top-K most semantically similar past questions using ChromaDB and `all-MiniLM-L6-v2`, providing them as few-shot examples. A Leave-One-Out strategy is used to prevent data leakage. |
| `graph` | **Graph RAG** | Routes the question to the relevant tables using an LLM-based schema linker and retrieves join rules and historical usage metadata from a property graph built from the ASTs of past golden queries using NetworkX. |
| `hybrid_v1` | **Hybrid RAG V1** | Concatenates the Vector RAG semantic examples with the Graph RAG structural context using vanilla fusion. |
| `hybrid_v2` | **Hybrid RAG V2** | Extends Hybrid V1 with entity resolution, filtered sample values for the selected tables, an ALPHA-weighted graph navigator that blends historical frequency with query-path semantic similarity, and negative-prompting instructions. |

> **Note on Ablation:** Three additional ablation variants of Hybrid RAG V2 are implemented to isolate the contribution of each module:
>
> - `hybrid_v2_no_entity`
> - `hybrid_v2_no_alpha`
> - `hybrid_v2_no_negprompt`
>
> An ALPHA sensitivity sweep is also implemented.

---

## Requirements & Setup

### Requirements

- **Python 3.10+**
- **Java JDK 8, 11, or 17**
- `JAVA_HOME` configured
- **Apache Spark**, installed automatically through `pyspark`
- No separate Spark cluster installation is required.

### 1. Clone the repository and create a virtual environment

```bash
git clone https://github.com/kevinsanchez04/tfg-text2sql.git
cd tfg-text2sql

python3 -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The SQLite JDBC driver is downloaded automatically into `jars/` the first time a Spark session is initialized.

### 3. Configure credentials

Create a `.env` file at the repository root with the credentials for whichever provider(s) you intend to use.

The default provider is `google` with `gemini-2.5-flash`.

```env
GOOGLE_API_KEY=your_api_key_here
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
NVIDIA_API_KEY=...
CLOUDFLARE_ACCOUNT_ID=...
CLOUDFLARE_API_TOKEN=...
```

Only the credentials required by the selected provider need to be configured.

---

## Data Preparation

This repository **does not include the BIRD dataset**.

You need to download the `dev` split from the [BIRD benchmark](https://bird-bench.github.io/) and place the files using the following structure:

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

---

# Quickstart & Pipeline

Before evaluating any architecture with a retrieval component, you must generate the **vector index** and the **property graph** for each database.

---

## 1. Offline Pipeline — Build Retrieval Databases

Set the target database:

```bash
DB=toxicology
```

Available databases:

```text
toxicology
financial
superhero
formula_1
codebase_community
```

### Build the vector store

The vector store is generated using ChromaDB:

```bash
python src/vector_rag/build_vector_db.py --db $DB
```

### Extract ASTs

Extract the AST of all golden SQL queries:

```bash
python src/graph_rag/ast_parser.py --db $DB
```

### Build the property graph

Consolidate the extracted AST information into the property graph:

```bash
python src/graph_rag/graph_builder.py --db $DB
```

For example, to prepare the `toxicology` database:

```bash
DB=toxicology

python src/vector_rag/build_vector_db.py --db $DB
python src/graph_rag/ast_parser.py --db $DB
python src/graph_rag/graph_builder.py --db $DB
```

Repeat this process for each database before running the corresponding retrieval-based architectures.

---

## 2. Run a Benchmark

Run the evaluation loop for a specific database and architecture:

```bash
python scripts/run_benchmark.py \
    --db toxicology \
    --arch hybrid_v2 \
    --provider google
```

### Available architectures

```text
baseline
vector
graph
hybrid_v1
hybrid_v2
```

### Example

Run Hybrid RAG V2 on the `financial` database:

```bash
python scripts/run_benchmark.py \
    --db financial \
    --arch hybrid_v2 \
    --provider google
```

---

## 3. Debug a Single Query

For debugging a specific natural-language query without running the entire benchmark:

```bash
python scripts/test_single_query.py \
    --qid 196 \
    --db toxicology
```

This is particularly useful for inspecting the behaviour of the Hybrid RAG V2 pipeline on individual questions.

---

## 4. Generate Results & Plots

Once the evaluation logs have been generated, the academic plots used in the thesis can be recreated.

The script expects logs for all five main architectures to exist:

```bash
python scripts/generate_plots.py --db toxicology
```

Generated plots are stored under:

```text
data/plots/
```

---

# Evaluation Metric

Correctness is measured using **Execution Accuracy**, implemented in:

```text
src/evaluation.py
```

A predicted result is considered correct if and only if:

1. Every column of the golden result appears in the predicted result.
2. Values match value-for-value.
3. The row order is preserved.
4. Extra predicted columns are allowed.

Before comparison, values are normalized to handle differences such as:

- Numeric representation
- `NULL` tokens
- Other equivalent representations

This evaluation focuses on whether the generated SQL produces the expected result when executed against SparkSQL.

---

# Repository Structure

```text
.
├── scripts/
│   ├── run_benchmark.py
│   │   └── CLI entry point: runs one architecture over one database
│   │
│   ├── test_single_query.py
│   │   └── Debug a single question with Hybrid RAG V2
│   │
│   ├── logs_summary.py
│   │   └── Flattens a raw log file for manual inspection
│   │
│   └── generate_plots.py
│       └── Builds comparative accuracy plots
│
├── src/
│   ├── config.py
│   │   └── Provider/model registry and global constants
│   │
│   ├── paths.py
│   │   └── Single source of truth for all on-disk paths
│   │
│   ├── llm.py
│   │   └── LLM client factory
│   │
│   ├── load_db.py
│   │   └── SQLite → Spark table loading
│   │
│   ├── evaluation.py
│   │   └── Execution Accuracy metric
│   │
│   ├── benchmark_runner.py
│   │   └── Orchestrates a full benchmark run
│   │
│   ├── context_strategies.py
│   │   └── Registry mapping architecture key → context builder
│   │
│   ├── utils.py
│   │   └── JDBC driver bootstrap and pretty printers
│   │
│   ├── spark_toolkit/
│   │   └── LangGraph ReAct agent + SparkSQL tools
│   │
│   ├── vector_rag/
│   │   └── ChromaDB index builder + retriever
│   │
│   ├── graph_rag/
│   │   └── AST extraction, graph builder and schema linker
│   │
│   ├── hybrid_rag_v1/
│   │   └── Hybrid RAG V1 orchestrator
│   │
│   └── hybrid_rag_v2/
│       ├── V2 orchestrator
│       ├── ALPHA-weighted graph navigator
│       └── ablation.py
│
├── db/
│   ├── bird-1/
│   │   └── BIRD benchmark data (not versioned)
│   │
│   └── vector_store/
│       └── Persisted ChromaDB collections (generated)
│
├── data/
│   ├── ast/
│   │   └── Extracted ASTs per domain (generated)
│   │
│   ├── graphs/
│   │   └── Persisted property graphs (generated)
│   │
│   └── plots/
│       └── Generated plots
│
├── docs/
│   └── AGENT_ARCHITECTURE_EXPLANATION.md
│
├── logs/
│   └── Raw evaluation results per domain/architecture
│
├── jars/
│   └── SQLite JDBC driver (auto-downloaded)
│
├── requirements.txt
├── .env
└── README.md
```

---

# Main Hyperparameters

| Parameter | Default Value | Location |
|---|---:|---|
| Top-K few-shot examples | `K = 3` | `src/vector_rag/retriever.py` |
| α — historical vs. semantic weight | `α = 0.4` | `src/hybrid_rag_v2/graph_navigator.py` |
| Attenuation factor per hop | `0.95` | `src/hybrid_rag_v2/graph_navigator.py` |
| Navigation hop limit | `cutoff = 3` | `src/graph_rag/graph_navigator.py`, `src/hybrid_rag_v2/graph_navigator.py` |
| Embeddings model | `all-MiniLM-L6-v2` | `src/vector_rag/retriever.py`, `src/hybrid_rag_v2/graph_navigator.py` |
| Embedding dimension | `384` | `all-MiniLM-L6-v2` |
| LLM Temperature | `0.0` | `src/config.py` |

---

# Architecture Overview

The five evaluated architectures progressively introduce additional retrieval and reasoning capabilities:

```text
                         Natural Language Question
                                   │
                                   ▼
                          ┌─────────────────┐
                          │    Baseline     │
                          │    Zero-Shot    │
                          └─────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │       Vector RAG         │
                    │ Semantic Few-Shot        │
                    │ ChromaDB + Embeddings    │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │        Graph RAG         │
                    │ Schema Linking           │
                    │ Property Graph           │
                    │ Join / Usage Metadata    │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │      Hybrid RAG V1       │
                    │ Vector + Graph Context   │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │      Hybrid RAG V2       │
                    │                          │
                    │ + Entity Resolution      │
                    │ + Sample Values          │
                    │ + ALPHA Navigator        │
                    │ + Semantic Path Scoring  │
                    │ + Negative Prompting     │
                    └──────────────────────────┘
                                   │
                                   ▼
                         ┌──────────────────┐
                         │ LangGraph ReAct  │
                         │ Agent + SparkSQL │
                         └──────────────────┘
                                   │
                                   ▼
                            SparkSQL Result
```

---

# Documentation

The repository includes additional documentation covering the system architecture, datasets, experimental methodology, and agent implementation.

- [`docs/Architecture.md`](docs/Architecture.md) — Overview of the RAG architectures, system components, and how the different retrieval strategies are integrated into the Text-to-SQL pipeline.
- [`docs/Dataset.md`](docs/Dataset.md) — Documentation of the BIRD databases used in the experiments, their characteristics, schemas, and data preparation.
- [`docs/Experiment_guide.md`](docs/Experiment_guide.md) — Step-by-step guide for reproducing the experiments, running the benchmarks, configuring the different architectures, and generating the results.
- [`docs/AGENT_ARCHITECTURE_EXPLANATION.md`](docs/AGENT_ARCHITECTURE_EXPLANATION.md) — Detailed explanation of the LangGraph ReAct agent, its execution cycle, EarlyExit strategy, and SparkSQL tool catalog.
---

# Reproducibility

To reproduce the experiments from scratch:

### Step 1 — Install the environment

```bash
git clone https://github.com/kevinsanchez04/tfg-text2sql.git
cd tfg-text2sql

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Step 2 — Configure the API provider

Create the `.env` file:

```env
GOOGLE_API_KEY=your_api_key_here
```

### Step 3 — Download BIRD

Place the required BIRD `dev` split under:

```text
db/bird-1/
```

### Step 4 — Build the retrieval resources

For each database:

```bash
DB=toxicology

python src/vector_rag/build_vector_db.py --db $DB
python src/graph_rag/ast_parser.py --db $DB
python src/graph_rag/graph_builder.py --db $DB
```

Repeat for:

```text
financial
superhero
formula_1
codebase_community
```

### Step 5 — Run the benchmarks

For example:

```bash
python scripts/run_benchmark.py --db toxicology --arch baseline --provider google
python scripts/run_benchmark.py --db toxicology --arch vector --provider google
python scripts/run_benchmark.py --db toxicology --arch graph --provider google
python scripts/run_benchmark.py --db toxicology --arch hybrid_v1 --provider google
python scripts/run_benchmark.py --db toxicology --arch hybrid_v2 --provider google
```

Repeat for the remaining databases.

### Step 6 — Generate the plots

```bash
python scripts/generate_plots.py --db toxicology
```

---

# License & Citation

This project was developed as a **Bachelor's Thesis (TFG) in Computer Engineering** at **Universitat Rovira i Virgili (URV)**.

If you use this repository or build upon the implementation, please cite the corresponding thesis.

```text
Text-to-SQL over Apache Spark: A Comparative Study of RAG Architectures
Bachelor's Final Degree Project (TFG)
Computer Engineering
Universitat Rovira i Virgili (URV)
```
