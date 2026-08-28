# Dataset

## Source

All evaluation is performed on five single-database subsets of the [BIRD benchmark](https://bird-bench.github.io/) `dev` split: **toxicology**, **financial**, **superhero**, **formula_1** and **codebase_community**.

The repository does not version the BIRD data. Expected layout (`src/config.py::DB_PATH`, `src/load_db.py`):

```
db/
└── bird-1/
    ├── dev.json
    ├── toxicology/toxicology.sqlite
    ├── financial/financial.sqlite
    ├── superhero/superhero.sqlite
    ├── formula_1/formula_1.sqlite
    └── codebase_community/codebase_community.sqlite
```

Download the official BIRD `dev` set and place the `dev.json` file and each database's `.sqlite` file according to the layout above.

## `dev.json` fields used

Each entry is a BIRD question record; the fields actually consumed by this codebase are:

| Field | Used by |
|---|---|
| `question_id` | query identifiers, leave-one-out exclusion in Vector RAG |
| `db_id` | filtering questions per database |
| `question` | the natural-language input to every architecture |
| `SQL` | the golden query, used both for offline graph/AST construction and as the ground truth for `execution_accuracy` |
| `difficulty` | used only for the plots (`scripts/generate_plots.py`) |
| `evidence` | present in the file but **intentionally never read** by the benchmarking pipeline — see [`architecture.md`](architecture.md#5-scope-decision-the-bird-evidence-field-is-never-used) |

## Loading into Spark

`src/load_db.py::load_bird_tables` reads the SQLite schema directly (via `sqlite3`) to detect `DATE`/`DATETIME`/`TIMESTAMP` columns and forces them to Spark's `string` type through a JDBC `customSchema` override before registering each table as a Spark temporary view. This avoids type-inference mismatches between SQLite's dynamic typing and Spark's JDBC reader for date-like columns.

## Known data quirks

- **`superhero`**: a subset of consecutive questions share an (near-)identical golden SQL output, independent of the specific entity named in the question. This produces a run of identical predicted/expected results in the logs that is not representative of true retrieval difficulty for those items, and is treated as a validity threat rather than a genuine model success — see the thesis's qualitative error analysis (Chapter 6) for the detailed breakdown.
- Execution Accuracy (`src/evaluation.py`) compares result columns **in row order** after normalisation; databases/queries where SQL does not enforce a deterministic row order (no `ORDER BY`, or ties within an `ORDER BY`) can therefore be scored as incorrect even when the *set* of returned values is right. See [`limitations.md`](limitations.md).
