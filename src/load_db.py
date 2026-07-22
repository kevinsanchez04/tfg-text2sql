from config import (
    DB_PATH,
    BENCHMARK_FILE
)
import os
import json

import sqlite3


SQLITE_DATE_TYPES = {"DATE", "DATETIME", "TIMESTAMP"}


def get_bird_db_path(db_name):
    return os.path.join(DB_PATH, "bird-1", db_name, f"{db_name}.sqlite")


def load_tables(spark_session, db_name):
    load_bird_tables(spark_session, db_name)


def _get_column_overrides(db_path, table):
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.execute(f'PRAGMA table_info("{table}")')
    overrides = []
    for row in cursor.fetchall():
        col_name = row[1]
        col_type = row[2].upper() if row[2] else "TEXT"
        sqlite_type = col_type.split("(")[0]
        if sqlite_type in SQLITE_DATE_TYPES:
            overrides.append(f"{col_name} string")
    db_conn.close()
    return overrides


def load_bird_tables(spark_session, db_name):
    db_path = get_bird_db_path(db_name)
    print(f"--- Scanning database: {db_path} ---")
    abs_db_path = os.path.abspath(db_path)
    jdbc_url = f"jdbc:sqlite:{abs_db_path}"
    db_connection = sqlite3.connect(db_path)
    cursor = db_connection.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    db_connection.close()

    if not tables:
        print("Warning: No tables found in the database!")
        return

    for table in tables:
        overrides = _get_column_overrides(db_path, table)
        custom_schema = ",".join(overrides) if overrides else None
        read_options = {
            "url": jdbc_url,
            "dbtable": f'"{table}"',
            "driver": "org.sqlite.JDBC",
        }
        if custom_schema:
            read_options["customSchema"] = custom_schema
        df = spark_session.read.format("jdbc").options(**read_options).load()

        df.createOrReplaceTempView(table)
        print(f" -> Registered table: '{table}'")


def load_query_info(query_id: int):

    query_data_file = os.path.join(DB_PATH, "bird-1", BENCHMARK_FILE)

    with open(query_data_file, 'r') as f:
        all_queries = json.load(f)

    query_info = None
    for query_entry in all_queries:
        if query_entry['question_id'] == query_id:
            query_info = query_entry
            break

    if query_info is None:
        raise ValueError(f"Query ID {query_id} not found")

    question = " ".join([
        query_info["question"],
        query_info.get("evidence", "")
    ])
    golden_query = query_info["SQL"]
    difficulty = query_info.get("difficulty", "unknown")

    return question, golden_query, difficulty
