"""Utility functions for SparkSQL Agent."""

import os
import urllib.request


# SQLite JDBC driver configuration
SQLITE_JDBC_VERSION = "3.47.2.0"
SQLITE_JDBC_JAR = f"sqlite-jdbc-{SQLITE_JDBC_VERSION}.jar"
SQLITE_JDBC_URL = f"https://repo1.maven.org/maven2/org/xerial/sqlite-jdbc/{SQLITE_JDBC_VERSION}/{SQLITE_JDBC_JAR}"


def ensure_sqlite_jdbc_driver(base_dir=None):
    """Download SQLite JDBC driver if not present.
    
    Args:
        base_dir: Base directory for jars folder. Defaults to parent of src/.
    
    Returns:
        Path to the JDBC jar file.
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(__file__))
    
    jar_dir = os.path.join(base_dir, "jars")
    jar_path = os.path.join(jar_dir, SQLITE_JDBC_JAR)
    
    if not os.path.exists(jar_path):
        print(f"Downloading SQLite JDBC driver...")
        os.makedirs(jar_dir, exist_ok=True)
        urllib.request.urlretrieve(SQLITE_JDBC_URL, jar_path)
        print(f"Downloaded to {jar_path}")
    
    return jar_path


def pretty_print_result(result_obj, max_rows=20):
    """Pretty print query results in a table format.
    
    Args:
        result_obj: Query result (list of rows).
        max_rows: Maximum number of rows to display.
    """
    if not result_obj:
        print("No results returned.")
        return
    
    if isinstance(result_obj, list) and len(result_obj) > 0:
        # Get column names from the first row
        first_row = result_obj[0]
        if hasattr(first_row, 'asDict'):
            # PySpark Row object
            columns = list(first_row.asDict().keys())
            rows = [list(row.asDict().values()) for row in result_obj[:max_rows]]
        elif hasattr(first_row, '_fields'):
            # Named tuple
            columns = list(first_row._fields)
            rows = [list(row) for row in result_obj[:max_rows]]
        else:
            # Plain list/tuple
            columns = [f"col_{i}" for i in range(len(first_row))]
            rows = [list(row) if hasattr(row, '__iter__') and not isinstance(row, str) else [row] 
                    for row in result_obj[:max_rows]]
        
        # Calculate column widths
        col_widths = []
        for i, col in enumerate(columns):
            max_width = len(str(col))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])[:50]))
            col_widths.append(min(max_width, 50))
        
        # Print header
        header = " | ".join(str(col).ljust(col_widths[i]) for i, col in enumerate(columns))
        separator = "-+-".join("-" * w for w in col_widths)
        
        print(header)
        print(separator)
        
        # Print rows
        for row in rows:
            row_str = " | ".join(
                str(row[i] if i < len(row) else "")[:50].ljust(col_widths[i]) 
                for i in range(len(columns))
            )
            print(row_str)
        
        if len(result_obj) > max_rows:
            print(f"\n... ({len(result_obj) - max_rows} more rows)")
        
        print(f"\nTotal rows: {len(result_obj)}")
    else:
        print(result_obj)
