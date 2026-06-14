import ast
import json

import sqlglot
import pandas as pd
import pyspark.sql
import numpy as np

NULL_TOKEN = "__NULL__"

def result_to_obj(s):
    if s and isinstance(s, str):
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                parsed = [{"value": s}]
        result = parsed
    else:
        result = s

    return result


def convert_to_dataframe(obj):
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, pyspark.sql.DataFrame):
        return obj.toPandas()
    else:
        return pd.DataFrame(obj)

def _flatten_values(df):
    arr = df.to_numpy(copy=False).astype(object, copy=False)

    mask = pd.isna(arr)
    if mask.any():
        arr = arr.copy()
        arr[mask] = NULL_TOKEN

    # normalize numerics -> "49729", 49729, 49729.0 ---> "49729"
    # convert everything to string
    def canon(v):
        try:
            if isinstance(v, str):
                v = v.strip()
                i = int(v)
                return str(i)
        except Exception:
            pass
        try:
            f = float(v)
            if float(f).is_integer():
                return str(int(f))
        except Exception:
            pass

        # convert all null values to a single token
        try:
            s = str(v).strip()
            s_low = s.lower()

            if s_low in {"none", "null", "nan", "na", "n/a"}:
                return NULL_TOKEN

            return s
        except Exception:
            pass

    out = np.empty(arr.size, dtype=object)
    flat = arr.ravel()
    for i, v in enumerate(flat):
        out[i] = canon(v)

    return out


def _normalize_value(v):
    """Normalize a single value to a canonical string representation."""
    if pd.isna(v):
        return NULL_TOKEN

    try:
        if isinstance(v, str):
            v = v.strip()
            i = int(v)
            return str(i)
    except Exception:
        pass

    try:
        f = float(v)
        if float(f).is_integer():
            return str(int(f))
    except Exception:
        pass

    try:
        s = str(v).strip()
        s_low = s.lower()

        if s_low in {"none", "null", "nan", "na", "n/a"}:
            return NULL_TOKEN

        return s
    except Exception:
        pass

    return str(v)


def _get_column_values(df, col_idx):
    """Return a tuple of normalized values for the given column index."""
    return tuple(_normalize_value(v) for v in df.iloc[:, col_idx])


def execution_accuracy(df_gt, df_inf):
    """Compute execution accuracy based on the Spider 2.0-lite definition.

    A predicted result is considered correct (indicator = 1) if and only if
    every column in the ground-truth result is present as an identical column
    (same values in the same order after normalization) in the predicted
    result. Extra columns in the predicted result are allowed.

    Args:
        df_gt: Ground truth result (DataFrame, list, or Spark DataFrame).
        df_inf: Predicted result (DataFrame, list, or Spark DataFrame).

    Returns:
        float: 1.0 if all gold columns are present in inferred, 0.0 otherwise.
    """
    df_gt = convert_to_dataframe(df_gt)
    df_inf = convert_to_dataframe(df_inf)

    if df_gt.empty and df_inf.empty:
        return 1.0

    if df_gt.empty or df_inf.empty:
        return 0.0

    # Build a list of inferred columns (as tuples of normalized values)
    inf_cols = [_get_column_values(df_inf, col_idx)
                for col_idx in range(df_inf.shape[1])]

    # For each gold column, find a matching inferred column (multiset semantics)
    used = [False] * len(inf_cols)
    for col_idx in range(df_gt.shape[1]):
        gt_col_values = _get_column_values(df_gt, col_idx)
        found = False
        for i in range(len(inf_cols)):
            if not used[i] and inf_cols[i] == gt_col_values:
                used[i] = True
                found = True
                break
        if not found:
            return 0.0

    return 1.0
