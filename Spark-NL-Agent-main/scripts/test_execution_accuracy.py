import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from evaluation import execution_accuracy


def test_both_empty():
    df_gt = pd.DataFrame()
    df_inf = pd.DataFrame()
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: both_empty")


def test_gt_empty():
    df_gt = pd.DataFrame()
    df_inf = pd.DataFrame({"a": [1, 2]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("PASS: gt_empty")


def test_inf_empty():
    df_gt = pd.DataFrame({"a": [1, 2]})
    df_inf = pd.DataFrame()
    result = execution_accuracy(df_gt, df_inf)
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("PASS: inf_empty")


def test_exact_match():
    df_gt = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_inf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: exact_match")


def test_different_column_names():
    """Column names should not matter, only values."""
    df_gt = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_inf = pd.DataFrame({"c": [1, 2, 3], "d": ["x", "y", "z"]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: different_column_names")


def test_extra_columns_in_inf():
    """Extra columns in inferred should be allowed."""
    df_gt = pd.DataFrame({"a": [1, 2, 3]})
    df_inf = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [4, 5, 6]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: extra_columns_in_inf")


def test_missing_column():
    """Missing a gold column should result in 0."""
    df_gt = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_inf = pd.DataFrame({"a": [1, 2, 3]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("PASS: missing_column")


def test_different_order():
    """Order of values matters - different order should result in 0."""
    df_gt = pd.DataFrame({"a": [1, 2, 3]})
    df_inf = pd.DataFrame({"a": [3, 2, 1]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("PASS: different_order")


def test_different_order_columns():
    """Order of columns should not matter as long as all gold columns are present."""
    df_gt = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    df_inf = pd.DataFrame({"b": ["x", "y", "z"], "a": [1, 2, 3]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: different_order_columns")


def test_numeric_normalization():
    """49729, 49729.0, and '49729' should be considered the same after normalization."""
    df_gt = pd.DataFrame({"a": [49729, 49729.0, "49729"]})
    df_inf = pd.DataFrame({"a": [49729, 49729, 49729]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: numeric_normalization")


def test_null_values():
    """None, null, nan, na, n/a should be normalized to the same token."""
    df_gt = pd.DataFrame({"a": [None, "hello"]})
    df_inf = pd.DataFrame({"a": ["null", "hello"]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: null_values")


def test_partial_column_match():
    """A column with some matching values but not all should not match."""
    df_gt = pd.DataFrame({"a": [1, 2, 3]})
    df_inf = pd.DataFrame({"a": [1, 2, 4]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("PASS: partial_column_match")


def test_reordered_columns_in_inf():
    """Gold columns can be found in inferred in any order."""
    df_gt = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df_inf = pd.DataFrame({"x": [3, 4], "y": [1, 2]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: reordered_columns_in_inf")


def test_single_row():
    df_gt = pd.DataFrame({"a": [1], "b": ["x"]})
    df_inf = pd.DataFrame({"a": [1], "b": ["x"]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: single_row")


def test_single_column_different_name():
    """Column names should not matter - same values with different name is a match."""
    df_gt = pd.DataFrame({"a": [1, 2, 3]})
    df_inf = pd.DataFrame({"b": [1, 2, 3]})
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: single_column_different_name")


def test_list_input():
    """Test with list input (should be converted to DataFrame)."""
    gt = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    inf = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    result = execution_accuracy(gt, inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: list_input")


def test_duplicate_columns():
    """If gold has duplicate columns, inferred must contain both."""
    df_gt = pd.DataFrame([[1, 1], [2, 2]], columns=["a", "a"])
    df_inf = pd.DataFrame([[1, 1], [2, 2]], columns=["b", "c"])
    result = execution_accuracy(df_gt, df_inf)
    assert result == 1.0, f"Expected 1.0, got {result}"
    print("PASS: duplicate_columns")


def test_duplicate_columns_missing():
    """If gold has duplicate columns but inferred only has one."""
    df_gt = pd.DataFrame([[1, 1], [2, 2]], columns=["a", "a"])
    df_inf = pd.DataFrame([[1], [2]], columns=["b"])
    result = execution_accuracy(df_gt, df_inf)
    assert result == 0.0, f"Expected 0.0, got {result}"
    print("PASS: duplicate_columns_missing")


if __name__ == "__main__":
    print("Running execution_accuracy tests...\n")
    test_both_empty()
    test_gt_empty()
    test_inf_empty()
    test_exact_match()
    test_different_column_names()
    test_extra_columns_in_inf()
    test_missing_column()
    test_different_order()
    test_different_order_columns()
    test_numeric_normalization()
    test_null_values()
    test_partial_column_match()
    test_reordered_columns_in_inf()
    test_single_row()
    test_single_column_different_name()
    test_list_input()
    test_duplicate_columns()
    test_duplicate_columns_missing()
    print("\nAll tests passed!")
