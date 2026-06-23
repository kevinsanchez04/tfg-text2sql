"""Spark SQL agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain_core.callbacks import BaseCallbackManager, Callbacks
from langchain_core.language_models import BaseLanguageModel

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

from spark_toolkit.prompt import SQL_PREFIX, SQL_SUFFIX
from spark_toolkit.toolkit import SparkSQLToolkit


def create_spark_sql_agent(
    llm: BaseLanguageModel,
    toolkit: SparkSQLToolkit,
    callback_manager: Optional[BaseCallbackManager] = None,
    callbacks: Callbacks = None,
    prefix: str = SQL_PREFIX,
    suffix: str = SQL_SUFFIX,
    format_instructions: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    top_k: int = 10,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    verbose: bool = False,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Runnable:
    """Construct a Spark SQL agent from an LLM and tools.

    Args:
        llm: The language model to use.
        toolkit: The Spark SQL toolkit.
        callback_manager: Optional. The callback manager. Default is None.
        callbacks: Optional. The callbacks. Default is None.
        prefix: Optional. The prefix for the prompt. Default is SQL_PREFIX.
        suffix: Optional. The suffix for the prompt. Default is SQL_SUFFIX.
        format_instructions: Optional. The format instructions for the prompt.
            Default is None.
        input_variables: Optional. The input variables for the prompt. Default is None.
        top_k: Optional. The top k for the prompt. Default is 10.
        max_iterations: Optional. The maximum iterations to run. Default is 15.
        max_execution_time: Optional. The maximum execution time. Default is None.
        early_stopping_method: Optional. The early stopping method. Default is "force".
        verbose: Optional. Whether to print verbose output. Default is False.
        agent_executor_kwargs: Optional. The agent executor kwargs. Default is None.
        kwargs: Any. Additional keyword arguments.

    Returns:
        The agent as a Runnable.
    """
    from langgraph.prebuilt import create_react_agent

    tools = toolkit.get_tools()
    try:
        prefix = prefix.format(top_k=top_k)
    except (KeyError, IndexError):
        # Prefix doesn't have {top_k} placeholder, use as-is
        pass
    
    agent = create_react_agent(llm, tools, prompt=prefix)
    return agent