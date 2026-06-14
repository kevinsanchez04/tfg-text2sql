import os
import types
import time
import json
import datetime
import uuid

from pyspark.sql import SparkSession
from langchain_core.callbacks import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_cloudflare import ChatCloudflareWorkersAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import config
from evaluation import result_to_obj
from llm import get_cloudflare_neuron_pricing
from spark_toolkit.toolkit import SparkSQLToolkit
from spark_toolkit.base import create_spark_sql_agent
from spark_toolkit.spark_sql import SparkSQL

class AgentEarlyExit(BaseException):
    def __init__(self, answer):
        self.answer = answer


class AgentLoopException(BaseException):
    pass


class AgentMonitoringCallback(BaseCallbackHandler):
    def __init__(self):
        self.count = 0
        self.chain_of_thought = []
        self.agent_thoughts = []  # Agent thoughts: textual LLM responses without metadata
        self.input_tokens = 0
        self.output_tokens = 0
        # token_details aggregated across ALL LLM calls
        self.input_token_details_total = {}   # e.g. cache_read
        self.output_token_details_total = {}  # e.g. reasoning
        self.last_prompt = None
        self.last_answer = None
        self.schema_sql_db_count = 0
        # store per-tool metrics
        self.tool_metrics = {}
        # store active tool runs
        self.active_tool_runs = {}
        # store active LLM runs
        self.active_llm_runs = {}
        # store LLM statistics per call
        self.llm_calls = {}
        # spans for time tracing (to catch nested calls)
        self.spans = {} # run_id -> span dict

    # -------------------------
    # helpers
    # -------------------------
    def _as_str(self, x):
        return str(x) if x is not None else None

    def _ensure_tool(self, tool_name: str):
        if tool_name not in self.tool_metrics:
            self.tool_metrics[tool_name] = {
                "call_count": 0,
                "total_duration": 0.0,
                "calls": {},
            }
        return self.tool_metrics[tool_name]

    def _ensure_tool_call(self, tool_name: str, tool_run_id: str):
        tool = self._ensure_tool(tool_name)
        if tool_run_id not in tool["calls"]:
            tool["calls"][tool_run_id] = {
                "duration": None,
                "input": None,
                "output": None,
            }
        return tool["calls"][tool_run_id]

    # identifier to link the token usage in response.generations to the tool calls
    def _parse_lc_run_identifier(self, message_id: str | None):
        # identifier in response.generations: 'lc_run--...-0' -> strip prefix and trailing '-0'
        if not message_id:
            return None
        s = message_id
        if s.startswith("lc_run--"):
            s = s.replace("lc_run--", "", 1)
        # remove trailing "-0"
        if "-" in s:
            s = s.rsplit("-", 1)[0]
        return s

    def _span_start(self, span_type: str, run_id: str, parent_run_id: str | None, name: str, meta: dict | None = None):
        if not run_id:
            return
        self.spans[run_id] = {
            "type": span_type,              # "llm" | "tool" | "spark"
            "name": name,                   # tool name / "llm_call" / "spark_run"
            "run_id": run_id,
            "parent_run_id": parent_run_id,
            "start": time.time(),
            "end": None,
            "duration": None,
            "meta": meta or {},
        }

    def _span_end(self, run_id: str, extra_meta: dict | None = None):
        time_end = time.time()
        if not run_id:
            return
        s = self.spans.get(run_id)
        if not s:
            return
        s["end"] = time_end
        duration = s["end"] - s["start"]
        s["duration"] = duration
        if extra_meta:
            s["meta"].update(extra_meta)

        return duration

    # -------------------------
    # LLM callbacks
    # -------------------------

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.count += 1
        if prompts:
            self.last_prompt = prompts[0]

        run_id = self._as_str(kwargs.get("run_id"))
        prompt = prompts[0] if prompts else ""

        start_time = time.time()
        self.active_llm_runs[run_id] = {"name": "llm_call", "prompt": prompt, "start": start_time}

        parent_run_id = self._as_str(kwargs.get("parent_run_id"))
        self._span_start(
            span_type="llm",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name="llm_call",
            meta={"prompt_len": len(prompt or "")},
        )

    # called before the tools are called (or inside the query checker) -> input and output tokens are known
    def on_llm_end(self, response, **kwargs):
        run_id = self._as_str(kwargs.get("run_id"))
        duration = self._span_end(run_id)

        # data to get the active run
        parent_run_id = self._as_str(kwargs.get("parent_run_id"))
        active = self.active_llm_runs.pop(run_id, None) if run_id else None

        if hasattr(response, "generations"):
            for g in response.generations:
                thought = ""
                for gen in g:
                    if hasattr(gen, "text"):
                        self.last_answer = gen.text
                        thought += gen.text + "\n"

                    gen_dict = getattr(gen, "__dict__", {}) or {}
                    message = gen_dict.get("message", None)
                    if message is None:
                        continue
                    message_dict = getattr(message, "__dict__", {}) or {}

                # Print and store agent thought (LLM textual response without metadata)
                if thought.strip():
                    self.agent_thoughts.append(thought.strip())
                    print(f"\n[Agent Thought] {thought.strip()}")

                # identifier to map token usage (given here) to the tool calls (given in on_tool_start)
                lc_run_identifier = self._parse_lc_run_identifier(message_dict.get("id"))

                usage = message_dict.get("usage_metadata", {}) or {}
                input_tokens = int(usage.get("input_tokens", 0) or 0)
                output_tokens = int(usage.get("output_tokens", 0) or 0)
                input_details = usage.get("input_token_details", {}) or {}
                output_details = usage.get("output_token_details", {}) or {}

                # global totals
                self.input_tokens += input_tokens
                self.output_tokens += output_tokens
                self.input_token_details_total[lc_run_identifier] = input_details
                self.output_token_details_total[lc_run_identifier] = output_details

                # print("Message_dict:", message_dict)

                tool_calls = message_dict.get("tool_calls", [])
                parsed_tool_calls = []
                for tc in tool_calls:
                    parsed_tool_calls.append({
                        "name": tc.get("name", "unknown_tool"),
                        "args": tc.get("args", {}) or {},
                        "id": tc.get("id"),
                        "type": tc.get("type"),
                    })

                # store per-LLM-call stats
                if active:
                    self.llm_calls[run_id] = {
                        "prompt": active["prompt"],
                        "duration": duration,
                        "run_id": run_id,
                        "parent_run_id": parent_run_id,
                        "total_tokens": input_tokens + output_tokens,
                        "input_tokens": input_tokens,
                        "input_token_details": input_details,
                        "output_tokens": output_tokens,
                        "output_token_details": output_details,
                        "tool_calls": parsed_tool_calls,
                        "thought": thought
                    }

    # -------------------------
    # Agent callbacks
    # -------------------------
    def on_agent_action(self, action, **kwargs):
        log_message = action.log
        self.chain_of_thought.append(log_message)
        print(f"\n[Real-time CoT - Agent Response] {log_message}")

        # Extract thought from action.log (text before "Action:")
        if log_message:
            # Split on "Action:" to get the thought part
            thought_text = log_message
            if "Action:" in log_message:
                parts = log_message.split("Action:", 1)
                thought_text = parts[0].strip()
                # Remove "Thought:" prefix if present
                if thought_text.startswith("Thought:"):
                    thought_text = thought_text[8:].strip()
            if thought_text:
                self.agent_thoughts.append(thought_text)
                print(f"\n[Agent Thought] {thought_text}")

        if action.tool == "schema_sql_db":
            self.schema_sql_db_count += 1
            if self.schema_sql_db_count > config.SCHEMA_LOOP_COUNT:
                raise AgentLoopException("Loop detected: schema_sql_db called too many times")

    def on_agent_finish(self, finish, **kwargs):
        log_message = finish.log
        self.chain_of_thought.append(log_message)
        print(f"\n[Real-time CoT - Agent Finish] {log_message}")

    # -------------------------
    # Tool callbacks
    # -------------------------

    def on_tool_start(self, serialized, input_str, **kwargs):
        run_id = self._as_str(kwargs.get("run_id"))
        parent_run_id = self._as_str(kwargs.get("parent_run_id"))

        tool_name = serialized.get("name") if serialized else "unknown"
        tool = self._ensure_tool(tool_name)
        tool["call_count"] += 1

        # initialize per-call entry
        call = self._ensure_tool_call(tool_name, run_id or f"tool_call_{tool['call_count']}")
        call["parent_run_id"] = parent_run_id
        call["input"] = input_str

        self.active_tool_runs[run_id] = {"name": tool_name, "tool_run_id": run_id or None}

        message = f"Action: {tool_name}\nAction Input: {input_str}"
        self.chain_of_thought.append(message)
        print(f"\n[Real-time CoT - TOOL Call] {message}")

        self._span_start(
            span_type="tool",
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=tool_name,
            meta={"input_len": len(input_str or "")},
        )

        if tool_name == "schema_sql_db":
            self.schema_sql_db_count += 1
            if self.schema_sql_db_count > config.SCHEMA_LOOP_COUNT:
                raise AgentLoopException("Loop detected: schema_sql_db called too many times")

    def on_tool_end(self, output, **kwargs):
        run_id = self._as_str(kwargs.get("run_id"))

        duration = self._span_end(run_id)

        tool_name = "unknown_tool"

        # If we have an active run entry, use it
        active = self.active_tool_runs.pop(run_id, None) if run_id else None
        if active:
            tool_name = active["name"]

        tool = self._ensure_tool(tool_name)
        call = self._ensure_tool_call(tool_name, run_id or f"tool_call_{tool['call_count']}")

        tool["total_duration"] += duration if duration is not None else 0.0
        call["duration"] = duration

        # cast output to str for json serializable storage
        if hasattr(output, "content"):
            clean_output = output.content
        else:
            clean_output = str(output)
        call["output"] = clean_output

        message = f"Observation: {output}"
        self.chain_of_thought.append(message)
        print(f"\n[Real-time CoT - TOOL Result] {message}")

def parsing_error_handler(error: Exception):
    str_error = str(error)
    # Check if this is the specific parsing error we want to catch
    if "Could not parse LLM output:" in str_error:
        print(f"[Internal Log] Parsing error detected. Asking LLM to retry...")
        return f"An output parsing error occurred. Please ensure you are using the correct format. Error: {str_error}"

    return f"Agent Error: {str_error}"


def get_spark_session(extra_configs=None, benchmark="bird"):
    builder = SparkSession.builder \
        .appName("SparkSQLAgentTimer") \
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
        .config("spark.driver.memory", "2g")

    if extra_configs:
        for key, value in extra_configs.items():
            builder = builder.config(key, value)

    spark = builder.getOrCreate()
    return spark


def get_schema_manually(self, table_names):
    all_schemas = []

    if not table_names:
        table_names = [t.name for t in self._spark.catalog.listTables()]

    for table in table_names:
        try:
            df = self._spark.table(table)
            columns = ", ".join(
                [f"{f.name} {f.dataType.simpleString()}" for f in df.schema]
            )
            all_schemas.append(f"CREATE TABLE {table} ({columns});")
        except Exception:
            pass

    return "\n\n".join(all_schemas)


def get_spark_sql():

    spark_sql = SparkSQL(schema=None)
    spark_sql.get_table_info = types.MethodType(get_schema_manually, spark_sql)
    return spark_sql


def run_sparksql_query(spark_session, query):
    """
    Executes a Spark SQL query immediately and measures execution time
    """
    start_t = time.time()
    error = None
    result_df = None
    result_obj = None

    try:
        # without .collect() we would only measure the Query Parsing Time
        result_df = spark_session.sql(query)
        result_obj = result_df.collect()
    except Exception as e:
        error = str(e)
        print(f"[Spark Error] Query failed: {error}")

    end_t = time.time()
    duration = end_t - start_t

    return result_df, result_obj, duration


def get_spark_agent(spark_sql, llm):

    original_run = spark_sql.run

    def timed_run(self, command, fetch="all", _no_early_exit=False):

        # Log to chain of thought if callback is attached
        if hasattr(self, 'cb') and self.cb:
            self.cb.chain_of_thought.append(f"Spark Query Executed: {command}")

        start_t = time.time()

        try:
            result = original_run(command, fetch)
            error = None
        except Exception as e:
            result = None
            error = str(e)

        end_t = time.time()
        duration = end_t - start_t

        config.metrics["query"] = command
        config.metrics["spark_time"] = duration
        config.metrics["result"] = result if error is None else None
        config.metrics["spark_error"] = error

        # manually create a time span for spark execution
        span_entry = {
            "type": "spark",
            "name": "spark_run",
            "run_id": f"spark_run_{uuid.uuid4()}",
            "parent_run_id": None,
            "start": start_t,
            "end": end_t,
            "duration": duration,
            "meta": {
                "query": command,
                "error": error,
            }
        }
        config.metrics["spark_span"] = span_entry

        print(f"\n[Agent_Internal_Log] Spark Query Executed in {duration:.4f}s")
        print("Query:", command)
        print("Result/Error:", result if error is None else error)

        # FORCE EARLY EXIT IMMEDIATELY AFTER FIRST SPARK QUERY
        # but only for the model, not for the golden query execution
        if error:
            if _no_early_exit:
                raise
            raise AgentEarlyExit(...)
        else:
            if _no_early_exit:
                return result
            raise AgentEarlyExit(...)

    spark_sql.run = types.MethodType(timed_run, spark_sql)
    toolkit = SparkSQLToolkit(db=spark_sql, llm=llm)
    
    # Choose prompt based on config
    if config.FORCE_THOUGHT_GENERATION:
        from spark_toolkit.prompt import SQL_PREFIX_WITH_THOUGHTS
        prefix = SQL_PREFIX_WITH_THOUGHTS
    else:
        from spark_toolkit.prompt import SQL_PREFIX
        prefix = SQL_PREFIX
    
    agent = create_spark_sql_agent(
        llm=llm,
        toolkit=toolkit, verbose=True,
        handle_parsing_errors=parsing_error_handler,
        prefix=prefix
    )

    return agent


# merge overlapping intervals
def _merge_intervals(intervals):
    """intervals: list[(start,end)] with start/end floats. Returns merged union intervals."""
    # discard invalid intervals
    intervals = [(s, e) for s, e in intervals if s is not None and e is not None and e >= s]
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ms, me = merged[-1]
        # if next one starts before current ends (or start at same time), merge
        if s <= me:
            # extend current interval
            merged[-1] = (ms, max(me, e))
        else:
            # new disjoint interval
            merged.append((s, e))
    return merged

# total length of merged intervals
def _interval_total(intervals):
    return sum((e - s) for s, e in intervals)

# compute set difference of merged intervals: base \ subtract
# return all intervals that are in base but NOT in subtract
# base:     [----------]
# subtract:    [----]
# result:   [--]    [--]
def _subtract_intervals(base, subtract):
    if not base:
        return []
    if not subtract:
        return base[:]

    out = []
    j = 0
    for bs, be in base:
        cur_s, cur_e = bs, be
        # skip subtract intervals that end before cur_s
        while j < len(subtract) and subtract[j][1] <= cur_s:
            j += 1
        k = j
        # scan overlapping subtract intervals
        while k < len(subtract) and subtract[k][0] < cur_e:
            ss, se = subtract[k]
            # subtract starts AFTER current start -> keep interval between subtract start and current start
            if ss > cur_s:
                out.append((cur_s, min(ss, cur_e)))
            cur_s = max(cur_s, se)
            # all consumed
            if cur_s >= cur_e:
                break
            k += 1
        # keep remaining part (after subtract end until current end)
        if cur_s < cur_e:
            out.append((cur_s, cur_e))
    return _merge_intervals(out)

def compute_time_breakdown_by_overlap(cb, total_start, total_end):
    spans = cb.spans
    # add the spark span if available
    spark_span = config.metrics.get("spark_span", None)
    if spark_span:
        spans["spark_span"] = spark_span

    llm_intervals = []
    tool_intervals = []
    spark_intervals = []

    for s in spans.values():
        st, en = s.get("start"), s.get("end")
        if s.get("type") == "llm":
            llm_intervals.append((st, en))
        elif s.get("type") == "tool":
            tool_intervals.append((st, en))
        elif s.get("type") == "spark":
            spark_intervals.append((st, en))

    # merge intervals into a list of disjoint intervals
    print("LLM INTERVALS:", llm_intervals)
    llm_union = _merge_intervals(llm_intervals)
    print("LLM UNION:", llm_union)
    print("TOOL INTERVALS:", tool_intervals)
    tool_union = _merge_intervals(tool_intervals)
    print("TOOL UNION:", tool_union)
    spark_union = _merge_intervals(spark_intervals)
    print("SPARK UNION:", spark_union)

    T_total = total_end - total_start
    # total lengths of the disjoint intervals
    T_llm = _interval_total(llm_union)
    T_spark = _interval_total(spark_union)

    # Tool overhead = tool time that is NOT overlapped by LLM or Spark
    # overlapped_by = merged intervals of (llm_union + spark_union)
    overlapped_by = _merge_intervals(llm_union + spark_union)
    # get a list of intervals that cover the time of tool_union but NOT the time of overlapped_by
    # -> all intervals  of tool time that do not overlap with llm or spark
    # -> to get rid of the time where the tool is calling LLM or Spark (e.g. for the query_checker)
    tool_overhead_union = _subtract_intervals(tool_union, overlapped_by)
    # = time of tools exclusive of llm + spark
    T_tool_overhead = _interval_total(tool_overhead_union)

    # Orchestration = remainder
    T_orchestr = max(0.0, T_total - (T_llm + T_spark + T_tool_overhead))

    return {
        "total_time": T_total,
        "llm_time": T_llm,
        "spark_time": T_spark,
        "tool_overhead_time": T_tool_overhead,
        "orchestration_time": T_orchestr,
        "llm_union": llm_union,
        "spark_union": spark_union,
        "tool_union": tool_union,
        "tool_overhead_union": tool_overhead_union,
    }

def run_nl_query(agent, nl_query, llm=None):

    print("--- Starting Agent ---")
    total_start = time.time()

    cb = AgentMonitoringCallback()

    # Attach callback to the db object so timed_run can access it
    # agent.tools is a list of tools. We need to find the one with the db.
    # Usually SparkSQLToolkit adds tools that share the same db instance.
    if hasattr(agent, 'tools'):
        for tool in agent.tools:
            if hasattr(tool, 'db'):
                tool.db.cb = cb
                break

    try:
        nl_query = nl_query + "\n\n" + config.DEFAULT_PROMPT_SUFIX
        from langchain_core.messages import HumanMessage
        response = agent.invoke({"messages": [HumanMessage(content=nl_query)]}, config={"callbacks": [cb]})
        final_answer = response["messages"][-1].content

    except AgentEarlyExit as e:
        print("--- Exit Triggered (Parsing Bypass) ---")
        print(e)
        final_answer = e.answer

    except AgentLoopException as e:
        print("--- Loop Detected ---")
        print(e)
        final_answer = str(e)
        config.metrics["query"] = None

    except Exception as e:
        print("--- Agent Error Occurred ---")
        print(str(e))
        final_answer = str(e)

    # end all open runs (happens in case of early exit)
    for run_id in list(cb.active_tool_runs.keys()):
        name = cb.active_tool_runs[run_id]["name"]
        if name == "schema_sql_db":
            cb.on_tool_end(output="Early Exit because of call to SQL. Duration is spark_time.", run_id=run_id)
        else:
            cb.on_tool_end(output="EXCEPTION: Early Exit. Duration is off.", run_id=run_id)


    total_end = time.time()

    llm_name = ""
    # TODO: add other LLMs as needed
    if isinstance(llm, ChatGoogleGenerativeAI):
        llm_name = "google"
    elif isinstance(llm, ChatCloudflareWorkersAI):
        llm_name = "cloudflare"
    elif isinstance(llm, ChatAnthropic):
        llm_name = "claude"
    elif isinstance(llm, ChatOpenAI):
        llm_name = "openai"
    else:
        llm_name = "unknown"

    config.metrics["llm"] = llm_name
    config.metrics["answer"] = final_answer
    total_time = total_end - total_start
    config.metrics["total_time"] = total_time
    spark_time = config.metrics.get("spark_time", total_time)
    config.metrics["translation_time"] = total_time - spark_time
    config.metrics["llm_requests"] = cb.count
    config.metrics["chain_of_thought"] = cb.chain_of_thought
    config.metrics["agent_thoughts"] = cb.agent_thoughts
    config.metrics["input_tokens"] = cb.input_tokens
    config.metrics["input_token_details"] = cb.input_token_details_total
    config.metrics["output_tokens"] = cb.output_tokens
    config.metrics["output_token_details"] = cb.output_token_details_total
    config.metrics["prompt"] = cb.last_prompt
    config.metrics["final_answer"] = cb.last_answer
    config.metrics["llm_calls"] = cb.llm_calls

    # Calculate Cloudflare Neurons if applicable
    if llm and isinstance(llm, ChatCloudflareWorkersAI):
        model_name = llm.model
        pricing = get_cloudflare_neuron_pricing(model_name)
        if pricing:
            input_neurons = (cb.input_tokens / 1_000_000) * pricing["input_neurons_per_m"]
            output_neurons = (cb.output_tokens / 1_000_000) * pricing["output_neurons_per_m"]
            total_neurons = input_neurons + output_neurons
            config.metrics["cloudflare_neurons"] = total_neurons
            print(f"[Cloudflare Cost] Estimated Neurons: {total_neurons:.2f}")

    print("\n--- Agent Finished Successfully ---")
    # Export tool metrics
    config.metrics["tool_metrics"] = {}
    for tool_name, tool_dict in cb.tool_metrics.items():
        config.metrics["tool_metrics"][f"{tool_name}"] = tool_dict
        config.metrics[f"time_{tool_name}"] = tool_dict["total_duration"]
        if "total_tokens" not in tool_dict:
            continue
        config.metrics[f"tokens_{tool_name}"] = tool_dict["total_tokens"]

    ############################################################################
    #               TIME COMPUTATION
    ############################################################################
    # Problem: QueryCheckerTool internally calls LLM too, so total_time_tools includes some LLM time as well
    # total times per tool (LangChain tools)
    total_times_per_tool = {tool_name: tool_dict.get("total_duration", 0) for tool_name, tool_dict in config.metrics.get("tool_metrics", {}).items()}
    total_time_tools = sum(total_times_per_tool.values())
    # total times per llm call
    total_times_per_llm_call = {}
    for llm_call_id, llm_call_data in config.metrics.get("llm_calls", {}).items():
        total_times_per_llm_call[llm_call_id] = llm_call_data.get("duration", 0)
    total_time_llm_calls = sum(total_times_per_llm_call.values())

    config.metrics["total_time_tools"] = total_time_tools
    config.metrics["total_times_per_tool"] = total_times_per_tool
    config.metrics["total_times_per_llm_call"] = total_times_per_llm_call
    config.metrics["total_time_llm_calls"] = total_time_llm_calls

    # compute disjoint time intervals to avoid double counting of overlapping times
    breakdown = compute_time_breakdown_by_overlap(cb, total_start, total_end)
    # total time of run_nl_query = total time needed for LLM execution + LangChain overhead with Spark execution time (inside tool time)
    total_time = breakdown["total_time"]
    config.metrics["total_time"] = total_time
    # Spark time is included in the tool time, output here for clarity
    spark_time = config.metrics.get("spark_time", total_time)
    config.metrics["translation_time"] = total_time - spark_time

    # pure LLM time (might overlap with some tool time)
    config.metrics["llm_time_total"] = breakdown["llm_time"]
    # pure Spark time (should not overlap with anything else)
    config.metrics["spark_time_spans"] = breakdown["spark_time"]
    # tool overhead time (excluding LLM and Spark time)
    # this means: e.g., query_checker_sql_db internally calls LLM, so the time for this LLM call is not counted here
    config.metrics["tool_overhead_time"] = breakdown["tool_overhead_time"]
    # everything else is orchestration time
    config.metrics["orchestration_time"] = breakdown["orchestration_time"]

    print("\n========================================")
    print("=== DETAILED TIME BREAKDOWN ===")
    print("========================================")
    print("[Internal Log] Disjoint Time Breakdown (overlap-based, sums to total_time):")
    print(f"  - Total Time           : {breakdown['total_time']:.6f} sec")
    print(f"  - LLM Time (union)     : {breakdown['llm_time']:.6f} sec")
    print(f"  - Spark Time (union)   : {breakdown['spark_time']:.6f} sec")
    print(f"  - Tool Overhead (excl LLM + Spark) : {breakdown['tool_overhead_time']:.6f} sec")
    print(f"  - Orchestration        : {breakdown['orchestration_time']:.6f} sec")
    print("  ---------------------------------------------")
    # total time = LLM time + Spark time + Tool Overhead + Orchestration
    print("  Sum buckets            : "
        f"{(breakdown['llm_time'] + breakdown['spark_time'] + breakdown['tool_overhead_time'] + breakdown['orchestration_time']):.6f} sec")
    ############################################################################


def process_result():
    result = config.metrics.get("result", None)
    result = result_to_obj(result)
    error = config.metrics.get("spark_error", None)

    json_result = {
        "llm": config.metrics.get("llm", None),
        "sparksql_query": config.metrics.get("query", None),
        "query_id": config.metrics.get("query_id", None),
        "iteration": config.metrics.get("iteration", None),
        "difficulty": config.metrics.get("difficulty", None),
        "execution_status": "ERROR" if error else ("VALID" if config.metrics.get("query", None) else "NOT_EXECUTED"),
        "query_result": result,
        "spark_error": error,
        "total_time": config.metrics.get("total_time", -1),
        "spark_time": config.metrics.get("spark_time", -1),
        "translation_time": config.metrics.get("translation_time", -1),
        "total_times_per_tool": config.metrics.get("total_times_per_tool", {}),
        "total_time_tools": config.metrics.get("total_time_tools", -1),
        "total_time_tools_excl_spark_llm": config.metrics.get("tool_overhead_time", -1),
        "total_times_per_llm_call": config.metrics.get("total_times_per_llm_call", {}),
        "total_time_llm_calls": config.metrics.get("total_time_llm_calls", -1),
        "orchestration_time": config.metrics.get("orchestration_time", -1),
        "llm_requests": config.metrics.get("llm_requests", 0),
        "chain_of_thought": config.metrics.get("chain_of_thought", []),
        "agent_thoughts": config.metrics.get("agent_thoughts", []),
        "total_tokens": config.metrics.get("input_tokens", 0) + config.metrics.get("output_tokens", 0),
        "input_tokens": config.metrics.get("input_tokens", 0),
        "input_token_details": config.metrics.get("input_token_details", {}),
        "output_tokens": config.metrics.get("output_tokens", 0),
        "output_token_details": config.metrics.get("output_token_details", {}),
        "cloudflare_neurons": config.metrics.get("cloudflare_neurons", None),
        "prompt": config.metrics.get("prompt", None),
        "final_answer": config.metrics.get("final_answer", None),
        "llm_calls": config.metrics.get("llm_calls", {}),
        "tool_metrics": config.metrics.get("tool_metrics", {}),
    }

    return json_result


def print_results(json_result, print_result=False):
    print("\n" + "="*40)
    print(" PERFORMANCE METRICS")
    print("="*40)

    total_time = json_result.get("total_time")
    spark_time = json_result.get("spark_time")
    translation_time = json_result.get("translation_time")

    status = json_result.get('execution_status')
    color_start = ""
    color_end = "\033[0m"

    if status == "VALID":
        color_start = "\033[92m"  # Green
    elif status == "ERROR":
        color_start = "\033[91m"  # Red
    elif status == "NOT_EXECUTED":
        color_start = "\033[93m"  # Yellow

    print(f"Execution Status: {color_start}{status}{color_end}")
    print(f"1. Total End-to-End Time    : {total_time:.4f} sec" if total_time is not None and total_time != -1 else "1. Total End-to-End Time    : N/A")
    print(f"2. Spark Execution Time     : {spark_time:.4f} sec" if spark_time is not None and spark_time != -1 else "2. Spark Execution Time     : N/A")
    print(f"3. Input Translation (LLM)  : {translation_time:.4f} sec" if translation_time is not None and translation_time != -1 else "3. Input Translation (LLM) Time  : N/A")
    print(f"4. LLM Requests             : {json_result.get('llm_requests')}")
    print(f"5. Input Tokens             : {json_result.get('input_tokens')}")
    print(f"6. Output Tokens            : {json_result.get('output_tokens')}")

    neurons = json_result.get('cloudflare_neurons')
    if neurons is not None:
        print(f"7. Cloudflare Neurons       : {neurons:.2f}")

    print(f"Spark Query: {color_start}{json_result.get('sparksql_query')}{color_end}")

    error = json_result.get("spark_error")
    print(f"Spark Error (first 50 chars): {error[:50] if error else 'None'}")
    print("="*40)

    if json_result.get('execution_status') == "VALID" and print_result:
        print(f"Query Result: {json_result.get('query_result')}")


def pretty_print_cot(json_result):
    print("\n" + "="*40)
    print(" CHAIN OF THOUGHT")
    print("="*40)

    cot = json_result.get("chain_of_thought", [])
    if not cot:
        print("No Chain of Thought available.")
        return

    for step in cot:
        print(step)
        print("-" * 20)
    print("="*40)


def save_results(results, output_file=None, query_id=None, iteration=1, additional_data=None, base_folder="."):
    if output_file is None:
        current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        output_file = f"{current_date}_ID_{query_id}_ITER_{iteration}_{random_suffix}.json"
    print(f"[Internal Log] Saving results to {output_file}")

    if additional_data:
        results.update(additional_data)

    os.makedirs(base_folder, exist_ok=True)

    with open(os.path.join(base_folder, output_file), 'w') as f:
        json.dump(results, f, indent=4)

    return os.path.join(base_folder, output_file)
