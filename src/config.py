DB_PATH = "db"
BENCHMARK_FILE = "dev.json"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_PROMPT_SUFIX = ""
SCHEMA_LOOP_COUNT = 3
FORCE_THOUGHT_GENERATION = False  # If True, agent generates text thoughts before tool calls

from enum import Enum

class Provider(Enum):
    GOOGLE = "google"
    CLOUDFLARE = "cloudflare"
    CLAUDE = "claude"
    OPENAI = "openai"
    NVIDIA = "nvidia"


metrics = {
    "total_time": -1,
    "spark_exec_time": -1,
    "translation_time": -1,
    "sparksql_query": None,
    "answer": None
}

DEFAULT_MODELS = {
    Provider.GOOGLE: "gemini-2.5-flash",
    Provider.CLOUDFLARE: "@cf/meta/llama-4-scout-17b-16e-instruct",
    Provider.CLAUDE: "claude-opus-4-5",
    Provider.OPENAI: "gpt-5.2",
    Provider.NVIDIA: "moonshotai/kimi-k2.5"
}

# OpenAI reasoning models that expose thinking/reasoning
REASONING_MODELS = {
    "o1": "o1",  # Full reasoning model
    "o1-mini": "o1-mini",  # Faster, cheaper reasoning
    "o3-mini": "o3-mini",  # Latest reasoning model
}
