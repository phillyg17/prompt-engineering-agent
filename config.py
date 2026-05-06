"""
Centralized configuration for prompt-engineering-agent.

Users can edit these constants directly or override them via environment variables.
"""

import os

# ---------------------------------------------------------------------------
# LLM model configuration
# ---------------------------------------------------------------------------
OLLAMA_MODEL = os.getenv("PROMPT_AGENT_MODEL", "gemma4:31b-cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
LLAMA_SERVER_MODEL = os.getenv("LLAMA_SERVER_MODEL", "Qwen3.6-35B-A3B-UD-Q6_K_XL")

# ---------------------------------------------------------------------------
# Execution defaults
# ---------------------------------------------------------------------------
DEFAULT_CONCURRENCY = int(os.getenv("PROMPT_AGENT_CONCURRENCY", "20"))
DEFAULT_MAX_ITERATIONS = int(os.getenv("PROMPT_AGENT_MAX_ITER", "20"))
