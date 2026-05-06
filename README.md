# Prompt Engineering Agent (LangGraph)

A multi-agent system built with **LangGraph** that iteratively writes and critiques prompts until they meet a quality threshold. It features an **intelligent document router** that selects only the relevant files from your `context/` folder, and an **automatic Ollama-to-llama-server fallback** that switches to a local llama.cpp server when Ollama is unavailable.

## Architecture

```
Entry
  |
  v
File Scanner ....................... lists all PDF / DOCX files in context/
  |
  v
Router Agent ....................... LLM-powered agent picks relevant files based on user context
  |
  v
Document Loader .................... reads only the selected files via tools
  |
  v
Writer Agent ....................... generates prompt using user context + document text
  |
  v
Critic Agent ....................... evaluates prompt on 3 dimensions
  |     (clarity, specificity, output_predictability)
  |     + execution test with dummy data (if no truth data exists)
  |
  v
Conditional Edge ................... scores ≥ 3.0 on all dims? → Tester
  |                                    otherwise → loop back to Writer
  v
Tester ............................. validates prompt against truth data from test_data/*.json
  |                                    (skipped if no truth data)
  v
END
```

### Execution Testing

- **Dummy data**: If `test_data/` has no JSON truth files, the Critic generates a dummy input and runs an execution test to observe real LLM output. This helps evaluate output predictability.
- **Truth data**: If `test_data/` contains JSON files, the Critic skips dummy execution tests and the Tester node validates the final prompt against real transcripts, comparing extracted scores to expected results.

### Why a Router?

If you drop 10 files into `context/`, the Writer doesn't need all 10. The **Router Agent** uses the LLM to inspect filenames (and your written context) and decide which documents are actually relevant. This saves tokens, reduces noise, and keeps the prompt focused.

## Setup

```bash
cd prompt-engineering-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Ollama (local LLM)

Make sure [Ollama](https://ollama.com) is installed and running. Pull the model you want to use (e.g.):

```bash
ollama pull llama3.1
```

By default the agent talks to `http://localhost:11434`. If your Ollama server is elsewhere, set:

```bash
export OLLAMA_BASE_URL="http://localhost:11434"
```

You can also change the model or iteration limit via env vars:

```bash
export PROMPT_AGENT_MODEL="llama3.1"
export PROMPT_AGENT_MAX_ITER="5"
```

### Automatic Fallback to llama-server

If Ollama is unavailable (connection errors, 502 Bad Gateway, etc.), the agent **automatically switches** to a local llama.cpp server (llama-server) running on `http://127.0.0.1:8080`. No manual intervention required.

To use the fallback, make sure llama-server is running with an OpenAI-compatible API:

```bash
ollama serve  # or your llama-server instance
```

Configure the fallback via env vars:

```bash
export LLAMA_SERVER_URL="http://127.0.0.1:8080"
export LLAMA_SERVER_MODEL="Qwen3.6-35B-A3B-UD-Q6_K_XL"
```

The fallback is **lazy** — Ollama is only probed on the first LLM call (writer node). Once Ollama fails, all subsequent calls use llama-server for the entire session.

## Usage

### 1. Add documents (optional)

Drop any `.pdf` or `.docx` files into the `context/` folder. The Router will intelligently choose which ones to use.

### 2. Run the agent

**Interactive mode:**

```bash
python prompt_agent.py
```

**Pass written context directly:**

```bash
python prompt_agent.py --context "A prompt that extracts action items from meeting transcripts and formats them as a markdown checklist"
```

**Documents-only mode** (skip written context by pressing Enter at the prompt):

```bash
python prompt_agent.py
> [press Enter]
```

## Real-time Logging

Every node execution is logged so you can watch the agents collaborate in real time:

```
12:34:56 | INFO     | --- FILE SCANNER NODE ---
12:34:56 | INFO     | Found 3 document(s) in context/: ['specs.pdf', 'style-guide.docx', 'invoice.pdf']
12:34:57 | INFO     | --- ROUTER NODE ---
12:34:58 | INFO     | Router selected 2 file(s): ['specs.pdf', 'style-guide.docx']
12:34:58 | INFO     | --- DOCUMENT LOADER NODE ---
12:34:58 | INFO     | Loading document: specs.pdf
12:34:59 | INFO     | Loading document: style-guide.docx
...
```

## Customization

- **Fallback behavior**: The agent automatically falls back to llama-server if Ollama fails. Configure via `LLAMA_SERVER_URL` and `LLAMA_SERVER_MODEL` env vars.
- **Score threshold**: Change `SCORE_THRESHOLD` in `prompt_agent.py` (default 3.0).
- **Number of critic tests**: Change `NUM_TESTS` (default 1).
- **Context directory**: Change `CONTEXT_DIR` in `prompt_agent.py`.
- **Test data directory**: Changed via `TEST_DATA_DIR` in `prompt_agent.py`.
- **Router behavior**: The router auto-selects the single available file if only one exists, and skips the LLM call. You can modify `ROUTER_SYSTEM` to tune its selection criteria.
