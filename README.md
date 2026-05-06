# Prompt Engineering Agent (LangGraph)

A multi-agent system built with **LangGraph** that iteratively writes and critiques prompts until they meet a quality threshold. It features an **intelligent document router** that selects only the relevant files from your `context/` folder, an **automatic Ollama-to-llama-server fallback**, and a standalone **Prompt Optimizer Agent** that maximizes accuracy against transcript-based truth data.

---

## Table of Contents

- [Architecture (Prompt Agent)](#architecture-prompt-agent)
- [Architecture (Optimizer Agent)](#architecture-optimizer-agent)
- [Setup](#setup)
- [Prompt Agent Usage](#prompt-agent-usage)
- [Optimizer Agent Usage](#optimizer-agent-usage)
- [Data Format](#data-format)
- [Anti-Contamination Guardrails](#anti-contamination-guardrails)
- [Real-time Logging](#real-time-logging)
- [Customization](#customization)

---

## Architecture (Prompt Agent)

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

---

## Architecture (Optimizer Agent)

The **Prompt Optimizer Agent** (`optimizer_agent.py`) is a standalone tool that iteratively rewrites a prompt to maximize its accuracy against transcript-based truth data. It is designed for batch prompt optimization against large datasets (3000+ transcripts) with full async parallelism.

```
Start
  |
  v
Load Transcripts + Truth Data ...... match by filename across CSV files
  |
  v
Test Prompt (async) .............. run prompt against ALL transcripts concurrently
  |                                    extract answer before "::", compare case-insensitive
  |
  v
Score ≥ target? .................. YES → save best prompt → STOP
  |                                    NO → continue
  |
  v
Max tests reached? ............... YES → save best prompt → STOP
  |                                    NO → continue
  |
  v
Critic Node ........................ analyzes failure patterns from transcripts
  |                                    (may cite specific evidence from transcripts)
  |
  v
Writer Node ........................ rewrites prompt using best prompt + critic feedback
  |                                    (must NOT embed verbatim transcript text)
  |
  v
Save prompt if improving .......... only if accuracy ≥ previous best
  |
  v
Loop back to Test Prompt
```

### Key Optimizer Features

| Feature | Description |
|---------|-------------|
| **Async batch testing** | Tests all transcripts concurrently with a bounded semaphore (default 20). Handles 3000+ transcripts efficiently. |
| **Filename matching** | Links transcript CSVs to truth CSVs by the `Filename` column. Multiple transcript rows per file are grouped by speaker. |
| **Answer extraction** | Extracts the first token before `::`, lowercases it, and compares to the truth answer case-insensitively. |
| **Smart saving** | Prompts are saved to `prompts/` **only if** the accuracy meets or exceeds the previous best. Worse prompts are discarded. |
| **Dual threshold control** | Set `--target-accuracy` (stop when reached) and/or `--max-iterations` (stop after N tests). Set either to `"none"` to ignore it. |
| **Seed from existing** | Use `--use-existing-prompt` to load `prompts/prompt.txt` as the starting point, or `--context` to generate from scratch. |
| **Ollama fallback** | Automatically switches to llama-server if Ollama is unavailable (shared logic with Prompt Agent). |

---

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

You can also change the model via env vars:

```bash
export PROMPT_AGENT_MODEL="llama3.1"
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

---

## Prompt Agent Usage

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

---

## Optimizer Agent Usage

The Optimizer Agent runs as a standalone script with its own CLI. It requires:
- **Transcript CSVs** in a folder (default: `transcripts/`)
- **Truth CSVs** in a folder (default: `truth data/`)
- A **starting prompt** from `prompts/prompt.txt` or generated from `--context`

### Data Format

**Transcript CSV** (`transcripts/*.csv`):
```csv
Filename,Party,StartOffset (sec),EndOffset (sec),Score,Text
9876543210123456789_9876543210123456789.nmf,Agent,0,0,1,"Hello, how can I help?"
9876543210123456789_9876543210123456789.nmf,Customer,0,0,1,"I need to check my balance."
```

**Truth CSV** (`truth data/*.csv`):
```csv
Filename,Answer,,,,
9876543210123456789_9876543210123456789.nmf,Reinforce
some_other_file.nmf,Refine
```

> **Note:** Files are matched by the value in the `Filename` column. Consecutive rows with the same filename are grouped into a single transcript. Consecutive same-speaker lines are merged into a single dialogue turn.

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--target-accuracy` | `"none"` | Target accuracy % (e.g. `90` or `95.0`). Use `"none"` to ignore. |
| `--max-iterations` | `"none"` | Max number of prompt tests to run. Use `"none"` to loop until target is met. If **both** are `"none"`, defaults to `20`. |
| `--max-transcripts` | `"none"` | Test only the first N transcripts (sorted by filename). Use `"none"` for all. |
| `--use-existing-prompt` | `False` | Load `prompts/prompt.txt` as the seed prompt. |
| `--context` | `""` | User description of the task (required when not using existing prompt). |
| `--prompts-folder` | `"prompts"` | Where to save improving prompts. |
| `--transcripts-folder` | `"transcripts"` | Where transcript CSVs are located. |
| `--truth-folder` | `"truth data"` | Where truth CSVs are located. |
| `--concurrency` | `20` | Max concurrent LLM calls during testing. |

### Example Commands

**Run until 90% accuracy is reached** (using existing prompt):

```bash
python optimizer_agent.py \
  --use-existing-prompt \
  --target-accuracy 90 \
  --max-iterations none \
  --concurrency 20
```

**Run exactly 5 tests regardless of score** (using existing prompt):

```bash
python optimizer_agent.py \
  --use-existing-prompt \
  --target-accuracy none \
  --max-iterations 5 \
  --concurrency 20
```

**Test only the first 100 transcripts** (using existing prompt):

```bash
python optimizer_agent.py \
  --use-existing-prompt \
  --target-accuracy 90 \
  --max-iterations none \
  --max-transcripts 100 \
  --concurrency 20
```

**Generate initial prompt from scratch** (with context):

```bash
python optimizer_agent.py \
  --context "Evaluate if the agent offered additional assistance at the end of a customer service call. Output the answer as Reinforce, Refine, or Redirect followed by a reason." \
  --target-accuracy 90 \
  --max-iterations none \
  --concurrency 20
```

**Use custom folders**:

```bash
python optimizer_agent.py \
  --use-existing-prompt \
  --target-accuracy 95 \
  --transcripts-folder my_transcripts \
  --truth-folder my_truth \
  --prompts-folder my_prompts \
  --concurrency 20
```

### Answer Matching Logic

The optimizer extracts the **first token before `::`** from the LLM output, trims whitespace, lowercases it, and compares it to the truth answer case-insensitively.

| LLM Output | Extracted | Truth | Match? |
|-----------|-----------|-------|--------|
| `Reinforce::agent met all criteria` | `reinforce` | `reinforce` | ✅ Yes |
| `1::n/a` | `1` | `reinforce` | ❌ No |
| `REFINE::assumptive close detected` | `refine` | `refine` | ✅ Yes |

### Output Files

Improving prompts are saved as:
```
prompts/prompt_iter_{test_number}_{accuracy}pct_{YYYYMMDD_HHMMSS}.txt
```

Example:
```
prompts/prompt_iter_2_90.0pct_20260506_135707.txt
```

If a new prompt scores **lower** than the previous best, it is **not saved** and the next rewrite uses the best-performing prompt so far.

---

## Anti-Contamination Guardrails

The Optimizer Agent separates what the **Critic** sees from what the **Writer** writes to prevent contaminated prompts.

| Node | Can see specific transcript text? | Can write verbatim transcript text into prompt? |
|------|-----------------------------------|-----------------------------------------------|
| **Critic** | ✅ Yes — full transcript excerpts, caller names, and LLM outputs to identify failure patterns | N/A (does not write prompts) |
| **Writer** | ❌ No — receives only abstract failure summaries (expected vs actual answers) and critic feedback | **Explicitly forbidden** — system prompt prohibits copying, quoting, or embedding exact transcript examples, caller names, dates, or real data |

This ensures rewritten prompts remain **generalizable** to unseen transcripts.

---

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

**Optimizer Agent logging:**
```
13:56:08 | INFO     | --- Test 1 ---
13:56:11 | INFO     | Progress: 10/10 tested
13:56:11 | INFO     | Accuracy: 70.00% (7/10) | Failures: 3
13:56:37 | INFO     | Critic feedback: [patterns identified]
13:56:54 | INFO     | Writer produced new prompt (3137 chars)
13:57:07 | INFO     | Accuracy: 90.00% (9/10) | Failures: 1
13:57:07 | INFO     | Target accuracy 90.00% reached!
```

---

## Customization

### Prompt Agent

- **Fallback behavior**: The agent automatically falls back to llama-server if Ollama fails. Configure via `LLAMA_SERVER_URL` and `LLAMA_SERVER_MODEL` env vars.
- **Score threshold**: Change `SCORE_THRESHOLD` in `prompt_agent.py` (default 3.0).
- **Number of critic tests**: Change `NUM_TESTS` (default 1).
- **Context directory**: Change `CONTEXT_DIR` in `prompt_agent.py`.
- **Test data directory**: Changed via `TEST_DATA_DIR` in `prompt_agent.py`.
- **Router behavior**: The router auto-selects the single available file if only one exists, and skips the LLM call. You can modify `ROUTER_SYSTEM` to tune its selection criteria.

### Optimizer Agent

- **Concurrency**: Adjust `--concurrency` (default 20) based on your Ollama/llama-server capacity. Higher = faster but more load.
- **Subset testing**: Use `--max-transcripts N` to test only the first N transcripts (sorted by filename). Useful for quickly validating changes on large datasets.
- **Answer extraction**: Modify `extract_answer()` in `optimizer_agent.py` if your output format differs from `token::reason`.
- **Maximum failure examples to critic**: Change `failures[:15]` in `critic_node()` (default 15).
- **Maximum failure examples to writer**: Change `failures[:10]` in `writer_node()` (default 10).
- **LLM models**: Edit `config.py` to change Ollama model, llama-server model, or base URLs.

---

## File Structure

```
prompt-engineering-agent/
├── prompt_agent.py           # Main LangGraph prompt engineering agent
├── optimizer_agent.py        # Standalone prompt optimizer agent
├── config.py                 # Centralized model & execution configuration (NEW)
├── requirements.txt
├── README.md
├── context/                  # Drop PDF/DOCX files here (for Prompt Agent)
├── transcripts/              # Drop transcript CSVs here (default for Optimizer)
├── truth data/               # Drop truth CSVs here (default for Optimizer)
├── prompts/                  # Generated prompts saved here
│   └── prompt.txt            # Starting prompt (can be used by Optimizer)
└── test_data/                # JSON truth files for Prompt Agent tester
```
