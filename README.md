# Prompt Engineering Agent (LangGraph)

A multi-agent system built with **LangGraph** that iteratively writes and critiques prompts until they meet a quality threshold. It now features an **intelligent document router** that selects only the relevant files from your `context/` folder before reading them.

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
Critic Agent ....................... runs 5 independent evaluation tests
  |                                    (clarity, specificity, output_predictability)
  |
  v
Conditional Edge ................... scores ≥ 3.0 on all dims? → END
  |                                    otherwise → loop back to Writer
  v
Loop
```

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

- **Swap LLM provider**: The agent uses `ChatOllama` by default. To switch back to OpenAI, replace `ChatOllama` with `ChatOpenAI` from `langchain_openai` and set your `OPENAI_API_KEY`.
- **Score threshold**: Change `SCORE_THRESHOLD` in `prompt_agent.py` (default 3.0).
- **Number of tests**: Change `NUM_TESTS` (default 5).
- **Context directory**: Change `CONTEXT_DIR` in `prompt_agent.py`.
- **Router behavior**: The router auto-selects the single available file if only one exists, and skips the LLM call. You can modify `ROUTER_SYSTEM` to tune its selection criteria.
