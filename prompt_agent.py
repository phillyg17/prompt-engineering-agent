#!/usr/bin/env python3
"""
LangGraph-powered Prompt Engineering Agent with Document Context & Intelligent Routing.

Flow:
1. User provides optional written context + drops documents into context/ folder.
2. File Scanner node lists all PDF / DOCX files in context/.
3. Router agent (LLM-powered) decides which files are relevant to the user's request.
4. Document Loader node parses only the selected files via tools.
5. Writer agent generates a prompt using written context + extracted document text.
6. Critic agent evaluates the prompt across 5 independent tests.
   For each test it grades clarity, specificity, and output predictability (1-5).
   Scores are aggregated (averaged) per dimension.
7. If any dimension scores < 3, the writer rewrites the prompt using critique feedback.
8. Loop continues until all dimensions >= 3 or a max iteration limit is reached.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, TypedDict

from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONTEXT_DIR = BASE_DIR / "context"

MODEL_NAME = os.getenv("PROMPT_AGENT_MODEL", "qwen3.5:397b-cloud")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MAX_ITERATIONS = int(os.getenv("PROMPT_AGENT_MAX_ITER", "10"))
SCORE_THRESHOLD = 3.0
NUM_TESTS = 5

# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------
llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.7)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    user_context: str          # typed context from CLI
    available_files: List[str] # files found in context/
    selected_files: List[str]  # files chosen by router
    doc_context: str           # extracted text from selected documents
    current_prompt: str
    iteration: int
    scores: Dict[str, float]
    logs: List[str]
    done: bool


# ---------------------------------------------------------------------------
# Document parsing (plain functions + tool wrappers)
# ---------------------------------------------------------------------------
def _read_pdf_impl(path: str) -> str:
    """Plain implementation — extract text from PDF."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise ImportError("pypdf is required for PDF support. Install: pip install pypdf") from exc

    reader = PdfReader(path)
    text_parts: List[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_parts.append(page_text)
    result = "\n".join(text_parts)
    logger.debug("PDF %s extracted %d chars", path, len(result))
    return result


def _read_docx_impl(path: str) -> str:
    """Plain implementation — extract text from DOCX."""
    try:
        import docx
    except ImportError as exc:
        raise ImportError("python-docx is required for DOCX support. Install: pip install python-docx") from exc

    document = docx.Document(path)
    text_parts: List[str] = [p.text for p in document.paragraphs if p.text.strip()]
    result = "\n".join(text_parts)
    logger.debug("DOCX %s extracted %d chars", path, len(result))
    if result == "":
        logger.warning("DOCX %s produced zero characters after extraction. File may be unparseable.", path)
    return result


@tool
def read_pdf(path: str) -> str:
    """Extract plain text from a PDF file given its filesystem path."""
    return _read_pdf_impl(path)


@tool
def read_docx(path: str) -> str:
    """Extract plain text from a Word document (.docx) given its filesystem path."""
    return _read_docx_impl(path)


@tool
def run_prompt_test(prompt: str, dummy_input: str) -> str:
    """
    Execute a prompt against a dummy input to observe the actual LLM output.
    This is useful for testing output predictability and consistency.
    """
    system_msg = (
        "You are a test executor. A prompt engineer wrote the PROMPT below. "
        "Your job is to apply that PROMPT to the DUMMY INPUT and produce ONLY the output. "
        "Do not explain, do not add commentary, do not use markdown unless the prompt asks for it. "
        "Just follow the prompt's instructions exactly."
    )
    human_msg = (
        f"--- PROMPT TO EXECUTE ---\n{prompt}\n\n"
        f"--- DUMMY INPUT ---\n{dummy_input}\n\n"
        f"Now produce the output by applying the prompt to the dummy input."
    )
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg),
    ]
    # Use a lower temperature for more deterministic test outputs
    test_llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, temperature=0.3)
    response = test_llm.invoke(messages)
    return response.content.strip()


# ---------------------------------------------------------------------------
# Helper: structured extraction
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """Naive JSON extractor – finds the first JSON object in the text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model output.")
    return json.loads(text[start:end + 1])


# ---------------------------------------------------------------------------
# Node: File Scanner
# ---------------------------------------------------------------------------
def file_scanner_node(state: AgentState) -> AgentState:
    logger.info("--- FILE SCANNER NODE ---")
    if not CONTEXT_DIR.exists():
        logger.info("Context directory %s does not exist.", CONTEXT_DIR)
        return {**state, "available_files": []}

    files = sorted(
        f.name for f in CONTEXT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in (".pdf", ".docx")
    )
    logger.info("Found %d document(s) in context/: %s", len(files), files)
    return {**state, "available_files": files}


# ---------------------------------------------------------------------------
# Node: Router
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = (
    "You are an intelligent document router. Your job is to decide which files "
    "from a provided list are relevant to the user's request. "
    "You must respond with a single JSON object and nothing else:\n"
    '{"selected_files": ["filename1.pdf", "filename2.docx"]}\n'
    "If no files are relevant, return an empty list."
)


def router_node(state: AgentState) -> AgentState:
    logger.info("--- ROUTER NODE ---")
    available = state.get("available_files", [])

    if not available:
        logger.info("No documents available to route.")
        return {**state, "selected_files": []}

    if len(available) == 1:
        logger.info("Only one document available (%s). Auto-selecting it.", available[0])
        return {**state, "selected_files": available}

    user_ctx = state.get("user_context", "").strip()
    files_list = "\n".join(f"  - {name}" for name in available)

    # If user provided no written context, route purely on filenames (select all by default)
    if not user_ctx:
        logger.info("No written context provided — routing purely on filenames.")

    human_msg = (
        f"The user wants a prompt for the following context:\n{user_ctx}\n\n"
        f"Available documents in context/ folder:\n{files_list}\n\n"
        f"Which of these documents are relevant? Return JSON."
    )
    messages = [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=human_msg),
    ]

    try:
        response = llm.invoke(messages)
        obj = _extract_json(response.content)
        selected = list(obj.get("selected_files", []))
        # Validate that selected files actually exist in available list
        valid = [f for f in selected if f in available]
        if len(valid) != len(selected):
            logger.warning("Router returned unknown files. Filtering to known files only.")
        selected = valid
    except Exception as exc:
        logger.warning("Router failed to parse response: %s. Defaulting to all files.", exc)
        selected = available

    # Fallback: if router returned nothing, use all files (safer than missing context)
    if not selected:
        logger.warning("Router returned zero files. Defaulting to all available documents.")
        selected = list(available)

    logger.info("Router selected %d file(s): %s", len(selected), selected)
    return {**state, "selected_files": selected}


# ---------------------------------------------------------------------------
# Node: Document Loader
# ---------------------------------------------------------------------------
def doc_loader_node(state: AgentState) -> AgentState:
    logger.info("--- DOCUMENT LOADER NODE ---")
    selected = state.get("selected_files", [])

    if not selected:
        logger.info("No files selected for loading.")
        return {**state, "doc_context": ""}

    combined_parts: List[str] = []
    for file_name in selected:
        file_path = CONTEXT_DIR / file_name
        logger.info("Loading document: %s (exists=%s)", file_name, file_path.exists())
        if not file_path.exists():
            logger.error("File not found on disk: %s", file_path)
            continue

        try:
            suffix = file_path.suffix.lower()
            if suffix == ".pdf":
                text = _read_pdf_impl(str(file_path))
            elif suffix == ".docx":
                text = _read_docx_impl(str(file_path))
            else:
                logger.warning("Unsupported suffix for %s", file_name)
                continue
        except Exception as exc:
            logger.warning("Failed to read %s: %s", file_name, exc, exc_info=True)
            continue

        text_len = len(text)
        logger.info("Document %s extracted %d characters.", file_name, text_len)
        if text_len == 0:
            logger.warning("Document %s produced zero characters! Content will be empty.", file_name)

        combined_parts.append(f"--- Document: {file_name} ---\n{text.strip()}")

    doc_text = "\n\n".join(combined_parts)
    logger.info("Total extracted doc_context length: %d characters from %d file(s).", len(doc_text), len(combined_parts))

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("===== DOC_CONTEXT START =====\n%s\n===== DOC_CONTEXT END =====", doc_text[:2000])

    return {**state, "doc_context": doc_text}


# ---------------------------------------------------------------------------
# Node: Writer
# ---------------------------------------------------------------------------
WRITER_SYSTEM = (
    "You are an expert prompt engineer. Your sole task is to write a single, "
    "self-contained prompt that an LLM can execute. Be detailed, specific, "
    "and clear. Output ONLY the final prompt text—no markdown fences, no preamble."
)


def _build_combined_context(state: AgentState) -> str:
    """Merge user context and document context into a single source block."""
    parts: List[str] = []
    user_ctx = state.get("user_context", "").strip()
    doc_ctx = state.get("doc_context", "").strip()

    if user_ctx:
        parts.append(f"User description:\n{user_ctx}")
    if doc_ctx:
        parts.append(f"Reference document(s):\n{doc_ctx}")

    result = "\n\n".join(parts)
    logger.debug("_build_combined_context produced %d characters.", len(result))
    return result


def writer_node(state: AgentState) -> AgentState:
    logger.info("--- WRITER NODE ---")
    combined_context = _build_combined_context(state)
    previous_prompt = state.get("current_prompt", "")
    iteration = state["iteration"]

    lines: List[str] = []
    if combined_context:
        lines.append(f"Use the following context to craft the prompt:\n{combined_context}\n")
    else:
        lines.append("Write a high-quality general-purpose prompt.\n")

    if iteration > 0 and previous_prompt:
        lines.append(f"Previous prompt (iteration {iteration}):\n{previous_prompt}\n")
        lines.append(
            "Rewrite the prompt to address the weaknesses identified by the critic. "
            "Output ONLY the improved prompt."
        )
    else:
        lines.append("Write the best possible prompt. Output ONLY the prompt text.")

    content = "\n".join(lines)
    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=content),
    ]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("===== WRITER PROMPT START =====\n%s\n===== WRITER PROMPT END =====", content[:2000])

    response = llm.invoke(messages)
    new_prompt = response.content.strip()

    logger.info("Writer produced prompt (%d chars):\n%s\n", len(new_prompt), new_prompt)

    return {
        **state,
        "current_prompt": new_prompt,
        "iteration": iteration + 1,
        "done": False,
    }


# ---------------------------------------------------------------------------
# Node: Critic
# ---------------------------------------------------------------------------
CRITIC_SYSTEM = (
    "You are a rigorous prompt critic. You evaluate prompts on three dimensions. "
    "You MUST use the FULL 1-5 scale. Avoid scoring only 1 or 5; most prompts are imperfect and deserve intermediate scores (2, 3, or 4).\n\n"
    "Scoring rubric for each dimension:\n"
    "  5 = Excellent. Near-flawless across all criteria for this dimension.\n"
    "  4 = Good. Strong with only minor gaps.\n"
    "  3 = Acceptable. Meets basic needs but has noticeable weaknesses.\n"
    "  2 = Weak. Major gaps or ambiguities that would likely cause problems.\n"
    "  1 = Very poor. Fails fundamentally at this dimension.\n\n"
    "Dimensions:\n"
    "1. clarity – Is the prompt unambiguous and easy to understand?\n"
    "   5: crystal clear; 4: mostly clear; 3: understandable but some fuzziness; 2: confusing in places; 1: incomprehensible.\n"
    "2. specificity – Does it include sufficient detail, constraints, format instructions, and examples?\n"
    "   5: exhaustive detail; 4: good detail with minor omissions; 3: adequate but thin; 2: vague, missing important constraints; 1: no actionable detail.\n"
    "3. output_predictability – Can a user reliably predict what the output will look like before running it?\n"
    "   5: output structure is fully specified; 4: mostly predictable; 3: somewhat predictable but could vary; 2: output shape is unclear; 1: no idea what will come out.\n\n"
    "IMPORTANT: Before giving your final scores, think step by step. "
    "Write 2-4 sentences of reasoning about the prompt's strengths and weaknesses. "
    "Consider the prompt text AND any execution results provided. "
    "This reasoning will be discarded — only the JSON matters for the final answer.\n\n"
    "Then output a single JSON object and nothing else:\n"
    '{"clarity": int, "specificity": int, "output_predictability": int}'
)


def _generate_dummy_input(prompt: str, context: str) -> str:
    """Generate a realistic dummy input for the given prompt."""
    system_msg = (
        "You are a data generator. Create a brief, realistic piece of input data "
        "that a user might feed to the given prompt. Be specific and concise (1-4 sentences). "
        "Output ONLY the dummy input — no explanations, no markdown."
    )
    human_msg = (
        f"Prompt that needs test input:\n{prompt}\n\n"
        f"Context:\n{context[:500]}\n\n"
        f"Generate a realistic dummy input for this prompt."
    )
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg),
    ]
    response = llm.invoke(messages)
    return response.content.strip()


def critic_node(state: AgentState) -> AgentState:
    logger.info("--- CRITIC NODE ---")
    prompt = state["current_prompt"]
    combined_context = _build_combined_context(state)
    logs = list(state.get("logs", []))

    # --- EXECUTION TESTING ---
    logger.info("Generating dummy input for execution testing...")
    try:
        dummy_input = _generate_dummy_input(prompt, combined_context)
        logger.info("Dummy input (%d chars): %s", len(dummy_input), dummy_input[:200])
    except Exception as exc:
        logger.warning("Failed to generate dummy input: %s. Using fallback.", exc)
        dummy_input = "This is a sample input for testing purposes."

    logger.info("Running execution test #1 (prompt + dummy input)...")
    try:
        output_1 = run_prompt_test.invoke({"prompt": prompt, "dummy_input": dummy_input})
        logger.info("Execution output #1 (%d chars): %s", len(output_1), output_1[:200])
    except Exception as exc:
        logger.warning("Execution test #1 failed: %s. Skipping execution evidence.", exc)
        output_1 = "[Execution failed]"

    logger.info("Running execution test #2 (same input, checking consistency)...")
    try:
        output_2 = run_prompt_test.invoke({"prompt": prompt, "dummy_input": dummy_input})
        logger.info("Execution output #2 (%d chars): %s", len(output_2), output_2[:200])
    except Exception as exc:
        logger.warning("Execution test #2 failed: %s. Skipping consistency check.", exc)
        output_2 = "[Execution failed]"

    # Compare consistency
    consistency_note = ""
    if output_1.strip() == output_2.strip():
        consistency_note = "The two executions with IDENTICAL input produced EXACTLY the same output (high consistency)."
    elif output_1[:300].strip() == output_2[:300].strip():
        consistency_note = "The two executions with identical input produced SIMILAR outputs with minor variations (moderate consistency)."
    else:
        consistency_note = "The two executions with identical input produced DIFFERENT outputs (low consistency / non-deterministic)."

    execution_evidence = (
        f"--- EXECUTION TEST RESULTS ---\n"
        f"Dummy input used for both runs:\n{dummy_input}\n\n"
        f"Run 1 output:\n{output_1}\n\n"
        f"Run 2 output (same input):\n{output_2}\n\n"
        f"Consistency check: {consistency_note}\n"
        f"--- END EXECUTION TESTS ---"
    )
    logger.info("Execution testing complete.\n%s", execution_evidence[:500])
    # --- END EXECUTION TESTING ---

    clarity_scores: List[int] = []
    specificity_scores: List[int] = []
    predictability_scores: List[int] = []

    for test_num in range(1, NUM_TESTS + 1):
        human_msg = (
            f"Test #{test_num} of {NUM_TESTS}\n"
            f"Context for which the prompt was written:\n{combined_context}\n\n"
            f"Prompt under evaluation:\n{prompt}\n\n"
            f"{execution_evidence}\n\n"
            f"Based on the prompt text AND the actual execution results above, provide your scores. "
            f"First, write 2-4 sentences of reasoning about the prompt's strengths and weaknesses. "
            f"Consider: Did the output match the claimed format? Was it consistent across runs? "
            f"Then output your scores in JSON."
        )
        messages = [
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(content=human_msg),
        ]

        try:
            response = llm.invoke(messages)
            scores_obj = _extract_json(response.content)
            c = int(scores_obj["clarity"])
            s = int(scores_obj["specificity"])
            o = int(scores_obj["output_predictability"])
        except Exception as exc:
            logger.warning("Critic test #%d failed to parse: %s. Defaulting to 1s.", test_num, exc)
            c = s = o = 1

        # Clamp to 1-5
        c, s, o = max(1, min(5, c)), max(1, min(5, s)), max(1, min(5, o))

        clarity_scores.append(c)
        specificity_scores.append(s)
        predictability_scores.append(o)

        log_entry = (
            f"  Test #{test_num}: clarity={c}, specificity={s}, predictability={o}"
        )
        logger.info(log_entry)
        logs.append(log_entry)

    # Aggregate (average)
    aggregated = {
        "clarity": sum(clarity_scores) / len(clarity_scores),
        "specificity": sum(specificity_scores) / len(specificity_scores),
        "output_predictability": sum(predictability_scores) / len(predictability_scores),
    }

    summary = (
        f"AGGREGATE SCORES after {NUM_TESTS} tests -> "
        f"clarity={aggregated['clarity']:.2f}, "
        f"specificity={aggregated['specificity']:.2f}, "
        f"output_predictability={aggregated['output_predictability']:.2f}"
    )
    logger.info(summary)
    logs.append(summary)

    return {
        **state,
        "scores": aggregated,
        "logs": logs,
    }


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------
def should_continue(state: AgentState) -> str:
    scores = state["scores"]
    iteration = state["iteration"]

    if iteration >= MAX_ITERATIONS:
        logger.warning("Max iterations (%d) reached. Ending flow.", MAX_ITERATIONS)
        return "__end__"

    all_above_threshold = all(score >= SCORE_THRESHOLD for score in scores.values())

    if all_above_threshold:
        logger.info("All scores >= %.1f. Prompt accepted!", SCORE_THRESHOLD)
        return "__end__"
    else:
        low_dims = [k for k, v in scores.items() if v < SCORE_THRESHOLD]
        logger.info(
            "Dimensions below threshold (%s): %s. Looping back to writer.",
            SCORE_THRESHOLD,
            ", ".join(low_dims),
        )
        return "writer"


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------
def build_graph() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("file_scanner", file_scanner_node)
    workflow.add_node("router", router_node)
    workflow.add_node("doc_loader", doc_loader_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("file_scanner")
    workflow.add_edge("file_scanner", "router")
    workflow.add_edge("router", "doc_loader")
    workflow.add_edge("doc_loader", "writer")
    workflow.add_edge("writer", "critic")
    workflow.add_conditional_edges(
        "critic",
        should_continue,
        {
            "writer": "writer",
            "__end__": END,
        },
    )

    return workflow.compile()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run(user_context: str) -> AgentState:
    graph = build_graph()

    initial_state: AgentState = {
        "user_context": user_context,
        "available_files": [],
        "selected_files": [],
        "doc_context": "",
        "current_prompt": "",
        "iteration": 0,
        "scores": {},
        "logs": [],
        "done": False,
    }

    logger.info("Starting prompt engineering agent.")
    if user_context:
        logger.info("User context provided: %r", user_context)
    final_state = graph.invoke(initial_state)
    return final_state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LangGraph Prompt Engineering Agent")
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Description of the prompt you want generated (optional if documents provided).",
    )
    args = parser.parse_args()

    if args.context:
        user_context = args.context
    else:
        print("Enter the context / description for the prompt you want (press Enter to skip if using documents only):")
        try:
            user_context = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborting.")
            sys.exit(0)

    # Check for documents in context/ folder
    docs_present = any(
        f.suffix.lower() in (".pdf", ".docx")
        for f in CONTEXT_DIR.iterdir()
        if f.is_file()
    ) if CONTEXT_DIR.exists() else False

    if not user_context and not docs_present:
        print("Error: No user context provided and no documents found in context/ folder.")
        print("Provide at least one source of context.")
        sys.exit(1)

    final = run(user_context)

    print("\n" + "=" * 60)
    print("FINAL PROMPT")
    print("=" * 60)
    print(final["current_prompt"])
    print("=" * 60)
    print("FINAL SCORES:")
    for dim, score in final["scores"].items():
        print(f"  {dim:25s}: {score:.2f}")
    print(f"Iterations: {final['iteration']}")
    if final.get("selected_files"):
        print(f"Documents used: {', '.join(final['selected_files'])}")
    print("=" * 60)
