#!/usr/bin/env python3
"""
Standalone Prompt Optimizer Agent.

Iteratively rewrites a prompt to maximize accuracy against transcript truth data.
Saves only prompts that meet or exceed the previous best accuracy.

Matching: truth answer is compared to the first token before '::' in the LLM output,
case-insensitive.
"""

import argparse
import asyncio
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

import config

# ---------------------------------------------------------------------------
# Logging
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

PLACEHOLDER = "{{insert transcript here}}"


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------
def get_llm(fallback: bool = False, temperature: float = 0.7):
    """Return LLM instance. Use llama-server fallback if requested."""
    if fallback:
        return ChatOpenAI(
            model=config.LLAMA_SERVER_MODEL,
            base_url=f"{config.LLAMA_SERVER_URL}/v1",
            temperature=temperature,
            api_key="",
        )
    return ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=temperature)


async def probe_ollama() -> bool:
    """Probe Ollama. Return True if we should fall back to llama-server."""
    try:
        llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
        await llm.ainvoke([HumanMessage(content="ok")])
        return False
    except Exception as exc:
        logger.warning("Ollama probe failed (%s). Switching to llama-server fallback.", exc)
        return True


async def llm_invoke_async(llm, messages):
    """Async LLM invoke with fallback to threaded sync call."""
    try:
        return await llm.ainvoke(messages)
    except (AttributeError, NotImplementedError):
        return await asyncio.to_thread(llm.invoke, messages)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_truth_data(truth_folder: Path) -> Dict[str, str]:
    """Load all truth CSVs. Returns dict: filename -> answer."""
    truth: Dict[str, str] = {}
    if not truth_folder.exists():
        logger.warning("Truth folder does not exist: %s", truth_folder)
        return truth

    for csv_file in sorted(truth_folder.glob("*.csv")):
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # skip header
            for row in reader:
                if not row or len(row) < 2:
                    continue
                filename = row[0].strip()
                answer = row[1].strip()
                if filename and answer:
                    truth[filename] = answer
        logger.info("Loaded truth data from %s (%d entries)", csv_file.name, len(truth))
    return truth


def load_transcripts(transcripts_folder: Path) -> Dict[str, List[Dict[str, str]]]:
    """Load all transcript CSVs. Returns dict: filename -> list of entries."""
    transcripts: Dict[str, List[Dict[str, str]]] = {}
    if not transcripts_folder.exists():
        logger.warning("Transcripts folder does not exist: %s", transcripts_folder)
        return transcripts

    for csv_file in sorted(transcripts_folder.glob("*.csv")):
        count = 0
        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("Filename", "").strip()
                if not filename:
                    continue
                speaker = row.get("Party", "unknown").strip()
                text = row.get("Text", "").strip()
                if not text:
                    continue
                transcripts.setdefault(filename, []).append({"speaker": speaker, "text": text})
                count += 1
        logger.info(
            "Loaded transcripts from %s (%d rows, %d unique filenames)",
            csv_file.name,
            count,
            len({k for k, v in transcripts.items() if any(e.get("_source") == csv_file.name for e in v)}),
        )
    return transcripts


# ---------------------------------------------------------------------------
# Transcript formatting
# ---------------------------------------------------------------------------
def format_transcript(entries: List[Dict[str, str]]) -> str:
    """Group consecutive same-speaker lines into dialogue."""
    lines: List[str] = []
    current_speaker = None
    current_parts: List[str] = []

    for entry in entries:
        speaker = entry["speaker"].capitalize()
        text = entry["text"]
        if not text:
            continue
        if speaker == current_speaker:
            current_parts.append(text)
        else:
            if current_speaker and current_parts:
                lines.append(f"{current_speaker}: {' '.join(current_parts)}")
            current_speaker = speaker
            current_parts = [text]

    if current_speaker and current_parts:
        lines.append(f"{current_speaker}: {' '.join(current_parts)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt cleaning
# ---------------------------------------------------------------------------
def clean_prompt(text: str) -> str:
    """Strip markdown fences and extra whitespace."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# Async testing engine
# ---------------------------------------------------------------------------
def extract_answer(output: str) -> str:
    """Extract first token before '::', lowercased."""
    if not output:
        return ""
    parts = output.split("::", 1)
    return parts[0].strip().lower()


async def test_single(
    transcript_text: str,
    prompt_template: str,
    llm,
    semaphore: asyncio.Semaphore,
) -> str:
    """Run one prompt against one transcript. Returns raw LLM output."""
    prompt = prompt_template.replace(PLACEHOLDER, transcript_text)
    async with semaphore:
        try:
            response = await llm_invoke_async(
                llm,
                [
                    SystemMessage(content="You are an LLM executing a prompt. Follow the instructions exactly."),
                    HumanMessage(content=prompt),
                ],
            )
            return response.content.strip()
        except Exception as exc:
            logger.error("LLM error during single test: %s", exc)
            return ""


async def run_tests(
    transcripts: Dict[str, List[Dict[str, str]]],
    truth: Dict[str, str],
    prompt_template: str,
    llm,
    concurrency: int = 20,
) -> Tuple[float, int, int, List[Dict]]:
    """Test prompt against all transcripts with truth data.

    Returns: (accuracy_pct, correct_count, total_count, failure_dicts)
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(filename: str, entries: List[Dict[str, str]]):
        text = format_transcript(entries)
        output = await test_single(text, prompt_template, llm, semaphore)
        expected = truth[filename].lower()
        actual = extract_answer(output)
        is_correct = actual == expected
        failure = None
        if not is_correct:
            excerpt = text[:500] + "..." if len(text) > 500 else text
            failure = {
                "filename": filename,
                "expected": expected,
                "actual": actual,
                "output": output,
                "excerpt": excerpt,
            }
        return filename, is_correct, failure

    tasks = []
    for filename, entries in transcripts.items():
        if filename not in truth:
            continue
        tasks.append(asyncio.create_task(run_one(filename, entries)))

    correct = 0
    total = len(tasks)
    failures: List[Dict] = []

    for i, coro in enumerate(asyncio.as_completed(tasks)):
        _, is_correct, failure = await coro
        if is_correct:
            correct += 1
        elif failure:
            failures.append(failure)

        if (i + 1) % 100 == 0 or (i + 1) == total:
            logger.info("Progress: %d/%d tested", i + 1, total)

    accuracy = (correct / total * 100) if total > 0 else 0.0
    return accuracy, correct, total, failures


# ---------------------------------------------------------------------------
# Prompt saving
# ---------------------------------------------------------------------------
def save_prompt(prompt: str, iteration: int, accuracy: float, prompts_folder: Path) -> Path:
    """Save prompt to file. Returns saved path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"prompt_iter_{iteration}_{accuracy:.1f}pct_{timestamp}.txt"
    path = prompts_folder / filename
    path.write_text(prompt, encoding="utf-8")
    logger.info("Prompt saved to %s", path)
    return path


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------
WRITER_SYSTEM = (
    "You are an expert prompt engineer. Your task is to rewrite a prompt to improve its accuracy on a "
    "transcript classification task. Output ONLY the rewritten prompt text—no markdown fences, no preamble. "
    f"The prompt MUST include `{PLACEHOLDER}` where the transcript text will be substituted. "
    "Do NOT include any transcript text in your output besides exactly that placeholder.\n\n"
    "CRITICAL GUARDRAILS:\n"
    "- Do NOT copy, quote, or embed exact examples from transcripts or truth data into the prompt.\n"
    "- Do NOT include caller names, agent names, dates, account numbers, or other real data.\n"
    "- If you need examples, write them as clearly fictional and generalized (e.g., 'Agent: How can I help you today?').\n"
    "- Keep instructions abstract and generalizable to unseen transcripts."
)


async def writer_node(
    best_prompt: str,
    best_accuracy: float,
    failures: List[Dict],
    critic_feedback: str,
    user_context: str,
    llm,
) -> str:
    """Generate improved prompt based on failures and critic feedback."""
    failure_text = ""
    for i, f in enumerate(failures[:10], 1):
        failure_text += (
            f"\n{i}. Expected answer: {f['expected']}\n"
            f"   Model answer:   {f['actual']}\n"
        )

    human = (
        f"Current best prompt (accuracy: {best_accuracy:.1f}%):\n"
        f"{best_prompt}\n\n"
        f"Failure summary (first 10 of {len(failures)}):\n"
        f"{failure_text}\n\n"
        f"Critic feedback:\n"
        f"{critic_feedback}\n\n"
        f"Rewrite the prompt to address the identified weaknesses and improve accuracy. "
        f"Be specific, clear, and structured. Output ONLY the prompt text. "
        f"Do NOT include exact transcript examples, caller names, or real data from the failures above."
    )

    if user_context:
        human = f"Original task context:\n{user_context}\n\n" + human

    response = await llm_invoke_async(llm, [SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=human)])
    new_prompt = clean_prompt(response.content.strip())

    if PLACEHOLDER not in new_prompt:
        new_prompt += f"\n\nTRANSCRIPT:\n{PLACEHOLDER}"

    return new_prompt


# ---------------------------------------------------------------------------
# Initial prompt generator
# ---------------------------------------------------------------------------
INITIAL_WRITER_SYSTEM = (
    "You are an expert prompt engineer. Write a self-contained prompt for a transcript classification task. "
    "Output ONLY the prompt text—no markdown fences, no preamble. "
    f"The prompt MUST include `{PLACEHOLDER}` where the transcript text will be substituted."
)


async def generate_initial_prompt(user_context: str, llm) -> str:
    """Generate a prompt from scratch given user context."""
    human = f"Write a prompt for the following task:\n{user_context}\n\nOutput ONLY the prompt text."
    response = await llm_invoke_async(llm, [SystemMessage(content=INITIAL_WRITER_SYSTEM), HumanMessage(content=human)])
    prompt = clean_prompt(response.content.strip())
    if PLACEHOLDER not in prompt:
        prompt += f"\n\nTRANSCRIPT:\n{PLACEHOLDER}"
    return prompt


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
CRITIC_SYSTEM = (
    "You are a rigorous prompt critic analyzing why a prompt failed on specific transcript classification examples. "
    "Given the current prompt and failure cases, identify:\n"
    "1. Concrete patterns of failure (e.g., misinterpreting certain phrases, unclear instructions, ambiguous answer options)\n"
    "2. Specific, actionable suggestions for how to rewrite the prompt to fix these issues\n\n"
    "You MAY cite specific phrases, speaker lines, or caller names from the transcripts as evidence. "
    "Be concise but thorough. Output your analysis as plain text bullet points."
)


async def critic_node(best_prompt: str, failures: List[Dict], llm) -> str:
    """Analyze failures and return concrete improvement suggestions."""
    failure_text = ""
    for i, f in enumerate(failures[:15], 1):
        excerpt = f["excerpt"][:300] + "..." if len(f["excerpt"]) > 300 else f["excerpt"]
        failure_text += (
            f"\n{i}. Expected: {f['expected']}, Actual: {f['actual']}\n"
            f"   LLM output: {f['output']}\n"
            f"   Excerpt: {excerpt}\n"
        )

    human = (
        f"Prompt being evaluated:\n"
        f"{best_prompt}\n\n"
        f"Failure cases ({len(failures)} total, showing first 15):\n"
        f"{failure_text}\n\n"
        f"Analyze the failures. What patterns do you see? What specific changes to the prompt would fix these errors?"
    )

    response = await llm_invoke_async(llm, [SystemMessage(content=CRITIC_SYSTEM), HumanMessage(content=human)])
    return response.content.strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Prompt Optimizer Agent")
    parser.add_argument(
        "--target-accuracy",
        default="none",
        help="Target accuracy %% (e.g., 95.0) or 'none' to ignore",
    )
    parser.add_argument(
        "--max-iterations",
        default="none",
        help="Max optimization iterations (int) or 'none' to ignore",
    )
    parser.add_argument(
        "--use-existing-prompt",
        action="store_true",
        help="Seed initial prompt from prompts/prompt.txt",
    )
    parser.add_argument(
        "--context",
        default="",
        help="User description / context for the prompt task (required if not using existing prompt)",
    )
    parser.add_argument(
        "--prompts-folder",
        default="prompts",
        help="Folder to save prompt iterations",
    )
    parser.add_argument(
        "--transcripts-folder",
        default="transcripts",
        help="Folder containing transcript CSVs",
    )
    parser.add_argument(
        "--truth-folder",
        default="truth data",
        help="Folder containing truth CSVs",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=config.DEFAULT_CONCURRENCY,
        help="Max concurrent LLM test calls",
    )
    parser.add_argument(
        "--max-transcripts",
        default="none",
        help="Test only the first N transcripts (e.g., 100) or 'none' for all",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    args = parse_args()

    # Parse thresholds
    target_accuracy: Optional[float] = None
    if args.target_accuracy.lower() != "none":
        target_accuracy = float(args.target_accuracy)

    max_iterations: Optional[int] = None
    if args.max_iterations.lower() != "none":
        max_iterations = int(args.max_iterations)

    if target_accuracy is None and max_iterations is None:
        logger.warning("Both target_accuracy and max_iterations are 'none'. Defaulting max_iterations to 20.")
        max_iterations = 20

    # Resolve paths
    prompts_folder = BASE_DIR / args.prompts_folder
    transcripts_folder = BASE_DIR / args.transcripts_folder
    truth_folder = BASE_DIR / args.truth_folder
    prompts_folder.mkdir(parents=True, exist_ok=True)

    # Load data
    truth = load_truth_data(truth_folder)
    transcripts = load_transcripts(transcripts_folder)

    logger.info("Loaded %d truth entries and %d transcript filenames", len(truth), len(transcripts))

    # Filter to testable transcripts
    testable = {k: v for k, v in transcripts.items() if k in truth}
    missing_truth = [k for k in transcripts if k not in truth]
    if missing_truth:
        logger.warning("%d transcripts have no matching truth data and will be skipped", len(missing_truth))

    if not testable:
        logger.error("No matching transcripts and truth data found! Exiting.")
        sys.exit(1)

    # Apply max-transcripts limit (preserves sorted filename insertion order)
    max_transcripts: Optional[int] = None
    total_testable = len(testable)
    if args.max_transcripts.lower() != "none":
        max_transcripts = int(args.max_transcripts)
        if max_transcripts < len(testable):
            testable = dict(list(testable.items())[:max_transcripts])
            logger.info("Limited testing to first %d transcripts (of %d testable available)", max_transcripts, total_testable)

    logger.info("Testable transcripts: %d", len(testable))

    # LLM setup
    use_fallback = await probe_ollama()
    test_llm = get_llm(fallback=use_fallback, temperature=0.0)
    writer_llm = get_llm(fallback=use_fallback, temperature=0.7)
    critic_llm = get_llm(fallback=use_fallback, temperature=0.5)

    # Seed initial prompt
    if args.use_existing_prompt:
        prompt_path = prompts_folder / "prompt.txt"
        if not prompt_path.exists():
            logger.error("Existing prompt not found at %s", prompt_path)
            sys.exit(1)
        current_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if PLACEHOLDER not in current_prompt:
            logger.warning("Loaded prompt missing placeholder '%s'. Appending it.", PLACEHOLDER)
            current_prompt += f"\n\nTRANSCRIPT:\n{PLACEHOLDER}"
        logger.info("Loaded existing prompt from %s (%d chars)", prompt_path, len(current_prompt))
    else:
        if not args.context:
            logger.error("No --context provided and --use-existing-prompt not set. Provide one.")
            sys.exit(1)
        current_prompt = await generate_initial_prompt(args.context, writer_llm)
        logger.info("Generated initial prompt from scratch (%d chars)", len(current_prompt))

    # Optimization loop state
    best_prompt = current_prompt
    best_accuracy = 0.0
    best_path: Optional[Path] = None
    tests_run = 0

    logger.info("=" * 60)
    logger.info("Starting optimization loop")
    logger.info("Target accuracy: %s | Max tests: %s", target_accuracy, max_iterations)
    logger.info("=" * 60)

    while True:
        tests_run += 1
        logger.info("--- Test %d ---", tests_run)

        # Test current prompt
        accuracy, correct, total, failures = await run_tests(
            testable, truth, current_prompt, test_llm, concurrency=args.concurrency
        )

        logger.info("Accuracy: %.2f%% (%d/%d) | Failures: %d", accuracy, correct, total, len(failures))

        # Save if improving (or equal)
        if accuracy >= best_accuracy:
            best_prompt = current_prompt
            best_accuracy = accuracy
            best_path = save_prompt(current_prompt, tests_run, accuracy, prompts_folder)
            logger.info("New best prompt saved (%.2f%%)", best_accuracy)
        else:
            logger.info(
                "Did not improve (current: %.2f%%, best: %.2f%%). Using best prompt for next rewrite.",
                accuracy,
                best_accuracy,
            )

        # Stop conditions
        if target_accuracy is not None and best_accuracy >= target_accuracy:
            logger.info("Target accuracy %.2f%% reached!", target_accuracy)
            break

        if max_iterations is not None and tests_run >= max_iterations:
            logger.info("Max tests (%d) reached.", max_iterations)
            break

        # Critic + Writer using BEST prompt
        logger.info("Running critic on %d failures...", len(failures))
        critic_feedback = await critic_node(best_prompt, failures, critic_llm)
        logger.info("Critic feedback:\n%s", critic_feedback[:800])

        logger.info("Running writer...")
        current_prompt = await writer_node(
            best_prompt,
            best_accuracy,
            failures,
            critic_feedback,
            args.context,
            writer_llm,
        )
        logger.info("Writer produced new prompt (%d chars)", len(current_prompt))

    # Final report
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Best accuracy:     {best_accuracy:.2f}%")
    print(f"Tests run:         {tests_run}")
    if best_path:
        print(f"Best prompt file:  {best_path}")
    print("\n" + "-" * 70)
    print("BEST PROMPT:")
    print("-" * 70)
    print(best_prompt)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
