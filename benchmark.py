#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai"]
# ///
"""
Usage:
    # OpenAI
    uv run benchmark.py --model gpt-5.2 --api-key $OPENAI_API_KEY

    # OpenRouter (with reasoning + vision)
    uv run benchmark.py --model openai/gpt-5.2 \
        --base-url https://openrouter.ai/api/v1 \
        --api-key $OPENROUTER_API_KEY \
        --image-mode vision \
        --extra-body '{"reasoning": {"enabled": true}}'

    # vLLM
    uv run benchmark.py --model my-model \
        --base-url http://localhost:8000/v1 --api-key dummy \
        --parallel 32
"""

import argparse
import base64
import json
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from leaderboard import update_leaderboard

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = Path(__file__).parent / "jmle2026_dataset.json"
IMAGES_DIR = Path(__file__).parent / "images"

SYSTEM_PROMPT_CHOICE = """\
あなたは医師国家試験を解く医学の専門家です。
問題を読み、正解を{n}つ選んでください。

最終回答は必ず以下の形式で出力してください:
【回答】{example}"""

SYSTEM_PROMPT_CALC = """\
あなたは医師国家試験を解く医学の専門家です。
問題を読み、数値で回答してください。

最終回答は必ず以下の形式で出力してください:
【回答】3.14"""


def get_system_prompt(entry: dict) -> str:
    if entry["question_type"] == "calculation":
        return SYSTEM_PROMPT_CALC
    n = entry["num_choices_to_select"]
    example = ",".join(list("ace")[:n])
    return SYSTEM_PROMPT_CHOICE.format(n=n, example=example)

ANSWER_PATTERN = re.compile(r"【回答】\s*(.+)")

REQUIRED_BLOCKS = {"B", "E"}

# Passing thresholds (reference: 119th exam)
REQUIRED_PASS_THRESHOLD = 160   # out of 200
GENERAL_PASS_THRESHOLD = 221    # out of 300


def question_points(entry: dict) -> int:
    """Return point value for a question.

    Blocks B and E: questions 1-25 are worth 1 point, questions 26-50 are worth 3 points.
    All other blocks: 1 point per question.
    """
    block = entry["block"]
    if block not in REQUIRED_BLOCKS:
        return 1
    num = int(entry["question_id"].split("-")[1])
    return 1 if num <= 25 else 3


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------
def build_user_content(entry: dict, image_mode: str) -> list[dict]:
    """Build the user message content array for a single question."""
    # Text part
    parts: list[str] = []
    if entry.get("serial_group"):
        parts.append(entry["serial_group"]["context_text"])
        parts.append("")  # blank line separator
    parts.append(entry["question_text"])
    text = "\n".join(parts)

    content: list[dict] = [{"type": "text", "text": text}]

    # Image parts
    if image_mode == "vision" and entry.get("clinical_images"):
        for img_name in entry["clinical_images"]:
            img_path = IMAGES_DIR / img_name
            if not img_path.exists():
                continue
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )

    return content


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------
def parse_answer(raw: str, question_type: str) -> tuple[list[str], bool]:
    """Extract the answer from the LLM response.

    Returns (parsed_answers, parse_success).
    """
    m = ANSWER_PATTERN.search(raw)
    if not m:
        return [], False

    answer_str = m.group(1).strip()

    if question_type == "calculation":
        # Return the numeric string as-is
        return [answer_str], True

    # Strip markdown formatting, then split on comma/space and keep only [a-e]
    cleaned = re.sub(r"[*_`#]", "", answer_str).strip()
    tokens = re.split(r"[,、\s]+", cleaned)
    answers = sorted(set(t.lower() for t in tokens if re.fullmatch(r"[a-eA-E]", t)))
    return answers, bool(answers)


def is_correct(predicted: list[str], gold: list[str]) -> bool:
    return sorted(predicted) == sorted(gold)


# ---------------------------------------------------------------------------
# Single-question worker
# ---------------------------------------------------------------------------
def process_question(
    client: OpenAI,
    model: str,
    entry: dict,
    image_mode: str,
    extra_body: dict | None = None,
) -> dict:
    """Call the LLM for one question and return the result dict."""
    qid = entry["question_id"]
    try:
        user_content = build_user_content(entry, image_mode)
        messages = [
            {"role": "system", "content": get_system_prompt(entry)},
            {"role": "user", "content": user_content},
        ]

        kwargs = dict(model=model, messages=messages)
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        raw = msg.content if msg.content is not None else ""
        reasoning_content = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

        predicted, parse_ok = parse_answer(raw, entry["question_type"])
        if not parse_ok and reasoning_content:
            predicted, parse_ok = parse_answer(reasoning_content, entry["question_type"])
        correct = is_correct(predicted, entry["answer"]) if parse_ok else False

        result = {
            "question_id": qid,
            "predicted": predicted,
            "gold": entry["answer"],
            "correct": correct,
            "raw_response": raw,
            "parse_success": parse_ok,
            "error": None,
        }
        if reasoning_content is not None:
            result["reasoning"] = reasoning_content
        return result

    except Exception as e:
        traceback.print_exc()
        return {
            "question_id": qid,
            "predicted": [],
            "gold": entry["answer"],
            "correct": False,
            "raw_response": "",
            "parse_success": False,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def _tally() -> dict:
    return {"correct": 0, "total": 0}


def _finalize_tally(groups: dict) -> None:
    for v in groups.values():
        v["accuracy"] = round(v["correct"] / v["total"], 4) if v["total"] else 0


def compute_summary(results: list[dict], entries: list[dict]) -> dict:
    entry_map = {e["question_id"]: e for e in entries}

    correct = parse_failures = errors = 0
    score = score_max = text_only_score = text_only_score_max = 0
    required_score = required_score_max = 0
    general_score = general_score_max = 0
    by_block: dict[str, dict] = defaultdict(_tally)
    by_type: dict[str, dict] = defaultdict(_tally)
    by_image: dict[str, dict] = defaultdict(_tally)

    for r in results:
        e = entry_map[r["question_id"]]
        pts = question_points(e)
        img_key = "with_image" if e["requires_image"] else "text_only"
        ok = r["correct"]

        correct += ok
        parse_failures += not r["parse_success"]
        errors += bool(r["error"])
        score_max += pts
        if ok:
            score += pts
        if not e["requires_image"]:
            text_only_score_max += pts
            if ok:
                text_only_score += pts

        if e["block"] in REQUIRED_BLOCKS:
            required_score_max += pts
            if ok:
                required_score += pts
        else:
            general_score_max += pts
            if ok:
                general_score += pts

        for group, key in [(by_block, e["block"]), (by_type, e["question_type"]), (by_image, img_key)]:
            group[key]["total"] += 1
            if ok:
                group[key]["correct"] += 1

    for groups in (by_block, by_type, by_image):
        _finalize_tally(groups)

    total = len(results)
    return {
        "accuracy": round(correct / total, 4) if total else 0,
        "correct": correct,
        "incorrect": total - correct,
        "score": score,
        "score_max": score_max,
        "score_pct": round(score / score_max, 4) if score_max else 0,
        "text_only_score": text_only_score,
        "text_only_score_max": text_only_score_max,
        "text_only_score_pct": round(text_only_score / text_only_score_max, 4) if text_only_score_max else 0,
        "required_score": required_score,
        "required_score_max": required_score_max,
        "required_score_pct": round(required_score / required_score_max, 4) if required_score_max else 0,
        "general_score": general_score,
        "general_score_max": general_score_max,
        "general_score_pct": round(general_score / general_score_max, 4) if general_score_max else 0,
        "required_pass": required_score >= REQUIRED_PASS_THRESHOLD if required_score_max == 200 else None,
        "general_pass": general_score >= GENERAL_PASS_THRESHOLD if general_score_max == 300 else None,
        "parse_failures": parse_failures,
        "errors": errors,
        "by_block": dict(sorted(by_block.items())),
        "by_type": dict(by_type),
        "by_image": dict(by_image),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="JMLE2026-Bench: Japanese Medical Licensing Examination LLM benchmark")
    p.add_argument("--model", required=True, help="Model name (e.g. gpt-5.2)")
    p.add_argument("--base-url", default="https://api.openai.com/v1", help="API base URL")
    p.add_argument("--api-key", required=True, help="API key")
    p.add_argument("--parallel", type=int, default=8, help="Max parallel workers")
    p.add_argument("--timeout", type=int, default=300, help="API timeout in seconds")
    p.add_argument("--image-mode", choices=["skip", "blind", "vision"], default="blind",
                   help="skip: exclude image questions, blind: answer without images, vision: send images (default: blind)")
    p.add_argument("--extra-body", default=None, help="JSON string passed as extra_body to the API (e.g. '{\"reasoning\": {\"enabled\": true}}')")
    p.add_argument("--resume", type=Path, default=None, help="Resume from a previous result JSON (retry errors only)")
    p.add_argument("--out", type=Path, default=None, help="Output JSON path")
    p.add_argument("--blocks", default=None, help="Comma-separated block filter (e.g. A,B)")
    p.add_argument("--display-name", default=None, help="Display name for leaderboard (default: same as --model)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()

    extra_body = json.loads(args.extra_body) if args.extra_body else None

    # Load dataset
    with open(DATASET_PATH) as f:
        dataset: list[dict] = json.load(f)

    # Filter by blocks
    if args.blocks:
        allowed = set(args.blocks.upper().split(","))
        dataset = [e for e in dataset if e["block"] in allowed]

    # Filter image questions in skip mode
    skipped_image = 0
    if args.image_mode == "skip":
        original = len(dataset)
        dataset = [e for e in dataset if not e["requires_image"]]
        skipped_image = original - len(dataset)

    # Resume: load previous results and filter to retry errors only
    prev_results: dict[str, dict] = {}
    if args.resume:
        with open(args.resume) as f:
            prev_data = json.load(f)
        for r in prev_data["results"]:
            if r["error"] is None and r["parse_success"]:
                prev_results[r["question_id"]] = r

    all_entries = dataset
    total_all = len(all_entries)
    dataset = [e for e in dataset if e["question_id"] not in prev_results]

    if total_all == 0:
        print("No questions to process.", file=sys.stderr)
        sys.exit(1)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"  model:       {args.model}")
    print(f"  base_url:    {args.base_url}")
    print(f"  image_mode:  {args.image_mode}")
    if extra_body:
        print(f"  extra_body:  {json.dumps(extra_body, ensure_ascii=False)}")
    print(f"  questions:   {total_all}")
    if skipped_image:
        print(f"  skipped:     {skipped_image} (image questions)")
    if prev_results:
        print(f"  resumed:     {len(prev_results)} cached, {len(dataset)} to retry")
    print(f"  parallel:    {args.parallel}")

    # Run remaining questions
    new_results: list[dict] = []
    if dataset:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
        total_run = len(dataset)
        max_workers = min(args.parallel, total_run)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for entry in dataset:
                future = executor.submit(process_question, client, args.model, entry, args.image_mode, extra_body)
                futures[future] = entry["question_id"]

            done = 0
            for future in as_completed(futures):
                result = future.result()
                new_results.append(result)
                done += 1
                mark = "o" if result["correct"] else "x"
                if result["error"]:
                    mark = "!"
                elif not result["parse_success"]:
                    mark = "?"
                print(f"  {mark} {result['question_id']}")
                if done % 10 == 0 or done == total_run:
                    correct_so_far = sum(1 for r in new_results if r["correct"])
                    print(f"  [{done}/{total_run}] {correct_so_far}/{done} correct")

    # Merge: previous successes + new results
    result_map = {r["question_id"]: r for r in prev_results.values()}
    for r in new_results:
        result_map[r["question_id"]] = r

    # Re-score all results against current dataset (gold answers may have changed)
    entry_map = {e["question_id"]: e for e in all_entries}
    for r in result_map.values():
        e = entry_map[r["question_id"]]
        if r["parse_success"]:
            r["gold"] = e["answer"]
            r["correct"] = is_correct(r["predicted"], e["answer"])

    results = sorted(result_map.values(), key=lambda r: r["question_id"])

    # Compute summary over all entries
    summary = compute_summary(results, all_entries)

    # Output
    display_name = args.display_name if args.display_name is not None else args.model
    metadata = {
        "model": args.model,
        "base_url": args.base_url,
        "timestamp": timestamp,
        "image_mode": args.image_mode,
        "extra_body": extra_body,
        "total_questions": total_all + skipped_image,
        "attempted": total_all,
        "skipped_image": skipped_image,
    }
    if args.display_name:
        metadata["display_name"] = args.display_name
    output = {
        "metadata": metadata,
        "summary": summary,
        "results": results,
    }

    # Determine output path (resume defaults to overwriting the same file)
    out_path = args.out if args.out is not None else args.resume
    if out_path is None:
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        safe_model = args.model.replace("/", "_")
        out_path = results_dir / f"{safe_model}_{timestamp}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Model:     {args.model}")
    print(f"  Score:     {summary['score']}/{summary['score_max']} ({summary['score_pct']:.1%})")
    print(f"    Required (B+E):      {summary['required_score']}/{summary['required_score_max']} ({summary['required_score_pct']:.1%})")
    print(f"    General  (A+C+D+F):  {summary['general_score']}/{summary['general_score_max']} ({summary['general_score_pct']:.1%})")
    print(f"  Accuracy:  {summary['correct']}/{total_all} ({summary['accuracy']:.1%})")
    print(f"  Parse failures: {summary['parse_failures']}")
    print(f"  Errors:    {summary['errors']}")
    print()
    print(f"  By block:")
    for block, s in summary["by_block"].items():
        print(f"    {block}: {s['correct']}/{s['total']} ({s['accuracy']:.1%})")
    print()
    print(f"  By type:")
    for qtype, s in summary["by_type"].items():
        print(f"    {qtype}: {s['correct']}/{s['total']} ({s['accuracy']:.1%})")
    print()
    print(f"  By image:")
    for key, s in summary["by_image"].items():
        if s["total"] > 0:
            print(f"    {key}: {s['correct']}/{s['total']} ({s['accuracy']:.1%})")
    print(f"{'='*60}")
    print(f"  Results saved to: {out_path}")

    # Update leaderboard
    update_leaderboard(display_name, args.image_mode, summary)



if __name__ == "__main__":
    main()
