#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib"]
# ///
"""Leaderboard management and plot generation for JMLE-Bench.

Usage:
    uv run leaderboard.py                             # rebuild README leaderboard + generate plot
    uv run leaderboard.py --rescore                   # re-parse answers, rescore, then rebuild
    uv run leaderboard.py --rescore results/foo.json  # rescore specific file(s) only
    uv run leaderboard.py --plot-only                 # only regenerate the plot
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
DATASET_PATH = Path(__file__).parent / "jmle2026_dataset.json"
README_PATH = Path(__file__).parent / "README.md"


# ---------------------------------------------------------------------------
# Leaderboard table helpers (moved from benchmark.py)
# ---------------------------------------------------------------------------
def _parse_leaderboard(text: str, start_marker: str, end_marker: str) -> tuple[int, int, list[list[str]]]:
    """Parse a markdown table between markers. Returns (start_pos, end_pos, rows).
    Each row is a list of cell strings (excluding header and separator)."""
    s = text.find(start_marker)
    e = text.find(end_marker)
    if s == -1 or e == -1:
        return -1, -1, []

    block = text[s + len(start_marker):e]
    rows = []
    for line in block.strip().splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= {"-"}:
            continue
        cells = [c.strip() for c in line.split("|")[1:-1]]
        # skip header row
        if cells and cells[0] == "Model":
            continue
        if cells:
            rows.append(cells)
    return s + len(start_marker), e, rows


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = "|".join("-" * max(len(h), 4) for h in headers)
    lines = [f"| {' | '.join(headers)} |", f"|{sep}|"]
    for r in rows:
        lines.append(f"| {' | '.join(r)} |")
    return "\n".join(lines) + "\n"


def _score_sort_key(row: list[str], score_col: int) -> float:
    """Parse score column like '485/500 (97.0%)' and return percentage for sorting."""
    try:
        return float(row[score_col].split("(")[1].rstrip("%)")) / 100
    except (IndexError, ValueError):
        return 0.0


def _strip_bold(s: str) -> str:
    """Remove markdown bold markers from a string."""
    if s.startswith("**") and s.endswith("**"):
        return s[2:-2]
    return s


def _fmt_ratio(num: int, denom: int, pct: float) -> str:
    return f"{num}/{denom} ({pct:.1%})"


def update_leaderboard(model: str, image_mode: str, summary: dict) -> None:
    """Update the leaderboard tables in README.md."""
    if not README_PATH.exists():
        return

    text = README_PATH.read_text()
    s = summary

    # Bold model name if both required and general pass thresholds are met
    passing = s.get("required_pass") is True and s.get("general_pass") is True
    display = f"**{model}**" if passing else model

    # All 400 questions table (blind/vision only)
    if image_mode != "skip":
        start, end, rows = _parse_leaderboard(text, "<!-- leaderboard-all-start -->\n", "<!-- leaderboard-all-end -->")
        if start != -1:
            total = s["correct"] + s["incorrect"]
            vision = "✓" if image_mode == "vision" else "-"
            entry = [display, vision, _fmt_ratio(s["score"], s["score_max"], s["score_pct"]),
                     _fmt_ratio(s["correct"], total, s["accuracy"])]
            rows = [r for r in rows if not (_strip_bold(r[0]) == model and r[1] == vision)]
            rows.append(entry)
            rows.sort(key=lambda r: _score_sort_key(r, 2), reverse=True)
            text = text[:start] + _format_table(["Model", "Vision", "Score", "Accuracy"], rows) + text[end:]

    # Text-only table (all modes)
    start, end, rows = _parse_leaderboard(text, "<!-- leaderboard-text-start -->\n", "<!-- leaderboard-text-end -->")
    if start != -1:
        if image_mode == "skip":
            score, score_max, score_pct = s["score"], s["score_max"], s["score_pct"]
            correct, total, acc = s["correct"], s["correct"] + s["incorrect"], s["accuracy"]
        else:
            ti = s["by_image"]["text_only"]
            score, score_max, score_pct = s["text_only_score"], s["text_only_score_max"], s["text_only_score_pct"]
            correct, total, acc = ti["correct"], ti["total"], ti["accuracy"]
        entry = [model, _fmt_ratio(score, score_max, score_pct), _fmt_ratio(correct, total, acc)]
        rows = [r for r in rows if _strip_bold(r[0]) != model]
        rows.append(entry)
        rows.sort(key=lambda r: _score_sort_key(r, 1), reverse=True)
        text = text[:start] + _format_table(["Model", "Score", "Accuracy"], rows) + text[end:]

    README_PATH.write_text(text)
    print(f"  Leaderboard updated in README.md")


def clear_leaderboard() -> None:
    """Clear all leaderboard rows in README.md so rescore can rebuild from scratch."""
    if not README_PATH.exists():
        return
    text = README_PATH.read_text()
    for start_marker, end_marker, headers in [
        ("<!-- leaderboard-all-start -->\n", "<!-- leaderboard-all-end -->",
         ["Model", "Vision", "Score", "Accuracy"]),
        ("<!-- leaderboard-text-start -->\n", "<!-- leaderboard-text-end -->",
         ["Model", "Score", "Accuracy"]),
    ]:
        start, end, _ = _parse_leaderboard(text, start_marker, end_marker)
        if start != -1:
            text = text[:start] + _format_table(headers, []) + text[end:]
    README_PATH.write_text(text)


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------
def plot_accuracy(output_path: Path | None = None) -> Path:
    """Generate a vertical grouped bar chart (all vs text-only) sorted by score.

    Returns the path to the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if output_path is None:
        output_path = RESULTS_DIR / "leaderboard.png"

    # Collect data from result files
    entries: list[dict] = []
    for p in sorted(RESULTS_DIR.glob("*.json")):
        with open(p) as f:
            data = json.load(f)
        meta = data["metadata"]
        summary = data["summary"]
        display_name = meta.get("display_name", meta["model"])
        entries.append({
            "name": display_name,
            "score_pct": summary["score_pct"],
            "text_only_score_pct": summary.get("text_only_score_pct", summary["score_pct"]),
        })

    if not entries:
        print("  No result files found, skipping plot.", file=sys.stderr)
        return output_path

    # Deduplicate: keep highest score_pct per name
    best: dict[str, dict] = {}
    for e in entries:
        if e["name"] not in best or e["score_pct"] > best[e["name"]]["score_pct"]:
            best[e["name"]] = e
    entries = sorted(best.values(), key=lambda e: e["score_pct"], reverse=True)

    names = [e["name"] for e in entries]
    all_scores = [e["score_pct"] * 100 for e in entries]
    text_scores = [e["text_only_score_pct"] * 100 for e in entries]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(entries) * 0.8), 6))
    bars_all = ax.bar(x - width / 2, all_scores, width, label="400 Questions", color="#2196F3")
    bars_text = ax.bar(x + width / 2, text_scores, width, label="302 Questions", color="#78909C")

    # Labels on bars
    for bar in bars_all:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
    for bar in bars_text:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Score (%)")
    ax.set_title("JMLE-Bench Leaderboard")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
    ax.set_ylim(0, max(max(all_scores), max(text_scores)) + 5)
    ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved to {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _rescore_file(result_path: Path, entries: list[dict]) -> None:
    """Re-parse and re-score a single result JSON with the current parser and dataset."""
    from benchmark import parse_answer, is_correct, compute_summary

    entry_map = {e["question_id"]: e for e in entries}

    with open(result_path) as f:
        data = json.load(f)

    results = data["results"]
    changes = []

    for r in results:
        e = entry_map.get(r["question_id"])
        if not e:
            continue

        raw = r.get("raw_response", "")
        predicted, parse_ok = parse_answer(raw, e["question_type"])
        if not parse_ok and r.get("reasoning"):
            predicted, parse_ok = parse_answer(r["reasoning"], e["question_type"])

        old_predicted = r["predicted"]
        old_correct = r["correct"]
        old_parse = r["parse_success"]

        correct = is_correct(predicted, e["answer"]) if parse_ok else False

        if predicted != old_predicted or correct != old_correct or parse_ok != old_parse:
            changes.append({
                "qid": r["question_id"],
                "old": {"predicted": old_predicted, "correct": old_correct, "parse": old_parse},
                "new": {"predicted": predicted, "correct": correct, "parse": parse_ok},
            })

        r["predicted"] = predicted
        r["parse_success"] = parse_ok
        r["gold"] = e["answer"]
        r["correct"] = correct

    result_qids = {r["question_id"] for r in results}
    relevant_entries = [e for e in entries if e["question_id"] in result_qids]

    summary = compute_summary(results, relevant_entries)
    data["summary"] = summary

    with open(result_path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    display_name = data["metadata"].get("display_name", data["metadata"]["model"])
    image_mode = data["metadata"].get("image_mode", "blind")
    update_leaderboard(display_name, image_mode, summary)

    total = summary["correct"] + summary["incorrect"]
    print(f"  {result_path.name}: {summary['score']}/{summary['score_max']} ({summary['score_pct']:.1%}), "
          f"{summary['correct']}/{total} ({summary['accuracy']:.1%})")
    if changes:
        for c in changes:
            mark = "+" if c["new"]["correct"] and not c["old"]["correct"] else "-" if not c["new"]["correct"] and c["old"]["correct"] else "~"
            print(f"    {mark} {c['qid']}: {c['old']['predicted']} -> {c['new']['predicted']} "
                  f"(correct: {c['old']['correct']} -> {c['new']['correct']})")
    else:
        print(f"    (no changes)")


def rescore(paths: list[Path] | None = None) -> None:
    """Re-parse and re-score result JSONs, then rebuild the leaderboard."""
    with open(DATASET_PATH) as f:
        entries = json.load(f)

    rescore_all = paths is None
    if rescore_all:
        paths = sorted(RESULTS_DIR.glob("*.json"))
        clear_leaderboard()

    print(f"Re-scoring {len(paths)} result file(s)...\n")
    for p in paths:
        _rescore_file(p, entries)
        print()

    if rescore_all:
        return  # leaderboard already rebuilt by _rescore_file calls

    # For partial rescore, rebuild full leaderboard to keep it consistent
    rebuild_leaderboard()


def rebuild_leaderboard() -> None:
    """Rebuild README leaderboard from existing result JSONs (no rescoring)."""
    paths = sorted(RESULTS_DIR.glob("*.json"))
    if not paths:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    clear_leaderboard()
    print(f"Rebuilding leaderboard from {len(paths)} result file(s)...\n")

    for p in paths:
        with open(p) as f:
            data = json.load(f)

        summary = data["summary"]
        display_name = data["metadata"].get("display_name", data["metadata"]["model"])
        image_mode = data["metadata"].get("image_mode", "blind")
        update_leaderboard(display_name, image_mode, summary)

        total = summary["correct"] + summary["incorrect"]
        print(f"  {p.name}: {summary['score']}/{summary['score_max']} ({summary['score_pct']:.1%}), "
              f"{summary['correct']}/{total} ({summary['accuracy']:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild JMLE-Bench leaderboard and generate plot")
    parser.add_argument("--rescore", nargs="*", metavar="FILE",
                        help="Re-parse and re-score result JSONs (optionally specify files; default: all)")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate the plot (skip README rebuild)")
    args = parser.parse_args()

    if args.plot_only:
        pass
    elif args.rescore is not None:
        paths = [Path(p) for p in args.rescore] if args.rescore else None
        rescore(paths)
    else:
        rebuild_leaderboard()

    plot_accuracy()


if __name__ == "__main__":
    main()
