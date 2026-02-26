#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["streamlit>=1.45", "pandas"]
# ///
"""JMLE2026-Bench Results Viewer — interactive Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATASET_PATH = BASE_DIR / "jmle2026_dataset.json"
RESULTS_DIR = BASE_DIR / "results"
IMAGES_DIR = BASE_DIR / "images"


def _discover_result_files() -> list[str]:
    """Discover result JSON files from results/ directory."""
    return sorted(p.name for p in RESULTS_DIR.glob("*.json"))


def _points(block: str, number: int) -> int:
    """Return point value for a question. B/E No.26-50 are 3pts, others 1pt."""
    if block in ("B", "E") and number >= 26:
        return 3
    return 1


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_dataset() -> list[dict]:
    with open(DATASET_PATH) as f:
        return json.load(f)


@st.cache_data
def load_result(filename: str) -> dict:
    with open(RESULTS_DIR / filename) as f:
        return json.load(f)


@st.cache_data
def load_all_summaries() -> pd.DataFrame:
    """Load metadata + summary from every result file into a DataFrame."""
    rows = []
    for fname in _discover_result_files():
        path = RESULTS_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        meta = data["metadata"]
        s = data["summary"]
        display_name = meta.get("display_name", meta["model"])
        rows.append(
            {
                "file": fname,
                "model": display_name,
                "score": s["score"],
                "score_max": s["score_max"],
                "score_pct": s["score_pct"],
                "accuracy": s["accuracy"],
                "correct": s["correct"],
                "incorrect": s["incorrect"],
                "text_only_score_pct": s.get("text_only_score_pct", s["score_pct"]),
                "required_pass": s.get("required_pass", False),
                "general_pass": s.get("general_pass", False),
                "pass": s.get("required_pass", False) and s.get("general_pass", False),
            }
        )
    df = pd.DataFrame(rows).sort_values("score_pct", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank
    return df


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="JMLE2026-Bench Viewer",
    page_icon=":material/biotech:",
    layout="wide",
)

st.title("JMLE2026-Bench Results Viewer")

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
view = st.segmented_control(
    "View",
    [
        ":material/leaderboard: Leaderboard",
        ":material/fact_check: Per Model",
        ":material/compare: Per Question",
    ],
    default=":material/leaderboard: Leaderboard",
    label_visibility="collapsed",
)

dataset = load_dataset()
question_map = {q["question_id"]: q for q in dataset}
leaderboard_df = load_all_summaries()

# Model display name -> filename mapping
model_file_map: dict[str, str] = {}
for _, row in leaderboard_df.iterrows():
    model_file_map[row["model"]] = row["file"]

# ---------------------------------------------------------------------------
# View: Leaderboard
# ---------------------------------------------------------------------------
if view == ":material/leaderboard: Leaderboard":
    st.header("Leaderboard", divider=True)

    # KPI row
    top = leaderboard_df.iloc[0]
    passing = leaderboard_df["pass"].sum()
    with st.container(horizontal=True):
        st.metric("Top Score", f"{top['score_pct']:.1%}", help=top["model"], border=True)
        st.metric("Passing", f"{passing}/{len(leaderboard_df)} models", border=True)

    # Leaderboard table
    st.caption("Score is weighted: Block B/E No.26-50 (required) = 3 pts, others = 1 pt. Max 500.")
    display_df = leaderboard_df.copy()
    display_df["score_label"] = display_df.apply(
        lambda r: f"{r['score']}/{r['score_max']}", axis=1
    )
    display_df["correct_label"] = display_df.apply(
        lambda r: f"{r['correct']}/{r['correct'] + r['incorrect']}", axis=1
    )
    st.dataframe(
        display_df[["model", "score_label", "score_pct", "correct_label", "pass"]],
        column_config={
            "model": st.column_config.TextColumn("Model", width="large"),
            "score_label": st.column_config.TextColumn("Score", width="small"),
            "score_pct": st.column_config.ProgressColumn(
                "Score %",
                min_value=0,
                max_value=1,
                format="%.1%%",
            ),
            "correct_label": st.column_config.TextColumn("Correct", width="small"),
            "pass": st.column_config.CheckboxColumn("Pass"),
        },
        hide_index=False,
        width="stretch",
    )

    # Score charts (full width, stacked vertically) with tooltips
    _axis_cfg = alt.Axis(labelAngle=-45, labelLimit=0, labelOverlap=False)

    st.subheader("All 400 Questions")
    all_df = leaderboard_df[["model", "score_pct"]].copy()
    all_df["Score (%)"] = (all_df["score_pct"] * 100).round(1)
    st.altair_chart(
        alt.Chart(all_df).mark_bar().encode(
            x=alt.X("model:N", sort="-y", title=None, axis=_axis_cfg),
            y=alt.Y("Score (%):Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("Score (%):Q", title="Score %")],
        ).properties(height=400),
        width="stretch",
    )

    st.subheader("Text-Only 302 Questions")
    text_df = leaderboard_df[["model", "text_only_score_pct"]].copy()
    text_df["Score (%)"] = (text_df["text_only_score_pct"] * 100).round(1)
    st.altair_chart(
        alt.Chart(text_df).mark_bar().encode(
            x=alt.X("model:N", sort="-y", title=None, axis=_axis_cfg),
            y=alt.Y("Score (%):Q", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("model:N", title="Model"), alt.Tooltip("Score (%):Q", title="Score %")],
        ).properties(height=400),
        width="stretch",
    )


# ---------------------------------------------------------------------------
# View: Question Browser
# ---------------------------------------------------------------------------
elif view == ":material/fact_check: Per Model":
    # Sidebar: model selection & filters
    with st.sidebar:
        st.header("Filters", divider=True)
        selected_model = st.selectbox(
            "Model",
            options=leaderboard_df["model"].tolist(),
            index=0,
        )
        filter_block = st.multiselect(
            "Block",
            options=["A", "B", "C", "D", "E", "F"],
            default=["A", "B", "C", "D", "E", "F"],
        )
        filter_correct = st.segmented_control(
            "Result",
            ["All", "Correct", "Incorrect"],
            default="All",
        )
        filter_image = st.segmented_control(
            "Image",
            ["All", "With Image", "Text Only"],
            default="All",
        )
        filter_type = st.segmented_control(
            "Question Type",
            ["All", "Multiple Choice", "Calculation"],
            default="All",
        )

    # Load selected model's results
    model_file = model_file_map[selected_model]
    result_data = load_result(model_file)
    results_list = result_data["results"]
    result_map = {r["question_id"]: r for r in results_list}

    st.header(f"Questions — {selected_model}", divider=True)

    # Summary metrics for this model
    s = result_data["summary"]
    with st.container(horizontal=True):
        st.metric("Score", f"{s['score']}/{s['score_max']} ({s['score_pct']:.1%})", border=True)
        st.metric("Accuracy", f"{s['correct']}/{s['correct'] + s['incorrect']} ({s['accuracy']:.1%})", border=True)
        req_label = "Pass" if s.get("required_pass") else "Fail"
        gen_label = "Pass" if s.get("general_pass") else "Fail"
        st.metric("Required", f"{req_label} ({s.get('required_score_pct', 0):.1%})", border=True)
        st.metric("General", f"{gen_label} ({s.get('general_score_pct', 0):.1%})", border=True)

    # Block-level accuracy breakdown
    by_block = s.get("by_block", {})
    if by_block:
        block_rows = []
        for block, info in sorted(by_block.items()):
            note = "No.26-50 are 3 pts (required)" if block in ("B", "E") else ""
            block_rows.append({"Block": block, "Correct": info["correct"], "Total": info["total"], "Accuracy": info["accuracy"], "Note": note})
        block_df = pd.DataFrame(block_rows)
        with st.expander("Block-level accuracy", expanded=False):
            st.dataframe(
                block_df,
                column_config={
                    "Block": st.column_config.TextColumn("Block", width="small"),
                    "Correct": st.column_config.NumberColumn("Correct", format="%d"),
                    "Total": st.column_config.NumberColumn("Total", format="%d"),
                    "Accuracy": st.column_config.ProgressColumn("Accuracy", min_value=0, max_value=1, format="%.1%%"),
                    "Note": st.column_config.TextColumn("Note", width="medium"),
                },
                hide_index=True,
                width="stretch",
            )

    # Build merged table
    merged_rows = []
    for q in dataset:
        qid = q["question_id"]
        r = result_map.get(qid)
        if r is None:
            continue
        merged_rows.append(
            {
                "question_id": qid,
                "block": q["block"],
                "number": q["number"],
                "pts": _points(q["block"], q["number"]),
                "type": q["question_type"],
                "has_image": bool(q.get("clinical_images")),
                "correct": r["correct"],
                "predicted": ", ".join(r["predicted"]) if r["predicted"] else "—",
                "gold": ", ".join(r["gold"]),
            }
        )
    merged_df = pd.DataFrame(merged_rows)

    # Apply filters
    if filter_block:
        merged_df = merged_df[merged_df["block"].isin(filter_block)]
    if filter_correct == "Correct":
        merged_df = merged_df[merged_df["correct"]]
    elif filter_correct == "Incorrect":
        merged_df = merged_df[~merged_df["correct"]]
    if filter_image == "With Image":
        merged_df = merged_df[merged_df["has_image"]]
    elif filter_image == "Text Only":
        merged_df = merged_df[~merged_df["has_image"]]
    if filter_type == "Multiple Choice":
        merged_df = merged_df[merged_df["type"] == "multiple_choice"]
    elif filter_type == "Calculation":
        merged_df = merged_df[merged_df["type"] == "calculation"]

    # Filtered metrics
    n_filtered = len(merged_df)
    n_correct = merged_df["correct"].sum()
    f_score = merged_df.loc[merged_df["correct"], "pts"].sum()
    f_score_max = merged_df["pts"].sum()
    f_acc = n_correct / n_filtered if n_filtered else 0
    f_score_pct = f_score / f_score_max if f_score_max else 0
    st.caption(
        f"{n_filtered} questions — "
        f"Score: {f_score}/{f_score_max} ({f_score_pct:.1%}) · "
        f"Accuracy: {n_correct}/{n_filtered} ({f_acc:.1%})"
    )

    # Question list
    event = st.dataframe(
        merged_df,
        column_config={
            "question_id": st.column_config.TextColumn("ID", width="small"),
            "block": st.column_config.TextColumn("Block", width="small"),
            "number": st.column_config.NumberColumn("No.", format="%d", width="small"),
            "pts": st.column_config.NumberColumn("Pts", format="%d", width="small"),
            "type": st.column_config.TextColumn("Type", width="small"),
            "has_image": st.column_config.CheckboxColumn("Image", width="small"),
            "correct": st.column_config.CheckboxColumn("Correct", width="small"),
            "predicted": st.column_config.TextColumn("Predicted", width="small"),
            "gold": st.column_config.TextColumn("Gold", width="small"),
        },
        hide_index=True,
        width="stretch",
        on_select="rerun",
        selection_mode="single-row",
    )

    # Detail panel
    selected_rows = event.selection.rows if event.selection else []
    if selected_rows:
        idx = selected_rows[0]
        qid = merged_df.iloc[idx]["question_id"]
        q = question_map[qid]
        r = result_map[qid]

        st.divider()
        pts = _points(q["block"], q["number"])
        pts_note = " — 3 pts (required)" if pts == 3 else ""
        st.subheader(f"Question {qid}{pts_note}")

        # Serial group context
        if q.get("serial_group") and q["serial_group"].get("context_text"):
            with st.container(border=True):
                st.caption("Context (shared across serial group)")
                st.markdown(q["serial_group"]["context_text"])

        # Question text
        with st.container(border=True):
            st.caption("Question")
            st.markdown(q["question_text"])

        # Clinical images
        if q.get("clinical_images"):
            with st.container(border=True):
                st.caption("Clinical Images")
                cols = st.columns(min(len(q["clinical_images"]), 3))
                for i, img_name in enumerate(q["clinical_images"]):
                    img_path = IMAGES_DIR / img_name
                    if img_path.exists():
                        cols[i % len(cols)].image(str(img_path), caption=img_name)
                    else:
                        cols[i % len(cols)].warning(f"Image not found: {img_name}")

        # Answer
        col_pred, col_gold = st.columns(2)
        with col_pred:
            predicted_str = ", ".join(r["predicted"]) if r["predicted"] else "—"
            if r["correct"]:
                st.success(f"Predicted: **{predicted_str}**")
            else:
                st.error(f"Predicted: **{predicted_str}**")
        with col_gold:
            st.info(f"Gold: **{', '.join(r['gold'])}**")

        # Reasoning
        with st.expander("Reasoning", expanded=False):
            reasoning = r.get("reasoning")
            if reasoning:
                st.markdown(reasoning)
            else:
                st.caption("N/A — this model does not provide reasoning traces.")

        # Raw response
        with st.expander("Raw Response", expanded=False):
            st.markdown(r.get("raw_response", "—"))


# ---------------------------------------------------------------------------
# View: Model Comparison
# ---------------------------------------------------------------------------
elif view == ":material/compare: Per Question":
    st.header("Model Comparison", divider=True)

    with st.sidebar:
        st.header("Settings", divider=True)
        compare_models = st.multiselect(
            "Models to compare",
            options=leaderboard_df["model"].tolist(),
            default=leaderboard_df["model"].tolist()[:3],
        )
        # Block filter to narrow question list
        compare_block = st.selectbox("Block", options=["All", "A", "B", "C", "D", "E", "F"], index=0)
        filtered_qs = dataset if compare_block == "All" else [q for q in dataset if q["block"] == compare_block]
        # Question selector with preview text
        q_options = [q["question_id"] for q in filtered_qs]

        def _q_label(qid: str) -> str:
            import re
            text = question_map[qid]["question_text"]
            # Strip leading question number like "1　" or "41　"
            text = re.sub(r"^\d+\s*", "", text)
            preview = text[:60].replace("\n", " ")
            return f"{qid} — {preview}..."

        selected_qid = st.selectbox(
            "Question",
            options=q_options,
            index=0 if q_options else None,
            format_func=_q_label,
        )

    if not compare_models:
        st.info("Select at least one model from the sidebar.")
    elif selected_qid:
        q = question_map[selected_qid]

        # Question display
        with st.container(border=True):
            if q.get("serial_group") and q["serial_group"].get("context_text"):
                st.caption("Context")
                st.markdown(q["serial_group"]["context_text"])
                st.divider()
            pts = _points(q["block"], q["number"])
            pts_note = " (3 pts, required)" if pts == 3 else ""
            st.caption(f"Question {selected_qid}{pts_note}")
            st.markdown(q["question_text"])

            if q.get("clinical_images"):
                cols = st.columns(min(len(q["clinical_images"]), 3))
                for i, img_name in enumerate(q["clinical_images"]):
                    img_path = IMAGES_DIR / img_name
                    if img_path.exists():
                        cols[i % len(cols)].image(str(img_path), caption=img_name)

            st.caption(f"Gold answer: **{', '.join(q['answer'])}**")

        # Side-by-side model responses
        n_models = len(compare_models)
        cols_per_row = min(n_models, 3)
        for row_start in range(0, n_models, cols_per_row):
            row_models = compare_models[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_models))
            for col, model_name in zip(cols, row_models):
                with col:
                    with st.container(border=True):
                        result_data = load_result(model_file_map[model_name])
                        r_map = {r["question_id"]: r for r in result_data["results"]}
                        r = r_map.get(selected_qid)

                        st.subheader(model_name)
                        if r is None:
                            st.warning("No result for this question.")
                            continue

                        predicted_str = ", ".join(r["predicted"]) if r["predicted"] else "—"
                        if r["correct"]:
                            st.success(f"Predicted: **{predicted_str}**")
                        else:
                            st.error(f"Predicted: **{predicted_str}**")

                        reasoning = r.get("reasoning")
                        with st.expander("Reasoning"):
                            if reasoning:
                                st.markdown(reasoning)
                            else:
                                st.caption("N/A")

                        with st.expander("Raw Response"):
                            st.markdown(r.get("raw_response", "—"))
