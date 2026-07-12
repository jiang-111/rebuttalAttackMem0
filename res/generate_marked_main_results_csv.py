#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Convert newly computed main-result summaries into a CSV whose cells already contain
LaTeX markers: \textbf{...} for the best result and \underline{...} for the second best.

Expected input: a summary CSV with one row per (query_type, victim_model, method),
for example the output of your corrected main/CI result script.

Supported common columns:
  query_type / QueryType / category
  victim_model / model / Target Model
  method_key / method / method_original / method_display / Method
  score_mean / score / Score
  refusal_mean / refusal / Refusal

Usage:
  python generate_marked_main_results_csv.py \
      --input bootstrap_ci_summary_all_methods.csv \
      --output main_results_marked.csv

Optional:
  python generate_marked_main_results_csv.py \
      --input bootstrap_ci_summary_all_methods.csv \
      --output main_results_marked.csv \
      --tex-output main_results_rows.tex
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# Table order and display names
# =========================

QUERY_ORDER = ["Factual Query", "Value Query"]

MODEL_ORDER = [
    "Gemini-2.5 Flash",
    "GPT-4o",
    "GPT-5.1",
    "LLaMA-4",
    "Qwen3-Next",
]

METHOD_ORDER = [
    "QueryAttack ",
    "Authority endorsement",
    "Evidence-based Persuasion",
    "Expert Endorsement",
    "Logical appeal",
    "Misrepresentation",
    "Rebuttal Attack (Ours)",
]


# =========================
# Normalization helpers
# =========================

def name_key(x: object) -> str:
    """Normalize names for robust matching."""
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    """Pick a column by exact or normalized-name match."""
    cols = list(df.columns)
    exact = {c: c for c in cols}
    norm = {name_key(c): c for c in cols}

    for c in candidates:
        if c in exact:
            return exact[c]
    for c in candidates:
        k = name_key(c)
        if k in norm:
            return norm[k]

    if required:
        raise ValueError(
            f"Cannot find required column. Tried {list(candidates)}. Existing columns: {cols}"
        )
    return None


def parse_first_float(x: object) -> float:
    """
    Parse numeric values from plain floats or strings like:
      0.734
      "0.734 [0.700, 0.760]"
      "\\textbf{0.734}"
    """
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x)
    m = re.search(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    return float(m.group(0)) if m else np.nan


def normalize_query_type(x: object) -> Optional[str]:
    k = name_key(x)
    mapping = {
        "stem": "Factual Query",
        "knowledge": "Factual Query",
        "factual": "Factual Query",
        "fact": "Factual Query",
        "factual_query": "Factual Query",
        "knowledge_query": "Factual Query",
        "value": "Value Query",
        "viewpoint": "Value Query",
        "view_point": "Value Query",
        "value_query": "Value Query",
        "viewpoint_query": "Value Query",
    }
    return mapping.get(k)


def normalize_model(x: object) -> Optional[str]:
    raw = str(x).strip()
    k = name_key(raw)

    # Gemini 2.5 Flash
    if "gemini" in k and "2_5" in k and "flash" in k:
        return "Gemini-2.5 Flash"
    if k in {"gemini_2_5_flash", "gemini_25_flash"}:
        return "Gemini-2.5 Flash"

    # GPT-4o
    if k in {"gpt_4o", "gpt4o", "gpt_4_o"}:
        return "GPT-4o"

    # GPT-5.1
    if k in {"gpt_5_1", "gpt5_1", "gpt_51", "gpt51"}:
        return "GPT-5.1"

    # LLaMA-4 variants
    if "llama" in k and "4" in k:
        return "LLaMA-4"
    if "llama_4_scout" in k:
        return "LLaMA-4"

    # Qwen3-Next variants
    if "qwen" in k and "next" in k:
        return "Qwen3-Next"
    if "qwen3" in k:
        return "Qwen3-Next"

    return None


def normalize_method(x: object) -> Optional[str]:
    raw = str(x).strip()
    k = name_key(raw)

    mapping = {
        "sql_responses": "QueryAttack ",
        "sql_response": "QueryAttack ",
        "queryattack": "QueryAttack ",
        "query_attack": "QueryAttack ",
        "queryattack_sql": "QueryAttack ",
        "query_attack_sql": "QueryAttack ",

        "pap_authority_endorsement": "Authority endorsement",
        "authority_endorsement": "Authority endorsement",
        "authority_endorsement_pap": "Authority endorsement",

        "pap_evidence_based_persuasion": "Evidence-based Persuasion",
        "evidence_based_persuasion": "Evidence-based Persuasion",
        "evidence_based_persuasion_pap": "Evidence-based Persuasion",

        "pap_expert_endorsement": "Expert Endorsement",
        "expert_endorsement": "Expert Endorsement",
        "expert_endorsement_pap": "Expert Endorsement",

        "pap_logical_appeal": "Logical appeal",
        "logical_appeal": "Logical appeal",
        "logical_appeal_pap": "Logical appeal",

        "pap_misrepresentation": "Misrepresentation",
        "misrepresentation": "Misrepresentation",
        "misrepresentation_pap": "Misrepresentation",

        "our_trap": "Rebuttal Attack (Ours)",
        "ourtrap": "Rebuttal Attack (Ours)",
        "rebuttalattack": "Rebuttal Attack (Ours)",
        "rebuttal_attack": "Rebuttal Attack (Ours)",
        "rebuttal_query": "Rebuttal Attack (Ours)",
        "rebuttal_query_full_method": "Rebuttal Attack (Ours)",
        "full_method": "Rebuttal Attack (Ours)",
    }

    if k in mapping:
        return mapping[k]

    # Fuzzy PAP matching
    if "pap" in k and "authority" in k:
        return "Authority endorsement"
    if "pap" in k and "evidence" in k:
        return "Evidence-based Persuasion"
    if "pap" in k and "expert" in k:
        return "Expert Endorsement"
    if "pap" in k and "logical" in k:
        return "Logical appeal"
    if "pap" in k and "misrepresentation" in k:
        return "Misrepresentation"

    # Fuzzy full-method matching
    if "rebuttal" in k or "our_trap" in k:
        return "Rebuttal Attack (Ours)"

    return None


def maybe_infer_metadata_from_file_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    If query_type/model/method columns are absent but a file/source_file column exists,
    try to infer metadata from path parts like:
      stem/gpt-4o/FlipAttack.csv
      value/gemini-2.5-flash/Our_trap.csv
    """
    out = df.copy()
    file_col = pick_col(out, ["source_file", "file", "filepath", "path"], required=False)
    if file_col is None:
        return out

    if "query_type" not in out.columns:
        out["query_type"] = np.nan
    if "victim_model" not in out.columns:
        out["victim_model"] = np.nan
    if "method" not in out.columns:
        out["method"] = np.nan

    for idx, p in out[file_col].items():
        parts = Path(str(p)).parts
        # Search for known query-type segment, then take next part as model and filename stem as method.
        for i, part in enumerate(parts):
            if normalize_query_type(part) is not None and i + 2 < len(parts):
                if pd.isna(out.at[idx, "query_type"]):
                    out.at[idx, "query_type"] = part
                if pd.isna(out.at[idx, "victim_model"]):
                    out.at[idx, "victim_model"] = parts[i + 1]
                if pd.isna(out.at[idx, "method"]):
                    out.at[idx, "method"] = Path(parts[-1]).stem
                break
    return out


# =========================
# Ranking and formatting
# =========================

def distinct_best_second(values: List[float], higher_is_better: bool, tol: float = 1e-12) -> Tuple[Optional[float], Optional[float]]:
    clean = [float(v) for v in values if not pd.isna(v)]
    if not clean:
        return None, None
    unique = []
    for v in sorted(clean, reverse=higher_is_better):
        if not any(abs(v - u) <= tol for u in unique):
            unique.append(v)
    best = unique[0]
    second = unique[1] if len(unique) > 1 else None
    return best, second


def mark_value(value: float, best: Optional[float], second: Optional[float], decimals: int = 3, tol: float = 5e-4) -> str:
    if pd.isna(value):
        return "--"
    s = f"{float(value):.{decimals}f}"
    rounded_value = round(float(value), decimals)

    if best is not None and abs(rounded_value - round(float(best), decimals)) <= tol:
        return rf"\textbf{{{s}}}"
    if second is not None and abs(rounded_value - round(float(second), decimals)) <= tol:
        return rf"\underline{{{s}}}"
    return s


def build_numeric_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query in QUERY_ORDER:
        qdf = summary[summary["query_display"] == query]
        for method in METHOD_ORDER:
            row = {"QueryType": query, "Method": method}
            mdf = qdf[qdf["method_display"] == method]

            score_values = []
            refusal_values = []
            for model in MODEL_ORDER:
                g = mdf[mdf["model_display"] == model]
                if len(g) == 0:
                    score = np.nan
                    refusal = np.nan
                else:
                    # If duplicates exist after alias normalization, average them.
                    score = g["score_value"].mean()
                    refusal = g["refusal_value"].mean()

                row[f"{model} Score"] = score
                row[f"{model} Refusal"] = refusal
                if not pd.isna(score):
                    score_values.append(score)
                if not pd.isna(refusal):
                    refusal_values.append(refusal)

            row["Average Score"] = float(np.mean(score_values)) if score_values else np.nan
            row["Average Refusal"] = float(np.mean(refusal_values)) if refusal_values else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def apply_best_second_marks(num_df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    out = num_df.copy()

    metric_cols = []
    for model in MODEL_ORDER:
        metric_cols.append((f"{model} Score", True))      # higher is better
        metric_cols.append((f"{model} Refusal", False))  # lower is better
    metric_cols.append(("Average Score", True))
    metric_cols.append(("Average Refusal", False))

    for query in QUERY_ORDER:
        mask = out["QueryType"] == query
        for col, higher_is_better in metric_cols:
            values = out.loc[mask, col].tolist()
            best, second = distinct_best_second(values, higher_is_better=higher_is_better)
            out.loc[mask, col] = out.loc[mask, col].apply(
                lambda v: mark_value(v, best, second, decimals=decimals)
            )

    return out


def write_tex_rows(marked_df: pd.DataFrame, tex_output: Path) -> None:
    lines = []
    for query in QUERY_ORDER:
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{13}}{{c}}{{{query}}} \\")
        lines.append(r"\midrule")
        qdf = marked_df[marked_df["QueryType"] == query]
        for _, row in qdf.iterrows():
            cells = [row["Method"]]
            for model in MODEL_ORDER:
                cells.append(row[f"{model} Score"])
                cells.append(row[f"{model} Refusal"])
            cells.append(row["Average Score"])
            cells.append(row["Average Refusal"])
            lines.append(" & ".join(str(c) for c in cells) + r" \\")
    tex_output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Input summary CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output marked CSV.")
    parser.add_argument("--tex-output", type=Path, default=None, help="Optional output .tex rows file.")
    parser.add_argument("--decimals", type=int, default=3, help="Number of decimals to display.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = maybe_infer_metadata_from_file_column(df)

    query_col = pick_col(df, ["query_type", "QueryType", "query", "category", "type"])
    model_col = pick_col(df, ["victim_model", "model", "target_model", "Target Model"])

    # Prefer method_key/original method over method_display because method_display may already be renamed.
    method_col = pick_col(
        df,
        ["method_key", "method", "method_original", "method_display", "Method", "attack_method"],
    )
    score_col = pick_col(df, ["score_mean", "score", "Score"])
    refusal_col = pick_col(df, ["refusal_mean", "refusal", "Refusal"])

    work = pd.DataFrame({
        "query_raw": df[query_col],
        "model_raw": df[model_col],
        "method_raw": df[method_col],
        "score_value": df[score_col].apply(parse_first_float),
        "refusal_value": df[refusal_col].apply(parse_first_float),
    })

    work["query_display"] = work["query_raw"].apply(normalize_query_type)
    work["model_display"] = work["model_raw"].apply(normalize_model)
    work["method_display"] = work["method_raw"].apply(normalize_method)

    bad = work[
        work[["query_display", "model_display", "method_display"]].isna().any(axis=1)
    ].copy()
    if len(bad) > 0:
        bad_path = args.output.with_name(args.output.stem + "_unmatched_rows.csv")
        bad.to_csv(bad_path, index=False)
        print(f"Warning: {len(bad)} rows were not matched to the target table and were saved to: {bad_path}")

    work = work.dropna(subset=["query_display", "model_display", "method_display"])
    work = work[
        work["query_display"].isin(QUERY_ORDER)
        & work["model_display"].isin(MODEL_ORDER)
        & work["method_display"].isin(METHOD_ORDER)
    ].copy()

    # Collapse duplicates created by alias normalization.
    summary = (
        work.groupby(["query_display", "model_display", "method_display"], as_index=False)
        .agg(score_value=("score_value", "mean"), refusal_value=("refusal_value", "mean"))
    )

    num_df = build_numeric_table(summary)
    marked_df = apply_best_second_marks(num_df, decimals=args.decimals)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    marked_df.to_csv(args.output, index=False, encoding="utf-8-sig")

    numeric_path = args.output.with_name(args.output.stem + "_numeric.csv")
    num_df.to_csv(numeric_path, index=False, encoding="utf-8-sig")

    if args.tex_output is not None:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        write_tex_rows(marked_df, args.tex_output)
        print(f"Saved LaTeX rows: {args.tex_output}")

    print(f"Saved marked CSV: {args.output}")
    print(f"Saved numeric CSV: {numeric_path}")


if __name__ == "__main__":
    main()
