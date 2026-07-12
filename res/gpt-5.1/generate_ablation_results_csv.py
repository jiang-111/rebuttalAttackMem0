#!/usr/bin/env python3
r"""
Generate ablation-result CSV/LaTeX rows from bootstrap_ci_summary_all_methods.csv.

Default ablation rows:
  - Our_meli_query        -> Direct Malicious Query
  - Our_neutral_question  -> Neutralized Query
  - Our_trap              -> Rebuttal Query (Full)

Input is expected to be the summary produced by the bootstrap CI script, with columns like:
  query_type, victim_model, method_key, method_original,
  score_mean, refusal_mean,
  score_ci_low, score_ci_high, refusal_ci_low, refusal_ci_high

Outputs:
  <prefix>_numeric.csv      numeric means only
  <prefix>_marked.csv       LaTeX-marked means: best=\textbf{}, second=\underline{}
  <prefix>_rows.tex         LaTeX table rows from the marked table
  <prefix>_ci.csv           optional CI-formatted values if CI columns exist
  <prefix>_unmatched.csv    input rows that were not used, useful for debugging aliases
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Display configuration
# -------------------------

MODEL_ORDER = [
    "Gemini-2.5 Flash",
    "GPT-4o",
    "GPT-5.1",
    "LLaMA-4",
    "Qwen3-Next",
]

QUERY_ORDER = [
    "Knowledge Query",
    "Viewpoint Query",
]

METHOD_ORDER = [
    "Direct Malicious Query",
    "Neutralized Query",
    "Rebuttal Query (Full)",
]

# You can edit these labels if your paper uses different wording.
METHOD_ALIASES = {
    # Direct malicious/original harmful query ablation
    "our_meli_query": "Direct Malicious Query",
    "our_malicious_query": "Direct Malicious Query",
    "meli_query": "Direct Malicious Query",
    "malicious_query": "Direct Malicious Query",
    "direct_malicious_query": "Direct Malicious Query",
    "direct_query": "Direct Malicious Query",
    "original_query": "Direct Malicious Query",
    "original_malicious_query": "Direct Malicious Query",

    # Semantic neutralization ablation
    "our_neutral_question": "Neutralized Query",
    "our_neutral_query": "Neutralized Query",
    "neutral_question": "Neutralized Query",
    "neutral_query": "Neutralized Query",
    "neutralized_query": "Neutralized Query",

    # Full method
    "our_trap": "Rebuttal Query (Full)",
    "ourtrap": "Rebuttal Query (Full)",
    "rebuttal_query": "Rebuttal Query (Full)",
    "rebuttalattack": "Rebuttal Query (Full)",
    "rebuttal_attack": "Rebuttal Query (Full)",
    "full_method": "Rebuttal Query (Full)",
}

QUERY_ALIASES = {
    "stem": "Knowledge Query",
    "knowledge": "Knowledge Query",
    "factual": "Knowledge Query",
    "fact": "Knowledge Query",
    "factual_query": "Knowledge Query",
    "knowledge_query": "Knowledge Query",

    "value": "Viewpoint Query",
    "viewpoint": "Viewpoint Query",
    "value_query": "Viewpoint Query",
    "viewpoint_query": "Viewpoint Query",
}


# -------------------------
# Normalization helpers
# -------------------------

def key(x: object) -> str:
    """Normalize strings for robust matching."""
    s = "" if pd.isna(x) else str(x)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def normalize_query_type(row: pd.Series) -> Optional[str]:
    candidates = []
    for col in ["query_type", "query_type_display", "dataset", "split"]:
        if col in row.index:
            candidates.append(row[col])
    for c in candidates:
        k = key(c)
        if k in QUERY_ALIASES:
            return QUERY_ALIASES[k]
    return None


def normalize_model(x: object) -> Optional[str]:
    k = key(x)

    if "gemini" in k and ("2_5" in k or "25" in k or "flash" in k):
        return "Gemini-2.5 Flash"
    if k in {"gemini", "gemini_flash"}:
        return "Gemini-2.5 Flash"

    if "gpt_4o" in k or "gpt4o" in k or k == "gpt_4_o":
        return "GPT-4o"

    if "gpt_5_1" in k or "gpt5_1" in k or "gpt_51" in k:
        return "GPT-5.1"

    if "llama" in k or "llama4" in k or "llama_4" in k:
        return "LLaMA-4"

    if "qwen" in k:
        return "Qwen3-Next"

    return None


def normalize_method(row: pd.Series) -> Optional[str]:
    candidates = []
    for col in ["method_key", "method", "method_original", "method_display"]:
        if col in row.index:
            candidates.append(row[col])
    for c in candidates:
        k = key(c)
        if k in METHOD_ALIASES:
            return METHOD_ALIASES[k]
    return None


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fmt_mean(x: object, digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "--"
    return f"{float(x):.{digits}f}"


def fmt_ci(mean: object, low: object, high: object, digits: int = 3) -> str:
    if any(pd.isna(v) for v in [mean, low, high]):
        return "--"
    return f"{float(mean):.{digits}f} [{float(low):.{digits}f}, {float(high):.{digits}f}]"


def latex_escape(s: str) -> str:
    return (
        str(s)
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


# -------------------------
# Core transformation
# -------------------------

def load_and_normalize(input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path)

    required_mean_cols = {"score_mean", "refusal_mean"}
    missing = required_mean_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    if "victim_model" not in df.columns:
        raise ValueError("Input must contain column: victim_model")

    norm_rows = []
    used_mask = []

    for _, row in df.iterrows():
        query_display = normalize_query_type(row)
        model_display = normalize_model(row["victim_model"])
        method_display = normalize_method(row)

        ok = (
            query_display in QUERY_ORDER
            and model_display in MODEL_ORDER
            and method_display in METHOD_ORDER
        )
        used_mask.append(ok)

        if not ok:
            continue

        new = row.copy()
        new["query_display"] = query_display
        new["model_display"] = model_display
        new["method_display_final"] = method_display
        norm_rows.append(new)

    normalized = pd.DataFrame(norm_rows)
    unmatched = df.loc[[not x for x in used_mask]].copy()
    return normalized, unmatched


def build_numeric_table(df: pd.DataFrame, average_mode: str = "strict") -> pd.DataFrame:
    """Build wide numeric table with columns Model Score/Refusal plus Average."""
    rows = []

    # If duplicate normalized entries exist, average them and warn later through console count.
    grouped = (
        df.groupby(["query_display", "model_display", "method_display_final"], as_index=False)
        .agg(
            score_mean=("score_mean", "mean"),
            refusal_mean=("refusal_mean", "mean"),
            n_rows=("score_mean", "size"),
        )
    )

    for query in QUERY_ORDER:
        for method in METHOD_ORDER:
            row = {"Query": query, "Method": method}
            score_values = []
            refusal_values = []

            for model in MODEL_ORDER:
                g = grouped[
                    (grouped["query_display"] == query)
                    & (grouped["model_display"] == model)
                    & (grouped["method_display_final"] == method)
                ]

                score_col = f"{model} Score"
                refusal_col = f"{model} Refusal"

                if len(g) == 0:
                    row[score_col] = np.nan
                    row[refusal_col] = np.nan
                    score_values.append(np.nan)
                    refusal_values.append(np.nan)
                else:
                    score = float(g["score_mean"].iloc[0])
                    refusal = float(g["refusal_mean"].iloc[0])
                    row[score_col] = score
                    row[refusal_col] = refusal
                    score_values.append(score)
                    refusal_values.append(refusal)

            if average_mode == "skipna":
                row["Average Score"] = float(np.nanmean(score_values)) if not np.all(pd.isna(score_values)) else np.nan
                row["Average Refusal"] = float(np.nanmean(refusal_values)) if not np.all(pd.isna(refusal_values)) else np.nan
            else:
                row["Average Score"] = float(np.mean(score_values)) if not any(pd.isna(score_values)) else np.nan
                row["Average Refusal"] = float(np.mean(refusal_values)) if not any(pd.isna(refusal_values)) else np.nan

            rows.append(row)

    return pd.DataFrame(rows)


def build_ci_table(df: pd.DataFrame, average_mode: str = "strict", digits: int = 3) -> Optional[pd.DataFrame]:
    required = {
        "score_ci_low",
        "score_ci_high",
        "refusal_ci_low",
        "refusal_ci_high",
    }
    if not required.issubset(df.columns):
        return None

    # Mean CIs cannot be exactly averaged from marginal bootstrap CIs. For the Average columns,
    # we output the average of means only and leave CI aggregation to raw paired/bootstrap scripts.
    grouped = (
        df.groupby(["query_display", "model_display", "method_display_final"], as_index=False)
        .agg(
            score_mean=("score_mean", "mean"),
            score_ci_low=("score_ci_low", "mean"),
            score_ci_high=("score_ci_high", "mean"),
            refusal_mean=("refusal_mean", "mean"),
            refusal_ci_low=("refusal_ci_low", "mean"),
            refusal_ci_high=("refusal_ci_high", "mean"),
        )
    )

    rows = []
    numeric = build_numeric_table(df, average_mode=average_mode)

    for query in QUERY_ORDER:
        for method in METHOD_ORDER:
            row = {"Query": query, "Method": method}
            for model in MODEL_ORDER:
                g = grouped[
                    (grouped["query_display"] == query)
                    & (grouped["model_display"] == model)
                    & (grouped["method_display_final"] == method)
                ]
                if len(g) == 0:
                    row[f"{model} Score"] = "--"
                    row[f"{model} Refusal"] = "--"
                else:
                    r = g.iloc[0]
                    row[f"{model} Score"] = fmt_ci(r["score_mean"], r["score_ci_low"], r["score_ci_high"], digits)
                    row[f"{model} Refusal"] = fmt_ci(r["refusal_mean"], r["refusal_ci_low"], r["refusal_ci_high"], digits)

            nrow = numeric[(numeric["Query"] == query) & (numeric["Method"] == method)].iloc[0]
            row["Average Score"] = fmt_mean(nrow["Average Score"], digits)
            row["Average Refusal"] = fmt_mean(nrow["Average Refusal"], digits)
            rows.append(row)

    return pd.DataFrame(rows)


# -------------------------
# Marking best and second best
# -------------------------

def second_distinct(values: Iterable[float], higher_is_better: bool) -> Tuple[Optional[float], Optional[float]]:
    clean = sorted({round(float(v), 12) for v in values if not pd.isna(v)}, reverse=higher_is_better)
    if not clean:
        return None, None
    best = clean[0]
    second = clean[1] if len(clean) > 1 else None
    return best, second


def mark_value(x: object, best: Optional[float], second: Optional[float], digits: int = 3) -> str:
    if x is None or pd.isna(x):
        return "--"
    value = round(float(x), 12)
    s = fmt_mean(value, digits)
    if best is not None and math.isclose(value, best, rel_tol=0, abs_tol=1e-12):
        return r"\textbf{" + s + "}"
    if second is not None and math.isclose(value, second, rel_tol=0, abs_tol=1e-12):
        return r"\underline{" + s + "}"
    return s


def apply_best_second_marks(num_df: pd.DataFrame, digits: int = 3) -> pd.DataFrame:
    out = num_df.copy().astype(object)

    metric_cols = [c for c in num_df.columns if c not in {"Query", "Method"}]

    for query in QUERY_ORDER:
        mask = num_df["Query"] == query
        for col in metric_cols:
            higher_is_better = col.endswith("Score")
            values = num_df.loc[mask, col].values
            best, second = second_distinct(values, higher_is_better=higher_is_better)
            out.loc[mask, col] = [mark_value(v, best, second, digits) for v in values]

    return out


# -------------------------
# LaTeX rows
# -------------------------

def table_columns() -> List[str]:
    cols = ["Query", "Method"]
    for model in MODEL_ORDER:
        cols.extend([f"{model} Score", f"{model} Refusal"])
    cols.extend(["Average Score", "Average Refusal"])
    return cols


def write_latex_rows(marked_df: pd.DataFrame, output_path: Path) -> None:
    cols = table_columns()
    lines = []

    for i, query in enumerate(QUERY_ORDER):
        if i > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{13}}{{c}}{{{latex_escape(query)}}} \\")
        lines.append(r"\midrule")

        block = marked_df[marked_df["Query"] == query]
        for _, row in block.iterrows():
            cells = [latex_escape(row["Method"])]
            for col in cols[2:]:
                cells.append(str(row[col]))
            lines.append(" & ".join(cells) + r" \\")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = table_columns()
    existing = [c for c in cols if c in df.columns]
    return df[existing]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ablation result CSV/LaTeX rows from bootstrap CI summary.")
    parser.add_argument("--input", default="bootstrap_ci_summary_all_methods.csv", help="Input summary CSV path.")
    parser.add_argument("--output-prefix", default="ablation_results", help="Output file prefix.")
    parser.add_argument("--digits", type=int, default=3, help="Decimal digits to show.")
    parser.add_argument(
        "--average-mode",
        choices=["strict", "skipna"],
        default="strict",
        help="strict: Average is -- unless all five models are present. skipna: average available models.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    prefix = Path(args.output_prefix)

    normalized, unmatched = load_and_normalize(input_path)

    if normalized.empty:
        raise RuntimeError(
            "No ablation rows were matched. Check method names in the input and edit METHOD_ALIASES in this script."
        )

    # Debug unmatched rows.
    unmatched_path = prefix.with_name(prefix.name + "_unmatched.csv")
    unmatched.to_csv(unmatched_path, index=False)

    numeric = build_numeric_table(normalized, average_mode=args.average_mode)
    numeric = reorder_columns(numeric)

    marked = apply_best_second_marks(numeric, digits=args.digits)
    marked = reorder_columns(marked)

    numeric_path = prefix.with_name(prefix.name + "_numeric.csv")
    marked_path = prefix.with_name(prefix.name + "_marked.csv")
    rows_path = prefix.with_name(prefix.name + "_rows.tex")
    ci_path = prefix.with_name(prefix.name + "_ci.csv")

    numeric.to_csv(numeric_path, index=False)
    marked.to_csv(marked_path, index=False)
    write_latex_rows(marked, rows_path)

    ci = build_ci_table(normalized, average_mode=args.average_mode, digits=args.digits)
    if ci is not None:
        ci = reorder_columns(ci)
        ci.to_csv(ci_path, index=False)

    # Console diagnostics.
    matched_methods = sorted(normalized["method_display_final"].unique())
    matched_queries = sorted(normalized["query_display"].unique())
    matched_models = sorted(normalized["model_display"].unique())

    duplicate_count = normalized.duplicated(["query_display", "model_display", "method_display_final"]).sum()

    print(f"Matched rows: {len(normalized)}")
    print(f"Matched queries: {matched_queries}")
    print(f"Matched models: {matched_models}")
    print(f"Matched methods: {matched_methods}")
    if duplicate_count:
        print(f"Warning: {duplicate_count} duplicate normalized rows were averaged.")
    if len(unmatched) > 0:
        print(f"Unmatched rows saved for debugging: {unmatched_path}")

    print(f"Saved numeric CSV: {numeric_path}")
    print(f"Saved marked CSV: {marked_path}")
    print(f"Saved LaTeX rows: {rows_path}")
    if ci is not None:
        print(f"Saved CI CSV: {ci_path}")


if __name__ == "__main__":
    main()
