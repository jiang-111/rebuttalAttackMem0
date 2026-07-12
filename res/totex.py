import pandas as pd

INPUT = "paired_bootstrap_vs_best_baseline.csv"
OUTPUT_TEX = "paired_bootstrap_compact.tex"
OUTPUT_CSV = "paired_bootstrap_compact.csv"

# =========================
# Display config
# =========================

QUERY_ORDER = ["stem", "value"]

QUERY_DISPLAY = {
    "stem": "Knowledge Query",
    "value": "Viewpoint Query",
}

MODEL_ORDER = [
    "gemini-2.5-flash",
    "gpt-4o",
    "gpt-5.1",
    "llama-4",
    "qwen3-next",
]

MODEL_DISPLAY = {
    "gemini-2.5-flash": "Gemini-2.5 Flash",
    "gpt-4o": "GPT-4o",
    "gpt-5.1": "GPT-5.1",
    "llama-4": "LLaMA-4",
    "qwen3-next": "Qwen3-Next",
}

BASELINE_DISPLAY = {
    "flipattack": "FlipAttack",
    "PAP_Misrepresentation": "Misrepresentation",
    "SQL_responses": "QueryAttack",
}

SCORE_COL = r"$\Delta$ Score $\uparrow$"
REFUSAL_COL = r"$\Delta$ Refusal $\uparrow$"


# =========================
# Helper functions
# =========================

def parse_bool(x):
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() == "true"


def fmt_delta(mean, low, high, significant):
    """
    Formatting rule:
    - Significant positive improvement: bold
    - Positive but not significant improvement: underline
    - Non-positive improvement: plain
    """
    s = f"{mean:.3f} [{low:.3f}, {high:.3f}]"

    if parse_bool(significant):
        return r"\textbf{" + s + "}"

    if mean > 0:
        return r"\underline{" + s + "}"

    return s


def latex_escape_text(s):
    """
    Escape ordinary text fields, not LaTeX math/formatted metric cells.
    """
    s = str(s)
    return (
        s.replace("&", r"\&")
         .replace("_", r"\_")
         .replace("%", r"\%")
    )


# =========================
# Load
# =========================

df = pd.read_csv(INPUT)

required_cols = {
    "query_type",
    "victim_model",
    "best_visible_baseline",
    "score_improvement_mean",
    "score_improvement_ci_low",
    "score_improvement_ci_high",
    "score_significant",
    "refusal_reduction_mean",
    "refusal_reduction_ci_low",
    "refusal_reduction_ci_high",
    "refusal_significant",
}

missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in {INPUT}: {missing}")


# =========================
# Format columns
# =========================

df["query_order"] = df["query_type"].apply(
    lambda x: QUERY_ORDER.index(x) if x in QUERY_ORDER else 999
)

df["model_order"] = df["victim_model"].apply(
    lambda x: MODEL_ORDER.index(x) if x in MODEL_ORDER else 999
)

df["Target Model"] = (
    df["victim_model"]
    .map(MODEL_DISPLAY)
    .fillna(df["victim_model"])
)

df["Strongest Baseline"] = (
    df["best_visible_baseline"]
    .map(BASELINE_DISPLAY)
    .fillna(df["best_visible_baseline"])
)

df[SCORE_COL] = df.apply(
    lambda r: fmt_delta(
        r["score_improvement_mean"],
        r["score_improvement_ci_low"],
        r["score_improvement_ci_high"],
        r["score_significant"],
    ),
    axis=1,
)

df[REFUSAL_COL] = df.apply(
    lambda r: fmt_delta(
        r["refusal_reduction_mean"],
        r["refusal_reduction_ci_low"],
        r["refusal_reduction_ci_high"],
        r["refusal_significant"],
    ),
    axis=1,
)

df = df.sort_values(["query_order", "model_order"])


# =========================
# Save compact CSV
# =========================

csv_rows = []

for query_type in QUERY_ORDER:
    sub = df[df["query_type"] == query_type]
    if sub.empty:
        continue

    for _, r in sub.iterrows():
        csv_rows.append({
            "Query Type": QUERY_DISPLAY.get(query_type, query_type),
            "Target Model": r["Target Model"],
            "Strongest Baseline": r["Strongest Baseline"],
            SCORE_COL: r[SCORE_COL],
            REFUSAL_COL: r[REFUSAL_COL],
        })

compact_csv = pd.DataFrame(csv_rows)
compact_csv.to_csv(OUTPUT_CSV, index=False)


# =========================
# Generate LaTeX manually
# =========================

lines = []

lines.append(r"\begin{tabular}{llcc}")
lines.append(r"\toprule")
lines.append(
    r"Target Model & Strongest Baseline & "
    + SCORE_COL
    + r" & "
    + REFUSAL_COL
    + r" \\"
)
lines.append(r"\midrule")

first_section = True

for query_type in QUERY_ORDER:
    sub = df[df["query_type"] == query_type]
    if sub.empty:
        continue

    if not first_section:
        lines.append(r"\midrule")
    first_section = False

    lines.append(
        r"\multicolumn{4}{c}{"
        + QUERY_DISPLAY.get(query_type, query_type)
        + r"} \\"
    )
    lines.append(r"\midrule")

    for _, r in sub.iterrows():
        target_model = latex_escape_text(r["Target Model"])
        baseline = latex_escape_text(r["Strongest Baseline"])
        score = r[SCORE_COL]
        refusal = r[REFUSAL_COL]

        line = (
            target_model
            + " & "
            + baseline
            + " & "
            + score
            + " & "
            + refusal
            + r" \\"
        )
        lines.append(line)

lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")

latex = "\n".join(lines)

with open(OUTPUT_TEX, "w", encoding="utf-8") as f:
    f.write(latex)

print(f"Saved {OUTPUT_CSV}")
print(f"Saved {OUTPUT_TEX}")