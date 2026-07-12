import re
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# Config
# =========================

ROOT = Path("./llama-4/eval")

# 每个 query type 的主实验样本数。你的 stem/value 都是 150。
DEFAULT_TOTAL_ITEMS = 150
TOTAL_ITEMS_BY_QUERY_TYPE = {
    "stem": 156,
    "value": 159,
    "knowledge": 156,
    "viewpoint": 159,
}

OUR_METHOD = "Our_trap"

VISIBLE_METHODS = [
    "flipattack",
    "cipher",
    "SQL_responses",
    "PAP_Authority endorsement",
    "PAP_Evidence-based Persuasion",
    "PAP_Expert Endorsement",
    "PAP_Logical appeal",
    "PAP_Misrepresentation",
]

VISIBLE_BASELINES = [
    "flipattack",
    "cipher",
    "SQL_responses",
    "PAP_Authority endorsement",
    "PAP_Evidence-based Persuasion",
    "PAP_Expert Endorsement",
    "PAP_Logical appeal",
    "PAP_Misrepresentation",
]

B = 10000
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# CI 表只需要 score/refusal；其他列存在就保留，缺失不强制报错。
CI_METRICS = ["score", "refusal"]
OPTIONAL_NUMERIC_COLS = ["convincingness", "specificity"]
TEXT_COL_CANDIDATES = [
    "response",
    "answer",
    "output",
    "model_output",
    "victim_response",
    "target_response",
    "assistant_response",
]

SKIP_OUTPUT_FILES = {
    "all_results_per_query.csv",
    "all_results_per_metric_main_rule.csv",
    "sample_counts_after_merge.csv",
    "sample_counts_after_main_rule.csv",
    "bootstrap_ci_summary.csv",
    "bootstrap_ci_summary_all_methods.csv",
    "bootstrap_ci_summary_visible_methods.csv",
    "paired_bootstrap_vs_best_baseline.csv",
    "paired_bootstrap_vs_visible_best_baseline.csv",
    "bad_invalid_rows.csv",
    "bad_invalid_metric_rows.csv",
    "bad_or_skipped_files.csv",
    "method_list.csv",
    "cleaning_audit_main_rule.csv",
}

METHOD_DISPLAY_RENAME = {
    "our_trap": "RebuttalAttack",
    "flipattack": "FlipAttack",
    "selfcipher": "SelfCipher",
    "queryattack": "QueryAttack",
    "pap": "PAP",
}

QUERY_TYPE_DISPLAY_RENAME = {
    "stem": "Knowledge",
    "value": "Viewpoint",
    "knowledge": "Knowledge",
    "viewpoint": "Viewpoint",
}


# =========================
# Helper functions
# =========================

def name_key(x):
    x = str(x).strip().lower()
    x = re.sub(r"[^a-z0-9]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def normalize_refusal(x):
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        x = x.strip().lower()
        if x in ["1", "true", "yes", "y", "refused", "refusal"]:
            return 1
        if x in [
            "0",
            "false",
            "no",
            "n",
            "not_refused",
            "not refused",
            "non-refusal",
            "non_refusal",
            "nonrefusal",
        ]:
            return 0

    try:
        return int(float(x))
    except Exception:
        return np.nan


def infer_metadata_from_path(csv_path):
    """
    Expected path under ROOT:
        stem/gpt-4o/Our trap.csv
        value/gemini-2.5-flash/flipattack.csv
    """
    try:
        rel = csv_path.relative_to(ROOT)
    except ValueError:
        rel = csv_path

    parts = rel.parts
    if len(parts) < 3:
        return None, None, None

    query_type = parts[0]
    victim_model = parts[1]
    method = csv_path.stem
    return query_type, victim_model, method


def get_total_items(query_type):
    return TOTAL_ITEMS_BY_QUERY_TYPE.get(str(query_type).lower(), DEFAULT_TOTAL_ITEMS)


def find_text_col(df):
    for col in TEXT_COL_CANDIDATES:
        if col in df.columns:
            return col
    return None


def has_response_text_value(x):
    if pd.isna(x):
        return False
    if isinstance(x, str):
        s = x.strip()
        return len(s) > 0 and s.lower() not in {"nan", "none", "null"}
    return True


def metric_to_numeric(df, metric):
    if metric not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    if metric == "refusal":
        return df[metric].apply(normalize_refusal).astype("float64")
    return pd.to_numeric(df[metric], errors="coerce")


def missing_fill_value(metric):
    """
    Fixed-size evaluation-set convention:
    - score/specificity/convincingness missing => 0, i.e., no useful harmful content.
    - refusal missing/no-response => 1, i.e., attack failed / no usable non-refusal output.
    """
    return 1.0 if metric == "refusal" else 0.0


def bootstrap_ci(values, b=B):
    """
    Non-parametric bootstrap CI for the mean.
    Returns mean, 2.5 percentile, 97.5 percentile.
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]

    n = len(values)
    if n == 0:
        return np.nan, np.nan, np.nan

    # vectorized bootstrap, much faster than Python loop
    sample_idx = rng.integers(0, n, size=(b, n))
    boot_means = values[sample_idx].mean(axis=1)

    mean = values.mean()
    low, high = np.percentile(boot_means, [2.5, 97.5])
    return mean, low, high


def fmt_ci(mean, low, high):
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def add_display_columns(df):
    df = df.copy()
    df["query_type_display"] = df["query_type"].astype(str).replace(QUERY_TYPE_DISPLAY_RENAME)
    df["method_display"] = df["method_key"].map(METHOD_DISPLAY_RENAME).fillna(df["method_original"])
    return df


def clean_metric_main_rule(df, metric, meta, text_col, total_items):
    """
    和修正后的 main 表一致的统计规则：
    1. 有回答无分数 -> drop
    2. 无回答无分数 -> score/spec/conv 补 0，refusal 补 1
    3. 总数不足 total_items -> score/spec/conv padding 0，refusal padding 1

    返回 long-format rows，用于普通 bootstrap CI。
    """
    numeric = metric_to_numeric(df, metric)
    rows = []
    bad_rows = []

    raw_n = len(df)
    valid_n = 0
    no_response_nan_to_zero_n = 0
    response_nan_dropped_n = 0

    for idx in df.index:
        val = numeric.loc[idx]
        query_id = df.at[idx, "meli_query"] if "meli_query" in df.columns else f"__ROW__{idx}"

        has_text = False
        if text_col is not None:
            has_text = has_response_text_value(df.at[idx, text_col])

        if pd.isna(val):
            if has_text:
                response_nan_dropped_n += 1
                bad_rows.append({
                    **meta,
                    "metric": metric,
                    "meli_query": query_id,
                    "row_index": idx,
                    "text_col": text_col,
                    "reason": "has_response_but_metric_nan_drop_then_padding_zero",
                })
                continue
            else:
                no_response_nan_to_zero_n += 1
                value = missing_fill_value(metric)
                reason = "no_response_and_metric_nan_to_fill_value"
        else:
            valid_n += 1
            value = float(val)
            reason = "valid_metric"

        rows.append({
            **meta,
            "metric": metric,
            "meli_query": query_id,
            "value": value,
            "is_padding": False,
            "reason": reason,
        })

    before_padding_n = len(rows)
    padding_n = 0
    if before_padding_n < total_items:
        padding_n = total_items - before_padding_n
        for i in range(padding_n):
            rows.append({
                **meta,
                "metric": metric,
                "meli_query": f"__PAD__::{meta['source_file']}::{metric}::{i}",
                "value": missing_fill_value(metric),
                "is_padding": True,
                "reason": "padding_to_total_items_with_fill_value",
            })

    audit = {
        **meta,
        "metric": metric,
        "total_items": total_items,
        "raw_n": raw_n,
        "valid_metric_n": valid_n,
        "no_response_nan_to_zero_n": no_response_nan_to_zero_n,
        "response_nan_dropped_n": response_nan_dropped_n,
        "before_padding_n": before_padding_n,
        "padding_n": padding_n,
        "final_n": len(rows),
        "text_col": text_col,
    }

    return rows, bad_rows, audit


def clean_metric_for_pairing(df, metric, meta, text_col):
    """
    paired bootstrap 需要保留 query id 才能对齐。
    因此这里把 NaN 按 metric 的失败值处理：
    score -> 0，refusal -> 1。
    """
    numeric = metric_to_numeric(df, metric)
    rows = []

    for idx in df.index:
        val = numeric.loc[idx]
        query_id = df.at[idx, "meli_query"] if "meli_query" in df.columns else f"__ROW__{idx}"
        rows.append({
            **meta,
            "metric": metric,
            "meli_query": query_id,
            "value": missing_fill_value(metric) if pd.isna(val) else float(val),
        })

    return rows


def summarize_metric(metric_df, metric_name):
    rows = []
    group_cols = ["query_type", "victim_model", "method_key"]

    for (query_type, victim_model, method_key), g in metric_df.groupby(group_cols):
        mean, low, high = bootstrap_ci(g["value"].to_numpy())
        original_methods = sorted(g["method"].astype(str).unique())
        method_original = " / ".join(original_methods)

        rows.append({
            "query_type": query_type,
            "victim_model": victim_model,
            "method_key": method_key,
            "method_original": method_original,
            f"n_{metric_name}": len(g),
            f"{metric_name}_mean": mean,
            f"{metric_name}_ci_low": low,
            f"{metric_name}_ci_high": high,
        })

    return pd.DataFrame(rows)


def aligned_values_for_two_methods(pair_metric_group, method_a_key, method_b_key, total_items, fill_value=0.0):
    """
    返回两个方法在同一组 query 上对齐后的值。
    - query 出现在一方但不在另一方：缺失方按 fill_value 补
    - 两方都缺失、导致总数不足 total_items：共同按 fill_value 补
    """
    a = pair_metric_group[pair_metric_group["method_key"] == method_a_key]
    b = pair_metric_group[pair_metric_group["method_key"] == method_b_key]

    def to_map(x):
        if len(x) == 0:
            return pd.Series(dtype=float)
        # 如有重复 query，取均值，避免 merge 后重复膨胀。
        return x.groupby("meli_query")["value"].mean()

    a_map = to_map(a)
    b_map = to_map(b)

    query_ids = sorted(set(a_map.index) | set(b_map.index))
    a_vals = np.array([a_map.get(q, fill_value) for q in query_ids], dtype=float)
    b_vals = np.array([b_map.get(q, fill_value) for q in query_ids], dtype=float)

    if len(query_ids) < total_items:
        pad_n = total_items - len(query_ids)
        a_vals = np.concatenate([a_vals, np.full(pad_n, fill_value)])
        b_vals = np.concatenate([b_vals, np.full(pad_n, fill_value)])

    return a_vals, b_vals, len(query_ids), len(a), len(b)


# =========================
# Step 1: Read and clean all CSV files with main-rule statistics
# =========================

metric_rows = []
pair_metric_rows = []
bad_metric_rows = []
audit_rows = []
skipped_files = []
method_rows = []

for csv_path in ROOT.rglob("*.csv"):
    if csv_path.name in SKIP_OUTPUT_FILES:
        continue

    query_type, victim_model, method = infer_metadata_from_path(csv_path)

    if query_type is None or victim_model is None or method is None:
        skipped_files.append({"file": str(csv_path), "reason": "Cannot infer metadata from path"})
        continue

    try:
        print(f"Reading: {csv_path}")
        df = pd.read_csv(csv_path)
    except Exception as e:
        skipped_files.append({"file": str(csv_path), "reason": f"Unreadable CSV: {e}"})
        continue

    if "meli_query" not in df.columns:
        skipped_files.append({"file": str(csv_path), "reason": "Missing required column: meli_query"})
        continue

    # 只保留需要的列，但不要丢掉 response/output 等文本列。
    keep_cols = [
        c for c in ["meli_query", "judge_model", *CI_METRICS, *OPTIONAL_NUMERIC_COLS, *TEXT_COL_CANDIDATES]
        if c in df.columns
    ]
    df = df[keep_cols].copy()

    text_col = find_text_col(df)
    total_items = get_total_items(query_type)

    meta = {
        "query_type": query_type,
        "victim_model": victim_model,
        "method": method,
        "method_key": name_key(method),
        "source_file": str(csv_path),
    }

    method_rows.append({"method": method, "method_key": name_key(method)})

    for metric in CI_METRICS:
        rows, bad_rows, audit = clean_metric_main_rule(df, metric, meta, text_col, total_items)
        metric_rows.extend(rows)
        bad_metric_rows.extend(bad_rows)
        audit_rows.append(audit)

        pair_rows = clean_metric_for_pairing(df, metric, meta, text_col)
        pair_metric_rows.extend(pair_rows)

if len(metric_rows) == 0:
    raise RuntimeError("No valid result CSV files were found. Please check your directory structure.")

metric_long = pd.DataFrame(metric_rows)
pair_metric_long = pd.DataFrame(pair_metric_rows)

metric_long.to_csv("all_results_per_metric_main_rule.csv", index=False)
print("Saved: all_results_per_metric_main_rule.csv")

# 为兼容旧文件名，也保存一份同内容文件。
metric_long.to_csv("all_results_per_query.csv", index=False)
print("Saved: all_results_per_query.csv")

bad_metric = pd.DataFrame(bad_metric_rows)
bad_metric.to_csv("bad_invalid_metric_rows.csv", index=False)
print("Saved: bad_invalid_metric_rows.csv")

cleaning_audit = pd.DataFrame(audit_rows)
cleaning_audit.to_csv("cleaning_audit_main_rule.csv", index=False)
print("Saved: cleaning_audit_main_rule.csv")

method_list = pd.DataFrame(method_rows).drop_duplicates().sort_values(["method_key", "method"])
method_list.to_csv("method_list.csv", index=False)
print("Saved: method_list.csv")

count_table = (
    metric_long
    .groupby(["query_type", "victim_model", "method", "method_key", "metric"])
    .size()
    .reset_index(name="n_after_main_rule")
    .sort_values(["query_type", "victim_model", "method_key", "metric"])
)
count_table.to_csv("sample_counts_after_main_rule.csv", index=False)
# 兼容旧文件名。
count_table.to_csv("sample_counts_after_merge.csv", index=False)
print("Saved: sample_counts_after_main_rule.csv")
print("Saved: sample_counts_after_merge.csv")


# =========================
# Step 2: Bootstrap CI for all methods
# =========================

score_summary = summarize_metric(metric_long[metric_long["metric"] == "score"], "score")
refusal_summary = summarize_metric(metric_long[metric_long["metric"] == "refusal"], "refusal")

summary_all = score_summary.merge(
    refusal_summary,
    on=["query_type", "victim_model", "method_key", "method_original"],
    how="outer",
)

# 通常 score/refusal 都是 150；如果不一致，保留两个 n 方便检查。
summary_all["n"] = summary_all["n_score"].fillna(summary_all["n_refusal"])

for col in [
    "score_mean",
    "score_ci_low",
    "score_ci_high",
    "refusal_mean",
    "refusal_ci_low",
    "refusal_ci_high",
]:
    summary_all[col] = summary_all[col].round(3)

summary_all = add_display_columns(summary_all)

summary_all["Score"] = summary_all.apply(
    lambda r: fmt_ci(r["score_mean"], r["score_ci_low"], r["score_ci_high"]),
    axis=1,
)
summary_all["Refusal"] = summary_all.apply(
    lambda r: fmt_ci(r["refusal_mean"], r["refusal_ci_low"], r["refusal_ci_high"]),
    axis=1,
)

summary_all = summary_all.sort_values(["query_type_display", "victim_model", "method_display"])
summary_all.to_csv("bootstrap_ci_summary_all_methods.csv", index=False)
print("Saved: bootstrap_ci_summary_all_methods.csv")


# =========================
# Step 3: Bootstrap CI for visible main methods
# =========================

visible_method_keys = {name_key(m) for m in VISIBLE_METHODS} | {name_key(OUR_METHOD)}
summary_visible = summary_all[summary_all["method_key"].isin(visible_method_keys)].copy()

missing_visible = sorted(visible_method_keys - set(summary_all["method_key"].unique()))
if missing_visible:
    print("\nWarning: these visible method keys were not found:")
    for m in missing_visible:
        print("  ", m)

summary_visible.to_csv("bootstrap_ci_summary_visible_methods.csv", index=False)
summary_visible.to_csv("bootstrap_ci_summary.csv", index=False)

print("\nVisible-method CI summary:")
print(summary_visible[["query_type_display", "victim_model", "method_display", "n", "Score", "Refusal"]])
print("Saved: bootstrap_ci_summary_visible_methods.csv")
print("Saved: bootstrap_ci_summary.csv")


# =========================
# Step 4: Save LaTeX CI table for visible methods
# =========================

latex_df = summary_visible[[
    "query_type_display",
    "victim_model",
    "method_display",
    "n",
    "Score",
    "Refusal",
]].copy()

latex_df = latex_df.rename(columns={
    "query_type_display": "Query Type",
    "victim_model": "Target Model",
    "method_display": "Method",
    "n": "$n$",
})

latex_str = latex_df.to_latex(index=False, escape=False, column_format="lllcll")

with open("bootstrap_ci_summary_visible_methods.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)

print("Saved: bootstrap_ci_summary_visible_methods.tex")


# =========================
# Step 5: Bootstrap vs strongest visible baseline
# =========================

paired_rows = []
our_key = name_key(OUR_METHOD)
baseline_keys = {name_key(m) for m in VISIBLE_BASELINES}

print("\nOUR_METHOD:")
print(f"  {OUR_METHOD} => {our_key}")

print("\nVISIBLE_BASELINES:")
for m in VISIBLE_BASELINES:
    print(f"  {m} => {name_key(m)}")

score_pair_long = pair_metric_long[pair_metric_long["metric"] == "score"].copy()
refusal_pair_long = pair_metric_long[pair_metric_long["metric"] == "refusal"].copy()

for (query_type, victim_model), score_group in score_pair_long.groupby(["query_type", "victim_model"]):
    total_items = get_total_items(query_type)

    ours_score_group = score_group[score_group["method_key"] == our_key]
    baseline_score_group = score_group[score_group["method_key"].isin(baseline_keys)]

    if len(ours_score_group) == 0:
        print(f"\nSkip bootstrap: {query_type}, {victim_model} has no OUR_METHOD={OUR_METHOD} ({our_key})")
        print("Available methods:", sorted(score_group["method"].astype(str).unique()))
        continue

    if len(baseline_score_group) == 0:
        print(f"\nSkip bootstrap: {query_type}, {victim_model} has no visible baselines")
        print("Available methods:", sorted(score_group["method"].astype(str).unique()))
        continue

    # strongest baseline 用 main-rule 后的 score mean 来选。
    summary_slice = summary_all[
        (summary_all["query_type"] == query_type)
        & (summary_all["victim_model"] == victim_model)
        & (summary_all["method_key"].isin(baseline_keys))
    ].copy()

    if len(summary_slice) == 0:
        continue

    best_row = summary_slice.loc[summary_slice["score_mean"].idxmax()]
    best_baseline_key = best_row["method_key"]
    best_baseline = best_row["method_original"]
    best_baseline_score_mean = best_row["score_mean"]

    ours_summary_row = summary_all[
        (summary_all["query_type"] == query_type)
        & (summary_all["victim_model"] == victim_model)
        & (summary_all["method_key"] == our_key)
    ]
    ours_score_mean = np.nan if len(ours_summary_row) == 0 else float(ours_summary_row.iloc[0]["score_mean"])

    # Score improvement: ours - best baseline
    score_a, score_b, n_union_score, n_ours_raw, n_base_raw = aligned_values_for_two_methods(
        score_group,
        our_key,
        best_baseline_key,
        total_items,
        fill_value=0.0,
    )
    score_diff = score_a - score_b
    score_mean, score_low, score_high = bootstrap_ci(score_diff)

    # Refusal reduction: baseline - ours
    refusal_group = refusal_pair_long[
        (refusal_pair_long["query_type"] == query_type)
        & (refusal_pair_long["victim_model"] == victim_model)
    ]
    refusal_a, refusal_b, n_union_refusal, _, _ = aligned_values_for_two_methods(
        refusal_group,
        our_key,
        best_baseline_key,
        total_items,
        fill_value=1.0,
    )
    refusal_reduction = refusal_b - refusal_a
    refusal_mean, refusal_low, refusal_high = bootstrap_ci(refusal_reduction)

    paired_rows.append({
        "query_type": query_type,
        "victim_model": victim_model,
        "our_method": OUR_METHOD,
        "our_method_key": our_key,
        "best_visible_baseline": best_baseline,
        "best_visible_baseline_key": best_baseline_key,
        "n_ours_raw": n_ours_raw,
        "n_baseline_raw": n_base_raw,
        "n_aligned_for_bootstrap": len(score_diff),
        "n_actual_query_union_score": n_union_score,
        "n_actual_query_union_refusal": n_union_refusal,
        "our_score_mean": ours_score_mean,
        "best_baseline_score_mean": best_baseline_score_mean,
        "score_improvement_mean": score_mean,
        "score_improvement_ci_low": score_low,
        "score_improvement_ci_high": score_high,
        "score_significant": score_low > 0,
        "our_refusal_mean": refusal_a.mean() if len(refusal_a) else np.nan,
        "best_baseline_refusal_mean": refusal_b.mean() if len(refusal_b) else np.nan,
        "refusal_reduction_mean": refusal_mean,
        "refusal_reduction_ci_low": refusal_low,
        "refusal_reduction_ci_high": refusal_high,
        "refusal_significant": refusal_low > 0,
    })

paired = pd.DataFrame(paired_rows)

if len(paired) > 0:
    numeric_cols = [
        "our_score_mean",
        "best_baseline_score_mean",
        "score_improvement_mean",
        "score_improvement_ci_low",
        "score_improvement_ci_high",
        "our_refusal_mean",
        "best_baseline_refusal_mean",
        "refusal_reduction_mean",
        "refusal_reduction_ci_low",
        "refusal_reduction_ci_high",
    ]
    for col in numeric_cols:
        paired[col] = paired[col].round(3)

    paired = paired.sort_values(["query_type", "victim_model"])
    paired.to_csv("paired_bootstrap_vs_visible_best_baseline.csv", index=False)
    paired.to_csv("paired_bootstrap_vs_best_baseline.csv", index=False)

    print("\nBootstrap vs strongest visible baseline:")
    print(paired)
    print("Saved: paired_bootstrap_vs_visible_best_baseline.csv")
    print("Saved: paired_bootstrap_vs_best_baseline.csv")
else:
    print("\nNo paired/bootstrap comparison results were generated.")


# =========================
# Step 6: Save skipped file log
# =========================

skipped = pd.DataFrame(skipped_files)
skipped.to_csv("bad_or_skipped_files.csv", index=False)

print("\nSaved: bad_or_skipped_files.csv")
print("Done.")
