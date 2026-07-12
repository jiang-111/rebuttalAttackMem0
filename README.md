# RebuttalAttack

This repository contains code and data for running rebuttal-style jailbreak / safety evaluation experiments.

## Repository Layout

- `attack.py`: main CLI for generating attack prompts and target model responses.
- `main.py`: legacy process-based generation pipeline.
- `probe_full_dataset.py`, `rcl.py`: refusal/continuation likelihood probing scripts.
- `rebuttal_man.py`, `rebuttal_analy.py`, `newana.py`: rebuttal experiment and analysis utilities.
- `methods/`: attack method implementations.
- `models/`: model wrappers for OpenAI-compatible, Anthropic, Gemini, LiteLLM, Ollama, and local vLLM backends.
- `statics/`: prompt templates, system prompts, schemas, and attack configuration.
- `utils/`: CSV/JSON/string helpers.
- `strong_reject/`, `strongReject.py`: StrongReject-style evaluation helpers.
- `data/`: curated experiment datasets and prompt sets.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set the API key required by your backend:

```bash
export OPENAI_API_KEY="..."
# Optional, depending on backend:
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

## Example Usage

Generate attack results:

```bash
python attack.py \
  --victim_llm gpt-4o \
  --attack trap \
  --dataset_pre stem \
  --lang en \
  --dataset_root data \
  --dataset_name ablation_stem_en_150.csv \
  --save_root result_150/stem/gpt-4o \
  --save_name trap
```

Probe refusal/continuation likelihoods with a local Hugging Face model:

```bash
python probe_full_dataset.py \
  --model Qwen/Qwen3-8B \
  --stem_csv data/stem_150_rechecked.csv \
  --value_csv data/value_150_rechecked.csv \
  --out_csv refusal_compare.csv
```

Enable Mem0-backed experiment memory:

```bash
pip install mem0ai

python attack.py \
  --victim_llm gpt-4o \
  --attack trap \
  --dataset_pre stem \
  --lang en \
  --dataset_root data \
  --dataset_name ablation_stem_en_150.csv \
  --save_root result_150 \
  --save_name trap \
  --memory
```

`--memory` retrieves prior three-layer strategy cards before generation and
writes a new run summary after saving results. It stores aggregate lessons, field coverage,
error counts, refusal-like counts, and hashes for repeated items rather than
full raw prompts or full model outputs. Use `--memory_dry_run` to inspect the
payload without calling Mem0. For hosted Mem0, set `MEM0_API_KEY` and pass
`--memory_backend platform`.

### Three-layer attack experience memory

Evaluated CSV files are mined into three balanced memory layers. Outcome mining uses an
LLM by default:

1. Rules use score/refusal/error fields only to build preliminary strata.
2. A schema-constrained LLM reads the original objective, actual target response, and judge
   signals. It compares the requested deliverable with
   what the response actually supplies; code treats only a substantially supplied central
   deliverable as fulfillment and derives attack `success`, judge `false_positive`,
   `failure`, or `unknown`. This outcome pass never receives method, model, source-path,
   or category metadata.
3. A separate schema-constrained pass extracts mechanisms once per *exact attack prompt*
   using only the original objective and prompt. Identical prompts share the same strategy
   observations. The global guide supplies broad-family definitions and prior canonical
   mechanisms, but not outcomes or method metadata; direct restatements normalize to
   `direct_baseline`.
4. A family-scoped canonicalization pass maps semantically equivalent fine-grained aliases
   to stable keys, recursively recovers incomplete schema batches, and records diagnostics.
   Broad built-in families remain organizational labels and do not replace specific
   mechanisms; novel mechanisms remain allowed.
5. A final schema-constrained LLM pass compares all cases sharing a canonical strategy
   and explains which conditions separate true success from false positives and failures.

All LLM passes use Pydantic response models through `models.models.OpenAIModel.chat_templ`
(Instructor `TOOLS` mode). If that repository model cannot be imported because optional
runtime dependencies are unavailable, the miner uses the same Instructor `TOOLS` mode
directly. Free-form or repaired-but-schema-invalid responses are not written to memory.

Mine the evaluation data already under `res/` into local JSONL strategy memory:

```bash
python mine_attack_strategies.py \
  'res/*/eval/stem/*/*.csv' \
  'res/*/eval/value/*/*.csv' \
  --mode outcome \
  --score_col score \
  --refusal_col refusal \
  --score_threshold 0.5 \
  --refusal_threshold 0.5 \
  --min_count 5 \
  --llm_model gpt-4o \
  --llm_base_url https://chatapi.littlewheat.com \
  --out memory/strategy_cards.jsonl
```

The default API budget samples a diverse maximum of 200 cases from each preliminary
stratum, including evidence-poor `unknown` rows for LLM re-adjudication. Use
`--llm_max_cases_per_outcome 0` to analyze every case. Validated per-case
attributions are checkpointed beside the card file as
`strategy_cards.case_attributions.jsonl`; prompt-level mechanism observations are separately
checkpointed as `strategy_cards.prompt_strategies.jsonl`, and the final per-case canonical
keys are materialized as `strategy_cards.canonical_attributions.jsonl`. Reruns resume matching
cases and exact prompts for the same model and schema version. Canonical strategy aliases are
persisted in `strategy_cards.strategy_registry.json`, so later runs reuse the same keys. Use
`--llm_no_resume` to force re-analysis. A final card needs at least three independent exact
attack prompts by default; `guidance_eligible` also requires two victim models and excludes
`direct_baseline`. With the default cap, card rates
and lift describe the LLM-reviewed stratified sample, not the untouched full-data
distribution.

If a structured batch omits case IDs, the miner keeps valid returned cases and retries
only the missing subset, recursively splitting down to individual cases when needed. The
same recovery behavior applies to prompt strategies and canonicalization. Add `--llm_debug`
to write `strategy_cards.llm_debug.jsonl` with case locations, prompt IDs, canonical mappings,
`finish_reason`, refusal metadata, token usage, accepted/missing IDs, and shortened API errors.
The diagnostic log omits API keys and raw attack transcripts.

Each final card records LLM-derived reasons, cross-outcome conditions, confidence,
outcome rates/lift, recommended action, and case hashes instead of raw prompts or
responses. To persist the same cards to Mem0, add `--memory`; both mining and attack
retrieval use the shared `attack_memory` agent namespace by default.

The previous keyword/statistical miner remains available for reproducibility with
`--reasoner heuristic`; it is no longer the default for outcome memory.

Guide a new attack-prompt generation run directly from the local cards:

```bash
python attack.py \
  --attack trap \
  --dataset_pre stem \
  --lang en \
  --dataset_root data \
  --dataset_name ablation_stem_en_150.csv \
  --memory_cards memory/strategy_cards.jsonl
```

Or retrieve the three layers from Mem0 with `--memory`. Retrieval is balanced
per outcome so success cards cannot crowd out false-positive and failure
lessons. Guidance is injected into attack-prompt construction stages. It is not
injected into `target`, `target_ablation`, or `judge` by default, which avoids
changing the victim-model evaluation condition; use `--memory_guide_target`
only when that behavior is intentional.

## Notes

- Generated outputs are intentionally ignored by git. Keep large result folders outside the repository or publish them as release artifacts.
- No API keys are required in source files. Use environment variables or pass CLI arguments where supported.
- Some scripts are retained for reproducibility and may reflect experiment-specific defaults; prefer `attack.py` for new runs.
