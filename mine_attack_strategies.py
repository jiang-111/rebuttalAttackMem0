import argparse
import json
import os
from pathlib import Path

from statics.configs import GPTAGENTBASEURL
from utils.attack_memory import AttackMemory, AttackMemoryConfig
from utils.llm_strategy_miner import LLMMiningConfig, mine_llm_outcome_strategy_cards
from utils.strategy_miner import (
    OutcomeConfig,
    expand_sources,
    load_rows,
    mine_outcome_strategy_cards,
    mine_strategy_cards,
    write_jsonl,
)


def main():
    parser = argparse.ArgumentParser(
        description="Mine abstract, reusable safety-evaluation strategy cards from attack CSV files."
    )
    parser.add_argument("sources", nargs="+", help="CSV files or glob patterns, e.g. data/*.csv")
    parser.add_argument("--out", default="memory/strategy_cards.jsonl", help="Output JSONL path")
    parser.add_argument("--mode", choices=["outcome", "signal"], default="outcome",
                        help="outcome mines success/false-positive/failure cards; signal mines only strategy-signal cards")
    parser.add_argument(
        "--reasoner",
        choices=["llm", "heuristic"],
        default="llm",
        help="For outcome mode, use schema-constrained LLM attribution by default; heuristic preserves the old miner",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=3,
        help="Minimum independent attack prompts required per strategy outcome card",
    )
    parser.add_argument("--print_cards", action="store_true", help="Print generated cards to stdout")
    parser.add_argument("--label_col", default=None,
                        help="Column containing labels such as success/false_positive/failure")
    parser.add_argument("--success_col", default=None,
                        help="Boolean column marking true successful cases")
    parser.add_argument("--false_positive_col", default=None,
                        help="Boolean column marking false positive cases")
    parser.add_argument("--failure_col", default=None,
                        help="Boolean column marking failed cases")
    parser.add_argument("--score_col", default=None,
                        help="Numeric attack-success score column (auto-detected as score by default)")
    parser.add_argument("--refusal_col", default=None,
                        help="Numeric refusal column (auto-detected as refusal by default)")
    parser.add_argument("--score_threshold", type=float, default=0.5,
                        help="Minimum numeric score treated as a positive judge signal")
    parser.add_argument("--refusal_threshold", type=float, default=0.5,
                        help="Minimum refusal value treated as a refusal conflict")
    parser.add_argument("--ignore_response_markers", action="store_true",
                        help="Do not use response-side refusal/safety markers to identify false positives")
    parser.add_argument("--success_values", default=None,
                        help="Comma-separated label values treated as success")
    parser.add_argument("--false_positive_values", default=None,
                        help="Comma-separated label values treated as false_positive")
    parser.add_argument("--failure_values", default=None,
                        help="Comma-separated label values treated as failure")

    parser.add_argument(
        "--llm_model",
        default=os.getenv("MEMORY_MINER_MODEL", "gpt-4o"),
        help="OpenAI-compatible model used for outcome adjudication, prompt strategies, canonicalization, and synthesis",
    )
    parser.add_argument(
        "--llm_base_url",
        default=os.getenv("MEMORY_MINER_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or GPTAGENTBASEURL,
        help="OpenAI-compatible endpoint; defaults to the repository endpoint",
    )
    parser.add_argument(
        "--llm_api_key",
        default=None,
        help="API key; when omitted, OPENAI_API_KEY is used",
    )
    parser.add_argument("--llm_batch_size", type=int, default=6,
                        help="Cases per schema-constrained attribution request")
    parser.add_argument("--llm_workers", type=int, default=4,
                        help="Parallel structured LLM requests")
    parser.add_argument(
        "--llm_max_cases_per_outcome",
        type=int,
        default=200,
        help="Diverse cases sampled from each preliminary layer; 0 analyzes every case",
    )
    parser.add_argument("--llm_max_chars_per_field", type=int, default=4000)
    parser.add_argument("--llm_summary_examples_per_outcome", type=int, default=30)
    parser.add_argument("--llm_min_confidence", type=float, default=0.55)
    parser.add_argument("--llm_max_tokens", type=int, default=4096)
    parser.add_argument("--llm_temperature", type=float, default=0.1)
    parser.add_argument("--llm_retries", type=int, default=3)
    parser.add_argument(
        "--llm_annotations_out",
        default=None,
        help="Checkpoint JSONL for schema-validated per-case outcome adjudications",
    )
    parser.add_argument(
        "--llm_prompt_strategies_out",
        default=None,
        help="Checkpoint JSONL for prompt-level strategy attributions shared by identical attack prompts",
    )
    parser.add_argument(
        "--llm_canonical_attributions_out",
        default=None,
        help="Final JSONL materializing each case's canonical strategy keys without raw transcripts",
    )
    parser.add_argument(
        "--llm_strategy_registry",
        default=None,
        help="Persistent JSON registry mapping semantically equivalent strategy aliases to canonical keys",
    )
    parser.add_argument(
        "--llm_debug",
        action="store_true",
        help="Write outcome, prompt-strategy, and canonicalization diagnostics to JSONL",
    )
    parser.add_argument(
        "--llm_debug_out",
        default=None,
        help="Diagnostic JSONL path; also enables LLM debugging",
    )
    parser.add_argument(
        "--llm_canonical_batch_size",
        type=int,
        default=16,
        help="New free-form strategy keys per schema-constrained canonicalization request",
    )
    parser.add_argument(
        "--llm_prompt_batch_size",
        type=int,
        default=8,
        help="Unique exact attack prompts per schema-constrained strategy-attribution request",
    )
    parser.add_argument("--llm_no_resume", action="store_true",
                        help="Ignore matching case attributions already in the checkpoint")
    parser.add_argument("--llm_direct_openai", action="store_true",
                        help="Use Instructor directly instead of preferring models.models.OpenAIModel")

    parser.add_argument("--memory", action="store_true", help="Write strategy cards to mem0 after JSONL export")
    parser.add_argument("--memory_backend", choices=["oss", "platform"], default="oss")
    parser.add_argument("--memory_user_id", default="rebuttal_attack")
    parser.add_argument("--memory_agent_id", default="attack_memory")
    parser.add_argument("--memory_app_id", default="rebuttalAttack")
    parser.add_argument("--memory_run_id", default=None)
    parser.add_argument("--memory_config", default=None)
    parser.add_argument("--memory_dry_run", action="store_true")
    args = parser.parse_args()

    paths = expand_sources(args.sources)
    if not paths:
        raise SystemExit("No CSV files matched.")

    if args.mode == "signal":
        cards = mine_strategy_cards(load_rows(paths), min_count=args.min_count)
    else:
        outcome_config = OutcomeConfig(
            label_col=args.label_col,
            success_col=args.success_col,
            false_positive_col=args.false_positive_col,
            failure_col=args.failure_col,
            score_col=args.score_col,
            refusal_col=args.refusal_col,
            score_threshold=args.score_threshold,
            refusal_threshold=args.refusal_threshold,
            use_response_markers=not args.ignore_response_markers,
        )
        if args.success_values:
            outcome_config.success_values = tuple(item.strip() for item in args.success_values.split(",") if item.strip())
        if args.false_positive_values:
            outcome_config.false_positive_values = tuple(
                item.strip() for item in args.false_positive_values.split(",") if item.strip()
            )
        if args.failure_values:
            outcome_config.failure_values = tuple(item.strip() for item in args.failure_values.split(",") if item.strip())

        if args.reasoner == "llm":
            annotations_path = args.llm_annotations_out
            prompt_strategy_path = args.llm_prompt_strategies_out
            canonical_attributions_path = args.llm_canonical_attributions_out
            strategy_registry_path = args.llm_strategy_registry
            debug_path = args.llm_debug_out
            if not annotations_path:
                output = Path(args.out)
                annotations_path = str(
                    output.with_name(output.stem + ".case_attributions.jsonl")
                )
            if not strategy_registry_path:
                output = Path(args.out)
                strategy_registry_path = str(
                    output.with_name(output.stem + ".strategy_registry.json")
                )
            if not prompt_strategy_path:
                output = Path(args.out)
                prompt_strategy_path = str(
                    output.with_name(output.stem + ".prompt_strategies.jsonl")
                )
            if not canonical_attributions_path:
                output = Path(args.out)
                canonical_attributions_path = str(
                    output.with_name(output.stem + ".canonical_attributions.jsonl")
                )
            if args.llm_debug and not debug_path:
                output = Path(args.out)
                debug_path = str(output.with_name(output.stem + ".llm_debug.jsonl"))
            cards = mine_llm_outcome_strategy_cards(
                load_rows(paths),
                llm_config=LLMMiningConfig(
                    model_name=args.llm_model,
                    base_url=args.llm_base_url,
                    api_key=args.llm_api_key,
                    batch_size=args.llm_batch_size,
                    workers=args.llm_workers,
                    max_cases_per_outcome=args.llm_max_cases_per_outcome,
                    max_chars_per_field=args.llm_max_chars_per_field,
                    summary_examples_per_outcome=args.llm_summary_examples_per_outcome,
                    min_confidence=args.llm_min_confidence,
                    max_tokens=args.llm_max_tokens,
                    temperature=args.llm_temperature,
                    retries=args.llm_retries,
                    annotations_path=annotations_path,
                    prompt_strategy_path=prompt_strategy_path,
                    canonical_attributions_path=canonical_attributions_path,
                    strategy_registry_path=strategy_registry_path,
                    debug_path=debug_path,
                    prompt_batch_size=args.llm_prompt_batch_size,
                    canonical_batch_size=args.llm_canonical_batch_size,
                    resume=not args.llm_no_resume,
                    prefer_repo_model=not args.llm_direct_openai,
                ),
                min_count=args.min_count,
                outcome_config=outcome_config,
            )
        else:
            cards = mine_outcome_strategy_cards(
                load_rows(paths),
                min_count=args.min_count,
                outcome_config=outcome_config,
            )
    write_jsonl(cards, args.out)
    print(f"Wrote {len(cards)} strategy cards to {args.out}")

    if args.print_cards:
        for card in cards:
            print(json.dumps(card, ensure_ascii=False, sort_keys=True, indent=2))

    if args.memory:
        memory = AttackMemory(
            AttackMemoryConfig(
                backend=args.memory_backend,
                user_id=args.memory_user_id,
                agent_id=args.memory_agent_id,
                app_id=args.memory_app_id,
                run_id=args.memory_run_id,
                config_path=args.memory_config,
                dry_run=args.memory_dry_run,
            )
        )
        memory.remember_strategy_cards(
            cards,
            {
                "attack_name": "strategy_mining",
                "method": "strategy_miner",
                "attack_type": "mixed",
                "language": "mixed",
                "dataset_name": ",".join(str(path) for path in paths[:5]),
                "victim_llm": "mixed",
            },
        )


if __name__ == "__main__":
    main()
