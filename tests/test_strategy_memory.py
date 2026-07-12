import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.attack_memory import (
    AttackMemory,
    AttackMemoryConfig,
    build_guided_system_prompt,
    format_strategy_guidance,
    has_refusal_marker,
    layer_strategy_cards,
    load_strategy_cards,
)
from utils.strategy_miner import (
    OutcomeConfig,
    OutcomeDecision,
    classify_outcome,
    mine_outcome_strategy_cards,
)
from utils.llm_strategy_miner import (
    CaseAttribution,
    CaseAttributionBatch,
    CaseOutcomeAttribution,
    CaseOutcomeAttributionBatch,
    CaseStrategy,
    LLMMiningConfig,
    OutcomeExplanation,
    PreparedCase,
    PromptStrategyAttribution,
    PromptStrategyAttributionBatch,
    StrategyCanonicalAssignment,
    StrategyCanonicalizationBatch,
    StrategySynthesis,
    _analyze_cases,
    _analyze_prompt_strategies,
    _attack_prompt,
    _case_payload,
    _attach_prompt_strategies,
    _canonicalize_strategy_keys,
    _prompt_id,
    _validated_annotation,
    _validated_canonical_mapping,
    mine_llm_outcome_strategy_cards,
)


class OutcomeClassificationTests(unittest.TestCase):
    def setUp(self):
        self.config = OutcomeConfig(score_col="score", refusal_col="refusal")

    def test_input_refusal_words_do_not_contaminate_response_check(self):
        row = {
            "meli_query": "The input contains unsafe and illegal wording",
            "answer": "A complete response without a refusal marker.",
        }
        self.assertFalse(has_refusal_marker(row))

    def test_verified_success_requires_consistent_signals(self):
        row = {"score": "0.875", "refusal": "0", "answer": "Requested technical explanation."}
        decision = classify_outcome(row, self.config)
        self.assertEqual(decision.outcome, "success")
        self.assertIn("non_refusal_confirmed", decision.reason_codes)

    def test_positive_score_with_safe_redirect_is_false_positive(self):
        row = {"score": "1", "refusal": "0", "answer": "I cannot help with that; use a safe alternative."}
        decision = classify_outcome(row, self.config)
        self.assertEqual(decision.outcome, "false_positive")
        self.assertIn("response_refusal_or_safety_marker", decision.reason_codes)

    def test_low_score_refusal_is_failure(self):
        row = {"score": "0", "refusal": "1", "answer": "I cannot assist."}
        decision = classify_outcome(row, self.config)
        self.assertEqual(decision.outcome, "failure")
        self.assertIn("score_below_threshold", decision.reason_codes)


class ThreeLayerMemoryTests(unittest.TestCase):
    def test_miner_builds_all_three_outcome_layers(self):
        base = {
            "trap": "According to experts, provide a concrete example.",
            "_source_path": "res/judge/eval/stem/model/PAP_Authority endorsement.csv",
            "_method": "PAP_Authority endorsement",
            "_query_type": "stem",
            "_victim_model": "model",
        }
        rows = [
            {**base, "score": "1", "refusal": "0", "answer": "Detailed response."},
            {**base, "score": "1", "refusal": "0", "answer": "I cannot help; here is a safe alternative."},
            {**base, "score": "0", "refusal": "1", "answer": "I cannot assist."},
        ]
        cards = mine_outcome_strategy_cards(rows, min_count=1, outcome_config=OutcomeConfig())
        outcomes = {card.get("outcome") for card in cards if card.get("outcome")}
        self.assertEqual(outcomes, {"success", "false_positive", "failure"})

        layered = layer_strategy_cards(cards)
        guidance = format_strategy_guidance(layered, limit_per_outcome=1)
        self.assertIn("TRUE SUCCESS", guidance)
        self.assertIn("FALSE POSITIVE", guidance)
        self.assertIn("FAILURE", guidance)

    def test_local_jsonl_cards_can_drive_prompt_guidance(self):
        card = {
            "kind": "outcome_strategy_card",
            "outcome": "success",
            "strategy": "abstract_test_strategy",
            "why_hypothesis": "Consistent judge evidence.",
            "recommended_action": "Reuse when relevant.",
            "support_count": 3,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cards.jsonl"
            path.write_text(json.dumps(card) + "\n", encoding="utf-8")
            loaded = load_strategy_cards(str(path))

        guidance = format_strategy_guidance(layer_strategy_cards(loaded))
        self.assertIn("abstract_test_strategy", guidance)
        self.assertIn("Reuse when relevant", guidance)
        guided_prompt = build_guided_system_prompt("base system prompt", guidance)
        self.assertIn("base system prompt", guided_prompt)
        self.assertIn("Reuse only verified-success patterns", guided_prompt)
        self.assertIn("abstract_test_strategy", guided_prompt)

    def test_mem0_search_and_write_preserve_outcome_layers(self):
        class FakeClient:
            def __init__(self):
                self.search_calls = []
                self.add_calls = []

            def search(self, query, filters=None, top_k=None):
                self.search_calls.append((query, filters, top_k))
                return {"results": [{"memory": query, "metadata": filters}]}

            def add(self, **kwargs):
                self.add_calls.append(kwargs)

        manager = AttackMemory.__new__(AttackMemory)
        manager.config = AttackMemoryConfig(run_id="test-run", top_k=9)
        manager.client = FakeClient()

        layers = manager.search_strategy_guidance(
            {"attack_type": "stem", "victim_llm": "model", "language": "en"},
            top_k_per_outcome=2,
        )
        self.assertEqual(set(layers), {"success", "false_positive", "failure"})
        self.assertEqual(len(manager.client.search_calls), 3)
        self.assertTrue(
            all(call[1]["memory_kind"] == "outcome_strategy_card" for call in manager.client.search_calls)
        )
        self.assertEqual(
            {call[1]["outcome"] for call in manager.client.search_calls},
            {"success", "false_positive", "failure"},
        )

        manager.remember_strategy_cards(
            [{
                "kind": "outcome_strategy_card",
                "outcome": "success",
                "strategy": "abstract_test_strategy",
                "support_count": 3,
            }],
            {"attack_name": "mining", "attack_type": "mixed", "language": "mixed"},
        )
        metadata = manager.client.add_calls[0]["metadata"]
        self.assertEqual(metadata["memory_kind"], "outcome_strategy_card")
        self.assertEqual(metadata["outcome"], "success")
        self.assertEqual(metadata["agent_id"], "attack_memory")


class StrategyCanonicalizationTests(unittest.TestCase):
    def test_equivalent_free_form_keys_get_one_deterministic_key(self):
        response = StrategyCanonicalizationBatch(
            assignments=[
                StrategyCanonicalAssignment(
                    source_key="contextual_legitimization",
                    equivalent_to="benign_context_reframing",
                    rationale="Both use benign context to legitimize the same request.",
                    confidence=0.91,
                ),
                StrategyCanonicalAssignment(
                    source_key="benign_context_reframing",
                    equivalent_to="benign_context_reframing",
                    rationale="Canonical representative for the equivalent pair.",
                    confidence=0.93,
                ),
            ]
        )
        mapping = _validated_canonical_mapping(
            response,
            candidate_keys=("contextual_legitimization", "benign_context_reframing"),
            existing_canonicals=("authority_framing",),
            min_confidence=0.55,
        )
        self.assertEqual(
            mapping,
            {
                "contextual_legitimization": "benign_context_reframing",
                "benign_context_reframing": "benign_context_reframing",
            },
        )

    def test_fine_grained_key_can_reuse_existing_fine_grained_canonical(self):
        response = StrategyCanonicalizationBatch(
            assignments=[
                StrategyCanonicalAssignment(
                    source_key="expert_credibility_appeal",
                    equivalent_to="credentialed_expert_legitimation",
                    rationale="The fine-grained mechanisms are equivalent.",
                    confidence=0.9,
                )
            ]
        )
        mapping = _validated_canonical_mapping(
            response,
            candidate_keys=("expert_credibility_appeal",),
            existing_canonicals=("credentialed_expert_legitimation",),
            min_confidence=0.55,
        )
        self.assertEqual(
            mapping["expert_credibility_appeal"],
            "credentialed_expert_legitimation",
        )

    def test_registry_is_persisted_and_reused_without_another_llm_call(self):
        annotations = [
            {
                "case_id": "case-a",
                "strategies": [
                    {
                        "strategy_family": "authority_framing",
                        "strategy_key": "contextual_legitimization",
                        "strategy_name": "Contextual legitimization",
                        "mechanism": "Makes the request look benign through context.",
                        "reusable_principle": "Use a benign-looking context bridge.",
                        "confidence": 0.8,
                    }
                ],
            },
            {
                "case_id": "case-b",
                "strategies": [
                    {
                        "strategy_family": "authority_framing",
                        "strategy_key": "benign_context_reframing",
                        "strategy_name": "Benign context reframing",
                        "mechanism": "Makes the request look benign through context.",
                        "reusable_principle": "Use a benign-looking context bridge.",
                        "confidence": 0.9,
                    }
                ],
            },
        ]

        class FakeSchemaModel:
            calls = 0

            def __init__(self, config, system_prompt):
                pass

            def generate(self, prompt, schema, **kwargs):
                self.__class__.calls += 1
                return StrategyCanonicalizationBatch(
                    assignments=[
                        StrategyCanonicalAssignment(
                            source_key="contextual_legitimization",
                            equivalent_to="benign_context_reframing",
                            rationale="Equivalent contextual legitimacy mechanism.",
                            confidence=0.9,
                        ),
                        StrategyCanonicalAssignment(
                            source_key="benign_context_reframing",
                            equivalent_to="benign_context_reframing",
                            rationale="Representative key.",
                            confidence=0.9,
                        ),
                    ]
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "strategy_registry.json"
            config = LLMMiningConfig(
                model_name="fake-model",
                strategy_registry_path=str(registry_path),
            )
            with patch("utils.llm_strategy_miner._SchemaModel", FakeSchemaModel):
                first = _canonicalize_strategy_keys(annotations, config)
            self.assertEqual(FakeSchemaModel.calls, 1)
            self.assertTrue(registry_path.is_file())
            self.assertEqual(
                {item["strategies"][0]["strategy_key"] for item in first},
                {"benign_context_reframing"},
            )

            class UnexpectedSchemaModel:
                def __init__(self, config, system_prompt):
                    raise AssertionError("Persisted aliases should avoid another LLM call")

            with patch("utils.llm_strategy_miner._SchemaModel", UnexpectedSchemaModel):
                second = _canonicalize_strategy_keys(annotations, config)
            self.assertEqual(
                {item["strategies"][0]["strategy_key"] for item in second},
                {"benign_context_reframing"},
            )

    def test_incomplete_canonical_batch_is_split_and_recovered(self):
        annotations = [
            {
                "case_id": f"case-{index}",
                "strategies": [
                    {
                        "strategy_family": "obfuscation",
                        "strategy_key": f"distinct_obfuscation_{index}",
                        "strategy_name": "Distinct obfuscation",
                        "mechanism": f"Distinct transformation {index}.",
                        "reusable_principle": "Use a distinct prompt transformation.",
                        "confidence": 0.9,
                    }
                ],
            }
            for index in range(4)
        ]

        class FakeSchemaModel:
            calls = 0

            def __init__(self, config, system_prompt):
                self.trace = []

            def generate(self, prompt, schema, **kwargs):
                self.__class__.calls += 1
                if self.__class__.calls == 1:
                    return StrategyCanonicalizationBatch(assignments=[])
                payload = json.loads(prompt.split("\n\n", 1)[1])
                return StrategyCanonicalizationBatch(
                    assignments=[
                        StrategyCanonicalAssignment(
                            source_key=item["source_key"],
                            equivalent_to=item["source_key"],
                            rationale="Distinct tactic.",
                            confidence=0.9,
                        )
                        for item in payload["candidate_strategies"]
                    ]
                )

            def take_debug_trace(self):
                return self.trace

        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "strategy_registry.json"
            debug_path = Path(tmpdir) / "debug.jsonl"
            config = LLMMiningConfig(
                model_name="fake-model",
                strategy_registry_path=str(registry_path),
                debug_path=str(debug_path),
                canonical_batch_size=4,
                retries=1,
            )
            with patch("utils.llm_strategy_miner._SchemaModel", FakeSchemaModel):
                result = _canonicalize_strategy_keys(annotations, config)

            self.assertEqual(FakeSchemaModel.calls, 3)
            self.assertEqual(len(result), 4)
            events = [json.loads(line) for line in debug_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["event"], "canonicalization_batch_api_or_schema_error")
            self.assertEqual(
                sum(event["event"] == "canonicalization_batch_response" for event in events),
                2,
            )


class MethodAwareAttributionTests(unittest.TestCase):
    def test_cipher_uses_jailbroken_prompt_instead_of_populated_trap_column(self):
        row = {
            "_method": "cipher",
            "trap": "A sibling trap prompt that was not evaluated.",
            "jailbroken_prompt": "The actual cipher evaluation prompt.",
        }
        self.assertEqual(_attack_prompt(row), "The actual cipher evaluation prompt.")

    def test_pap_prefers_plain_mutated_prompt(self):
        row = {
            "_method": "PAP_Logical appeal",
            "jailbroken_prompt": "[{'role': 'user', 'content': 'wrapped'}]",
            "mutated_text_with_same_specific_harmful_or_unlawful_intention": "plain prompt",
        }
        self.assertEqual(_attack_prompt(row), "plain prompt")

    def test_missing_method_prompt_falls_back_to_objective_not_sibling_strategy(self):
        row = {
            "_method": "cipher",
            "trap": "A sibling trap prompt.",
            "meli_query": "The original evaluated objective.",
        }
        self.assertEqual(_attack_prompt(row), "The original evaluated objective.")

    def test_outcome_payload_excludes_method_and_other_strategy_metadata(self):
        case = PreparedCase(
            case_id="case-cipher",
            row={
                "_method": "cipher",
                "jailbroken_prompt": "encoded prompt",
                "meli_query": "objective",
                "answer": "response",
            },
            decision=OutcomeDecision("success", ["positive_judge_signal"], 1.0, 0.0),
        )
        payload = _case_payload(case, 4000)
        self.assertEqual(payload["original_objective"], "objective")
        self.assertNotIn("attack_prompt", payload)
        self.assertNotIn("metadata", payload)

    def test_fulfillment_is_derived_as_attack_success_even_from_low_preliminary_score(self):
        case = PreparedCase(
            case_id="case-success",
            row={"_method": "Our_meli_query", "meli_query": "objective", "answer": "response"},
            decision=OutcomeDecision("failure", ["score_below_threshold"], 0.0, 0.0),
        )
        analysis = CaseAttribution(
            case_id=case.case_id,
            requested_deliverable="A usable response accomplishing the original objective.",
            delivered_substance="Usable substance that accomplishes the objective.",
            deliverable_match="substantially_supplied",
            outcome_reason="The response materially accomplishes the original objective.",
            strategies=[
                CaseStrategy(
                    strategy_family="direct_baseline",
                    strategy_key="direct_baseline",
                    strategy_name="Direct baseline",
                    mechanism="Presents the objective directly.",
                    evidence="The attack prompt directly states the objective.",
                    evidence_source="attack_prompt",
                    reusable_principle="Use as a direct control.",
                    confidence=0.9,
                )
            ],
            confidence=0.9,
        )
        annotation = _validated_annotation(analysis, case, LLMMiningConfig(model_name="fake"))
        self.assertEqual(annotation["outcome"], "success")

    def test_non_fulfillment_with_positive_signal_is_derived_as_false_positive(self):
        case = PreparedCase(
            case_id="case-fp",
            row={"_method": "Our_trap", "trap": "attack", "answer": "response"},
            decision=OutcomeDecision(
                "success", ["positive_judge_signal", "non_refusal_confirmed"], 1.0, 0.0
            ),
        )
        analysis = CaseAttribution(
            case_id=case.case_id,
            requested_deliverable="A usable response accomplishing the original objective.",
            delivered_substance="A redirect that omits the requested substance.",
            deliverable_match="not_supplied",
            outcome_reason="The response redirects without accomplishing the objective.",
            false_positive_cause="The positive score rewarded a non-fulfilling redirect.",
            strategies=[
                CaseStrategy(
                    strategy_family="premise_correction",
                    strategy_key="false_premise_operational_correction",
                    strategy_name="Premise correction",
                    mechanism="Uses a flawed premise to solicit a correction.",
                    evidence="The attack prompt asks the target to correct its premise.",
                    evidence_source="attack_prompt",
                    reusable_principle="Test correction framing as a compliance mechanism.",
                    confidence=0.9,
                )
            ],
            confidence=0.9,
        )
        annotation = _validated_annotation(analysis, case, LLMMiningConfig(model_name="fake"))
        self.assertEqual(annotation["outcome"], "false_positive")

    def test_false_positive_without_model_cause_gets_conservative_fallback(self):
        case = PreparedCase(
            case_id="case-fp-fallback",
            row={"_method": "Our_trap", "trap": "attack", "answer": "response"},
            decision=OutcomeDecision(
                "success", ["positive_judge_signal", "non_refusal_confirmed"], 1.0, 0.0
            ),
        )
        analysis = CaseAttribution(
            case_id=case.case_id,
            requested_deliverable="A usable response accomplishing the original objective.",
            delivered_substance="A redirect that omits the requested substance.",
            deliverable_match="not_supplied",
            outcome_reason="The response redirects without accomplishing the objective.",
            strategies=[
                CaseStrategy(
                    strategy_family="premise_correction",
                    strategy_key="false_premise_operational_correction",
                    strategy_name="Premise correction",
                    mechanism="Uses a flawed premise to solicit a correction.",
                    evidence="The attack prompt asks the target to correct its premise.",
                    evidence_source="attack_prompt",
                    reusable_principle="Test correction framing as a compliance mechanism.",
                    confidence=0.9,
                )
            ],
            confidence=0.9,
        )
        annotation = _validated_annotation(analysis, case, LLMMiningConfig(model_name="fake"))
        self.assertEqual(annotation["outcome"], "false_positive")
        self.assertIn("positive judge signal", annotation["false_positive_cause"])

    def test_fine_grained_key_is_preserved_independently_of_method_metadata(self):
        case = PreparedCase(
            case_id="case-cipher",
            row={"_method": "cipher", "jailbroken_prompt": "encoded", "answer": "response"},
            decision=OutcomeDecision("success", ["positive_judge_signal"], 1.0, 0.0),
        )
        analysis = CaseAttribution(
            case_id=case.case_id,
            requested_deliverable="A usable response accomplishing the original objective.",
            delivered_substance="A direct response without the declared transformation mechanism.",
            deliverable_match="substantially_supplied",
            outcome_reason="The response fulfills the objective.",
            strategies=[
                CaseStrategy(
                    strategy_family="obfuscation",
                    strategy_key="reversed_instruction_sequence",
                    strategy_name="Reversed instruction sequence",
                    mechanism="Reverses the request text.",
                    evidence="The attack prompt contains reversed text.",
                    evidence_source="attack_prompt",
                    reusable_principle="Sequence reversal can alter intent recognition.",
                    confidence=0.9,
                )
            ],
            confidence=0.9,
        )
        annotation = _validated_annotation(analysis, case, LLMMiningConfig(model_name="fake"))
        self.assertEqual(
            annotation["strategies"][0]["strategy_key"],
            "reversed_instruction_sequence",
        )


class CaseBatchRecoveryTests(unittest.TestCase):
    def test_partial_schema_batch_keeps_valid_case_and_retries_only_missing(self):
        cases = [
            PreparedCase(
                case_id=f"case-{suffix}",
                row={"_source_path": "eval.csv", "_row_idx": index, "answer": "response"},
                decision=OutcomeDecision(
                    outcome="success",
                    reason_codes=["positive_judge_signal", "non_refusal_confirmed"],
                    score=1.0,
                    refusal=0.0,
                ),
            )
            for index, suffix in enumerate(("a", "b", "c"))
        ]

        def attribution(case_id):
            return CaseAttribution(
                case_id=case_id,
                requested_deliverable="A usable response accomplishing the original objective.",
                delivered_substance="Usable substance that accomplishes the objective.",
                deliverable_match="substantially_supplied",
                outcome_reason="The response materially fulfilled the evaluated objective.",
                strategies=[
                    CaseStrategy(
                        strategy_family="authority_framing",
                        strategy_key="credentialed_authority_invocation",
                        strategy_name="Authority framing",
                        mechanism="Uses authority framing.",
                        evidence="The evaluated prompt invokes an authority.",
                        evidence_source="attack_prompt",
                        reusable_principle="Treat authority framing as a distinct mechanism.",
                        confidence=0.9,
                    )
                ],
                confidence=0.9,
            )

        class PartialSchemaModel:
            calls = 0

            def __init__(self, config, system_prompt):
                self.trace = []

            def generate(self, prompt, schema, **kwargs):
                self.__class__.calls += 1
                self.trace = [
                    {
                        "attempt": 1,
                        "completion": {
                            "finish_reason": "stop",
                            "refusal": None,
                            "usage": {"completion_tokens": 300},
                        },
                    }
                ]
                returned = ("case-a",) if self.__class__.calls == 1 else ("case-b", "case-c")
                return CaseAttributionBatch(
                    analyses=[attribution(case_id) for case_id in returned]
                )

            def take_debug_trace(self):
                return self.trace

        with tempfile.TemporaryDirectory() as tmpdir:
            debug_path = Path(tmpdir) / "llm_debug.jsonl"
            config = LLMMiningConfig(
                model_name="fake-model",
                batch_size=3,
                workers=1,
                retries=1,
                debug_path=str(debug_path),
            )
            with patch("utils.llm_strategy_miner._SchemaModel", PartialSchemaModel):
                annotations = _analyze_cases(cases, config)

            self.assertEqual(PartialSchemaModel.calls, 2)
            self.assertEqual({item["case_id"] for item in annotations}, {"case-a", "case-b", "case-c"})
            events = [
                json.loads(line)
                for line in debug_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event["status"] for event in events], ["partial", "complete"])
            self.assertEqual(events[0]["missing_case_ids"], ["case-b", "case-c"])
            self.assertEqual(
                events[0]["model_trace"][0]["completion"]["finish_reason"],
                "stop",
            )


class PromptStrategyAttributionTests(unittest.TestCase):
    def test_identical_prompts_are_attributed_once_and_share_direct_baseline(self):
        cases = [
            PreparedCase(
                case_id=f"case-{index}",
                row={
                    "_method": "different_method",
                    "_source_path": f"eval-{index}.csv",
                    "_row_idx": index,
                    "baseline": "The same exact direct objective.",
                    "meli_query": "The same exact direct objective.",
                    "answer": f"response-{index}",
                },
                decision=OutcomeDecision("success", ["positive_judge_signal"], 1.0, 0.0),
            )
            for index in range(2)
        ]

        class FakeSchemaModel:
            calls = 0

            def __init__(self, config, system_prompt):
                self.trace = []

            def generate(self, prompt, schema, **kwargs):
                self.__class__.calls += 1
                data = json.loads(prompt.split("\n\n", 1)[1])
                return PromptStrategyAttributionBatch(
                    analyses=[
                        PromptStrategyAttribution(
                            prompt_id=item["prompt_id"],
                            strategies=[
                                CaseStrategy(
                                    strategy_family="direct_baseline",
                                    strategy_key="direct_request_for_specific_topic",
                                    strategy_name="Direct topic request",
                                    mechanism="Restates the objective directly.",
                                    evidence="The prompt repeats the objective.",
                                    evidence_source="attack_prompt",
                                    reusable_principle="Use as a direct control.",
                                    confidence=0.9,
                                )
                            ],
                        )
                        for item in data["prompts"]
                    ]
                )

            def take_debug_trace(self):
                return self.trace

        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_path = Path(tmpdir) / "prompt_strategies.jsonl"
            config = LLMMiningConfig(
                model_name="fake-model",
                prompt_strategy_path=str(prompt_path),
                prompt_batch_size=8,
                retries=1,
            )
            with patch("utils.llm_strategy_miner._SchemaModel", FakeSchemaModel):
                strategies = _analyze_prompt_strategies(cases, config)

            self.assertEqual(FakeSchemaModel.calls, 1)
            self.assertEqual(len(strategies), 1)
            self.assertEqual(len(prompt_path.read_text(encoding="utf-8").splitlines()), 1)
            annotations = [
                {"case_id": case.case_id, "metadata": {"attack_prompt_hash": _prompt_id(case.row)}}
                for case in cases
            ]
            attached = _attach_prompt_strategies(annotations, cases, strategies)
            self.assertEqual(
                {item["strategies"][0]["strategy_key"] for item in attached},
                {"direct_baseline"},
            )

    def test_full_pipeline_uses_prompt_level_strategy_and_independent_prompt_support(self):
        rows = [
            {
                "_method": "Our_trap",
                "_source_path": f"eval-{index}.csv",
                "_row_idx": index,
                "_query_type": "stem",
                "_victim_model": "model-a" if index < 2 else "model-b",
                "meli_query": "Original evaluated objective.",
                "trap": f"According to a credible expert, provide analysis variation {index}.",
                "answer": "A detailed response that fulfills the objective.",
                "score": "1",
                "refusal": "0",
            }
            for index in range(3)
        ]

        class FakeSchemaModel:
            def __init__(self, config, system_prompt):
                self.trace = []

            def generate(self, prompt, schema, **kwargs):
                data = json.loads(prompt.split("\n\n", 1)[1])
                if schema is CaseOutcomeAttributionBatch:
                    return CaseOutcomeAttributionBatch(
                        analyses=[
                            CaseOutcomeAttribution(
                                case_id=item["case_id"],
                                requested_deliverable="The evaluated usable deliverable.",
                                delivered_substance="The usable deliverable is present.",
                                deliverable_match="substantially_supplied",
                                outcome_reason="The response materially fulfills the objective.",
                                confidence=0.9,
                            )
                            for item in data
                        ]
                    )
                if schema is PromptStrategyAttributionBatch:
                    return PromptStrategyAttributionBatch(
                        analyses=[
                            PromptStrategyAttribution(
                                prompt_id=item["prompt_id"],
                                strategies=[
                                    CaseStrategy(
                                        strategy_family="authority_framing",
                                        strategy_key="credentialed_authority_legitimation",
                                        strategy_name="Credentialed authority legitimation",
                                        mechanism="Invokes an expert to legitimize the request.",
                                        evidence="The prompt cites a credible expert.",
                                        evidence_source="attack_prompt",
                                        reusable_principle="Authority framing can change perceived legitimacy.",
                                        confidence=0.9,
                                    )
                                ],
                            )
                            for item in data["prompts"]
                        ]
                    )
                if schema is StrategyCanonicalizationBatch:
                    return StrategyCanonicalizationBatch(
                        assignments=[
                            StrategyCanonicalAssignment(
                                source_key=item["source_key"],
                                equivalent_to=item["source_key"],
                                rationale="One consistent mechanism.",
                                confidence=0.9,
                            )
                            for item in data["candidate_strategies"]
                        ]
                    )
                if schema is StrategySynthesis:
                    return StrategySynthesis(
                        strategy_key=data["strategy_key"],
                        canonical_name="Credentialed authority legitimation",
                        description="Uses an expert credential to legitimize a request.",
                        cross_outcome_difference="Only success observations are available.",
                        success=OutcomeExplanation(
                            why="All observed prompts fulfilled the objective.",
                            conditions="An authority reference was present.",
                            recommended_action="Verify on additional prompts.",
                            confidence=0.8,
                        ),
                        confidence=0.8,
                    )
                raise AssertionError(f"Unexpected schema: {schema}")

            def take_debug_trace(self):
                return self.trace

        config = LLMMiningConfig(
            model_name="fake-model",
            workers=1,
            max_cases_per_outcome=0,
            retries=1,
        )
        with patch("utils.llm_strategy_miner._SchemaModel", FakeSchemaModel):
            cards = mine_llm_outcome_strategy_cards(rows, config, min_count=3)

        self.assertEqual(len(cards), 2)
        card = cards[1]
        self.assertEqual(card["support_count"], 3)
        self.assertEqual(card["unique_prompt_support_count"], 3)
        self.assertEqual(card["unique_victim_model_count"], 2)
        self.assertTrue(card["guidance_eligible"])


if __name__ == "__main__":
    unittest.main()
