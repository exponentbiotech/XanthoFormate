"""Tests for the LLM-backed chat module.

We do not exercise the actual Groq HTTP call here — those tests would require
network access and a paid quota. Instead we verify:

* The deterministic fallback is invoked when no API key is present so the
  chat keeps working in offline / unconfigured environments.
* ``_build_messages`` packages the system prompt, locked snapshot, the
  ``tea.py`` / ``lca.py`` source excerpts, prior chat history, and the
  current user question in the right order.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from formate_biorefinery_model.app_support import comprehensive_chat_context
from formate_biorefinery_model.config import (
    AmmoniaRecoveryMethod,
    CO2Source,
    ElectricityCase,
    FeedstockType,
    ScenarioCategory,
    ScenarioConfig,
)
from formate_biorefinery_model.model_chat import (
    _build_messages,
    answer_question,
    is_llm_available,
)
from formate_biorefinery_model.run_scenarios import evaluate_scenario


def _struvite_snapshot():
    cfg = ScenarioConfig(
        category=ScenarioCategory.AMMONIA_SCP,
        annual_primary_product_tpy=1_000.0,
        feedstock_type=FeedstockType.FORMATE,
        electricity_case=ElectricityCase.RENEWABLE,
        co2_source=CO2Source.BIOGENIC_WASTE,
        ammonia_recovery_method=AmmoniaRecoveryMethod.STRUVITE_MAP,
        use_biogenic_carbon_credit=True,
        use_scp_displacement_credit=False,
    )
    active = evaluate_scenario(cfg)
    return comprehensive_chat_context(cfg, {}, active, active.source_rows)


class ModelChatTest(unittest.TestCase):
    def test_is_llm_available_requires_a_key(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(is_llm_available(api_key=None))
            self.assertFalse(is_llm_available(api_key=""))
            self.assertTrue(is_llm_available(api_key="gsk-test"))
        with mock.patch.dict(os.environ, {"GROQ_API_KEY": "gsk-env"}, clear=True):
            self.assertTrue(is_llm_available(api_key=None))

    def test_no_key_falls_back_to_deterministic_interpreter(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            answer = answer_question(
                "Most reasonable scenario to become profitable",
                _struvite_snapshot(),
                history=[],
                api_key=None,
            )
        self.assertIn("USD 52.1M", answer)
        self.assertIn("The strongest route", answer)

    def test_build_messages_includes_code_and_locked_results(self) -> None:
        snapshot = _struvite_snapshot()
        messages = _build_messages(
            "How is NPV calculated?",
            snapshot,
            history=[
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier reply"},
            ],
        )

        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "system")
        grounding = messages[1]["content"]
        self.assertIn("LIVE_RESULTS", grounding)
        self.assertIn("SOURCE_CODE", grounding)
        self.assertIn("def evaluate_tea", grounding)
        self.assertIn("def evaluate_lca", grounding)
        self.assertIn("Active scenario", grounding)
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "How is NPV calculated?")
        self.assertEqual(messages[-2]["role"], "assistant")
        self.assertEqual(messages[-3]["role"], "user")


if __name__ == "__main__":
    unittest.main()
