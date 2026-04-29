import unittest

from formate_biorefinery_model.app_support import comprehensive_chat_context
from formate_biorefinery_model.config import (
    AmmoniaRecoveryMethod,
    CO2Source,
    ElectricityCase,
    FeedstockType,
    ScenarioCategory,
    ScenarioConfig,
)
from formate_biorefinery_model.model_interpreter import answer_model_question
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


class ModelInterpreterTest(unittest.TestCase):
    def test_profitability_answer_uses_model_values_not_old_llm_numbers(self) -> None:
        answer = answer_model_question("Most reasonable scenario to become profitable", _struvite_snapshot())

        self.assertIn("USD 52.1M", answer)
        self.assertIn("USD 607.2M", answer)
        self.assertNotIn("USD 120", answer)
        self.assertNotIn("USD 144", answer)
        self.assertIn("The strongest route", answer)
        self.assertNotIn("no LLM", answer)


if __name__ == "__main__":
    unittest.main()
