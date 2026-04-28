import unittest

from formate_biorefinery_model.config import (
    AmmoniaRecoveryMethod,
    CO2Source,
    ElectricityCase,
    FeedstockType,
    ScenarioCategory,
    ScenarioConfig,
)
from formate_biorefinery_model.run_scenarios import evaluate_scenario


class FertilizerRevenueAccountingTest(unittest.TestCase):
    def test_struvite_revenue_is_not_double_counted_in_npv_cashflow(self) -> None:
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
        result = evaluate_scenario(cfg)
        metrics = result.tea.metrics
        credits = result.tea.credits_usd_per_y

        fertilizer_credit = credits["struvite_credit"] + credits["map_fert_credit"]
        expected_revenue = (
            metrics["benchmark_primary_revenue_usd_per_y"]
            + sum(credits.values())
            - fertilizer_credit
        )

        self.assertGreater(fertilizer_credit, 0.0)
        self.assertAlmostEqual(metrics["benchmark_total_revenue_usd_per_y"], expected_revenue)
        self.assertAlmostEqual(metrics["npv_usd"] / 1e6, 52.12, places=1)


if __name__ == "__main__":
    unittest.main()
