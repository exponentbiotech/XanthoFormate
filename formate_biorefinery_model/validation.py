from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config import ScenarioCategory, build_default_inputs
from .run_scenarios import evaluate_scenario, run_sensitivity_cases
from .config import ScenarioConfig


@dataclass
class ValidationCheck:
    name: str
    passed: bool
    details: str


def run_validation_suite() -> List[ValidationCheck]:
    _, _, technology, _ = build_default_inputs()
    checks: List[ValidationCheck] = []

    for category in (ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP):
        result = evaluate_scenario(ScenarioConfig(category=category, annual_primary_product_tpy=1_000.0))
        closure = result.foreground.closure
        checks.append(
            ValidationCheck(
                name=f"{category.value}_carbon_balance",
                passed=abs(closure["carbon_balance_error_kg"]) < 1e-6,
                details=f"carbon balance error = {closure['carbon_balance_error_kg']:.3e} kg/y",
            )
        )
        checks.append(
            ValidationCheck(
                name=f"{category.value}_nitrogen_balance",
                passed=abs(closure["nitrogen_balance_error_kg"]) < 1e-6,
                details=f"nitrogen balance error = {closure['nitrogen_balance_error_kg']:.3e} kg/y",
            )
        )

    ammonia = evaluate_scenario(ScenarioConfig(category=ScenarioCategory.AMMONIA_SCP, annual_primary_product_tpy=1_000.0))
    ammonia_formate_ratio = ammonia.foreground.ledger.mass_kg_per_y["formate"] / ammonia.foreground.sellable_primary_product_kg
    expected_ammonia_ratio = technology.formate_to_ammonia_kg_per_kg / technology.ammonia_recovery_efficiency
    checks.append(
        ValidationCheck(
            name="ammonia_formate_ratio",
            passed=abs(ammonia_formate_ratio - expected_ammonia_ratio) < 1e-6,
            details=f"calculated = {ammonia_formate_ratio:.6f}, expected = {expected_ammonia_ratio:.6f}",
        )
    )

    urea = evaluate_scenario(ScenarioConfig(category=ScenarioCategory.BIO_UREA_SCP, annual_primary_product_tpy=1_000.0))
    urea_formate_ratio = urea.foreground.ledger.mass_kg_per_y["formate"] / urea.foreground.sellable_primary_product_kg
    expected_urea_ratio = technology.formate_to_urea_kg_per_kg / technology.urea_recovery_efficiency
    checks.append(
        ValidationCheck(
            name="urea_formate_ratio",
            passed=abs(urea_formate_ratio - expected_urea_ratio) < 1e-6,
            details=f"calculated = {urea_formate_ratio:.6f}, expected = {expected_urea_ratio:.6f}",
        )
    )

    for category in (ScenarioCategory.AMMONIA_SCP, ScenarioCategory.BIO_UREA_SCP):
        low = evaluate_scenario(ScenarioConfig(category=category, annual_primary_product_tpy=100.0))
        mid = evaluate_scenario(ScenarioConfig(category=category, annual_primary_product_tpy=1_000.0))
        high = evaluate_scenario(ScenarioConfig(category=category, annual_primary_product_tpy=10_000.0))
        checks.append(
            ValidationCheck(
                name=f"{category.value}_scale_economy",
                passed=(
                    low.tea.metrics["gross_primary_lcox_usd_per_kg"]
                    >= mid.tea.metrics["gross_primary_lcox_usd_per_kg"]
                    >= high.tea.metrics["gross_primary_lcox_usd_per_kg"]
                ),
                details=(
                    f"gross LCOX values = "
                    f"{low.tea.metrics['gross_primary_lcox_usd_per_kg']:.3f}, "
                    f"{mid.tea.metrics['gross_primary_lcox_usd_per_kg']:.3f}, "
                    f"{high.tea.metrics['gross_primary_lcox_usd_per_kg']:.3f}"
                ),
            )
        )

    sensitivity = run_sensitivity_cases(
        ScenarioConfig(category=ScenarioCategory.BIO_UREA_SCP, annual_primary_product_tpy=1_000.0),
        "electricity_price_usd_per_kwh",
    )
    low_cost = sensitivity[0].tea.metrics["net_primary_lcox_usd_per_kg"]
    base_cost = sensitivity[1].tea.metrics["net_primary_lcox_usd_per_kg"]
    high_cost = sensitivity[2].tea.metrics["net_primary_lcox_usd_per_kg"]
    checks.append(
        ValidationCheck(
            name="electricity_sensitivity_direction",
            passed=low_cost < base_cost < high_cost,
            details=f"net LCOX values = {low_cost:.3f}, {base_cost:.3f}, {high_cost:.3f}",
        )
    )
    return checks


def format_validation_report(checks: List[ValidationCheck]) -> str:
    lines = []
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"{status:>4}  {check.name:35s}  {check.details}")
    return "\n".join(lines)
