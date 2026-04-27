from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"


class ScenarioCategory(str, Enum):
    AMMONIA_SCP = "ammonia_scp"
    BIO_UREA_SCP = "bio_urea_scp"


class FeedstockType(str, Enum):
    FORMATE = "formate"       # CO2 electrolysis -> formate -> fermentation (existing)
    H2_CO2 = "h2_co2"         # Water electrolysis -> H2, purchased CO2 -> autotrophic fermentation
    METHANOL = "methanol"     # Purchased methanol -> methylotrophic fermentation


class ElectricityCase(str, Enum):
    US_GRID = "us_grid"
    RENEWABLE = "renewable"


class CO2Source(str, Enum):
    """Origin of the CO₂ feedstock for formate electrolysis.

    The choice determines both the supply-chain emission burden (the process cost of
    obtaining the CO₂) and whether biogenic carbon accounting rules apply.
    """
    BIOGENIC_WASTE = "biogenic_waste"  # Corn ethanol or other fermentation off-gas — biogenic, near-zero burden
    FOSSIL_CAPTURE  = "fossil_capture"  # Post-combustion CCS from industrial point source
    DAC             = "dac"             # Direct air capture — highest energy burden


class AmmoniaRecoveryMethod(str, Enum):
    VACUUM_STRIPPING = "vacuum_stripping"
    MEMBRANE = "membrane"
    AIR_STRIPPING = "air_stripping"
    STRUVITE_MAP = "struvite_map"      # MgNH4PO4 precipitation — low energy, produces struvite fertilizer
    MAP_FERTILIZER = "map_fertilizer"  # Membrane strip NH3, absorb in H3PO4 → NH4H2PO4 (11-52-0 fertilizer)


class UreaRecoveryMethod(str, Enum):
    EVAPORATION = "evaporation_crystallization"        # Single-effect evaporation + crystallization
    HYBRID = "membrane_polishing"                      # (kept for backward compat)
    MVR_CRYSTALLIZATION = "mvr_crystallization"        # Mechanical vapor recompression — lower steam, higher electricity


@dataclass(frozen=True)
class DataRecord:
    key: str
    value: float
    unit: str
    source_name: str
    source_url: str
    source_year: int
    confidence: str
    notes: str


def _load_csv_records(filename: str) -> Dict[str, DataRecord]:
    path = DATA_DIR / filename
    records: Dict[str, DataRecord] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records[row["key"]] = DataRecord(
                key=row["key"],
                value=float(row["value"]),
                unit=row["unit"],
                source_name=row["source_name"],
                source_url=row["source_url"],
                source_year=int(row["source_year"]),
                confidence=row["confidence"],
                notes=row["notes"],
            )
    return records


def load_source_metadata() -> Dict[str, object]:
    path = DATA_DIR / "source_metadata.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _values(records: Mapping[str, DataRecord]) -> Dict[str, float]:
    return {key: record.value for key, record in records.items()}


@dataclass(frozen=True)
class EconomicInputs:
    electricity_price_usd_per_kwh: float
    water_price_usd_per_kg: float
    wastewater_price_usd_per_kg: float
    naoh_price_usd_per_kg: float
    h2so4_price_usd_per_kg: float
    h3po4_price_usd_per_kg: float
    mgcl2_price_usd_per_kg: float
    co2_price_usd_per_kg: float
    ammonia_market_price_usd_per_kg: float
    urea_market_price_usd_per_kg: float
    scp_market_price_usd_per_kg: float
    operator_loaded_wage_usd_per_h: float
    maintenance_factor: float
    admin_overhead_factor: float
    lang_factor: float
    working_capital_fraction: float
    discount_rate: float
    plant_life_years: float
    electrolyzer_installed_cost_usd_per_kw: float
    electrolyzer_stack_replacement_fraction: float
    electrolyzer_stack_life_years: float
    membrane_cost_usd_per_m2: float
    steam_price_usd_per_kg: float
    struvite_market_price_usd_per_kg: float       # Bulk struvite slow-release fertilizer benchmark
    map_fert_market_price_usd_per_kg: float       # MAP fertilizer (11-52-0) commodity benchmark
    methanol_price_usd_per_kg: float              # Industrial methanol commodity price

    @classmethod
    def from_records(cls, records: Mapping[str, DataRecord]) -> "EconomicInputs":
        values = _values(records)
        return cls(
            electricity_price_usd_per_kwh=values["electricity_price_usd_per_kwh"],
            water_price_usd_per_kg=values["water_price_usd_per_kg"],
            wastewater_price_usd_per_kg=values["wastewater_price_usd_per_kg"],
            naoh_price_usd_per_kg=values["naoh_price_usd_per_kg"],
            h2so4_price_usd_per_kg=values["h2so4_price_usd_per_kg"],
            h3po4_price_usd_per_kg=values["h3po4_price_usd_per_kg"],
            mgcl2_price_usd_per_kg=values["mgcl2_price_usd_per_kg"],
            co2_price_usd_per_kg=values["co2_price_usd_per_kg"],
            ammonia_market_price_usd_per_kg=values["ammonia_market_price_usd_per_kg"],
            urea_market_price_usd_per_kg=values["urea_market_price_usd_per_kg"],
            scp_market_price_usd_per_kg=values["scp_market_price_usd_per_kg"],
            operator_loaded_wage_usd_per_h=values["operator_loaded_wage_usd_per_h"],
            maintenance_factor=values["maintenance_factor"],
            admin_overhead_factor=values["admin_overhead_factor"],
            lang_factor=values["lang_factor"],
            working_capital_fraction=values["working_capital_fraction"],
            discount_rate=values["discount_rate"],
            plant_life_years=values["plant_life_years"],
            electrolyzer_installed_cost_usd_per_kw=values["electrolyzer_installed_cost_usd_per_kw"],
            electrolyzer_stack_replacement_fraction=values["electrolyzer_stack_replacement_fraction"],
            electrolyzer_stack_life_years=values["electrolyzer_stack_life_years"],
            membrane_cost_usd_per_m2=values["membrane_cost_usd_per_m2"],
            steam_price_usd_per_kg=values["steam_price_usd_per_kg"],
            struvite_market_price_usd_per_kg=values["struvite_market_price_usd_per_kg"],
            map_fert_market_price_usd_per_kg=values["map_fert_market_price_usd_per_kg"],
            methanol_price_usd_per_kg=values["methanol_price_usd_per_kg"],
        )


@dataclass(frozen=True)
class LCAFactors:
    grid_electricity_kgco2e_per_kwh: float
    renewable_electricity_kgco2e_per_kwh: float
    water_supply_kgco2e_per_kg: float
    wastewater_treatment_kgco2e_per_kg: float
    steam_kgco2e_per_kg: float
    naoh_kgco2e_per_kg: float
    h2so4_kgco2e_per_kg: float
    h3po4_kgco2e_per_kg: float
    mgcl2_kgco2e_per_kg: float
    # CO₂ supply-chain burden: three source categories
    co2_supply_kgco2e_per_kg: float         # Fossil point-source capture (post-combustion CCS)
    biogenic_co2_kgco2e_per_kg: float       # Waste biogenic CO₂ (corn ethanol off-gas): only compression+transport burden
    dac_co2_kgco2e_per_kg: float            # Direct air capture — highest energy penalty
    membrane_kgco2e_per_m2: float
    # Methanol supply chain
    methanol_kgco2e_per_kg: float           # Cradle-to-gate GWP of purchased methanol feedstock
    # Biogenic carbon and system-expansion credits
    scp_carbon_fraction: float              # Mass fraction of C in dry SCP (≈ 0.47)
    scp_displacement_kgco2e_per_kg: float   # GWP of conventional protein displaced per kg SCP (soy meal, no LUC)

    @classmethod
    def from_records(cls, records: Mapping[str, DataRecord]) -> "LCAFactors":
        values = _values(records)
        return cls(**values)

    def electricity_factor(self, electricity_case: ElectricityCase) -> float:
        if electricity_case == ElectricityCase.RENEWABLE:
            return self.renewable_electricity_kgco2e_per_kwh
        return self.grid_electricity_kgco2e_per_kwh

    def co2_source_factor(self, co2_source: CO2Source) -> float:
        """Supply-chain emission factor for the chosen CO₂ feedstock."""
        if co2_source == CO2Source.BIOGENIC_WASTE:
            return self.biogenic_co2_kgco2e_per_kg
        if co2_source == CO2Source.DAC:
            return self.dac_co2_kgco2e_per_kg
        return self.co2_supply_kgco2e_per_kg  # FOSSIL_CAPTURE


@dataclass(frozen=True)
class TechnologyInputs:
    capacity_factor: float
    electrolyzer_energy_kwh_per_mol_formate: float
    formate_faradaic_efficiency: float
    h2_faradaic_efficiency: float
    water_kg_per_mol_formate: float
    co2_kg_per_kg_formate: float
    formate_to_ammonia_kg_per_kg: float
    scp_to_ammonia_kg_per_kg: float
    formate_to_urea_kg_per_kg: float
    scp_to_urea_kg_per_kg: float
    nitrogen_mass_fraction_scp: float
    carbon_mass_fraction_scp: float
    ammonia_recovery_efficiency: float
    urea_recovery_efficiency: float
    vacuum_energy_kwh_per_kg_ammonia: float
    vacuum_naoh_mol_per_mol_nh3: float
    membrane_energy_kwh_per_kg_ammonia: float
    membrane_flux_kg_per_m2_h: float
    air_stripping_energy_kwh_per_kg_ammonia: float
    evaporation_kwh_per_kg_urea: float
    evaporation_steam_kg_per_kg_urea: float
    ammonia_productivity_kg_per_m3_h: float
    urea_productivity_kg_per_m3_h: float
    broth_water_kg_per_kg_ammonia: float
    broth_water_kg_per_kg_urea: float
    scp_cake_moisture: float
    scp_product_moisture: float
    scp_drying_kwh_per_kg_water: float
    centrifuge_kwh_per_m3_broth: float
    agitation_aeration_kwh_per_m3_h: float
    air_compression_kwh_per_kg_n2: float
    bioreactor_base_cost_usd: float
    bioreactor_reference_m3: float
    recovery_base_cost_usd: float
    urea_recovery_base_cost_usd: float
    scp_processing_base_cost_usd: float
    equipment_scaling_exponent: float
    # --- Struvite / MAP precipitation (AmmoniaRecoveryMethod.STRUVITE_MAP) ---
    map_electricity_kwh_per_kg_nh3: float
    map_mg_mol_per_mol_nh3: float
    map_phosphate_mol_per_mol_nh3: float
    map_recovery_efficiency: float
    struvite_kg_per_kg_nh3: float
    map_settler_base_cost_usd: float
    # --- MAP Fertilizer route (AmmoniaRecoveryMethod.MAP_FERTILIZER) ---
    # Membrane strip NH3, absorb into H3PO4 trap → NH4H2PO4 (monoammonium phosphate, 11-52-0)
    map_fert_h3po4_mol_per_mol_nh3: float      # Slight excess H3PO4 for complete neutralization
    map_fert_nh4h2po4_kg_per_kg_nh3: float     # MAP fertilizer yield per kg NH3 (fixed by stoichiometry)
    map_fert_granulation_base_cost_usd: float   # Neutralization tank + spray dryer base cost
    # --- MVR crystallization (UreaRecoveryMethod.MVR_CRYSTALLIZATION) ---
    mvr_kwh_per_kg_urea: float
    mvr_steam_kg_per_kg_urea: float
    mvr_base_cost_usd: float
    # --- H2/CO2 autotrophic feedstock (FeedstockType.H2_CO2) ---
    h2_to_ammonia_kg_per_kg: float              # kg H2 consumed per kg NH3 produced
    h2_to_urea_kg_per_kg: float                 # kg H2 consumed per kg urea produced
    co2_to_biomass_h2co2_kg_per_kg: float       # kg CO2 required per kg SCP (CBB cycle)
    scp_to_ammonia_h2co2_kg_per_kg: float       # kg SCP co-product per kg NH3 (autotrophic)
    scp_to_urea_h2co2_kg_per_kg: float          # kg SCP co-product per kg urea (autotrophic)
    electrolyzer_energy_kwh_per_mol_h2: float   # PEM water electrolysis energy per mol H2
    water_kg_per_mol_h2: float                  # Water stoichiometry for H2 electrolysis
    ammonia_productivity_h2co2_kg_per_m3_h: float  # Volumetric productivity on H2/CO2
    urea_productivity_h2co2_kg_per_m3_h: float     # Volumetric productivity on H2/CO2
    # --- Methanol methylotrophic feedstock (FeedstockType.METHANOL) ---
    methanol_to_ammonia_kg_per_kg: float        # kg methanol consumed per kg NH3 produced
    methanol_to_urea_kg_per_kg: float           # kg methanol consumed per kg urea produced
    scp_to_ammonia_methanol_kg_per_kg: float    # kg SCP co-product per kg NH3 (methylotrophic)
    scp_to_urea_methanol_kg_per_kg: float       # kg SCP co-product per kg urea (methylotrophic)
    ammonia_productivity_methanol_kg_per_m3_h: float  # Volumetric productivity on methanol
    urea_productivity_methanol_kg_per_m3_h: float     # Volumetric productivity on methanol

    @classmethod
    def from_records(cls, records: Mapping[str, DataRecord]) -> "TechnologyInputs":
        return cls(**_values(records))


@dataclass(frozen=True)
class ScenarioConfig:
    category: ScenarioCategory
    annual_primary_product_tpy: float
    feedstock_type: FeedstockType = FeedstockType.FORMATE
    # Design-basis electricity is renewable (PPA wind/solar, IPCC AR5 WGIII
    # Annex III lifecycle factor). The grid case remains available as a
    # downside comparator and appears in the LCA waterfall figure.
    electricity_case: ElectricityCase = ElectricityCase.RENEWABLE
    ammonia_recovery_method: AmmoniaRecoveryMethod = AmmoniaRecoveryMethod.VACUUM_STRIPPING
    urea_recovery_method: UreaRecoveryMethod = UreaRecoveryMethod.EVAPORATION
    use_scp_credit: bool = True
    use_h2_credit: bool = False
    h2_credit_usd_per_kg: float = 0.0
    co2_credit_usd_per_kg: float = 0.0
    # --- LCA options ---
    co2_source: CO2Source = CO2Source.BIOGENIC_WASTE  # Corn ethanol off-gas is the design basis
    use_biogenic_carbon_credit: bool = True   # Credit biogenic C stored in SCP at cradle-to-gate boundary
    use_scp_displacement_credit: bool = False  # System expansion: credit for displacing soy meal protein
    user_overrides: Dict[str, float] = field(default_factory=dict)

    @property
    def annual_primary_product_kg(self) -> float:
        return self.annual_primary_product_tpy * 1000.0


def load_datasets() -> Dict[str, Dict[str, DataRecord]]:
    return {
        "economic": _load_csv_records("economic_inputs.csv"),
        "lca": _load_csv_records("lca_factors.csv"),
        "technology": _load_csv_records("technology_inputs.csv"),
    }


def build_default_inputs(
    overrides: Optional[Mapping[str, float]] = None,
) -> tuple[EconomicInputs, LCAFactors, TechnologyInputs, Dict[str, Dict[str, DataRecord]]]:
    datasets = load_datasets()
    overrides = overrides or {}

    def merged(name: str) -> Dict[str, DataRecord]:
        out = dict(datasets[name])
        for key, value in overrides.items():
            if key in out:
                record = out[key]
                out[key] = DataRecord(
                    key=record.key,
                    value=float(value),
                    unit=record.unit,
                    source_name="user_override",
                    source_url="",
                    source_year=load_source_metadata()["base_year"],
                    confidence="user",
                    notes=f"User override applied to {record.key}",
                )
        return out

    econ_records = merged("economic")
    lca_records = merged("lca")
    tech_records = merged("technology")
    return (
        EconomicInputs.from_records(econ_records),
        LCAFactors.from_records(lca_records),
        TechnologyInputs.from_records(tech_records),
        {"economic": econ_records, "lca": lca_records, "technology": tech_records},
    )


def default_scenarios() -> List[ScenarioConfig]:
    capacities = [100.0, 1_000.0, 10_000.0]
    scenarios: List[ScenarioConfig] = []
    for capacity in capacities:
        scenarios.append(
            ScenarioConfig(
                category=ScenarioCategory.AMMONIA_SCP,
                annual_primary_product_tpy=capacity,
            )
        )
        scenarios.append(
            ScenarioConfig(
                category=ScenarioCategory.BIO_UREA_SCP,
                annual_primary_product_tpy=capacity,
            )
        )
    return scenarios


def record_table(records: Mapping[str, DataRecord]) -> List[Dict[str, object]]:
    return [
        {
            "key": record.key,
            "value": record.value,
            "unit": record.unit,
            "source": record.source_name,
            "source_url": record.source_url,
            "year": record.source_year,
            "confidence": record.confidence,
            "notes": record.notes,
            "is_override": record.source_name == "user_override",
        }
        for record in records.values()
    ]


def flatten_record_tables(dataset_map: Mapping[str, Mapping[str, DataRecord]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset_name, records in dataset_map.items():
        for row in record_table(records):
            row["dataset"] = dataset_name
            rows.append(row)
    return rows


def records_by_key(dataset_map: Mapping[str, Mapping[str, DataRecord]]) -> Dict[str, Dict[str, object]]:
    """Return a flat lookup of assumption metadata keyed by input name."""
    out: Dict[str, Dict[str, object]] = {}
    for dataset_name, records in dataset_map.items():
        for key, record in records.items():
            out[key] = {
                "dataset": dataset_name,
                "key": record.key,
                "value": record.value,
                "unit": record.unit,
                "source": record.source_name,
                "source_url": record.source_url,
                "year": record.source_year,
                "confidence": record.confidence,
                "notes": record.notes,
                "is_override": record.source_name == "user_override",
            }
    return out


def iter_overrideable_keys() -> Iterable[str]:
    for records in load_datasets().values():
        for key in records:
            yield key
