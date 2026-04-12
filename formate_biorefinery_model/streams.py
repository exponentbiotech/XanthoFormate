from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class StreamLedger:
    mass_kg_per_y: Dict[str, float] = field(default_factory=dict)
    electricity_kwh_per_y: Dict[str, float] = field(default_factory=dict)
    steam_kg_per_y: Dict[str, float] = field(default_factory=dict)
    area_m2: Dict[str, float] = field(default_factory=dict)

    def add_mass(self, name: str, value: float) -> None:
        self.mass_kg_per_y[name] = self.mass_kg_per_y.get(name, 0.0) + value

    def add_electricity(self, name: str, value: float) -> None:
        self.electricity_kwh_per_y[name] = self.electricity_kwh_per_y.get(name, 0.0) + value

    def add_steam(self, name: str, value: float) -> None:
        self.steam_kg_per_y[name] = self.steam_kg_per_y.get(name, 0.0) + value

    def add_area(self, name: str, value: float) -> None:
        self.area_m2[name] = self.area_m2.get(name, 0.0) + value

    @property
    def total_electricity_kwh_per_y(self) -> float:
        return sum(self.electricity_kwh_per_y.values())

    @property
    def total_steam_kg_per_y(self) -> float:
        return sum(self.steam_kg_per_y.values())


def water_removed_from_scp(
    dry_scp_kg: float,
    cake_moisture: float,
    final_moisture: float,
) -> float:
    water_cake = dry_scp_kg * cake_moisture / max(1e-9, 1.0 - cake_moisture)
    water_final = dry_scp_kg * final_moisture / max(1e-9, 1.0 - final_moisture)
    return max(0.0, water_cake - water_final)
