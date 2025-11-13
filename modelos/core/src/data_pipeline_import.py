from __future__ import annotations

"""
Compat layer: reexport domain-specific loaders from data_pipeline package.
"""

from data_pipeline import (
    FinanceImportConfig,
    LogisticsImportConfig,
    HealthImportConfig,
    PhysicsImportConfig,
    load_finance_dataset,
    load_logistics_dataset,
    load_health_dataset,
    load_physics_dataset,
)

__all__ = [
    "FinanceImportConfig",
    "LogisticsImportConfig",
    "HealthImportConfig",
    "PhysicsImportConfig",
    "load_finance_dataset",
    "load_logistics_dataset",
    "load_health_dataset",
    "load_physics_dataset",
]
