from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import shap  # type: ignore

    SHAP_AVAILABLE = True
except Exception:  # pragma: no cover
    shap = None
    SHAP_AVAILABLE = False


@dataclass
class FeatureImportance:
    feature: str
    importance: float


class ExplainabilityHelper:
    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = list(feature_names)

    def export_feature_importance(self, model, output_path: Path) -> pd.DataFrame:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            raise ValueError("Model does not expose feature_importances_ or coef_.")
        df = pd.DataFrame({"feature": self.feature_names, "importance": importances})
        df.sort_values("importance", ascending=False, inplace=True)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    def compute_shap_values(self, model, X: np.ndarray, scaler=None) -> Optional[np.ndarray]:
        if not SHAP_AVAILABLE:
            return None
        if scaler is not None:
            X = scaler.transform(X)
        explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") else shap.KernelExplainer(model.predict, X[:100])
        shap_values = explainer.shap_values(X)
        return shap_values

    def save_shap_summary(self, model, X: np.ndarray, output_path: Path, scaler=None) -> bool:
        if not SHAP_AVAILABLE:
            return False
        shap_values = self.compute_shap_values(model, X, scaler=scaler)
        if shap_values is None:
            return False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shap.summary_plot(shap_values, X, feature_names=self.feature_names, show=False)
        import matplotlib.pyplot as plt

        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        return True
