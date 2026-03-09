from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _risk_bias(level: str) -> float:
    key = str(level or "").strip().lower()
    if key in {"dispersion", "stable", "estavel"}:
        return 0.12
    if key in {"transition", "transicao"}:
        return -0.02
    if key in {"stress", "estresse"}:
        return -0.16
    return -0.05


def _pbo_bonus(verdict: str) -> float:
    key = str(verdict or "").strip().lower()
    if key == "robusto":
        return 0.08
    if key == "aceitavel":
        return 0.03
    return -0.06


def _alert_penalty(status: str, alert_count: int) -> float:
    key = str(status or "").strip().lower()
    if key == "fail":
        return 0.18 + 0.01 * max(0, int(alert_count) - 1)
    if key == "warn":
        return 0.07 + 0.01 * max(0, int(alert_count) - 2)
    return 0.0


def _confidence_level(score: float) -> str:
    if score >= 0.72:
        return "alta"
    if score >= 0.52:
        return "média"
    return "baixa"


@dataclass(frozen=True)
class ModeConfidenceDecision:
    recommended_mode: str
    confidence_score: float
    confidence_level: str
    attack_score: float
    protection_score: float
    scenario_bad: str
    scenario_base: str
    scenario_good: str
    reasons: list[str]
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_attack_vs_protection(
    *,
    structural_risk_level: str,
    structural_confidence_score: float | None,
    vigilance_status: str,
    vigilance_alert_count: int,
    pbo_verdict: str,
    attack_underperform_prob_63: float | None,
    attack_top3_retention: float | None,
    attack_drawdown: float | None,
    protection_drawdown: float | None,
    execution_winner: str | None,
) -> ModeConfidenceDecision:
    struct_conf = _clip01(float(structural_confidence_score or 0.0))
    risk_bias = _risk_bias(structural_risk_level)
    pbo = _pbo_bonus(pbo_verdict)
    alert_penalty = _alert_penalty(vigilance_status, int(vigilance_alert_count))
    underperf = _clip01(float(attack_underperform_prob_63 or 0.5))
    retention = _clip01(float(attack_top3_retention or 0.0))
    attack_dd = abs(float(attack_drawdown)) if attack_drawdown is not None else 0.80
    protect_dd = abs(float(protection_drawdown)) if protection_drawdown is not None else 0.60

    attack_score = (
        0.46
        + 0.24 * struct_conf
        + risk_bias
        + pbo
        - 0.35 * max(0.0, underperf - 0.50)
        - 0.24 * max(0.0, 0.45 - retention)
        - 0.14 * max(0.0, attack_dd - 0.65)
        - alert_penalty
    )
    if str(execution_winner or "").strip().lower() in {"attack", "modo ataque"}:
        attack_score += 0.04

    protection_score = (
        0.50
        + 0.18 * (1.0 - struct_conf)
        - risk_bias * 0.35
        + pbo * 0.6
        + 0.10 * max(0.0, 0.75 - protect_dd)
        + 0.08 * max(0.0, 0.60 - underperf)
        + 0.18 * max(0.0, 0.45 - retention)
        + 0.12 * max(0.0, attack_dd - protect_dd)
    )
    if str(vigilance_status or "").strip().lower() in {"warn", "fail"}:
        protection_score += 0.05
    if str(execution_winner or "").strip().lower() in {"protection", "main_guard", "modo principal com guarda"}:
        protection_score += 0.08

    attack_score = _clip01(attack_score)
    protection_score = _clip01(protection_score)
    if attack_score > protection_score:
        recommended_mode = "ataque"
        confidence_score = _clip01(0.55 * attack_score + 0.45 * (attack_score - protection_score + 0.5))
    else:
        recommended_mode = "proteção"
        confidence_score = _clip01(0.55 * protection_score + 0.45 * (protection_score - attack_score + 0.5))

    reasons: list[str] = []
    if retention < 0.35:
        reasons.append("O ganho continua dependente dos nomes cripto mais fortes.")
    if underperf > 0.55:
        reasons.append("No curto e médio prazo, o modo agressivo ainda pode ficar abaixo do benchmark por bastante tempo.")
    if str(vigilance_status or "").strip().lower() in {"warn", "fail"}:
        reasons.append("Os alertas diários pedem leitura mais defensiva.")
    if str(structural_risk_level or "").strip().lower() in {"dispersion", "stable", "estavel"} and struct_conf >= 0.55:
        reasons.append("A leitura estrutural está mais limpa, então ainda existe espaço para ataque.")
    if not reasons:
        reasons.append("Os sinais estão mistos; a recomendação atual favorece o lado com menor fragilidade operacional.")

    confidence_level = _confidence_level(confidence_score)
    scenario_bad = (
        "Mercado mais fraco que o esperado: proteção tende a sofrer menos e ataque pode ficar para trás."
    )
    scenario_base = (
        "Cenário central: manter o modo recomendado e revisar só quando a leitura estrutural ou a vigilância mudarem."
    )
    scenario_good = (
        "Mercado mais limpo e com breadth forte: ataque volta a ter mais chance de abrir vantagem."
    )

    return ModeConfidenceDecision(
        recommended_mode=recommended_mode,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        attack_score=attack_score,
        protection_score=protection_score,
        scenario_bad=scenario_bad,
        scenario_base=scenario_base,
        scenario_good=scenario_good,
        reasons=reasons,
        metrics={
            "structural_risk_level": str(structural_risk_level or ""),
            "structural_confidence_score": struct_conf,
            "vigilance_status": str(vigilance_status or ""),
            "vigilance_alert_count": int(vigilance_alert_count),
            "pbo_verdict": str(pbo_verdict or ""),
            "attack_underperform_prob_63": underperf,
            "attack_top3_retention": retention,
            "attack_drawdown_abs": attack_dd,
            "protection_drawdown_abs": protect_dd,
            "execution_winner": str(execution_winner or ""),
        },
    )
