from engine.portfolio.period_loss_guards import (
    PeriodLossGuardConfig,
    combine_guard_actions,
    monthly_loss_guard,
    quarterly_loss_guard,
)


def test_monthly_loss_guard_thresholds():
    cfg = PeriodLossGuardConfig()
    assert monthly_loss_guard(0.01, cfg) == "NORMAL"
    assert monthly_loss_guard(-0.04, cfg) == "REDUCED_ATTACK"
    assert monthly_loss_guard(-0.07, cfg) == "PROTECTED"
    assert monthly_loss_guard(-0.11, cfg) == "CASH_HEAVY"


def test_quarterly_loss_guard_thresholds():
    cfg = PeriodLossGuardConfig()
    assert quarterly_loss_guard(0.02, cfg) == "NORMAL"
    assert quarterly_loss_guard(-0.06, cfg) == "REDUCED_ATTACK"
    assert quarterly_loss_guard(-0.10, cfg) == "PROTECTED"
    assert quarterly_loss_guard(-0.15, cfg) == "CASH_HEAVY"


def test_combine_guard_actions_returns_strongest():
    assert combine_guard_actions("NORMAL", "REDUCED_ATTACK") == "REDUCED_ATTACK"
    assert combine_guard_actions("PROTECTED", "CASH_HEAVY") == "CASH_HEAVY"
    assert combine_guard_actions("NORMAL", "NORMAL") == "NORMAL"
