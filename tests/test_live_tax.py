from __future__ import annotations

from pathlib import Path

from execution.live_tax import build_live_tax_summary


def test_live_tax_summary_keeps_exempt_month_tax_zero(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "\n".join(
            [
                "plan_run_id,ticket_id,ticker,side,planned_notional_brl,filled_notional_brl,filled_quantity,avg_price_brl,fee_brl,executed_at,status",
                "r1,t1,BTC-USD,buy,1000,1000,0.01,100000,0,2026-01-10T10:00:00,filled",
                "r1,t2,BTC-USD,sell,1200,1200,0.01,120000,0,2026-01-20T10:00:00,filled",
            ]
        ),
        encoding="utf-8",
    )
    summary = build_live_tax_summary(
        ledger_csv=ledger,
        net_assumptions_config="config/profit_net_assumptions.json",
    )
    assert summary["status"] == "ok"
    assert summary["monthly_rows"][0]["estimated_tax_due_brl"] == 0.0


def test_live_tax_summary_flags_tax_when_sales_above_exemption(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.csv"
    ledger.write_text(
        "\n".join(
            [
                "plan_run_id,ticket_id,ticker,side,planned_notional_brl,filled_notional_brl,filled_quantity,avg_price_brl,fee_brl,executed_at,status",
                "r1,t1,BTC-USD,buy,30000,30000,1.0,30000,0,2026-01-10T10:00:00,filled",
                "r1,t2,BTC-USD,buy,10000,10000,0.25,40000,0,2026-01-12T10:00:00,filled",
                "r1,t3,BTC-USD,sell,50000,50000,1.0,50000,0,2026-01-20T10:00:00,filled",
            ]
        ),
        encoding="utf-8",
    )
    summary = build_live_tax_summary(
        ledger_csv=ledger,
        net_assumptions_config="config/profit_net_assumptions.json",
    )
    assert summary["monthly_rows"][0]["sales_above_exemption"] is True
    assert summary["monthly_rows"][0]["estimated_tax_due_brl"] > 0.0
