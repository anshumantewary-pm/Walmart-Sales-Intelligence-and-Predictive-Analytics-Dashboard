import pytest

from src.analysis import insights


def test_holiday_lift_has_each_store_type(sales):
    out = insights.holiday_lift(sales)
    assert set(out["Type"]) == set(sales["Type"])
    assert {"normal_week_avg", "holiday_week_avg", "lift_pct"} <= set(out.columns)


def test_markdown_effect_periods(sales):
    out = insights.markdown_effect(sales)
    assert "no_markdown" in set(out["period"])
    base = out.loc[out["period"] == "no_markdown", "vs_no_markdown_pct"].iloc[0]
    assert base == pytest.approx(0)


def test_macro_correlation_in_range(sales):
    out = insights.macro_correlation(sales)
    assert out["spearman_corr"].between(-1, 1).all()


def test_store_segments_one_row_per_store(sales):
    out = insights.store_segments(sales, k=2)
    assert len(out) == sales["Store"].nunique()
    assert out["segment"].notna().all()


def test_inventory_cost():
    out = insights.inventory_cost([100, 100], [110, 80], overstock_cost=0.5, understock_cost=1)
    assert out["overstock_$"] == 10 and out["understock_$"] == 20
    assert out["est_cost_$"] == 25
