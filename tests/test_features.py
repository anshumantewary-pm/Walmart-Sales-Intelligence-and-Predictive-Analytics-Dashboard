from src.config import TEST_WEEKS


def test_lag52_is_same_week_last_year(sales):
    g = sales[(sales["Store"] == 1) & (sales["Dept"] == 1)].reset_index(drop=True)
    assert g.loc[52, "Sales_Lag52"] == g.loc[0, "Weekly_Sales"]
    assert g["Sales_Lag52"].head(52).isna().all()


def test_no_lag_shorter_than_forecast_horizon(sales):
    # Any sales-based feature must look back at least TEST_WEEKS weeks.
    g = sales[(sales["Store"] == 1) & (sales["Dept"] == 1)].reset_index(drop=True)
    i = 60
    g.loc[i - TEST_WEEKS + 1 : i, "Weekly_Sales"] = -1e9  # poison recent weeks
    from src.features.build import add_lag_features
    rebuilt = add_lag_features(g).reset_index(drop=True)
    assert rebuilt.loc[i, "Sales_Lag52"] == g.loc[i - 52, "Weekly_Sales"]
    assert rebuilt.loc[i, "Sales_Lag52_Smooth"] > 0


def test_calendar_features(sales):
    assert sales["Week"].between(1, 53).all()
    assert sales["TypeCode"].notna().all()
