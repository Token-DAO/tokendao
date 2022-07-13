import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models

ROOT_DIR = os.path.abspath(os.curdir)
ROOT_DIR = ROOT_DIR.replace("\\", "/").replace("tests", "")
sys.path.insert(0, ROOT_DIR)

from tokendao import stock_to_flow as sf

ticker = "btc"
df, params = sf.stock_to_flow_model(ticker)
results = sf.regression_analysis(df)


def test_timeseries():
    timeseries = sf.timeseries(ticker)
    assert isinstance(timeseries, pd.DataFrame), "Data type should be DataFrame."


def test_stock_to_flow_data():
    data = sf.stock_to_flow_data(ticker)
    assert isinstance(data, pd.DataFrame), "Data type should be DataFrame."


def test_stock_to_flow_model():
    assert isinstance(df, pd.DataFrame), "Data type should be DataFrame."
    assert isinstance(params, np.ndarray), "Data type should be DataFrame."


def test_regression_analysis():
    assert results.summary() is not None, "Results object returned NoneType."


def test_model_significance():
    model_significance = sf.model_significance(ticker, results)
    assert isinstance(model_significance, pd.DataFrame)


def test_confidence_interval():
    obs_ci_lower, obs_ci_upper = sf.confidence_interval(df, ticker, results)
    assert isinstance(obs_ci_lower, pd.Series), "Data type should be Series."
    assert isinstance(obs_ci_upper, pd.Series), "Data type should be Series."


def test_breuschpagan():
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        sf.breuschpagan(results)
        sys.stdout = old_stdout  # reset old stdout
    except:
        raise AssertionError("Exception raised.")


def test_durbinwatson():
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        sf.durbinwatson(results)
        sys.stdout = old_stdout  # reset old stdout
    except:
        raise AssertionError("Exception raised.")


def test_shapiro():
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        sf.shapiro_test(results)
        sys.stdout = old_stdout  # reset old stdout
    except:
        raise AssertionError("Exception raised.")


def test_adfuller():
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        sf.adfuller_test(df)
        sys.stdout = old_stdout  # reset old stdout
    except:
        raise AssertionError("Exception raised.")


def test_cointegration():
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        sf.cointegration(df)
        sys.stdout = old_stdout  # reset old stdout
    except:
        raise AssertionError("Exception raised.")


def test_forecast():
    halvening_dates = ['2024-04-02', '2028-06-01', '2032-06-01']
    daily_average_mined = 905.80  # currently pre-2024-04-02
    daily_std_mined = 105.82  # currently pre-2024-04-02
    full_df, sample_df = sf.forecast(df, params, halvening_dates, daily_average_mined, daily_std_mined)
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        sf.forecast_chart(sample_df, full_df, params)
        sys.stdout = old_stdout  # reset old stdout
    except:
        raise AssertionError("Exception raised.")
    assert isinstance(full_df, pd.DataFrame)
    assert isinstance(sample_df, pd.DataFrame)


def test_charts():
    sf.charts(df, ticker, params, chart=1, figsize=(16, 7))


def test_selected_forecast():
    try:
        old_stdout = sys.stdout  # backup current stdout
        sys.stdout = open(os.devnull, "w")
        sf.forecast_chart(sample_df, full_df, params_sf)
        sys.stdout = old_stdout  # reset old stdout
    except:
        raise AssertionError("Exception raised.")


if __name__ == "__main__":
    test_timeseries()
    test_stock_to_flow_data()
    test_stock_to_flow_model()
    test_regression_analysis()
    test_model_significance()
    test_confidence_interval()
    test_breuschpagan()
    print("Everything passed")
