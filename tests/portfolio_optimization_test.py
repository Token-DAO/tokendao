import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from pypfopt import risk_models

ROOT_DIR = os.path.abspath(os.curdir)
ROOT_DIR = ROOT_DIR.replace("\\", "/").replace("tests", "")
sys.path.insert(0, ROOT_DIR)

from tokendao import portfolio_optimization as opt

tickers = [
    "BTC-USD",
    "ETH-USD",
    "ADA-USD",
    "SOL-USD",
    "USDC-USD"
]

prices = opt.price_history(tickers)
info = opt.get_info(tickers)
covariance_matrix = risk_models.risk_matrix(prices, 'oracle_approximating')
expected_returns = opt.returns_model(info, prices, covariance_matrix)


def test_price_history():
    assert isinstance(prices, pd.DataFrame), "Data type should be DataFrame."


def test_get_info():
    assert isinstance(info, pd.DataFrame), "Data type should be DataFrame."


def test_covariance_matrix():
    assert isinstance(covariance_matrix, pd.DataFrame), "Data type should be DataFrame."


def test_returns_model():
    assert isinstance(expected_returns, pd.Series), "Data type should be Series."


def test_constraints_model():
    bounds = opt.constraints_model(prices)
    assert isinstance(bounds, list), "Data type should be list."


def test_clean_weights():
    weights = pd.DataFrame({1: [0.50, 0.30, 0.15, 0.025, 0.025]}, index=tickers)
    weights = opt.clean_weights(weights, cutoff=0.10)
    assert isinstance(weights, pd.DataFrame), "Data type should be DataFrame."


def test_optimize_portfolio():
    weightings, results = opt.optimize_portfolio(expected_returns, covariance_matrix)
    assert isinstance(weightings, pd.DataFrame), "Data type should be DataFrame."
    assert isinstance(results, pd.DataFrame), "Data type should be DataFrame."


def test_min_max_risk():
    min_risk = opt.min_risk(expected_returns, covariance_matrix)
    max_risk = opt.max_risk(expected_returns, covariance_matrix)
    assert isinstance(min_risk, float), "Data type should be float."
    assert isinstance(max_risk, float), "Data type should be float."


def test_compute_efficient_frontier():
    optimized_portfolios, results = opt.compute_efficient_frontier(expected_returns, covariance_matrix)
    assert isinstance(optimized_portfolios, pd.DataFrame), "Data type should be DataFrame."
    assert isinstance(results, pd.DataFrame), "Data type should be DataFrame."


def test_risk_weightings():
    optimized_portfolios, results = opt.compute_efficient_frontier(expected_returns, covariance_matrix)
    df = opt.risk_weightings(optimized_portfolios, covariance_matrix)
    assert isinstance(df, pd.DataFrame), "Data type should be DataFrame."


def test_compute_ticker_vols():
    ticker_vols = opt.compute_ticker_vols(tickers, covariance_matrix)
    assert isinstance(ticker_vols, pd.Series), "Data type should be Series"


if __name__ == "__main__":
    test_price_history()
    test_get_info()
    test_returns_model()
    test_constraints_model()
    test_clean_weights()
    test_optimize_portfolio()
    test_min_max_risk()
    test_compute_efficient_frontier()
    test_risk_weightings()
    test_compute_ticker_vols()
    print("Everything passed")
