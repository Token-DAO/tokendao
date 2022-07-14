import sys
import os
import bt
import ffn
import patsy
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from tqdm import tqdm
from pypfopt import black_litterman, risk_models, efficient_frontier, objective_functions


def price_history(tickers, period='max', column='Adj Close', start_date=None, slicing_method='dropna_rows'):
    """
    Downloads historical price data including Open, High, Close, Adj Close, or Volume.

    :param tickers: (list) List of tickers.
    :type period: (str) Optional, valid options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
    :param column: Optional, valid options are Open, High, Close, Adj Close, or Volume.
    :param start_date: Optional, start date if slicing method is 'start_date'. Format 'YYYY-MM-DD'.
    :param slicing_method: Optional, splices DataFrame response. Valid options are dropna_rows, dropna_cols, and start_date.
    :return: (pd.DataFrame) DataFrame of historical prices.
    """
    resp = yf.download(tickers, period=period)[column]
    if slicing_method is None or slicing_method == 'None':
        return resp.copy()  # may contain NaNs
    elif slicing_method == 'start_date':
        if start_date in resp.copy().index:
            return resp.copy().loc[start_date:].dropna(axis=1)  # omits tickers without enough price data
        else:
            print('ERROR: Start date not found. Enter a valid start date.')
    elif slicing_method == 'dropna_cols':
        return resp.copy().dropna(axis=1)
    elif slicing_method == 'dropna_rows':
        return resp.copy().dropna(axis=0)
    else:
        raise ValueError(
            'Invalid slicing method. Slicing method can only be "start_date", "dropna_cols", or "dropna_rows"')


def get_info(tickers):
    """
    Downloads ticker info.

    :param tickers: (list) List of tickers.
    :return: (pd.DataFrame) DataFrame of ticker info.
    """
    yf_tickers = yf.Tickers(tickers)
    info_dict = {}
    for ticker in tqdm(tickers):
        info_dict[ticker] = yf_tickers.tickers[ticker].info
    return pd.DataFrame.from_dict(info_dict).T.sort_index(axis=1)


def returns_model(info, prices, covariance_matrix):
    """
    Computes expected returns using Black-Litterman model.

    :param info: (pd.DataFrame) DataFrame of ticker info.
    :param prices: (pd.DataFrame) DataFrame of historical prices.
    :param covariance_matrix: (pd.DataFrame) DataFrame containing risk model covariance matrix.
    :return: (pd.Series) Series of expected returns for each ticker.
    """
    market_caps = info['marketCap'].astype(float)
    risk_aversion = black_litterman.market_implied_risk_aversion(prices, frequency=365)
    return black_litterman.market_implied_prior_returns(market_caps, risk_aversion, covariance_matrix)


def constraints_model(prices):
    """
    Downloads historical price data including Open, High, Close, Adj Close, or Volume.

    :param prices: (pd.DataFrame) DataFrame of historical prices.
    :return: (list of tuples) List of (minimum, maximum) weighting constraints.
    """
    returns = prices.pct_change().dropna()
    volatility = returns.std() * 252 ** 0.5
    volatility_rank = pd.DataFrame(volatility.rank(pct=True, ascending=False), columns=['Volatility_Rank'])
    volatility_rank.index.name = 'Ticker'
    constraints_dict = {
        'Volatility_Rank': [0, .2, .4, .6, .8],
        'Max': [.0625, .125, .25, .5, 1.0]
    }
    constraints_table = pd.DataFrame(constraints_dict, columns=['Volatility_Rank', 'Max'])
    constraints_table['Volatility_Rank'] = constraints_table[['Volatility_Rank']].astype('float64')
    constraints = pd.merge_asof(volatility_rank.reset_index().sort_values(by='Volatility_Rank'), constraints_table,
                                on='Volatility_Rank').set_index('Ticker')
    constraints = constraints[['Max']]
    constraints.insert(0, 'Min', 0)
    constraints.sort_index(inplace=True)
    bounds = [tuple(x) for x in constraints[['Min', 'Max']].to_numpy()]
    return bounds


def clean_weights(weights, cutoff=0.0001, rounding=4):
    """
    Helper method to clean the raw weights, setting any weights whose absolute
    values are below the cutoff to zero, and rounding the rest.

    :param weights: (pd.DataFrame) DataFrame of weightings.
    :type cutoff: (float) Optional, cutoff level to clean weights.
    :param rounding: Number of decimal places to round the weights, defaults to 4.
                     Set to None if rounding is not desired.
    :return: (pd.DataFrame) DataFrame of cleaned weights.
    """
    if weights is None:
        raise AttributeError("Weights not yet computed")
    weights[np.abs(weights) < cutoff] = 0
    if rounding is not None:
        if not isinstance(rounding, int) or rounding < 1:
            raise ValueError("Rounding must be a positive integer")
        weights = np.round(weights, rounding)
    weights = weights.div(weights.sum())
    weights = weights[weights != 0]
    return weights


def optimize_portfolio(
        expected_returns, covariance_matrix, bounds=(0, 1),
        objective='max_sharpe', gamma=0.1, cutoff=0.0001, target_volatility=0.10,
        target_return=0.10, risk_free_rate=0.02, market_neutral=False,
        risk_aversion=1
):
    """
    Compute the optimal portfolio.

    :param expected_returns: (pd.Series) Expected returns for each asset.
    :param covariance_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive
                                             semidefinite, otherwise optimization will fail.
    :param bounds: (list of tuples) List of (minimum, maximum) weighting constraints.
    :param objective: (str) Objective function used in the portfolio optimization. Defaults to 'max_sharpe'.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible
                          weights.
    :type cutoff: (float) Optional, cutoff level to clean weights.
    :param target_volatility: (float) Optional, the desired maximum volatility of the resulting portfolio. Required if
                                      objective function is 'efficient_risk', otherwise, parameter is ignored. Defaults
                                      to 0.01.
    :param target_return: (float) Optional, the desired return of the resulting portfolio. Required if objective
                                  function is 'efficient return', otherwise, parameter is ignored. Defaults to 0.2.
    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02. Required if objective function
                                   is 'max_sharpe', otherwise, parameter is ignored.
    :param market_neutral: (bool) Optional, if weights are allowed to be negative (i.e. short). Defaults to False.
    :param risk_aversion: (positive float) Optional, risk aversion parameter (must be greater than 0). Required if
                                           objective function is 'max_quadratic_utility'. Defaults to 1.
    :return: (tuple) Tuple of weightings (pd.DataFrame) and results (pd.DataFrame) showing risk, return, sharpe ratio
                     metrics.
    """
    weightings = pd.DataFrame()
    results = pd.DataFrame()
    ef = efficient_frontier.EfficientFrontier(expected_returns, covariance_matrix, bounds)
    ef.add_objective(objective_functions.L2_reg, gamma=gamma)
    if objective == 'min_volatility':
        ef.min_volatility()
    elif objective == 'max_sharpe':
        ef.max_sharpe(risk_free_rate)
    elif objective == 'max_quadratic_utility':
        ef.max_quadratic_utility(risk_aversion, market_neutral)
    elif objective == 'efficient_risk':
        ef.efficient_risk(target_volatility, market_neutral)
    elif objective == 'efficient_return':
        ef.efficient_return(target_return, market_neutral)
    else:
        raise NotImplementedError('Check objective parameter. Double-check spelling.')
    weights = ef.clean_weights(cutoff)
    weights = pd.DataFrame.from_dict(weights, orient='index', columns=[int(1)]).round(4)
    weights = clean_weights(weights, cutoff)
    performance = pd.DataFrame(ef.portfolio_performance(risk_free_rate)).round(4)
    weightings = pd.concat([weightings, weights], axis=1).round(4).fillna(0)
    weightings.index.name = 'Ticker'
    results = pd.concat([results, performance], axis=1)
    results.columns = ['Portfolio']
    results = results.rename(index={0: 'Expected_Return', 1: 'Volatility', 2: 'Sharpe_Ratio'})
    return weightings, results


def min_risk(
        expected_returns, covariance_matrix, bounds=(0, 1),
        gamma=0.1, cutoff=0.0001, risk_free_rate=0.02
):
    """
    Compute the minimum risk level.

    :param expected_returns: (pd.Series) Expected returns for each asset.
    :param covariance_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param bounds: (list of tuples) List of (minimum, maximum) weighting constraints.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible weights.
    :type cutoff: (float) Optional, cutoff level to clean weights.
    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02. Required if objective function
                                   is 'max_sharpe', otherwise, parameter is ignored.
    :return: (float) Volatility of the minimum risk portfolio.
    """
    return optimize_portfolio(
        expected_returns, covariance_matrix, bounds, objective='min_volatility',
        gamma=gamma, cutoff=cutoff, risk_free_rate=risk_free_rate
    )[1].loc['Volatility'].squeeze()


def max_risk(
        expected_returns, covariance_matrix, bounds=(0, 1), target_volatility=2.0,
        gamma=0.1, cutoff=0.0001, risk_free_rate=0.02
):
    """
    Compute the maximum risk level.

    :param expected_returns: (pd.Series) Expected returns for each asset.
    :param covariance_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive
                                             semidefinite, otherwise optimization will fail.
    :param bounds: (list of tuples) List of (minimum, maximum) weighting constraints.
    :param target_volatility: (float or int) Optional, sets portfolio volatility % to this level.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible
                          weights.
    :type cutoff: (float) Optional, cutoff level to clean weights.
    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02. Required if objective function
                                   is 'max_sharpe', otherwise, parameter is ignored.
    :return: (float) Volatility of the maximum risk portfolio.
    """
    return optimize_portfolio(
        expected_returns, covariance_matrix, bounds, objective='efficient_risk',
        target_volatility=target_volatility, gamma=gamma, cutoff=cutoff,
        risk_free_rate=risk_free_rate
    )[1].loc['Volatility'].squeeze()


def compute_efficient_frontier(
        expected_returns, covariance_matrix, bounds=(0, 1),
        gamma=0.1, cutoff=0.0001, risk_free_rate=0.02
):
    """
    Compute the efficient frontier portfolios and results.

    :param expected_returns: (pd.Series) Expected returns for each asset.
    :param covariance_matrix: (pd.DataFrame) Covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param bounds: (list of tuples) List of (minimum, maximum) weighting constraints.
    :param gamma: (float) Optional, L2 regularisation parameter, defaults to 0. Increase if you want more non-negligible weights.
    :type cutoff: (float) Optional, cutoff level to clean weights.
    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02. Required if objective function
                                   is 'max_sharpe', otherwise, parameter is ignored.
    :return: (tuple) Tuple of DataFrames containing portfolio weightings and results of efficient frontier portfolios.
    """
    old_stdout = sys.stdout  # backup current stdout
    sys.stdout = open(os.devnull, "w")
    min_volatility = min_risk(expected_returns, covariance_matrix, bounds, gamma, cutoff, risk_free_rate)
    max_volatility = max_risk(expected_returns, covariance_matrix, bounds, gamma, cutoff, risk_free_rate)
    optimized_portfolios = pd.DataFrame()
    results = pd.DataFrame()
    counter = 1
    for i in tqdm(np.linspace(min_volatility + .001, max_volatility, 20).round(4)):
        optimized_portfolio, optimized_performance = optimize_portfolio(
            expected_returns, covariance_matrix, bounds, objective='efficient_risk',
            gamma=gamma, cutoff=cutoff, target_volatility=i, risk_free_rate=risk_free_rate
        )
        portfolio = optimized_portfolio[int(1)]
        portfolio.name = counter
        optimized_portfolios = pd.concat([optimized_portfolios, portfolio], axis=1)
        result = optimized_performance
        result.columns = [counter]
        results = pd.concat([results, result], axis=1)
        counter += 1
    optimized_portfolios = optimized_portfolios.fillna(0)
    sys.stdout = old_stdout  # reset old stdout
    return optimized_portfolios, results


def risk_weightings(optimized_portfolios, covariance_matrix):
    """
    Computes the risk weightings of all portfolios.
    :param optimized_portfolios: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param covariance_matrix: (pd.DataFrame) Covariance of returns for each asset.
    :return: (pd.DataFrame) Risk-weightings for efficient frontier portfolios.
    """
    cash_weightings = optimized_portfolios.copy()
    df = pd.DataFrame(index=cash_weightings.index)
    for i in range(1, cash_weightings.shape[1] + 1):
        w = cash_weightings[i]
        pvar = np.dot(w.T, np.dot(covariance_matrix, w))
        pvolw = ((np.dot(w, covariance_matrix)) / pvar) * w
        df = pd.concat([df, pvolw], axis=1)
    return df


def compute_ticker_vols(tickers, covariance_matrix):
    """
    Computes the volatility of each ticker.

    :param tickers: (list) List of tickers.
    :param covariance_matrix: (pd.DataFrame) Covariance of returns for each asset.
    :return: (pd.Series) Volatility for each ticker.
    """
    count = 0
    ticker_stds = []
    weight_vector = [1] + [0] * (len(tickers) - 1)
    while count < len(tickers):
        ticker_stds.append(np.dot(weight_vector, np.dot(covariance_matrix, weight_vector)).round(4))
        try:
            weight_vector[count], weight_vector[count + 1] = 0, 1
        except IndexError:
            break
        count += 1
    return pd.Series(ticker_stds, tickers)


def eff_frontier_plot(covariance_matrix, expected_returns, results, figsize=(12, 6)):
    """
    Plots the efficient frontier and individual assets.

    :param covariance_matrix: (pd.DataFrame) covariance of returns for each asset. This **must** be positive semidefinite,
                                      otherwise optimization will fail.
    :param expected_returns: (pd.Series) expected returns for each asset.
    :param results: (pd.DataFrame) risk, return, and sharpe ratio for all efficient frontier portfolios. Input the
                                   results DataFrame computed using the compute_efficient_frontier() function.
    :param figsize: (float, float) optional, multiple by which to multiply the maximum weighting constraints at the
                                   ticker level. Defaults to (12, 6).
    :return: (fig) plot of efficient frontier and individual assets.
    """
    portfolio_volatilities = list(results.iloc[1:2, :].squeeze())
    returns = list(results.iloc[:1, :].squeeze())
    sharpe_ratios = list(results.iloc[2:3, :].squeeze())
    max_sharpe_ratio_index = sharpe_ratios.index(max(sharpe_ratios))
    min_volatility_index = portfolio_volatilities.index(min(portfolio_volatilities))
    plt.figure(figsize=figsize)
    plt.plot(portfolio_volatilities, returns, c='black', label='Constrained Efficient Frontier')
    plt.scatter(portfolio_volatilities[max_sharpe_ratio_index],
                returns[max_sharpe_ratio_index],
                marker='*',
                color='g',
                s=400,
                label='Maximum Sharpe Ratio')
    plt.scatter(portfolio_volatilities[min_volatility_index],
                returns[min_volatility_index],
                marker='*',
                color='r',
                s=400,
                label='Minimum Volatility')
    plt.scatter(np.sqrt(np.diag(covariance_matrix)),
                expected_returns,
                marker='.',
                color='black',
                s=100,
                label='Individual Assets')
    plt.title('Efficient Frontier with Individual Assets')
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.legend(loc='upper left')
    plt.show()


class OrderedWeights(bt.Algo):
    def __init__(self, weights):
        self.target_weights = weights

    def __call__(self, target):
        target.temp['weights'] = dict(zip(target.temp['selected'], self.target_weights))
        return True


def rebalance_module(rebalance_freq='once', tolerance=0.2):
    """
    Plots the efficient frontier and individual assets.

    :param rebalance_freq: (str) Desired frequency of rebalancing back to target weights. Options are 'once', 'daily',
                                 'weekly', 'monthly', 'quarterly', 'yearly', or 'outofbounds'. 'Once' implies a buy and
                                 hold portfolio. 'outofbounds' rebalances the portfolio if portfolio weightings deviate
                                 more than the tolerance allowed away from the target weightings.
    :param tolerance: (float) Allowed deviation of each security weight. If a security has a 10% target weight, then
                              setting tolerance to 0.2 means the strategy would rebalance once the security weight surpassed
                              8% or 12%.
    :return: (obj) Object containing rebalancing criteria implemented in backtests.
    """
    if rebalance_freq == 'once':
        return bt.algos.RunOnce()
    elif rebalance_freq == 'weekly':
        return bt.algos.RunWeekly()
    elif rebalance_freq == 'monthly':
        return bt.algos.RunMonthly()
    elif rebalance_freq == 'quarterly':
        return bt.algos.RunQuarterly()
    elif rebalance_freq == 'yearly':
        return bt.algos.RunYearly()
    elif rebalance_freq == 'outofbounds':
        return bt.algos.RunIfOutOfBounds(tolerance)
    else:
        raise ValueError(
            'Invalid input into rebalance_freq parameter. Valid options are once, weekly, monthly, quarterly, or yearly')


def backtest_parameters(portfolio, weightings, prices, rebalance_freq='once', tolerance=0.2):
    """
    Creates Backtest object combining Strategy object with price data.

    :param portfolio: (int) Choose any portfolio from 1-20.
    :param weightings: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :param rebalance_freq: (str) Desired frequency of rebalancing back to target weights. Options are 'once', 'daily',
                                 'weekly', 'monthly', 'quarterly', 'yearly', or 'outofbounds'. 'Once' implies a buy and
                                 hold portfolio. 'outofbounds' rebalances the portfolio if portfolio weightings deviate
                                 more than the tolerance allowed away from the target weightings.
    :param tolerance: (float) Allowed deviation of each security weight. If a security has a 10% target weight, then
                              setting tolerance to 0.2 means the strategy would rebalance once the security weight surpassed
                              8% or 12%.
    :return: (obj) Backtest object combining Strategy object with price data.
    """
    target_weights = weightings[portfolio]
    target_weights = target_weights[target_weights != 0].to_frame()
    tickers = list(target_weights.index)
    weights_dict = target_weights.to_dict().get(portfolio)
    prices_df = prices[tickers]
    rebalance = rebalance_module(rebalance_freq, tolerance)
    strategy = bt.Strategy('{}'.format(portfolio),
                           [rebalance,
                            bt.algos.SelectAll(tickers),
                            OrderedWeights(list(weights_dict.values())),
                            bt.algos.Rebalance()])
    return bt.Backtest(strategy, prices_df)


def compile_backtests(weightings, prices, rebalance_freq='once', tolerance=0.2):
    """
    Compiles multiple backtest objects.

    :param weightings: (pd.DataFrame) Weightings for efficient frontier portfolios.
    :param prices: (pd.DataFrame) Dataframe where each column is a series of prices for an asset.
    :param rebalance_freq: (str) Desired frequency of rebalancing back to target weights. Options are 'once', 'daily',
                                 'weekly', 'monthly', 'quarterly', 'yearly', or 'outofbounds'. 'Once' implies a buy and
                                 hold portfolio. 'outofbounds' rebalances the portfolio if portfolio weightings deviate
                                 more than the tolerance allowed away from the target weightings.
    :param tolerance: (float) Allowed deviation of each security weight. If a security has a 10% target weight, then
                              setting tolerance to 0.2 means the strategy would rebalance once the security weight surpassed
                              8% or 12%.
    :return: (list) List of Backtest objects, one for each efficient frontier portfolio.
    """
    backtests = []
    for backtest in list(weightings.columns):
        backtests.append(backtest_parameters(backtest, weightings, prices, rebalance_freq, tolerance))
    return backtests


def benchmark_strategy(benchmark_ticker='SPY', rebalance_freq='once', tolerance=0.2):
    """
    Creates a Strategy object for the benchmark ticker.

    :param benchmark_ticker: (str) Optional, benchmark ticker. Defaults to 'SPY'.
    :param rebalance_freq: (str) Desired frequency of rebalancing back to target weights. Options are 'once', 'daily',
                                 'weekly', 'monthly', 'quarterly', 'yearly', or 'outofbounds'. 'Once' implies a buy and
                                 hold portfolio. 'outofbounds' rebalances the portfolio if portfolio weightings deviate
                                 more than the tolerance allowed away from the target weightings.
    :param tolerance: (float) Allowed deviation of each security weight. If a security has a 10% target weight, then
                              setting tolerance to 0.2 means the strategy would rebalance once the security weight surpassed
                              8% or 12%.
    :return: (obj) Strategy object assigned to the benchmark.
    """
    rebalance = rebalance_module(rebalance_freq, tolerance)
    return bt.Strategy(
        benchmark_ticker,
        algos=[rebalance,
               bt.algos.SelectAll(),
               bt.algos.SelectThese([benchmark_ticker]),
               bt.algos.WeighEqually(),
               bt.algos.Rebalance()],
    )


def benchmark_backtest(benchmark_ticker):
    """
    Creates Backtest object combining Strategy object with price data from the benchmark.

    :param benchmark_ticker: (str) Optional, benchmark ticker.
    :return: (obj) Backtest object combining Strategy object with price data.
    """
    benchmark_prices = pd.DataFrame(price_history(benchmark_ticker))
    benchmark_prices.columns = [benchmark_ticker]
    return bt.Backtest(benchmark_strategy(benchmark_ticker), benchmark_prices)


def run_backtest(backtests, benchmark):
    """
    Runs the backtest.

    :param backtests: (list) List of Backtest objects, one for each efficient frontier portfolio.
    :param benchmark: (list) Backtest object for the benchmark_strategy.
    :return: (obj) Result object containing backtest results.
    """
    np.seterr(divide='ignore')
    return bt.run(
        backtests[0], backtests[1], backtests[2], backtests[3], backtests[4],
        backtests[5], backtests[6], backtests[7], backtests[8], backtests[9],
        backtests[10], backtests[11], backtests[12], backtests[13], backtests[14],
        backtests[15], backtests[16], backtests[17], backtests[18], backtests[19],
        benchmark
    )


def linechart(Results, title='Backtest Results', figsize=(15, 9), colormap='jet'):
    """
    Plots the performance for all efficient frontier portfolios.

    :param Results: (object) Results object from bt.backtest.Result(*backtests). Refer to the following documentation
                                 https://pmorissette.github.io/bt/bt.html?highlight=display#bt.backtest.Result
    :param title: (str) Optional. Defaults to 'Backtest Results'.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the ticker level.
                                   Defaults to (15, 9).
    :param colormap: (str or matplotlib colormap object) Colormap to select colors from. If string, load colormap with
                                                         that name from matplotlib. Defaults to 'jet'.
    :return: (fig) Plot of performance for all efficient frontier portfolios.
    """
    plot = Results.plot(title=title, figsize=figsize, colormap=colormap)
    fig = plot.get_figure()
    plt.legend()


def backtest_timeseries(Results, freq='d'):
    """
    Plots the performance for all efficient frontier portfolios.

    :param Results: (object) Results object from bt.backtest.Result(*backtests). Refer to the following documentation
                                 https://pmorissette.github.io/bt/bt.html?highlight=display#bt.backtest.Result
    :param freq: (str) Data frequency used for display purposes. Refer to pandas docs for valid freq strings.
    :return: (pd.DataFrame) Time series of each portfolio's value over time according to the backtest Results object.
    """
    return Results._get_series(freq).drop_duplicates().iloc[1:].rebase(100)


def performance_stats(
        backtest_timeseries, benchmark_ticker='BTC-USD', risk_free_rate=0.02, freq=252):
    """
    Computes the cumulative performance statistics based on data from the backtest_timeseries.

    :param backtest_timeseries: (pd.DataFrame) Timeseries performance of efficient frontier portfolios.
    :param benchmark_ticker: (str) Optional, benchmark ticker. Takes only one ticker. Defaults to 'SPY'.
    :param risk_free_rate: (float) Optional, annualized risk-free rate, defaults to 0.02.
    :param freq: (str) Data frequency used for display purposes. Refer to pandas docs for valid freq strings.
    :return: (pd.DataFrame) DataFrame of cumulative performance statistics for all efficient frontier portfolios.
    """
    perf = ffn.core.GroupStats(backtest_timeseries)
    perf.set_riskfree_rate(float(risk_free_rate))
    portfolios = backtest_timeseries.columns
    start_date = backtest_timeseries.index[0].strftime('%m-%d-%Y')
    end_date = backtest_timeseries.index[-1].strftime('%m-%d-%Y')

    cagrs = {}
    vols = {}
    capms = {}
    betas = {}
    jensen_alphas = {}
    appraisal_ratios = {}
    sharpes = {}
    treynors = {}
    information_ratios = {}
    sortinos = {}
    capture_ratios = {}
    drawdowns = {}
    ulcers = {}
    m2s = {}
    m2_alphas = {}

    for portfolio in portfolios[:-1]:
        p = backtest_timeseries.copy()[[portfolio, benchmark_ticker]]
        r = p.pct_change().dropna()
        p.name, r.name = portfolio, benchmark_ticker
        # return
        cagr = (1 + r).prod() ** (freq / (freq if r.shape[0] < freq else r.shape[0])) - 1
        # risk
        vol = r.std() * (freq if r.shape[0] > freq else r.shape[0]) ** 0.5
        # client regression model
        y, x = r[portfolio], r[benchmark_ticker]
        yx = pd.concat([y, x], axis=1)
        y, X = patsy.dmatrices(
            'y ~ x',
            data=yx,
            return_type='dataframe'
        )
        mod = sm.OLS(y, X)
        res = mod.fit()
        # benchmark regression model
        y_b, x_b = r[benchmark_ticker], r[benchmark_ticker]
        yx_b = pd.concat([y_b, x_b], axis=1)
        y_b, X_b = patsy.dmatrices(
            'y_b ~ x_b',
            data=yx_b,
            return_type='dataframe'
        )
        mod_b = sm.OLS(y_b, X_b)
        res_b = mod_b.fit()
        # capm
        capm = risk_free_rate + res.params.values[1] * (cagr[benchmark_ticker] - risk_free_rate)
        beta = res.params.values[1]
        capm_b = risk_free_rate + res_b.params.values[1] * (cagr[benchmark_ticker] - risk_free_rate)
        beta_b = res_b.params.values[1]
        # jensen's alpha
        non_systematic_risk = (
                vol[portfolio] ** 2
                - res.params.values[1] ** 2
                * vol[benchmark_ticker] ** 2
        )
        non_systematic_risk_b = (
                vol[benchmark_ticker] ** 2
                - res_b.params.values[1] ** 2
                * vol[benchmark_ticker] ** 2
        )
        jensen_alpha = float(cagr[portfolio] - capm)
        jensen_alpha_b = float(cagr[benchmark_ticker] - capm_b)
        # appraisal ratio
        appraisal_ratio = jensen_alpha / (non_systematic_risk ** 0.5)
        appraisal_ratio_b = jensen_alpha_b / (non_systematic_risk_b ** 0.5)
        # sharpe ratio
        sharpe = (cagr[portfolio] - risk_free_rate) / vol[portfolio]
        sharpe_b = (cagr[benchmark_ticker] - risk_free_rate) / vol[benchmark_ticker]
        # treynor ratio
        treynor = cagr[portfolio] / beta
        treynor_b = cagr[benchmark_ticker] / 1.
        # information ratio
        yx1 = yx.copy()
        yx1['Active_Return'] = yx1[portfolio] - yx1[benchmark_ticker]
        information_ratio = yx1['Active_Return'].mean() / yx1['Active_Return'].std()
        # sortino ratio
        downside_returns = (yx1[yx1[portfolio] < 0])[portfolio].values
        downside_deviation = downside_returns.std() * (freq if r.shape[0] > freq else r.shape[0]) ** 0.5
        sortino = cagr[portfolio] / downside_deviation
        downside_returns_b = (yx1[yx1[benchmark_ticker] < 0])[[benchmark_ticker]].values
        downside_deviation_b = downside_returns_b.std() * (freq if r.shape[0] > freq else r.shape[0]) ** 0.5
        sortino_b = cagr[benchmark_ticker] / downside_deviation_b
        # capture ratio
        up_returns = yx[yx[portfolio] >= 0].round(4)
        try:
            up_geo_avg = (1 + up_returns[portfolio]).prod() ** (1 / len(up_returns.index)) - 1
            up_geo_avg_b = (1 + up_returns[benchmark_ticker]).prod() ** (1 / len(up_returns.index)) - 1
            down_returns = yx[yx[portfolio] < 0].round(4)
            down_geo_avg = (1 + down_returns[portfolio]).prod() ** (1 / len(down_returns.index)) - 1
            down_geo_avg_b = (1 + down_returns[benchmark_ticker]).prod() ** (1 / len(down_returns.index)) - 1
            up_capture = up_geo_avg / up_geo_avg_b
            down_capture = down_geo_avg / down_geo_avg_b
            capture_ratio = up_capture / down_capture
            capture_ratio_b = 1.
        except ZeroDivisionError:
            capture_ratio = np.nan
            capture_ratio_b = 1.
        # drawdown
        drawdown = p.copy()[[portfolio]]
        drawdown = drawdown.fillna(method='ffill')
        drawdown[np.isnan(drawdown)] = -np.Inf
        roll_max = np.maximum.accumulate(drawdown)
        drawdown = drawdown / roll_max - 1.
        drawdown = drawdown.round(4)
        drawdown = drawdown.iloc[-1:, :].squeeze()
        drawdown_b = p.copy()[[benchmark_ticker]]
        drawdown_b = drawdown_b.fillna(method='ffill')
        drawdown_b[np.isnan(drawdown_b)] = -np.Inf
        roll_max_b = np.maximum.accumulate(drawdown_b)
        drawdown_b = drawdown_b / roll_max_b - 1.
        drawdown_b = drawdown_b.round(4)
        drawdown_b = drawdown_b.iloc[-1:, :].squeeze()
        # ulcer performance index
        ulcer = \
            ffn.core.to_ulcer_performance_index(
                p[[portfolio]], risk_free_rate, nperiods=freq).to_frame('ulcer_index').values[0].squeeze()
        ulcer_b = ffn.core.to_ulcer_performance_index(
            p[[benchmark_ticker]], risk_free_rate, nperiods=freq).to_frame('ulcer_index').values[0].squeeze()
        # M^2 alpha
        m2 = float(sharpe * vol[benchmark_ticker] + risk_free_rate)
        m2_b = float(sharpe_b * vol[benchmark_ticker] + risk_free_rate)
        m2_alpha = m2 - cagr[benchmark_ticker]
        m2_alpha_b = m2_b - cagr[benchmark_ticker]
        # record results
        cagrs[portfolio] = cagr[portfolio]
        vols[portfolio] = vol[portfolio]
        capms[portfolio] = capm
        betas[portfolio] = beta
        jensen_alphas[portfolio] = jensen_alpha
        appraisal_ratios[portfolio] = appraisal_ratio
        sharpes[portfolio] = sharpe
        treynors[portfolio] = treynor
        information_ratios[portfolio] = information_ratio
        sortinos[portfolio] = sortino
        capture_ratios[portfolio] = capture_ratio
        drawdowns[portfolio] = drawdown
        ulcers[portfolio] = ulcer.round(4)
        m2s[portfolio] = m2
        m2_alphas[portfolio] = m2_alpha
    cagrs[benchmark_ticker] = cagr[benchmark_ticker]
    vols[benchmark_ticker] = vol[benchmark_ticker]
    capms[benchmark_ticker] = capm_b
    betas[benchmark_ticker] = beta_b
    jensen_alphas[benchmark_ticker] = jensen_alpha_b
    appraisal_ratios[benchmark_ticker] = 0
    sharpes[benchmark_ticker] = sharpe_b
    treynors[benchmark_ticker] = treynor_b
    information_ratios[benchmark_ticker] = 0
    sortinos[benchmark_ticker] = sortino_b
    capture_ratios[benchmark_ticker] = capture_ratio_b
    drawdowns[benchmark_ticker] = drawdown_b
    ulcers[benchmark_ticker] = ulcer_b.round(4)
    m2s[benchmark_ticker] = m2_b
    m2_alphas[benchmark_ticker] = m2_alpha_b

    cols = [
        'vol',
        'beta',
        'cagr',
        'drawdown',
        'capm',
        'jensen_alpha',
        'm2',
        'm2_alpha',
        'sharpe',
        'treynor',
        'sortino',
        'info_ratio',
        'capture_ratio',
        'appraisal_ratio',
        'ulcer',
    ]

    dicts = [
        vols,
        betas,
        cagrs,
        drawdowns,
        capms,
        jensen_alphas,
        m2s,
        m2_alphas,
        sharpes,
        treynors,
        sortinos,
        information_ratios,
        capture_ratios,
        appraisal_ratios,
        ulcers,
    ]

    performance_data = pd.DataFrame(index=list(cagrs.keys()), columns=cols).reset_index()
    for col, d in zip(cols, dicts):
        performance_data[col] = performance_data['index'].map(d)
    performance_data = performance_data.set_index('index')
    performance_data.index.name = start_date + ' - ' + end_date
    return performance_data.round(4)
