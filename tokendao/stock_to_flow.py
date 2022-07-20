import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from statsmodels.compat import lzip
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, coint
from matplotlib import cm
from matplotlib.ticker import FuncFormatter, ScalarFormatter
from IPython.display import display, Math
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import stats


def timeseries(ticker, fields=None):
    """
    Helper function to filter out tickers which have no available data based on the search term parameter.

    :param ticker: (str) Ticker of cryptocurrency you want to look up.
    :param fields: (str or list of str) Optional, data fields for which to include in timeseries DataFrame.
    :returns: (pd.DateFrame or pd.Series) DataFrame or Series (if only one field) containing requested data.
    """
    if fields is None:
        fields = ['PriceUSD']
    filename = '/coinmetrics/data/blob/master/csv/{}.csv'.format(ticker)
    df = pd.read_csv('https://github.com' + filename + '?raw=true').set_index('time', drop=True)
    df.index = pd.to_datetime(df.index)
    return df[fields].dropna()


def stock_to_flow_data(ticker):
    """
    Compiles required data to compute stock-to-flow model.

    :param ticker: (str) Ticker of cryptocurrency you want to look up.
    :returns: (pd.DateFrame) DataFrame containing stock-to-flow data.
    """
    df = timeseries(ticker, ['CapMrktCurUSD', 'PriceUSD', 'BlkCnt', 'SplyCur'])
    df.insert(3, 'TotalBlks', df.BlkCnt.cumsum().values)
    df['StocktoFlow'] = df['SplyCur'] / (df['SplyCur'] - df['SplyCur'].shift(365))
    return df.dropna().round(2)


def objective(x, a, b):
    """
    Power Law Function
    """
    return np.exp(a) * (x ** b)


def stock_to_flow_model(ticker, p0=None, show=False):
    """
    Computes a fitted stock-to-flow model to observed data, computes Spearman Correlation Coefficient, and tests the
    null hypothesis that there is no correlation between the computed stock-to-flow model and observations. Rejecting
    the null hypothesis means accepting the alternative hypothesis that there is correlation between the stock-to-flow
    model and observed values.

    :param ticker: (str) Ticker of cryptocurrency.
    :param p0: (list of floats) Optional, initial guesses for coefficients a and b in the objective function. Defaults
                                to None.
    :param show: (bool) Optional, prints the results from fitting a power law function to the stock-to-flow data.
                        Defaults to False.
    :returns: (pd.DataFrame, np.array) DataFrame containing data necessary to compute stock-to-flow model along with a
                                       np.array containing the fitted values for coefficients a and b in the objective
                                       function.
    """
    df = stock_to_flow_data(ticker)
    xdata = df.StocktoFlow.values
    ydata = df.CapMrktCurUSD.values
    params, cov = curve_fit(objective, xdata, ydata, p0)
    drawdown = df.CapMrktCurUSD.fillna(method='ffill')
    drawdown[np.isnan(drawdown)] = -np.Inf
    roll_max = np.maximum.accumulate(drawdown)
    drawdown = drawdown / roll_max - 1.
    df['ModelCapMrktCurUSD'] = (np.exp(params[0]) * (df['StocktoFlow'] ** params[1])).round(4)
    df['ModelPriceUSD'] = df['ModelCapMrktCurUSD'] / df['SplyCur']
    df['Difference%'] = df['ModelCapMrktCurUSD'] / df['CapMrktCurUSD'] - 1
    df['MaxDrawdown%'] = drawdown.round(4)
    df.insert(2, 'BlkCntMonthly', df['TotalBlks'] - df['TotalBlks'].shift(30))
    sf = df.StocktoFlow.values[-1].round(2)
    p0 = df.CapMrktCurUSD[-1].round(2)
    p1 = df.ModelCapMrktCurUSD[-1].round(2)
    r, p = (stats.spearmanr(df['CapMrktCurUSD'], df['ModelCapMrktCurUSD']))
    r2 = r ** 2
    n = len(xdata)
    k = len(params)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))
    if show:
        print('Current Stock-to-Flow: {}'.format(sf))
        print('Current Market Value: ${:,.2f}'.format(p0))
        print('Model Prediction: ${:,.2f}'.format(p1))
        print('Potential Return%: {:,.2f}%'.format((p1 / p0 - 1) * 100))
        print('')
        print('Fitted Power Law Model: MarketCapUSD = e^{:.3f} * SF^{:.3f}'.format(*params))
        print('Equivalent Regression Model: ln(MarketCapUSD) = {:.3f} * ln(SF) + {:.3f}'.format(params[1], params[0]))
        print('Spearman R-Squared: {}'.format(r2.round(4)))
        print('Adj. Spearman R-Squared: {}'.format(adj_r2.round(4)))
        print('P-value of Correlation Coefficient: {}'.format(p.round(4)))
        print(' ')
        print('Conclusion: ')
        if p < 0.05:
            print('[1] Correlation detected. Reject null hypothesis that correlation is equal to 0.')
            print('[2] Statistically significant at the 95% confidence level.')
            print(
                '[3] The independent variable explains approximately {}% of the variation in the dependent variable.'
                ''.format((r2 * 100).round(2)))
        else:
            print('[1] No correlation detected. Fail to reject null hypothesis that correlation is equal to 0.')
            print('[2] Statistically insignificant at the 95% confidence level.')
            print(
                '[3] The independent variable explains approximately {}% of the variation in the dependent variable.'
                ''.format((r2 * 100).round(2)))
        print('')
        print('Notes: ')
        print(
            '[1] Assumes model is correctly specified with no violations of the classic normal linear regression '
            'model assumptions.')
        print(
            '[2] Conclusion could be the result of spurious correlation. Test for cointegration to confirm. Use with '
            'caution.')
    return df, params


def regression_analysis(df, show=False, cov_type='HAC'):
    """
    Tests the null hypothesis that the computed stock-to-flow model does not correlate with actual observed values.
    Rejecting the null hypothesis means accepting the alternative hypothesis that the stock-to-flow model does
    correlate with observed data.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model.
    :param show: (bool) Optional, if True, prints the results of the regression analysis and hypothesis test. Defaults
                        to False.
    :param cov_type: (str) Optional, the type of robust sandwich estimator to use. See
                           https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html#statsmodels.regression.linear_model.OLSResults.get_robustcov_results
                           for more information. Defaults to 'HAC'. See
                           https://www.statsmodels.org/devel/generated/statsmodels.stats.sandwich_covariance.cov_hac.html#statsmodels.stats.sandwich_covariance.cov_hac
                           for more information.
    :returns: (obj) Results instance with the requested robust covariance as the default
                    covariance of the parameters. Inferential statistics like p-values and hypothesis tests will be
                    based on this covariance matrix.
    """
    x = df['ModelCapMrktCurUSD']
    y = df['CapMrktCurUSD']
    X = sm.add_constant(x)
    results = sm.OLS(y, X).fit().get_robustcov_results(cov_type, maxlags=1)  # 'HAC' uses Newey-West method
    if show:
        print(results.summary())
        print('\nConclusion: ')
        if results.f_pvalue < 0.05:

            print(
                '[1] Reject H\N{SUBSCRIPT ZERO} because \N{greek small letter beta}\N{SUBSCRIPT ONE} is statistically '
                'different from 0.')
            print('[2] Model may have explanatory value.')
        else:
            print(
                '[1] Fail to reject H\N{SUBSCRIPT ZERO} because \N{greek small letter beta}\N{SUBSCRIPT ONE} is not '
                'statistically different from 0.')
            print('[2] Model does not appear to have explanatory value.')
    return results


def model_significance(ticker, results):
    """
    Generates DataFrame containing statistical significance and correlation data for quick reference.

    :param ticker: (str) Ticker of cryptocurrency.
    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be based on this
                          covariance matrix.
    :returns: (pd.DataFrame) DataFrame containing statistical significance and correlation data.
    """
    return pd.DataFrame(
        index=['f_pvalue', 'const_pvalue', 'beta_pvalue', 'rsquared', 'rsquared_adj'],
        columns=[ticker],
        data=[results.f_pvalue, results.pvalues[0], results.pvalues[1], results.rsquared, results.rsquared_adj]
    ).round(3)


def confidence_interval(df, ticker, results, show=False):
    """
    Generates confidence interval data based on regression analysis.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model.
    :param ticker: (str) Ticker of cryptocurrency.
    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be
                          based on this covariance matrix.
    :param show: (bool) Optional, if True, prints the results of the regression analysis and hypothesis test. Defaults
                        to false.
    :returns: (pd.Series, pd.Series) Contains tuple of two pd.Series containing the lower confidence level and upper
                                    confidence level.
    """
    get_prediction = results.get_prediction().summary_frame()
    obs_ci_lower, obs_ci_upper = get_prediction.obs_ci_lower, get_prediction.obs_ci_upper
    if show:
        print('Ticker: {}'.format(ticker))
        print('Confidence Level: 95%')
        print('Current Market Value: ${:,.2f}'.format(df['CapMrktCurUSD'][-1]))
        print('Lower 95%: ${:,.2f} or {:,.2f}%'.format(obs_ci_lower[-1],
                                                       (obs_ci_lower[-1] / df['CapMrktCurUSD'][-1] - 1) * 100))
        print('Mean Estimate: ${:,.2f} or {:,.2f}%'.format(results.predict()[-1],
                                                           (results.predict()[-1] / df['CapMrktCurUSD'][-1] - 1) * 100))
        print('Upper 95%: ${:,.2f} or {:,.2f}%'.format(obs_ci_upper[-1],
                                                       (obs_ci_upper[-1] / df['CapMrktCurUSD'][-1] - 1) * 100))
    return obs_ci_lower, obs_ci_upper


def markdown_model(params):
    """
    Helper function to display specified regression model in markdown format.

    :param params: (list of [float, float]) Parameters of stock-to-flow model computed using stock_to_flow_model
                                            function.
    :returns: (markdown text) Generates markdown text containing the computed stock-to-flow model.
    """
    a, b = params[0].round(3), params[1].round(3)
    print('Power Law Model:')
    display(Math(r'MarketCapUSD = e^{{{}}} * SF^{{{}}}'.format(a, b)))
    print('which is equivalent to the linear function:')
    display(Math(r'ln(MarketCapUSD) = {{{}}} * ln(SF) + {{{}}}'.format(b, a)))
    print('which is a linear function.')


def breuschpagan(results, alpha=0.05):
    """
    Prints results from Breusch-Pagan Lagrange Multiplier test for heteroscedasticity. The null hypothesis of the test
    is that there is no heteroskedasticity.

    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be
                          based on this covariance matrix.
    :param alpha: (float) Significance level.
    :returns: None
    """
    name = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    test = sms.het_breuschpagan(results.resid, results.model.exog)
    het_breuschpagan = lzip(name, test)
    print('Breusch-Pagan f p-value = {}'.format(het_breuschpagan[3][1]))
    print('')
    print('Conclusion: ')
    if het_breuschpagan[3][1] < alpha:
        print('[1] Heteroskedasticity detected. Reject null hypothesis of no heteroskedasticity.')
        print('[2] The variance of the error terms may not be the same for all observations.')
        print('[3] OLS standard errors in this regression likely significantly underestimate the true standard errors.')
        print(
            '[4] t-statistics for the significance of individual regression coefficients likely to be inflated and '
            'unreliable.')
        print('[5] Estimators of the standard error of regression coefficients likely to be biased and unreliable.')
    else:
        print('[1] Heteroskedasticity was not detected. Fail to reject null hypothesis of no heteroskedasticity.')


def durbinwatson(results, critical_value=1.925):
    """
    Prints results from Durbin-Watson test for serial correlation. The null hypothesis of the test is that there is no
    serial correlation in the residuals.

    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be
                          based on this covariance matrix.
    :param critical_value: (float) Critical value with which to compare the Durbin-Watson statistic to test for serial
                                   correlation.
    :returns: None
    """
    dw = durbin_watson(results.resid).round(3)
    print('Durbin-Watson = {}'.format(dw))
    print('')
    print('Conclusion: ')
    if dw < critical_value:
        print('[1] Positive serial correlation detected. Reject null hypothesis of no positive serial correlation.')
        print('[2] F-statistic to test overall significance of the regression likely to be inflated.')
        print('[3] OLS standard errors in this regression likely significantly underestimate the true standard errors.')
        print(
            '[4] t-statistics for the significance of individual regression coefficients likely to be inflated and '
            'unreliable.')
        print('[5] Estimators of the standard error of regression coefficients likely to be biased and unreliable.')
    else:
        print(
            '[1] No positive serial correlation detected. Fail to reject null hypothesis of no positive serial '
            'correlation.')


def shapiro_test(results, alpha=0.05):
    """
    Prints results from Shapiro-Wilk test for normality. The Shapiro-Wilk test tests the null hypothesis that the data
    was drawn from a normal distribution.

    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be
                          based on this covariance matrix.
    :param alpha: (float) Significance level.
    :returns: None
    """
    shapiro_test = stats.shapiro(results.resid)
    print(shapiro_test)
    print('')
    print('Conclusion:')
    if shapiro_test[1] < alpha:
        print('[1] Non-normality detected. Reject null hypothesis that the residuals are normal.')
        print('[2] Regression model may violate assumption of normality in linear regression.')
    else:
        print('[1] Normality detected. Fail to reject null hypothesis that the residuals are normal.')
        print('[2] Regression model appears to satisfy the assumption of normality in linear regression.')


def adfuller_test(df, alpha=0.05):
    """
    Prints results from Augmented Dickey-Fuller unit root test. The Augmented Dickey-Fuller test can be used to test
    for a unit root in a univariate process in the presence of serial correlation.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model.
    :param alpha: (float) Significance level.
    :returns: None
    """
    adf_sf = adfuller(df['StocktoFlow'])[1].round(4)
    adf_mktcap = adfuller(df['CapMrktCurUSD'])[1].round(4)
    print('Augmented Dickey-Fuller = {} (StocktoFlow)'.format(adf_sf))
    print('Augmented Dickey-Fuller = {} (CapMrktCurUSD)'.format(adf_mktcap))
    print('')
    print('Conclusion: ')
    if (adf_sf > alpha) & (adf_mktcap > alpha):
        print(
            '[1] Unit root and nonstationarity detected in both time series. Reject null hypothesis of no unit root '
            'and stationarity.')
        print(
            '[2] Expected value of the error term may not be 0 which may result in inconsistent regression '
            'coefficients and standard errors.')
        print(
            '[3] Variance of the error term may not be constant for all observations indicating presence of '
            'heteroskedasticity.')
        print('[4] Error term may be correlated across observations indicating presence of serial correlation.')
        print(
            '[5] Regression model appears to violate nonstationarity assumption of linear regression and may need to '
            'be corrected.')
    elif (adf_sf > alpha) | (adf_mktcap > alpha):
        print(
            '[1] Unit root and nonstationarity detected in one of the time series. Reject null hypothesis of no unit '
            'root and stationarity.')
        print(
            '[2] Expected value of the error term may not be 0 which may result in inconsistent regression '
            'coefficients and standard errors.')
        print(
            '[3] Variance of the error term may not be constant for all observations indicating presence of '
            'heteroskedasticity.')
        print('[4] Error term may be correlated across observations indicating presence of serial correlation.')
        print(
            '[5] Regression model appears to violate nonstationarity assumption of linear regression and may need to '
            'be corrected.')
    else:
        print(
            '[1] No positive serial correlation detected. Fail to reject null hypothesis of no positive serial '
            'correlation. ')
        print('[2] Regression model does not appear to violate nonstationarity assumption of linear regression.')


def cointegration(df, alpha=0.05):
    """
    Prints results from test for no-cointegration of a univariate equation. The null hypothesis is no cointegration.
    This uses the augmented Engle-Granger two-step cointegration test. Constant or trend is included in 1st stage
    regression, i.e. in cointegrating equation.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model.
    :param alpha: (float) Significance level.
    :returns: None
    """
    coint_test = coint(df['StocktoFlow'], df['CapMrktCurUSD'])
    coint_pvalue = coint_test[1]
    print('Cointegration p-value = {}'.format(coint_pvalue.round(3)))
    print('')
    print('Conclusion: ')
    if coint_pvalue > alpha:
        print('[1] No cointegration detected at the 5% level. Fail to reject null hypothesis.')
        print('[2] There may not exist a cointegrated relationship between the dependent and independent variables.')
        print('[3] Regression model is likely to show spurious correlation and be unreliable.')
    else:
        print('[1] Cointegration detected at the 5% level. Reject the null hypothesis.')
        print('[2] There may exist a cointegrated relationship between the dependent and independent variables.')
        print('[3] Regression model is unlikely to show spurious correlation and may be reliable.')


def conf_int_chart(df, ticker, results, figsize=(12, 6), save=False, show=True):
    """
    Generates a plot of regression model data and confidence interval.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model
    :param ticker: (str) Ticker of cryptocurrency.
    :param results: (obj) Results instance with the requested robust covariance as the default covariance of the
                          parameters. Inferential statistics like p-values and hypothesis tests will be
                          based on this covariance matrix.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the
                                   ticker level. Defaults to (12,6).
    :param save: (bool) Optional, saves the chart as a png file to charts folder. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :returns: (plt) Generates log scale pyplot of CapMrktCurUSD over CapMrktCurUSD with 95% confidence interval.
    """
    params = results.params
    ytrue = df['CapMrktCurUSD'].to_numpy()
    ypred = df['ModelCapMrktCurUSD'].to_numpy()
    obs_ci_lower, obs_ci_upper = confidence_interval(df, ticker, results)
    plt.style.use('default')
    fig = plt.gcf()
    fig.set_size_inches(figsize)
    plt.plot(ypred, ytrue, 'bo')
    plt.plot(ypred, results.predict(), 'r-')
    plt.plot(ypred, sorted(obs_ci_lower), 'r--')
    plt.plot(ypred, sorted(obs_ci_upper), 'r--')
    plt.title("CapMrktCurUSD vs. ModelCapMrktCurUSD ({})\n {} to {}".format(
        ticker,
        df.index[0].strftime('%m-%d-%Y'),
        df.index[-1].strftime('%m-%d-%Y')))
    plt.legend([
        'CapMrktCurUSD / ModelCapMrktCurUSD ({})'.format(ticker),
        'Linear Model: {:.4f}x + {:.4f}'.format(params[1], params[0]),
        '95% Confidence Interval'
    ])
    plt.xlabel('ModelCapMrktCurUSD ({})'.format(ticker))
    plt.ylabel('CapMrktCurUSD ({})'.format(ticker))
    plt.xscale('log')
    plt.yscale('log')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:,.16g}'.format(ytrue)))
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda ypred, _: '{:,.16g}'.format(ypred)))
    if save: plt.savefig(
        '../charts/conf_int_chart_{}.png'.format(datetime.today().strftime('%m-%d-%Y')), bbox_inches='tight')
    if not show: plt.close()


def charts(df, ticker, params, chart=1, figsize=(12, 6), save=False, show=True):
    """
    Helper function of preformatted charts to show the results of the stock-to-flow model curve fitting.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model
    :param ticker: (str) Ticker of cryptocurrency.
    :param params: (list of [float,float]) Ticker of cryptocurrency.
    :param chart: (int) Select one of 3 pre-formatted charts labeled 1, 2 and 3. Defaults to 1.
    :param figsize: (float, float) Optional, multiple by which to multiply the maximum weighting constraints at the
                                   ticker level.
    :param save: (bool) Optional, saves the chart as a png file to charts folder. Defaults to False.
    :param show: (bool) Optional, displays plot. Defaults to True.
    :returns: (pyplot) Generates log scale pyplot.
    """
    dates = np.array(df.index)
    sf = df['StocktoFlow'].to_numpy()
    d = (df['MaxDrawdown%'] * 100).to_numpy()
    ytrue = df['CapMrktCurUSD'].to_numpy()
    ypred = df['ModelCapMrktCurUSD'].to_numpy()
    if chart == 1:
        plt.style.use('grayscale')
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.scatter(dates, ytrue, c=d, cmap=cm.jet, lw=1, alpha=1, zorder=5, label=ticker)
        plt.yscale('log', subsy=[1])
        ax.plot(dates, ypred, c='black', label='ModelCapMrktCurUSD: e^{:.3f} * SF^{:.3f}'.format(*params))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:,.16g}'.format(ytrue)))
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Maximum Drawdown%')
        plt.xlabel('Year')
        plt.ylabel('CapMrktCurUSD ({})'.format(ticker))
        plt.title("CapMrktCurUSD and ModelCapMrktCurUSD ({})\n {} to {}".format(
            ticker,
            df.index[0].strftime('%m-%d-%Y'),
            df.index[-1].strftime('%m-%d-%Y')))
        plt.legend()
        plt.show()
    elif chart == 2:
        plt.style.use('default')
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        plt.yscale('log')
        plt.plot(dates, ytrue, '-b')
        plt.plot(dates, ypred, 'r')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        plt.gca().xaxis.set_major_locator(mdates.YearLocator(1))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:,.16g}'.format(ytrue)))
        plt.legend(['CapMrktCurUSD ({})'.format(ticker), 'ModelCapMrktCurUSD: e^{:.3f} * SF^{:.3f}'.format(*params)])
        plt.title("CapMrktCurUSD and ModelCapMrktCurUSD ({})\n {} to {}".format(
            ticker,
            df.index[0].strftime('%m-%d-%Y'),
            df.index[-1].strftime('%m-%d-%Y')))
        plt.xlabel('Year')
        plt.ylabel('CapMrktCurUSD ({})'.format(ticker))
    elif chart == 3:
        plt.style.use('default')
        fig = plt.gcf()
        fig.set_size_inches(figsize)
        plt.yscale('log')
        plt.plot(sf, ytrue, 'bo', label='data')
        plt.plot(sf, objective(sf, *params), 'r-', label='curve_fit')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:,.16g}'.format(ytrue)))
        plt.legend(
            ['CapMrktCurUSD ({})'.format(ticker), 'Fitted Power Law Model: e^{:.3f} * SF^{:.3f}'.format(*params)])
        plt.title("CapMrktCurUSD vs. Stock-to-Flow ({})\n {} to {}".format(
            ticker,
            df.index[0].strftime('%m-%d-%Y'),
            df.index[-1].strftime('%m-%d-%Y')))
        plt.xlabel('Stock-to-Flow ({})'.format(ticker))
        plt.ylabel('CapMrktCurUSD ({})'.format(ticker))
    elif chart == 4:
        a1, b1 = 14.6, 3.3
        a2, b2 = params[0], params[1]
        drawdown = (df['MaxDrawdown%'] * 100).to_numpy()
        ytrue = df['CapMrktCurUSD'].to_numpy()
        ypred_planb, ypred_tokendao = np.exp(a1) * sf ** b1, np.exp(a2) * sf ** b2
        plt.style.use('grayscale')
        fig, ax = plt.subplots(figsize=(15, 7))
        im = ax.scatter(sf, ytrue, c=drawdown, cmap=cm.jet, lw=1, alpha=1, zorder=5, label=ticker)
        plt.xscale('log')
        plt.yscale('log', subsy=[1])
        ax.plot(sf, ypred_planb, c='black', label='Plan B Model: e^{:.3f} * SF^{:.3f}'.format(a1, b1))
        ax.plot(sf, ypred_tokendao, c='red', label='tokendao Model: e^{:.3f} * SF^{:.3f}'.format(a2, b2))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:,.16g}'.format(ytrue)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda sf, _: '{:,.16g}'.format(sf)))
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Maximum Drawdown%')
        plt.xlabel('Stock-to-Flow')
        plt.ylabel('CapMrktCurUSD ({})'.format(ticker))
        plt.title("Stock-to-Flow and CapMrktCurUSD ({})\n {} to {}".format(
            ticker,
            df.index[0].strftime('%m-%d-%Y'),
            df.index[-1].strftime('%m-%d-%Y')))
        plt.legend()
    else:
        raise ValueError('Invalid chart number. Type a valid number to the chart parameter.')
    if save:
        plt.savefig('../charts/chart{}_{}.png'.format(chart, datetime.today().strftime('%m-%d-%Y')),
                    bbox_inches='tight')
    if not show:
        plt.close()


def forecast(df, params, halvening_dates, daily_average_mined, daily_std_mined, divisor=None):
    """
    Forecasts the future expected price based on the computed stock-to-flow model.

    :param df: (pd.DataFrame) DataFrame containing data necessary to compute stock-to-flow model.
    :param params: (list of [float, float]) Parameters of stock-to-flow model computed using stock_to_flow_model
                                            function.
    :param halvening_dates: (list) List of halvening dates. Date format is 'YYYY-MM-DD'.
    :param daily_average_mined: (list of [float,float]) Ticker of cryptocurrency.
    :param daily_std_mined: (int) Select one of 3 pre-formatted charts labeled 1, 2 and 3. Defaults to 1.
    :param divisor: (int) Divisor applied to block reward during halving events.
    :returns: (tuple of pd.DataFrame) Generates tuple of dataframes containing forecasted data and historical sample
                                      stock-to-flow data.
    """
    past_df = df.copy()[['SplyCur']]
    past_df['SplyCurDiffDaily'] = past_df['SplyCur'].diff()
    past_df['SplyCurDiffAnnual'] = past_df['SplyCur'].diff(365)

    dfs = {'past': past_df}
    for halvening_date, i in tqdm(zip(halvening_dates, range(0, len(halvening_dates)))):
        days_until = (pd.to_datetime(halvening_date, format='%Y-%m-%d') - pd.to_datetime(
            dfs[list(dfs.keys())[-1]].index[-1])).days
        future_df = pd.DataFrame(
            index=pd.date_range(pd.to_datetime(dfs[list(dfs.keys())[-1]].index[-1] + timedelta(days=1)),
                                periods=days_until))
        if i == 0:
            divisor = 1
        else:
            if divisor is None:
                divisor = 2
            else:
                pass
        daily_average_mined = daily_average_mined / divisor
        daily_std_mined = daily_std_mined / divisor
        future_df['SplyCurDiffDaily'] = np.random.normal(daily_average_mined, daily_std_mined, days_until).round(2)
        splycur = [dfs[list(dfs.keys())[-1]]['SplyCur'][-1]]
        [splycur.append((splycur[-1] + sply).round(2)) for sply in future_df.SplyCurDiffDaily.values]
        future_df['SplyCur'] = splycur[1:]
        future_df = future_df[['SplyCur', 'SplyCurDiffDaily']]
        future_df = pd.concat([dfs[list(dfs.keys())[-1]], future_df])
        future_df['StocktoFlowAnnual'] = round(
            future_df['SplyCur'] / (future_df['SplyCur'] - future_df['SplyCur'].shift(365)), 2)
        dfs[halvening_date] = future_df
    full_df = dfs[list(dfs.keys())[-1]]
    full_df['SplyCurDiffAnnual'] = (full_df['SplyCur'] - full_df['SplyCur'].shift(365))
    full_df['ModelCapMrktCurUSD'] = (np.exp(params[0]) * (full_df['StocktoFlowAnnual'] ** params[1])).round(4)
    full_df['ModelPriceUSD'] = (full_df['ModelCapMrktCurUSD'] / full_df['SplyCur']).round(2)
    full_df = full_df.dropna()
    sample_df = df.loc[full_df.index[0]:]
    return full_df, sample_df


def forecast_chart(sample_df, full_df, params, ticker):
    """
    Forecasts the future expected price based on the computed stock-to-flow model.

    :param sample_df: (pd.DataFrame) DataFrame containing historical sample stock-to-flow data.
    :param full_df: (pd.DataFrame) DataFrame containing forecasted stock-to-flow data.
    :param params: (list) Parameters of stock-to-flow model computed using stock_to_flow_model function.
    :param ticker: (str) Ticker of cryptocurrency you want to look up.
    :returns: (plot) Generates a plot showing stock-to-flow model forecast versus historical data.
    """
    plot_df = pd.DataFrame({'PriceUSD': sample_df.PriceUSD, 'ModelPriceUSD': full_df.ModelPriceUSD})
    plt.figure(figsize=(12, 6))
    plt.yscale('log')
    plt.title("PriceUSD versus ModelPriceUSD ({})\n {} to {}".format(
        ticker.upper(),
        full_df.index[0].strftime('%m-%d-%Y'),
        full_df.index[-1].strftime('%m-%d-%Y')))
    plt.xlabel('Year')
    plt.ylabel('PriceUSD ({})'.format(ticker.upper()))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda ytrue, _: '{:,.16g}'.format(ytrue)))
    plt.plot(plot_df.PriceUSD, 'b')
    plt.plot(plot_df.ModelPriceUSD, 'r')
    plt.legend(['PriceUSD ({})'.format(ticker.upper()), 'ModelPriceUSD: e^{:.3f} * SF^{:.3f}'.format(*params)])
    plt.show()


def selected_forecast(full_df, ticker):
    """
    Displays current and annual price forecasts based on computed stock-to-flow model.

    :param full_df: (pd.DataFrame) DataFrame containing forecasted stock-to-flow data.
    :param ticker: (str) Ticker of cryptocurrency you want to look up.
    :returns: (print) Prints current and annual price forecasts based on computed stock-to-flow model.
    """
    selected_future_dates = [datetime.today().strftime('%Y-%m-%d'),
                             datetime(datetime.today().year, 12, 31).strftime('%Y-%m-%d')]
    for i in range(0, relativedelta(full_df.index[-1], datetime.today()).years):
        selected_future_dates.append(datetime(datetime.today().year + i + 1, 12, 31).strftime('%Y-%m-%d'))
    try:
        select_df = full_df.loc[selected_future_dates]
    except KeyError:
        select_df = full_df.loc[selected_future_dates[:-1]]
    print('Forecasted {} Price:\n'.format(ticker.upper()))
    for i in range(0, len(select_df.index)):
        print(select_df.index[i].strftime('%Y-%m-%d') + ': {}'.format('${:,.0f}'.format(select_df.ModelPriceUSD[i])))
