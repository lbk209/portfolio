import pandas as pd
import matplotlib.pyplot as plt

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

import bt
from bt.algos import (
    RunWeekly, RunMonthly, RunQuarterly, RunYearly, 
    SelectN, SelectMomentum
)

from pf_custom import SelectKRatio

import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)

metrics = [
    'total_return', 'cagr', 'calmar', 
    'max_drawdown', 'avg_drawdown', 'avg_drawdown_days', 
    'daily_vol', 'daily_sharpe', 'daily_sortino', 
    'monthly_vol', 'monthly_sharpe', 'monthly_sortino'
]

WEEKS_IN_YEAR = 51


def import_rate1(file, path='.', cols=['date', None]):
    """
    file: historical of cumulative rate of return in long format
    data_check: [(기준일1, 기준가1), (기준일2, 기준가2)]
    """
    df_rate = pd.read_csv(f'{path}/{file}', parse_dates=[0], index_col=[0])
    if df_rate.columns.size > 1:
        print('WARNING: taking the 1st two columns only.')
    # make sure to get series
    df_rate = df_rate.iloc[:, 0]
    df_rate = df_rate.rename_axis(cols[0])
    
    col_data = cols[1]
    if col_data is None:
        col_data = file.split('.')[0]
    df_rate.name = col_data

    return df_rate


def import_rate2(file, path='.', cols=['date', None], n_headers=1):
    """
    file: historical of cumulative rate of return in wide format
    data_check: [(기준일1, 기준가1), (기준일2, 기준가2)]
    """
    df_rate = pd.read_csv(f'{path}/{file}')
    df_rate = df_rate.T.iloc[n_headers:, 0]

    df_rate.index = pd.to_datetime(df_rate.index)
    df_rate = df_rate.rename_axis(cols[0])
    
    col_data = cols[1]
    if col_data is None:
        col_data = file.split('.')[0]
    df_rate.name = col_data

    return df_rate
    

def get_price(df_rate, data_check, rate_is_percent=True):
    """
    calc price from rate of return
    """
    # date check
    for dt, _ in data_check:
        try:
            dt = pd.to_datetime(dt)
            rate = df_rate.loc[dt]
        except KeyError as e:
            return print(f'ERROR: KeyError {e}')
    
    # convert to price with data_check[0]
    dt, price = data_check[0]
    dt = pd.to_datetime(dt)
    rate = df_rate.loc[dt]
    if rate_is_percent:
        rate = rate/100
        df_rate = df_rate/100
    price_base = price / (rate+1)
    df_price = (df_rate + 1) * price_base 

    # check price
    dt, price = data_check[1]
    e = df_price.loc[dt]/price - 1
    print(f'error: {e*100:.2f} %')
    
    return df_price


def convert_rate_to_price(data, n_headers=1, path=None, 
                          rate_is_percent=True, df_rate=None, rate_only=False):
    """
    data: series or dict
    df_rate: historical given as dataframe
    """
    data_type = data['data_type']
    if data_type == 1:
        import_rate = import_rate1
    elif data_type == 2:
        import_rate = lambda *args, **kwargs: import_rate2(*args, n_headers=n_headers, **kwargs)
    else:
        if df_rate is None:
            return print(f'ERROR: no data type {data_type} exists')
        else:
            import_rate = lambda *args, **kwargs: df_rate.rename_axis(kwargs['cols'][0]).rename(kwargs['cols'][1])
    
    ticker = data['ticker']
    name = data['name']
    file = f'{data['file']}.csv'
    data_check = [
        (data['check1_date'], data['check1_price']),
        (data['check2_date'], data['check2_price']),
    ]
    
    df = import_rate(file, path=path, cols=['date', ticker])
    if rate_only:
       return df
        
    df = get_price(df, data_check, rate_is_percent=rate_is_percent)
    if df is None:
        return print(f'ERROR: check {ticker}')
    else:
        return df


def get_date_range(dfs, symbol_name=None, slice_input=False):
    """
    symbol_name: dict of symbols to names
    """
    df = dfs.apply(lambda x: x[x.notna()].index.min()).to_frame('start date')
    df = df.join(dfs.apply(lambda x: x[x.notna()].index.max()).to_frame('end date'))
    if symbol_name is not None:
        df = pd.Series(symbol_name).to_frame('name').join(df)

    if slice_input:
        start_date = df.iloc[:, 0].max()
        end_date = df.iloc[:, 1].min()
        return dfs.loc[start_date:end_date]
    else:
        return df.sort_values('start date')



def valuate_bond(face, rate, year, ytm, n_pay=1):
    """
    face: face value
    rate: coupon rate (annual)
    year: years to maturity
    ytm: discount rate (annual)
    n_pay: number of payments per year
    """
    c = face * rate / n_pay
    vc = 0
    r_discount = ytm/n_pay
    # calc The present value of expected cash flows
    for t in range(1, year*n_pay+1):
        vc += c/(1+r_discount)**t
    # the present value of the face value of the bond added
    return vc + face/(1+r_discount)**(year*n_pay)



class StaticPortfolio():
    """
    backtest fixed weight portfolio
    """
    def __init__(self, df_equity, align_axis=0, metrics=None, name_prfx='Portfolio', 
                 initial_capital=1000000, commissions=None, equity_names=None):
        # df of equities (equities in columns) which of each has its own periods.
        # the periods will be aligned for equities in a portfolio. see self.build
        if isinstance(df_equity, pd.Series):
            return print('ERROR: df_equity must be Dataframe')
        self.df_equity = df_equity
        self.align_axis = align_axis # how to set time periods intersection with equities
        self.portfolios = dict() # dict of bt.backtest.Backtest
        self.pf_weights = dict()
        self.metrics = metrics
        self.name_prfx = name_prfx
        self.n_names = 0 # see self._check_name
        self.initial_capital = initial_capital
        # commissions of all equities across portfolios (per year)
        self.commissions = commissions 
        self.equity_names = equity_names # names of all equities across portfolios
        self.run_results = None


    def align_period(self, df_equity, axis=0, dt_format='%Y-%m-%d',
                     fill_na=True, print_msg=True, n_indent=2):
        """
        axis: 0 : Drop time index which contain missing prices.
              1 : Drop equity columns whose length is less than max from missing value.
        fill_na: set False to drop nan fields
        """
        if axis == 0:
            df = get_date_range(df_equity, slice_input=True)
            if len(df) < len(df_equity):
                dts = [x.strftime(dt_format) for x in (df.index.min(), df.index.max())]
                print(f"period reset: {' ~ '.join(dts)}")
        elif axis == 1:
            c_all = df_equity.columns
            cond = df_equity.apply(lambda x: x.dropna().count()) < len(df_equity)
            c_drop = c_all[cond]
            df = df_equity[c_all.difference(c_drop)]
            n_c = len(c_drop)
            if n_c > 0:
                n_all = len(c_all)
                print(f'{n_c} equities removed for shorter periods ({n_c/n_all*100:.1f}%)')
        else:
            pass

        if print_msg:
            stats = df.isna().sum().div(df.count())
            t = 'filled forward' if fill_na else 'dropped'
            print(f'ratio of nan {t}::')
            indent = ' '*n_indent
            _ = [print(f'{indent}{i}: {stats[i]:.3f}') for i in stats.index]

        if fill_na:
            return df.ffill()
        else:
            return df.dropna()

    
    def _check_name(self, name=None):
        if name is None:
            self.n_names += 1
            name = f'{self.name_prfx}{self.n_names}'
        return name

    
    def _check_weights(self, dfs, weights):
        if weights is None:
            cols = dfs.columns
            weights = dict(zip(cols, [1/len(cols)]*len(cols)))
        return weights

    
    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg


    def _calc_commissions(self, commissions, weights, freq='Y', rate_is_percent=True):
        """
        commissions: dict of equity to fee
        """
        a = 100 if rate_is_percent else 1
        
        if freq == 'W':
            a *= 52
        elif freq == 'Q':
            a *= 4
        elif freq == 'M':
            a *= 12
        else:
            pass

        try:
            c = sum([v*commissions[k]/a for k,v in weights.items()])
            return lambda q, p: abs(q*p*c)
        except Exception as e:
            print(f'WARNING: commissions set to 0 as {e}')
            return None


    def backtest(self, dfs, weights=None, name='portfolio', 
                 run_freq=bt.algos.RunOnce(), 
                 capital_flow=0, **kwargs):
        """
        kwargs: keyword args for bt.Backtest
        """
        strategy = bt.Strategy(name, [
            bt.algos.SelectAll(),
            bt.algos.CapitalFlow(capital_flow),
            bt.algos.WeighSpecified(**weights),
            run_freq,
            bt.algos.Rebalance()
        ])
        return bt.Backtest(strategy, dfs, **kwargs)
        

    def build(self, weights=None, name=None, freq=None, 
              initial_capital=None, commissions=None, capital_flow=0,
              align_axis=None, fill_na=True):
        """
        make backtest of a strategy with tickers in weights
        """
        dfs = self.df_equity
        weights = self._check_weights(dfs, weights)
        name = self._check_name(name)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        align_axis = self._check_var(align_axis, self.align_axis)
        
        try:
            dfs = dfs[weights.keys()] # dataframe even if one weight given
        except KeyError as e:
            return print(f'ERROR: check weights as {e}')

        dfs = self.align_period(dfs, axis=align_axis, fill_na=fill_na)
                
        run_freq = self._get_run_freq(freq)
        
        commissions = self._check_var(commissions, self.commissions)
        if commissions is None:
            c_avg = None
        else:
            c_avg = self._calc_commissions(commissions, weights, freq)
        
        self.portfolios[name] = self.backtest(dfs, weights=weights, name=name, run_freq=run_freq, 
                                              capital_flow=capital_flow,
                                              initial_capital=initial_capital, commissions=c_avg)
        self.pf_weights[name] = weights
        return None


    def _get_run_freq(self, freq='M'):
        if freq == 'W':
            run_freq = RunWeekly()
        elif freq == 'Q':
            run_freq = RunQuarterly()
        elif freq == 'Y':
            run_freq = RunYearly()
        else: # default monthly
            run_freq = RunMonthly()
        return run_freq

    
    def buy_n_hold(self, weights=None, name=None, **kwargs):
        if isinstance(weights, str):
            weights = {weights: 1}
        return self.build(weights=weights, name=name, freq=None, **kwargs)


    def build_batch(self, kwa_list, reset_portfolios=False, **kwargs):
        """
        kwa_list: list of k/w args for each backtest
        kwargs: k/w args common for all backtest
        """
        if reset_portfolios:
            self.portfolios = {}
        else:
            return print('WARNING: set reset_portfolios to True to run')

        for kwa in kwa_list:
            self.build(**{**kwa, **kwargs})
        return None

    
    def run(self, pf_list=None, metrics=None, plot=True, freq='d', figsize=None, stats=True):
        """
        pf_list: List of backtests or list of index of backtest
        """
        if len(self.portfolios) == 0:
            return print('ERROR: no strategy to backtest. build strategies first')
            
        if pf_list is None:
            bt_list = self.portfolios.values()
        else:
            c = [0 if isinstance(x, int) else 1 for x in pf_list]
            if sum(c) == 0: # pf_list is list of index
                bt_list = [x for i, x in enumerate(self.portfolios.values()) if i in pf_list]
            else: # pf_list is list of names
                bt_list = [v for k, v in self.portfolios.items() if k in pf_list]

        try:
            results = bt.run(*bt_list)
        except Exception as e:
            return print(f'ERROR: {e}')
            
        self.run_results = results
        
        if plot:
            results.plot(freq=freq, figsize=figsize);

        if stats:
            print('Returning stats')
            return self.get_stats(pf_list=pf_list, metrics=metrics)
        else:
            print('Returning backtest results')
            return results


    def check_portfolios(self, pf_list=None, run=True, convert_index=True):
        """
        convert_index: convert pf_list of index to pf_list of portfolio names 
        """
        if run:
            if self.run_results is None:
                return print('ERROR: run backtest first')
            else:
                pf_list_all = list(self.run_results.keys())
        else:
            pf_list_all = list(self.portfolios.keys())
    
        if pf_list is None:
            return pf_list_all
            
        if not isinstance(pf_list, list):
            pf_list = [pf_list]
    
        try: # assuming list of int
            if max(pf_list) >= len(pf_list_all):
                print('WARNING: check pf_list')
                pf_list = pf_list_all
            else:
                if convert_index:
                    pf_list = [pf_list_all[x] for x in pf_list]

        except TypeError: # pf_list is list of str
            if len(set(pf_list) - set(pf_list_all)) > 0:
                print('WARNING: check pf_list')
                pf_list = pf_list_all
            
        return pf_list


    def get_stats(self, pf_list=None, metrics=None, sort_by=None):
        pf_list  = self.check_portfolios(pf_list, run=True)
        if pf_list is None:
            return None
        else:
            results = self.run_results
            
        metrics = self._check_var(metrics, self.metrics)
        if (metrics is None) or (metrics == 'all'):
            df = results.stats[pf_list]
        else:
            metrics = ['start', 'end'] + metrics
            df = results.stats.loc[metrics, pf_list]

        if sort_by is not None:
            try:
                df = df.sort_values(sort_by, axis=1, ascending=False)
            except KeyError as e:
                print(f'WARNING: no sorting as {e}')

        return df


    def _plot_portfolios(self, plot_func, pf_list, ncols=2, sharex=True, sharey=True, 
                         figsize=(10,5), legend=True):
        n = len(pf_list)
        if n == 1:
            ncols = 1
            
        nrows = n // ncols + min(n % ncols, 1)
        fig, axes = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                                #figsize=figsize
                               )
        if nrows == 1:
            axes = [axes]
            if ncols == 1:
                axes = [axes]
        
        k = 0
        finished = False
        for i in range(nrows):
            for j in range(ncols):
                ax = axes[i][j]
                _ = plot_func(pf_list[k], ax=ax, legend=legend, figsize=figsize)
                k += 1
                if k == n:
                    finished = True
                    break
            if finished:
                break
    
    
    def plot_security_weights(self, pf_list=None, **kwargs):
        pf_list  = self.check_portfolios(pf_list, run=True)
        if pf_list is None:
            return None
        
        plot_func = self.run_results.plot_security_weights
        return self._plot_portfolios(plot_func, pf_list, **kwargs)
        

    def plot_weights(self, pf_list=None, **kwargs):
        pf_list  = self.check_portfolios(pf_list, run=True)
        if pf_list is None:
            return None
        
        plot_func = self.run_results.plot_weights
        return self._plot_portfolios(plot_func, pf_list, **kwargs)


    def plot_histogram(self, pf_list=None, **kwargs):
        pf_list  = self.check_portfolios(pf_list, run=True)
        if pf_list is None:
            return None
        
        if len(pf_list) > 1:
            print('WARNING: passed axis not bound to passed figure')

        for x in pf_list:
            _ = self.run_results.plot_histogram(x, **kwargs)
        return None


    def get_weights(self, pf_list=None, equity_names=None, as_series=True):
        pf_list  = self.check_portfolios(pf_list, run=False, convert_index=True)
        if pf_list is None:
            return None
        
        weights = {k: self.pf_weights[k] for k in pf_list}
        quity_names = self._check_var(equity_names, self.equity_names)
        if equity_names is not None:
            weights = {k: {equity_names[x]:y for x,y in v.items()} for k,v in weights.items()}
        if as_series:
            weights = pd.DataFrame().from_dict(weights).fillna(0)
        return weights


    def _retrieve_results(self, pf_list, func_result):
        """
        generalized function to retrieve results of pf_list from func_result
        func_result is func with ffn.core.PerformanceStats or bt.backtest.Backtest
        """
        pf_list  = self.check_portfolios(pf_list, run=True, convert_index=True)
        if pf_list is None:
            return None

        df_all = None
        for rp in pf_list:
            df = func_result(rp)
            if df_all is None:
                df_all = df.to_frame()
            else:
                df_all = df_all.join(df)
        return df_all


    def get_historical(self, pf_list=None):
        func_result = lambda x: self.run_results[x].prices
        return self._retrieve_results(pf_list, func_result)


    def get_turnover(self, pf_list=None, drop_zero=True):
        """
        Calculate the turnover for the backtest
        """
        func_result = lambda x: self.portfolios[x].turnover.rename(x)
        df = self._retrieve_results(pf_list, func_result)

        if drop_zero:
            df = df.loc[(df.sum(axis=1) > 0)]
        return df

    
    def get_security_weights(self, pf=0):
        pf_list  = self.check_portfolios(pf, run=True, convert_index=True)
        if pf_list is None:
            return None
        else:
            pf = pf_list[0]
        return self.run_results.get_security_weights(pf)
        

    def get_transactions(self, pf=0):
        if isinstance(pf, list):
            return print('WARNING: set one portfolio')
            
        pf_list  = self.check_portfolios(pf, run=True, convert_index=True)
        if pf_list is None:
            return None
        else:
            pf = pf_list[0]
            
        return self.run_results.get_transactions(pf)



class DynamicPortfolio(StaticPortfolio):
    def __init__(self, df_equity, align_axis=0, metrics=None, name_prfx='Portfolio', 
                  initial_capital=1000000, commissions=None, equity_names=None):
        self.df_equity = df_equity
        self.align_axis = align_axis
        self.portfolios = dict()
        #self.pf_weights = dict()
        self.metrics = metrics
        self.name_prfx = name_prfx
        self.n_names = 0 # see self._check_name
        self.initial_capital = initial_capital
        # commissions of all equities across portfolios (per year)
        self.commissions = commissions 
        self.equity_names = equity_names # names of all equities across portfolios
        self.run_results = None


    def backtest(self, dfs, 
                 algo_select=SelectMomentum, n_equities=2, lookback=12, lag=30, 
                 name='portfolio', run_freq=RunMonthly(), **kwargs):
        """
        lookback: month
        lag: day
        kwargs: keyword args for bt.Backtest
        """
        strategy = bt.Strategy(name, [
            bt.algos.SelectAll(),
            run_freq,
            algo_select(n=n_equities, lookback=pd.DateOffset(months=lookback),
                       lag=pd.DateOffset(days=lag)),
            bt.algos.WeighERC(lookback=pd.DateOffset(months=lookback)),
            bt.algos.Rebalance()
        ])
        return bt.Backtest(strategy, dfs, **kwargs)


    def build(self, 
              method='simple', n_equities=2, lookback=12, lag=30, 
              name=None, freq='M', 
              initial_capital=None, commissions=None, rate_is_percent=True,
              align_axis=None, fill_na=True):
        """
        make backtest of a strategy with tickers in weights
        method: rule how to select equities
        lookback: month
        lag: day
        commissions: same for all equities
        """
        dfs = self.df_equity
        name = self._check_name(name)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        align_axis = self._check_var(align_axis, self.align_axis)
                
        dfs = self.align_period(dfs, axis=align_axis, fill_na=fill_na, print_msg=False)
        
        run_freq = self._get_run_freq(freq)

        if commissions is not None:
            c = 100 if rate_is_percent else 1
            c = commissions * c
            commissions = lambda q, p: abs(q*p*c)

        algo_select = self._get_algo_select(method)
        
        self.portfolios[name] = self.backtest(dfs, algo_select=algo_select,
                                              n_equities=n_equities, 
                                              lookback=lookback, lag=lag,
                                              name=name, run_freq=run_freq, 
                                              initial_capital=initial_capital, 
                                              commissions=commissions)
        return None


    def _get_algo_select(self, method='simple'):
        if method == 'k-ratio':
            return SelectKRatio
        else:
            return SelectMomentum


    def benchmark(self, dfs, name='BM', align_axis=None):
        """
        dfs: str or list of str if dfs in self.df_equity or historical of tickers
        """
        df_equity = self.df_equity
        
        if isinstance(dfs, str):
            dfs = [dfs]

        if isinstance(dfs, list):
            if pd.Index(dfs).isin(self.df_equity.columns).sum() != len(dfs):
                return print('ERROR: check arg dfs')
            else:
                dfs = df_equity[dfs]
        else:
            dfs = dfs.loc[self.df_equity.index]
            
        if isinstance(dfs, pd.Series):
            if dfs.name is None:
                dfs = dfs.to_frame(name)
                weights = name
            else:
                dfs = dfs.to_frame()
                weights = None
        else:
            weights = None # equal weights
            
        spf = StaticPortfolio(dfs, initial_capital=self.initial_capital)
        spf.buy_n_hold(weights, name=name, align_axis=align_axis)
        self.portfolios[name] = spf.portfolios[name]
        return None

    
    def build_batch(self, *args, **kwargs):
        return super().build_batch(*args, **kwargs)

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)

    def get_stats(self, *args, **kwargs):
        return super().get_stats(*args, **kwargs)

    def _check_name(self, *args, **kwargs):
        return super()._check_name(*args, **kwargs)

    def _check_var(self, *args, **kwargs):
        return super()._check_var(*args, **kwargs)

    def _get_run_freq(self,  *args, **kwargs):
        return super()._get_run_freq(*args, **kwargs)

    def align_period(self, *args, **kwargs):
        return super().align_period(*args, **kwargs)

    def plot_security_weights(self, *args, **kwargs):
        return super().plot_security_weights(*args, **kwargs)

    def plot_weights(self, *args, **kwargs):
        return super().plot_weights(*args, **kwargs)

    def plot_histogram(self, *args, **kwargs):
        return super().plot_histogram(*args, **kwargs)

    def get_historical(self, *args, **kwargs):
        return super().get_historical(*args, **kwargs)

    def get_turnover(self, *args, **kwargs):
        return super().get_turnover(*args, **kwargs)
    
    def get_security_weights(self, *args, **kwargs):
        return super().get_security_weights(*args, **kwargs)
        
    def get_transactions(self, *args, **kwargs):
        return super().get_transactions(*args, **kwargs)
        


class AssetEvaluator():
    def __init__(self, df_prices, days_in_year=252):
        # df of equities (equities in columns) which of each might have its own periods.
        # the periods of all equities will be aligned in every calculation.
        df_prices = df_prices.to_frame() if isinstance(df_prices, pd.Series) else df_prices
        if df_prices.index.name is None:
            df_prices.index.name = 'date' # set index name to run check_days_in_year
        _ = self.check_days_in_year(df_prices, days_in_year, freq='M')
        
        self.df_prices = df_prices
        self.days_in_year = days_in_year
        self.bayesian_data = None
        

    def check_days_in_year(self, df, days_in_year, freq='M'):
        """
        freq: freq to check days_in_year in df
        """
        if freq == 'Y':
            grp_format = '%Y'
            #days_in_freq = days_in_year
            factor = 1
        elif freq == 'W':
            grp_format = '%Y%m%U'
            #days_in_freq = round(days_in_year/12/WEEKS_IN_YEAR)
            factor = 12 * WEEKS_IN_YEAR
        else: # default month
            grp_format = '%Y%m'
            #days_in_freq = round(days_in_year/12)
            factor = 12

        # calc mean days for each equity
        days_freq_calc = (df.assign(gb=df.index.strftime(grp_format)).set_index('gb')
                            .apply(lambda x: x.dropna().groupby('gb').count()[1:-1]
                            .mean().round()))

        cond = (days_freq_calc != round(days_in_year / factor))
        if cond.sum() > 0:
            days_in_year_new = days_freq_calc.loc[cond].mul(factor).round()
            n = len(days_in_year_new)
            if n < 5:
                print(f'WARNING: the number of days in a year with followings is not {days_in_year} in setting:')
                _ = [print(f'{k}: {int(v)}') for k,v in days_in_year_new.to_dict().items()]
            else:
                print(f'WARNING: the number of days in a year with {n} equities is not {days_in_year} in setting:')
            return days_in_year_new
        else:
            return days_in_year


    def get_freq_days(self, freq='daily'):
        if freq == 'yearly':
            n = self.days_in_year
        elif freq == 'monthly':
            n = round(self.days_in_year/12)
        elif freq == 'weekly':
            n = round(self.days_in_year/WEEKS_IN_YEAR)
        else: # default daily
            n = 1
            freq = 'daily'
        return (n, freq)

        
    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg


    def calc_cagr(self, df_prices=None, days_in_year=None, align_period=False):
        # calc cagr's of equities
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        if align_period:
            df_prices = self.align_period(df_prices, fill_na=True)
        return df_prices.apply(lambda x: self._calc_cagr(x, days_in_year))


    def _calc_cagr(self, sr_prices, days_in_year):
        # calc cagr of a equity
        sr = sr_prices.ffill().dropna()
        t = days_in_year / len(sr)
        return (sr.iloc[-1]/sr.iloc[0]) ** t - 1


    def calc_mean_return(self, df_prices=None, days_in_year=None, freq='daily', annualize=True):
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        periods, _ = self.get_freq_days(freq)
        scale = (days_in_year/periods) if annualize else 1
        return df_prices.apply(lambda x: self._calc_mean_return(x, periods, scale))
        

    def _calc_mean_return(self, sr_prices, periods, scale):
        return sr_prices.pct_change(periods).dropna().mean() * scale
    

    def calc_volatility(self, df_prices=None, days_in_year=None, freq='daily', annualize=True):
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        periods, _ = self.get_freq_days(freq)
        scale = (days_in_year/periods) ** .5 if annualize else 1
        return df_prices.apply(lambda x: self._calc_volatility(x, periods, scale))
        

    def _calc_volatility(self, sr_prices, periods, scale):
        return sr_prices.pct_change(periods).dropna().std() * scale
    

    def calc_sharpe(self, df_prices=None, days_in_year=None, freq='daily', annualize=True, rf=0):
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        periods, _ = self.get_freq_days(freq)
        scale = (days_in_year/periods) ** .5 if annualize else 1
        return df_prices.apply(lambda x: self._calc_sharpe(x, periods, scale, rf))
        

    def _calc_sharpe(self, sr_prices, periods, scale, rf):
        mean = self._calc_mean_return(sr_prices, periods, 1)
        std = self._calc_volatility(sr_prices, periods, 1)
        return (mean - rf) / std * scale


    def summary(self, freq='yearly', annualize=True, rf=0, align_period=False):
        df_prices = self.df_prices
        if align_period:
            df_prices = self.align_period(df_prices, fill_na=True)

        _, freq = self.get_freq_days(freq) # check freq for naming
        kwargs = dict(
            df_prices=df_prices, freq=freq, annualize=annualize
        )
        df = df_prices.apply(lambda x: f'{len(x.dropna())/self.days_in_year:.1f}')
        # work even with df_prices of single asset as df_prices is always series (see __init__)
        return df.to_frame('years').join(
            self.calc_cagr(df_prices).to_frame('cagr').join(
                self.calc_mean_return(**kwargs).to_frame(f'{freq}_mean').join(
                    self.calc_volatility(**kwargs).to_frame(f'{freq}_vol').join(
                        self.calc_sharpe(rf=rf, **kwargs).to_frame(f'{freq}_sharpe')
                    )
                )
            )
        ).T


    def bayesian_sample(self, freq='yearly', annualize=True, rf=0, align_period=False,
                        sample_draws=1000, sample_tune=1000, target_accept=0.9,
                        multiplier_std=1000, 
                        rate_nu = 29, normality_sharpe=True, debug_annualize=False):
        """
        normality_sharpe: set to True if 
         -. You are making comparisons to Sharpe ratios calculated under the assumption of normality.
         -. You want to account for the higher variability due to the heavy tails of the t-distribution.
        """
        days_in_year = self.days_in_year
        periods, freq = self.get_freq_days(freq)
        factor_year = days_in_year/periods if annualize else 1

        df_prices = self.df_prices
        assets = list(df_prices.columns)
        
        if align_period:
            df_prices = self.align_period(df_prices, fill_na=True)
            df_ret = df_prices.pct_change(periods).dropna() * factor_year
            mean_prior = df_ret.mean()
            std_prior = df_ret.std()
            std_low = std_prior / multiplier_std
            std_high = std_prior * multiplier_std
        else:
            ret_list = [df_prices[x].pct_change(periods).dropna() * factor_year for x in assets]
            mean_prior = [x.mean() for x in ret_list]
            std_prior = [x.std() for x in ret_list]
            std_low = [x / multiplier_std for x in std_prior]
            std_high = [x * multiplier_std for x in std_prior]
            returns = dict()
        
        num_assets = len(assets) # flag for comparisson of two assets
        coords={'asset': assets}

        with pm.Model(coords=coords) as model:
            # nu: degree of freedom (normality parameter)
            nu = pm.Exponential('nu_minus_two', 1 / rate_nu, testval=4) + 2.
            mean = pm.Normal('mean', mu=mean_prior, sigma=std_prior, dims='asset')
            std = pm.Uniform('vol', lower=std_low, upper=std_high, dims='asset')
            
            if align_period:
                returns = pm.StudentT(f'{freq}_returns', nu=nu, mu=mean, sigma=std, observed=df_ret)
            else:
                func = lambda x: dict(mu=mean[x], sigma=std[x], observed=ret_list[x])
                returns = {i: pm.StudentT(f'{freq}_returns[{x}]', nu=nu, **func(i)) for i, x in enumerate(assets)}

            fy2 = 1 if debug_annualize else factor_year
            pm.Deterministic(f'{freq}_mean', mean * fy2, dims='asset')
            pm.Deterministic(f'{freq}_vol', std * (fy2 ** .5), dims='asset')
            std_sr = std * pt.sqrt(nu / (nu - 2)) if normality_sharpe else std
            sharpe = pm.Deterministic(f'{freq}_sharpe', ((mean-rf) / std_sr) * (fy2 ** .5), dims='asset')
            
            if num_assets == 2:
                #mean_diff = pm.Deterministic('mean diff', mean[0] - mean[1])
                #pm.Deterministic('effect size', mean_diff / (std[0] ** 2 + std[1] ** 2) ** .5 / 2)
                sharpe_diff = pm.Deterministic('sharpe diff', sharpe[0] - sharpe[1])
    
            trace = pm.sample(draws=sample_draws, tune=sample_tune,
                              #chains=chains, cores=cores,
                              target_accept=target_accept,
                              #return_inferencedata=False, # TODO: what's for?
                              progressbar=True)
            
        self.bayesian_data = {'trace':trace, 'coords':coords, 'align_period':align_period,
                              'freq':freq, 'annualize':annualize, 'rf':rf}
        return None
        
    
    def bayesian_summary(self, var_names=None, filter_vars='like', **kwargs):
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
            return az.summary(trace, var_names=var_names, filter_vars=filter_vars, **kwargs)


    def bayesian_plot(self, var_names=None, filter_vars='like', ref_val=None, **kwargs):
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
            coords = self.bayesian_data['coords']
            freq = self.bayesian_data['freq']
            annualize = self.bayesian_data['annualize']
            rf = self.bayesian_data['rf']
            align_period = self.bayesian_data['align_period']

        if ref_val is None:
            df = self.summary(freq=freq, annualize=annualize, rf=rf, align_period=align_period)
            metrics = [x for x in df.index if x.startswith(freq)]
            ref_val = df.loc[metrics].to_dict(orient='index')
            col_name = list(coords.keys())[0]
            ref_val = {k: [{col_name:at, 'ref_val':rv} for at, rv in v.items()] for k,v in ref_val.items()}
        ref_val.update({'mean diff': [{'ref_val': 0}], 'sharpe diff': [{'ref_val': 0}]})

        _ = az.plot_posterior(trace, var_names=var_names, filter_vars=filter_vars,
                              ref_val=ref_val, **kwargs)
        #return ref_val
        return None


    def plot_trace(self, var_names=None, filter_vars='like', legend=False, figsize=(12,6), **kwargs):
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
            return az.plot_trace(trace, var_names=var_names, filter_vars=filter_vars, 
                                 legend=legend, figsize=figsize, **kwargs)


    def plot_energy(self, **kwargs):
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
            return az.plot_energy(trace, **kwargs)


    def align_period(self, df, fill_na=True):
        return Backtest(pd.Series()).align_period(df, fill_na=fill_na)
