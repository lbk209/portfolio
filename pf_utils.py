import bt
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
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
    def __init__(self, df_equity, metrics=None, name_prfx='Portfolio', 
                 initial_capital=1000000, commissions=None, equity_names=None):
        # df of equities (equities in columns) which of each has its own periods.
        # the periods will be aligned for equities in a portfolio. see self.build
        if isinstance(df_equity, pd.Series):
            return print('ERROR: df_equity must be Dataframe')
        self.df_equity = df_equity
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


    def align_period(self, df_equity, dt_format='%Y-%m-%d', n_indent=2, fill_na=True, print_msg=True):
        """
        fill_na: set False to drop nan fields
        """
        df = get_date_range(df_equity, slice_input=True)
        dts = [x.strftime(dt_format) for x in (df.index.min(), df.index.max())]
        print(f"period reset: {' ~ '.join(dts)}")

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
              fill_na=True):
        """
        make backtest of a strategy with tickers in weights
        """
        dfs = self.df_equity
        weights = self._check_weights(dfs, weights)
        name = self._check_name(name)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        
        try:
            dfs = dfs[weights.keys()] # dataframe even if one weight given
        except KeyError as e:
            return print(f'ERROR: check weights as {e}')

        dfs = self.align_period(dfs, fill_na=fill_na)
                
        if freq == 'W':
            run_freq = bt.algos.RunWeekly()
        elif freq == 'Q':
            run_freq = bt.algos.RunQuarterly()
        elif freq == 'Y':
            run_freq = bt.algos.RunYearly()
        elif freq == 'M':
            run_freq = bt.algos.RunMonthly()
        else: # default: buy & hold
            run_freq = bt.algos.RunOnce()
        
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


    def get_stats(self, pf_list=None, metrics=None):
        pf_list  = self.check_portfolios(pf_list, run=True)
        if pf_list is None:
            return None
        else:
            results = self.run_results
            
        metrics = self._check_var(metrics, self.metrics)
        if (metrics is None) or (metrics == 'all'):
            return results.stats[pf_list]
        else:
            metrics = ['start', 'end'] + metrics
            return results.stats.loc[metrics, pf_list]


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


    def get_historical(self, pf_list=None):
        """
        calc weighted sum of securities for each portfolio
        """
        pf_list  = self.check_portfolios(pf_list, run=True, convert_index=True)
        if pf_list is None:
            return None

        df_all = None
        for rp in pf_list:
            # ffn.core.PerformanceStats get only a key or None
            df = self.run_results[rp].prices
            if df_all is None:
                df_all = df.to_frame()
            else:
                df_all = df_all.join(df)
        return df_all


    def get_turnover(self, pf_list=None, drop_zero=True):
        """
        Calculate the turnover for the backtest
        """
        pf_list  = self.check_portfolios(pf_list, run=True, convert_index=True)
        if pf_list is None:
            return None

        df_all = None
        for rp in pf_list:
            # turnover saved in bt.backtest.Backtest not ffn.core.PerformanceStats
            df = self.portfolios[rp].turnover.rename(rp)
            if df_all is None:
                df_all = df.to_frame()
            else:
                df_all = df_all.join(df)

        if drop_zero:
            df_all = df_all.loc[(df_all.sum(axis=1) > 0)]
        return df_all



class DynamicPortfolio(StaticPortfolio):
    def __init__(self, df_equity, metrics=None, name_prfx='Portfolio', 
                  initial_capital=1000000, commissions=None, equity_names=None):
        self.df_equity = df_equity
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


    def backtest(self, dfs, n_equities=2, lookback_months=12, name='portfolio',                 
                 run_freq=bt.algos.RunMonthly(), **kwargs):
        """
        kwargs: keyword args for bt.Backtest
        """
        strategy = bt.Strategy(name, [
            bt.algos.SelectAll(),
            bt.algos.SelectMomentum(n=n_equities, lookback=pd.DateOffset(months=lookback_months)),
            bt.algos.WeighERC(lookback=pd.DateOffset(months=lookback_months)),
            run_freq,
            bt.algos.Rebalance()
        ])
        return bt.Backtest(strategy, dfs, **kwargs)


    def build(self, n_equities=2, lookback_months=12, name=None, freq='M', 
              initial_capital=None, commissions=None, rate_is_percent=True,
              fill_na=True):
        """
        make backtest of a strategy with tickers in weights
        commissions: same for all equities
        """
        dfs = self.df_equity
        dfs = self.align_period(dfs, fill_na=fill_na, print_msg=False)
        
        name = self._check_name(name)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
                
        if freq == 'W':
            run_freq = bt.algos.RunWeekly()
        elif freq == 'Q':
            run_freq = bt.algos.RunQuarterly()
        elif freq == 'Y':
            run_freq = bt.algos.RunYearly()
        else: # default monthly
            run_freq = bt.algos.RunMonthly()

        if commissions is not None:
            c = 100 if rate_is_percent else 1
            c = commissions * c
            commissions = lambda q, p: abs(q*p*c)
            
        self.portfolios[name] = self.backtest(dfs, n_equities=n_equities, 
                                              lookback_months=lookback_months, 
                                              name=name, run_freq=run_freq, 
                                              initial_capital=initial_capital, 
                                              commissions=commissions)
        return None


    def benchmark(self, dfs, name='BM'):
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
        spf.buy_n_hold(weights, name=name)
        self.portfolios[name] = spf.portfolios[name]
        return None


    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)

    def get_stats(self, *args, **kwargs):
        return super().get_stats(*args, **kwargs)

    def _check_name(self, *args, **kwargs):
        return super()._check_name(*args, **kwargs)

    def _check_var(self, *args, **kwargs):
        return super()._check_var(*args, **kwargs)

    def align_period(self, *args, **kwargs):
        return super().align_period(*args, **kwargs)

    def plot_security_weights(self, *args, **kwargs):
        return super().plot_security_weights(*args, **kwargs)

    def plot_weights(self, *args, **kwargs):
        return super().plot_weights(*args, **kwargs)

    def plot_histogram(self, *args, **kwargs):
        return super().plot_histogram(*args, **kwargs)