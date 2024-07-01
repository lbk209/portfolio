import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import FinanceDataReader as fdr
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

import bt
from pf_custom import SelectKRatio

import warnings

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)

# support korean lang
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumBarunGothic'

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


def check_days_in_year(df, days_in_year=252, freq='M'):
    """
    freq: unit to check days_in_year in df
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
        df_days = days_freq_calc.loc[cond].mul(factor).round()
        n = len(df_days)
        if n < 5:
            print(f'WARNING: the number of days in a year with followings is not {days_in_year} in setting:')
            _ = [print(f'{k}: {int(v)}') for k,v in df_days.to_dict().items()]
        else:
            print(f'WARNING: the number of days in a year with {n} equities is not {days_in_year} in setting:')
        return df_days
    else:
        return None



class BacktestManager():
    def __init__(self, df_equity, align_axis=0, metrics=metrics, name_prfx='Portfolio', 
                 initial_capital=1000000, commissions=None, days_in_year=252):
        # df of equities (equities in columns) which of each has its own periods.
        # the periods will be aligned for equities in a portfolio. see self.build
        if isinstance(df_equity, pd.Series):
            return print('ERROR: df_equity must be Dataframe')

        self.df_equity = df_equity
        self.align_axis = align_axis # how to set time periods intersection with equities
        self.portfolios = dict() # dict of bt.backtest.Backtest
        self.metrics = metrics
        self.name_prfx = name_prfx
        self.n_names = 0 # see self._check_name
        self.initial_capital = initial_capital
        # commissions of all equities across portfolios
        self.commissions = commissions  # unit %
        self.run_results = None
        self.days_in_year = days_in_year
        _ = check_days_in_year(df_equity, days_in_year, freq='M')


    def align_period(self, df_equity, axis=0, dt_format='%Y-%m-%d',
                     fill_na=True, print_msg1=True, print_msg2=True, n_indent=2):
        """
        axis: 0 : Drop time index which contain missing prices.
              1 : Drop equity columns whose length is less than max from missing value.
        fill_na: set False to drop nan fields
        """
        msg1 = None
        if axis == 0:
            df_aligned = get_date_range(df_equity, slice_input=True)
            if len(df_aligned) < len(df_equity):
                dts = [x.strftime(dt_format) for x in (df_aligned.index.min(), df_aligned.index.max())]
                msg1 = f"period reset: {' ~ '.join(dts)}"
        elif axis == 1:
            c_all = df_equity.columns
            df_cnt = df_equity.apply(lambda x: x.dropna().count())
            cond = (df_cnt < df_cnt.max())
            c_drop = c_all[cond]
            df_aligned = df_equity[c_all.difference(c_drop)]
            n_c = len(c_drop)
            if n_c > 0:
                n_all = len(c_all)
                msg1 = f'{n_c} equities removed for shorter periods ({n_c/n_all*100:.1f}%)'
        else:
            pass

        if print_msg1:
            print(msg1) if msg1 is not None else None
            if print_msg2:
                stats = df_aligned.isna().sum().div(df.count())
                t = 'filled forward' if fill_na else 'dropped'
                print(f'ratio of nan {t}::')
                indent = ' '*n_indent
                _ = [print(f'{indent}{i}: {stats[i]:.3f}') for i in stats.index]

        if fill_na:
            return df_aligned.ffill()
        else:
            return df_aligned.dropna()

    
    def _check_name(self, name=None):
        if name is None:
            self.n_names += 1
            name = f'{self.name_prfx}{self.n_names}'
        return name

    
    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg


    def _check_weights(self, weights, dfs):
        """
        weights: str, list of str, dict, or None
        """
        if isinstance(weights, str):
            if weights in dfs.columns:
                return {weights: 1}
            else:
                return print(f'ERROR: No {weights} in the dfs')
        elif isinstance(weights, list):
            cols = pd.Index(weights).difference(dfs.columns)
            if len(cols) == 0:
                return {k:1/len(weights) for k in weights}
            else:
                cols = ', '.join(cols)
                return print(f'ERROR: No {cols} in the dfs')
        else: # assuming dict
            return weights


    def _check_algos(self, select, freq, weigh):
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        # managed to make it work
        if cond(select['select'], 'randomly') and cond(weigh['weigh'], 'ERC'):
            #return print('WARNING: random select does not work with ERC weighting')
            return None
        else:
            return None
    

    def backtest(self, dfs, name='portfolio', 
                 select={'select':'all'}, freq={'freq':'year'}, weigh={'weigh':'equally'},
                 algos=None, commissions=None, **kwargs):
        """
        kwargs: keyword args for bt.Backtest except commissions
        algos: List of Algos
        """
        _ = self._check_algos(select, freq, weigh)
        if algos is None:
            algos = [
                self._get_algo_select(**select), 
                self._get_algo_freq(**freq), 
                self._get_algo_weigh(**weigh),
                bt.algos.Rebalance()
            ]
        strategy = bt.Strategy(name, algos)
        if commissions is not None:
            c = lambda q, p: abs(q) * p * commissions
        return bt.Backtest(strategy, dfs, commissions=c, **kwargs)


    def _get_algo_select(self, select='all', n_equities=0, lookback=0, lag=0):
        """
        select: all, momentum, kratio, randomly
        """
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        
        if cond(select, 'Momentum'):
            algo_select = bt.algos.SelectMomentum(n=n_equities, lookback=pd.DateOffset(months=lookback),
                                                  lag=pd.DateOffset(days=lag))
            # SelectAll() or similar should be called before SelectMomentum(), 
            # as StatTotalReturn uses values of temp[‘selected’]
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'k-ratio'):
            algo_select = SelectKRatio(n=n_equities, lookback=pd.DateOffset(months=lookback),
                                       lag=pd.DateOffset(days=lag))
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'randomly'):
            algo_select = bt.algos.SelectRandomly(n=n_equities)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'all'):
            algo_select = bt.algos.SelectAll()
        else:
            print('SelectAll selected')
            algo_select = bt.algos.SelectAll()
            
        return algo_select
        

    def _get_algo_freq(self, freq='M', offset=0, days_in_year=252):
        """
        freq: W, M, Q, Y
        """
        cond = lambda x, y: False if x is None else x[0].lower() == y[0].lower()
        if cond(freq, 'W'):
            n = round(days_in_year / WEEKS_IN_YEAR)
        elif cond(freq, 'M'):
            n = round(days_in_year / 12)
        elif cond(freq, 'Q'):
            n = round(days_in_year / 4)
        elif cond(freq, 'Y'):
            n = days_in_year
        else:  # default run once
            n = -1

        if n > 0:
            algo_freq = bt.algos.RunEveryNPeriods(n, offset=offset)
        else:
            print('RunOnce selected')
            algo_freq = bt.algos.RunOnce()
        return algo_freq


    def _get_algo_weigh(self, weigh='equally', 
                         weights=None, lookback=0, lag=0, rf=0, bounds=(0.0, 1.0)):
        """
        weigh: equally, erc, specified, randomly, invvol, meanvar
        lookback: month
        lag: day
        """
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        
        # reset weigh if weights not given
        if cond(weigh, 'Specified') and (weights is None):
            weigh = 'equally'
        
        if cond(weigh, 'ERC'):
            algo_weigh = bt.algos.WeighERC(lookback=pd.DateOffset(months=lookback), 
                                          lag=pd.DateOffset(days=lag))
            # Use SelectHasData to avoid LedoitWolf ERROR; other weights like InvVol work fine without it.
            algo_weigh = bt.AlgoStack(bt.algos.SelectHasData(lookback=pd.DateOffset(months=lookback)), 
                                      algo_weigh)
        elif cond(weigh, 'Specified'):
            algo_weigh = bt.algos.WeighSpecified(**weights)
        elif cond(weigh, 'Randomly'):
            algo_weigh = bt.algos.WeighRandomly()
        elif cond(weigh, 'InvVol'): # risk parity
            algo_weigh = bt.algos.WeighInvVol(lookback=pd.DateOffset(months=lookback), 
                                             lag=pd.DateOffset(days=lag))
        elif cond(weigh, 'MeanVar'): # Markowitz’s mean-variance optimization
            algo_weigh = bt.algos.WeighMeanVar(lookback=pd.DateOffset(months=lookback), 
                                              lag=pd.DateOffset(days=lag),
                                              rf=rf, bounds=bounds)
        elif cond(weigh, 'equally'):
            algo_weigh = bt.algos.WeighEqually()
        else:
            print('WeighEqually selected')
            algo_weigh = bt.algos.WeighEqually()
            
        return algo_weigh
        

    def build(self, name=None, 
              freq='M', offset=0,
              select='all', n_equities=0, lookback=0, lag=0,
              weigh='equally', weights=None, rf=0, bounds=(0.0, 1.0),
              initial_capital=None, commissions=None, 
              align_axis=None, fill_na=True, algos=None):
        """
        make backtest of a strategy
        lookback: month
        lag: day
        commissions: %; same for all equities
        algos: set List of Algos to build backtest directly
        """
        dfs = self.df_equity
        name = self._check_name(name)
        align_axis = self._check_var(align_axis, self.align_axis)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        commissions = self._check_var(commissions, self.commissions)
       
        dfs = self.align_period(dfs, axis=align_axis, fill_na=fill_na, print_msg2=False)
        weights = self._check_weights(weights, dfs)
        
        select = {'select':select, 'n_equities':n_equities, 'lookback':lookback, 'lag':lag}
        freq = {'freq':freq, 'offset':offset, 'days_in_year':self.days_in_year}
        weigh = {'weigh':weigh, 'weights':weights, 'rf':rf, 'bounds':bounds,
                 'lookback':lookback, 'lag':lag}

        self.portfolios[name] = self.backtest(dfs, name=name, 
                                              select=select, freq=freq, weigh=weigh, algos=algos,
                                              initial_capital=initial_capital, 
                                              commissions=commissions)
        return None
    
    
    def buy_n_hold(self, name=None, weights=None, **kwargs):
        """
        weights: dict of ticker to weight. str if one equity portfolio
        kwargs: set initial_capital, commissions, align_axis or fill_na 
                to use different onces with other strategies
        """
        return self.build(name=name, freq=None, select='all', weigh='specified',
                          weights=weights, **kwargs)


    def benchmark(self, dfs, name=None, weights=None, 
                  initial_capital=None, commissions=None, 
                  align_axis=None, fill_na=True):
        """
        dfs: str or list of str if dfs in self.df_equity or historical of tickers
        """
        align_axis = self._check_var(align_axis, self.align_axis)
        df_equity = self.align_period(self.df_equity, axis=align_axis, fill_na=fill_na, print_msg1=False)
        
        if isinstance(dfs, str):
            dfs = [dfs]

        if isinstance(dfs, list): # dfs is list of columns in self.df_equity
            if pd.Index(dfs).isin(df_equity.columns).sum() != len(dfs):
                return print('ERROR: check arg dfs')
            else:
                dfs = df_equity[dfs]
        else:
            dfs = dfs.loc[df_equity.index.min():df_equity.index.max()]

        if isinstance(dfs, pd.Series):
            if dfs.name is None:
                if name is None:
                    return print('ERROR')
                else:
                    dfs = dfs.to_frame(name)
            else:
                if name is None:
                    name = dfs.name
                dfs = dfs.to_frame()
        else:
            if name is None:
                name = list(dfs.columns)[0]
                print(f'WARNING: name set to {name}')
        
        dfs = self.align_period(dfs, axis=align_axis, fill_na=fill_na, print_msg2=False)
        weights = self._check_weights(weights, dfs)
        weigh = {'weigh':'specified', 'weights':weights}
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        commissions = self._check_var(commissions, self.commissions)
       
        self.portfolios[name] = self.backtest(dfs, name=name, select={'select':'all'}, 
                                              freq={'freq':None}, weigh=weigh, 
                                              initial_capital=initial_capital, 
                                              commissions=commissions)
        return None


    def build_batch(self, kwa_list, reset_portfolios=False, **kwargs):
        """
        kwa_list: list of k/w args for each backtest
        kwargs: k/w args common for all backtest
        """
        if reset_portfolios:
            self.portfolios = {}
        else:
            #return print('WARNING: set reset_portfolios to True to run')
            pass
            
        for kwa in kwa_list:
            self.build(**{**kwa, **kwargs})
        return None

    
    def run(self, pf_list=None, metrics=None, plot=True, freq='D', figsize=None, stats=True):
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
                print(f'REMINDER: max index of reset set to {len(bt_list)-1}')
            else: # pf_list is list of names
                bt_list = [v for k, v in self.portfolios.items() if k in pf_list]

        try:
            results = bt.run(*bt_list)
        except Exception as e:
            return print(f'ERROR: {e}')
            
        self.run_results = results
        
        if plot:
            results.plot(freq=freq, figsize=figsize)

        if stats:
            print('Returning stats')
            # pf_list not given as self.run_results recreated
            return self.get_stats(metrics=metrics) 
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
            df_stats = results.stats[pf_list]
        else:
            metrics = ['start', 'end'] + metrics
            df_stats = results.stats.loc[metrics, pf_list]

        if sort_by is not None:
            try:
                df_stats = df_stats.sort_values(sort_by, axis=1, ascending=False)
            except KeyError as e:
                print(f'WARNING: no sorting as {e}')

        return df_stats


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


    def plot(self, freq='D', figsize=None):
        if self.run_results is None:
            return print('ERROR: run backtest first')
        else:
            return self.run_results.plot(freq=freq, figsize=figsize)


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


    def util_import_data(self, symbol, col='Close', name=None, 
                         align_axis=None, date_format='%Y-%m-%d'):
        """
        import historical of symbol by using FinanceDataReader.DataReader
        """
        if name is None:
            name = symbol

        df_equity = self.df_equity
        align_axis = self._check_var(align_axis, self.align_axis)
        dfs = self.align_period(df_equity, axis=align_axis, print_msg1=False)
        
        start = dfs.index[0].strftime(date_format)
        end = dfs.index[-1].strftime(date_format)
        try:
            df = fdr.DataReader(symbol, start, end)
            return df[col].rename(name)
        except Exception as e:
            return print(f'ERROR: {e}')



class AssetEvaluator():
    def __init__(self, df_prices, days_in_year=252):
        # df of equities (equities in columns) which of each might have its own periods.
        # the periods of all equities will be aligned in every calculation.
        df_prices = df_prices.to_frame() if isinstance(df_prices, pd.Series) else df_prices
        if df_prices.index.name is None:
            df_prices.index.name = 'date' # set index name to run check_days_in_year
        _ = check_days_in_year(df_prices, days_in_year, freq='M')
        
        self.df_prices = df_prices
        self.days_in_year = days_in_year
        self.bayesian_data = None


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
