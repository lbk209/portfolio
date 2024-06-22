import bt
import pandas as pd
import matplotlib.pyplot as plt

import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

metrics = [
    'total_return', 'cagr', 
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



class Backtest():
    """
    backtest fixed weight portfolio
    """
    def __init__(self, df_equity, metrics=None, name_prfx='Portfolio', 
                 initial_capital=1000000, commissions=None, equity_names=None):
        # df of equities (equities in columns) which of each has its own periods.
        # the periods will be aligned for equities in a portfolio. see self.build
        self.df_equity = df_equity
        self.portfolios = dict()
        self.pf_weights = dict()
        self.metrics = metrics
        self.name_prfx = name_prfx
        self.n_names = 0 # see self._check_name
        self.initial_capital = initial_capital
        # commissions of all equities across portfolios (per year)
        self.commissions = commissions 
        self.equity_names = equity_names # names of all equities across portfolios
        self.run_results = None


    def align_period(self, df_equity, dt_format='%Y-%m-%d', n_indent=2, fill_na=True):
        """
        fill_na: set False to drop nan fields
        """
        df = get_date_range(df_equity, slice_input=True)
        dts = [x.strftime(dt_format) for x in (df.index.min(), df.index.max())]
        print(f"period reset: {' ~ '.join(dts)}")
        
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
            name = f'{self.name_prfx} {self.n_names}'
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
            metrics = self._check_var(metrics, self.metrics)
            if (metrics is None) or (metrics == 'all'):
                return results.stats
            else:
                metrics = ['start', 'end'] + metrics
                return results.stats.loc[metrics]
        else:
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


    def _plot_portfolios(self, plot_func, pf_list, ncols=2, sharex=True, sharey=True, figsize=(10,5)):
        n = len(pf_list)
        nrows = n // ncols + min(n % ncols, 1)
        fig, axs = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                                #figsize=figsize
                               )
        if nrows == 1:
            axs = [axs]
        
        k = 0
        finished = False
        for i in range(nrows):
            for j in range(ncols):
                ax = axs[i][j]
                _ = plot_func(pf_list[k], ax=ax, figsize=figsize)
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


    def get_historical(self, pf_list=None, normalize=True):
        """
        calc weighted sum of securities for each portfolio
        """
        pf_list  = self.check_portfolios(pf_list, run=True, convert_index=True)
        if pf_list is None:
            return None

        df_all = None
        for rp in pf_list:
            df = self._calc_weigthed(self.portfolios[rp])
            if normalize:
                df = df / df.dropna().iloc[0] * 100
            if df_all is None:
                df_all = df
            else:
                df_all = df_all.join(df)
        return df_all
        

    def _calc_weigthed(self, result_portfolio):
        df_d = result_portfolio.data
        df_w = result_portfolio.weights
        cols_d = df_d.columns
        cols_w = df_w.columns
        col_p = cols_w[0]
        
        di = list(range(len(cols_d)))
        wi = [[i+len(di) for i, cw in enumerate(cols_w) if cw.endswith(cd)][0] for cd in cols_d]
        f = lambda la, lb: [la[i] for i in lb]
        return (df_d.join(df_w, rsuffix='_w')
                    .apply(lambda x: np.inner(f(x, di), f(x, wi)), axis=1)
                    .to_frame(col_p).dropna())


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
                            .apply(lambda x: x.dropna()[1:-1]
                            .groupby('gb').count().mean().round()))

        cond = (days_freq_calc != round(days_in_year / factor))
        if cond.sum() > 0:
            days_in_year_new = days_freq_calc.loc[cond].mul(factor).round()
            print(f'WARNING: the number of days in a year with followings is not {days_in_year} in setting:')
            _ = [print(f'{k}: {int(v)}') for k,v in days_in_year_new.to_dict().items()]
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


    def calc_cagr(self, df_prices=None, days_in_year=None):
        # calc cagr's of equities
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        return df_prices.apply(lambda x: self._calc_cagr(x, days_in_year))


    def _calc_cagr(self, sr_prices, days_in_year):
        # calc cagr of a equity
        sr = sr_prices.dropna()
        t = days_in_year / len(sr)
        return (sr[-1]/sr[0]) ** t - 1


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
            df_prices = self.align_period(df_prices, fill_na=False)

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
                        multiplier_std=10):

        self.bayesian_sample_warning(freq)

        days_in_year = self.days_in_year
        periods, freq = self.get_freq_days(freq)
        factor_year = days_in_year/periods if annualize else 1

        df_prices = self.df_prices
        assets = list(df_prices.columns)
        
        if align_period:
            df_prices = self.align_period(df_prices, fill_na=False)
            df_ret = df_prices.pct_change(periods).dropna()
            mean_prior = df_ret.mean()
            std_prior = df_ret.std()
            std_low = std_prior / multiplier_std
            std_high = std_prior * multiplier_std
        else:
            ret_list = [df_prices[x].pct_change(periods).dropna() for x in assets]
            mean_prior = [x.mean() for x in ret_list]
            std_prior = [x.std() for x in ret_list]
            std_low = [x / multiplier_std for x in std_prior]
            std_high = [x * multiplier_std for x in std_prior]
            returns = dict()
        
        num_assets = len(assets) # flag for comparisson of two assets
        coords={'asset': assets}

        with pm.Model(coords=coords) as model:
            nu = pm.Exponential('nu_minus_two', 1 / 29, testval=4) + 2.
            #nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            #nu = pm.Deterministic("nu", nu_minus_one + 1)
            
            mean = pm.Normal('mean', mu=mean_prior, sigma=std_prior, dims='asset')
            std = pm.Uniform('std', lower=std_low, upper=std_high, dims='asset')
            if align_period:
                returns = pm.StudentT('returns', nu=nu, mu=mean, sigma=std, observed=df_ret)
            else:
                func = lambda x: dict(mu=mean[x], sigma=std[x], observed=ret_list[x])
                returns = {i: pm.StudentT(f'returns[{i}]', nu=nu, **func(i)) for i, _ in enumerate(assets)}
            
            std = std * pt.sqrt(nu / (nu - 2))
            pm.Deterministic(f'{freq}_mean',  mean * factor_year, dims='asset')
            pm.Deterministic(f'{freq}_vol',  std * (factor_year ** .5), dims='asset')
            pm.Deterministic(f'{freq}_sharpe', ((mean-rf) / std) * (factor_year ** .5), dims='asset')
    
            if num_assets == 2:
                mean_diff = pm.Deterministic('mean diff', mean[0] - mean[1])
                pm.Deterministic('std diff', std[0] - std[1])
                pm.Deterministic('effect size', mean_diff / (std[0] ** 2 + std[1] ** 2) ** .5 / 2)
    
            trace = pm.sample(draws=sample_draws, tune=sample_tune,
                              #chains=chains, cores=cores,
                              target_accept=target_accept,
                              #return_inferencedata=False, # TODO: what's for?
                              progressbar=True)
            
        self.bayesian_data = {'trace':trace, 'coords':coords, 
                              'freq':freq, 'annualize':annualize, 'rf':rf}
        return None


    
    def bayesian_sample_working(self, freq='yearly', annualize=True, rf=0, align_period=False,
                        sample_draws=1000, sample_tune=1000, target_accept=0.9,
                        multiplier_std=10):

        self.bayesian_sample_warning(freq)

        days_in_year = self.days_in_year
        periods, freq = self.get_freq_days(freq)
        factor_year = days_in_year/periods if annualize else 1
        
        df_prices = self.df_prices
        if align_period:
            df_prices = self.align_period(df_prices, fill_na=False)
        df_ret = df_prices.pct_change(periods)

        mean, std, returns = {}, {}, {}
        assets = list(df_ret.columns)
        num_assets = len(assets) # flag for comparisson of two assets
        with pm.Model() as model:
            nu = pm.Exponential('nu_minus_two', 1 / 29, testval=4) + 2.
            
            for i, x in enumerate(assets):
                mean[i] = pm.Normal(f'mean[{x}]', mu=mean_prior, sigma=std_prior)
                std[i] = pm.Uniform(f'std[{x}]', lower=std_low, upper=std_high)
                returns[i] = pm.StudentT('returns', nu=nu, mu=mean, sigma=std, observed=df_ret)




        ###
        
        mean_prior = df_ret.mean()
        std_prior = df_ret.std()
        std_low = std_prior / multiplier_std
        std_high = std_prior * multiplier_std
        factor_year = days_in_year/periods if annualize else 1
    
        assets = list(df_ret.columns)
        num_assets = len(assets) # flag for comparisson of two assets
        coords={'asset': assets}
        
        with pm.Model() as model:
            nu = pm.Exponential('nu_minus_two', 1 / 29, testval=4) + 2.
            
            mean = pm.Normal('mean', mu=mean_prior, sigma=std_prior)
            std = pm.Uniform('std', lower=std_low, upper=std_high)
            returns = pm.StudentT('returns', nu=nu, mu=mean, sigma=std, observed=df_ret)
            
            std = std * pt.sqrt(nu / (nu - 2))
            pm.Deterministic(f'{freq}_mean',  mean * factor_year, dims='asset')
            pm.Deterministic(f'{freq}_vol',  std * (factor_year ** .5), dims='asset')
            pm.Deterministic(f'{freq}_sharpe', ((mean-rf) / std) * (factor_year ** .5), dims='asset')
    
            if num_assets == 2:
                mean_diff = pm.Deterministic('mean diff', mean[0] - mean[1])
                pm.Deterministic('std diff', std[0] - std[1])
                pm.Deterministic('effect size', mean_diff / (std[0] ** 2 + std[1] ** 2) ** .5 / 2)
    
            trace = pm.sample(draws=sample_draws, tune=sample_tune,
                              #chains=chains, cores=cores,
                              target_accept=target_accept,
                              #return_inferencedata=False, # TODO: what's for?
                              progressbar=True)
            
        self.bayesian_data = {'trace':trace, 'coords':coords, 
                              'freq':freq, 'annualize':annualize, 'rf':rf}
        return None


    def bayesian_sample_old(self, freq='yearly', annualize=True, rf=0,
                        sample_draws=1000, sample_tune=1000, target_accept=0.9,
                        multiplier_std=10):

        self.bayesian_sample_warning(freq)

        days_in_year = self.days_in_year
        periods, freq = self.get_freq_days(freq)

        #df_ret = self.df_prices.pct_change(periods) # ImputationWarning & taking more time
        #df_ret = self.df_prices.pct_change(periods).dropna()
        df_ret = self.align_period(self.df_prices, fill_na=False)
        df_ret = df_ret.pct_change(periods).dropna()
        
        mean_prior = df_ret.mean()
        std_prior = df_ret.std()
        std_low = std_prior / multiplier_std
        std_high = std_prior * multiplier_std
        factor_year = days_in_year/periods if annualize else 1
    
        assets = list(df_ret.columns)
        num_assets = len(assets) # flag for comparisson of two assets
        coords={'asset': assets}
        
        with pm.Model(coords=coords) as model:
            nu = pm.Exponential('nu_minus_two', 1 / 29, testval=4) + 2.
            
            #nu_minus_one = pm.Exponential("nu_minus_one", 1 / 29.0)
            #nu = pm.Deterministic("nu", nu_minus_one + 1)
            
            mean = pm.Normal('mean', mu=mean_prior, sigma=std_prior, dims='asset')
            std = pm.Uniform('std', lower=std_low, upper=std_high, dims='asset')
            returns = pm.StudentT('returns', nu=nu, mu=mean, sigma=std, observed=df_ret)
            
            std = std * pt.sqrt(nu / (nu - 2))
            pm.Deterministic(f'{freq}_mean',  mean * factor_year, dims='asset')
            pm.Deterministic(f'{freq}_vol',  std * (factor_year ** .5), dims='asset')
            pm.Deterministic(f'{freq}_sharpe', ((mean-rf) / std) * (factor_year ** .5), dims='asset')
    
            if num_assets == 2:
                mean_diff = pm.Deterministic('mean diff', mean[0] - mean[1])
                pm.Deterministic('std diff', std[0] - std[1])
                pm.Deterministic('effect size', mean_diff / (std[0] ** 2 + std[1] ** 2) ** .5 / 2)
    
            trace = pm.sample(draws=sample_draws, tune=sample_tune,
                              #chains=chains, cores=cores,
                              target_accept=target_accept,
                              #return_inferencedata=False, # TODO: what's for?
                              progressbar=True)
            
        self.bayesian_data = {'trace':trace, 'coords':coords, 
                              'freq':freq, 'annualize':annualize, 'rf':rf}
        return None
        
    
    def bayesian_summary(self, var_names=None, filter_vars='like', **kwargs):
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
            return az.summary(trace, var_names=var_names, filter_vars=filter_vars, **kwargs)


    def bayesian_plot(self, var_names=None, filter_vars='like', **kwargs):
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
            coords = self.bayesian_data['coords']
            freq = self.bayesian_data['freq']
            annualize = self.bayesian_data['annualize']
            rf = self.bayesian_data['rf']

        df = self.summary(freq=freq, annualize=annualize, rf=rf)
        metrics = [x for x in df.index if x.startswith(freq)]
        ref_val = df.loc[metrics].to_dict(orient='index')
        col_name = list(coords.keys())[0]
        ref_val = {k: [{col_name:at, 'ref_val':rv} for at, rv in v.items()] for k,v in ref_val.items()}
            
        _ = az.plot_posterior(trace, var_names=var_names, filter_vars=filter_vars,
                              ref_val=ref_val, **kwargs)
        return None


    def align_period(self, df, fill_na=False):
        return Backtest(pd.Series()).align_period(df, fill_na=fill_na)


    def bayesian_sample_warning(self, freq='yearly'):
        msg = """ 
        Bayesian estimation of certain TDFs yielded dubious posteriors for daily or monthly frequencies. 
        However, the estimation for some ETFs appeared reasonable. The issue might lie in the historical prices of TDFs, 
        which were derived from cumulative rates of return, with data for weekends seemingly filled forward.
        """
        if freq != 'yearly':
            return print(f'WARNING: {msg}')
        else:
            return None