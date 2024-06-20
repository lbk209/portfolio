import bt
import pandas as pd
import matplotlib.pyplot as plt

metrics = [
    'total_return', 'cagr', 
    'max_drawdown', 'avg_drawdown', 'avg_drawdown_days', 
    'daily_vol', 'daily_sharpe', 'daily_sortino', 
    'monthly_vol', 'monthly_sharpe', 'monthly_sortino'
]


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
        self.df_equity = self.align_period(df_equity)
        self.portfolios = dict()
        self.pf_weights = dict()
        self.metrics = metrics
        self.name_prfx = name_prfx
        self.n_names = 0
        self.initial_capital = initial_capital
        # commissions of all equities across portfolios (per year)
        self.commissions = commissions 
        self.equity_names = equity_names # names of all equities across portfolios
        self.run_results = None


    def align_period(self, df_equity, dt_format='%Y-%m-%d', n_indent=2):
        df = get_date_range(df_equity, slice_input=True)
        dts = [x.strftime(dt_format) for x in (df.index.min(), df.index.max())]
        print(f"backtest period reset: {' ~ '.join(dts)}")
        
        stats = df.isna().sum().div(df.count())
        print('rate of nan filled forward::')
        indent = ' '*n_indent
        _ = [print(f'{indent}{i}: {stats[i]:.3f}') for i in stats.index]
        
        return df.ffill()
        
    
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


    def _calc_commissions(self, commissions, weights, period='Y', rate_is_percent=True):
        """
        commissions: dict of equity to fee
        """
        a = 100 if rate_is_percent else 1
        
        if period == 'W':
            a *= 52
        elif period == 'Q':
            a *= 4
        elif period == 'M':
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
                 run_period=bt.algos.RunOnce(), 
                 capital_flow=0, **kwargs):
        """
        kwargs: keyword args for bt.Backtest
        """
        strategy = bt.Strategy(name, [
            bt.algos.SelectAll(),
            bt.algos.CapitalFlow(capital_flow),
            bt.algos.WeighSpecified(**weights),
            run_period,
            bt.algos.Rebalance()
        ])
        return bt.Backtest(strategy, dfs, **kwargs)
        

    def build(self, weights=None, name=None, period=None, 
              initial_capital=None, commissions=None, capital_flow=0):

        dfs = self.df_equity
        weights = self._check_weights(dfs, weights)
        name = self._check_name(name)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        
        try:
            dfs = dfs[weights.keys()] # dataframe even if one weight given
        except KeyError as e:
            return print(f'ERROR: check weights as {e}')
        
        if period == 'W':
            run_period = bt.algos.RunWeekly()
        elif period == 'Q':
            run_period = bt.algos.RunQuarterly()
        elif period == 'Y':
            run_period = bt.algos.RunYearly()
        elif period == 'M':
            run_period = bt.algos.RunMonthly()
        else: # default: buy & hold
            run_period = bt.algos.RunOnce()
        
        commissions = self._check_var(commissions, self.commissions)
        if commissions is None:
            c_avg = None
        else:
            c_avg = self._calc_commissions(commissions, weights, period)
        
        self.portfolios[name] = self.backtest(dfs, weights=weights, name=name, run_period=run_period, 
                                              capital_flow=capital_flow,
                                              initial_capital=initial_capital, commissions=c_avg)
        self.pf_weights[name] = weights

    
    def buy_n_hold(self, weights=None, name=None, **kwargs):
        if isinstance(weights, str):
            weights = {weights: 1}
        return self.build(weights=weights, name=name, period=None, **kwargs)

    
    def run(self, pf_list=None, metrics=None, plot=True, freq='d', figsize=None, stats=True):
        """
        pf_list: List of backtests or index
        """
        if pf_list is None:
            pf_list = self.portfolios.values()
        else:
            c = [0 if isinstance(x, int) else 1 for x in pf_list]
            if sum(c) == 0:
                pf_list = [x for i, x in enumerate(self.portfolios.values()) if i in pf_list]
            else:
                pf_list = [v for k, v in self.portfolios.items() if k in pf_list]
        
        results = bt.run(*pf_list)
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
        if run:
            if self.run_results is None:
                return print('ERROR')
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
        if self.run_results is None:
            return print('ERROR: run backtest first')
        
        plot_func = self.run_results.plot_security_weights
        pf_list  = self.check_portfolios(pf_list)
        return self._plot_portfolios(plot_func, pf_list, **kwargs)
        

    def plot_weights(self, pf_list=None, **kwargs):
        if self.run_results is None:
            return print('ERROR: run backtest first')
        
        plot_func = self.run_results.plot_weights
        pf_list  = self.check_portfolios(pf_list)
        return self._plot_portfolios(plot_func, pf_list, **kwargs)


    def plot_histogram(self, pf_list=None, **kwargs):
        if self.run_results is None:
            return print('ERROR: run backtest first')
        
        pf_list  = self.check_portfolios(pf_list)
        if len(pf_list) > 1:
            print('WARNING: passed axis not bound to passed figure')

        for x in pf_list:
            _ = self.run_results.plot_histogram(x, **kwargs)
        return None


    def get_weights(self, pf_list=None, equity_names=None, as_series=True):
        pf_list  = self.check_portfolios(pf_list, convert_index=True)
        weights = {k: self.pf_weights[k] for k in pf_list}
        quity_names = self._check_var(equity_names, self.equity_names)
        if equity_names is not None:
            weights = {k: {equity_names[x]:y for x,y in v.items()} for k,v in weights.items()}
        if as_series:
            weights = pd.DataFrame().from_dict(weights).fillna(0)
        return weights



class AssetEvaluator():
    def __init__(self, df_prices, days_in_year=252):
        self.df_prices = df_prices.to_frame() if isinstance(df_prices, pd.Series) else df_prices
        self.days_in_year = days_in_year
        self.bayesian_input = None
        self.bayesian_trace = None
        return self.check_days_in_year(df_prices, days_in_year)
     

    def check_days_in_year(self, df_prices=None, days_in_year=None):
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        
        df = (pd.Series(1, index=df_prices.index.strftime('%Y%m'))
                .groupby(df_prices.index.name).count())
        df = df[1:-1]
        
        avg_mdays = round(df.mean())
        days_in_month = round(days_in_year/12)
        if avg_mdays != days_in_month:
            return print(f'WARNING: avg days in a month, {avg_mdays} differs with {days_in_month}')
        else:
            return None


    def get_freq_days(self, freq='daily'):
        if freq == 'yearly':
            return self.days_in_year
        elif freq == 'monthly':
            return round(self.days_in_year/12)
        elif freq == 'weekly':
            return round(self.days_in_year/51)
        else: # default daily
            return 1

        
    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg

    
    def calc_cagr(self, df_prices=None, days_in_year=None):
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)

        t = self.days_in_year / len(df_prices)
        cagr = lambda x: (x[-1]/x[0]) ** t -1
        
        return df_prices.apply(lambda x: cagr(x.dropna()))
        

    def calc_mean_return(self, df_prices=None, days_in_year=None, freq='daily', annualize=True):
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)

        periods = self.get_freq_days(freq)
        res = df.pct_change(periods).dropna().mean()
        if annualize:
            return res * (days_in_year/periods)
        else:
            return res
        

    def calc_volatility(self, df_prices=None, days_in_year=None, freq='daily', annualize=True):
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)

        periods = self.get_freq_days(freq)
        res = df.pct_change(periods).dropna()
        res = res.std()
        if annualize:
            return res * ((days_in_year/periods) ** .5)
        else:
            return res


    def calc_sharpe(self, df_prices=None, days_in_year=None, freq='daily', annualize=True, rf=0):
        df_prices = self._check_var(df_prices, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)

        periods = self.get_freq_days(freq)
        res = df.pct_change(periods).dropna()
        res = (res.mean() - rf) / res.std()
        if annualize:
            return res * ((days_in_year/periods) ** .5)
        else:
            return res


    def summary(self, freq='daily', annualize=True, rf=0):
        kwargs = dict(
            freq=freq, annualize=annualize
        )
        df = self.df_prices.apply(lambda x: f'{len(x.dropna())/self.days_in_year:.1f}')
        # work even with df_prices of single asset as df_prices is always series (see __init__)
        return df.to_frame('years').join(
            self.calc_cagr().to_frame('cagr').join(
                self.calc_mean_return(**kwargs).to_frame(f'{freq}_mean').join(
                    self.calc_volatility(**kwargs).to_frame(f'{freq}_vol').join(
                        self.calc_sharpe(rf=rf, **kwargs).to_frame(f'{freq}_sharpe')
                    )
                )
            )
        ).T
    
    
    def bayesian_sample(self, freq='daily', annualize=True, rf=0,
                          sample_draws=1000, sample_tune=1000, target_accept=0.9,
                          multiplier_std=10):

        freq = self._check_var(freq, self.bayesian_freq)
        days_in_year = self.days_in_year
        periods = self.get_freq_days(freq)
        df_ret = self.df_prices.pct_change(periods).dropna()
        
        mean_prior = df_ret.mean()
        std_prior = df_ret.std()
        std_low = std_prior / multiplier_std
        std_high = std_prior * multiplier_std
    
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
            t = days_in_year/periods # annualizing
            pm.Deterministic(f'{freq}_mean',  mean * t, dims='asset')
            pm.Deterministic(f'{freq}_vol',  std * (t ** .5), dims='asset')
            pm.Deterministic(f'{freq}_sharpe', ((mean-rf) / std) * (t ** .5), dims='asset')
    
            if num_assets == 2:
                mean_diff = pm.Deterministic('mean diff', mean[0] - mean[1])
                pm.Deterministic('std diff', std[0] - std[1])
                pm.Deterministic('effect size', mean_diff / (std[0] ** 2 + std[1] ** 2) ** .5 / 2)
    
            trace = pm.sample(draws=sample_draws, tune=sample_tune,
                              #chains=chains, cores=cores,
                              target_accept=target_accept,
                              #return_inferencedata=False, # TODO: what's for?
                              progressbar=True)
        self.bayesian_trace = trace
        self.bayesian_freq = freq
        return None

    
    def bayesian_summary(self, var_names=None, filter_vars=None, **kwargs):
        trace = self.bayesian_trace
        if trace is None:
            return print('ERROR: run bayesian_sample first')
            
        return az.summary(trace, var_names=var_names, filter_vars=filter_vars, **kwargs)


    def bayesian_plot(self, var_names=None, filter_vars=None, **kwargs):
        trace = self.bayesian_trace
        if trace is None:
            return print('ERROR: run bayesian_sample first')
        else:
            freq = self.bayesian_freq

        self.summary(freq=freq, annualize=True, rf=0)
            
        if ref_val:
                self.summary(freq='monthly')
        _ = az.plot_posterior(trace, var_names=var_names, filter_vars=filter_vars,
                              ref_val=ref_val, **kwargs)
        return None
