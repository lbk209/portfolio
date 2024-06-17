import bt
import pandas as pd

metrics = [
    'total_return', 'cagr', 
    'max_drawdown', 'avg_drawdown', 'avg_drawdown_days', 
    'daily_vol', 'daily_sharpe', 'daily_sortino', 
    'monthly_vol', 'monthly_sharpe', 'monthly_sortino'
]


def import_rate1(file, path='.', cols=['date', None]):
    """
    data_check: [(기준일1, 기준가1), (기준일2, 기준가2)]
    """
    df_rate = pd.read_csv(f'{path}/{file}', parse_dates=[0], index_col=[0])
    if df_rate.columns.size > 1:
        print('WARNING: taking the 1st two columns only ...')
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


def convert_rate_to_price(data, n_headers=1, path=None, rate_is_percent=True, df_rate=None):
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
    df = get_price(df, data_check, rate_is_percent=rate_is_percent)
    if df is None:
        return print(f'ERROR: check {ticker}')
    else:
        return df


def get_date_range(dfs, symbol_name=None):
    """
    symbol_name: dict of symbols to names
    """
    df = dfs.apply(lambda x: x[x.notna()].index.min()).to_frame('start date')
    df = df.join(dfs.apply(lambda x: x[x.notna()].index.max()).to_frame('end date'))
    if symbol_name is not None:
        df = pd.Series(symbol_name).to_frame('name').join(df)
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


    def align_period(self, df_equity, dt_format='%Y-%m-%d', n_indent=2):
        df = get_date_range(df_equity)
        start_date = df.iloc[:, 0].max()
        end_date = df.iloc[:, 1].min()
        df = df_equity.loc[start_date:end_date]
        dts = [x.strftime(dt_format) for x in (start_date, end_date)]
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

    
    def run(self, pf_list=None, metrics=None, plot=True, freq='d', figsize=None):
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
        
        if plot:
            results.plot(freq=freq, figsize=figsize);
        
        metrics = self._check_var(metrics, self.metrics)
        if (metrics is None) or (metrics == 'all'):
            return results.stats
        else:
            metrics = ['start', 'end'] + metrics
            return results.stats.loc[metrics]

    
    def show_weights(self, name=None, equity_names=None, as_series=True):
        if name is None: # return weights of all portfolios
            weights = self.pf_weights
        else:
            if name in self.portfolios.keys():
                weights = self.pf_weights[name]
                equity_names = self._check_var(equity_names, self.equity_names)
                if equity_names is not None:
                    weights = {equity_names[k] if k in equity_names else k:v for k,v in weights.items()}
            else:
                print(f'WARNING: no portfolio {name}')
                weights = self.pf_weights
        if as_series:
            weights = pd.Series(weights)
        return weights