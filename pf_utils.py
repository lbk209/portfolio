import bt
import pandas as pd

metrics = [
    'total_return', 'cagr', 
    'max_drawdown', 'avg_drawdown', 'avg_drawdown_days', 
    'daily_vol', 'daily_sharpe', 'daily_sortino', 
    'monthly_vol', 'monthly_sharpe', 'monthly_sortino'
]


def buy_and_hold(df, names=None, name_stg=None):
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if names is not None:
        if not isinstance(names, list):
            names = [names]
        cols = df.columns
        if len(cols) == len(names):
            df = df.rename(columns=dict(zip(cols, names)))
        else:
            print('WARNING: check num of names')

    if name_stg is None:
        name_stg = names[0]
        
    strategy = bt.Strategy(name_stg, [
        bt.algos.SelectAll(),
        bt.algos.WeighEqually(),
        bt.algos.RunOnce(),
        bt.algos.Rebalance()
    ])
    return bt.Backtest(strategy, df)



def backtest(dfs, weights=None, name='portfolio', period='M', **kwargs):
    if weights is None:
        cols = dfs.columns
        weights = dict(zip(cols, [1]*len(cols)))

    if period == 'W':
        run_period = bt.algos.RunWeekly()
    elif period == 'Q':
        run_period = bt.algos.RunQuarterly()
    elif period == 'Y':
        run_period = bt.algos.RunYearly()
    else: # default montly
        run_period = bt.algos.RunMonthly()
        
    strategy = bt.Strategy(name, [
        bt.algos.SelectAll(),
        bt.algos.WeighSpecified(**weights),
        run_period,
        bt.algos.Rebalance()
    ])
    return bt.Backtest(strategy, dfs, **kwargs)



def get_start_dates(dfs, symbol_name=None):
    """
    symbol_name: dict of symbols to names
    """
    df = dfs.apply(lambda x: x[x.notna()].index.min()).to_frame('start date')
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