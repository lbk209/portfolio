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
    
    # convert to price with data_check[0]
    dt, price = data_check[0]
    dt = pd.to_datetime(dt)
    rate = df_rate[dt]
    if rate_is_percent:
        rate = rate/100
        df_rate = df_rate/100
    price_base = price / (rate+1)
    df_price = (df_rate + 1) * price_base 

    # check price
    dt, price = data_check[1]
    e = df_price[dt]/price - 1
    print(f'error: {e*100:.2f} %')
    
    return df_price


def backtest(dfs, weights=None, name='portfolio', period=None, **kwargs):
    if weights is None:
        cols = dfs.columns
        weights = dict(zip(cols, [1/len(cols)]*len(cols)))

    if period == 'W':
        run_period = bt.algos.RunWeekly()
    elif period == 'Q':
        run_period = bt.algos.RunQuarterly()
    elif period == 'Y':
        run_period = bt.algos.RunYearly()
    elif period == 'M':
        run_period = bt.algos.RunMonthly()
    else: # default montly
        run_period = bt.algos.RunOnce()
        
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