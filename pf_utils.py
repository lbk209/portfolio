import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import FinanceDataReader as fdr
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt

import os, time, re, sys
from datetime import datetime, timedelta
from contextlib import contextmanager

from os import listdir
from os.path import isfile, join, splitext

import bt
from pf_custom import AlgoSelectKRatio, AlgoRunAfter, calc_kratio

from ffn import calc_stats, calc_perf_stats

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
    file = get_file_latest(file, path) # latest file
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


def get_date_range(dfs, symbol_name=None, return_intersection=False):
    """
    get datetime range of each ticker (columns) or datetime index of intersection
    dfs: index date, columns tickers
    symbol_name: dict of symbols to names
    """
    df = dfs.apply(lambda x: x[x.notna()].index.min()).to_frame('start date')
    df = df.join(dfs.apply(lambda x: x[x.notna()].index.max()).to_frame('end date'))
    if symbol_name is not None:
        df = pd.Series(symbol_name).to_frame('name').join(df)

    if return_intersection:
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


def check_days_in_year(df, days_in_year=252, freq='M', n_thr=10):
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

    # calc mean days for each asset
    df_days = (df.assign(gb=df.index.strftime(grp_format)).set_index('gb')
                 .apply(lambda x: x.dropna().groupby('gb').count()[1:-1])
                 .fillna(0) # fill nan with zero for assets invested recently
                 .mul(factor).mean().round()
                 .fillna(0) # for the case no asset has enough days for the calc
              )

    cond = (df_days != days_in_year)
    if cond.sum() > 0:
        df = df_days.loc[cond]
        n = len(df)
        if n < n_thr:
            #print(f'WARNING: the number of days in a year with followings is not {days_in_year} in setting:')
            print(f'WARNING: the number of days in a year with followings is {df.mean()} in avg.:')
            _ = [print(f'{k}: {int(v)}') for k,v in df.to_dict().items()]
        else:
            p = n / len(df_days) * 100
            #print(f'WARNING: the number of days in a year with {n} assets ({p:.0f}%) is not {days_in_year} in setting:')
            print(f'WARNING: the number of days in a year with {n} assets ({p:.0f}%) is {df.mean()} in avg.')
    
    return df_days


def align_period(df_assets, axis=0, date_format='%Y-%m-%d',
                 fill_na=True, print_msg1=True, print_msg2=True, n_indent=2):
    """
    axis: Determines the operation for handling missing data.
        0 : Drop rows (time index) with missing prices.
        1 : Drop columns (assets) with a count of non-missing prices less than the maximum found.
    fill_na: set False to drop nan fields
    """
    msg1 = None
    if axis == 0:
        df_aligned = get_date_range(df_assets, return_intersection=True)
        if len(df_aligned) < len(df_assets):
            dts = [x.strftime(date_format) for x in (df_aligned.index.min(), df_aligned.index.max())]
            msg1 = f"period reset: {' ~ '.join(dts)}"
    elif axis == 1:
        c_all = df_assets.columns
        df_cnt = df_assets.apply(lambda x: x.dropna().count())
        cond = (df_cnt < df_cnt.max())
        c_drop = c_all[cond]
        df_aligned = df_assets[c_all.difference(c_drop)]
        n_c = len(c_drop)
        if n_c > 0:
            n_all = len(c_all)
            msg1 = f'{n_c} assets removed for shorter periods ({n_c/n_all*100:.1f}%)'
    else:
        pass

    if print_msg1:
        print(msg1) if msg1 is not None else None
        if print_msg2:
            stats = df_aligned.isna().sum().div(df_aligned.count())
            t = 'filled forward' if fill_na else 'dropped'
            print(f'ratio of nan {t}::')
            indent = ' '*n_indent
            _ = [print(f'{indent}{i}: {stats[i]:.3f}') for i in stats.index]

    if fill_na:
        return df_aligned.ffill()
    else:
        return df_aligned.dropna()


def print_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time:.2f} secs")
        return result
    return wrapper


def get_file_list(file, path='.'):
    """
    find files starting with str file
    """
    name, ext = splitext(file)
    name = name.replace('*', r'(.*?)')
    try:
        flist = [f for f in listdir(path) if isfile(join(path, f)) and re.search(name, f)]
    except Exception as e:
        print(f'ERROR: {e}')
        flist = []
    return sorted(flist)


def get_file_latest(file, path='.'):
    files = get_file_list(file, path)
    if len(files) == 0:
        return None
    else:
        return files[-1] # latest file


def performance_stats(df_prices, metrics=None, sort_by=None, align_period=True, idx_dt=['start', 'end']):
    if isinstance(df_prices, pd.Series):
        df_prices = df_prices.to_frame()
        
    if align_period:
        df_stats = calc_stats(df_prices).stats
    else:
        #df_stats = df_prices.apply(lambda x: calc_stats(x.dropna()).stats)
        df_stats = df_prices.apply(lambda x: calc_perf_stats(x.dropna()).stats)

    if (metrics is not None) and (metrics != 'all'):
        metrics = idx_dt + metrics
        df_stats = df_stats.loc[metrics]

    for i in df_stats.index:
        if i in idx_dt:
            df_stats.loc[i] = df_stats.loc[i].apply(lambda x: x.strftime('%Y-%m-%d'))

    if sort_by is not None:
        try:
            df_stats = df_stats.sort_values(sort_by, axis=1, ascending=False)
        except KeyError as e:
            print(f'WARNING: no sorting as {e}')

    return df_stats



class AssetDict(dict):
    """
    A dictionary subclass that associates keys (ex:asset tickers) with names.
    Attributes:
        names (dict): Optional dictionary mapping tickers to names.
    """
    def __init__(self, *args, names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.names = names

    def __repr__(self):
        output = ""
        for i, key in enumerate(self.keys()):
            name = self.get_name(key)
            if name is None:
                output += f"{i}) {key}\n"
            else:
                output += f"{i}) {key}: {name}\n"
        return output

    def get_name(self, key):
        if self.names is None:
            return None
        else:
            try:
                return self.names[key]
            except KeyError:
                return None


class IndentOutput:
    def __init__(self, indent=4):
        self.indent = ' ' * indent
        self.old_target = sys.stdout
    
    def write(self, text):
        if text.strip():  # Only indent non-empty lines
            indented_text = f"{self.indent}{text.replace('\n', f'\n{self.indent}')}"
            self.old_target.write(indented_text)
        else:
            self.old_target.write(text)
    
    def flush(self):
        pass  # This method is needed to match the interface of sys.stdout

    @contextmanager
    def indented_output(self):
        old_stdout = sys.stdout
        sys.stdout = self
        try:
            yield
        finally:
            sys.stdout = old_stdout
            

class DataManager():
    def __init__(self, file=None, path='.', 
                 universe='kospi200', upload_type='price'):
        """
        universe: kospi200, etf
        """
        self.file_historicals = get_file_latest(file, path) # latest file
        self.path = path
        self.universe = universe
        self.asset_names = None
        self.upload_type = upload_type
        self.df_prices = None

    
    def upload(self, file=None, path=None):
        """
        load df_prices from saved file
        """
        file = self._check_var(file, self.file_historicals)
        path = self._check_var(path, self.path)
        if file is None:
            return print('ERROR: no file to load.')
        else:
            df_prices = self._upload(file, path, upload_type=self.upload_type)
        self.df_prices = df_prices
        return print('df_prices updated')
        

    @print_runtime
    def download(self, start_date=None, n_years=3, tickers=None,
                 save=True, date_format='%Y-%m-%d', close_today=False):
        """
        download df_prices by using FinanceDataReader
        """
        if start_date is None:
            today = datetime.today()
            start_date = (today.replace(year=today.year - n_years)
                               .replace(month=1, day=1).strftime(date_format))
            
        print('Downloading ...', end=' ')
        if tickers is None:
            asset_names = self._get_tickers(self.universe)
            if (asset_names is None) or len(asset_names) == 0:
                return print('ERROR: no ticker found')
            else:
                tickers = list(asset_names.keys())
                self.asset_names = asset_names
            
        try:
            df_prices = fdr.DataReader(tickers, start_date)
            if not close_today: # market today not closed yet
                df_prices = df_prices.loc[:datetime.today() - timedelta(days=1)]
            print('done.')
            self._print_info(df_prices, str_sfx='downloaded.')
        except Exception as e:
            return print(f'ERROR: {e}')
            
        self.df_prices = df_prices
        if save:
            self.save(date=df_prices.index.max())
        return print('df_prices updated')

    
    def save(self, file=None, path=None, date=None, date_format='%y%m%d'):
        file = self._check_var(file, self.file_historicals)
        path = self._check_var(path, self.path)
        df_prices = self.df_prices
        if (file is None) or (df_prices is None):
            return print('ERROR: check file or df_prices')

        if date is None:
            date = datetime.now()
        if not isinstance(date, str):
            date = date.strftime(date_format)
        
        file = re.sub(r"_\d+(?=\.\w+$)", f'_{date}', file)
        f = os.path.join(path, file)
        if os.path.exists(f):
            return print(f'ERROR: failed to save as {file} exists')
        else:
            df_prices.to_csv(f)    
            return print(f'{f} saved.')

    
    def _check_var(self, var_arg, var_self):
        return var_self if var_arg is None else var_arg


    def _print_info(self, df_prices, str_pfx='', str_sfx='', date_format='%Y-%m-%d'):
        dt0 = df_prices.index.min().strftime(date_format)
        dt1 = df_prices.index.max().strftime(date_format)
        n = df_prices.columns.size
        s1  = str_pfx + " " if str_pfx else ""
        s2  = " " + str_sfx if str_sfx else ""
        return print(f'{s1}{n} assets from {dt0} to {dt1}{s2}')


    def _get_tickers(self, universe='kospi200'):
        if universe.lower() == 'kospi200':
            func = self._get_tickers_kospi200
        elif universe.lower() == 'etf':
            func = self._get_tickers_etf
        elif universe.lower() == 'tdf':
            func = self._get_tickers_tdf
        else:
            func = lambda: None

        try:
            return func()
        except Exception as e:
            return print(f'ERROR: failed to download tickers as {e}')
            

    def _get_tickers_kospi200(self, ticker='KRX/INDEX/STOCK/1028', 
                              col_asset='Code', col_name='Name'):
        tickers = fdr.SnapDataReader(ticker)
        return tickers.set_index(col_asset)[col_name].to_dict()

    
    def _get_tickers_etf(self, ticker='ETF/KR', 
                         col_asset='Symbol', col_name='Name'):
        tickers = fdr.StockListing(ticker) # 한국 ETF 전종목
        return tickers.set_index(col_asset)[col_name].to_dict()


    def _get_tickers_tdf(self, col_asset='ticker', col_name='name'):
        file = self.file_historicals
        path = self.path
        tickers = pd.read_csv(f'{path}/{file}')
        return tickers.set_index(col_asset)[col_name].to_dict()
        

    def _upload(self, file, path, upload_type='price'):
        if upload_type.lower() == 'rate':
            func = self._upload_from_rate
        else: # default price
            func = lambda f, p: pd.read_csv(f'{p}/{f}', parse_dates=[0], index_col=[0])
        
        try:
            df_prices = func(file, path)
            self._print_info(df_prices, str_sfx='uploaded.')
            return df_prices
        except Exception as e:
            return print(f'ERROR: {e}')


    def _upload_from_rate(self, file, path):
        """
        master file of assets with ticker, file, adjusting data, etc
        """
        df_info = pd.read_csv(f'{path}/{file}')
        df_prices = None
        print('Estimating price from rate ...')
        for _, data in df_info.iterrows():
            # Using the combined class with the context manager
            with IndentOutput(indent=2).indented_output():
                df = convert_rate_to_price(data, path=path)
            if df_prices is None:
                df_prices = df.to_frame()
            else:
                df_prices = df_prices.join(df, how='outer')
        #print('Done.')
        return df_prices

    
    def get_names(self, tickers=None, reset=False):
        asset_names = self.asset_names
        df_prices = self.df_prices
        if reset or (asset_names is None):
            asset_names = self._get_tickers(self.universe)
            self.asset_names = asset_names

        try:
            if tickers is None:
                if df_prices is None:
                    res = asset_names
                else:
                    res = {k: asset_names[k] for k in df_prices.columns}
            else:
                res = {k: asset_names[k] for k in tickers}
            return AssetDict(res, names=asset_names)
        except KeyError as e:
            return print(f'ERROR: {e}')


    def get_date_range(self, return_intersection=False):
        df_prices = self.df_prices
        if df_prices is None:
            return print('ERROR')
        else:
            return get_date_range(df_prices, return_intersection=return_intersection)



class StaticPortfolio():
    def __init__(self, df_universe, file=None, path='.', 
                 method_weigh='ERC', lookback=12, lag=0,
                 days_in_year=246, align_axis=0, asset_names=None, name='portfolio',
                 cols_record = {'date':'date', 'ast':'asset', 'name':'name', 'prc':'price', 
                                'trs':'transaction', 'net':'net', 'wgt':'weight', 'wgta':'weight*'}
                ):
        """
        file: file of transaction history. 
              Do not update the asset prices with the actual purchase price, 
               as the new df_universe may be adjusted with updated prices after the purchase.
        asset_names: dict of ticker to name
        """
        bm = BacktestManager(df_universe, days_in_year=days_in_year, align_axis=align_axis)
        self.df_universe = bm.df_assets
        self._check_weights = bm._check_weights

        if file is None:
            file = 'tmp.csv' # set temp name for self._load_transaction
        file = self._retrieve_transaction_file(file, path)
        
        record = self._load_transaction(file, path)
        if record is None:
            print('REMINDER: make sure this is 1st transaction as no records provided')
        self.record = record
        self.cols_record = cols_record
        
        self.selected = None
        self.lookback = lookback 
        self.lag = lag 
        self.method_weigh = method_weigh
        self.file = file
        self.path = path
        self.asset_names = asset_names
        self.df_rec = None # record updated with new transaction
        self.name = name # portfolio name
        
        
    def select(self, date=None):
        """
        date: transaction date
        """
        df_data = self.df_universe 
        if date is not None:
            df_data = df_data.loc[:date]

        # prepare data for weigh procedure
        date = df_data.index.max()
        dt1 = date - pd.DateOffset(days=self.lag)
        dt0 = dt1 - pd.DateOffset(months=self.lookback)
        df_data = df_data.loc[dt0:dt1] 

        dts = df_data.index
        dts = [x.strftime('%Y-%m-%d') for x in (dts.min(), dts.max())]
        n_assets = df_data.columns.size # all assets in the universe selected
        print(f'{n_assets} assets from {dts[0]} to {dts[1]} prepared for weight analysis')
        
        self.selected = {'date': date, 'data': df_data}
        return None

    
    def weigh(self, method=None, weights=None):
        """
        method: ERC, InvVol, Equally, Specified
        weights: str, list of str, dict, or None
        """
        selected = self.selected
        method = self._check_var(method, self.method_weigh)
        if selected is None:
            return print('ERROR')
        else:
            df_data = selected['data']
            assets = df_data.columns
            
        if method.lower() == 'erc':
            weights = bt.ffn.calc_erc_weights(df_data.pct_change(1).dropna())
            method = 'ERC'
        elif method.lower() == 'invvol':
            weights = bt.ffn.calc_inv_vol_weights(df_data.pct_change(1).dropna())
            method = 'Inv.Vol'
        elif method.lower() == 'specified':
            w = self._check_weights(weights, df_data)
            weights = {x:0 for x in assets}
            weights.update(w)
            weights = pd.Series(weights)
            method = 'Specified'
        else: # default equal. no need to set arg weights
            weights = {x:1/len(assets) for x in assets}
            weights = pd.Series(weights)
            method = 'Equal weights'
        weigths = AssetDict(weights, names=self.asset_names)

        self.selected['weights'] = weights # weights is series
        print(f'Weights of assets determined by {method}.')
        return weights
        

    def allocate(self, capital=10000000, commissions=0):
        """
        calc number of each asset with price and weights
        capital: rebalance assets without cash flows if set to 0
        commissions: percentage
        """
        col_date = self.cols_record['date']
        col_ast = self.cols_record['ast']
        col_prc = self.cols_record['prc']
        col_net = self.cols_record['net']
        col_wgt = self.cols_record['wgt']
        col_wgta = self.cols_record['wgta']
        col_name = self.cols_record['name']
        cols_all = [col_name, col_prc, col_net, col_wgt, col_wgta]
        
        asset_names = self.asset_names
        selected = self.selected
        if selected is None:
            return print('ERROR')
        
        try:
            date = selected['date']
            weights = selected['weights']
            assets = weights.index
        except KeyError as e:
            return print('ERROR')

        # sum capital and asset value
        record = self.record
        if record is None:
            if capital == 0:
                return print('ERROR: Neither capital nor assets to rebalance exists')
        else:
            msg = 'WARNING: No rebalance as no new transaction'
            if self._check_new_transaction(date, record, col_date, msg):
                # the arg capital is now cash flows
                capital += self.calc_value(record, False)

        # calc quantity of each asset by weights and capital
        df_prc = self.df_universe
        wi = pd.Series(weights, name=col_wgt).rename_axis(col_ast) # ideal weights
        wvi = wi * capital / (1+commissions/100) # weighted asset value
        df_net = wvi / df_prc.loc[date, assets] # stock quantity float
        df_net = df_net.apply(np.floor).astype(int).to_frame(col_net) # stock quant int
        df_net = df_net.assign(**{col_date: date})
        df_net = df_prc.loc[date].to_frame(col_prc).join(df_net, how='right')
        # index is multiindex of date and asset
        df_net = df_net.set_index(col_date, append=True).swaplevel()

        # calc error between ideal and actual weights of assets 
        wva = df_net[col_prc].mul(df_net[col_net]).rename(col_wgta) # actual value of each asset
        mae = (wva.to_frame().join(wvi)
                  .apply(lambda x: x[col_wgta]/x[col_wgt] - 1, axis=1).abs().mean() * 100)
        print(f'Mean absolute error of weights: {mae:.0f} %')
        
        # add weights as new cols
        func = lambda x: f'{x:.03f}'
        wa = wva / wva.groupby(col_date).sum()
        df_net = df_net.join(wi.apply(func)).join(wa.apply(func))

        # add asset names
        if asset_names is None:
            df_net[col_name] = None
        else:
            df_net = df_net.join(pd.Series(asset_names, name=col_name), on=col_ast)
            
        return df_net[cols_all]
    

    def transaction(self, df_net, record=None):
        """
        add new transaction to records
        df_net: output of self.allocate
        record: transaction record given as dataframe
        """
        col_date = self.cols_record['date']
        col_ast = self.cols_record['ast']
        col_name = self.cols_record['name']
        col_prc = self.cols_record['prc']
        col_trs = self.cols_record['trs']
        col_net = self.cols_record['net']
        col_wgt = self.cols_record['wgt']
        col_wgta = self.cols_record['wgta']
        cols_short = [col_net, col_wgt, col_wgta]
        cols_all = [col_name, col_prc, col_trs, *cols_short]
        cols_int = [col_prc, col_trs, col_net]
                
        date = df_net.index.get_level_values(0).max()
        record = self._check_var(record, self.record)
        if record is None:
            # allocation is same as transaction for the 1st time
            df_rec = df_net.assign(**{col_trs: df_net[col_net]})
        else:
            if self._check_new_transaction(date, record, col_date):
                # add new to record after removing additional info except for cols_all in record
                df_rec = pd.concat([record[cols_all], df_net])
                df_prc = self.df_universe
            else:
                return None
            
            # fill missing prices (ex: old price of new assets, new price of old assets)
            # use purchase prices in the record before possible adjustment of stock prices
            lidx = [df_rec.index.get_level_values(i).unique() for i in range(2)]
            midx = pd.MultiIndex.from_product(lidx).difference(df_rec.index)
            df_m = (df_prc[lidx[1]].stack().loc[midx]
                     .rename_axis([col_date, col_ast]).to_frame(col_prc))
            df_rec = pd.concat([df_rec, df_m])
            
            # the net amount of the assets not in hold on the date is 0
            cond = df_rec[col_net].isna()
            cond = cond & (df_rec.index.get_level_values(0) == date)
            df_rec.loc[cond, cols_short] = 0  
            
            # update transaction on the date by using the assets on the date 
            # and all the transaction before the date
            df_trs = (df_rec.loc[date, col_net]
                      .sub(df_rec.groupby(col_ast)[col_trs].sum())
                      .to_frame(col_trs).assign(**{col_date:date})
                      .set_index(col_date, append=True).swaplevel())
            df_rec.update(df_trs)
            df_rec = df_rec.dropna(subset=cols_short) # drop new assets before the date

        df_rec = df_rec[cols_all]
        df_rec[cols_int] = df_rec[cols_int].astype(int).sort_index(level=[0,1])

        # calc value and profit
        v, p = [self.calc_value(df_rec, x) for x in [False, True]]
        print(f'Value {v:,}, Profit {p:,}')
        
        self.df_rec = df_rec
        return df_rec


    def calc_value(self, record, profit=False):
        """
        calc asset value profit
        """
        col_prc = self.cols_record['prc']
        col_trs = self.cols_record['trs']
        col_net = self.cols_record['net']
        cost = record[col_prc].mul(record[col_trs]).sum()
        date = record.index.get_level_values(0).max()
        val = record.loc[date].apply(lambda x: x[col_prc] * x[col_net], axis=1).sum()
        if profit: # return profit
            return val-cost
        else: # return total value (asset value + profit)
            return 2*val-cost
        

    def transaction_pipeline(self, date=None, method_weigh=None, weights=None,
                             capital=10000000, commissions=0, 
                             record=None, save=False):
        method_weigh = self._check_var(method_weigh, self.method_weigh)
        self.select(date=date)
        if not self.check_new_transaction():
            # calc profit at the last transaction
            p = self.calc_value(self.record, True)
            print(f'The profit from the most recent transaction: {p:,}')
            return self.record

        _ = self.weigh(method_weigh, weights)
        df_net = self.allocate(capital=capital, commissions=commissions)
        if df_net is None:
            return None
            
        df_rec = self.transaction(df_net, record=record)
        if df_rec is not None: # new transaction updated
            if save:
                self.save_transaction(df_rec)
            else:
                print('Set save=True to save transaction record')
        else:
            print('Nothing to save')
        return df_rec


    def _calc_historical(self, df_rec, name):
        """
        calc historical of portfolio value from tranaction
        """
        col_net = self.cols_record['net']
        end = datetime.today()
        sr_tot = pd.Series()
        dates_trs = df_rec.index.get_level_values(0).unique()
        
        # loop for transaction dates in descending order
        for start in dates_trs.sort_values(ascending=False):
            n_assets = df_rec.loc[start, col_net]
            # calc combined asset value history from prv transaction (start) to current (end) 
            sr_i = (self.df_universe.loc[start:end, n_assets.index]
                    .apply(lambda x: x*n_assets.loc[x.name]).sum(axis=1)) # x.name: index name
            # add to history the profit by substracting initial value of the history from total value (to start)
            sr_i += self.calc_value(df_rec.loc[:start], profit=False) - sr_i[0] 
            # concat histories        
            sr_tot = pd.concat([sr_tot, sr_i])
            end = start - pd.DateOffset(days=1)
        # sort by date
        return sr_tot.rename(name).sort_index()


    def get_historical(self):
        df_rec = self._check_result()
        if df_rec is None:
            return None
        else:
            return self._calc_historical(df_rec, self.name)

    
    def plot(self, figsize=(10,4)):
        """
        plot total value of portfolio
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
        else:
            sr_historical = self._calc_historical(df_rec, self.name)
            dates_trs = df_rec.index.get_level_values(0).unique()
            
        ax = sr_historical.plot(figsize=figsize, title='Portfolio Growth')
        ax.vlines(dates_trs, 0, 1, transform=ax.get_xaxis_transform(), lw=0.5, color='grey')
        ax.autoscale(enable=True, axis='x', tight=True)
        ax2 = ax.twinx()
        _ = sr_historical.mul(1/sr_historical[0]).plot(ax=ax2)
        ax2.autoscale(enable=True, axis='x', tight=True)
        return None


    def performance(self, metrics=None, sort_by=None):
        df_rec = self._check_result()
        if df_rec is None:
            return None
        else:
            sr_historical = self._calc_historical(df_rec, self.name)
            return performance_stats(sr_historical, metrics=metrics, sort_by=sort_by)


    def _check_result(self):
        if self.df_rec is None:
            if self.record is None:
                return print('ERROR: No transaction record')
            else:
                return self.record
        else:
            return self.df_rec
    

    def check_new_transaction(self):
        record = self.record
        if record is None:
            print('WARNING: no record loaded')
            return True
        else:
            col_date = record.index.names[0]
        
        selected = self.selected
        if selected is None:
            print('ERROR: run select first')
            return False
        else:
            date = selected['date']
        
        return self._check_new_transaction(date, record, col_date)
    
    
    def _check_new_transaction(self, date, record, col_date, 
                               msg='ERROR: check the date as no new transaction'):
        dt = record.index.get_level_values(col_date).max()
        if dt >= date:
            print(msg) if msg is not None else None
            return False
        else:
            return True


    def save_transaction(self, df_rec, file=None, path=None):
        file = self._check_var(file, self.file)
        path = self._check_var(path, self.path)
        return self._save_transaction(df_rec, file, path)
        

    def _load_transaction(self, file, path, print_msg=True, date_format='%Y-%m-%d'):
        f = os.path.join(path, file)
        if os.path.exists(f):
            df_rec = pd.read_csv(f, parse_dates=[0], index_col=[0,1], dtype={'asset':str})
        else:
            return None
            
        if print_msg:
            dt = df_rec.index.get_level_values(0).max().strftime(date_format)
            print(f'Transaction record to {dt} loaded.')
        return df_rec
        

    def _save_transaction(self, df_rec, file, path, 
                          pattern=r"_\d+(?=\.\w+$)", date_format='%Y-%m-%d'):
        # add date to file name
        dt = df_rec.index.get_level_values(0).max()
        dt = f"_{dt.strftime('%y%m%d')}"
        match = re.search(pattern, file)
        if bool(match):
            file = re.sub(match[0], dt, file)
        else:
            name, ext = splitext(file)
            file = f'{name}{dt}{ext}'
        # save after duplicate chekc
        f = os.path.join(path, file)
        if os.path.exists(f):
            return print(f'ERROR: failed to save as {file} exists')
        else:    
            df_rec.to_csv(f)
            print(f'All transactions saved to {file}')
        
    
    def _check_var(self, arg, arg_self):
        return arg_self if arg is None else arg


    def _retrieve_transaction_file(self, file, path):
        """
        get the latest transaction file
        """
        files = get_file_list(file, path)
        if len(files) == 0:
            name, ext = splitext(file)
            print(f'WARNING: no {name}* exists')
            return file
        else:
            return files[-1]



class MomentumPortfolio(StaticPortfolio):
    def __init__(self, *args, align_axis=1, method_select='simple', **kwargs):
        super().__init__(*args, align_axis=align_axis, **kwargs)
        self.method_select = method_select
        
    
    def select(self, date=None, n_assets=5, method=None):
        """
        date: transaction date
        method: simple, k-ratio
        """
        df_data = self.df_universe
        method = self._check_var(method, self.method_select)
        if date is not None:
            df_data = df_data.loc[:date]

        # prepare data for weigh procedure
        date = df_data.index.max()
        dt1 = date - pd.DateOffset(days=self.lag)
        dt0 = dt1 - pd.DateOffset(months=self.lookback)
        df_data = df_data.loc[dt0:dt1]

        dts = df_data.index
        dts = [x.strftime('%Y-%m-%d') for x in (dts.min(), dts.max())]
        info_date = f'from {dts[0]} to {dts[1]}'
        
        if method.lower() == 'k-ratio':
            rank = df_data.pct_change(1).apply(lambda x: calc_kratio(x.dropna())).sort_values(ascending=False)[:n_assets]
            method = 'K-ratio'
        else: # default simple
            #rank = bt.ffn.calc_total_return(df_data).sort_values(ascending=False)[:n_assets]
            # no difference with calc_total_return as align_axis=1
            rank = df_data.apply(lambda x: x.dropna().iloc[-1]/x.dropna().iloc[0]-1).sort_values(ascending=False)[:n_assets]
            method = 'Total return'

        assets = rank.index
        self.selected = {'date': date, 'rank': rank, 'data': df_data[assets]}
        print(f'{n_assets} assets selected by {method} {info_date}')
        return rank

    
    def transaction_pipeline(self, date=None, n_assets=5, 
                             method_select=None, method_weigh=None, weights=None,
                             capital=10000000, commissions=0, 
                             record=None, save=False):
        method_select = self._check_var(method_select, self.method_select)
        method_weigh = self._check_var(method_weigh, self.method_weigh)
        
        self.select(date=date, n_assets=n_assets, method=method_select)
        if not self.check_new_transaction():
            # calc profit at the last transaction
            p = self.calc_value(self.record, True)
            print(f'The profit from the most recent transaction: {p:,}')
            return self.record

        _ = self.weigh(method_weigh, weights)
        df_net = self.allocate(capital=capital, commissions=commissions)
        if df_net is None:
            return None
            
        df_rec = self.transaction(df_net, record=record)
        if df_rec is not None: # new transaction updated
            if save:
                self.save_transaction(df_rec)
            else:
                print('Set save=True to save transaction record')
        else:
            print('Nothing to save')
        return df_rec



class BacktestManager():
    def __init__(self, df_assets, name_prfx='Portfolio',
                 align_axis=0, fill_na=True, metrics=metrics,  
                 initial_capital=1000000, commissions=None, 
                 days_in_year=252, asset_names=None):
        """
        align_axis: how to set time periods intersection with assets
        fill_na: fill forward na in df_assets if True, drop na if False 
        """
        # df of assets (assets in columns) which of each has its own periods.
        # the periods will be aligned for assets in a portfolio. see self.build
        if isinstance(df_assets, pd.Series):
            return print('ERROR: df_assets must be Dataframe')
        
        df_assets = self.align_period(df_assets, axis=align_axis, fill_na=fill_na, print_msg2=False)
        self.df_assets = df_assets
        self.portfolios = AssetDict(names=asset_names) # dict of bt.backtest.Backtest
        self.cv_strategies = AssetDict(names=asset_names) # dict of args of strategies to cross-validate
        self.metrics = metrics
        self.name_prfx = name_prfx
        self.n_names = 0 # see self._check_name
        self.initial_capital = initial_capital
        # commissions of all assets across portfolios
        self.commissions = commissions  # unit %
        self.run_results = None
        self.days_in_year = days_in_year # only for self._get_algo_freq
        # saving to apply the same rule in benchmark data
        self.align_axis = align_axis
        self.fill_na = fill_na
        self.asset_names = asset_names
        self.print_algos_msg = True # control msg print in self._get_algo_*

        # run after set self.df_assets
        print('running self.util_check_days_in_year to check days in a year')
        _ = self.util_check_days_in_year(df_assets, days_in_year, freq='M')


    def align_period(self, df_assets, axis=0, date_format='%Y-%m-%d',
                     fill_na=True, print_msg1=True, print_msg2=True, n_indent=2):
        return align_period(df_assets, axis=axis, date_format=date_format,
                     fill_na=fill_na, print_msg1=print_msg1, print_msg2=print_msg2, n_indent=n_indent)


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


    def _get_algo_select(self, select='all', n_assets=0, lookback=0, lag=0):
        """
        select: all, momentum, kratio, randomly
        """
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        
        if cond(select, 'Momentum'):
            algo_select = bt.algos.SelectMomentum(n=n_assets, lookback=pd.DateOffset(months=lookback),
                                                  lag=pd.DateOffset(days=lag))
            # SelectAll() or similar should be called before SelectMomentum(), 
            # as StatTotalReturn uses values of temp[‘selected’]
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'k-ratio'):
            algo_select = AlgoSelectKRatio(n=n_assets, lookback=pd.DateOffset(months=lookback),
                                       lag=pd.DateOffset(days=lag))
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'randomly'):
            algo_after = AlgoRunAfter(lookback=pd.DateOffset(months=lookback), 
                                      lag=pd.DateOffset(days=lag))
            algo_select = bt.algos.SelectRandomly(n=n_assets)
            algo_select = bt.AlgoStack(algo_after, bt.algos.SelectAll(), algo_select)
        else:
            algo_after = AlgoRunAfter(lookback=pd.DateOffset(months=lookback), 
                                      lag=pd.DateOffset(days=lag))
            algo_select = bt.AlgoStack(algo_after, bt.algos.SelectAll())
            if not cond(select, 'all'):
                print('WARNING:SelectAll selected') if self.print_algos_msg else None
          
        return algo_select
        

    def _get_algo_freq(self, freq='M', offset=0, days_in_year=252):
        """
        freq: W, M, Q, Y, or num of days
        """
        if isinstance(freq, int):
            n = freq
        else:
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
                if freq.lower() != 'once':
                    print('WARNING:RunOnce selected') if self.print_algos_msg else None
            
        if n > 0:
            algo_freq = bt.algos.RunEveryNPeriods(n, offset=offset)
        else:
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
            algo_weigh = bt.AlgoStack(bt.algos.SelectHasData(lookback=pd.DateOffset(months=lookback)), 
                                      algo_weigh)
        else:
            algo_weigh = bt.algos.WeighEqually()
            if not cond(weigh, 'equally'):
                print('WARNING:WeighEqually selected') if self.print_algos_msg else None
            
        return algo_weigh
        

    def build(self, name=None, 
              freq='M', offset=0,
              select='all', n_assets=0, lookback=0, lag=0,
              weigh='equally', weights=None, rf=0, bounds=(0.0, 1.0),
              initial_capital=None, commissions=None, algos=None, run_cv=False):
        """
        make backtest of a strategy
        lookback: month
        lag: day
        commissions: %; same for all assets
        algos: set List of Algos to build backtest directly
        run_cv: flag to cross-validate
        """
        dfs = self.df_assets
        weights = self._check_weights(weights, dfs)
        name = self._check_name(name)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        commissions = self._check_var(commissions, self.commissions)

        # build args for self._get_algo_* from build args
        select = {'select':select, 'n_assets':n_assets, 'lookback':lookback, 'lag':lag}
        freq = {'freq':freq} # offset being saved when running backtest
        weigh = {'weigh':weigh, 'weights':weights, 'rf':rf, 'bounds':bounds,
                 'lookback':lookback, 'lag':lag}
        
        if run_cv:
            self.print_algos_msg = False
        else:
            self.print_algos_msg = True
            self.cv_strategies[name] = {
                # convert args for self.build_batch in self._cross_validate_strategy
                **select, **freq, **weigh, 'algos':None,
                'initial_capital':initial_capital, 'commissions':commissions
            }

        freq.update({'offset':offset, 'days_in_year':self.days_in_year})
        kwargs = {'select':select, 'freq':freq, 'weigh':weigh, 'algos':algos,
                  'initial_capital':initial_capital, 'commissions':commissions}
        self.portfolios[name] = self.backtest(dfs, name=name, **kwargs)
        
        return None
        

    def buy_n_hold(self, name=None, weights=None, **kwargs):
        """
        weights: dict of ticker to weight. str if one asset portfolio
        kwargs: set initial_capital or commissions
        """
        return self.build(name=name, freq='once', select='all', weigh='specified',
                          weights=weights, **kwargs)


    def benchmark(self, dfs, name=None, weights=None, 
                  initial_capital=None, commissions=None,
                  lookback=0, lag=0):
        """
        dfs: str or list of str if dfs in self.df_assets or historical of tickers
        no cv possible with benchmark
        lookback & lag to set start date same as momentum stragegy with lookback & lag
        """
        df_assets = self.df_assets
        
        if isinstance(dfs, str):
            dfs = [dfs]

        if isinstance(dfs, list): # dfs is list of columns in self.df_assets
            if pd.Index(dfs).isin(df_assets.columns).sum() != len(dfs):
                return print('ERROR: check arg dfs')
            else:
                dfs = df_assets[dfs]
        else:
            dfs = dfs.loc[df_assets.index.min():df_assets.index.max()]

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
        
        dfs = self.align_period(dfs, axis=self.align_axis, fill_na=self.fill_na, print_msg2=False)
        weights = self._check_weights(weights, dfs)
        weigh = {'weigh':'specified', 'weights':weights}
        select = {'select':'all', 'lookback':lookback, 'lag':lag}
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        commissions = self._check_var(commissions, self.commissions)
       
        self.portfolios[name] = self.backtest(dfs, name=name, select=select, 
                                              freq={'freq':'once'}, weigh=weigh, 
                                              initial_capital=initial_capital, 
                                              commissions=commissions)
        return None


    def benchmark_ticker(self, ticker='069500', name='KODEX200', **kwargs):
        print(f'Benchmark is {name}')
        df = self.util_import_data(ticker, name=name)
        return self.benchmark(df, **kwargs)


    def build_batch(self, *kwa_list, reset_portfolios=False, run_cv=False, **kwargs):
        """
        kwa_list: list of k/w args for each backtest
        kwargs: k/w args common for all backtest
        run_cv: set to True when runing self.cross_validate
        """
        if reset_portfolios:
            self.portfolios = AssetDict(names=self.asset_names)
        else:
            #return print('WARNING: set reset_portfolios to True to run')
            pass
        for kwa in kwa_list:
            self.build(**{**kwa, **kwargs, 'run_cv':run_cv})
        return None

    
    def run(self, pf_list=None, metrics=None, stats=True, stats_sort_by=None,
            plot=True, freq='D', figsize=None):
        """
        pf_list: List of backtests or list of index of backtest
        """
        # convert pf_list for printing msg purpose
        pf_list = self.check_portfolios(pf_list, run_results=None, convert_index=True, run_cv=False)
        self._print_strategies(pf_list, n_max=5, work='Backtesting')
        
        run_results = self._run(pf_list)
        if run_results is None:
            return None
        else:
            self.run_results = run_results
        
        if plot:
            run_results.plot(freq=freq, figsize=figsize)

        if stats:
            print('Returning stats')
            # pf_list not given as self.run_results recreated
            return self.get_stats(metrics=metrics, run_results=run_results, sort_by=stats_sort_by) 
        else:
            print('Returning backtest results')
            return run_results
        

    def _run(self, pf_list=None):
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
            return bt.run(*bt_list)
        except Exception as e:
            return print(f'ERROR: {e}')


    def _print_strategies(self, pf_list, n_max=5, work='Backtesting'):
        n = len(pf_list)
        if n > n_max:
            pf_str = f"{', '.join(pf_list[:2])}, ... , {pf_list[-1]}"
        else:
            pf_str = ', '.join(pf_list)
        print(f"{work} {n} strategies: {pf_str}")


    def cross_validate(self, pf_list=None, lag=None, n_sample=10, sampling='random',
                       metrics=None, simplify=True, remove_portfolios=True):
        """
        pf_list: str, index, list of str or list of index
        simplify: result format mean ± std if True, dict of cv if False 
        """
        if len(self.cv_strategies) == 0:
            return print('ERROR: no strategy to evaluate')
        else:
            n_given = 0 if pf_list is None else len(pf_list)
            pf_list = self.check_portfolios(pf_list, run_results=None, convert_index=True, run_cv=True)
            if (n_given > 0) and len(pf_list) != n_given:
                return print('ERROR: run after checking pf_list')
            else:
                self._print_strategies(pf_list, n_max=5, work='Cross-validating')
            
        metrics = self._check_var(metrics, self.metrics)
        
        lag = self._check_var(lag, self.days_in_year)
        if lag <= n_sample:
            n_sample = lag
            print(f'WARNING: n_sample set to lag {lag}')
            
        if sampling == 'random':
            offset_list = np.random.randint(lag, size=n_sample)
        else:
            offset_list = range(0, lag+1, round(lag/n_sample))

        result = dict()
        for name in pf_list:
            kwargs_build = self.cv_strategies[name]
            result[name] = self._cross_validate_strategy(name, offset_list, **kwargs_build)
        
        if remove_portfolios:
            remove = [k for k in self.portfolios.keys() for name in pf_list if k.startswith(f'CV[{name}]')]
            remain = {k: v for k, v in self.portfolios.items() if k not in remove}
            self.portfolios = AssetDict(remain)

        if simplify:
            df_cv = None
            for name, stats in result.items():
                df = stats.apply(lambda x: f'{x['mean']:.02f} ± {x['std']:.03f}', axis=1).to_frame(name)
                if df_cv is None:
                    df_cv = df
                else:
                    df_cv = df_cv.join(df)
            result = df_cv
            
        return result
        
        
    def _cross_validate_strategy(self, name, offset_list, metrics=None, **kwargs_build):
        keys = ['name', 'offset']
        kwa_list = [dict(zip(keys, [f'CV[{name}]: offset {x}', x])) for x in offset_list]
        kwargs_build = {k:v for k,v in kwargs_build.items() if k not in keys}
        # no saving param study in cv_strategies by setting run_cv to True
        self.build_batch(*kwa_list, run_cv=True, **kwargs_build)
            
        pf_list = [x['name'] for x in kwa_list]
        run_results = self._run(pf_list)
        stats = self.get_stats(metrics=metrics, run_results=run_results) 
        idx = stats.index.difference(['start', 'end'])
        return stats.loc[idx].agg(['mean', 'std', 'min', 'max'], axis=1)
     

    def check_portfolios(self, pf_list=None, run_results=None, convert_index=True, run_cv=False):
        """
        run_results: output from bt.run
        convert_index: convert pf_list of index to pf_list of portfolio names 
        run_cv: search porfolio args from self.cv_strategies
        """
        if run_results is None:
            if run_cv:
                pf_list_all = list(self.cv_strategies.keys())
            else:
                pf_list_all = list(self.portfolios.keys())
        else:
            pf_list_all = list(run_results.keys())
    
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


    def get_stats(self, pf_list=None, metrics=None, sort_by=None, run_results=None,
                  idx_dt=['start', 'end']):
        """
        run_results: arg for cross_validate. use self.run_results if set to None
        """
        if run_results is None:
            run_results = self.run_results
            
        pf_list  = self.check_portfolios(pf_list, run_results=run_results)
        if pf_list is None:
            return None
            
        metrics = self._check_var(metrics, self.metrics)
        if (metrics is None) or (metrics == 'all'):
            df_stats = run_results.stats[pf_list]
        else:
            metrics = idx_dt + metrics
            df_stats = run_results.stats.loc[metrics, pf_list]

        for i in df_stats.index:
            if i in idx_dt:
                df_stats.loc[i] = df_stats.loc[i].apply(lambda x: x.strftime('%Y-%m-%d'))

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
                _ = plot_func(pf_list[k], title=pf_list[k], 
                              ax=ax, legend=legend, figsize=figsize)
                k += 1
                if k == n:
                    finished = True
                    break
            if finished:
                break
    
    
    def plot_security_weights(self, pf_list=None, **kwargs):
        run_results = self.run_results
        pf_list = self.check_portfolios(pf_list, run_results=run_results)
        if pf_list is None:
            return None
        
        plot_func = run_results.plot_security_weights
        return self._plot_portfolios(plot_func, pf_list, **kwargs)
        

    def plot_weights(self, pf_list=None, **kwargs):
        run_results = self.run_results
        pf_list  = self.check_portfolios(pf_list, run_results=run_results)
        if pf_list is None:
            return None
        
        plot_func = run_results.plot_weights
        return self._plot_portfolios(plot_func, pf_list, **kwargs)


    def plot_histogram(self, pf_list=None, **kwargs):
        run_results = self.run_results
        pf_list  = self.check_portfolios(pf_list, run_results=run_results)
        if pf_list is None:
            return None
        
        if len(pf_list) > 1:
            print('WARNING: passed axis not bound to passed figure')

        for x in pf_list:
            _ = run_results.plot_histogram(x, **kwargs)
        return None


    def _retrieve_results(self, pf_list, func_result):
        """
        generalized function to retrieve results of pf_list from func_result
        func_result is func with ffn.core.PerformanceStats or bt.backtest.Backtest
        """
        pf_list  = self.check_portfolios(pf_list, run_results=self.run_results, convert_index=True)
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


    def get_historical(self, pf_list=None, raw=False):
        """
        drop dates before 1st transaction if raw=False
        """
        if raw:
            func_result = lambda x: self.run_results[x].prices
        else:
            def func_result(x):
                df = self.get_transactions(x, msg=False)
                start = df.index.get_level_values(0).min()
                return self.run_results[x].prices.loc[start:]
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

    
    def get_security_weights(self, pf=0, transaction_only=True, stack=False):
        """
        stack: convert to mutiindex of date and tickers allocated if True
        """
        if isinstance(pf, list):
            return print('WARNING: set one portfolio')
        run_results = self.run_results
        pf_list  = self.check_portfolios(pf, run_results=run_results, convert_index=True)
        if pf_list is None:
            return None
        else:
            pf = pf_list[0]
            
        df_w = run_results.get_security_weights(pf)
        if transaction_only:
            dts = self.get_transactions(pf, msg=False).index.get_level_values(0).unique()
            df_w = df_w.loc[dts]
            print(f'{pf}: weights at transactions returned')
        else:
            print(f'{pf}: weights returned')

        if stack: 
            df_w = df_w.stack().rename('weight')
            df_w = df_w.loc[df_w > 0]
        return df_w
        

    def get_transactions(self, pf=0, msg=True):
        if isinstance(pf, list):
            return print('WARNING: set one portfolio')
        run_results = self.run_results
        pf_list  = self.check_portfolios(pf, run_results=run_results, convert_index=True)
        if pf_list is None:
            return None
        else:
            pf = pf_list[0]

        print(f'{pf}: transactions returned') if msg else None
        return run_results.get_transactions(pf)


    def get_balance(self, pf=0, date=None, transpose=False, col='quantity'):
        """
        cal volume of each security on date
        """
        df_trans = self.get_transactions(pf, msg=False)
        if df_trans is None:
            return None
        
        if date is not None:
            df_trans = df_trans.loc[df_trans.index.get_level_values(0) <= date]
    
        date = df_trans.index.get_level_values(0).strftime('%Y-%m-%d')[-1]
        pf_list  = self.check_portfolios(pf, run_results=self.run_results, convert_index=True)
        print(f'{pf_list[0]}: quantity of securities on {date} returned')
        
        df_bal = df_trans[col].unstack().fillna(0).sum()
        df_bal = df_bal.rename('volume').loc[df_bal>0].astype('int')
        if transpose:
            return df_bal.to_frame().T
        else:
            return df_bal
        
        
    def util_import_data(self, symbol, col='Close', name=None, date_format='%Y-%m-%d'):
        """
        import historical of symbol by using FinanceDataReader.DataReader
        """
        if name is None:
            name = symbol

        df_assets = self.df_assets
        start = df_assets.index[0].strftime(date_format)
        end = df_assets.index[-1].strftime(date_format)
        
        try:
            df = fdr.DataReader(symbol, start, end)
            return df[col].rename(name)
        except Exception as e:
            return print(f'ERROR: {e}')

    
    def util_check_days_in_year(self, df=None, days_in_year=None, freq='M', n_thr=10):
        df = self._check_var(df, self.df_assets)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        return check_days_in_year(df, days_in_year=days_in_year, freq=freq, n_thr=n_thr)



class AssetEvaluator():
    def __init__(self, df_prices, days_in_year=252, metrics=metrics):
        # df of assets (assets in columns) which of each might have its own periods.
        # the periods of all assets will be aligned in every calculation.
        df_prices = df_prices.to_frame() if isinstance(df_prices, pd.Series) else df_prices
        if df_prices.index.name is None:
            df_prices.index.name = 'date' # set index name to run check_days_in_year
        _ = check_days_in_year(df_prices, days_in_year, freq='M')
        
        self.df_prices = df_prices
        self.days_in_year = days_in_year
        self.metrics = metrics
        self.bayesian_data = None


    def get_stats(self, metrics=None, sort_by=None, align_period=True, idx_dt=['start', 'end']):
        metrics = self._check_var(metrics, self.metrics)
        df_prices = self.df_prices
        return performance_stats(df_prices, metrics=metrics, sort_by=sort_by, align_period=align_period, idx_dt=idx_dt)

        
    def plot_historical(self, figsize=(10,4), title='Portfolio Growth'):
        """
        plot total value of portfolio
        """
        df_prices = self.df_prices
        ax = df_prices.plot(figsize=figsize, title=title)
        ax.autoscale(enable=True, axis='x', tight=True)
        return None
        

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


    def _calc_mean_return(self, df_prices, periods, days_in_year, annualize=True):
        scale = (days_in_year/periods) if annualize else 1
        return df_prices.apply(lambda x: x.pct_change(periods).dropna().mean() * scale)
        

    def _calc_volatility(self, df_prices, periods, days_in_year, annualize=True):
        scale = (days_in_year/periods) ** .5 if annualize else 1
        return df_prices.apply(lambda x: x.pct_change(periods).dropna().std() * scale)
        

    def _calc_sharpe(self, df_prices, periods, days_in_year, annualize=True, rf=0):
        mean = self._calc_mean_return(df_prices, periods, 0, False)
        std = self._calc_volatility(df_prices, periods, 0, False)
        scale = (days_in_year/periods) ** .5 if annualize else 1
        return (mean - rf) / std * scale


    def get_ref_val(self, freq='yearly', annualize=True, rf=0, align_period=False):
        """
        get ref val for 
        """
        df_prices = self.df_prices
        if align_period:
            df_prices = self.align_period(df_prices, axis=0, fill_na=True)
        days_in_year = self.days_in_year
        periods, freq = self.get_freq_days(freq)
        args = [df_prices, periods, days_in_year, annualize]
        return {
            f'{freq}_mean': self._calc_mean_return(*args).to_dict(),
            f'{freq}_vol': self._calc_volatility(*args).to_dict(),
            f'{freq}_sharpe': self._calc_sharpe(*args).to_dict()
        }


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
            df_prices = self.align_period(df_prices, axis=0, fill_na=True)
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
            ref_val = self.get_ref_val(freq=freq, annualize=annualize, rf=rf, align_period=align_period)
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


    def align_period(self, df, axis=0, fill_na=True, **kwargs):
        return align_period(df, axis=axis, fill_na=fill_na, **kwargs)
