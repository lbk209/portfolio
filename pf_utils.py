import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import FinanceDataReader as fdr
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xml.etree.ElementTree as ET
import os, time, re, sys, pickle
import bt
import warnings
import seaborn as sns
import yfinance as yf

from datetime import datetime, timedelta
from contextlib import contextmanager
from os import listdir
from os.path import isfile, join, splitext
from pf_custom import (AlgoSelectKRatio, AlgoRunAfter, calc_kratio, AlgoSelectIDiscrete, 
                       AlgoSelectIDRank, SelectMomentum, AlgoSelectFinRatio)
from ffn import calc_stats, calc_perf_stats
from pykrx import stock as pyk
from tqdm import tqdm
from matplotlib.dates import num2date, date2num
from numbers import Number

from pf_data import PortfolioData

warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)

# support korean lang
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumBarunGothic'

METRICS = [
    'total_return', 'cagr', 'calmar', 
    'max_drawdown', 'avg_drawdown', 'avg_drawdown_days', 
    'daily_vol', 'daily_sharpe', 'daily_sortino', 
    'monthly_vol', 'monthly_sharpe', 'monthly_sortino'
]

WEEKS_IN_YEAR = 51


def valuate_bond(face, rate, year, ytm, n_pay=1):
    """
    Bond Valuation: see www.investopedia.com/terms/b/bond-valuation.asp
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
    df_rate = df_rate.T.iloc[n_headers:, 0].astype(float)

    df_rate.index = pd.to_datetime(df_rate.index)
    df_rate = df_rate.rename_axis(cols[0])
    
    col_data = cols[1]
    if col_data is None:
        col_data = file.split('.')[0]
    df_rate.name = col_data

    return df_rate


def import_xml_rate(file, path='.', cols=['date', None], 
                    tag_iter='prfRtList', tag_date='standardDt', 
                    tag_val='managePrfRate'):
    """
    file: xml. historical of cumulative rate of return in long format
    data_check: [(기준일1, 기준가1), (기준일2, 기준가2)]
    """
    tree = ET.parse(f'{path}/{file}')
    root = tree.getroot()
    
    data = list()
    for x in root.iter(tag_iter):
        date = x.find(tag_date).text
        val = x.find(tag_val).text
        data.append((date,val))
    
    df_val = pd.DataFrame().from_records(data, columns=cols).set_index(cols[0]).astype(float)
    df_val.index = pd.to_datetime(df_val.index)
    # make sure to get series
    df_val = df_val.iloc[:, 0]
    
    return df_val
    

def get_price(df_rate, data_check, rate_is_percent=True, print_msg=True):
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
    if print_msg:
        print(f'error: {e*100:.2f} %')
    return (df_price, e)


def convert_rate_to_price(data, n_headers=1, path=None, 
                          rate_is_percent=True, df_rate=None, rate_only=False,
                          print_msg=False):
    """
    data: series or dict
    df_rate: historical given as dataframe
    """
    data_type = data['data_type']
    if data_type == 1:
        import_rate = import_rate1
    elif data_type == 2:
        import_rate = lambda *args, **kwargs: import_rate2(*args, n_headers=n_headers, **kwargs)
    elif data_type == 3:
        import_rate = import_xml_rate
    else:
        if df_rate is None:
            return print(f'ERROR: no data type {data_type} exists')
        else:
            import_rate = lambda *args, **kwargs: df_rate.rename_axis(kwargs['cols'][0]).rename(kwargs['cols'][1])
    
    ticker = data['ticker']
    name = data['name']
    file = get_file_latest(data['file'], path) # latest file
    data_check = [
        (data['check1_date'], data['check1_price']),
        (data['check2_date'], data['check2_price']),
    ]
    
    df = import_rate(file, path=path, cols=['date', ticker])
    if rate_only:
       return df
        
    df_n_err = get_price(df, data_check, rate_is_percent=rate_is_percent, print_msg=print_msg)
    if df_n_err is None:
        return print(f'ERROR: check {ticker}')
    else:
        return df_n_err


def get_date_range(dfs, symbol_name=None, return_intersection=False):
    """
    get datetime range of each ticker (columns) or datetime index of intersection
    dfs: index date, columns tickers
    symbol_name: dict of symbols to names
    """
    df = dfs.apply(lambda x: x[x.notna()].index.min()).to_frame('start date')
    df = df.join(dfs.apply(lambda x: x[x.notna()].index.max()).rename('end date'))
    df = df.join(dfs.apply(lambda x: x[x.notna()].count()).rename('n'))
    if symbol_name is not None:
        df = pd.Series(symbol_name).to_frame('name').join(df, how='right')

    if return_intersection:
        start_date = df.iloc[:, 0].max()
        end_date = df.iloc[:, 1].min()
        return dfs.loc[start_date:end_date]
    else:
        return df.sort_values('start date')


def get_date_minmax(df, date_format=None, level=0):
    """
    get min & max from the datetime index of df
    """
    dts = df.index.get_level_values(level)
    dts = [dts.min(), dts.max()]
    if date_format is not None:
        dts = [x.strftime(date_format) for x in dts]
    return dts
    

def check_days_in_year(df, days_in_year=252, freq='M', n_thr=10, msg=True):
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
                 #.fillna(0) # comment as it distorts mean
                 .mul(factor).mean().round()
                 .fillna(0) # for the case no asset has enough days for the calc
              )

    cond = (df_days != days_in_year)
    if (cond.sum() > 0) and msg:
        df = df_days.loc[cond]
        n = len(df)
        if n < n_thr:
            #print(f'WARNING: the number of days in a year with followings is not {days_in_year} in setting:')
            print(f'WARNING: the number of days in a year with followings is {df.mean():.0f} in avg.:')
            _ = [print(f'{k}: {int(v)}') for k,v in df.to_dict().items()]
        else:
            p = n / len(df_days) * 100
            #print(f'WARNING: the number of days in a year with {n} assets ({p:.0f}%) is not {days_in_year} in setting:')
            print(f'WARNING: the number of days in a year with {n} assets ({p:.0f}%) is {df.mean():.0f} in avg.')
    
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
            dts = get_date_minmax(df_aligned, date_format)
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
        rex = f'^{name}.*{ext}'
        flist = [f for f in listdir(path) if isfile(join(path, f)) and re.search(rex, f)]
    except Exception as e:
        print(f'ERROR: {e}')
        flist = []
    return sorted(flist)


def get_file_latest(file, path='.', msg=False, file_type=None):
    files = get_file_list(file, path)
    if len(files) == 0:
        if msg and (file is not None):
            name, ext = splitext(file)
            name = name if file_type is None else f'{file_type} {name}'
            print(f'WARNING: no {name}*{ext} exists')
        return file
    else:
        return files[-1] # latest file


def get_filename(file, repl, pattern=r"_\d+(?=\.\w+$)"):
    """
    replace pattern of file with repl or insert repl if no pattern in file
    """
    match = re.search(pattern, file)
    if bool(match):
        file = re.sub(match[0], repl, file)
    else:
        name, ext = splitext(file)
        file = f'{name}{repl}{ext}'
    return file


def set_filename(file, ext=None, default=None):
    """
    return default for file name if file is None
    set extension if no extension in file
    defaule: ex) 'temp.csv', 'temp', None
    ext: ex) '.csv', 'csv', None
    """
    # set dault file name and extension
    if default is not None:
        name, _ext = splitext(default)
        ext = _ext if ext is None else ext
        ext = ext.replace('.', '')
    # return default if file is None
    if file is None:
        if default is not None:
            default = name if ext is None else f'{name}.{ext}'
        return default
    # set ext if no ext in file    
    name, _ext = splitext(file)
    if len(_ext) == 0:
        file = name if ext is None else f'{name}.{ext}'
    return file


def save_dataframe(df, file, path='.', 
                   msg_succeed='file saved.',
                   msg_fail='ERROR: failed to save as the file exists'):
    f = os.path.join(path, file)
    if os.path.exists(f):
        print(msg_fail)
        return False
    else:
        df.to_csv(f)    
        print(msg_succeed)
        return True
        

def performance_stats(df_prices, metrics=None, sort_by=None, align_period=True, idx_dt=['start', 'end']):
    if isinstance(df_prices, pd.Series):
        df_prices = df_prices.to_frame()

    if len(df_prices) <= 1:
        return print('ERROR: Need more data to measure')
        
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


def convert_to_daily(df, method='ffill'):
    """
    convert df to daily time series
    """
    start = df.index.min()
    end = df.index.max()
    index = pd.date_range(start, end)
    return df.reindex(index, method=method)


def mldate(date, date_format='%Y-%m-%d'):
    """
    convert date of str or datetime to Matplotlib date format 
     or date of Matplotlib date format to str
    """
    if isinstance(date, Number):
        return num2date(date).strftime(date_format)
    elif isinstance(date, str):
        date = datetime.strptime(date, date_format)
        return date2num(date)
    elif isinstance(date, datetime):
        return date2num(date)
    else:
        return date


def set_matplotlib_twins(ax1, ax2, legend=True):
    axes = [ax1, ax2]
    # set tick color
    _ = [x.tick_params(axis='y', labelcolor=x.get_lines()[0].get_color()) for x in axes]
    # drop individual legends
    _ = [None if x.get_legend() is None else x.get_legend().remove() for x in axes]
    if legend:
        # set legend
        h1, h2 = [x.get_legend_handles_labels()[0] for x in axes]
        if len(h1)*len(h2) > 0:
            ax1.legend(handles=h1+h2)
    return (ax1, ax2)


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
            name = self.get_names(key)
            if name is None:
                output += f"{i}) {key}\n"
            else:
                output += f"{i}) {key}: {name}\n"
        return output

    def get_names(self, key):
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



class TimeTracker:
    def __init__(self, auto_start=False):
        """
        Initialize the TimeTracker instance.
        If auto_start is True, the timer will start automatically upon initialization.
        """
        self.reset()
        if auto_start:
            self.start()

    def start(self):
        """Start the timer if it's not started or resume it if it was stopped."""
        if self.start_time is None:
            # Start the timer for the first time
            self.start_time = time.time()
            self.last_pause_time = self.start_time
            self.is_stopped = False
            self.total_paused_time = 0
        elif self.is_stopped:
            # Resume the timer if it's stopped
            current_time = time.time()
            # Update the total paused time by subtracting the time since last pause
            self.total_paused_time += current_time - self.last_pause_time
            self.is_stopped = False
        else:
            print("Timer is already running.")

    def elapsed(self):
        """
        Return the time elapsed since the timer was started, excluding any paused time.
        Raises an error if the timer has not been started.
        """
        if self.start_time is None:
            s = "Timer has not been started. Use the 'start()' method."
            #raise ValueError(s)
            return print(f'ERROR: {s}')
        
        current_time = self.last_pause_time if self.is_stopped else time.time()
        te = current_time - self.start_time
        tp = self.total_paused_time
        
        gets = lambda x: f'{x:.1f} secs' if x//60 < 1 else f'{x/60:.1f} mins'
        return print(f'{gets(te)} elapsed, {gets(tp)} paused ({tp/te:.1%})')


    def pause(self, interval=5, pause_duration=1, msg=True):
        """
        Pauses execution for 'pause_duration' seconds if 'interval' seconds have passed
        since the last pause or start. Tracks total pause duration and increments pause count.
        """
        if interval < 0 or pause_duration < 0:
            s = "Interval and pause duration must be non-negative."
            #raise ValueError(s)
            return print(f'ERROR: {s}')

        current_time = time.time()
        time_since_last_pause = current_time - self.last_pause_time
        
        if time_since_last_pause >= interval:
            if msg:
                print(f"Pausing for {pause_duration} second(s) after {interval} seconds elapsed...")
            time.sleep(pause_duration)
            self.last_pause_time = current_time  # Update the last pause time
            self.total_paused_time += pause_duration  # Add to total paused time
            self.pause_count += 1  # Increment the pause counter

    def stop(self, msg=True):
        """
        Stops the timer by marking it as stopped.
        Further calls to 'elapsed' will reflect the time until this point.
        """
        if not self.is_stopped:
            self.is_stopped = True
            self.last_pause_time = time.time()
            if msg:
                self.elapsed()
        
    def reset(self):
        """Reset the timer, clearing all tracked time and resetting the pause counter."""
        self.start_time = None
        self.last_pause_time = None
        self.total_paused_time = 0
        self.is_stopped = False
        self.pause_count = 0  # Reset the pause counter



class DataManager():
    def __init__(self, file=None, path='.', 
                 universe='kospi200', upload_type='price', 
                 daily=True, days_in_year=12):
        """
        universe: kospi200, etf, krx, file. used only for ticker name
        """
        file = set_filename(file, 'csv') 
        self.file_historical = get_file_latest(file, path) # latest file
        self.path = path
        self.universe = universe
        self.asset_names = None 
        self.upload_type = upload_type
        self.df_prices = None
        self.days_in_year = days_in_year # only for convert_to_daily
        self.upload(self.file_historical)
        self.convert_to_daily(True, days_in_year) if not daily else None
        # asset_names set in downloading except for 'file' universe
        _ = self.get_names() if universe == 'file' else None
    
    def upload(self, file=None, path=None):
        """
        load df_prices from saved file
        """
        file = self._check_var(file, self.file_historical)
        path = self._check_var(path, self.path)
        if file is None:
            return print('ERROR: no file to load.')
        else:
            df_prices = self._upload(file, path, upload_type=self.upload_type,
                                    msg_exception='WARNING: uploading failed as ')

        if df_prices is None:
            return None # error msg printed out by self._upload
        else:
            self.df_prices = df_prices
            return print('df_prices loaded')
        

    @print_runtime
    def download(self, start_date=None, end_date=None, n_years=3, tickers=None,
                 save=True, date_format='%Y-%m-%d', close_today=False,
                 **kwargs_download):
        """
        download df_prices by using FinanceDataReader
        n_years: int
        tickers: None, 'selected', list of tickers
        kwargs_download: args for krx. ex) interval=5, pause_duration=1, msg=False
        """
        start_date, end_date = DataManager.get_start_end_dates(start_date, end_date, 
                                                               close_today, n_years, date_format)
        print('Downloading ...')
        if tickers is None:
            asset_names = self._get_tickers(self.universe)
            if (asset_names is None) or len(asset_names) == 0:
                return print('ERROR: Failed to get ticker names')
            else:
                tickers = list(asset_names.keys())
        else: 
            if isinstance(tickers, str) and (tickers.lower() == 'selected'):
                if self.df_prices is None:
                    return print('ERROR: No selected tickers as no file exists')
                else:
                    tickers = self.df_prices.columns.to_list()
            elif not isinstance(tickers, list):
                return print('ERROR: check tickers set')
            asset_names = self._get_tickers(self.universe, tickers)
                   
        try:
            df_prices = self._download(self.universe, tickers, start_date, end_date,
                                       **kwargs_download)
            if not close_today: # market today not closed yet
                df_prices = df_prices.loc[:datetime.today() - timedelta(days=1)]
            print('... done')
            DataManager.print_info(df_prices, str_sfx='downloaded.')
        except Exception as e:
            return print(f'ERROR: {e}')
            
        self.df_prices = df_prices
        self.asset_names = asset_names     
        if save:
            self.save(date=df_prices.index.max())
        return print('df_prices updated')

    
    def save(self, file=None, path=None, date=None, date_format='%y%m%d'):
        file = self._check_var(file, self.file_historical)
        path = self._check_var(path, self.path)
        df_prices = self.df_prices
        if (file is None) or (df_prices is None):
            return print('ERROR: check file or df_prices')

        if date is None:
            date = datetime.now()
        if not isinstance(date, str):
            date = date.strftime(date_format)

        file = get_filename(file, f'_{date}', r"_\d+(?=\.\w+$)")
        _ = save_dataframe(df_prices, file, path, msg_succeed=f'{file} saved',
                           msg_fail=f'ERROR: failed to save as {file} exists')
        return None

    
    def _check_var(self, var_arg, var_self):
        return var_self if var_arg is None else var_arg


    @staticmethod
    def print_info(df_prices, str_pfx='', str_sfx='', date_format='%Y-%m-%d'):
        dt0, dt1 = get_date_minmax(df_prices, date_format)
        n = df_prices.columns.size
        s1  = str_pfx + " " if str_pfx else ""
        s2  = " " + str_sfx if str_sfx else ""
        return print(f'{s1}{n} assets from {dt0} to {dt1}{s2}')


    def _get_tickers(self, universe='kospi200', tickers=None, **kwargs):
        if universe.lower() == 'kospi200':
            func = self._get_tickers_kospi200
        elif universe.lower() == 'etf':
            func = self._get_tickers_etf
        elif universe.lower() == 'file':
            func = self._get_tickers_file
        elif universe.lower() == 'krx':
            func = self._get_tickers_krx
        elif universe.lower() == 'yahoo':
            tickers = self.df_prices.columns.to_list() if tickers is None else tickers
            func = lambda *a, **k: self._get_tickers_yahoo(tickers, *a, **k)
        else:
            func = lambda **x: None

        try:
            return func(**kwargs)
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


    def _get_tickers_krx(self, ticker=['KOSPI', 'KOSDAQ'], 
                         col_asset='Code', col_name='Name'):
        tickers = dict()
        for x in ticker:
            df = fdr.StockListing(x)
            tickers.update(df.set_index(col_asset)[col_name].to_dict())
        return tickers


    def _get_tickers_file(self, col_asset='ticker', col_name='name'):
        file = self.file_historical
        path = self.path
        tickers = pd.read_csv(f'{path}/{file}')
        return tickers.set_index(col_asset)[col_name].to_dict()


    def _get_tickers_yahoo(self, tickers, col_name='longName'):
        if tickers is None:
            return print('ERROR: set tickers for names')
        if isinstance(tickers, str):
            tickers = [tickers]
        yft = yf.Tickers(' '.join(tickers))
        return {x:yft.tickers[x].info[col_name] for x in tickers}


    def _download(self, universe, *args, **kwargs):
        """
        args: args for DataReader. ex) ticker, start_date, end_date
        """
        uv = universe.lower()
        if uv == 'krx':
            # use pykrx as fdr seems ineffective to download all tickers in krx
            func = DataManager.download_krx
        elif uv == 'yahoo':
            func = DataManager.download_yahoo
        elif uv == 'file':
            return print("ERROR: Downloading not supported for universe 'file'")
        else:
            func = fdr.DataReader
            
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return print(f'ERROR: failed to download prices as {e}')

    @staticmethod
    def download_krx(tickers, start_date, end_date,
                      interval=5, pause_duration=1, msg=False):
        """
        tickers: list of tickers
        """
        krx = KRXDownloader(start_date, end_date)
        krx.get_tickers(tickers=tickers)
        krx.download(interval=interval, pause_duration=pause_duration, msg=msg)
        return krx.df_data

    @staticmethod
    def download_yahoo(tickers, start_date, end_date, col_price='Adj Close'):
        if isinstance(tickers, str):
            tickers = [tickers]
        df_data = yf.download(tickers, start_date, end_date)
        df_data = df_data[col_price]
        try:
            df_data.index = df_data.index.tz_convert(None)
        except:
            pass
        return df_data
        

    def _upload(self, file, path, upload_type='price', msg_exception=''):
        if upload_type.lower() == 'rate':
            func = self._upload_from_rate
        else: # default price
            func = lambda f, p: pd.read_csv(f'{p}/{f}', parse_dates=[0], index_col=[0])
        
        try:
            df_prices = func(file, path)
            DataManager.print_info(df_prices, str_sfx='uploaded.')
            return df_prices
        except Exception as e:
            return print(f'{msg_exception}{e}')


    def _upload_from_rate(self, file, path, print_msg_threshold=5):
        """
        file: master file of assets with ticker, file, adjusting data, etc
        """
        pfile = f'{path}/{file}'
        df_info = pd.read_csv(pfile)

        n_df_info = len(df_info)
        if n_df_info > print_msg_threshold:
            print_ind = False
        else:
            print_ind = True
        df_prices = None
        errors = list()
        
        print(f'Estimating price from "{pfile}" ...')
        for _, data in df_info.iterrows():
            # Using the combined class with the context manager
            with IndentOutput(indent=2).indented_output():
                df, err = convert_rate_to_price(data, path=path, print_msg=print_ind)
            if df_prices is None:
                df_prices = df.to_frame()
            else:
                df_prices = df_prices.join(df, how='outer')
            errors.append(err)

        if not print_ind:
            print(f'Max error of {n_df_info} conversions: {max(errors)*100:.2f} %')
        #print('Done.')
        return df_prices

    
    def get_names(self, tickers=None, reset=False):
        asset_names = self.asset_names
        df_prices = self.df_prices
        if reset or (asset_names is None):
            asset_names = self._get_tickers(self.universe, tickers=tickers)
            if (asset_names is None) or len(asset_names) == 0:
                return print('ERROR: Failed to get ticker names')
            else:
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
            return print(f'ERROR from get_names: {e}')


    def get_date_range(self, df_prices=None, return_intersection=False):
        df_prices = self._check_var(df_prices, self.df_prices)
        asset_names = self.asset_names
        if df_prices is None:
            return print('ERROR')
        else:
            return get_date_range(df_prices, asset_names, 
                                  return_intersection=return_intersection)


    def check_days_in_year(self, days_in_year=251, freq='M', n_thr=10):
        df_prices = self.df_prices
        if df_prices is None:
            return print('ERROR')
        else:
            return check_days_in_year(df_prices, days_in_year=days_in_year, freq=freq, n_thr=n_thr)


    def _convert_assets_to_daily(self, confirm=False, assets=None):
        df_prices = self.df_prices
        if df_prices is None:
            return print('ERROR')
            
        if confirm:
            if assets is None:
                assets = df_prices.columns
            # convert assets to daily
            df = df_prices[assets].apply(lambda x: convert_to_daily(x.dropna()))
            # update self.df_prices with the converted by unstack, concat and unstack 
            # to makes sure outer join of datetime index 
            self.df_prices = (pd.concat([df_prices.drop(assets, axis=1).unstack(), df.unstack()])
                              .unstack(0).ffill())
            days_in_year = 365
            print(f'REMINDER: {len(assets)} equities converted to daily (days in year: {days_in_year})')
            print('Daily metrics in Performance statistics must be meaningless')
            return None
        else:
            return print('WARNING: set confirm to True to convert df_assets to daily')


    def convert_to_daily(self, confirm=False, days_in_year=None):
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        df = self.check_days_in_year(252)
        if df is None:
            return None
        else:
            cols = df.loc[df==days_in_year].index
        return self._convert_assets_to_daily(confirm, cols)


    def performance(self, metrics=None, sort_by=None):
        df_prices = self.df_prices
        if df_prices is None:
            return print('ERROR')
        
        df_stat = performance_stats(df_prices, metrics=None)
        df_stat = df_stat.T
        
        if metrics is None:
            metrics = df_stat.columns
        else:
            if isinstance(metrics, str):
                metrics = [metrics]
            metrics = [y for x in metrics for y in df_stat.columns if x in y]
            df_stat = df_stat[metrics]
        
        if self.asset_names is not None:
            df_stat = pd.Series(self.asset_names).to_frame('name').join(df_stat, how='right')
        
        if sort_by is not None:
            sort_by = [x for x in metrics if sort_by in x]
            if len(sort_by) > 0:
                df_stat = df_stat.sort_values(sort_by[0], ascending=False)
        return df_stat

    
    @staticmethod
    def get_stats(df, start_date=None, end_date=None, stats=['mean', 'median', 'std'], 
                  stats_daily=True, date_format='%Y-%m-%d', msg=True,
                  plot=False, figsize=(7,3)):
        """
        df: df of date index and ticker columns. ex) self.df_prices 
        stats_daily:
         set to True to calculate statistics of daily averages for all stocks
         set to False to calculate statistics of stock averages for the period
        """
        df = df.loc[start_date:end_date]
        axis = 1 if stats_daily else 0
        df_m = df.mean(axis=axis)
        df_stats = df_m.agg(stats)
        
        if msg:
            dt0, dt1 = get_date_minmax(df, date_format)
            ps = 'daily' if stats_daily else 'stock'
            print(f'Stats of {ps} averages from {dt0} to {dt1}:')
            print(df_stats.map(lambda x: f'{x:.1f}').to_frame(ps).T.to_string())
    
        if plot and stats_daily:
            ax = df_m.plot(figsize=figsize)
            
        return df_m

    
    @staticmethod
    def get_start_end_dates(start_date: str = None, end_date: str = None, 
                            close_today: bool = True, n_years: int = 3, 
                            date_format: str = '%Y-%m-%d') -> tuple[str, str]:
        """
        get start & end dates for downloading stock price data 
        """
        today = datetime.today()
        
        # Set end_date
        if end_date is None:
            end_date = today
        else:
            end_date = datetime.strptime(end_date, date_format)
        
        # Adjust end_date if market should be closed today
        if not close_today and end_date.date() == today.date():
            end_date -= timedelta(days=1)
        
        # Set start_date
        if start_date is None:
            start_date = end_date.replace(year=end_date.year - n_years, month=1, day=1)
        else:
            start_date = datetime.strptime(start_date, date_format)
        
        return [x.strftime(date_format) for x in (start_date, end_date)]

        

class KRXDownloader():
    def __init__(self, start_date, end_date=None, close_today=True,
                 cols_pykrx={'ticker':'Symbol', 'price':'종가', 'vol':'거래량', 'date':'date'}):
        _, end_date = DataManager.get_start_end_dates(start_date, end_date, close_today, 
                                                      date_format='%Y-%m-%d')
        self.start_date = start_date
        self.end_date = end_date
        self.cols_pykrx = cols_pykrx
        self.market = None
        self.tickers = None
        self.df_data = None
        self.failed = [] # tickers failed to download
    
    def get_tickers(self, market=['KOSPI', 'KOSDAQ'], tickers=None):
        """
        market: KOSPI, KOSDAQ
        tickers: list of tickers to download
        """
        if tickers is not None:
            self.market = None
            self.tickers = tickers
            return print(f'REMINDER: {len(tickers)} tickers set regardless of market')
            
        if isinstance(market, str):
            market = [market]
        if not isinstance(market, list):
            return print('ERROR')
       
        date = self.end_date
        col_symbol = self.cols_pykrx['ticker']
        
        tickers = list()
        for x in market:
            tickers += pyk.get_market_ticker_list(date, market=x)
    
        #excluded = fdr.StockListing('KRX-DELISTING') # error
        excluded = fdr.StockListing('KRX-ADMIN')[col_symbol]
        tickers = list(set(tickers) - set(excluded))
        
        self.market = market
        self.tickers = tickers
        return None
            
    def download(self, interval=5, pause_duration=1, msg=False):
        cols = self.cols_pykrx
        col_price = cols['price']
        col_vol = cols['vol']
        col_date = cols['date']
        tickers = self.tickers
        self.failed = [] # reset for new downloading
        
        get_price = lambda x: pyk.get_market_ohlcv(self.start_date, self.end_date, x)

        tracker = TimeTracker(auto_start=True)
        df_data = None
        for x in tqdm(tickers):
            df = get_price(x)
            if len(df) > 0:
                df = df[col_price].rename(x)
                df_data = df if df_data is None else pd.concat([df_data, df], axis=1)
            else:
                self.failed.append(x)
            tracker.pause(interval=interval, pause_duration=pause_duration, msg=msg)
        tracker.stop()
        
        self.df_data = df_data.rename_axis(col_date)
        n = len(self.failed)
        return print(f'WARNING: {n} tickers failed to download') if n>0 else None

    def save(self, file='krx_prices.csv', path='.'):
        if self.df_data is None:
            print('ERROR')
        file = set_filename(file, 'csv')
        name, ext = splitext(file)
        start = self.convert_date_format(self.start_date)
        end = self.convert_date_format(self.end_date)
        file = f'{name}_{start}_{end}{ext}'
        f = f'{path}/{file}'
        self.df_data.to_csv(f)
        print(f'{file} saved')

    def convert_date_format(self, date, format_from='%Y-%m-%d', format_to='%Y%m%d'):
        if isinstance(date, str):
            return datetime.strptime(date, format_from).strftime(format_to)
        else:
            return date.strftime(date, format_to)
            


class PortfolioBuilder():
    def __init__(self, df_universe, file=None, path='.', name='portfolio',
                 method_select='all', sort_ascending=False, n_assets=0, lookback=0, lag=0, assets=None, 
                 method_weigh='Equally', weights=None, lookback_w=None, lag_w=None, 
                 df_additional=None, asset_names=None, 
                 cols_record = {'date':'date', 'ast':'asset', 'name':'name', 'prc':'price', 
                                'trs':'transaction', 'net':'net', 'wgt':'weight', 'wgta':'weight*'},
                 date_format='%Y-%m-%d'
                ):
        """
        file: file of transaction history. 
              Do not update the asset prices with the actual purchase price, 
               as the new df_universe may be adjusted with updated prices after the purchase.
        method_select: 'all', 'selected' for static, 'momentum', 'k-ratio', 'f-ratio' for dynamic
        lookback_w, lag_w: for weigh. reuse those for select if None
        sort_ascending: set to False for momentum & k-ratio, True for PER of f-ratio
        asset_names: dict of ticker to name
        """
        self.df_universe = df_universe
        # set temp name for self._load_transaction
        file = set_filename(file, default='tmp.csv')
        file = self._retrieve_transaction_file(file, path)
        self.file = file
        self.path = path
        
        self.method_select = method_select
        self.sort_ascending = sort_ascending
        self.n_assets = n_assets
        self.lookback = lookback # period for select
        self.lag = lag # days
        self.assets = assets # see select
        self.method_weigh = method_weigh
        self.weights = weights
        self.lookback_w = self._check_var(lookback_w, self.lookback) # for weigh
        self.lag_w = self._check_var(lag_w, self.lag)
        self.df_additional = df_additional
        self.asset_names = asset_names
        self.name = name # portfolio name
        self.cols_record = cols_record
        self.date_format = date_format # date str format for record & printing
        
        self.selected = None # data for select, weigh and allocate
        self.df_rec = None # record updated with new transaction
        self.liquidation = Liquidation() # for instance of Liquidation
        # different from the record file if profit_on_transaction_date is True
        self.record = self.import_record()
            

    def import_record(self, record=None, profit_on_transaction_date=False, print_msg=True):
        """
        read record from file and update transaction dates
        """
        if record is None:
            record = self._load_transaction(self.file, self.path, print_msg=print_msg)
    
        if record is None:
            print('REMINDER: make sure this is 1st transaction as no records provided')
        else:
            if profit_on_transaction_date:
                record = self._update_transaction_dates(record, self.df_universe, self.cols_record['date'])
                #print('Transaction dates updated for profit/loss on the dates')
        return record


    def select(self, date=None, method=None, sort_ascending=None, 
               n_assets=None, lookback=None, lag=None, assets=None,  
               df_additional=None):
        """
        date: transaction date
        method: all, selected, momentum, k-ratio, f-ratio
        assets: list of assets in the universe
        df_additional: ex) df_ratio for f-ratio method
        """
        method = self._check_var(method, self.method_select)
        n_assets = self._check_var(n_assets, self.n_assets)
        lookback = self._check_var(lookback, self.lookback)
        lag = self._check_var(lag, self.lag)
        assets = self._check_var(assets, self.assets)
        sort_ascending = self._check_var(sort_ascending, self.sort_ascending)
        df_additional = self._check_var(df_additional, self.df_additional)

        if (n_assets is not None) and (assets is not None):
            if n_assets > len(assets):
                return print('ERROR: n_assets greater than length of assets')
        
        # search transaction date from universe
        kwa = dict(date=date, assets=assets)
        date = self._get_data(0, 0, **kwa).index.max()
        df_data = self._get_data(lookback, lag, **kwa)
        dts = get_date_minmax(df_data, self.date_format)
        info_date = f'from {dts[0]} to {dts[1]}'
        
        method = method.lower()
        if method == 'k-ratio':
            rank = (df_data.pct_change(1).apply(lambda x: calc_kratio(x.dropna()))
                    .sort_values(ascending=sort_ascending)[:n_assets])
            method = 'K-ratio'
        elif method == 'f-ratio':
            if df_additional is None:
                return print('ERROR: no df_additional available')

            try:
                dts = df_data.index.strftime(self.date_format) # cast to str for err msg
                stat = df_additional.loc[dts].mean()
            except KeyError as e:
                return print(f'ERROR: no ratio for {e}')
                
            stat = stat.loc[stat > 0]
            if len(stat) == 0:
                return print('ERROR: check df_additional')
            
            rank = stat.sort_values(ascending=sort_ascending)[:n_assets]
            if rank.index.difference(df_data.columns).size > 0:
                print('ERROR: check selected assets if price data given')
            method = 'Financial Ratio'
        elif method == 'momentum':
            #rank = bt.ffn.calc_total_return(df_data).sort_values(ascending=False)[:n_assets]
            # no difference with calc_total_return as align_axis=1
            rank = (df_data.apply(lambda x: x.dropna().iloc[-1]/x.dropna().iloc[0]-1)
                    .sort_values(ascending=sort_ascending)[:n_assets])
            method = 'Total return'
        else: # default all for static
            rank = pd.Series(1, index=df_data.columns)
            n_assets = rank.count()
            method = 'All' if assets is None else 'Selected'
                
        assets = rank.index
        self.selected = {'date': date, 'assets': assets, 'rank': rank} 
        print(f'{n_assets} assets selected by {method} {info_date}')
        return rank    

    
    def weigh(self, method=None, weights=None, lookback=None, lag=None):
        """
        method: ERC, InvVol, Equally, Specified
        weights: str, list of str, dict, or None. Used only for 'Specified' method
        """
        selected = self.selected
        method = self._check_var(method, self.method_weigh)
        weights = self._check_var(weights, self.weights)
        lookback = self._check_var(lookback, self.lookback_w)
        lag = self._check_var(lag, self.lag_w)
        
        if selected is None:
            return print('ERROR')
        else:
            date = selected['date']
            assets = selected['assets']

        df_data = self._get_data(lookback, lag, date=date, assets=assets)
        if method.lower() == 'erc':
            weights = bt.ffn.calc_erc_weights(df_data.pct_change(1).dropna())
            method = 'ERC'
        elif method.lower() == 'invvol':
            weights = bt.ffn.calc_inv_vol_weights(df_data.pct_change(1).dropna())
            method = 'Inv.Vol'
        elif method.lower() == 'specified':
            w = self._check_weights(weights, df_data, none_weight_is_error=True)
            if w is None:
                return self.liquidation.check_weights(weights)
            weights = {x:0 for x in assets}
            weights.update(w)
            weights = pd.Series(weights)
            method = 'Specified'
        else: # default equal. no need to set arg weights
            weights = {x:1/len(assets) for x in assets}
            weights = pd.Series(weights)
            method = 'Equal weights'

        self.selected['weights'] = weights # weights is series
        print(f'Weights of assets determined by {method}.')
        return weights
        

    def allocate(self, capital=10000000, commissions=0):
        """
        calc number of each asset with price and weights
        capital: rebalance assets without cash flows if set to 0
        commissions: percentage
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_ast = cols_record['ast']
        col_prc = cols_record['prc']
        col_net = cols_record['net']
        col_wgt = cols_record['wgt']
        col_wgta = cols_record['wgta']
        col_name = cols_record['name']
        # transaction being included in transaction step
        cols_all = [col_name, col_prc, col_net, col_wgt, col_wgta]
        
        asset_names = self.asset_names
        selected = self.selected
        if selected is None:
            return print('ERROR')
        
        try:
            date = selected['date']
            weights = selected['weights']
            assets = selected['assets']
        except KeyError as e:
            return print('ERROR')

        # sum capital and asset value
        record = self.record
        if record is None:
            if capital == 0:
                return print('ERROR: Neither capital nor assets to rebalance exists')
        else:
            msg = 'WARNING: No rebalance as no new transaction'
            if self.check_new_transaction(date, msg):
                # the arg capital is now cash flows
                print(f'New cash inflows of {capital:,}' ) if capital>0 else None
                val, _ = self.valuate(date, print_msg=False, plot=False)
                capital += val # add porfolio value to capital

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
        mae = (wva.loc[wva != 0] # drop assets of zero weight for mae
                  .to_frame().join(wvi)
                  .apply(lambda x: x[col_wgta]/x[col_wgt] - 1, axis=1).abs().mean() * 100)
        print(f'Mean absolute error of weights: {mae:.0f} %')
        
        # add weights as new cols
        wa = self.calc_weight_actual(df_net)
        df_net = df_net.join(wi.apply(lambda x: f'{x:.03f}')).join(wa)

        # add asset names
        if asset_names is None:
            df_net[col_name] = None
        else:
            df_net = df_net.join(pd.Series(asset_names, name=col_name), on=col_ast)
            
        return df_net[cols_all]


    def calc_weight_actual(self, df_rec):
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_prc = cols_record['prc']
        col_net = cols_record['net']
        col_wgta = cols_record['wgta']
        wva = df_rec[col_prc].mul(df_rec[col_net]).rename(col_wgta) # actual value of each asset
        wa = wva / wva.groupby(col_date).sum()
        return wa.apply(lambda x: f'{x:.04f}')


    def transaction(self, df_net, record=None):
        """
        add new transaction to records
        df_net: output of self.allocate
        record: transaction record given as dataframe
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_ast = cols_record['ast']
        col_name = cols_record['name']
        col_prc = cols_record['prc']
        col_trs = cols_record['trs']
        col_net = cols_record['net']
        col_wgt = cols_record['wgt']
        col_wgta = cols_record['wgta']
        cols_val = [col_trs, col_net] # valid record only if not None nor zero
        cols_idx = [col_date, col_ast]
        cols_all = [x for x in cols_record.values() if x not in cols_idx]
        cols_int = [col_prc, col_trs, col_net]
                
        date = df_net.index.get_level_values(0).max()
        record = self._check_var(record, self.record)
        if record is None:
            # allocation is same as transaction for the 1st time
            df_rec = df_net.assign(**{col_trs: df_net[col_net]})
        else:
            if self.check_new_transaction(date):
                # add new to record after removing additional info except for cols_all in record
                df_rec = pd.concat([record[cols_all], df_net])
                # update universe by adding assets not in the universe but in the past transactions
                df_prc = self._update_universe(df_rec, msg=True)
                df_prc = self.liquidation.set_price(df_prc)
            else:
                return None
            
            # fill missing prices (ex: old price of new assets, new price of old assets)
            # use purchase prices in the record before possible adjustment of stock prices
            lidx = [df_rec.index.get_level_values(i).unique() for i in range(2)]
            midx = pd.MultiIndex.from_product(lidx).difference(df_rec.index)
            df_m = (df_prc[lidx[1]]
                    # fill NaN with zero price to avoid missing assets on some dates when stacking
                    # it will be removed if no transaction or net 
                    .fillna(0) 
                    .stack().loc[midx]
                    .rename_axis([col_date, col_ast]).to_frame(col_prc))
            if self.asset_names is not None: # add asset names
                df_m = df_m.join(pd.Series(self.asset_names, name=col_name), on=col_ast)
            df_rec = pd.concat([df_rec, df_m])
            
            # the net amount of the assets not in hold on the date is 0
            cond = df_rec[col_net].isna()
            cond = cond & (df_rec.index.get_level_values(0) == date)
            df_rec.loc[cond, [col_net, col_wgta]] = 0  
            
            # update transaction on the date by using the assets on the date 
            # and all the transaction before the date
            df_trs = (df_rec.loc[date, col_net]
                      .sub(df_rec.groupby(col_ast)[col_trs].sum())
                      .to_frame(col_trs).assign(**{col_date:date})
                      .set_index(col_date, append=True).swaplevel())
            df_rec.update(df_trs)
            # drop new assets before the date
            df_rec = df_rec.dropna(subset=cols_val) 
            # drop rows with neither transaction nor net 
            cond = (df_rec.transaction == 0) & (df_rec.net == 0)
            df_rec = df_rec.loc[~cond]

        df_rec = df_rec[cols_all]
        df_rec[cols_int] = df_rec[cols_int].astype(int).sort_index(level=[0,1])
        self.df_rec = df_rec
        # print portfolio value and profit/loss after self.df_rec updated
        _ = self.valuate(plot=False)
        return df_rec


    def valuate(self, date=None, print_msg=True, plot=True, **kw_plot):
        """
        calc cashflow, portfolio value and profit/loss of self.record or self.df_rec
        date_format: self.date_format
        """
        cols_record = self.cols_record
        col_prc = cols_record['prc']
        col_trs = cols_record['trs']
        col_net = cols_record['net']
        date_format = self.date_format
        
        # get latest record
        df_rec = self._check_result()
        if df_rec is None:
            return None
        
        # update price data by adding assets not in the universe if existing
        #df_prices = self._update_universe(df_rec, msg=print_msg)
        df_prices = self._update_universe(df_rec, msg=False)

        # check date by price data
        if date is None: # to get the value of latest day after last transaction
            date = df_prices.index.max()
            date_lt = df_rec.index.get_level_values(0).max()
            if date_lt > date:
                dt = date_lt.strftime(date_format)
                return print(f"ERROR: Price data predates the last transaction on {dt}")
        else: # to get the value of the date regardeless of the last transaction
            if isinstance(date, str):
                date = datetime.strptime(date, date_format)
            date_lp = df_prices.index.max()
            date_ft = df_rec.index.get_level_values(0).min()
            if not (date_ft <= date <= date_lp):
                dt = date.strftime(date_format)
                return print(f'ERROR: No price data on {dt} or no transaction before {dt}')
        
        # get record to date
        df_rec = df_rec.loc[:date]
        date_lt = df_rec.index.get_level_values(0).max()
        # cashflow (cost if positive) on date. see _calc_cashflow_history for cf history
        cflow = df_rec[col_prc].mul(df_rec[col_trs]).sum()
        # calc value
        n_assets = df_rec.loc[date_lt, col_net]
        val = n_assets.mul(df_prices.loc[date, n_assets.index]).sum().astype(int)
        
        if print_msg:
            dt = date.strftime(date_format)
            print(f'Portfolio value {val:,}, Profit {val/cflow-1:.1%} on {dt}')

        if plot:
            self.plot(msg_cr=False, **kw_plot)
        else:
            return (val, cflow)


    def transaction_pipeline(self, date=None, capital=10000000, commissions=0, 
                             record=None, save=False, **kw_liq):
        """
        kw_liq: kwargs for Liquidation.prepare
        """        
        self.liquidation.prepare(self.record, **kw_liq)
        rank = self.select(date=date)
        if rank is None:
            return None # rank is not None even for static portfolio (method_select='all')
        
        if not self.check_new_transaction():
            # calc profit at the last transaction
            dt = self.selected['date'] # selected defined by self.select
            _ = self.valuate(dt, plot=False)
            return self.record

        weights = self.weigh()
        if weights is None:
            return None
        
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


    def get_value_history(self):
        """
        get history of portfolio value
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
        else:
            return self._calc_value_history(df_rec, self.name, msg=True)


    def get_cash_history(self):
        """
        get history of cash-flow
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
        else:
            return self._calc_cashflow_history(df_rec)


    def get_profit_history(self, percent=True, log=False, msg=True):
        """
        get history of profit/loss
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
            
        sr_val = self._calc_value_history(df_rec, self.name, msg=msg)
        if (sr_val is None) or (len(sr_val)==1):
            return print('ERROR: need more data to plot')

        sr_cf = self._calc_cashflow_history(df_rec) # cashflow
        sr_prf = self._calc_profit(sr_val, sr_cf, percent, log) # profit
        return sr_prf

    
    def plot(self, figsize=(10,4), legend=True, 
             msg_cr=True, start_date=None, end_date=None, margin=0.02, 
             pnl_percent=True, pnl_log=False):
        """
        plot total value of portfolio
        """
        df_rec = self._check_result(msg_cr)
        if df_rec is None:
            return None
        
        sr_val = self._calc_value_history(df_rec, self.name, msg=True)
        if (sr_val is None) or (len(sr_val)==1):
            return print('ERROR: need more data to plot')

        sr_cf = self._calc_cashflow_history(df_rec) # cashflow
        sr_prf = self._calc_profit(sr_val, sr_cf, pnl_percent, pnl_log) # profit

        # transaction dates
        dates_trs = df_rec.index.get_level_values(0).unique()
            
        # plot historical of portfolio value
        ax1 = sr_val.plot(figsize=figsize, label='Value', title='Portfolio Growth')
        ax1.vlines(dates_trs, 0, 1, transform=ax1.get_xaxis_transform(), lw=0.5, color='grey')
        ax1.tick_params(axis='y', labelcolor=ax1.get_lines()[0].get_color())
        #ax1.autoscale(enable=True, axis='x', tight=True)
    
        # set x & y lim
        ax1.set_xlim(mldate(start_date), mldate(end_date))
        sr = sr_val.loc[start_date:end_date]
        ax1.set_ylim(sr.min()*(1-margin), sr.max()*(1+margin))
        
        # plot profit history
        label = 'Profit'
        label = f'{label} (%)' if pnl_percent else label
        ax2 = sr_prf.plot(ax=ax1.twinx(), label=label, alpha=0.4, color='orange')
        # set env for the twins
        _ = set_matplotlib_twins(ax1, ax2, legend=legend)
    
        return None
        

    def performance(self, metrics=None, sort_by=None):
        """
        calc performance of ideal portfolio excluding slippage
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
        
        sr_val = self._calc_value_history(df_rec, self.name)
        if sr_val is None:
            return None
        else:
            return performance_stats(sr_val, metrics=metrics, sort_by=sort_by)

    
    def check_new_transaction(self, date=None,
                              msg='ERROR: check the date as no new transaction'):
        record = self.record
        if record is None:
            print('WARNING: No record loaded')
            return True
        else:
            date_lt = record.index.get_level_values(0).max()

        if date is None:
            selected = self.selected
            if selected is None:
                print('ERROR: run select first')
                return False
            else:
                date = selected['date']

        if date_lt >= date:
            print(msg) if msg is not None else None
            return False
        else:
            return True


    def save_transaction(self, df_rec):
        file, path = self.file, self.path
        df_rec = self.liquidation.recover_record(df_rec, self.cols_record)
        self.file = self._save_transaction(df_rec, file, path)
        if self.file is not None:
            self.record = self.import_record(df_rec)
        return None
        

    def update_record(self, asset_names=None, save=True, update_var=True):
        """
        update and save record: ticker names of None and actual weight
        save: overwrite record file if True
        """
        # load record and check if any ticker name is None
        # reload record as self.record could been modified for liquidation
        record = self.import_record()
        if record is None:
            return None
        
        # update ticker name
        cond = record.name.isna()
        if cond.sum() > 0:
            # check if ticker name provided
            asset_names = self._check_var(asset_names, self.asset_names)
            if asset_names is None:
                print('WARNING: Set asset_names to update names of None')
            else:
                try:
                    col_name = self.cols_record['name']
                    record.loc[cond, col_name] = record.loc[cond].apply(lambda x: asset_names[x.name[1]], axis=1)
                    print('Ticker names of None updated')
                except KeyError as e:
                    print(f'ERROR: KeyError {e} to update names')

        # update actual weights
        wa = self.calc_weight_actual(record)
        record.update(wa)
        print('Actual weights updated')

        if save:
            file, path = self.file, self.path
            record.to_csv(f'{path}/{file}')
            print(f'Transaction file {file} updated')
            if update_var:
                self.record = self.import_record(print_msg=False)
                print(f'self.record updated')
        return record


    def copy_record(self, date=None, save=True):
        """
        copy latest transaction of record with new price on date
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_name = cols_record['name']
        col_prc = cols_record['prc']
        col_trs = cols_record['trs']
        col_net = cols_record['net']
        col_wgta = cols_record['wgta']
    
        # get transaction record not modified for liquidation
        df_rec = self.import_record()
        if df_rec is None:
            return None
        else:
            df_rec = self.liquidation.recover_record(df_rec, cols_record)
    
        # get universe and adjust copy date
        df_data = self.df_universe
        if date is not None:
            df_data = df_data.loc[:date]
        date = df_data.index.max()
    
        # check if copy date is latest
        if not self.check_new_transaction(date):
           return None
    
        # copy the latest transaction
        date_lt = df_rec.index.get_level_values(0).max()
        df_net = df_rec.loc[date_lt, [col_name, col_net]]
        df_net = df_net.loc[df_net[col_net]>0]
        df_net[col_trs] = 0
    
        # get price at the date
        df_prc = df_data.loc[date, df_net.index]
        df_net = df_net.join(df_prc.to_frame(col_prc)).assign(date=date)
        df_net = df_net.set_index(col_date, append=True).swaplevel()
    
        # calc actual weight
        wa = self.calc_weight_actual(df_net)
        df_net = df_net.join(wa)
    
        # append copied as new transaction
        df_rec = pd.concat([df_rec, df_net])
    
        if save:
            self.save_transaction(df_rec)
    
        return df_rec

    
    def view_record(self, n_latest=0, df_rec=None, msg=True):
        """
        get 'n_latest' latest or oldest transaction record 
        """
        if df_rec is None:
            df_rec = self._check_result(msg)
        if df_rec is None:
            return None

        idx = df_rec.index.get_level_values(0).unique().sort_values(ascending=True)
        if n_latest > 0:
            idx = idx[:n_latest]
        elif n_latest < 0:
            idx = idx[n_latest:]
        else:
            pass
        return df_rec.loc[idx]


    def check_weights(self, *args, **kwargs):
        return BacktestManager.check_weights(*args, **kwargs)
        

    def check_additional(self, date=None, df_additional=None, 
                         stats=['mean', 'median', 'std'], 
                         plot=False, figsize=(8,5), title='History of Additional data', legend=True,
                         market_label='Market', market_color='grey', market_alpha=0.5, market_line='--'):
        """
        check df_additional
        date: a transaction date from record
        """
        df_additional = self._check_var(df_additional, self.df_additional)
        if df_additional is None:
            return print('ERROR: no df_additional available')
    
        df_rec = self._check_result(False)
        if df_rec is None:
            print('No record')
            assets = None
        else:
            # Retrieve the date and assets for the transaction closest to the arg date
            df = df_rec.loc[:date]
            if len(df) > 0:
                date = df.index.get_level_values(0).max() 
                assets = df.loc[date].index.to_list()
            else:
                # date is not None since df is df_rec then
                print(f'No record on {date}')
                assets = None
        df_all = df_additional.loc[date:]
        if len(df_all) == 0:
            return print('ERROR: update df_additional first')
    
        try:
            df_res = None if assets is None else df_all[assets] 
        except KeyError as e:
            return print(f'ERROR: KeyError {e}')
    
        if plot:
            ax1 = (df_all.mean(axis=1)
                   .plot(figsize=figsize, title=title, label=market_label, 
                         alpha=market_alpha, c=market_color, linestyle=market_line))
            if df_res is not None:
                ax2 = df_res.plot(ax=ax1.twinx())
                _ = set_matplotlib_twins(ax1, ax2, legend=legend)
    
        return df_res
        
        
    def _calc_cashflow_history(self, record):
        """
        Returns a series of resultant cash flows at each transaction.
         negative for outflows, positive for inflows.
        """
        cols_record = self.cols_record
        col_prc = cols_record['prc']
        col_trs = cols_record['trs']
        col_date = cols_record['date']
        return (record[col_prc].mul(record[col_trs])
                 .groupby(col_date).sum().cumsum().mul(-1))
        

    def _update_universe(self, df_rec, msg=False):
        """
        create price histories from record to update universe
        """
        df_prices = self.df_universe
        cols_record = self.cols_record
        col_prc = cols_record['prc']
        col_ast = cols_record['ast']
        # assets not in the universe
        out = df_rec.index.get_level_values(1).unique().difference(df_prices.columns)
        if out.size > 0:
            idx = pd.IndexSlice
            df_out = df_rec.sort_index().loc[idx[:, out], col_prc].unstack(col_ast)
            df_new = pd.concat([df_prices, df_out], axis=1)
            df_new[out] = df_new[out].ffill().bfill()
            if msg:
                s = ', '.join(out.to_list())
                print(f'Assets {s} added to universe')
            return df_new
        else:
            return df_prices


    def _calc_value_history(self, df_rec, name, msg=False):
        """
        calc historical of portfolio value from transaction
        """
        col_net = self.cols_record['net']
        end = datetime.today()
        sr_tot = pd.Series()
        dates_trs = df_rec.index.get_level_values(0).unique()
        # update price data with df_rec
        df_universe = self._update_universe(df_rec, msg=msg)
        
        # loop for transaction dates in descending order
        for start in dates_trs.sort_values(ascending=False):
            n_assets = df_rec.loc[start, col_net]
            df_i = df_universe.loc[start:end, n_assets.index]
            if len(df_i) == 0: # no price data from transaction date start
                continue
            # calc combined asset value history from prv transaction (start) to current (end) 
            sr_i = df_i.apply(lambda x: x*n_assets.loc[x.name]).sum(axis=1) # x.name: index name
            # concat histories        
            sr_tot = pd.concat([sr_tot, sr_i])
            end = start - pd.DateOffset(days=1)

        if len(sr_tot) > 0:
            # sort by date
            return sr_tot.astype(int).rename(name).sort_index()
        else:
            return print('ERROR: no historical')
            
            
    def _plot_cashflow(self, ax, sr_cashflow_history, xmax, 
                       label='Cash Flows', alpha=0.4, colors=('r','g')):
        df_cf = sr_cashflow_history.rename('y').rename_axis('x1').reset_index()
        df_cf = df_cf.join(df_cf.x1.shift(-1).rename('x2')).fillna(xmax)
        df_cf = df_cf[['y', 'x1', 'x2']]
        args_vline = [x.to_list() for _, x in df_cf.iterrows()]
        kwargs = dict(label=label, alpha=alpha)
        return [ax.hlines(*args, color= colors[0] if args[0] < 0 else colors[1], **kwargs)
                for args in args_vline]


    def _calc_profit(self, sr_val, sr_cashflow_history, percent=True, log=False):
        df = (sr_val.to_frame('value')
              .join(sr_cashflow_history.abs().rename('cflow'), how='outer')
              .ffill().fillna(0))
        if percent:
            if log:
                df = df.apply(lambda x: np.log(x.value / x.cflow), axis=1).mul(100)
            else:
                df = df.apply(lambda x: x.value / x.cflow - 1, axis=1).mul(100)
        else:
            df = df.apply(lambda x: x.value - x.cflow, axis=1)
        return df
        

    def _check_result(self, msg=True):
        if self.df_rec is None:
            if self.record is None:
                return print('ERROR: No transaction record') if msg else None
            else:
                df_res = self.record
        else:
            df_res = self.df_rec
        return df_res
    

    def _load_transaction(self, file, path, print_msg=True):
        col_ast = self.cols_record['ast']
        f = os.path.join(path, file)
        if os.path.exists(f):
            df_rec = pd.read_csv(f, parse_dates=[0], index_col=[0,1], dtype={col_ast:str})
        else:
            return None
        # check if ticker of asset is 6 digits
        df_rec = (df_rec.reset_index(level=1)
                        .assign(asset=lambda x: x.asset.str.zfill(6))
                        .set_index(col_ast, append=True))
        
        if print_msg:
            dt = df_rec.index.get_level_values(0).max().strftime(self.date_format)
            print(f'Transaction record to {dt} loaded')
        return df_rec
        

    def _save_transaction(self, df_rec, file, path, pattern=r"_\d+(?=\.\w+$)"):
        """
        save df_rec and return file name
        """
        # add date to file name
        dt = df_rec.index.get_level_values(0).max()
        dt = f"_{dt.strftime('%y%m%d')}"

        file = get_filename(file, dt, r"_\d+(?=\.\w+$)")
        _ = save_dataframe(df_rec, file, path, 
                           msg_succeed=f'All transactions saved to {file}',
                           msg_fail=f'ERROR: failed to save as {file} exists')
        return file
        
    
    def _check_var(self, arg, arg_self):
        return arg_self if arg is None else arg


    def _retrieve_transaction_file(self, file, path):
        """
        get the latest transaction file
        """
        return get_file_latest(file, path, msg=True, file_type='record')


    def _update_transaction_dates(self, record, df_universe, col_date):
        """
        modify transaction dates to be able to compare purchase price 
         and close price on the transaction date 
        """
        dts_trs = record.index.get_level_values(0).unique()
        dts_dict = {x: df_universe.loc[:x - timedelta(days=1)].index.max() for x in dts_trs}
        return (record.reset_index(level=0)
                      .assign(**{col_date: lambda x: x[col_date].apply(lambda x: dts_dict[x])})
                      .set_index(col_date, append=True).swaplevel())

    
    def _get_date_offset(self, *args, **kwargs):
        return BacktestManager.get_date_offset(*args, **kwargs)


    def _get_data(self, lookback, lag, date=None, assets=None):
        """
        get data for select or weigh
        """
        df_data = self.df_universe
        if date is not None:
            df_data = df_data.loc[:date]
        if assets is not None:
            df_data = df_data[assets]
            
        # setting liquidation
        df_data = self.liquidation.set_price(df_data, select=True)
        # set date range
        date = df_data.index.max()
        dt1 = date - self._get_date_offset(lag, 'weeks')
        dt0 = dt1 - self._get_date_offset(lookback, 'month')
        return df_data.loc[dt0:dt1].dropna(axis=1)
    

class Liquidation():
    def __init__(self):
        self.assets_to_sell = None
        
    def prepare(self, record, assets_to_sell=None, hold=False):
        """
        convert assets_to_sell to dict of tickers to sell price
        record: PortfolioBuilder.record
        assets_to_sell: str of a ticker; list of tickers; dict of the tickers to its sell price
        hold:     
        - If set to True, all assets in `assets_to_sell` will be held and not liquidated.    
        - If set to False, you can selectively hold certain assets by setting their sell price to zero. 
          In this case, only the specified assets in `assets_to_sell` will be held, while others may still be liquidated.
        """
        # set self.assets_to_sell first to data check
        self.assets_to_sell = assets_to_sell
        if assets_to_sell is None:
            return print('Liquidation set to None')
        
        if record is None:
            return print('ERROR: no record to liquidate')
    
        if isinstance(assets_to_sell, str):
            liq = [assets_to_sell]
        elif isinstance(assets_to_sell, dict):
            liq = [x for x, _ in assets_to_sell.items()]
        elif isinstance(assets_to_sell, list):
            liq = assets_to_sell
        else:
            return print('ERROR: check arg assets_to_sell')
            
        # check if tickers to sell exist in record
        date_lt = record.index.get_level_values(0).max()
        record_lt = record.loc[date_lt]
        if pd.Index(liq).difference(record_lt.index).size > 0:
            return print('ERROR: some tickers not in record')
    
        if not isinstance(assets_to_sell, dict):
            price = 0 if hold else None
            # liq is list regardless of type of assets_to_sell
            assets_to_sell = {x:price for x in liq}
    
        self.assets_to_sell = assets_to_sell
        return print('Liquidation prepared')

    
    def set_price(self, df_prices, select=False):
        """
        update df_prices for liquidation
        df_prices: df_universe for select or transaction
        select: exclude tickers to liquidate in record from universe.
                set to True for self.select
        """
        liq_dict = self.assets_to_sell
        if liq_dict is None:
            return df_prices
        else:
            df_data = df_prices.copy()

        if select: # exclude tickers to liquidate in record from universe
            df_data = df_data.drop(list(liq_dict.keys()), axis=1, 
                                   errors='ignore' # tickers might be delisted from kospi200
                                  )
        else: # reset price of liquidating
            tickers_all = df_data.columns
            for ticker, prc in liq_dict.items():
                if prc is None:
                    if ticker not in tickers_all:
                        print(f'ERROR: {ticker} has no sell price')
                        return df_prices
                else: # set sell price
                    df_data[ticker] = prc
       
        return df_data


    def check_weights(self, weights):
        """
        check if weights has tickers to liquidate
        """
        liq_dict = self.assets_to_sell
        if liq_dict is None:
            return None

        if isinstance(weights, str):
            w_list = [weights]
        elif isinstance(weights, dict):
            w_list = weights.keys()
        elif isinstance(weights, list):
            w_list = weights
        else:
            return print('ERROR: check type of weights')
            
        w = [x for x in liq_dict.keys() if x in w_list]
        if len(w) > 0:
            return print('ERROR: assets to liquidate in weights')
        else:
            return None

    
    def recover_record(self, df_rec, cols_rec):
        """
        reset net and transaction of assets in hold
        """
        liq_dict = self.assets_to_sell
        if liq_dict is None:
            return df_rec
        else:
            liq = list(liq_dict.keys())

        if df_rec is None:
            return print('ERROR')

        date_lt = df_rec.index.get_level_values(0).max()
        df = df_rec.loc[date_lt]
        if pd.Index(liq).difference(df.index).size > 0:
            print('ERROR: some tickers not in record')
            return df_rec
            
        col_prc = cols_rec['prc']
        col_net = cols_rec['net']
        col_trs = cols_rec['trs']
        cond = (df_rec[col_prc] == 0) & (df_rec.net == 0)
        if cond.sum() > 0:
            df_rec.loc[cond, col_net] = -df_rec.loc[cond, col_trs]
            df_rec.loc[cond, col_trs] = 0
            print('Holdings recovered')
        return df_rec



class BacktestManager():
    def __init__(self, df_assets, name_prfx='Portfolio',
                 align_axis=0, fill_na=True, metrics=METRICS,  
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
        self.cv_result = None # dict of cv result
        self.metrics = metrics
        self.name_prfx = name_prfx
        self.n_names = 0 # see self._check_name
        self.initial_capital = initial_capital
        # commissions of all assets across portfolios
        self.commissions = commissions  # unit %
        self.run_results = None # output of bt.run
        self.days_in_year = days_in_year # only for self._get_algo_freq
        # saving to apply the same rule in benchmark data
        self.align_axis = align_axis
        self.fill_na = fill_na
        self.asset_names = asset_names
        self.print_algos_msg = True # control msg print in self._get_algo_*

        # run after set self.df_assets
        DataManager.print_info(df_assets, str_sfx='uploaded.')
        if days_in_year > 0:
            print('running self.util_check_days_in_year to check days in a year')
            _ = self.util_check_days_in_year(df_assets, days_in_year, freq='M', n_thr=1)
        

    def align_period(self, df_assets, axis=0, date_format='%Y-%m-%d',
                     fill_na=True, print_msg1=True, print_msg2=True, n_indent=2):
        if axis is None:
            return df_assets
        else:
            return align_period(df_assets, axis=axis, date_format=date_format, fill_na=fill_na, 
                                print_msg1=print_msg1, print_msg2=print_msg2, n_indent=n_indent)


    def _check_name(self, name=None):
        if name is None:
            self.n_names += 1
            name = f'{self.name_prfx}{self.n_names}'
        return name

    
    def _check_var(self, var_arg, var_self):
        if var_arg is None:
            var_arg = var_self
        return var_arg

    @staticmethod
    def check_weights(weights, dfs, none_weight_is_error=False):
        """
        return equal weights if weights is str or list
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
        elif isinstance(weights, dict):
            c = list(weights.keys())
            cols = pd.Index(c).difference(dfs.columns)
            if len(cols) == 0:
                w = list(weights.values())
                if round(sum(w),2) == 1:
                    return weights
                else:
                    return print('ERROR: sum of weights is not 1')
            else:
                cols = ', '.join(cols)
                return print(f'ERROR: No {cols} in the dfs')
        else:
            if none_weight_is_error:
                print('ERROR: weights is None')
            return weights
    

    def backtest(self, dfs, name='portfolio', 
                 select={'select':'all'}, freq={'freq':'year'}, weigh={'weigh':'equally'},
                 algos=None, commissions=None, **kwargs):
        """
        kwargs: keyword args for bt.Backtest except commissions
        algos: List of Algos
        """
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


    def _get_algo_select(self, select='all', n_assets=0, lookback=0, lag=0, 
                         id_scale=1, threshold=None, df_ratio=None, ratio_descending=None,
                         tickers=None):
        """
        select: all, momentum, kratio, randomly, specified, list of tickers
        ratio_descending, df_ratio: args for AlgoSelectFinRatio
        tickers: list of tickers to select in SelectThese algo
        """
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        lb = self._get_date_offset(lookback)
        lg = self._get_date_offset(lag, 'weeks')

        if isinstance(select, list): # set for SelectThese
            tickers = select
            select = 'Specified'
              
        if cond(select, 'Momentum'):
            algo_select = SelectMomentum(n=n_assets, lookback=lb, lag=lg, threshold=threshold)
            # SelectAll() or similar should be called before SelectMomentum(), 
            # as StatTotalReturn uses values of temp[‘selected’]
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'f-ratio'):
            algo_select = AlgoSelectFinRatio(df_ratio, n_assets, 
                                             lookback_days=lb,
                                             sort_descending=ratio_descending)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'k-ratio'):
            algo_select = AlgoSelectKRatio(n=n_assets, lookback=lb, lag=lg)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'ID'):
            id_scale = id_scale if id_scale > 1 else 2
            n_pool = round(n_assets * id_scale)
            algo_select1 = bt.algos.SelectMomentum(n=n_pool, lookback=lb, lag=lg)
            algo_select2 = AlgoSelectIDiscrete(n=n_assets, lookback=lb, lag=lg)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select1, algo_select2)
        elif cond(select, 'IDRank'):
            algo_select = AlgoSelectIDRank(n=n_assets, lookback=lb, lag=lg, scale=id_scale)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'randomly'):
            algo_after = AlgoRunAfter(lookback=lb, lag=lg)
            algo_select = bt.algos.SelectRandomly(n=n_assets)
            algo_select = bt.AlgoStack(algo_after, bt.algos.SelectAll(), algo_select)
        elif cond(select, 'specified'):
            algo_after = AlgoRunAfter(lookback=lb, lag=lg)
            algo_select = bt.algos.SelectThese(tickers)
            algo_select = bt.AlgoStack(algo_after, bt.algos.SelectAll(), algo_select)
        else:
            algo_after = AlgoRunAfter(lookback=lb, lag=lg)
            algo_select = bt.AlgoStack(algo_after, bt.algos.SelectAll())
            if not cond(select, 'all'):
                print('WARNING:SelectAll selected') if self.print_algos_msg else None
 
        return algo_select
        

    def _get_algo_freq(self, freq='M', offset=0, days_in_year=252):
        """
        freq: W, M, Q, Y, or num of months considering days_in_year
        offset (int): Applies to the first run. If 0, this algo will run the first time it is called.
        """
        if isinstance(freq, str) and (freq.lower() == 'once'):
            n_t = (0, 'once')
        else:
            n_t = BacktestManager.split_int_n_temporal(freq, 'M') # default month
    
        if n_t is None:
            return
        else:
            n, temporal = n_t        

        cond = lambda x, y: False if x is None else x[0].lower() == y[0].lower()
        if cond(temporal, 'W'):
            n *= round(days_in_year / WEEKS_IN_YEAR)
        elif cond(temporal, 'M'):
            n *= round(days_in_year / 12)
        elif cond(temporal, 'Q'):
            n *= round(days_in_year / 4)
        elif cond(temporal, 'Y'):
            n *= days_in_year
        elif cond(temporal, 'D'):
            return print(f'ERROR: days freq not supported')
        else:  # default run once
            n = 0
            if freq.lower() != 'once':
                print('WARNING:RunOnce selected') if self.print_algos_msg else None
            
        if n > 0:
            # RunEveryNPeriods counts number of index in data (not actual days)
            algo_freq = bt.algos.RunEveryNPeriods(n, offset=offset)
        else:
            algo_freq = bt.algos.RunOnce()
        return algo_freq


    def _get_algo_weigh(self, weigh='equally', 
                         weights=None, lookback=0, lag=0, rf=0, bounds=(0.0, 1.0)):
        """
        weigh: equally, erc, specified, randomly, invvol, meanvar
        """
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        lb = self._get_date_offset(lookback) # default month
        lg = self._get_date_offset(lag, 'weeks') # default week
        
        # reset weigh if weights not given
        if cond(weigh, 'Specified') and (weights is None):
            weigh = 'equally'
        
        if cond(weigh, 'ERC'):
            algo_weigh = bt.algos.WeighERC(lookback=lb, lag=lg)
            # Use SelectHasData to avoid LedoitWolf ERROR; other weights like InvVol work fine without it.
            algo_weigh = bt.AlgoStack(bt.algos.SelectHasData(lookback=lb), algo_weigh)
        elif cond(weigh, 'Specified'):
            algo_weigh = bt.algos.WeighSpecified(**weights)
        elif cond(weigh, 'Randomly'):
            algo_weigh = bt.algos.WeighRandomly()
        elif cond(weigh, 'InvVol'): # risk parity
            algo_weigh = bt.algos.WeighInvVol(lookback=lb, lag=lg)
        elif cond(weigh, 'MeanVar'): # Markowitz’s mean-variance optimization
            algo_weigh = bt.algos.WeighMeanVar(lookback=lb, lag=lg, rf=rf, bounds=bounds)
            algo_weigh = bt.AlgoStack(bt.algos.SelectHasData(lookback=lb), algo_weigh)
        else:
            algo_weigh = bt.algos.WeighEqually()
            if not cond(weigh, 'equally'):
                print('WARNING:WeighEqually selected') if self.print_algos_msg else None
            
        return algo_weigh


    @staticmethod
    def get_date_offset(n, unit='months'):
        """
        n: int or str such as '1m', '2months', '3M'
        unit: default unit
        """
        n_t = BacktestManager.split_int_n_temporal(n, unit)
        if n_t is None:
            return
        else:
            v, k = n_t

        cond = lambda x, y: False if x is None else x[0].lower() == y[0].lower()            
        if cond(k, 'd'):
            #k = 'days'
            return print(f'ERROR: days offset not supported')
        elif  cond(k, 'w'):
            k = 'weeks'
        elif  cond(k, 'm'):
            k = 'months'
        elif  cond(k, 'q'):
            k = 'months'
            v *= 3
        elif  cond(k, 'y'):
            k = 'years'
        else:
            return print(f'ERROR: {n}')
        kwarg = {k: v}

        return pd.DateOffset(**kwarg)


    @staticmethod
    def split_int_n_temporal(nt, unit='months'):
        """
        nt: int or str. ex) 0, '1m', '2months', '3M'
        unit: default unit
        """
        if isinstance(nt, int):
            return (nt, unit)
        
        match = re.match(r"(\d+)\s*([a-zA-Z]+)", nt)
        try:
            n = int(match.group(1)) 
            t = match.group(2).lower()[0]
            return (n,t)
        except Exception as e:
            return print(f'ERROR: {e}')
                

    def _get_date_offset(self, n, unit='months'):
        return BacktestManager.get_date_offset(n, unit=unit)
        

    def build(self, name=None, 
              freq='M', offset=0,
              select='all', n_assets=0, lookback=0, lag=0,
              lookback_w=None, lag_w=None,
              id_scale=1, threshold=None,
              df_ratio=None, ratio_descending=None, # args for select 'f-ratio'
              weigh='equally', weights=None, rf=0, bounds=(0.0, 1.0),
              initial_capital=None, commissions=None, algos=None, build_cv=False):
        """
        make backtest of a strategy
        lookback, lag: for select
        lookback_w, lag_w: for weigh. reuse those for select if None
        commissions: %; same for all assets
        algos: set List of Algos to build backtest directly
        build_cv: set to True to prepare cross-validate,
                  which makes self.portfolios only when running cross-validate
        """
        dfs = self.df_assets
        weights = BacktestManager.check_weights(weights, dfs)
        name = self._check_name(name)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        commissions = self._check_var(commissions, self.commissions)

        # build args for self._get_algo_* from build args
        select = {'select':select, 'n_assets':n_assets, 'lookback':lookback, 'lag':lag, 
                  'id_scale':id_scale, 'threshold':threshold,
                  'df_ratio':df_ratio, 'ratio_descending':ratio_descending}
        freq = {'freq':freq} # offset being saved when running backtest
        weigh = {'weigh':weigh, 'weights':weights, 'rf':rf, 'bounds':bounds,
                 'lookback':self._check_var(lookback_w, lookback), 
                 'lag':self._check_var(lag_w, lag)}
        
        if build_cv: # backtest inserted to self.portfolios when running cross_validate
            self.cv_strategies[name] = {
                # convert args for self.build_batch in self._cross_validate_strategy
                **select, **freq, **weigh, 'algos':None,
                'initial_capital':initial_capital, 'commissions':commissions
            }
        else:
            freq.update({'offset':offset, 'days_in_year':self.days_in_year})
            kwargs = {'select':select, 'freq':freq, 'weigh':weigh, 'algos':algos,
                      'initial_capital':initial_capital, 'commissions':commissions}
            self.portfolios[name] = self.backtest(dfs, name=name, **kwargs)
        
        return None
        

    def buy_n_hold(self, name=None, weigh='specified', weights=None, **kwargs):
        """
        weights: dict of ticker to weight. str if one asset portfolio
        kwargs: set initial_capital or commissions
        """
        return self.build(name=name, freq='once', select='all', weigh=weigh,
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
        weights = BacktestManager.check_weights(weights, dfs)
        weigh = {'weigh':'specified', 'weights':weights}
        select = {'select':'all', 'lookback':lookback, 'lag':lag}
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        commissions = self._check_var(commissions, self.commissions)
       
        self.portfolios[name] = self.backtest(dfs, name=name, select=select, 
                                              freq={'freq':'once'}, weigh=weigh, 
                                              initial_capital=initial_capital, 
                                              commissions=commissions)
        return None


    def benchmark_ticker(self, name='KODEX200', ticker=None, **kwargs):
        start, end = get_date_minmax(self.df_assets, date_format='%Y-%m-%d')
        df = BacktestManager.util_import_data(name, ticker, start_date=start, end_date=end)
        if df is None:
            return None
        else:
            print(f'Benchmark is {df.name}')
            return self.benchmark(df, **kwargs)


    def build_batch(self, *kwa_list, reset_portfolios=False, build_cv=False, **kwargs):
        """
        kwa_list: list of k/w args for each backtest
        kwargs: k/w args common for all backtest
        build_cv: set to True to prepare cross-validation.
        reset_portfolios: reset portfolios and cv_strategies
        """
        if reset_portfolios:
            self.portfolios = AssetDict(names=self.asset_names)
            self.cv_strategies = AssetDict(names=self.asset_names)
            
        _ = [self.build(**{**kwa, **kwargs, 'build_cv':build_cv}) for kwa in kwa_list]
        return print(f'{len(kwa_list)} jobs prepared for cross-validation') if build_cv else None
        
    
    def run(self, pf_list=None, metrics=None, stats=True, stats_sort_by=None,
            plot=True, start=None, end=None, freq='D', figsize=None):
        """
        pf_list: List of backtests or list of index of backtest
        """
        # convert pf_list for printing msg purpose
        pf_list = self.check_portfolios(pf_list, run_results=None, convert_index=True, build_cv=False)
        self._print_strategies(pf_list, n_max=5, work='Backtesting')
        
        run_results = self._run(pf_list)
        if run_results is None:
            return None
        else:
            self.run_results = run_results
        
        if plot:
            _ = self.plot(start=start, end=end, freq=freq, figsize=figsize)

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
                       metrics=None, simplify=False,
                       size_batch=0, file_batch='tmp_batch', path_batch='.',  delete_batch=True):
        """
        pf_list: str, index, list of str or list of index
        simplify: result format mean ± std if True, dict of cv if False 
        """
        if len(self.cv_strategies) == 0:
            return print('ERROR: no strategy to evaluate')
        else:
            n_given = 0 if pf_list is None else len(pf_list)
            pf_list = self.check_portfolios(pf_list, run_results=None, convert_index=True, build_cv=True)
            if (n_given > 0) and len(pf_list) != n_given:
                return print('ERROR: run after checking pf_list')
            else:
                self._print_strategies(pf_list, n_max=5, work='Cross-validating')
                print_algos_msg_backup = self.print_algos_msg
                self.print_algos_msg = False # hide msg from self._get_algo_*
            
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
        batch = BatchCV(size_batch, start=True, path=path_batch)
        tracker = TimeTracker(auto_start=True)
        for name in pf_list:
            if batch.check(name):
                continue
            kwargs_build = self.cv_strategies[name]
            result[name] = self._cross_validate_strategy(name, offset_list, 
                                                         metrics=metrics, **kwargs_build)
            result = batch.update(result) # result reset if saved
        tracker.stop()
        result = batch.finish(result, delete=delete_batch)    
        
        self.cv_result = result
        self.print_algos_msg = print_algos_msg_backup # restore the flag
        if simplify:
            return self.get_cv_simple(result)
        else:
            return None
        
        
    def _cross_validate_strategy(self, name, offset_list, metrics=None, **kwargs_build):
        keys = ['name', 'offset']
        kwa_list = [dict(zip(keys, [f'CV[{name}]: offset {x}', x])) for x in offset_list]
        kwargs_build = {k:v for k,v in kwargs_build.items() if k not in keys}
        # set build_cv to False to run self.backtest which saved into self.portfolios
        self.build_batch(*kwa_list, build_cv=False, **kwargs_build)
            
        pf_list = [x['name'] for x in kwa_list]
        run_results = self._run(pf_list)
        # drop backtests after getting run_results hopelly to save mem
        self.portfolios = {k: v for k, v in self.portfolios.items() if k not in pf_list}
        return self.get_stats(metrics=metrics, run_results=run_results)


    def get_cat_data(self, kwa_list, cv_result=None, file=None, path='.'):
        """
        convert cross-validation result to catplot data
        kwa_list: list of dicts of parameter sets. see build_batch for detail
        cv_result: output of cross_validate with simplify=False
        """
        cv_result = self._check_var(cv_result, self.cv_result)
        if (cv_result is None) or not isinstance(cv_result, dict):
            return print('ERROR: cv result is not dict')
            
        df_parm = pd.DataFrame(kwa_list).set_index('name')
        df_cv = pd.DataFrame()
        for k, df in cv_result.items():
            n = df.columns.size
            idx = [[k], range(n)]
            idx = pd.MultiIndex.from_product(idx)
            df = df.T.set_index(idx).assign(**df_parm.loc[k].to_dict())
            df_cv = pd.concat([df_cv, df])
        df_cv.index.names = ['set','iteration']

        if file is not None:
            _ = save_dataframe(df_cv, file, path, msg_succeed=f'{file} saved')
        
        BacktestManager.print_cv(df_cv)
        return df_cv


    def get_cv_simple(self, cv_result=None):
        cv_result = self._check_var(cv_result, self.cv_result)
        if (cv_result is None) or not isinstance(cv_result, dict):
            return print('ERROR: cv result is not dict')
        df_cv = None

        # Suppress warnings as the Sortino can sometimes be infinite when calculating std.
        with warnings.catch_warnings(category=RuntimeWarning):
            warnings.simplefilter("ignore")
            for name, stats in cv_result.items():
                idx = stats.index.difference(['start', 'end'])
                df = (stats.loc[idx]
                      .agg(['mean', 'std', 'min', 'max'], axis=1)
                      .apply(lambda x: f'{x['mean']:.02f} ± {x['std']:.03f}', axis=1)
                      .to_frame(name))
                if df_cv is None:
                    df_cv = df
                else:
                    df_cv = df_cv.join(df)
        return df_cv     


    @staticmethod
    def catplot(data, path='.', ref_val=None, **kw):
        """
        data: output of get_cat_data or its file
        kw: kwargs of sns.catplot. 
            ex) {'y':'cagr', 'x':'freq', 'row':'n_assets', 'col':'lookback', 'hue':'lag'}
        """
        if isinstance(data, str): # data is file
            try:
                f = f'{path}/{data}'
                data = pd.read_csv(f, index_col=[0,1])
                print(f'Returning {f}')
                BacktestManager.print_cv(data)
                return data
            except FileNotFoundError as e:
                return print('ERROR: FileNotFoundError {e}')
        else:
            g = sns.catplot(data=data, **kw)
            if ref_val is not None:
                if isinstance(ref_val, str):
                    ref_val = BacktestManager.benchmark_stats(kw['y'], data=data, name=ref_val)
                g.refline(y=ref_val)
            return g

    
    @staticmethod
    def print_cv(df_cv):
        n_s = df_cv.index.get_level_values(0).nunique()
        n_i = df_cv.index.get_level_values(1).size
        return print(f'{n_s} param sets with {round(n_i/n_s)} iterations per set')


    @staticmethod
    def benchmark_stats(metric='cagr', name='KODEX200', ticker=None,
                        start_date=None, end_date=None, data=None):
        """
        data: output of get_cat_data
        """
        if data is not None:
            start_date, end_date = data['start'].min(), data['end'].max()

        df = BacktestManager.util_import_data(name, ticker, start_date=start_date, end_date=end_date)
        if df is None:
            return None
        else:
            dt0, dt1 = get_date_minmax(df, date_format='%Y-%m-%d')
            print(f'Returning {metric} of {df.name} from {dt0} to {dt1}')
            return performance_stats(df).loc[metric][0]
        

    def check_portfolios(self, pf_list=None, run_results=None, convert_index=True, build_cv=False):
        """
        run_results: output from bt.run
        convert_index: convert pf_list of index to pf_list of portfolio names 
        build_cv: search porfolio args from self.cv_strategies
        """
        if run_results is None:
            if build_cv:
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
        if len(pf_list) == 0:
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
        if len(pf_list) == 0:
            return None
        
        plot_func = run_results.plot_security_weights
        return self._plot_portfolios(plot_func, pf_list, **kwargs)
        

    def plot_weights(self, pf_list=None, **kwargs):
        run_results = self.run_results
        pf_list  = self.check_portfolios(pf_list, run_results=run_results)
        if len(pf_list) == 0:
            return None
        
        plot_func = run_results.plot_weights
        return self._plot_portfolios(plot_func, pf_list, **kwargs)


    def plot_histogram(self, pf_list=None, **kwargs):
        run_results = self.run_results
        pf_list  = self.check_portfolios(pf_list, run_results=run_results)
        if len(pf_list) == 0:
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
        if len(pf_list) == 0:
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


    def plot(self, pf_list=None, start=None, end=None, freq='D', 
             figsize=None, legend=True):
        run_results = self.run_results
        if run_results is None:
            return print('ERROR: run backtest first')
        
        if pf_list is None:
            ax = run_results.plot(freq=freq, figsize=figsize, legend=legend)
        else:
            pf_list  = self.check_portfolios(pf_list, run_results=run_results, convert_index=True)
            if len(pf_list) == 0:
                return None
            fig, ax = plt.subplots()
            _ = [run_results[x].plot(ax=ax, freq=freq, figsize=figsize) for x in pf_list]
            ax.legend() if legend else None
            ax.set_title('daily Price Series')
        dates = self._get_xlim(run_results.prices, start=start, end=end)
        ax.set_xlim(*dates)
        return ax


    def _get_xlim(self, df_prices, start=None, end=None):
        """
        start, end: str or None
        """
        if start is None:
            df = df_prices.diff(-1).abs().cumsum()
            start = df.loc[df.iloc[:,0]>0].index.min()
        start = mldate(start)
        
        if end is None:
            end = df_prices.index.max()
        end = mldate(end)
        
        return (start, end)


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
        if len(pf_list) == 0:
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
        if len(pf_list) == 0:
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
        

    @staticmethod
    def util_import_data(name='KODEX200', ticker=None, start_date=None, end_date=None,
                         col='Close', date_format='%Y-%m-%d',
                         tickers={'KODEX200':'069500', 'KOSPI':'KS11', 'KOSDAQ':'KQ11', 'KOSPI200':'KS200',
                                  'S&P500':'S&P500', 'DOW':'DJI', 'NASDAQ':'IXIC'}):
        """
        import historical of ticker by using FinanceDataReader.DataReader
        """
        # set name and set ticker if name in tickers
        if name is None: 
            name = ticker
        else:
            _name = name.upper()
            if _name in tickers.keys():
                ticker = tickers[_name]
        # show available name for benchmark if no ticker given
        if ticker is None:
            names = ', '.join(tickers.keys())
            return print(f'ERROR: Set ticker or name from {names}')
        # download
        try:
            df = fdr.DataReader(ticker, start_date, end_date)
            return df[col].rename(name)
        except Exception as e:
            return print(f'ERROR: {e}')

    
    def util_check_days_in_year(self, df=None, days_in_year=None, freq='M', n_thr=10):
        df = self._check_var(df, self.df_assets)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        return check_days_in_year(df, days_in_year=days_in_year, freq=freq, n_thr=n_thr)
            


class BayesianEstimator():
    def __init__(self, df_prices, days_in_year=252, metrics=METRICS):
        # df of assets (assets in columns) which of each might have its own periods.
        # the periods of all assets will be aligned in every calculation.
        df_prices = df_prices.to_frame() if isinstance(df_prices, pd.Series) else df_prices
        if df_prices.index.name is None:
            df_prices.index.name = 'date' # set index name to run check_days_in_year
        _ = check_days_in_year(df_prices, days_in_year, freq='M', n_thr=1)
        
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



class FinancialRatios():
    def __init__(self, file, path='.', date_format='%Y-%m-%d',
                 cols_index={'date':'date', 'ticker':'ticker'},
                 ratios={'BPS':False, 'PER':True, 'PBR':True, 
                         'EPS':False, 'DIV':False, 'DPS':False}):
        file = set_filename(file, 'csv') 
        self.file = get_file_latest(file, path) # latest file
        self.path = path
        self.date_format = date_format
        self.ratios = ratios # ratios and its ascending order
        self.cols_index = cols_index
        self.df_ratios = None
        return self.upload()


    def upload(self):
        """
        load financial ratios from a file
        """
        file = self.file
        path = self.path
        if file is None:
            return print('ERROR: Download first')

        f = os.path.join(path, file)
        if os.path.exists(f):
            col_ticker = self.cols_index['ticker']
            df_ratios = pd.read_csv(f, index_col=[0,1], parse_dates=[1], dtype={col_ticker:str})
            self.df_ratios = df_ratios
            return self._print_info(df_ratios, str_sfx='loaded')
        else:
            return print(f'WARNING: No \'{file}\' exists')
        

    def download(self, tickers, start_date, end_date=None, 
                 freq='daily', close_today=False, save=True,
                 # args for TimeTracker.pause
                 interval=50, pause_duration=2, msg=False):
        """
        freq: day, month, year
        """
        col_date = self.cols_index['date']
        col_ticker = self.cols_index['ticker']
        date_format = self.date_format

        # check end date and closing
        _, end_date = DataManager.get_start_end_dates(start_date, end_date, close_today, 
                                                      date_format=date_format)
        tracker = TimeTracker(auto_start=True)
        df_ratios = pd.DataFrame()
        try:
            for ticker in tqdm(tickers):
                df = pyk.get_market_fundamental(start_date, end_date, ticker, freq=freq[0].lower())
                df = df.assign(**{col_ticker:ticker})
                df_ratios = pd.concat([df_ratios, df])
                tracker.pause(interval=interval, pause_duration=pause_duration, msg=msg)
            tracker.stop()
        except Exception as e:
            return print(f'ERROR: {e}')
        
        df_ratios = (df_ratios.rename_axis(col_date)
                     .loc[df_ratios.index <= end_date] # remove fictitious end date of month
                     .set_index(col_ticker, append=True)
                     .swaplevel())
        self._print_info(df_ratios, str_sfx='downloaded')
        self.df_ratios = df_ratios
        if save:
            self.save(self.file, self.path)

    
    def save(self, file=None, path=None, date_format='%y%m%d'):
        """
        date_format: date format for file name
        """
        file = self._check_var(file, self.file)
        path = self._check_var(path, self.path)
        df_ratios = self.df_ratios
        if (file is None) or (df_ratios is None):
            return print('ERROR: check file or df_ratios')

        date = df_ratios.index.get_level_values(1).max().strftime(date_format)
        file = get_filename(file, f'_{date}', r"_\d+(?=\.\w+$)")
        _ = save_dataframe(df_ratios, file, path, msg_succeed=f'{file} saved',
                           msg_fail=f'ERROR: failed to save as {file} exists')
        return None


    def get_ratios(self, date=None, metrics=None):
        """
        metrics: list or str
        """
        df_ratios = self.df_ratios
        if df_ratios is None:
            return print('ERROR: load ratios first')
        else:
            date = self._check_date(df_ratios, date)
            if date is None:
                return None

        if metrics is None:
            metrics = df_ratios.columns.to_list()

        col_date = self.cols_index['date']
        df_res = self._get_ratio(df_ratios, date, metrics).droplevel(col_date)

        metrics = '/'.join(metrics) if isinstance(metrics, list) else metrics
        print(f'{metrics} on {date}')
        return df_res
    

    def calc_rank(self, date=None, metrics='PER', topn=10, scale='minmax'):
        """
        calc the rank of financial ratios
        metrics: list or str
        """
        df_ratios = self.df_ratios
        if df_ratios is None:
            return print('ERROR: load ratios first')
        else:
            date = self._check_date(df_ratios, date)
            if date is None:
                return None

        if isinstance(metrics, str):
            metrics = [metrics]

        if len(metrics) > 1:
            if scale not in ['minmax', 'zscore']:
                return print('ERROR: Set scale to sum up multiple ranks')
        
        res_rank = None
        for m in metrics:
            sr_rank = self._calc_rank(df_ratios, date, m, scale)
            if res_rank is None:
                res_rank = sr_rank
            else:
                res_rank += sr_rank 
                
        metrics = '+'.join(metrics)
        s = metrics if topn is None else f'top {topn} stocks of low {metrics}'
        print(f'Ranking score of {s} on {date}')
        col_date = self.cols_index['date']
        return res_rank.droplevel(col_date).sort_values(ascending=True).iloc[:topn]


    def get_stats(self, metrics=None, start_date=None, end_date=None, stats=['mean', 'std'],
                  stats_daily=True):
        """
        stats_daily:
         set to True to calculate statistics of daily averages for all stocks
         set to False to calculate statistics of stock averages over the given period
        """
        df_ratios = self.df_ratios
        if df_ratios is None:
            return print('ERROR: load ratios first')
    
        if metrics is None:
            metrics = df_ratios.columns
        if isinstance(metrics, str):
            metrics = [metrics]
    
        col_date = self.cols_index['date']
        col_ticker = self.cols_index['ticker']
        col_metric = 'metric'
        idx = pd.IndexSlice
        df_r = (df_ratios.sort_index().loc[idx[:, start_date:end_date], metrics]
                    .rename_axis(col_metric, axis=1))
        # get dates
        cols_fn = ['first', 'last']
        cols_se = ['start','end']
        df_d = (df_r.stack().dropna().reset_index(col_date)
                .groupby(col_metric).date.agg(cols_fn)
                .apply(lambda x: x.dt.strftime(self.date_format))
                .rename(columns=dict(zip(cols_fn, cols_se)))
                .T
                .loc[:, metrics] # set to original order of metrics
               )
        # calc stats
        by = col_date if stats_daily else col_ticker
        ps = 'daily' if stats_daily else 'stock'
        df_s = (df_r.dropna()
                .groupby(by).mean() # daily mean of each metric
                .agg(stats) # stats of daily mean of each metric
                .map(lambda x: f'{x:.1f}'))

        print(f'Returning stats of {ps} averages')
        # calc stats and append to date
        return pd.concat([df_d, df_s])
        

    def calc_historical(self, metrics='PER', scale='minmax'):
        df_ratios = self.df_ratios
        if df_ratios is None:
            return print('ERROR: load ratios first')
        
        if isinstance(metrics, str):
            metrics = [metrics]
    
        try:
            df_r = df_ratios[metrics]
        except KeyError as e:
            return print(f'ERROR: KeyError {e}')
        
        if len(metrics) > 1:
            if scale not in ['minmax', 'zscore']:
                return print('ERROR: Set scale to sum up multiple ranks')    
    
        col_date = self.cols_index['date']
        sr_historical = None
        for m in metrics:
            ascending = self.ratios[m]
            sr_h = self._calc_historical(df_r[m], ascending, scale, col_date)
            if sr_historical is None:
                sr_historical = sr_h
            else:
                sr_historical += sr_h

        metrics = '+'.join(metrics)
        print(f'Historical of {metrics} ranking score created')
        return sr_historical


    def interpolate(self, sr_prices, metric='PER', freq='M'):
        """
        calculates an interpolated ratio for date range intersection of price & ratio
        sr_prices: series of price with index of (ticker, date)
        """
        col_ticker = self.cols_index['ticker']
        col_date = self.cols_index['date']
        col_price = 'price'
        col_mpl = 'multiplier'
        col_ym = 'year_month'
    
        df_ratios = self.df_ratios
        if df_ratios is None:
            return print('ERROR: load ratios first')
            
        # Copy metric column to avoid modifying original
        df_m = df_ratios[metric].copy()
    
        # check if ratios missing
        gtck = lambda x: x.index.get_level_values(0).unique()
        n = gtck(sr_prices).difference(gtck(df_m)).size 
        if n > 0:
            return print(f'ERROR: ratios of {n} stocks missing')
    
        # check frequency
        if not self._check_freq(df_ratios, col_date, n_in_year=12):
            print('WARNING: No interpolation as data is not monthly')
            return df_m

        # set price date range with metric range
        start_date, end_date = get_date_minmax(df_m, level=1)
        if freq.lower() == 'm':
            end_date = end_date + pd.DateOffset(months=1)
        else:
            raise NotImplementedError
    
        # check date intersection
        start, end = get_date_minmax(sr_prices, level=1)
        if (start_date > end) or (end_date < start):
            return print('ERROR: no intersection of dates btw price and ratios')
    
        # get multiplier to calc ratio from price
        i0 = df_m.index.get_level_values(0).unique()
        i1 = pd.date_range(start=start_date, end=end_date)
        idx = pd.MultiIndex.from_product([i0, i1], names=df_m.index.names)
        df_m = (sr_prices.to_frame(col_price)
                .join(df_m, how='outer').ffill()  # combine all dates of both price and ratio
                .apply(lambda x: x[metric] / x[col_price], axis=1).rename(col_mpl) # calc multiplier
                .loc[df_m.index].to_frame(col_mpl) # slice multipliers with dates in ratio
                # ffill multipliers from start_date and end_date
                .join(pd.DataFrame(index=idx), how='right').ffill()
        )
        
        # interpolate ratio
        idx = pd.IndexSlice
        df_res = (sr_prices.loc[idx[:, start_date:end_date]]
                  .to_frame(col_price)
                  .join(df_m) # join with ratio multiplier
                  .apply(lambda x: x[col_price] * x[col_mpl], axis=1)
                  .rename(metric)
                 )
        if len(df_res) == 0:
            # redundant?
            return print('ERROR: no intersection of dates btw price and ratios')
        else:    
            dt0, dt1 = get_date_minmax(df_res, self.date_format, 1)
            print(f'{metric} interpolated from {dt0} to {dt1}')
            return df_res
    

    def _calc_historical(self, sr_ratio, ascending, scale, col_date):
        return (sr_ratio.groupby(level=col_date).rank(ascending=ascending)
                .groupby(level=col_date, group_keys=False).apply(lambda x: self._scale(x, scale))
               )
    

    def _get_ratio(self, df_ratios, date, metrics):
        """
        get financial ratios on date
        metrics: list or str
        """
        try:
            idx = pd.IndexSlice
            return df_ratios.sort_index().loc[idx[:,date], metrics]
        except KeyError as e:
            return print(f'ERROR: KeyError {e}')
        

    def _calc_rank(self, df_ratios, date, metric, scale, drop_zero=True):
        """
        calc the rank of a financial ratio
        metric: str
        """
        sr_ratio = self._get_ratio(df_ratios, date, metric)
        if sr_ratio is None:
            return # see _get_ratio for error msg
        
        if drop_zero:
           sr_ratio = sr_ratio.loc[sr_ratio>0]
            
        ascending = self.ratios[metric]
        sr_rank = sr_ratio.rank(ascending=ascending)
        return self._scale(sr_rank, scale)


    def _scale(self, sr_rank, scale):
        """
        scale rank
        """
        if scale == 'minmax':
            sr_rank = (sr_rank-sr_rank.min()) / (sr_rank.max()-sr_rank.min())
        elif scale == 'zscore':
            sr_rank  = (sr_rank-sr_rank.mean()) / sr_rank.std()
        else:
            pass
        return sr_rank


    def _check_date(self, df_ratios, date, return_str=True):
        """
        date: set date for ratios in df_ratios.
              'start' for the earlist date in df_ratios, 
              'end' or None for the latest date,
        """
        dates = df_ratios.index.get_level_values(1).unique()
        if date == 'start':
            date = dates.min()
        elif date in [None, 'end']:
            date = dates.max()
        else:
            cond = dates <= date
            if cond.any():
                date = dates[cond].max()
            else:
                date = dates.min()
                print(f'WARNING: date set to {date.strftime(self.date_format)}')
        
        if return_str:
            date = date.strftime(self.date_format)
        return date

    
    def _print_info(self, df, str_pfx='Financial ratios of', str_sfx=''):
        dt0, dt1 = get_date_minmax(df, self.date_format, 1)
        n = df.index.get_level_values(0).nunique()
        s1  = str_pfx + " " if str_pfx else ""
        s2  = " " + str_sfx if str_sfx else ""
        return print(f'{s1}{n} stocks from {dt0} to {dt1}{s2}')
        

    def _check_var(self, var_arg, var_self):
        return var_self if var_arg is None else var_arg


    def _check_freq(self, df, col_date, n_in_year=12):
        df = df.groupby(col_date).last()
        n = check_days_in_year(df, msg=False).mean()
        if n == n_in_year:
            return True
        else:
            return False


    def util_reshape(self, df_data=None, stack=True, swaplevel=True):
        """
        Converts price data from **DataManager** 
         to f-ratios in **FinancialRatios** format, or vice versa 
        df_data: price or f-ratios
        """
        cols_index = self.cols_index
        col_ticker = cols_index['ticker']
        col_date = cols_index['date']

        df_ratios = self.df_ratios
        if df_data is None:
            df_data = df_ratios
        else:
            if isinstance(df_data, str) and df_ratios is not None:
                try:
                    df_data = df_ratios[df_data]
                except KeyError as e:
                    return print(f'ERROR: No {df_data} in ratio')
                
        if df_data is None:
            return print('ERROR: load or set ratios first')
            
        try:
            if stack: # convert to DataManager format
                df = df_data.stack()
                if swaplevel:
                    df = df.swaplevel().rename_axis([col_ticker,col_date]).sort_index()
            else: # convert to FinancialRatios format
                df = df_data.unstack(0)
        except Exception as e:
            return print(f'ERROR: {e}')

        return df


    def util_compare_periods(self, df, level=0):
        """
        compare date index between df_ratios and df (ex. price)
        level: index level of date
        """
        df_ratios = self.df_ratios
        date_format = self.date_format

        getd = lambda x, l=1: x.index.get_level_values(l).unique()
        dts = getd(df, level).difference(getd(df_ratios))
        n = dts.size
        if n > 0:
            d0 = dts.min().strftime(date_format)
            d1 = dts.max().strftime(date_format)
            print(f'WARNING: {n} days from {d0} to {d1} missing in the ratio')
        return None
        


class BatchCV():
    """
    manage batch process of cross_validation
    """
    def __init__(self, size=0, result_reset=dict(), start=True,
                 file=None, path='.', pattern=r"_\d+(?=\.\w+$)"):
        self.size = size # batch size
        self.result_reset = result_reset # result reset
        # temp file to save result
        self.file = set_filename(file, 'pkl', 'tmp_batch')
        self.path = path
        self.pattern = pattern # suffix for temp file name
        # the number of batches done in the latest cross-validation
        self.n_last = 0
        # list of cross-validation finished last time
        self.jobs_finished = list()
        self.start() if start else None

    def start(self, msg=True):
        if self.size == 0: # not batch process
            return None
        result = self.load()
        self.n_last = self._get_last()
        self.jobs_finished = self._get_finished(result)
        n = len(self.jobs_finished)
        if msg and (n>0):
            print(f'Make sure {n} jobs done before')
        return None
    
    def load(self, files=None):
        result = self._reset_result()
        if files is None:
            files = get_file_list(self.file, self.path)
        for file in files:
            f = f'{self.path}/{file}'
            with open(f, 'rb') as handle:
                res = pickle.load(handle)
                result = self._append(result, res)
        return result

    def check(self, job):
        """
        use to process cross-validation
        job: name of cv iteration
        """
        if self.size == 0:
            # skip continue and do the following process as not batch process
            return False 
            
        if job in self.jobs_finished:
            return True # skip to next job as the job done before
        else:
            return False # do the following process

    def update(self, result, forced=False):
        """
        save result of a batch and reset the result for next batch
        result: dict of results of _cross_validate_strategy
        forced: set to True to save regardless of size_batch
        """
        size_batch = self.size
        if size_batch == 0: # not batch
            return result

        # save result every size_batch jobs finished
        if (len(result) % size_batch == 0) or forced:
            self.n_last += 1
            self._save(result, self.n_last)
            return self._reset_result() # reset the result for next batch
        else: # pass result w/o saving batch
            return result
            
    def finish(self, result, delete=False):
        """
        run after batch loop to save the rest of jobs
        delete: set to True to delete temp files
        """
        # pass result if no batch or no temp saved
        if self.size * self.n_last == 0:
            return result
            
        # save the rest of jobs
        if not self._is_reset(result):
            _ = self.update(result, forced=True)
    
        # reload all batch files
        result = self.load()
        # update batch status
        self.n_last = self._get_last()
        self.jobs_finished = self._get_finished(result)
        if delete:
            files = get_file_list(self.file, self.path)
            _ = [os.remove(f'{self.path}/{x}') for x in files]
            print('Temp batch files deleted')
        return result

    def _reset_result(self):
        """
        return reset value for batch
        """
        result_reset = self.result_reset
        if isinstance(result_reset, dict):
            return result_reset.copy()
        elif result_reset is None:
            return None
        else:
            raise NotImplementedError
            
    def _is_reset(self, result):
        """
        return True if result is reset
        """
        result_reset = self.result_reset
        if isinstance(result_reset, dict):
            return len(result) == 0
        elif result_reset is None:
            return result is None
        else:
            raise NotImplementedError
    
    def _get_last(self):
        """
        get the last batch number from temp files saved during the last cross-validation
        """
        file = get_file_latest(self.file, self.path)
        match = re.search(self.pattern, file)
        return int(match[0].lstrip('_')) if bool(match) else 0

    def _get_finished(self, result):
        """
        get the list of names of jobs done the last cross-validation
        """
        if isinstance(result, dict):
            return list(result.keys())
        else:
            raise NotImplementedError

    def _append(self, result, new_result):
        """
        combine results in temp files to one obj such as dict
        """
        if result is None:
            result = new_result
        elif isinstance(result, dict):
            if isinstance(new_result, dict):
                result.update(new_result)
            else:
                print('ERROR: Update failed as result is not dict')
        else:
            raise NotImplementedError
        return result
        
    def _save(self, result, n_current):
        """
        save a batch to temp file of pickle
        """
        file, ext = splitext(self.file)
        file = f'{self.path}/{file}_{n_current:03}{ext}'
        with open(file, 'wb') as handle:
            pickle.dump(result, handle)



class PortfolioManager():
    """
    manage multiple portfolios 
    """
    def __init__(self, pf_names=None):
        """
        pf_names: list of portfolio names
        """
        self.pf_data = PortfolioData()
        self.pf_names = self.check_portfolios(pf_names)
        self.portfolios = self.load(self.pf_names)

    
    def load(self, pf_names=None):
        """
        loading multiple portfolios (no individual args except for PortfolioData)
        pf_names: list of portfolio names
        """
        pf_dict = dict()
        for name in pf_names:
            print(f'{name}:')
            pf_dict[name] = PortfolioManager.create_portfolio(name)
            print()
        return pf_dict


    @staticmethod
    def review(space=None):
        pfd = PortfolioData()
        return pfd.review(space)
    
    @staticmethod
    def review_portfolio(pf_name, strategy=False, universe=False):
        pfd = PortfolioData()
        return pfd.review_portfolio(pf_name, strategy=strategy, universe=universe)

    @staticmethod
    def review_universe(name):
        pfd = PortfolioData()
        return pfd.review_universe(name)

    @staticmethod
    def review_strategy(name):
        pfd = PortfolioData()
        return pfd.review_strategy(name)

    
    @staticmethod
    def create_universe(name, *args, **kwargs):
        """
        args, kwargs: args & kwargs for DataManager
        """
        pfd = PortfolioData()
        kwa_u = pfd.review_universe(name)
        return DataManager(*args, **{**kwa_u, **kwargs})

    
    @staticmethod
    def create_portfolio(name, *args, **kwargs):
        """
        name: portfolio name
        args, kwargs: additional args & kwargs for PortfolioBuilder
        """
        # get args of portfolios
        pfd = PortfolioData()
        
        # get the instance of DataManager
        kwa_p = pfd.review_portfolio(name, strategy=False, universe=False)
        dm = PortfolioManager.create_universe(kwa_p['universe'])
        
        # get the instance of *Portfolio
        kwa_s = pfd.review_portfolio(name, strategy=True, universe=False)
        kwa_s = {**kwa_s, 'name':name, 'asset_names':dm.get_names()}
        return PortfolioBuilder(dm.df_prices, *args, **{**kwa_s, **kwargs})


    def check_portfolios(self, pf_names=None):
        pf_names = self.pf_names if pf_names is None else pf_names
        pf_all = self.pf_data.portfolios.keys()
        if pf_names is None:
            pf_names = pf_all
        else:
            pf_names = [pf_names] if isinstance(pf_names, str) else pf_names
            if len(set(pf_names)-set(pf_all)) > 0:
                return print('ERROR: check portfolio names')
        return pf_names

    
    def plot(self, pf_names=None, start_date=None, end_date=None, percent=True,
             figsize=(8,4), legend=True, 
             colors = plt.cm.Spectral):
        """
        start_date: date of beginning of the return plot
        end_date: date to calc return
        """
        # check portfolios
        pf_names = self.check_portfolios(pf_names)
        if pf_names is None:
            return None
        else: # cacl portfolio return for title
            df = self._valuate(pf_names, end_date)
            r = df.loc['Total'].to_dict()
            totr = r['Return']
            title = f"Total Return: {round(totr,-3):,.0f} ({r['Percentage']:.1%})"
    
        # individual return
        dfs = [self.portfolios[x].get_profit_history(percent=percent, msg=False) for x in pf_names]
        dfs = [v.rename(k) for k,v in zip(pf_names, dfs) if v is not None]
        ax1 = pd.concat(dfs, axis=1).ffill().loc[start_date:end_date].plot(alpha=0.5)
        ax1.set_prop_cycle(color=colors(np.linspace(0,1,len(pf_names))))
        ax1.set_ylabel('Portfolio Returns (%)')
        
        # total profit/loss
        color_tot = 'gray'
        dfs = [self.portfolios[x].get_profit_history(percent=False, msg=False) for x in pf_names]
        ax2 = ax1.twinx()
        df_ttl = (pd.concat(dfs, axis=1).ffill().sum(axis=1).rename('Total')
              .loc[start_date:end_date])
        _ = df_ttl.plot(ax=ax2, c=color_tot, ls='--', title=title, figsize=figsize)
        ax2.set_ylabel('Total Return')
        _ = set_matplotlib_twins(ax2, ax1, legend=legend)

        # return marker
        _ = ax2.plot(df_ttl.index[-1], totr, 
                     markeredgecolor=color_tot, markersize=10, alpha=0.6,
                     marker=6 if totr >0 else 7,  
                     markerfacecolor= 'blue' if totr >0 else 'red')
        
        
    def valuate(self, pf_names=None, date=None):
        pf_names = self.check_portfolios(pf_names)
        if pf_names is None:
            return None
        else:
            return self._valuate(pf_names, date)


    def _valuate(self, pf_names, date):
        """
        calc gain/loss and rate of return for each portfolio
        pf_names: list of portfolio names
        """
        l_ret, l_pct = list(), list()
        val, cflow = 0, 0
        for name in pf_names:
            pf = self.portfolios[name]
            try:
                v, c = pf.valuate(date=date, plot=False, print_msg=False)
                l_ret.append(v - c)
                l_pct.append(v/c - 1)
                val += v
                cflow += c
            except Exception as e:
                print(f'ERROR:({name}) {e}')
        index = pf_names.copy()
        #if cflow != c: # no total if one portfolio given
        if True: # always add total for convenience
            index.append('Total')
            l_ret.append(val-cflow)
            l_pct.append(val/cflow-1)
        data = {'Return':l_ret, 'Percentage':l_pct}
        return pd.DataFrame(data=data, index=index)