import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.units as munits
import FinanceDataReader as fdr
import arviz as az
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import xml.etree.ElementTree as ET
import os, time, re, sys, pickle, random
import bt
import warnings
import seaborn as sns
import yfinance as yf
import requests

from datetime import datetime, timedelta
from datetime import date as datetime_date
from contextlib import contextmanager
from os import listdir
from os.path import isfile, join, splitext
from ffn import calc_stats, calc_perf_stats
from pykrx import stock as pyk
from tqdm import tqdm
from matplotlib.dates import num2date, date2num, ConciseDateConverter
from matplotlib.gridspec import GridSpec
from numbers import Number

from pf_data import PORTFOLIOS, STRATEGIES, UNIVERSES
from pf_custom import (AlgoSelectKRatio, AlgoRunAfter, calc_kratio, AlgoSelectIDiscrete, 
                       AlgoSelectIDRank, SelectMomentum, AlgoSelectFinRatio, 
                       RedistributeWeights, redistribute_weights)


warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=pd.errors.PerformanceWarning)

# support korean lang
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams["font.family"] = 'NanumBarunGothic'

# all calls to axes that have dates are to be made using this converter
converter = ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[datetime_date] = converter
munits.registry[datetime] = converter


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

    # calc mean days for each ticker
    df_days = (df.assign(gb=df.index.strftime(grp_format)).set_index('gb')
                 .apply(lambda x: x.dropna().groupby('gb').count()[1:-1])
                 #.fillna(0) # comment as it distorts mean
                 .mul(factor).mean().round()
                 .fillna(0) # for the case no ticker has enough days for the calc
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
            #print(f'WARNING: the number of days in a year with {n} tickers ({p:.0f}%) is not {days_in_year} in setting:')
            print(f'WARNING: the number of days in a year with {n} tickers ({p:.0f}%) is {df.mean():.0f} in avg.')
    
    return df_days


def align_period(df_prices, axis=0, date_format='%Y-%m-%d',
                 fill_na=True, print_msg1=True, print_msg2=True, n_indent=2):
    """
    axis: Determines the operation for handling missing data.
        0 : Drop rows (time index) with missing prices.
        1 : Drop columns (tickers) with a count of non-missing prices less than the maximum found.
    fill_na: set False to drop nan fields
    """
    msg1 = None
    if axis == 0:
        df_aligned = get_date_range(df_prices, return_intersection=True)
        if len(df_aligned) < len(df_prices):
            dts = get_date_minmax(df_aligned, date_format)
            msg1 = f"period reset: {' ~ '.join(dts)}"
    elif axis == 1:
        c_all = df_prices.columns
        df_cnt = df_prices.apply(lambda x: x.dropna().count())
        cond = (df_cnt < df_cnt.max())
        c_drop = c_all[cond]
        df_aligned = df_prices[c_all.difference(c_drop)]
        n_c = len(c_drop)
        if n_c > 0:
            n_all = len(c_all)
            msg1 = f'{n_c} tickers removed for shorter periods ({n_c/n_all*100:.1f}%)'
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


def set_filename(file, ext=None, default='test'):
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


def save_dataframe(df, file, path='.', overwrite=False,
                   msg_succeed='file saved.',
                   msg_fail='ERROR: failed to save as the file exists',
                   **kwargs):
    """
    kwargs: kwargs for to_csv
    """
    f = os.path.join(path, file)
    if os.path.exists(f) and not overwrite:
        print(msg_fail)
        return False
    else:
        df.to_csv(f, **kwargs)    
        print(msg_succeed)
        return True
        

def performance_stats(df_prices, metrics=METRICS, sort_by=None, align_period=True, idx_dt=['start', 'end']):
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


def convert_to_daily(df, method='linear'):
    """
    convert df to daily time series
    method: 'ffill', 'linear'
    """
    start = df.index.min()
    end = df.index.max()
    index = pd.date_range(start, end)
    if method == 'linear':
        return df.reindex(index, method=None).interpolate()
    else:
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


def set_matplotlib_twins(ax1, ax2, legend=True, colors=None, loc='upper left'):
    """
    colors: list of color for ax1 and ax2
    """
    axes = [ax1, ax2]
    # set tick & y label colors
    colors = [x.get_lines()[0].get_color() for x in axes] if colors is None else colors
    _ = [x.tick_params(axis='y', labelcolor=colors[i]) for i, x in enumerate(axes)]
    _ = [x.yaxis.label.set_color(colors[i]) for i,x in enumerate(axes)]  
    # drop individual legends
    _ = [None if x.get_legend() is None else x.get_legend().remove() for x in axes]
    if legend:
        # set legend
        h1, h2 = [x.get_legend_handles_labels()[0] for x in axes]
        if len(h1)*len(h2) > 0:
            ax1.legend(handles=h1+h2, loc=loc)
    return (ax1, ax2)


def format_rounded_string(*ns, string='ROI: {:.1%}, UGL: {:,.0f}', n_round=3):
    """
    Formats a customizable string with rounded numeric values.
    Parameters:
        *ns (float): Numbers to be inserted into the template string.
        template (str): String template for formatting, with placeholders for values.
        n_round (int): Number of digits to round values based on magnitude.
    Returns:
        str:
    """
    myround = lambda x: x if abs(x) < 10**(n_round+1) else round(x,-n_round)
    ns = [myround(x) for x in ns]
    return string.format(*ns)


def sum_dateoffsets(offset1, offset2):
    """
    Sums two pandas DateOffset objects by combining their kwargs.
    
    Parameters:
        offset1 (pd.DateOffset): The first DateOffset.
        offset2 (pd.DateOffset): The second DateOffset.
        
    Returns:
        pd.DateOffset: The resulting DateOffset.
    """
    # Extract kwargs from both offsets
    kwargs1 = offset1.kwds
    kwargs2 = offset2.kwds
    
    # Sum the respective kwargs
    combined_kwargs = {
        key: kwargs1.get(key, 0) + kwargs2.get(key, 0)
        for key in set(kwargs1) | set(kwargs2)
    }
    
    # Return a new DateOffset with the combined kwargs
    return pd.DateOffset(**combined_kwargs)


def string_shortener(x, n=20, r=1, ellipsis="..."):
    """
    Clips a string to a specified length, inserting an ellipsis ('...') 
     and cleaning up any surrounding special characters to ensure a tidy output.
    """
    if len(x) <= n:
        return x

    if r == 1:
        result = f"{x[:n]}{ellipsis}"
    elif r == 0:
        result = f"{ellipsis}{x[-n:]}"
    else:
        n1 = int(n * r)
        n2 = int(n * (1 - r))
        result = f"{x[:n1]}{ellipsis}{x[-n2:]}"

    # Remove special characters immediately surrounding the custom ellipsis
    result = re.sub(r"([^a-zA-Z0-9\s])" + re.escape(ellipsis), f"{ellipsis}", result)  # Before the ellipsis
    result = re.sub(re.escape(ellipsis) + r"([^a-zA-Z0-9\s])", f"{ellipsis}", result)  # After the ellipsis
    return result


def create_split_axes(figsize=(10, 6), vertical_split=True, 
                      ratios=(3, 1), share_axis=False, space=0):
    """
    Creates a figure with two subplots arranged either vertically or horizontally.

    Parameters:
    -----------
    figsize : tuple, optional
        The size of the figure (width, height) in inches. Default is (10, 6).
    vertical_split : bool, optional
        If True, splits the figure vertically (stacked subplots).
        If False, splits the figure horizontally (side-by-side subplots). Default is True.
    ratios : tuple, optional
        Ratios of the sizes of the two subplots. Default is (3, 1).
    share_axis : bool, optional
        If True, the axes will share either the x-axis (for vertical split) or the y-axis (for horizontal split).
        Default is True.

    Returns:
    --------
    tuple
        A tuple containing the two subplot axes (ax1, ax2).

    Example:
    --------
    >>> ax1, ax2 = create_split_axes(figsize=(12, 8), vertical_split=False, ratios=(2, 1), share_axis=False)
    """
    fig = plt.figure(figsize=figsize)
    if vertical_split:
        gs = GridSpec(2, 1, figure=fig, hspace=space, height_ratios=ratios)
        sp1 = gs[:-1, :]
        sp2 = gs[-1, :]
    else:
        gs = GridSpec(1, 2, figure=fig, wspace=space, width_ratios=ratios)
        sp1 = gs[:, :-1]
        sp2 = gs[:, -1]
        
    ax1 = fig.add_subplot(sp1)
    ax2 = fig.add_subplot(sp2)
    
    if share_axis:
        if vertical_split:
            ax1.sharex(ax2)
        else:
            ax1.sharey(ax2)
    
    return (ax1, ax2)
    

class SecurityDict(dict):
    """
    A dictionary subclass that associates keys (ex:security tickers) with names.
    Attributes:
        names (dict): Optional dictionary mapping tickers to names.
    """
    def __init__(self, *args, names=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.names = names

    def __repr__(self):
        return self._print(self.keys())

    def _print(self, keys):
        output = ""
        for i, key in enumerate(keys):
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

    def get_names(self, keys):
        if isinstance(keys, str):
            keys = [keys]
        res = self._print(keys)
        print(res)


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
                 universe='kospi200', tickers='KRX/INDEX/STOCK/1028', 
                 daily=True, days_in_year=12):
        """
        universe: kospi200, etf, krx, fund, etc. only for getting tickers. see pf_data for more
        file: price history. set as kw for pf_data
        tickers: ticker for getting pool of tickers. can be a file name for tickers as well.
        daily: set to False if price is monthly
        days_in_year: only for convert_to_daily
        """
        file = set_filename(file, 'csv') 
        self.file_historical = get_file_latest(file, path) # latest file
        self.path = path
        self.universe = universe
        self.tickers = tickers
        self.security_names = None 
        self.df_prices = None
        self.daily = daily
        self.days_in_year = days_in_year
        # update self.df_prices
        self.upload(self.file_historical, get_names=True, convert_to_daily=not daily)

    
    def upload(self, file=None, path=None, get_names=False, convert_to_daily=False):
        """
        load df_prices from saved file
        """
        file = self._check_var(file, self.file_historical)
        path = self._check_var(path, self.path)
        if file is None:
            return print('ERROR: no file to load.')
        else:
            df_prices = self._upload(file, path, msg_exception='WARNING: uploading failed as ')

        if df_prices is None:
            return None # error msg printed out by self._upload
        
        self.df_prices = df_prices
        self.get_names(reset=True) if get_names else None
        self.convert_to_daily(True, self.days_in_year) if convert_to_daily and not self.daily else None
        return print('Price data loaded')
        

    @print_runtime
    def download(self, start_date=None, end_date=None, n_years=3, tickers=None,
                 save=True, overwrite=False, date_format='%Y-%m-%d', close_today=False,
                 **kwargs_download):
        """
        download df_prices by using FinanceDataReader
        n_years: int
        tickers: None for all in new universe, 'selected' for all in df_prices, 
                 or list of tickers in new universe
        kwargs_download: args for krx. ex) interval=5, pause_duration=1, msg=False
        """
        start_date, end_date = DataManager.get_start_end_dates(start_date, end_date, 
                                                               close_today, n_years, date_format)
        print('Downloading ...')
        
        security_names = self._get_tickers(tickers)
        if security_names is None:
            return None # see _get_tickers for error msg
        else:
            tickers = list(security_names.keys())
                   
        try:
            df_prices = self._download_universe(tickers, start_date, end_date, **kwargs_download)
            if not close_today: # market today not closed yet
                df_prices = df_prices.loc[:datetime.today() - timedelta(days=1)]
            print('... done')
            DataManager.print_info(df_prices, str_sfx='downloaded.')
        except Exception as e:
            return print(f'ERROR: {e}')
            
        self.df_prices = df_prices
        self.security_names = security_names
        if save:
            if not self.save(date=df_prices.index.max(), overwrite=overwrite):
                return None
        # convert to daily after saving original monthly
        self.convert_to_daily(True, self.days_in_year) if not self.daily else None
        return print('df_prices updated')

    
    def save(self, file=None, path=None, date=None, overwrite=False, date_format='%y%m%d'):
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
        return save_dataframe(df_prices, file, path, overwrite=overwrite,
                               msg_succeed=f'{file} saved',
                               msg_fail=f'ERROR: failed to save as {file} exists')

    
    def _check_var(self, var_arg, var_self):
        return var_self if var_arg is None else var_arg


    @staticmethod
    def print_info(df_prices, str_pfx='', str_sfx='', date_format='%Y-%m-%d'):
        dt0, dt1 = get_date_minmax(df_prices, date_format)
        n = df_prices.columns.size
        s1  = str_pfx + " " if str_pfx else ""
        s2  = " " + str_sfx if str_sfx else ""
        return print(f'{s1}{n} securities from {dt0} to {dt1}{s2}')


    def _get_tickers(self, tickers=None, **kwargs):
        """
        tickers: None for all in new universe, 'selected' for all in df_prices, 
                 or list of tickers in new universe
        """
        uv = self.universe.lower()
        if uv == 'kospi200':
            func = self._get_tickers_kospi200
        elif uv == 'etf':
            func = self._get_tickers_etf
        elif uv == 'fund':
            func = self._get_tickers_fund
        elif uv == 'file':
            func = self._get_tickers_file
        elif uv == 'krx':
            func = self._get_tickers_krx
        elif uv == 'yahoo':
            func = self._get_tickers_yahoo
        else:
            func = lambda **x: None
     
        try:
            security_names = func(tickers, **kwargs)
            failed = 'ERROR: Failed to get ticker names' if len(security_names) == 0 else None
        except Exception as e:
            failed = f'ERROR: Failed to get ticker names as {e}'

        if failed:
            return print(failed)
        else:
            return security_names
            

    def _get_tickers_kospi200(self, tickers=None, col_ticker='Code', col_name='Name'):
        df = fdr.SnapDataReader(self.tickers)
        security_names = df.set_index(col_ticker)[col_name].to_dict()
        return self._check_tickers(security_names, tickers)
        
    
    def _get_tickers_etf(self, tickers=None, col_ticker='Symbol', col_name='Name'):
        """
        한국 ETF 전종목
        """
        df = fdr.StockListing(self.tickers) 
        security_names = df.set_index(col_ticker)[col_name].to_dict()
        return self._check_tickers(security_names, tickers)
        

    def _get_tickers_krx(self, tickers=None, col_ticker='Code', col_name='Name'):
        """
        self.tickers: KOSPI,KOSDAQ
        """
        security_names = dict()
        for x in [x.replace(' ', '') for x in self.tickers.split(',')]:
            df = fdr.StockListing(x)
            security_names.update(df.set_index(col_ticker)[col_name].to_dict())
        return self._check_tickers(security_names, tickers)


    def _get_tickers_fund(self, tickers=None, path=None, col_name='name'):
        """
        self.fickers: file name for tickers
        """
        file = self.tickers # file of tickers
        path = self._check_var(path, self.path)
        fd = FundDownloader(file, path, check_master=False, msg=False)
        if fd.check_master() is not None:
            fd.update_master(save=True)
        security_names = fd.data_tickers[col_name].to_dict()
        return self._check_tickers(security_names, tickers)
        

    def _get_tickers_file(self, tickers=None, path=None, col_ticker='ticker', col_name='name'):
        """
        tickers: file for names of tickers
        """
        file = self.tickers # file of tickers
        path = self._check_var(path, self.path)
        df = pd.read_csv(f'{path}/{file}')
        security_names = df.set_index(col_ticker)[col_name].to_dict()
        return self._check_tickers(security_names, tickers)


    def _get_tickers_yahoo(self, tickers, col_name='longName'):
        if tickers is None:
            print('ERROR: Set tickers for names')
            return dict()
        if isinstance(tickers, str):
            tickers = [tickers]
        yft = yf.Tickers(' '.join(tickers))
        security_names = {x:yft.tickers[x].info[col_name] for x in tickers}
        return self._check_tickers(security_names, tickers)
        

    def _check_tickers(self, security_names, tickers, msg=True):
        """
        get security_names for tickers, checking missing ones as well
        tickers: None, 'selected' or list of tickers
        """
        # get ticker list for 'selected' before reset
        if isinstance(tickers, str) and (tickers.lower() == 'selected'):
            if self.df_prices is None:
                print(f"WARNING: Load price data first for '{tickers}' option")
                tickers = None
            else:
                tickers = self.df_prices.columns.to_list()
                
        if tickers is not None:
            security_names = {k:v for k,v in security_names.items() if k in tickers}
            _ = self._check_security_names(security_names)
        return security_names
        

    def _download_universe(self, *args, **kwargs):
        """
        return df of price history if multiple tickers set, series if a single ticker set
        args, kwargs: for DataManager.download_*
        """
        universe = self.universe
        file = self.tickers
        path = self.path
        return DataManager.download_universe(universe, *args, file=file, path=path, **kwargs)

    @staticmethod
    def download_universe(universe, *args, file=None, path=None, **kwargs):
        """
        return df of price history if multiple tickers set, series if a single ticker set
        universe: krx, yahoo, file, default(fdr)
                  use yahoo for us stocks instead of fdr which shows some inconsitancy of price data
        args, kwargs: for DataManager.download_*
        file, path: for universe fund
        """
        uv = universe.lower() if isinstance(universe, str) else 'default'
        if uv == 'krx':
            # use pykrx as fdr seems ineffective to download all tickers in krx
            func = DataManager.download_krx
        elif uv == 'fund':
            func = lambda *a, **k: DataManager.download_fund(*a, file=file, path=path, **k)
        elif uv == 'fund':
            func = DataManager.download_fund
        elif uv == 'yahoo':
            func = DataManager.download_yahoo
        elif uv == 'file':
            return print("ERROR: Downloading not supported for universe 'file'")
        else:
            func = DataManager.download_fdr
            
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return print(f'ERROR: Failed to download prices as {e}')

    @staticmethod
    def download_fdr(tickers, start_date, end_date, col_price1='Adj Close', col_price2='Close'):
        if isinstance(tickers, str):
            tickers = [tickers]
        df_data = fdr.DataReader(tickers, start_date, end_date)
        cols = df_data.columns
        if col_price1 in cols: # data of signle us stock
            df_data = df_data[col_price1]
        elif col_price2 in cols: # data of signle kr stock (kr stock has no adj close)
            df_data = df_data[col_price2]
        else:# data of multiple tickers
            pass
        return df_data
        
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
    def download_fund(tickers, start_date, end_date,
                      interval=5, pause_duration=1, msg=False,
                      file=None, path='.'):
        """
        file: master file of fund data
        """
        fd = FundDownloader(file, path, check_master=True, msg=False)
        fd.set_tickers()
        ers = fd.download(start_date, end_date, file=None, msg=msg)
        return fd.df_prices

    @staticmethod
    def download_yahoo(tickers, start_date, end_date, col_price='Adj Close'):
        if isinstance(tickers, str):
            tickers = [tickers]
        df_data = yf.download(tickers, start_date, end_date)
        df_data = df_data[col_price]
        try: # df of multiple tickers had index of tz something
            df_data.index = df_data.index.tz_convert(None)
        except:
            pass
        return df_data
        

    def _upload(self, file, path, msg_exception=''):
        try:
            df_prices = pd.read_csv(f'{path}/{file}', parse_dates=[0], index_col=[0])
            DataManager.print_info(df_prices, str_sfx='uploaded.')
            return df_prices
        except Exception as e:
            return print(f'{msg_exception}{e}')

    
    def get_names(self, tickers=None, reset=False, **kwargs):
        """
        tickers: None, 'selected' or list of tickers
        reset: True to get security_names aftre resetting first
        kwargs: additional args for _get_tickers
        """
        security_names = self.security_names
        if reset or (security_names is None):
            security_names = self._get_tickers(tickers, **kwargs)
            if security_names is None:
                return None
            else: # reset security_names
                self.security_names = self._check_security_names(security_names)
        else:
            security_names = self._check_tickers(security_names, tickers, msg=True)
        return SecurityDict(security_names, names=security_names)


    def _check_security_names(self, security_names, update=True):
        """
        check if all tickers in security_names in price data
        """
        df_prices = self.df_prices
        if df_prices is not None:
            tickers = df_prices.columns
            out = [x for x in tickers if x not in security_names.keys()]
            n_out = len(out)
            if n_out > 0:
                print(f'WARNING: Update price data as {n_out} tickers not in universe')
                security_names.update(dict(zip(out, out))) if update else None
        return security_names


    def get_date_range(self, tickers=None, df_prices=None, return_intersection=False):
        df_prices = self._check_var(df_prices, self.df_prices)
        security_names = self.security_names
        if df_prices is None:
            return print('ERROR from get_date_range')
        if tickers is not None:
            try:
                df_prices = df_prices[tickers]
            except KeyError as e:
                print('ERROR: KeyError {e}')
        return get_date_range(df_prices, security_names, 
                              return_intersection=return_intersection)


    def check_days_in_year(self, days_in_year=251, freq='M', n_thr=10):
        df_prices = self.df_prices
        if df_prices is None:
            return print('ERROR from check_days_in_year')
        else:
            return check_days_in_year(df_prices, days_in_year=days_in_year, freq=freq, n_thr=n_thr)


    def _convert_price_to_daily(self, confirm=False, tickers=None):
        df_prices = self.df_prices
        if df_prices is None:
            return print('ERROR from _convert_price_to_daily')
            
        if confirm:
            if tickers is None:
                tickers = df_prices.columns
            # convert tickers to daily
            df = df_prices[tickers].apply(lambda x: convert_to_daily(x.dropna()))
            # update self.df_prices with the converted by unstack, concat and unstack 
            # to makes sure outer join of datetime index 
            self.df_prices = (pd.concat([df_prices.drop(tickers, axis=1).unstack(), df.unstack()])
                              .unstack(0).ffill())
            days_in_year = 365
            print(f'REMINDER: {len(tickers)} equities converted to daily (days in year: {days_in_year})')
            print('Daily metrics in Performance statistics must be meaningless')
            return None
        else:
            return print('WARNING: set confirm to True to convert df_prices to daily')


    def convert_to_daily(self, confirm=False, days_in_year=None):
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        df = self.check_days_in_year(252)
        if df is None:
            return None
        else:
            cols = df.loc[df==days_in_year].index
        return self._convert_price_to_daily(confirm, cols)


    def performance(self, tickers=None, metrics=None, 
                    sort_by=None, start_date=None, end_date=None,
                    fee=None, period_fee=3, percent_fee=True):
        df_prices = self._get_prices(tickers=tickers, start_date=start_date, 
                                      end_date=end_date, n_max=-1)
        if df_prices is None:
            return None

        if fee is not None:
            df_prices = self._get_prices_after_fee(df_prices, fee, 
                                                   period=period_fee, percent=percent_fee)
            
        return self._performance(df_prices, metrics=metrics, sort_by=sort_by)


    def _performance(self, df_prices, metrics=None, sort_by=None):
        df_stat = performance_stats(df_prices, metrics=None, align_period=False)
        df_stat = df_stat.T
        
        if metrics is None:
            metrics = df_stat.columns
        else:
            if isinstance(metrics, str):
                metrics = [metrics]
            metrics = [y for x in metrics for y in df_stat.columns if x in y]
            df_stat = df_stat[metrics]
        
        if self.security_names is not None:
            df_stat = pd.Series(self.security_names).to_frame('name').join(df_stat, how='right')
        
        if sort_by is not None:
            sort_by = [x for x in metrics if sort_by in x]
            if len(sort_by) > 0:
                df_stat = df_stat.sort_values(sort_by[0], ascending=False)
        return df_stat


    def plot(self, tickers, start_date=None, end_date=None, metric='cagr', compare_fees=[True, True],
             base=1000, fee=None, period_fee=3, percent_fee=True,
             length=20, ratio=1,
             figsize=(12,4), ratios=(7, 3)):
        """
        plot total returns of tickers and bar chart of metric
        """
        kw_tkrs = dict(tickers=tickers, start_date=start_date, end_date=end_date)
        kw_fees = dict(fee=fee, period_fee=period_fee, percent_fee=percent_fee)
        # create gridspec
        ax1, ax2 = create_split_axes(figsize=figsize, ratios=ratios, vertical_split=False)
        
        # plot total returns
        kw = dict(base=base, compare_fees=compare_fees[0], length=length, ratio=ratio)
        ax1 = self.plot_return(ax=ax1, **kw_tkrs, **kw_fees, **kw)
        
        # plot bar chart of metric
        kw_tkrs.update({'start_date':mldate(ax1.get_xlim()[0])}) # update start date according to price adjustment
        colors = [ax1.get_lines()[i].get_color() for i, _ in enumerate(tickers)]
        kw = dict(metric=metric, colors=colors, compare_fees=compare_fees[1], length=length, ratio=ratio)
        ax2 = self.plot_bar(ax=ax2, **kw_tkrs, **kw_fees, **kw)
        if ax2 is not None:
            ax2.get_legend().remove()
            ax2.yaxis.tick_right()
        return None


    def plot_return(self, tickers=None, start_date=None, end_date=None,
             base=-1, n_max=-1, 
             fee=None, period_fee=3, percent_fee=True, compare_fees=True,
             ax=None, figsize=(8,5), lw=1, loc='upper left', length=20, ratio=1):
        """
        compare tickers by plot
        tickers: list of tickers to plot
        base: set value for adjusting price so the starting values are identical
        n_max: max num of tickers to plot
        length, ratio: see legend
        """
        df_tickers = self._get_prices(tickers=tickers, start_date=start_date, 
                                      end_date=end_date, base=base, n_max=n_max)
        if df_tickers is None:
            return None

        title = 'Total returns'
        if fee is None:
            compare_fees = False # force to False as no fee provided for comparison
            df_tf = None
        else:
            df_tf = self._get_prices_after_fee(df_tickers, fee, period=period_fee, 
                                               percent=percent_fee)
            if not compare_fees:
                df_tickers = df_tf.copy()
                df_tf = None
                title = 'Total returns after fees'
        
        ax = self._plot_return(df_tickers, df_tf, ax=ax, figsize=figsize, lw=lw, loc=loc,
                        length=length, ratio=ratio)
            
        if base > 0:
            title = f'{title} (adjusted for comparison)'
        ax.set_title(title)       
        return ax
    

    def _plot_return(self, df_prices, df_prices_compare=None,
              ax=None, figsize=(8,5), lw=1, loc='upper left', length=20, ratio=1):
        """
        df_prices: price date of selected tickers
        df_prices_compare: additional data to compare with df_prices such as price after fees
         whose legend assumed same as df_prices
        length, ratio: args for xtick labels 
        """
        security_names = self.security_names
        # rename legend if security_names exists
        clip = lambda x: string_shortener(x, n=length, r=ratio)
        if security_names is not None:
            df_prices.columns = [clip(security_names[x]) for x in df_prices.columns]
            if df_prices_compare is None:
                compare_fees = False
            else:
                compare_fees = True
                df_prices_compare.columns = df_prices.columns 
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        ax = df_prices.plot(ax=ax, lw=lw)
        
        # add plot of df_prices_compare with same color as df_prices
        legend = ax.get_legend_handles_labels()[1] # save legend before adding return after fees
        if compare_fees: 
            colors = {x: ax.get_lines()[i].get_color() for i,x in enumerate(df_prices.columns)}
            _ = df_prices_compare.apply(lambda x: x.plot(c=colors[x.name], ls='--', lw=lw, ax=ax))
        ax.legend(legend, loc=loc) # remove return after fees from legend
        return ax


    def plot_bar(self, metric='cagr', tickers=None, start_date=None, end_date=None, n_max=-1, 
                 fee=None, period_fee=3, percent_fee=True, compare_fees=True,
                 ax=None, figsize=(6,4), length=20, ratio=1,
                 colors=None, alphas=[0.4, 0.8]):
        df_tickers = self._get_prices(tickers=tickers, start_date=start_date, 
                                      end_date=end_date, n_max=n_max)
        if df_tickers is None:
            return None

        label = metric.upper()
        if fee is None:
            df_tf = None
            labels = [label]
        else:
            df_tf = self._get_prices_after_fee(df_tickers, fee, 
                                               period=period_fee, percent=percent_fee)
            if compare_fees:
                labels = [label, f'{label} after fees']
            else:
                df_tickers = df_tf.copy()
                df_tf = None
                labels = [f'{label} after fees']
                
        return self._plot_bar(df_tickers, df_tf, metric=metric, labels=labels, 
                              ax=ax, figsize=figsize, length=length, ratio=ratio,
                              colors=colors, alphas=alphas)
    
    
    def _plot_bar(self, df_prices, df_prices_compare=None, 
                  metric='cagr', labels=['base', 'compare'], 
                  ax=None, figsize=(6,4), length=20, ratio=1,
                  colors=None, alphas=[0.4, 0.8]):
        df_stat = self._performance(df_prices, metrics=None, sort_by=None)
        try:
            df_stat = df_stat[metric]
            df_stat = df_stat.to_frame(labels[0]) # for bar loop
        except KeyError:
            return print(f'ERROR: No metric such as {metric}')

        if df_prices_compare is not None:
            df_stat_f = self._performance(df_prices_compare, metrics=None, sort_by=None)
            df_stat_f = df_stat_f[metric].to_frame(labels[1])
            df_stat = df_stat.join(df_stat_f)
    
        if self.security_names is not None:
            clip = lambda x: string_shortener(x, n=length, r=ratio)
            df_stat.index = [clip(self.security_names[x]) for x in df_stat.index]
    
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cols = df_stat.columns.size
        alphas = [max(alphas)] if cols == 1 else alphas
        x = df_stat.index.to_list()
        _ = [ax.bar(x, df_stat.iloc[:, i], color=colors, alpha=alphas[i]) for i in range(cols)]
        #ax.tick_params(axis='x', labelrotation=45)
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_title(f'{metric.upper()}')
        ax.legend(df_stat.columns.to_list())
        return ax

    
    def _get_prices(self, tickers=None, start_date=None, end_date=None, base=-1, n_max=-1):
        """
        return price data of tickers with date of adjustment for comparison
        n_max: num of random tickers from universe
        base: base value to adjust tickers
        start_date: date to adjust if base > 0
        """
        df_prices = self.df_prices
        if df_prices is None:
            return print('ERROR')
        else:
            df_prices = df_prices.loc[start_date:end_date]
        
        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = self._check_var(tickers, self.df_prices.columns.to_list())
        if len(tickers) > n_max > 0:
            tickers = random.sample(tickers, n_max)
    
        df_tickers = df_prices[tickers]
        dts = df_tickers.apply(lambda x: x.dropna().index.min()) # start date of each tickers
        if base > 0: # adjust price of tickers
            dt_adj = df_tickers.index.min()
            dt_max = dts.max() # min start date where all tickers have data 
            dt_adj = dt_max if dt_adj < dt_max else dt_adj
            df_tickers = df_tickers.apply(lambda x: x / x.loc[dt_adj] * base)
        else:
            dt_adj = dts.min() # drop dates of all None
        return df_tickers.loc[dt_adj:] 


    def _get_prices_after_fee(self, df_prices, sr_fee, period=3, percent=True):
        """
        get df_prices after annual fee
        sr_fee: dict or series of ticker to annual fee. rate
        """
        out = df_prices.columns.difference(sr_fee.index)
        n = out.size
        if n > 0:
            print(f'WARNING: Fee of {n} tickers set to 0 as missing fee data')
            sr = pd.Series(0, index=out)
            sr_fee = pd.concat([sr_fee, sr])
        return CostManager.get_history_with_fee(df_prices, sr_fee, period=period, percent=percent)

    
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



class FundDownloader():
    def __init__(self, file, path='.', file_historical=None, check_master=True, msg=True,
                 col_ticker='ticker', 
                 cols_check=['check1_date', 'check1_price', 'check2_date', 'check2_price'],
                 url = "https://dis.kofia.or.kr/proframeWeb/XMLSERVICES/",
                 headers = {"Content-Type": "application/xml"}):
        # master file of securities with ticker, name, adjusting data, etc
        self.file_master = get_file_latest(file, path)
        # price file name
        file_historical = None if file_historical is None else get_file_latest(file_historical, path)
        self.file_historical = file_historical
        self.path = path
        self.col_ticker = col_ticker
        self.cols_check = cols_check
        # df of master file
        self.data_tickers = self._load_master(msg)
        self.url = url
        self.headers = headers
        self.tickers = None
        self.df_prices = None
        self.failed = [] # tickers failed to download
        # check missing data for conversion
        _ = self.check_master() if check_master else None


    def _load_master(self, msg=True):
        """
        load master file of tickers and its info such as values for conversion from rate to price
        """
        file = self.file_master
        path = self.path
        col_ticker = self.col_ticker
        cols_check = self.cols_check
        f = f'{path}/{file}'
        try:
            data_tickers = pd.read_csv(f, index_col=col_ticker)
        except Exception as e:
            return print(f'ERROR: failed to load {file} as {e}')
            
        cols = pd.Index(cols_check).difference(data_tickers.columns)
        if cols.size > 0:
            data_tickers[cols] = None
        print(f'Data for {len(data_tickers)} funds loaded.') if msg else None
        return data_tickers
        

    def check_master(self):
        """
        check if missing data in data_tickers
        """
        data_tickers = self.data_tickers
        if data_tickers is None:
            return print('ERROR: no ticker data loaded yet')

        # check ticker duplication
        idx = data_tickers.index
        idx = idx[idx.duplicated()].to_list()
        if len(idx) > 0:
            print(f'ERROR: tickers duplicated')
            return idx

        # check conversion data
        cond = data_tickers[self.cols_check].isna().any(axis=1)
        n = cond.sum()
        if n > 0:
            print(f'{n} tickers missing data for conversion from rate to price.')
            return data_tickers.loc[cond].index.to_list()
        else:
            return None


    def update_master(self, save=True, 
                      interval=5, pause_duration=.1, msg=False):
        """
        download data and update ticker data for self.cols_check
        """
        data_tickers = self.data_tickers
        if data_tickers is None:
            return print('ERROR')

        col_ticker = self.col_ticker
        cols_check = self.cols_check
        cols_check_float = ['check1_price', 'check2_price']
        
        data, failed = list(), list()
        tracker = TimeTracker(auto_start=True)
        for x in tqdm(data_tickers.index):
            # download settlements history to get dates for price history
            df = self.download_settlements(x)
            if df is None:
                failed.append(x)
                continue
            else:
                cond = (df['type'] == '결산')
                start = df.loc[cond, 'start'].max()
                end = df.set_index('start').loc[start, 'end']
                
            # download date & price for conversion from rate to price
            df = self.download_price(x, start, end)
            if df is None:
                failed.append(x)
                continue
            else:
                sr = df['price']
                start = sr.index.min()
                end = sr.index.max()
                data.append([x, start, sr[start], end, sr[end]])
            
            tracker.pause(interval=interval, pause_duration=pause_duration, msg=msg)
        tracker.stop()

        if len(data) > 0:
            cols = [col_ticker, *cols_check]
            df = pd.DataFrame().from_records(data, columns=cols).set_index(col_ticker)
            df[cols_check_float] = df[cols_check_float].astype(float)
            data_tickers.update(df, join='left', overwrite=True)
            print('data_tickers updated')
            self.data_tickers = data_tickers
            self.save_master(overwrite=True) if save else None # overwite after updating

        if len(failed) > 0:
            print('WARNING: check output of failed')
            return failed
        

    def set_tickers(self, tickers=None, col_ticker='ticker'):
        """
        set tickers to download prices
        """
        data_tickers = self.data_tickers
        if data_tickers is None:
            return print('ERROR')
        else:
            tickers_all = data_tickers.index.to_list()

        if tickers is None:
            tickers = tickers_all
        else:
            tickers = [tickers] if isinstance(tickers, str) else tickers
            n = pd.Index(tickers).difference(tickers_all).size
            if n > 0:
                print(f'WARNING: {n} funds missing in the data')
                tickers = pd.Index(tickers).intersection(tickers_all).to_list()

        print(f'{len(tickers)} tickers set')
        self.tickers = tickers
        return None


    def download(self, start_date, end_date, freq='monthly',
                 url=None, headers=None,
                 interval=5, pause_duration=.1, msg=False,
                 file=None, path='.'):
        """
        download rate and convert to price using prices in settlement info
        """
        tickers = self.tickers
        if tickers is None:
            return print('ERROR: load tickers first')
            
        url = self._check_var(url, self.url)
        headers = self._check_var(headers, self.headers)
        data_tickers = self.data_tickers
        self.failed = [] # reset for new downloading

        # download rates
        tracker = TimeTracker(auto_start=True)
        df_rates = None
        for x in tqdm(tickers):
            df = self.download_rate(x, start_date, end_date, freq=freq, msg=msg)
            if df is None:
                self.failed.append(x)
            else:
                sr = df['rate'].rename(x)
                df_rates = sr.to_frame() if df_rates is None else pd.concat([df_rates, sr], axis=1)
            tracker.pause(interval=interval, pause_duration=pause_duration, msg=msg)
        tracker.stop()

        if df_rates is None:
            return print('ERROR')
        else:
            n = len(self.failed)
            print(f'WARNING: {n} tickers failed to download') if n>0 else None

        # convert to price
        df_prices = None
        errors, index_errors = list(), list()
        for x in df_rates.columns:
            data = data_tickers.loc[x].to_dict()
            sr_rate = df_rates[x]
            sr_n_err = self._convert_rate(data, sr_rate, percentage=True, msg=msg)
            if sr_n_err is None:
                print(f'ERROR: check data for {x}')
            else:
                sr, err = sr_n_err
                df_prices = sr.to_frame() if df_prices is None else pd.concat([df_prices, sr], axis=1)
                index_errors.append(x)
                errors.append(err)
                
        if len(errors) > 0:
            print(f'Max error of conversions: {max(errors):.2e}')
            self.df_prices = df_prices.sort_index()
            self.save(file, path) if file is not None else None
            return pd.Series(errors, index=index_errors, name='error')
        else:
            return None


    def save(self, file=None, path=None):
        """
        save price data
        """
        file = self._check_var(file, self.file_historical)
        path = self._check_var(path, self.path)
        df_prices = self.df_prices
        if df_prices is None:
            print('ERROR')
        else:
            self._save(df_prices, file, path)
        return None


    def save_master(self, file=None, path=None, overwrite=False):
        """
        save master data
        """
        file = self._check_var(file, self.file_master)
        path = self._check_var(path, self.path)
        data_tickers = self.data_tickers
        if data_tickers is None:
            print('ERROR')
        else:
            self._save(data_tickers, file, path, overwrite=overwrite)
        return None


    def _save(self, df_result, file, path, date=None, date_format='%y%m%d', overwrite=False):
        if date is None:
            date = datetime.now()
        if not isinstance(date, str):
            date = date.strftime(date_format)
        file = get_filename(file, f'_{date}', r"_\d+(?=\.\w+$)")
        _ = save_dataframe(df_result, file, path, overwrite=overwrite,
                           msg_succeed=f'{file} saved',
                           msg_fail=f'ERROR: failed to save as {file} exists')
        return None


    def download_rate(self, ticker, start_date, end_date, freq='m', msg=False,
                       url=None, headers=None, date_format='%Y%m%d',
                       payload="""<?xml version="1.0" encoding="utf-8"?>
                                    <message>
                                      <proframeHeader>
                                        <pfmAppName>FS-COM</pfmAppName>
                                        <pfmSvcName>COMFundUnityPrfRtSO</pfmSvcName>
                                        <pfmFnName>prfRtAllSrch</pfmFnName>
                                      </proframeHeader>
                                      <systemHeader></systemHeader>
                                      <COMFundUnityInfoInputDTO>
                                        <standardCd>{ticker:}</standardCd>
                                        <vSrchTrmFrom>{start_date:}</vSrchTrmFrom>
                                        <vSrchTrmTo>{end_date:}</vSrchTrmTo>
                                        <vSrchStd>{code_freq:}</vSrchStd>
                                      </COMFundUnityInfoInputDTO>
                                    </message>""",
                       tag_iter='prfRtList', 
                       tags={'date':'standardDt', 'rate':'managePrfRate'}
                      ):
        # convert inputs for request
        start_date, end_date = self._convert_dates([start_date, end_date], date_format)
        code_freq = 2 if freq.upper()[0] == 'M' else 1
        kwargs = dict(ticker=ticker, start_date=start_date, end_date=end_date, code_freq=code_freq)
        
        df = self._download_data(payload, tag_iter, tags, **kwargs)
        if df is not None:
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df['rate'] = df['rate'].astype('float')
        return df
        

    def download_settlements(self, ticker, msg=False,
                             url=None, headers=None,
                             payload = """<?xml version="1.0" encoding="utf-8"?>
                                            <message>
                                              <proframeHeader>
                                                <pfmAppName>FS-COM</pfmAppName>
                                                <pfmSvcName>COMFundSettleExSO</pfmSvcName>
                                                <pfmFnName>settleExSrch</pfmFnName>
                                              </proframeHeader>
                                              <systemHeader></systemHeader>
                                                <COMFundUnityInfoInputDTO>
                                                <standardCd>{ticker:}</standardCd>
                                            </COMFundUnityInfoInputDTO>
                                            </message>""",
                             tag_iter='settleExList', 
                             tags={'start':'trustAccSrt', 'end':'trustAccend', 'price':'standardCot', 
                                   'amount':'uOriginalAmt', 'type':'vSettleGbNm'}
                            ):
        df = self._download_data(payload, tag_iter, tags, ticker=ticker)
        if df is not None:
            df['price'] = df['price'].astype('float')
            df['amount'] = df['amount'].astype('int')
            df['start'] = pd.to_datetime(df['start'])
            df['end'] = pd.to_datetime(df['end'])
        return df


    def download_price(self, ticker, start_date, end_date, freq='m', msg=False,
                        url=None, headers=None, date_format='%Y%m%d',
                        payload = """<?xml version="1.0" encoding="utf-8"?>
                                    <message>
                                      <proframeHeader>
                                        <pfmAppName>FS-COM</pfmAppName>
                                        <pfmSvcName>COMFundPriceModSO</pfmSvcName>
                                        <pfmFnName>priceModSrch</pfmFnName>
                                      </proframeHeader>
                                      <systemHeader></systemHeader>
                                        <COMFundUnityInfoInputDTO>
                                        <standardCd>{ticker:}</standardCd>
                                        <companyCd></companyCd>
                                        <vSrchTrmFrom>{start_date:}</vSrchTrmFrom>
                                        <vSrchTrmTo>{end_date:}</vSrchTrmTo>
                                        <vSrchStd>{code_freq:}</vSrchStd>
                                    </COMFundUnityInfoInputDTO>
                                    </message>""",
                        tag_iter='priceModList', 
                        tags={'date':'standardDt', 'price':'standardCot', 'amount':'uOriginalAmt'}
                        ):
        """
        Do not use for long-term history of prices as it seems the prices are not adjusted
        """
        start_date, end_date = self._convert_dates([start_date, end_date], date_format)
        code_freq = 2 if freq.upper()[0] == 'M' else 1
        kwargs = dict(ticker=ticker, start_date=start_date, end_date=end_date, code_freq=code_freq)
        df = self._download_data(payload, tag_iter, tags, **kwargs)
        if df is not None:
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df['price'] = df['price'].astype('float')
            df['amount'] = df['amount'].astype('int')
        return df


    def _download_data(self, payload, tag_iter, tags, 
                       url=None, headers=None, **kwargs_payload):

        url = self._check_var(url, self.url)
        headers = self._check_var(headers, self.headers)
        payload = payload.format(**kwargs_payload)
        xml = FundDownloader.fetch_data(url, headers, payload, msg=False)
        return None if xml is None else FundDownloader.parse_xml(xml, tag_iter, tags)

    
    @staticmethod
    def fetch_data(url, headers, payload, msg=False):
        # Sending the POST request
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            return print(f"An error occurred: {e}" if msg else None)

    
    @staticmethod
    def parse_xml(xml, tag_iter, tags):
        """
        tags: dict of column name to tag in list or list of tags
        """
        if isinstance(tags, dict):
            cols = list(tags.keys())
            tags = list(tags.values())
        elif isinstance(tags, (list, tuple)):
            cols = tags
        else:
            return print('ERROR')
            
        root = ET.fromstring(xml)
        data = list()
        try:
            for itr in root.iter(tag_iter):
                d = [itr.find(x).text for x in tags]
                data.append(d)
        except Exception as e:
            return print(f'ERROR: {e}')
    
        if len(data) == 0:
            return print('ERROR')
        
        return pd.DataFrame().from_records(data, columns=cols)
    

    def _convert_rate(self, data, sr_rate, percentage=True, msg=False):
        """
        calc price from rate of return
        data: series or dict
        """
        data_check = [
            (data['check1_date'], data['check1_price']),
            (data['check2_date'], data['check2_price']),
        ]
        # date check
        for dt, _ in data_check:
            try:
                rate = sr_rate.loc[dt]
            except KeyError as e:
                return print(f'ERROR: KeyError {e}') if msg else None
        
        # convert to price with data_check[0]
        dt, price = data_check[0]
        #dt = pd.to_datetime(dt)
        rate = sr_rate.loc[dt]
        if percentage:
            rate = rate/100
            sr_rate = sr_rate/100
        price_base = price / (rate+1)
        sr_price = (sr_rate + 1) * price_base 
    
        # check price
        dt, price = data_check[1]
        e = sr_price.loc[dt]/price - 1
        print(f'error: {e:.2f}') if msg else None
        
        return (sr_price, e)


    def _check_var(self, var_arg, var_self):
        return var_self if var_arg is None else var_arg


    def _convert_dates(self, dates, date_format='%Y%m%d'):
        """
        convert dates for inputs of request
        """
        if not isinstance(dates, list):
            dates = [dates]
        if isinstance(dates[0], str):
            return pd.to_datetime(dates).strftime(date_format)
        else: # assuming datetime
            return pd.Index(dates).strftime(date_format)

    @staticmethod
    def create(file, path='.'):
        """
        file: master file or instance of DataManager
        """
        if isinstance(file, DataManager):
            path = file.path
            file = file.tickers
        return FundDownloader(file, path)

    
    @staticmethod
    def export_master(file, path='.'):
        """
        get df of fund list (master)
        """
        fd = FundDownloader.create(file, path=path)
        return fd.data_tickers

    
    def export_cost(self, universe, file=None, path='.', update=True,
                    cols_cost=['buy', 'sell', 'fee', 'tax'],
                    col_uv='universe', col_ticker='ticker'):
        """
        universe: universe name. see keys of UNIVERSES
        update: True to update the file with new cost data
        """
        data_tickers = self.data_tickers
        if data_tickers is None:
            return print('ERROR: no ticker data loaded yet')

        cols = [col_ticker, *cols_cost]
        df_cost = (data_tickers.reset_index().loc[:, cols]
                   .fillna(0)
                   .assign(universe=universe)
                   .loc[:, [col_uv, *cols]])
       
        if file: # save cost data 
            file = set_filename(file, 'csv')
            if update: # update existing cost data
                df = CostManager.load_cost(file, path)
                if df is None:
                    return None
                idx = [col_uv, col_ticker]
                df = df.set_index(idx)
                df = df.loc[~df.index.isin(df_cost.set_index(idx).index)]
                df_cost = pd.concat([df.reset_index(), df_cost])
                # save as new file name
                dt = datetime.today().strftime('%y%m%d')
                file = get_filename(file, f'_{dt}', r"_\d+(?=\.\w+$)")
            save_dataframe(df_cost, file, path, 
                           msg_succeed=f'Cost data saved to {file}',
                           index=False)
            return None
        else:
            return df_cost
            


class PortfolioBuilder():
    def __init__(self, df_universe, file=None, path='.', name='portfolio',
                 method_select='all', sort_ascending=False, n_tickers=0, lookback=0, lag=0, tickers=None, 
                 method_weigh='Equally', weights=None, lookback_w=None, lag_w=None, weight_min=0,
                 df_additional=None, security_names=None, 
                 cols_record = {'date':'date', 'tkr':'ticker', 'name':'name', 'rat':'ratio',
                                'trs':'transaction', 'net':'net', 'wgt':'weight', 'dttr':'date*', 'prc':'price'},
                 date_format='%Y-%m-%d', cost=None
                ):
        """
        file: file of transaction history. 
              Do not update the ticker prices with the actual purchase price, 
               as the new df_universe may be adjusted with updated prices after the purchase.
        method_select: 'all', 'selected' for static, 'momentum', 'k-ratio', 'f-ratio' for dynamic
        lookback_w, lag_w: for weigh. reuse those for select if None
        sort_ascending: set to False for momentum & k-ratio, True for PER of f-ratio
        security_names: dict of ticker to name
        cols_record: all the data fields of transaction file
        """
        self.df_universe = df_universe
        # set temp name for self._load_transaction
        file = set_filename(file, default='tmp.csv')
        file = self._retrieve_transaction_file(file, path)
        self.file = file
        self.path = path
        
        self.method_select = method_select
        self.sort_ascending = sort_ascending
        self.n_tickers = n_tickers
        self.lookback = lookback # period for select
        self.lag = lag # days
        self.tickers = tickers # see select
        self.method_weigh = method_weigh
        self.weights = weights
        self.lookback_w = self._check_var(lookback_w, self.lookback) # for weigh
        self.lag_w = self._check_var(lag_w, self.lag)
        self.weight_min = weight_min
        self.df_additional = df_additional
        self.security_names = security_names
        self.name = name # portfolio name
        self.cols_record = cols_record
        self.date_format = date_format # date str format for record & printing
        self.cost = cost # dict of buy/sell commissions, fee and tax. see CostManager
        
        self.selected = None # data for select, weigh and allocate
        self.df_rec = None # record updated with new transaction
        self.liquidation = Liquidation() # for instance of Liquidation
        self.record = self.import_record()
            

    def import_record(self, record=None, msg=True):
        """
        read record from file and update transaction dates
        """
        if record is None:
            record = self._load_transaction(self.file, self.path, print_msg=msg)
        if record is None:
            print('REMINDER: make sure this is 1st transaction as no records provided')
        elif record[self.cols_record['prc']].notna().any():
            print('WARNING: Run update_record first after editing record') if msg else None
        return record


    def select(self, date=None, method=None, sort_ascending=None, 
               n_tickers=None, lookback=None, lag=None, tickers=None,  
               df_additional=None):
        """
        date: transaction date
        method: all, selected, momentum, k-ratio, f-ratio
        tickers: list of tickers in the universe
        df_additional: ex) df_ratio for f-ratio method
        """
        method = self._check_var(method, self.method_select)
        n_tickers = self._check_var(n_tickers, self.n_tickers)
        lookback = self._check_var(lookback, self.lookback)
        lag = self._check_var(lag, self.lag)
        tickers = self._check_var(tickers, self.tickers)
        sort_ascending = self._check_var(sort_ascending, self.sort_ascending)
        df_additional = self._check_var(df_additional, self.df_additional)

        if (n_tickers is not None) and (tickers is not None):
            if n_tickers > len(tickers):
                return print('ERROR: n_tickers greater than length of tickers')
        
        # search transaction date from universe
        kwa = dict(date=date, tickers=tickers)
        date = self._get_data(0, 0, **kwa).index.max()
        df_data = self._get_data(lookback, lag, **kwa)
        dts = get_date_minmax(df_data, self.date_format)
        info_date = f'from {dts[0]} to {dts[1]}'
        
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        if cond(method, 'k-ratio'):
            rank = (df_data.pct_change(1).apply(lambda x: calc_kratio(x.dropna()))
                    .sort_values(ascending=sort_ascending)[:n_tickers])
            method = 'K-ratio'
        elif cond(method, 'f-ratio'):
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
            
            rank = stat.sort_values(ascending=sort_ascending)[:n_tickers]
            if rank.index.difference(df_data.columns).size > 0:
                print('ERROR: check selected tickers if price data given')
            method = 'Financial Ratio'
        elif cond(method, 'momentum'):
            #rank = bt.ffn.calc_total_return(df_data).sort_values(ascending=False)[:n_tickers]
            # no difference with calc_total_return as align_axis=1
            rank = (df_data.apply(lambda x: x.dropna().iloc[-1]/x.dropna().iloc[0]-1)
                    .sort_values(ascending=sort_ascending)[:n_tickers])
            method = 'Total return'
        else: # default all for static
            rank = pd.Series(1, index=df_data.columns)
            n_tickers = rank.count()
            method = 'All' if tickers is None else 'Selected'
                
        tickers = rank.index
        self.selected = {'date': date, 'tickers': tickers, 'rank': rank} 
        print(f'{n_tickers} tickers selected by {method} {info_date}')
        return rank    

    
    def weigh(self, method=None, weights=None, lookback=None, lag=None, weight_min=None, **kwargs):
        """
        method: ERC, InvVol, Equally, Specified
        weights: str, list of str, dict, or None. Used only for 'Specified' method
        weight_min: min weight for every equity. not work with method specified or equal weights
        """
        selected = self.selected
        method = self._check_var(method, self.method_weigh)
        weights = self._check_var(weights, self.weights)
        lookback = self._check_var(lookback, self.lookback_w)
        lag = self._check_var(lag, self.lag_w)
        weight_min = self._check_var(weight_min, self.weight_min)
        
        if selected is None:
            return print('ERROR')
        else:
            date = selected['date']
            tickers = selected['tickers']

        df_data = self._get_data(lookback, lag, date=date, tickers=tickers)
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        if cond(method, 'erc'):
            weights = bt.ffn.calc_erc_weights(df_data.pct_change(1).dropna(), **kwargs)
            method = 'ERC'
        elif cond(method, 'invvol'):
            weights = bt.ffn.calc_inv_vol_weights(df_data.pct_change(1).dropna())
            method = 'Inv.Vol'
        elif cond(method, 'meanvar'):
            weights = bt.ffn.calc_mean_var_weights(df_data.pct_change(1).dropna(), **kwargs)
            method = 'MeanVar'
        elif cond(method, 'specified'):
            w = self.check_weights(weights, df_data, none_weight_is_error=True)
            if w is None:
                return self.liquidation.check_weights(weights)
            weights = {x:0 for x in tickers}
            weights.update(w)
            weights = pd.Series(weights)
            method = 'Specified'
            weight_min = 0
        else: # default equal. no need to set arg weights
            weights = {x:1/len(tickers) for x in tickers}
            weights = pd.Series(weights)
            method = 'Equal weights'
            weight_min = 0

        if weight_min > 0: # drop equities of weight lt weight_min
            w = redistribute_weights(weights, weight_min, n_min=1, none_if_fail=True)
            weights = pd.Series(0, index=weights.index) if w is None else pd.Series(w)

        
        self.selected['weights'] = np.round(weights, 4) # weights is series
        print(f'Weights of tickers determined by {method}.')
        return weights
        

    def allocate(self, capital=10000000, commissions=0, int_nshares=True):
        """
        calc amount of each security for net on the transaction date 
        capital: rebalance tickers without cash flows if set to 0
        commissions: percentage
        int_nshares: True if transaction by number of shares, False for fund
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
        col_rat = cols_record['rat']
        col_net = cols_record['net']
        col_wgt = cols_record['wgt']
        col_name = cols_record['name']
        col_dttr = cols_record['dttr'] # field for actual transaction date
        col_prc = cols_record['prc']
        # column transaction being included in transaction step
        cols_all = [col_name, col_rat, col_net, col_wgt, col_dttr, col_prc]
        col_wgta = 'weight*'
        
        security_names = self.security_names
        selected = self.selected
        if selected is None:
            return print('ERROR')
        
        try:
            date = selected['date']
            weights = selected['weights']
            tickers = selected['tickers']
        except KeyError as e:
            return print('ERROR')

        # sum capital and security value
        record = self.record
        if record is None:
            if capital == 0:
                return print('ERROR: Neither capital nor tickers to rebalance exists')
        else:
            if self.check_new_transaction(date):
                # the arg capital is now cash flows
                print(f'New cash inflows of {capital:,}' ) if capital > 0 else None
                self.df_rec = None # reset df_rec to calc capital
                sr = self.valuate(date, print_msg=False)
                capital += sr['value'] # add porfolio value to capital

        # calc amount of each security by weights and capital
        df_prc = self.df_universe # no _update_universe to work on tickers in the universe
        wi = pd.Series(weights, name=col_wgt).rename_axis(col_tkr) # ideal weights
        sr_net = wi * capital / (1+commissions/100) # weighted security value
        
        if int_nshares: # allocating by considering num of shares from price data
            sr_net = sr_net / df_prc.loc[date, tickers] # stock quantity float
            sr_net = sr_net.rename(col_net).rename_axis(col_tkr) # missing index name in some cases
            # floor-towards-zero for int shares of buy or sell
            sr_net = sr_net.apply(np.fix).astype(int) # Round to nearest integer towards zero
            sr_net = (df_prc.loc[date].to_frame(col_prc).join(sr_net, how='right')
                      .apply(lambda x: x[col_prc] * x[col_net], axis=1) # calc amount from int shares
                      .to_frame(col_net).assign(**{col_date: date}))
        else:
            sr_net = sr_net.to_frame(col_net).assign(date=date)
        # index is multiindex of date and security
        sr_net = sr_net.set_index(col_date, append=True).swaplevel().loc[:, col_net]
        
        # calc error of weights
        wa = self._calc_weight_actual(sr_net, decimals=3)
        mae = (wa.to_frame(col_wgta).join(wi)
               .apply(lambda x: x[col_wgta]/x[col_wgt] - 1, axis=1)
               .abs().mean() * 100)
        print(f'Mean absolute error of weights: {mae:.0f} %')
        
        df_net = (sr_net.to_frame().join(wi)
                  .assign(**{col_rat: 1, col_dttr:date, col_prc:None})) # assigning default values
        df_net = df_net.loc[df_net[col_net]>0] # drop equities of zero net
        if len(df_net) == 0:
            return print('ERROR: No allocation at all')

        # add security names
        if security_names is None:
            df_net[col_name] = None
        else:
            df_net = df_net.join(pd.Series(security_names, name=col_name), on=col_tkr)
            
        return df_net[cols_all]


    def transaction(self, df_net, record=None):
        """
        add new transaction to records
        df_net: output of self.allocate
        record: transaction record given as dataframe
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
        col_name = cols_record['name']
        col_rat = cols_record['rat']
        col_trs = cols_record['trs']
        col_net = cols_record['net']
        col_wgt = cols_record['wgt']
        col_dttr = cols_record['dttr']
        cols_val = [col_trs, col_net] # valid record only if not None nor zero
        cols_idx = [col_date, col_tkr]
        cols_all = [x for x in cols_record.values() if x not in cols_idx]
        cols_int = [col_trs, col_net]
                
        date = df_net.index.get_level_values(0).max()
        record = self._check_var(record, self.record)
        if record is None: # no transation record saved
            # allocation is same as transaction for the 1st time
            df_rec = df_net.assign(**{col_trs: df_net[col_net]})
        else:
            if self.check_new_transaction(date):
                # confine tickers on transaction date
                date_lt = record.index.get_level_values(col_date).max()
                tickers_lt = record.loc[date_lt].index
                tickers_lt = tickers_lt.union(df_net.index.get_level_values(col_tkr))
                # add new to record after removing additional info except for cols_all in record
                df_rec = pd.concat([record[cols_all], df_net])
                # update universe by adding tickers not in the universe but in the past transactions
                df_prc = self._update_universe(df_rec, msg=True)
                df_prc = self.liquidation.set_price(df_prc)
            else: # return None if no new transaction
                return None
            
            # get assets of zero net and concat to df_rec
            lidx = [df_rec.index.get_level_values(i).unique() for i in range(2)]
            midx = pd.MultiIndex.from_product(lidx).difference(df_rec.index)
            df_m = pd.DataFrame({col_rat:1, col_net:0, col_wgt:0}, index=midx)
            # add security names
            if self.security_names is not None: 
                df_m = df_m.join(pd.Series(self.security_names, name=col_name), on=col_tkr)
            df_rec = pd.concat([df_rec, df_m])

            # get num of shares for transaction & net with price history
            # where num of shares is ratio of value to close from latest data
            df_nshares = self._get_nshares(df_rec, df_prc, cols_record, int_nshares=False)
            # get transaction amount for the transaction date
            df_trs = (df_nshares.loc[date, col_net]
                      .sub(df_nshares.groupby(col_tkr)[col_trs].sum())
                      .mul(df_prc.loc[date]).dropna() # get amount by multiplying price
                      .round() # round very small transaction to zero for the cond later
                      .to_frame(col_trs).assign(**{col_date:date, col_dttr:date})
                      .set_index(col_date, append=True).swaplevel())
            # confine tickers on the transaction date
            df_trs = df_trs.loc[df_trs.index.get_level_values(1).isin(tickers_lt)]
            df_rec.update(df_trs)
            # drop new tickers before the date
            df_rec = df_rec.dropna(subset=cols_val) 
            # drop rows with neither transaction nor net 
            cond = (df_rec.transaction == 0) & (df_rec.net == 0)
            df_rec = df_rec.loc[~cond]

        df_rec = df_rec[cols_all]
        df_rec[cols_int] = df_rec[cols_int].astype(int).sort_index(level=[0,1])
        self.df_rec = df_rec # overwrite existing df_rec with new transaction
        # print portfolio value and profit/loss after self.df_rec updated
        _ = self.valuate(print_msg=True)
        return df_rec


    def valuate(self, date=None, print_msg=False, cost_excluded=False):
        """
        calc date, buy/sell prices & portfolio value from self.record or self.df_rec
        """
        date_format = self.date_format
        
        # get latest record
        df_rec = self._check_result(print_msg)
        if df_rec is None:
            return None
        
        # update price data by adding tickers not in the universe if existing
        df_prices = self._update_universe(df_rec, msg=print_msg)
        df_prices = self.liquidation.set_price(df_prices)
    
        # check date by price data
        date = df_prices.loc[:date].index.max() # works even if date None
        date_ft = df_rec.index.get_level_values(0).min()
        if date_ft > date:
            dt = date.strftime(date_format)
            return print(f'ERROR: No transaction before {dt}') if print_msg else None
        else:
            # convert dates to str for series as no more comparison
            date_ft, date = [x.strftime(date_format) for x in (date_ft, date)]
                
        # get record to date
        df_rec = df_rec.loc[:date]
        date_lt = df_rec.index.get_level_values(0).max()
        
        # buy & sell prices to date.
        cost = None if cost_excluded else self.cost
        cf = self._calc_cashflow_history(df_rec, cost).sort_index().iloc[-1].astype(int)
        sell, buy = cf['sell'], cf['buy']
        # calc value
        val = self._calc_value_history(df_rec, date, self.name, msg=False).sort_index().iloc[-1]
        # calc roi & unrealized gain/loss
        ugl = val + sell - buy
        roi = ugl / buy
        if print_msg:
            s = format_rounded_string(roi, ugl)
            print(s, f' ({date})')
             
        data = [date_ft, date, buy, sell, val, ugl, roi]
        index = ['start', 'date', 'buy', 'sell', 'value', 'UGL', 'ROI']
        return pd.Series(data, index=index)


    def transaction_pipeline(self, date=None, capital=10000000, commissions=0, 
                             record=None, save=False, nshares=False, **kw_liq):
        """
        kw_liq: kwargs for Liquidation.prepare
        nshares: set to True if saving last transaction as num of shares for the convenience of trading
        """        
        self.liquidation.prepare(self.record, **kw_liq)
        rank = self.select(date=date)
        if rank is None:
            return None # rank is not None even for static portfolio (method_select='all')
        
        if not self.check_new_transaction(msg=True):
            # calc profit at the last transaction
            dt = self.selected['date'] # selected defined by self.select
            _ = self.valuate(dt, print_msg=True)
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
                # save transaction as num of shares for the convenience of trading
                if nshares:
                    df_prc = self._update_universe(df_rec, msg=False)
                    df_prc = self.liquidation.set_price(df_prc)
                    # DO NOT SAVE transaction & net as int. Set int_nshares to False
                    df_rec = self._convert_to_nshares(df_rec, df_prc, int_nshares=False)
                    # set ratio to None for the new transaction which is a flag for calc of ratio with buy/sell price
                    date_lt = df_rec.index.get_level_values(0).max()
                    col_rat = self.cols_record['rat']
                    df_rec.loc[date_lt, col_rat] = None
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
            return self._calc_value_history(df_rec, name=self.name, msg=True)


    def get_cash_history(self, cost_excluded=False):
        """
        get history of buy and sell prices
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
        else:
            cost = None if cost_excluded else self.cost
            return self._calc_cashflow_history(df_rec, cost)


    def get_profit_history(self, result='ROI', roi_log=False, msg=True, cost_excluded=False):
        """
        get history of profit/loss
        result: 'ROI', 'UGL' or 'all'
        """
        df_rec = self._check_result(msg)
        if df_rec is None:
            return None
            
        sr_val = self._calc_value_history(df_rec, name=self.name, msg=msg)
        if (sr_val is None) or (len(sr_val)==1):
            return print('ERROR: need more data to plot')

        cost = None if cost_excluded else self.cost
        df_cf = self._calc_cashflow_history(df_rec, cost) # buy & sell
        sr_prf = self._calc_profit(sr_val, df_cf, result=result, roi_log=roi_log)
        return sr_prf

    
    def plot(self, start_date=None, end_date=None, 
             figsize=(10,6), legend=True, height_ratios=(3,1), loc='upper left',
             msg_cr=True, roi=True, roi_log=False, cashflow=True, cost_excluded=False):
        """
        plot total, net and profit histories of portfolio
        """
        df_rec = self._check_result(msg_cr)
        if df_rec is None:
            return None
            
        col_net = 'Net'
        col_sell = 'sell'
        sr_val = self._calc_value_history(df_rec, name=self.name, msg=True).rename(col_net)
        if (sr_val is None) or (len(sr_val)==1):
            return print('ERROR: need more data to plot')

        cost = None if cost_excluded else self.cost
        df_cf = self._calc_cashflow_history(df_rec, cost) # cashflow
        res_prf = 'ROI' if roi else 'UGL'
        sr_prf = self._calc_profit(sr_val, df_cf, result=res_prf, roi_log=roi_log) # profit
    
        # total: value + sell price
        sr_ttl = (df_cf.join(sr_val, how='right').ffill()
                  .apply(lambda x: x[col_net] + x[col_sell], axis=1))
    
        func = lambda x: x.loc[start_date:end_date]
        sr_ttl = func(sr_ttl)
        sr_val = func(sr_val)
        sr_prf = func(sr_prf)
        
       # set title
        sr = self.valuate(end_date, print_msg=False, cost_excluded=cost_excluded)
        title = format_rounded_string(sr['ROI'], sr['UGL'])
        title = f"{title} ({sr['date']})"
            
        # plot historical of portfolio value
        ax1, ax2 = self._plot_get_axes(figsize=figsize, height_ratios=height_ratios)
        line_ttl = {'c':'darkgray', 'ls':'--'}
        _ = sr_ttl.plot(ax=ax1, label='Total', title=title, figsize=figsize, **line_ttl)
        _ = sr_val.plot(ax=ax1, c=line_ttl['c'])
        ax1.fill_between(sr_ttl.index, sr_ttl, ax1.get_ylim()[0], facecolor=line_ttl['c'], alpha=0.1)
        ax1.fill_between(sr_val.index, sr_val, ax1.get_ylim()[0], facecolor=line_ttl['c'], alpha=0.2)
    
        # plot vline for transaction dates
        dates_trs = func(df_rec).index.get_level_values(0).unique()
        ax1.vlines(dates_trs, 0, 1, transform=ax1.get_xaxis_transform(), lw=0.5, color='gray')
        ax1.set_ylabel('Value')
        
        # plot profit history
        ax1t = sr_prf.plot(ax=ax1.twinx(), label=res_prf, lw=1, color='orange')
        ax1t.set_ylabel('Return On Investment (%)' if roi else 'Unrealized Gain/Loss')
        # set env for the twins
        _ = set_matplotlib_twins(ax1, ax1t, legend=legend, loc=loc)
        ax1.margins(0)
        ax1t.margins(0)
        
        # plot cashflow
        if cashflow:
            # set slice for record with a single transaction
            start, end = get_date_minmax(sr_val)
            ax2 = self.plot_cashflow(df_rec=df_rec, start_date=start, end_date=end, 
                                     cost_excluded=cost_excluded, ax=ax2)
        return None


    def plot_cashflow(self, df_rec=None, start_date=None, end_date=None, cost_excluded=False,
                      ax=None, figsize=(8,2), alpha=0.4, colors=('r', 'g'),
                      labels=['Buy', 'Sell'], loc='upper left'):
        df_rec = self._check_result() if df_rec is None else df_rec
        if df_rec is None:
            return None

        cost = None if cost_excluded else self.cost
        df_cf = self._calc_cashflow_history(df_rec, cost)
        df_cf = self._plot_cashflow_slice(df_cf, start_date, end_date)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        kw = lambda i: {'label':labels[i], 'color':colors[i]}
        _ = [self._plot_cashflow(ax, df_cf[x], end_date, **kw(i)) for i, x in enumerate(df_cf.columns)]
        ax.legend(loc=loc)
        return ax  
    
    
    def performance(self, metrics=METRICS, sort_by=None):
        """
        calc performance of ideal portfolio excluding slippage
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
        
        sr_val = self._calc_value_history(df_rec, name=self.name)
        if sr_val is None:
            return None
        else:
            return performance_stats(sr_val, metrics=metrics, sort_by=sort_by)

    
    def check_new_transaction(self, date=None, msg=True):
        record = self.record
        if record is None:
            print('WARNING: No record loaded') if msg else None
            return True
        else:
            date_lt = record.index.get_level_values(0).max()

        if date is None:
            selected = self.selected
            if selected is None:
                print('ERROR: run select first') if msg else None
                return False
            else:
                date = selected['date']

        if date_lt >= date:
            print('ERROR: check the date as no new transaction') if msg else None
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
        

    def update_record(self, security_names=None, save=True, update_var=True):
        """
        update and save record: amount-based & ticker names
        save: overwrite record file if True
        """
        # load record and check if any ticker name is None
        # reload record as self.record could been modified for liquidation
        record = self.import_record(msg=False)
        if record is None:
            return None
        else:
            df_rec = record.copy()

        # update col_rat and convert record from num of shares to amount
        df_prc = self._update_universe(df_rec, msg=False)
        df_prc = self.liquidation.set_price(df_prc)
        df_rec = self._update_price_ratio(df_rec, df_prc)
        df_rec = self._convert_to_amount(df_rec, df_prc)
        
        # update ticker name
        df_rec = self._update_ticker_name(df_rec, security_names)
        
        if not df_rec.equals(record): # change exists
            if save:
                self._overwrite_record(df_rec, update_var=update_var)
            else:
                print('REMINDER: Set save to True to save update')                
        #return df_rec
        return None


    def view_record(self, n_latest=0, df_rec=None, nshares=False, value=False,
                    weight_actual=True, msg=True, int_nshares=True):
        """
        get 'n_latest' latest or oldest transaction record 
        nshares: True if num of shares for transaction & net, False if amount of tradings
        value: True if to add value of assets on the next transaction date
        weight_actual: True if to add actual weights of assets
        """
        if df_rec is None:
            df_rec = self._check_result(msg)
        if df_rec is None: # record is None or nshares-based to edit
            return self.record # see _check_result for err msg

        if weight_actual:# add actual weights
            df_rec = self.insert_weight_actual(df_rec)
        
        if nshares or value:
            df_prc = self._update_universe(df_rec, msg=False)
            df_prc = self.liquidation.set_price(df_prc)

        if value: # run before nshares
            df_val = self._calc_periodic_value(df_rec, df_prc)

        # list of cols and types for displaying
        cols_record = self.cols_record
        cols = [cols_record[x] for x in ['rat', 'prc', 'trs', 'net']]
        col_rat, col_prc, col_trs, col_net = cols
        cols = df_rec.columns.drop(col_prc)
        cols_int = [col_trs, col_net]
        
        # show number of shares instead of amount
        if nshares: 
            df_rec = self._convert_to_nshares(df_rec, df_prc, int_nshares=int_nshares)
            # upadte list of cols and types for nshares display
            i = cols.get_loc(col_rat)
            cols = cols.insert(i, col_prc).drop(col_rat)
            cols_int = [*cols_int, col_prc]
            
        df_rec[cols_int] = df_rec[cols_int].astype(int)
        df_rec = df_rec[cols]

        idx = df_rec.index.get_level_values(0).unique().sort_values(ascending=True)
        if n_latest > 0:
            idx = idx[:n_latest]
        elif n_latest < 0:
            idx = idx[n_latest:]
        else:
            pass

        if value:
            return df_rec.loc[idx].join(df_val)
        else:
            return df_rec.loc[idx]


    def check_weights(self, *args, **kwargs):
        return BacktestManager.check_weights(*args, **kwargs)
        

    def check_additional(self, date=None, df_additional=None, 
                         stats=['mean', 'median', 'std'], 
                         plot=False, figsize=(8,5), title='History of Additional data', legend=True,
                         market_label='Market', market_color='grey', market_alpha=0.5, market_line='--'):
        """
        check df_additional from date
        date: a transaction date from record
        """
        df_additional = self._check_var(df_additional, self.df_additional)
        if df_additional is None:
            return print('ERROR: no df_additional available')
    
        df_rec = self._check_result(False)
        if df_rec is None:
            print('No record')
            tickers = None
        else:
            # Retrieve the date and tickers for the transaction closest to the arg date
            df = df_rec.loc[:date]
            if len(df) > 0:
                date = df.index.get_level_values(0).max().strftime(self.date_format)
                tickers = df.loc[date].index.to_list()
            else:
                # date is not None since df is df_rec then
                print(f'No record on {date}')
                tickers = None
        df_all = df_additional.loc[date:]
        if len(df_all) == 0:
            print(f'WARNING: No data after {date}')
            df_all = df_additional
    
        try:
            df_res = None if tickers is None else df_all[tickers] 
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


    def get_names(self, tickers=None):
        security_names = self.security_names
        if security_names is None:
            return print('ERROR: Set security_names first')
        if tickers is None:
            if self.tickers is None:
                tickers = self.df_universe.columns
            else:
                tickers = self.tickers
        s = {k:v for k,v in security_names.items() if k in tickers}
        return SecurityDict(s, names=s)


    def _calc_cashflow_history(self, record, cost=None):
        """
        Returns df of cumulative buy and sell prices at each transaction.
        """
        # add value to record to calc year-fee
        df_rec = self.view_record(0, df_rec=record, nshares=False, value=True, 
                                  weight_actual=False, msg=False, int_nshares=False)
        cm = CostManager(df_rec, self.cols_record, self.date_format)
        return cm.calc_cashflow_history(cost=cost)
    

    def _update_universe(self, df_rec, msg=False):
        """
        create price histories from record to update universe
         the amount of transaction assumed as a stock price itself.
        df_rec: transaction record with amount
        """
        df_prices = self.df_universe
        cols_record = self.cols_record
        col_trs = cols_record['trs']
        col_tkr = cols_record['tkr']
        # tickers not in the universe
        out = df_rec.index.get_level_values(col_tkr).unique().difference(df_prices.columns)
        if out.size > 0:
            idx = pd.IndexSlice
            df_out = df_rec.sort_index().loc[idx[:, out], col_trs].unstack(col_tkr)
            df_new = pd.concat([df_prices, df_out], axis=1)
            df_new[out] = df_new[out].ffill().bfill()
            if msg:
                s = ', '.join(out.to_list())
                print(f'Tickers {s} added to universe')
            return df_new
        else:
            return df_prices

        
    def _calc_periodic_value(self, df_rec, df_prices, date=None, msg=False,
                             col_val='value', col_end='end'):
        """
        get record of transactions with values by asset 
         which is for CostManager._calc_fee_annual
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
        col_net = cols_record['net']
        col_prc = cols_record['prc']
        col_stt = 'start' # defined inside as not included in output df_val
        col_nshares = 'n'
        idx = [col_end, col_tkr]
        
        # calc num of shares for asset value on col_end
        df_val = self._get_nshares(df_rec, df_prices, cols_record, 
                                   int_nshares=False, add_price=True)
        sr_prc = df_val[col_prc] # buy/sell price
        df_val = df_val[col_net].to_frame(col_nshares) 
        # transaction date to calc period for fee calc
        df_val[col_stt] = df_val.index.get_level_values(col_date) 
        # get end date before next transaction
        date = df_prices.index.max() if date is None else date
        df_val[col_end] = (df_val.groupby(col_tkr, group_keys=False)
                          .apply(lambda x: x[col_stt].shift(-1)).fillna(date))
        # calc amout by buy/sell price on col_end
        df_val[col_val] = (df_val.join(sr_prc.rename_axis(idx), on=idx)
                           .apply(lambda x: x[col_nshares] * x[col_prc], axis=1))
        return df_val.loc[:, [col_end, col_val]]
    

    def _convert_to_nshares(self, df_rec, df_universe, int_nshares=False):
        """
        convert transaction & net of df_rec from amount to number of shares 
        """
        df_rec_ns = df_rec.copy()
        cols_record = self.cols_record
        df_nshares = self._get_nshares(df_rec_ns, df_universe, cols_record, 
                                       int_nshares=int_nshares, add_price=True)
        if df_nshares is not None:
            df_rec_ns.update(df_nshares, overwrite=True)
        return df_rec_ns
        

    def _convert_to_amount(self, df_rec, df_universe):
        """
        convert df_rec from num of shares to amount-based
        """
        cols_record = self.cols_record
        col_trs = cols_record['trs']
        col_net = cols_record['net']
        col_prc = cols_record['prc']
        cols_int = [col_trs, col_net]
        # check if record is # of share
        if df_rec[col_prc].isna().any():
            return df_rec # col_prc must be not None to convert to shares
        # calc amounts for transaction & net
        df_rec.loc[:, cols_int] = df_rec[cols_int].mul(df_rec[col_prc], axis=0).astype(int)
        df_rec.loc[:, col_prc] = None # set col_prc to None as flag
        return df_rec


    def _get_trading_price(self, df_rec, df_universe, col_rat, col_prc, col_close=None):
        """
        get buy/sell price from ratio and close price
        col_close: set column name to get close price instead of trading price
        """
        sr_rat = df_rec[col_rat]
        idx = sr_rat.index.names
        sr_close = df_universe.stack().rename_axis(idx)
        if col_close: # return close price
            return sr_close.rename(col_close)
        if sr_rat.isna().any(): # all col_rat must be filled to get trading price
            return print('ERROR: Missing ratio\'s exit')
        return sr_close.div(sr_rat).rename(col_prc).dropna(axis=0)
        

    def _get_nshares(self, df_rec, df_universe, cols_record, 
                     int_nshares=False, add_price=True):
        """
        calc number of shares for amount net & transaction
        """
        cols = [cols_record[x] for x in ['rat','prc','trs','net']]
        col_rat, col_prc, col_trs, col_net = cols
        if df_rec[col_prc].notna().any():
            return print(f'ERROR: {col_prc} is not None')
 
        # get buy/sell price to calc num of shares
        sr_prc = self._get_trading_price(df_rec, df_universe, col_rat, col_prc)
        if sr_prc is None:
            return None # see _get_trading_price for err msg

        df_nshares = df_rec[[col_trs, col_net]].div(sr_prc, axis=0)
        df_nshares = df_nshares.join(sr_prc) if add_price else df_nshares
        return df_nshares.map(np.fix).astype(int) if int_nshares else df_nshares


    def _update_price_ratio(self, df_rec, df_universe):
        """
        calc the ratio of trading price to close price on the trading date
        """
        col_rat, col_prc = [self.cols_record[x] for x in ['rat','prc']]
        col_close = 'close'
        # check if record is nshares mode where price col cannot be None
        if df_rec[col_prc].isna().any(): # no update if missing price
            if df_rec[col_prc].notna().any():
                print(f'ERROR: {col_prc} must be all None or all not None')
            return df_rec
        
        df_rat = df_rec.loc[df_rec[col_rat].isna()]
        if len(df_rat) == 0: # no calc of ratio
            return df_rec

        sr_close = self._get_trading_price(df_rec, df_universe, col_rat, col_prc, 
                                           col_close=col_close)
        df_rat = df_rat.join(sr_close).apply(lambda x: x[col_close] / x[col_prc], axis=1)
        df_rec.update(df_rat.rename(col_rat))
        return df_rec


    def _update_ticker_name(self, df_rec, security_names=None, overwrite=False):
        """
        upadte ticker name in record
        """
        cond = df_rec.name.isna()
        if cond.sum() > 0:
            # check if ticker name provided
            security_names = self._check_var(security_names, self.security_names)
            if security_names is None:
                print('WARNING: Set security_names to update names of None')
            else:
                try:
                    col_name = self.cols_record['name']
                    df_rec.loc[cond, col_name] = df_rec.loc[cond].apply(lambda x: security_names[x.name[1]], axis=1)
                    print('Ticker names of None updated')
                except KeyError as e:
                    print(f'ERROR: KeyError {e} to update names')
        return df_rec
        

    def _calc_weight_actual(self, sr_net, decimals=-1):
        """
        calc actual weights from 'net'
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_wgta = 'weight*'
        sr_w = sr_net.rename(col_wgta)
        sr_w = sr_w / sr_w.groupby(col_date).sum()
        return sr_w.round(decimals) if decimals > 0 else sr_w


    def insert_weight_actual(self, df_rec, decimals=3):
        cols_record = self.cols_record
        col_net = cols_record['net']
        col_wgt = cols_record['wgt']
        col_wgta = 'weight*'
        cols = df_rec.columns
        i = cols.get_loc(col_wgt)
        return (df_rec.join(self._calc_weight_actual(df_rec[col_net], decimals=decimals))
                .loc[:, cols.insert(i+1, col_wgta)])


    def _calc_value_history(self, df_rec, end_date=None, name=None, msg=False):
        """
        calc historical of portfolio value from transaction
        end_date: calc value from 1st transaction of df_rec to end_date.
        name: name of output series
        """
        cols_record = self.cols_record
        col_rat = cols_record['rat']
        col_net = cols_record['net']
        end = datetime.today() if end_date is None else end_date
        sr_ttl = pd.Series()
        dates_trs = df_rec.index.get_level_values(0).unique()
        # update price data with df_rec
        df_universe = self._update_universe(df_rec, msg=msg)
        df_universe = self.liquidation.set_price(df_universe)
        # get number of shares
        sr_nshares = self._get_nshares(df_rec, df_universe, cols_record, int_nshares=False)
        sr_nshares = sr_nshares[col_net]
        df_unit = df_universe.copy()
        df_unit.loc[:,:] = None
        
        # loop for transaction dates in descending order
        for start in dates_trs.sort_values(ascending=False):
            n_tickers = sr_nshares.loc[start]
            df_c = df_universe.loc[start:end, n_tickers.index]
            if len(df_c) == 0: # no price data from transaction date start
                continue
                
            # get ratio of closed to buy/sell price on transaction date 'start'
            rat_i = df_unit.loc[start:end, n_tickers.index]
            rat_i.loc[start] = df_rec.loc[start, col_rat]
            # calc combined security value history from prv transaction (start) to current (end) 
            sr_i = (df_c.div(rat_i, fill_value=1) # trading price
                    .apply(lambda x: x*n_tickers.loc[x.name]).sum(axis=1)) # x.name: index name
            # concat histories        
            sr_ttl = pd.concat([sr_ttl, sr_i])
            end = start - pd.DateOffset(days=1)
    
        if len(sr_ttl) > 0:
            # sort by date
            sr = sr_ttl.astype(int).sort_index()
            return sr if name is None else sr.rename(name)
        else:
            return print('ERROR: no historical')


    def _plot_get_axes(self, figsize=(10,6), height_ratios=(3, 1), sharex=True):
        """
        create axes for self.plot
        """
        return create_split_axes(figsize=figsize, vertical_split=True, 
                                 ratios=height_ratios, share_axis=sharex, space=0)
        

    def _plot_cashflow(self, ax, sr_cashflow_history, date=None, 
                       label='Cash Flows', alpha=0.4, color='g'):
        sr_cashflow_history = sr_cashflow_history.loc[:date]
        df_cf = sr_cashflow_history.rename('y').rename_axis('x1').reset_index()
        df_cf = (df_cf.join(df_cf.x1.shift(-1).rename('x2'))
                 .apply(lambda x: x if date is None else x.fillna(date)))
        df_cf = df_cf[['y', 'x1', 'x2']]
        args_line = [x.to_list() for _, x in df_cf.iterrows()]
        _ = [ax.hlines(*args, color=color, alpha=alpha, label=label if i==0 else None) 
             for i, args in enumerate(args_line)]
        
        df_cf = sr_cashflow_history.rename('y2').rename_axis('x').reset_index()
        df_cf = df_cf.join(df_cf.y2.shift(1).rename('y1')).dropna()
        df_cf = df_cf[['x', 'y1', 'y2']]
        args_line = [x.to_list() for _, x in df_cf.iterrows()]
        _ = [ax.vlines(*args, color=color, alpha=alpha) for args in args_line]
            
    
    def _plot_cashflow_slice(self, df_cf, start_date, end_date):
        """
        slice cashflow history
        """
        for x in (start_date, end_date):
            if (x is not None) and x not in df_cf.index:
                x = datetime.strptime(x, self.date_format) if isinstance(x, str) else x
                df_cf.loc[x] = None
        return df_cf.sort_index().ffill().fillna(0).loc[start_date:end_date]


    def _calc_profit(self, sr_val, df_cashflow_history, result='ROI', 
                     roi_log=False, roi_percent=True,
                     col_val='value', col_sell='sell', col_buy='buy'):
        """
        calc history of roi or unrealized gain/loss
        sr_val: output of _calc_value_history
        result: ROI, UGL or None
        """
        df = (sr_val.to_frame(col_val)
              .join(df_cashflow_history, how='outer')
              .ffill().fillna(0))
        
        result = result.upper()
        if result == 'ROI':
            ratio = lambda x: (x[col_val] + x[col_sell]) / x[col_buy]
            m = 100 if roi_percent else 1
            if roi_log:
                df = df.apply(lambda x: np.log(ratio(x)), axis=1).mul(m)
            else:
                df = df.apply(lambda x: ratio(x) - 1, axis=1).mul(m)
        elif result == 'UGL': # unrealized gain/loss
            df = df.apply(lambda x: x[col_val] + x[col_sell] - x[col_buy], axis=1)
        else:
            pass # return df of col_val, col_sell and col_buy
        return df
        

    def _check_result(self, msg=True):
        if self.df_rec is None:
            if self.record is None:
                return print('ERROR: No transaction record') if msg else None
            else:
                df_res = self.record
        else:
            df_res = self.df_rec

        col_prc = self.cols_record['prc']
        if df_res[col_prc].notna().any(): 
            # seems like record saved as nshares for editing
            print(f'ERROR: Run update_record first after editing record')
        
        # self.df_rec or self.record could be modified if not copied
        return df_res.copy() 
    

    def _load_transaction(self, file, path, print_msg=True):
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
        cols_dts = [col_date, cols_record['dttr']]
        idx = [col_date, col_tkr]
        f = os.path.join(path, file)
        if os.path.exists(f):
            df_rec = pd.read_csv(f, parse_dates=cols_dts, index_col=idx, dtype={col_tkr:str})
        else:
            return None
        # fill ticker less than 6 digits with zeros
        df_rec = (df_rec.reset_index(level=col_tkr)
                        .assign(**{col_tkr: lambda x: x.ticker.str.zfill(6)})
                        .set_index(col_tkr, append=True))
        
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

    
    def _get_date_offset(self, *args, **kwargs):
        return BacktestManager.get_date_offset(*args, **kwargs)


    def _get_data(self, lookback, lag, date=None, tickers=None):
        """
        get data for select or weigh
        """
        df_data = self.df_universe
        if date is not None:
            df_data = df_data.loc[:date]
        if tickers is not None:
            df_data = df_data[tickers]
            
        # setting liquidation
        df_data = self.liquidation.set_price(df_data, select=True)
        # set date range
        date = df_data.index.max()
        dt1 = date - self._get_date_offset(lag, 'weeks')
        dt0 = dt1 - self._get_date_offset(lookback, 'month')
        return df_data.loc[dt0:dt1].dropna(axis=1)


    def _overwrite_record(self, record, update_var=True):
        file, path = self.file, self.path
        record.to_csv(f'{path}/{file}')
        print(f'Transaction file {file} updated')
        if update_var:
            self.record = self.import_record(msg=False)
            print(f'self.record updated')
        return None



class CostManager():
    def __init__(self, df_rec, cols_record, date_format='%Y-%m-%d',
                 file=None, path='.'):
        """
        df_rec: transaction record from PortfolioBuilder
        cols_record: dict of var name to col name of df_rec
        """
        self.df_rec = df_rec
        self.cols_record = cols_record
        self.date_format = date_format
        
    
    def calc_cashflow_history(self, date=None, percent=True,
                              cost=dict(buy=0.00363960, sell=0.00363960, tax=0.18, fee=0)):
        """
        calc net buy & sell prices at each transaction date
        cost: dict of cost items
        """
        df_rec = self.df_rec
        if df_rec is None:
            return None
        else:
            cols_record = self.cols_record
        
        df_cf = CostManager._calc_cashflow_history(df_rec, cols_record)
        df_cf = df_cf.loc[:date]
        if cost is None: # cashflow without cost
            return df_cf
            
        # calc cost
        df_cost = self.calc_cost(date=date, percent=percent, **cost)
        df_cost = df_cost.groupby('date').sum().rename(columns={'buy':'cost_buy'})
        df_cost['cost_sell'] = df_cost['sell'] + df_cost['fee'] + df_cost['tax']
        df_cost = df_cost[['cost_buy', 'cost_sell']].cumsum()
        
        # calc net cashflow
        df_net = df_cf.join(df_cost, how='outer').ffill()
        df_net['buy']  = df_net['buy'] + df_net['cost_buy']
        df_net['sell']  = df_net['sell'] - df_net['cost_sell']
        return df_net[['buy', 'sell']]


    def calc_cost(self, date=None, buy=0, sell=0, tax=0, fee=0, percent=True):
        """
        buy, sell, tax, fee: float, series or dict of ticker to annual fee
        """
        df_rec = self.df_rec
        if df_rec is None:
            return None
        else:    
            cols_record = self.cols_record
            date = datetime.today().strftime(self.date_format) if date is None else date
            df_rec = df_rec.loc[:date]

        m = 0.01 if percent else 1
        sr_buy = CostManager._calc_fee_trading(df_rec, cols_record, m*buy, transaction='buy')
        sr_sell = CostManager._calc_fee_trading(df_rec, cols_record, m*sell, transaction='sell')
        sr_tax = CostManager._calc_fee_trading(df_rec, cols_record, m*tax, transaction='tax')
        sr_fee = CostManager._calc_fee_annual(df_rec, cols_record, m*fee, date)
        return (sr_buy.to_frame().join(sr_sell, how='outer')
                .join(sr_fee, how='outer').join(sr_tax, how='outer')
                .fillna(0))


    def calc_fee_trading(self, commission, date=None, transaction='all', percent=True):
        df_rec = self.df_rec
        if df_rec is None:
            return None
        else:
            cols_record = self.cols_record
            df_rec = df_rec.loc[:date]
        
        commission = commission * (0.01 if percent else 1)
        return CostManager._calc_fee_trading(df_rec, cols_record, commission, transaction=transaction)


    def calc_tax(self, tax, date=None, percent=True):
        return self.calc_fee_trading(tax, date=date, transaction='tax', percent=percent)
        

    def calc_fee_annual(self, fee, date=None, percent=True):
        df_rec = self.df_rec
        if df_rec is None:
            return None
        else:
            cols_record = self.cols_record
            date = datetime.today().strftime(self.date_format) if date is None else date
            df_rec = df_rec.loc[:date]
        
        fee = fee * (0.01 if percent else 1)
        return CostManager._calc_fee_annual(df_rec, cols_record, fee, date)


    @staticmethod
    def _calc_cashflow_history(df_rec, cols_record):
        """
        Returns df of cumulative buy and sell prices at each transaction.
        """
        col_trs = cols_record['trs']
        col_date = cols_record['date']
        
        df = df_rec.loc[df_rec[col_trs]>0]
        df_cf = df[col_trs].groupby(col_date).sum().cumsum().to_frame('buy')
        df = df_rec.loc[df_rec[col_trs]<0]
        df_sell = df[col_trs].groupby(col_date).sum().cumsum().mul(-1)
        return df_cf.join(df_sell.rename('sell'), how='outer').ffill().fillna(0)


    @staticmethod
    def _calc_fee_trading(df_rec, cols_record, sr_fee, transaction='all'):
        """
        calc trading fee
        sr_fee: sell/buy commissions. rate of float or seires or dict
        transaction: 'all', 'buy', 'sell', 'tax'
        """
        col_tkr = cols_record['tkr']
        col_trs = cols_record['trs']
        
        sr_val = df_rec[col_trs]
        # now sr_fee series or float
        sr_fee = pd.Series(sr_fee) if isinstance(sr_fee, dict) else sr_fee
        # rename axis to multiply if series
        sr_fee = sr_fee.rename_axis(col_tkr) if isinstance(sr_fee, pd.Series) else sr_fee
        if transaction == 'buy':
            sr_val = sr_val.loc[sr_val > 0]
        elif transaction in ('sell', 'tax'):
            sr_val = sr_val.loc[sr_val < 0]
        else:
            pass
        # fillna for missing tickers in sr_fee
        return sr_val.abs().mul(sr_fee).fillna(0).rename(transaction)


    @staticmethod
    def _calc_fee_annual(df_rec, cols_record, sr_fee, date, name='fee',
                         col_val='value', col_end='end'):
        """
        calc annual fee
        sr_fee: dict or series of ticker to annual fee. rate
        """
        col_tkr = cols_record['tkr']
        col_date = cols_record['date']
        col_prd, col_rate = 'period', 'rate'
        cols = [col_end, col_val]
        if pd.Index(cols).difference(df_rec.columns).size > 0:
            return print('ERROR: No value in record')
        else:
            df_val = df_rec[cols]
        # value for fee calc is avg of buy and valuated on next transaction date
        df_val.loc[:, 'value'] = (df_val['value'] + df_rec['net'])/2
        sr_p = df_rec[col_end].sub(df_rec.index.get_level_values(col_date)).dt.days.rename(col_prd)
        df_val = df_val.join(sr_p).loc[df_val[col_val] > 0]
        
        if isinstance(sr_fee, dict):
            sr_fee = pd.Series(sr_fee)
        elif isinstance(sr_fee, Number):
            sr_fee = pd.Series(sr_fee, index=df_rec.index.get_level_values(col_tkr).unique())
        sr_fee = sr_fee.rename_axis(col_tkr).rename(name)
    
        df_val[col_rate] = (df_val.join(sr_fee)
                            # year fee converted to fee for period of x[col_prd] days
                           .apply(lambda x: -1 + (1 + x[name]) ** (x[col_prd]/365), axis=1)
                           .fillna(0)) # fillna for missing tickers in sr_fee
        return (df_val.apply(lambda x: x[col_val] * x[col_rate], axis=1) # amount of fee for period
                .rename(name).swaplevel().sort_index())


    @staticmethod
    def get_history_with_fee(df_val, sr_fee, period=3, percent=True):
        """
        df_val: history of value or price such as DataManager.df_prices
        sr_fee: dict or series of ticker to annual fee. rate
        period: add fee every period of months
        """
        # calc fee every period
        def calc_fee(df, fee, period=period, percent=percent):
            fee = fee/100 if percent else fee
            fee = fee.apply(lambda x: -1 + (1+x)**(period/12)) # get equivalent rate of fee for period
            days = check_days_in_year(df, msg=False) # get days fo a year
            days = days.mul(period/12).round().astype(int) # get dats for a period
            return df.apply(lambda x: x.dropna().iloc[::days[x.name]] * fee[x.name]).fillna(0)
        # add fees to value history
        df_fee = df_val.copy()
        df_fee.loc[:,:] = None
        df_fee.update(calc_fee(df_val, sr_fee)) # get fee for every period
        df_fee = df_fee.fillna(0).cumsum() # get history of fees
        return df_val.sub(df_fee)

    
    @staticmethod
    def load_cost(file, path='.', col_uv='universe', col_ticker='ticker'):
        """
        load cost data of strategy, universe & ticker
        """
        try:
            file = get_file_latest(file, path)
            df_cost = pd.read_csv(f'{path}/{file}', dtype={col_ticker:str}, comment='#')
            print(f'Cost data {file} loaded')
        except FileNotFoundError:
            return print('ERROR: Failed to load')
        return df_cost

    
    @staticmethod
    def get_cost(universe, file, path='.', 
                 cols_cost=['buy', 'sell', 'fee', 'tax'],
                 col_uv='universe', col_ticker='ticker'):
        """
        load cost file and get dict of commission for the universe
        """
        df_kw = CostManager.load_cost(file, path)
        if df_kw is None:
            #return print('ERROR: Load cost file first')
            return None

        df_kw = df_kw.loc[df_kw[col_uv] == universe]
        if (len(df_kw) == 1) and df_kw[col_ticker].isna().all(): # same cost for all tickers
            return df_kw[cols_cost].to_dict('records')[0]
        elif len(df_kw) > 1: # cost items are series of ticker to cost
            return df_kw.set_index(col_ticker)[cols_cost].to_dict('series')
        else:
            return print('WARNING: No cost data available')


    @staticmethod
    def check_cost(file, path='.', universe=None,
                   col_uv='universe', col_ticker='ticker'):
        """
        check cost file and cost data for the universe
        """
        # load cost file
        df_cst = CostManager.load_cost(file, path)
        if df_cst is None:
            return None
    
        # check cost data for given univese
        if universe is not None:
            # check if universe in cost data
            df_cst_uv = df_cst.loc[df_cst[col_uv] == universe]
            if len(df_cst_uv) == 0:
                print(f'ERROR: No cost data for {universe} exists')
                return df_cst
            else:
                df_cst = df_cst_uv
    
            # check missing tickers in cost data
            tickers = df_cst[col_ticker]
            if (len(tickers)>1) or tickers.notna().all():
                dm = PortfolioManager.create_universe(universe) # instance of DataManager
                if dm is None:
                    return None
                no_cost = dm.df_prices.columns.difference(tickers.to_list()).to_list()
                n = len(no_cost)
                if n > 0:
                    print(f'ERROR: {n} tickers missing cost data')
                    return no_cost
    
        # check duplication for col_uv & col_ticker as key
        key = [col_uv, col_ticker]
        dupli = df_cst.duplicated(key, keep=False)
        if dupli.any():
            print(f'ERROR: Check duplicates for {key}')
            return df_cst.loc[dupli]



class Liquidation():
    def __init__(self):
        self.securities_to_sell = None
        
    def prepare(self, record, securities_to_sell=None, hold=False):
        """
        convert securities_to_sell to dict of tickers to sell price
        record: PortfolioBuilder.record
        securities_to_sell: str of a ticker; list of tickers; dict of the tickers to its sell price
        hold:     
        - If set to True, all securities in `securities_to_sell` will be held and not liquidated.    
        - If set to False, you can selectively hold certain securities by setting their sell price to zero. 
          In this case, only the specified securities in `securities_to_sell` will be held, while others may still be liquidated.
        """
        # set self.securities_to_sell first to data check
        self.securities_to_sell = securities_to_sell
        if securities_to_sell is None:
            return print('Liquidation set to None')
        
        if record is None:
            return print('ERROR: no record to liquidate')
    
        if isinstance(securities_to_sell, str):
            liq = [securities_to_sell]
        elif isinstance(securities_to_sell, dict):
            liq = [x for x, _ in securities_to_sell.items()]
        elif isinstance(securities_to_sell, list):
            liq = securities_to_sell
        else:
            return print('ERROR: check arg securities_to_sell')
            
        # check if tickers to sell exist in record
        date_lt = record.index.get_level_values(0).max()
        record_lt = record.loc[date_lt]
        if pd.Index(liq).difference(record_lt.index).size > 0:
            return print('ERROR: some tickers not in record')
    
        if not isinstance(securities_to_sell, dict):
            price = 0 if hold else None
            # liq is list regardless of type of securities_to_sell
            securities_to_sell = {x:price for x in liq}
    
        self.securities_to_sell = securities_to_sell
        return print('Liquidation prepared')

    
    def set_price(self, df_prices, select=False):
        """
        update df_prices for liquidation
        df_prices: df_universe for select or transaction
        select: exclude tickers to liquidate in record from universe.
                set to True for self.select
        """
        liq_dict = self.securities_to_sell
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
        liq_dict = self.securities_to_sell
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
            return print('ERROR: securities to liquidate in weights')
        else:
            return None

    
    def recover_record(self, df_rec, cols_rec):
        """
        reset net and transaction of securities in hold
        """
        liq_dict = self.securities_to_sell
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
    def __init__(self, df_prices, name_prfx='Portfolio',
                 metrics=METRICS, initial_capital=1000000, commissions=None, 
                 days_in_year=252, security_names=None):
        # df of tickers (tickers in columns) which of each has its own periods.
        # the periods will be aligned for tickers in a portfolio. see self.build
        if isinstance(df_prices, pd.Series):
            return print('ERROR: df_prices must be Dataframe')
        
        self.df_prices = df_prices
        self.portfolios = SecurityDict(names=security_names) # dict of bt.backtest.Backtest
        self.cv_strategies = SecurityDict(names=security_names) # dict of args of strategies to cross-validate
        self.cv_result = None # dict of cv result
        self.metrics = metrics
        self.name_prfx = name_prfx
        self.n_names = 0 # see self._check_name
        self.initial_capital = initial_capital
        # commissions of all tickers across portfolios
        self.commissions = commissions  # unit %
        self.run_results = None # output of bt.run
        self.days_in_year = days_in_year # only for self._get_algo_freq
        self.security_names = security_names
        self.print_algos_msg = True # control msg print in self._get_algo_*

        # run after set self.df_prices
        DataManager.print_info(df_prices, str_sfx='uploaded.')
        if days_in_year > 0:
            print('running self.util_check_days_in_year to check days in a year')
            _ = self.util_check_days_in_year(df_prices, days_in_year, freq='M', n_thr=1)
        

    def align_period(self, df_prices, axis=0, date_format='%Y-%m-%d',
                     fill_na=True, print_msg1=False, print_msg2=False, n_indent=2):
        if axis is None:
            return df_prices
        else:
            return align_period(df_prices, axis=axis, date_format=date_format, fill_na=fill_na, 
                                print_msg1=print_msg1, print_msg2=print_msg2, n_indent=n_indent)


    def _check_name(self, name=None):
        if name is None:
            self.n_names += 1
            name = f'{self.name_prfx}{self.n_names}'
        return name

    
    def _check_var(self, var_arg, var_self):
        return var_self if var_arg is None else var_arg

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
                self._get_algo_freq(**freq), 
                self._get_algo_select(**select), 
                self._get_algo_weigh(**weigh),
                bt.algos.Rebalance()
            ]
        strategy = bt.Strategy(name, algos)
        if commissions is not None:
            c = lambda q, p: abs(q) * p * commissions
        return bt.Backtest(strategy, dfs, commissions=c, **kwargs)


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
            algo_freq = bt.AlgoStack(bt.algos.RunAfterDays(offset), bt.algos.RunOnce())
        return algo_freq


    def _get_algo_select(self, select='all', n_tickers=0, 
                         lookback=pd.DateOffset(days=0), lag=pd.DateOffset(days=0), 
                         id_scale=1, threshold=None, df_ratio=None, ratio_descending=None):
        """
        select: all, momentum, kratio, randomly, list of tickers
        ratio_descending, df_ratio: args for AlgoSelectFinRatio
        tickers: list of tickers to select in SelectThese algo
        """
        cond = lambda x,y: False if x is None else x.lower() == y.lower()

        if isinstance(select, list): # set for SelectThese
            tickers = select
            select = 'Specified'
              
        if cond(select, 'Momentum'):
            algo_select = SelectMomentum(n=n_tickers, lookback=lookback, lag=lag, threshold=threshold)
            # SelectAll() or similar should be called before SelectMomentum(), 
            # as StatTotalReturn uses values of temp[‘selected’]
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'f-ratio'):
            algo_select = AlgoSelectFinRatio(df_ratio, n_tickers, 
                                             lookback_days=lookback,
                                             sort_descending=ratio_descending)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'k-ratio'):
            algo_select = AlgoSelectKRatio(n=n_tickers, lookback=lookback, lag=lag)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'ID'):
            id_scale = id_scale if id_scale > 1 else 2
            n_pool = round(n_tickers * id_scale)
            algo_select1 = bt.algos.SelectMomentum(n=n_pool, lookback=lookback, lag=lag)
            algo_select2 = AlgoSelectIDiscrete(n=n_tickers, lookback=lookback, lag=lag)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select1, algo_select2)
        elif cond(select, 'IDRank'):
            algo_select = AlgoSelectIDRank(n=n_tickers, lookback=lookback, lag=lag, scale=id_scale)
            algo_select = bt.AlgoStack(bt.algos.SelectAll(), algo_select)
        elif cond(select, 'randomly'):
            algo_after = AlgoRunAfter(lookback=lookback, lag=lag)
            algo_select = bt.algos.SelectRandomly(n=n_tickers)
            algo_select = bt.AlgoStack(algo_after, bt.algos.SelectAll(), algo_select)
        elif cond(select, 'specified'):
            algo_after = AlgoRunAfter(lookback=lookback, lag=lag)
            algo_select = bt.algos.SelectThese(tickers)
            algo_select = bt.AlgoStack(algo_after, bt.algos.SelectAll(), algo_select)
        else:
            algo_after = AlgoRunAfter(lookback=lookback, lag=lag)
            algo_select = bt.AlgoStack(algo_after, bt.algos.SelectAll())
            if not cond(select, 'all'):
                print('WARNING:SelectAll selected') if self.print_algos_msg else None
 
        return algo_select
        

    def _get_algo_weigh(self, weigh='equally', weights=None, lookback=pd.DateOffset(days=0), 
                        lag=pd.DateOffset(days=0), rf=0, bounds=(0.0, 1.0), threshold=0):
        """
        weigh: equally, erc, specified, randomly, invvol, meanvar
        """
        cond = lambda x,y: False if x is None else x.lower() == y.lower()
        
        # reset weigh if weights not given
        if cond(weigh, 'Specified') and (weights is None):
            weigh = 'equally'
        
        if cond(weigh, 'ERC'):
            algo_weigh = bt.algos.WeighERC(lookback=lookback, lag=lag)
            # Use SelectHasData to avoid LedoitWolf ERROR; other weights like InvVol work fine without it.
            lb = sum_dateoffsets(lookback, lag)
            algo_weigh = bt.AlgoStack(bt.algos.SelectHasData(lookback=lb), algo_weigh)
        elif cond(weigh, 'Specified'):
            algo_after = AlgoRunAfter(lookback=lookback, lag=lag)
            algo_weigh = bt.algos.WeighSpecified(**weights)
            algo_weigh = bt.AlgoStack(algo_after, algo_weigh)
        elif cond(weigh, 'Randomly'):
            algo_after = AlgoRunAfter(lookback=lookback, lag=lag)
            algo_weigh = bt.algos.WeighRandomly()
            algo_weigh = bt.AlgoStack(algo_after, algo_weigh)
        elif cond(weigh, 'InvVol'): # risk parity
            algo_weigh = bt.algos.WeighInvVol(lookback=lookback, lag=lag)
        elif cond(weigh, 'MeanVar'): # Markowitz’s mean-variance optimization
            algo_weigh = bt.algos.WeighMeanVar(lookback=lookback, lag=lag, rf=rf, bounds=bounds)
            lb = sum_dateoffsets(lookback, lag)
            algo_weigh = bt.AlgoStack(bt.algos.SelectHasData(lookback=lb), algo_weigh)
        else:
            algo_after = AlgoRunAfter(lookback=lookback, lag=lag)
            algo_weigh = bt.algos.WeighEqually()
            algo_weigh = bt.AlgoStack(algo_after, algo_weigh)
            if not cond(weigh, 'equally'):
                print('WARNING:WeighEqually selected') if self.print_algos_msg else None

        if threshold > 0: # drop equities of weight lt threshold
            algo_redist = RedistributeWeights(threshold, n_min=1, false_if_fail=True)
            algo_weigh = bt.AlgoStack(algo_weigh, algo_redist)
            
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


    def build(self, name=None, build_cv=False, **kwargs):
        """
        prepare for cv or run self.backtest for each run/iteration
        kwargs: all kwargs for self._build
        """
        if build_cv:
            # prepare cv process after which each iteration of cv will run in cross_validate
            # which use self.build by setting build_cv to False
            self.cv_strategies[name] = kwargs
        else:
            self._build(name=name, **kwargs)
        return None
        

    def _build(self, name=None, 
               freq='M', offset=0,
               select='all', n_tickers=0, lookback=0, lag=0,
               lookback_w=None, lag_w=None,
               id_scale=1, threshold=None,
               df_ratio=None, ratio_descending=None, # args for select 'f-ratio'
               weigh='equally', weights=None, rf=0, bounds=(0.0, 1.0), weight_min=0,
               initial_capital=None, commissions=None, algos=None, df_prices=None,
               align_axis=0):
        """
        make backtest of a strategy
        offset: int, for freq
        lookback, lag: for select
        lookback_w, lag_w: for weigh. reuse those for select if None
        commissions: %; same for all tickers
        algos: set List of Algos to build backtest directly
        align_axis: None, 0, 1. option for select list or all
        """
        dfs = self._check_var(df_prices, self.df_prices)
        if isinstance(select, list):
            try:
                dfs = self.align_period(dfs[select], axis=align_axis)
            except KeyError as e:
                return print('ERROR: KeyError {e}')
        elif select.lower() == 'all':
            dfs = self.align_period(dfs, axis=align_axis)
                
        weights = BacktestManager.check_weights(weights, dfs)
        name = self._check_name(name)
        initial_capital = self._check_var(initial_capital, self.initial_capital)
        commissions = self._check_var(commissions, self.commissions)

        # convert lookback & lag to DateOffset just before running self.backtest
        lookback = self._get_date_offset(lookback) # default month
        lag = self._get_date_offset(lag, 'weeks') # default week
        lookback_w = lookback if lookback_w is None else self._get_date_offset(lookback_w) 
        lag_w = lag if lag_w is None else self._get_date_offset(lag_w, 'weeks')
        
        # calc init offset for RunEveryNPeriods in _get_algo_freq
        lags = [(lookback, lag), (lookback_w, lag_w)]
        lags = [len(dfs.loc[ : dfs.index[0] + x + y ]) for x, y in lags]
        # 1st run is after max(lags) for select or weigh, 
        # without which 1st run could be after freq + max(lags) in worst case 
        offset += max(lags) 
        
        # build args for self._get_algo_* from build args
        select = {'select':select, 'n_tickers':n_tickers, 'lookback':lookback, 'lag':lag, 
                  'id_scale':id_scale, 'threshold':threshold,
                  'df_ratio':df_ratio, 'ratio_descending':ratio_descending}
        freq = {'freq':freq, 'offset':offset, 'days_in_year':self.days_in_year} 
        weigh = {'weigh':weigh, 'weights':weights, 'rf':rf, 'bounds':bounds,
                 'lookback':lookback_w, 'lag':lag_w, 'threshold':weight_min}
        
        kwargs = {'select':select, 'freq':freq, 'weigh':weigh, 'algos':algos,
                  'initial_capital':initial_capital, 'commissions':commissions}
        self.portfolios[name] = self.backtest(dfs, name=name, **kwargs)
        
        return None
        

    def buy_n_hold(self, name=None, weigh='specified', weights=None, **kwargs):
        """
        weights: dict of ticker to weight. str if one security portfolio
        kwargs: set initial_capital or commissions
        """
        return self.build(name=name, freq='once', select='all', weigh=weigh,
                          weights=weights, **kwargs)


    def _benchmark(self, dfs, name=None, weights=None, 
                  initial_capital=None, commissions=None,
                  lookback=0, lag=0):
        """
        dfs: historical of tickers
        no cv possible with benchmark
        lookback & lag to set start date same as momentum stragegy with lookback & lag
        """
        print('REMINDER: Make sure all strtategies built to align backtest periods')
        
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
        else: # dfs assumed dataframe
            if name is None:
                name = list(dfs.columns)[0]
                #print(f'WARNING: name set to {name}')

        # check time period of benchmark
        if len(self.portfolios) > 0:
            df = [get_date_minmax(v.data) for k,v in self.portfolios.items()]
            df = pd.DataFrame(df, columns=['start', 'end'])
            start, end = df['start'].min(), df['end'].max()
        else:
            start, end = dfs.index.min(), dfs.index.max()
        dfs = dfs.loc[start:end]

        return self.build(name=name, build_cv=False, freq='once', 
                          select='all', lookback=lookback, lag=lag,
                          weigh='specified', weights=weights, 
                          initial_capital=initial_capital, commissions=commissions, 
                          df_prices=dfs)


    def benchmark(self, name='KODEX200', ticker=None, **kwargs):
        if isinstance(name, (pd.Series, pd.DataFrame)):
            dfs = name
        else:
            df_prices = self.df_prices
            cols = df_prices.columns
            if isinstance(name, str):
                if name in cols:
                    dfs = df_prices[name]
                else: # download
                    start, end = get_date_minmax(df_prices, date_format='%Y-%m-%d')
                    dfs = BacktestManager.util_import_data(name, ticker, start_date=start, end_date=end)
                    if dfs is None:
                        return print('ERROR')
            else: # name is list
                if pd.Index(name).isin(cols).sum() == len(names):
                    dfs = df_prices[name]
                else:
                    return print('ERROR')
        return self._benchmark(dfs, **kwargs)


    def build_batch(self, *kwa_list, reset_portfolios=False, build_cv=False, **kwargs):
        """
        kwa_list: list of k/w args for each backtest
        kwargs: k/w args common for all backtest
        build_cv: set to True to prepare cross-validation.
        reset_portfolios: reset portfolios and cv_strategies
        """
        if reset_portfolios:
            self.portfolios = SecurityDict(names=self.security_names)
            self.cv_strategies = SecurityDict(names=self.security_names)
            
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
        lag: int
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
        ref_val: name kwarg for util_import_data if str, ticker if list/tuple or value
            ex) s&p500, ('LRGF', 'yahoo'), (005930, None)
        kw: kwargs of sns.catplot. 
            ex) {'y':'cagr', 'x':'freq', 'row':'n_tickers', 'col':'lookback', 'hue':'lag'}
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
                if not isinstance(ref_val, Number):
                    start_date, end_date = data['start'].min(), data['end'].max()
                    if isinstance(ref_val, str):
                        name, ticker = ref_val, None
                    elif isinstance(ref_val, (list, tuple)):
                        name, ticker = None, ref_val
                    else:
                        pass # ERROR?
                    kw = dict(metric=kw['y'], name=name, ticker=ticker, 
                              start_date=start_date, end_date=end_date)
                    ref_val = BacktestManager.benchmark_stats(**kw)
                g.refline(y=ref_val)
            return g

    
    @staticmethod
    def print_cv(df_cv):
        n_s = df_cv.index.get_level_values(0).nunique()
        n_i = df_cv.index.get_level_values(1).size
        return print(f'{n_s} param sets with {round(n_i/n_s)} iterations per set')


    @staticmethod
    def benchmark_stats(metric='cagr', name='KODEX200', ticker=None,
                        start_date=None, end_date=None):
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
            start = None if start is pd.NaT else start
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
                         universe='default', date_format='%Y-%m-%d',
                         tickers={'KODEX200':'069500', 'KOSPI':'KS11', 'KOSDAQ':'KQ11', 
                                  'KOSPI200':'KS200',
                                  'S&P500':('SPY', 'yahoo'), 'DOW':('^DJI', 'yahoo'), 
                                  'NASDAQ':('^IXIC', 'yahoo')}):
        """
        import historical of a single ticker by using DataManager.download_universe
        ticker: list/tuple of ticker and universe. str if it's in default universe
        tickers: predefined to use with name w/o ticker
        """
        if (name is None) and (ticker is None):
            return print('ERROR: Set ticker to download')
            
        # set tickers to list format
        cvt = lambda x: [x, universe] if isinstance(x ,str) else x 
        ticker = cvt(ticker) # list or None
        tickers = {k.upper():cvt(v) for k,v in tickers.items()}
    
        # set ticker
        if ticker is None:
            _name = name.upper() # name is not None if ticker is None
            if _name in tickers.keys():
                ticker, universe = tickers[_name]
            else: # show available names for benchmark if no ticker found
                names = ', '.join(tickers.keys())
                return print(f'ERROR: Set ticker or name from {names}')
        else:
            ticker, universe = ticker
            name = ticker if name is None else name
    
        # download
        try:
            sr = DataManager.download_universe(universe, ticker, start_date, end_date)
            sr.name = name
            return sr
        except Exception as e:
            return print(f'ERROR: {e}')

    
    def util_check_days_in_year(self, df=None, days_in_year=None, freq='M', n_thr=10):
        df = self._check_var(df, self.df_prices)
        days_in_year = self._check_var(days_in_year, self.days_in_year)
        return check_days_in_year(df, days_in_year=days_in_year, freq=freq, n_thr=n_thr)
            


class BayesianEstimator():
    def __init__(self, df_prices, days_in_year=252, metrics=METRICS):
        # df of tickers (tickers in columns) which of each might have its own periods.
        # the periods of all tickers will be aligned in every calculation.
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


    def _check_var(self, arg, arg_self):
        return arg_self if arg is None else arg

        
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
        tickers = list(df_prices.columns)
        
        if align_period:
            df_prices = self.align_period(df_prices, axis=0, fill_na=True)
            df_ret = df_prices.pct_change(periods).dropna() * factor_year
            mean_prior = df_ret.mean()
            std_prior = df_ret.std()
            std_low = std_prior / multiplier_std
            std_high = std_prior * multiplier_std
        else:
            ret_list = [df_prices[x].pct_change(periods).dropna() * factor_year for x in tickers]
            mean_prior = [x.mean() for x in ret_list]
            std_prior = [x.std() for x in ret_list]
            std_low = [x / multiplier_std for x in std_prior]
            std_high = [x * multiplier_std for x in std_prior]
            returns = dict()
        
        num_tickers = len(tickers) # flag for comparisson of two tickers
        coords={'ticker': tickers}

        with pm.Model(coords=coords) as model:
            # nu: degree of freedom (normality parameter)
            nu = pm.Exponential('nu_minus_two', 1 / rate_nu, testval=4) + 2.
            mean = pm.Normal('mean', mu=mean_prior, sigma=std_prior, dims='ticker')
            std = pm.Uniform('vol', lower=std_low, upper=std_high, dims='ticker')
            
            if align_period:
                returns = pm.StudentT(f'{freq}_returns', nu=nu, mu=mean, sigma=std, observed=df_ret)
            else:
                func = lambda x: dict(mu=mean[x], sigma=std[x], observed=ret_list[x])
                returns = {i: pm.StudentT(f'{freq}_returns[{x}]', nu=nu, **func(i)) for i, x in enumerate(tickers)}

            fy2 = 1 if debug_annualize else factor_year
            pm.Deterministic(f'{freq}_mean', mean * fy2, dims='ticker')
            pm.Deterministic(f'{freq}_vol', std * (fy2 ** .5), dims='ticker')
            std_sr = std * pt.sqrt(nu / (nu - 2)) if normality_sharpe else std
            sharpe = pm.Deterministic(f'{freq}_sharpe', ((mean-rf) / std_sr) * (fy2 ** .5), dims='ticker')
            
            if num_tickers == 2:
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


    def util_compare_periods(self, df, level=0, name='Price'):
        """
        compare date index between df_ratios and df (ex. price)
        level: index level of date
        """
        df_ratios = self.df_ratios
        date_format = self.date_format
        d0, d1 = get_date_minmax(df_ratios, date_format, level=1)
        print(f'Ratio: {d0} ~ {d1}')
        d0, d1 = get_date_minmax(df, date_format)
        print(f'{name}: {d0} ~ {d1}')
        return None

    
    @staticmethod
    def util_get_ratio(metric, file, path='.', reshape=True):
        fr = FinancialRatios(file, path)
        if reshape:
            df_ratio = fr.util_reshape(metric, stack=False)
        else:
            df_ratio = fr.df_ratios[metric]
        return df_ratio
        


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
    def __init__(self, pf_names):
        """
        pf_names: list of portfolio names
        """
        self.pf_data = PortfolioData()
        self.portfolios = dict()
        self.load(pf_names)

    
    def load(self, pf_names, reload=False):
        """
        loading multiple portfolios (no individual args except for PortfolioData)
        pf_names: list of portfolio names
        """
        if isinstance(pf_names, str):
            pf_names = [pf_names]

        pf_names = self.check_portfolios(pf_names, loading=True)
        if len(pf_names) == 0:
            return None
            
        if reload:
            pf_dict = dict()
        else:
            pf_dict = self.portfolios
            
        for name in pf_names:
            if name in pf_dict.keys():
                print(f'{name} already exists')
            else:
                print(f'{name}:')
                pf_dict[name] = PortfolioManager.create_portfolio(name)
                print()
        self.portfolios = pf_dict
        return None
        

    @staticmethod
    def review(space=None, output=False):
        pfd = PortfolioData()
        return pfd.review(space, output=output)
    
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
        universe_data = pfd.review_universe(name)
        if universe_data is None:
            return None # see review_universe for err msg 
            
        universe = name
        if len(kwargs) > 0:
            universe_data = {**universe_data, **kwargs}
            universe = None
        dm = DataManager(*args, **universe_data)
        dm.portfolio_data = {'universe': {'data': universe_data, 'name':universe}}
        return dm
        
    
    @staticmethod
    def create_portfolio(name, *args, df_universe=None, df_additional=None, **kwargs):
        """
        name: portfolio name
        args, kwargs: additional args & kwargs for PortfolioBuilder
        df_additional: explicit set to exlcude from kwargs
        """
        # removal for comparison with strategy_data
        security_names = kwargs.pop('security_names', None)
        _ = kwargs.pop('name', None) # drop name if it's in kwargs
        
        # get kwarg sets of portfolios
        pfd = PortfolioData()
        kwa_pf = pfd.review_portfolio(name, strategy=False, universe=False)
        if kwa_pf is None:
            return None
            
        name_strategy = kwa_pf['strategy']
        name_universe = kwa_pf['universe']
        
        # update transaction file & path if given in kwargs
        kwa_tran = ['file', 'path']
        kwa_tran = {x: kwargs.pop(x, None) for x in kwa_tran}
        kwa_tran = {k:kwa_pf[k] if v is None else v for k,v in kwa_tran.items()}
        # create portfolio_data with transaction data of the portfolio
        portfolio_data = {**kwa_tran}

        # get universe
        if df_universe is None:
            dm = PortfolioManager.create_universe(name_universe) # instance of DataManager
            if dm is None:
                return None
            security_names = dm.get_names() if security_names is None else security_names # update security_names
            df_universe = dm.df_prices
            portfolio_data.update(dm.portfolio_data)
            
        # get kwargs of PortfolioBuilder
        strategy_data = pfd.review_portfolio(name, strategy=True, universe=False)
        if strategy_data is None:
            return None # see review_portfolio for err msg 
            
        # update strategy_data if input kwargs given
        tmp = [k for k in kwargs.keys() if k in strategy_data.keys()]
        name_strategy = None if len(tmp) > 0 else name_strategy 
        strategy_data = {**strategy_data, **kwargs}
        # set portfolio_data for ref
        portfolio_data['strategy'] = {'data':strategy_data, 'name':name_strategy}

        # create cost if its file given
        cost = strategy_data.pop('cost', None)
        if isinstance(cost, str): # cost is file name
            path = kwa_tran['path'] # cost file in the same dir with transaction file
            cost = PortfolioManager.get_cost(name_universe, cost, path=path)
        
        kws = {**strategy_data, 'name':name, 'security_names':security_names, 
               'cost':cost, **kwa_tran}
        pb = PortfolioBuilder(df_universe, *args, df_additional=df_additional, **kws)
        pb.portfolio_data = portfolio_data
        return pb

    @staticmethod
    def get_cost(name, file, path='.'):
        """
        name: universe name
        """
        return CostManager.get_cost(name, file, path=path)

    @staticmethod
    def check_cost(name, file, path='.'):
        """
        name: universe name. set to None to check cost data regardless of universe
        """
        return CostManager.check_cost(file, path=path, universe=name)


    def check_portfolios(self, pf_names=None, loading=False):
        if loading:
            pf_all = self.pf_data.portfolios.keys()
        else:
            pf_all = self.portfolios.keys()
        if pf_names is None:
            pf_names = pf_all
        else:
            pf_names = [pf_names] if isinstance(pf_names, str) else pf_names
            out = set(pf_names)-set(pf_all)
            if len(out) > 0:
                out = ', '.join(out)
                print(f'ERROR: No portfolio such as {out}')
                p = PortfolioManager.review('portfolio', output=True)
                p =', '.join(p)
                print(f'Portfolios available: {p}')
                pf_names = list()
        return pf_names

    
    def plot(self, pf_names=None, start_date=None, end_date=None, roi=True,
             figsize=(10,5), legend=True, colors = plt.cm.Spectral,
             col_val='value', col_sell='sell', col_buy='buy'):
        """
        start_date: date of beginning of the return plot
        end_date: date to calc return
        roi: ROI plot if True, UGL plot if False
        """
        # check portfolios
        pf_names = self.check_portfolios(pf_names)
        if len(pf_names) == 0:
            return None
        else: # calc portfolio return for title
            df = self._valuate(pf_names, end_date)
            sr = df['Total']
            title = format_rounded_string(sr['ROI'], sr['UGL'])
            title = f"Total {title} ({sr['date']})"
    
        # total value
        line_ttl = {'c':'gray', 'ls':'--'}
        dfs = [v._calc_value_history(v.record, name=k, msg=False) for k,v in self.portfolios.items() if k in pf_names]
        sr_ttl = pd.concat(dfs, axis=1).ffill().sum(axis=1).rename('Total Value').loc[start_date:end_date]
        ax1 = sr_ttl.plot(title=title, figsize=figsize, **line_ttl)
        ax1.set_ylabel('Total Value')
    
        # roi or ugl total
        if roi:
            result_indv = 'ROI'
            ylabel_indv = 'Return On Investment (%)'
            func_ttl = lambda x: ((x[col_val] + x[col_sell]) / x[col_buy] - 1)*100
        else:
            result_indv = 'UGL'
            ylabel_indv = 'Unrealized Gain/Loss'
            func_ttl = lambda x: x[col_val] + x[col_sell] - x[col_buy]
        
        ax2 = ax1.twinx()
        list_df = [self.portfolios[x].get_profit_history(result='all', msg=False) for x in pf_names]
        func = lambda x: pd.concat([df[x] for df in list_df], axis=1).ffill().sum(axis=1).rename(x)
        dfs = [func(x) for x in (col_val, col_sell, col_buy)]
        df_ttl = pd.concat(dfs, axis=1).ffill()
        sr_ttl = df_ttl.apply(func_ttl, axis=1).rename(f'Total {result_indv}')
        sr_ttl = sr_ttl.loc[start_date:end_date]
        _ = sr_ttl.plot(ax=ax2, lw=1)
        
        # roi or ugl individuals
        dfs = [self.portfolios[x].get_profit_history(result=result_indv, msg=False) for x in pf_names]
        dfs = [v.rename(k) for k,v in zip(pf_names, dfs) if v is not None]
        _ = pd.concat(dfs, axis=1).ffill().loc[start_date:end_date].plot(ax=ax2, alpha=0.5, lw=1)
        ax2.set_prop_cycle(color=colors(np.linspace(0,1,len(pf_names))))
        ax2.set_ylabel(ylabel_indv)
        _ = set_matplotlib_twins(ax1, ax2, legend=legend)
        
        # fill total roi/ugl
        ax2.fill_between(sr_ttl.index, sr_ttl, ax2.get_ylim()[0], 
                         facecolor=ax2.get_lines()[0].get_color(), alpha=0.1)
        ax1.margins(0)
        ax2.margins(0)
        
        
    def valuate(self, pf_names=None, date=None):
        pf_names = self.check_portfolios(pf_names)
        if len(pf_names) == 0:
            return None
        else:
            return self._valuate(pf_names, date)


    def _valuate(self, pf_names, date):
        """
        return evaluation summary df the portfolios in pf_names
        pf_names: list of portfolio names
        """
        col_total = 'Total'
        r_start, r_date, r_roi, r_ugl, r_buy = ('start', 'date', 'ROI', 'UGL', 'buy')
        df_res = None
        no_res = []
        for name in pf_names:
            pf = self.portfolios[name]
            sr = pf.valuate(date=date, print_msg=False)
            if sr is None:
                no_res.append(name)
            else:
                df_res = sr.to_frame(name) if df_res is None else df_res.join(sr.rename(name)) 
        # set total
        df_res[col_total] = [df_res.loc[r_start].min(), df_res.loc[r_date].max(), 
                             *df_res.iloc[2:].sum(axis=1).to_list()]
        df_ttl = df_res[col_total]
        df_res.loc[r_roi, col_total] = df_ttl[r_ugl] / df_ttl[r_buy]
        if no_res is not None:
            df_res[no_res] = None
        return df_res



class PortfolioData():
    def __init__(self, portfolios=PORTFOLIOS, strategies=STRATEGIES, universes=UNIVERSES):
        """
        portfolios: dict of portfolios (name to strategy name, universe name & transaction data)
        strategies: dict of strategies (name to strategy data)
        universes: dict of universes (name to universe data)
        """
        self.portfolios = portfolios
        self.strategies = strategies
        self.universes = universes

    def review(self, space=None, output=False):
        """
        get list of names of portfolios, strategies or universes
        space: universe, stragegy, portfolio
        """
        space = 'P' if space is None else space[0].upper()
        if space == 'U':
            args = [self.universes, 'Universe']
        elif space == 'S':
            args = [self.strategies, 'Strategy']
        else: # default portfolio names
            args = [self.portfolios, 'Portfolio']
        return self._print_items(*args, output=output)
        
    def review_portfolio(self, name, strategy=False, universe=False):
        """
        review param values of a portfolio
        name: portfolio name
        """
        if self.portfolios is None:
            return print('ERROR: no portfolios set')
        
        result = self._get_item(name, self.portfolios)
        if result is None:
            return None

        res_s = self.review_strategy(result['strategy'])
        res_u = self.review_universe(result['universe'])
        if strategy:
            if universe:
                result = {'strategy': res_s, 'universe': res_u}
            else:
                result = res_s
        else:
            if universe:
                result = res_u
            else:
                pass
        return result

    def review_strategy(self, name):
        """
        name: universe name
        """
        if self.strategies is None:
            return print('ERROR: no strategies set')
        else:
            return self._get_item(name, self.strategies)

    def review_universe(self, name):
        """
        name: universe name
        """
        if self.universes is None:
            return print('ERROR: no universes set')
        else:
            return self._get_item(name, self.universes)

    def _print_items(self, items, space, output=False):
        if items is None:
            return print(f'ERROR: No {space} set')
        else:
            names = items.keys()
            if output:
                return names
            else:
                return print(f"{space}: {', '.join(names)}")
        
    def _get_item(self, name, data):
        """
        name: universe, strategy, or portfolio name
        """
        try:
            return data[name]
        except KeyError as e:
            names = ', '.join(data.keys())
            return print(f'ERROR: No {e}. select one of {names}')