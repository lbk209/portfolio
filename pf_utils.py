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
import os, time, re, sys, pickle, random, io
import bt
import warnings
import seaborn as sns
import yfinance as yf
import requests
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
import asyncio, nest_asyncio
import math

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
from scipy.stats import gaussian_kde

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

METRICS2 = [
    'total_return', 'cagr', 'calmar', 
    'max_drawdown', 'avg_drawdown', 'avg_drawdown_days', 
    'monthly_vol', 'monthly_sharpe', 'monthly_sortino',
    'yearly_vol', 'yearly_sharpe', 'yearly_sortino'
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


def calculate_hdi(df_density, hdi_prob=0.94):
    """
    Function to calculate the 94% HDI for each Ticker
    """
    hdi_results = dict()  # To store results
    df_density = df_density.sort_index()
    
    for ticker in df_density.columns:
        sr_d = df_density[ticker]
        
        # Normalize density to sum to 1 (probability distribution)
        sr_cd = sr_d.cumsum() / sr_d.sum()
    
        # Find the values where cumulative density is within the desired interval
        lower = sr_cd.loc[sr_cd >= (1 - hdi_prob) / 2].index[0]
        upper = sr_cd.loc[sr_cd <= 1 - (1 - hdi_prob) / 2].index[-1]
        
        hdi_results[ticker] = {
            'x': [lower, upper], 
            'y':[sr_d.loc[lower], sr_d.loc[lower]]
        }
    return hdi_results


def diversification_score(weights, scale=True):
    """
    Compute HHI-based diversification score.
    If scale=True, returns a normalized score in [0, 1], where 1 = equal weights.
    """
    weights = np.array(weights)
    hhi = np.sum(weights**2)
    score = 1 / hhi

    if not scale:
        return score

    n = len(weights)
    scaled = (score - 1) / (n - 1) if n > 1 else 0
    return scaled
    
    
def diversification_ratio(weights, returns, scale=True):
    """
    Compute Diversification Ratio.
    (How much risk reduction am I getting from combining the assets?)
    If scale=True, returns a normalized score in [0, 1], where 1 = max diversification.
    """
    epsilon = 1e-12

    if returns.isnull().values.any():
        #raise ValueError("Returns contain NaN values.")
        pass

    vols = returns.std().values
    if (vols == 0).any():
        raise ValueError("Zero volatility found in returns.")

    corr = returns.corr().values
    cov = corr * np.outer(vols, vols)

    weights = np.asarray(weights)
    port_var = weights @ cov @ weights
    port_var = max(port_var, 0)
    port_vol = np.sqrt(port_var + epsilon)
    wa_vol = np.sum(weights * vols)
    dr = wa_vol / port_vol

    if not scale:
        return dr

    # Equal-weight benchmark
    n = len(weights)
    w_eq = np.ones(n) / n
    port_var_eq = w_eq @ cov @ w_eq
    port_var_eq = max(port_var_eq, 0)
    port_vol_eq = np.sqrt(port_var_eq + epsilon)
    wa_vol_eq = np.sum(w_eq * vols)
    dr_max = wa_vol_eq / port_vol_eq

    denom = dr_max - 1
    if np.abs(denom) < epsilon:
        return 0.0

    scaled = (dr - 1) / denom
    #return np.clip(scaled, 0, 1)
    return scaled



def effective_number_of_risk_bets(weights, returns, scale=False):
    """
    Calculate the Effective Number of Risk Bets (ENRB), optionally scaled to [0, 1].
    (How many independent bets am I making?)
    """
    if len(weights) != returns.shape[1]:
        raise ValueError("Length of weights must match number of assets (columns in returns).")

    vols = returns.std().values
    corr = returns.corr().values
    cov = corr * np.outer(vols, vols)
    
    w = np.asarray(weights).reshape(-1, 1)
    sigma = np.asarray(cov)

    numerator = (w.T @ sigma @ w) ** 2
    denominator = w.T @ sigma @ sigma @ w
    epsilon = 1e-12
    enrb = float(numerator / (denominator + epsilon))

    if scale:
        n_assets = len(weights)
        return (enrb - 1) / (n_assets - 1) if n_assets > 1 else 0.0
    else:
        return enrb
    

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
    df: df of date index and ticker column
    freq: unit to check days_in_year in df and min data size of df as well
    """
    if freq == 'Y':
        grp_format = '%Y'
        factor = 1 
    elif freq == 'W':
        grp_format = '%Y%U' # ex) 202400, 202401, ... 202452
        factor = WEEKS_IN_YEAR
    else: # default month
        grp_format = '%Y%m'
        factor = 12

    # calc mean days for each ticker
    df_days = (df.assign(gb=df.index.strftime(grp_format)).set_index('gb')
                 .apply(lambda x: x.dropna().groupby('gb').count()[1:-1])
                 #.fillna(0) # commented as it distorts mean
                 .mul(factor).mean().round()
                 #.fillna(days_in_year) # for the case no ticker has enough days for the calc
              )

    n = df_days.isna().sum()
    if n > 0: # use input days_in_year for tickers w/o enough days to calc
        df_days = df_days.fillna(days_in_year)
        if msg:
            print(f'WARNING: {n} tickers assumed having {days_in_year} days for a year')

    # check result
    cond = (df_days != days_in_year)
    if (cond.sum() > 0) and msg:
        df = df_days.loc[cond]
        n = len(df)
        if n < n_thr:
            print(f'WARNING: the number of days in a year with followings is {df.mean():.0f} in avg.:')
            _ = [print(f'{k}: {int(v)}') for k,v in df.to_dict().items()]
        else:
            p = n / len(df_days) * 100
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
        t = end_time - start_time
        t, u = (t/60, 'mins') if t > 100 else (t, 'secs')
        print(f"Execution time of {func.__name__}: {t:.0f} {u}")
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
            print(f'WARNING: No {name}*{ext} exists')
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
        print(msg_fail) if msg_fail else None
        return False
    else:
        df.to_csv(f, **kwargs)    
        print(msg_succeed) if msg_succeed else None
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
            print(f'WARNING: No sorting as {e}')

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


def format_price(x, digits=3, min_x=1000, sig_figs=3, int_to_str=True):
    """
    Formats a number by rounding and optionally converting it to a string with comma separators.
    """
    if isinstance(x, Number):
        if abs(x) >= min_x:
            y = int(round(x, -digits))
            y = f'{y:,.0f}' if int_to_str else y
        elif abs(x) > 0:
            y = round(x, -int(math.floor(math.log10(abs(x)))) + (sig_figs - 1))
        else: # x == 0 or None
            y = x
    else:
        y = x
    return y


def print_list(x, print_str='The items in a list: {}'):
    """
    print a string by joining all the elements of an iterable 
    x: list of str
    """
    x = ', '.join(x)
    return print(print_str.format(x))


def add_suffix(s: str, existing_list: list) -> str:
    """
    Returns `s` with the next available numeric suffix if `s` or its numbered versions exist in `existing_list`.

    Args:
        s (str): The base string.
        existing_list (list): List of existing strings.

    Returns:
        str: `s` or `sN` where N is the next available integer suffix.
    """
    # Match s or s1, s2, etc.
    pattern = re.compile(rf"^{re.escape(s)}(\d*)$")
    max_suffix = 0

    for item in existing_list:
        match = pattern.match(item)
        if match:
            suffix = match.group(1)
            if suffix == "":
                max_suffix = max(max_suffix, 1)
            else:
                max_suffix = max(max_suffix, int(suffix) + 1)

    return f"{s}{max_suffix}" if max_suffix > 0 else s


class SuppressPrint:
    def __init__(self, suppress=True):
        self.suppress = suppress

    def __enter__(self):
        if self.suppress:
            self._stdout = sys.stdout
            self._buffer = io.StringIO()
            sys.stdout = self._buffer

    def __exit__(self, exc_type, exc_value, traceback):
        if self.suppress:
            sys.stdout = self._stdout  # Restore original stdout
    

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
                 to_daily=False, days_in_year=12, **kwargs):
        """
        universe: kospi200, etf, krx, fund, etc. only for getting tickers. see pf_data for more
        file: price history. set as kw for pf_data
        tickers: ticker for getting pool of tickers. can be a file name for tickers as well.
        to_daily: set to True if convert montly price to daily
        days_in_year: only for convert_to_daily
        kwargs: additional kwargs for each universe
        """
        file = set_filename(file, 'csv') 
        self.file_historical = get_file_latest(file, path) # latest file
        self.path = path
        self.universe = universe
        self.tickers = tickers
        self.security_names = None 
        self.df_prices = None
        self.to_daily = to_daily
        self.days_in_year = days_in_year
        self.kwargs_universe = kwargs
        self.visualization = None
        # update self.df_prices
        self.upload(self.file_historical, get_names=True, convert_to_daily=to_daily)

    
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
        self.convert_to_daily(True, self.days_in_year) if convert_to_daily and self.to_daily else None
        return print('Price data loaded')
        

    @print_runtime
    def download(self, start_date=None, end_date=None, n_years=3, tickers=None,
                 save=True, overwrite=False, close_today=False, append=False,
                 date_format='%Y-%m-%d', **kwargs_download):
        """
        download df_prices by using FinanceDataReader
        n_years: int
        tickers: None for all in new universe, 'selected' for all in df_prices, 
                 or list of tickers in new universe
        append: set to True to just donwload new tickers to update existing price data
        kwargs_download: args for krx. ex) interval=5, pause_duration=1, msg=False
        """
        df_prices = self.df_prices
        start_date, end_date = DataManager.get_start_end_dates(start_date, end_date, 
                                                               close_today, n_years, date_format)
        print('Downloading ...')
        security_names = self._get_tickers(tickers)
        if security_names is None:
            return None # see _get_tickers for error msg
        else:
            tickers = list(security_names.keys())

        if append and (df_prices is not None):
            tickers = pd.Index(tickers).difference(df_prices.columns)
            if tickers.size > 0:
                print(f'Update existing data with {tickers.size} tickers')
                tickers = tickers.to_list()
            else:
                return print('ERROR: No new tickers to download. Set append=False to download all')
        else:
            append = False # set to False to avoid later appending 
                   
        try:
            df_prices_new = self._download_universe(tickers, start_date, end_date, **kwargs_download)
            if not close_today: # market today not closed yet
                df_prices_new = df_prices_new.loc[:datetime.today() - timedelta(days=1)]
            print('... done')

            if append:
                dt = df_prices.index.max() # follow the last date of existing data
                df_prices = pd.concat([df_prices, df_prices_new.loc[:dt]], axis=1)
            else:
                df_prices = df_prices_new
            
            DataManager.print_info(df_prices, str_sfx='downloaded.')
        except Exception as e:
            return print(f'ERROR: {e}')
            
        self.df_prices = df_prices
        self.security_names = security_names
        if save:
            if not self.save(overwrite=overwrite):
                return None
        # convert to daily after saving original monthly
        self.convert_to_daily(True, self.days_in_year) if self.to_daily else None
        return print('df_prices updated')

    
    def save(self, file=None, path=None, date=None, overwrite=False, date_format='%y%m%d'):
        file = self._check_var(file, self.file_historical)
        path = self._check_var(path, self.path)
        df_prices = self.df_prices
        if (file is None) or (df_prices is None):
            return print('ERROR: check file or df_prices')

        if date is None:
            #date = datetime.now()
            date = df_prices.index.max()
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


    def _get_tickers(self, tickers=None):
        """
        tickers: None for all in new universe, 'selected' for all in df_prices, 
                 or list of tickers in new universe
        """
        kwargs = self.kwargs_universe
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
            #print(f'WARNING: Set tickers when downloading universe {uv}')
            #func = self._get_tickers_kospi200
            def func(tickers=tickers):
                if tickers is None or (isinstance(tickers, str) and tickers == 'selected'):
                    if self.df_prices is None:
                        return None
                    tickers = self.df_prices.columns
                elif isinstance(tickers, str):
                    tickers = [tickers]
                return {x:x for x in tickers}
        
        try:
            security_names = func(**kwargs)
            security_names = self._check_tickers(security_names, tickers)
            failed = 'ERROR: Failed to get ticker names' if len(security_names) == 0 else None
        except Exception as e:
            failed = f'ERROR: Failed to get ticker names as {e}'

        if failed:
            return print(failed)
        else:
            return security_names
            

    def _get_tickers_kospi200(self, col_ticker='Code', col_name='Name', **kw):
        """
        kw: dummy for other _get_tickers_*
        """
        df = fdr.SnapDataReader(self.tickers)
        return df.set_index(col_ticker)[col_name].to_dict()
        
    
    def _get_tickers_etf(self, col_ticker='Symbol', col_name='Name', **kw):
        """
        한국 ETF 전종목
        """
        df = fdr.StockListing(self.tickers) 
        return df.set_index(col_ticker)[col_name].to_dict()
        

    def _get_tickers_krx(self, **kw):
        """
        self.tickers: KOSPI,KOSDAQ
        """
        security_names = dict()
        for x in [x.replace(' ', '') for x in self.tickers.split(',')]:
            security_names.update(DataManager.get_tickers_krx(x))
        return security_names


    @staticmethod
    def get_tickers_krx(x, col_ticker='Code', col_name='Name'):
        """
        self.tickers: KOSPI,KOSDAQ
        """
        df = fdr.StockListing(x)
        return df.set_index(col_ticker)[col_name].to_dict()


    def _get_tickers_fund(self, path=None, col_name='name', **kw):
        """
        self.fickers: file name for tickers
        """
        file = self.tickers # file of tickers
        path = self._check_var(path, self.path)
        # get kwa to init FundDownloader
        ks = ['check_master', 'msg', 'freq', 'batch_size']
        kwargs = {k:v for k,v in self.kwargs_universe.items() if k in ks}
        fd = FundDownloader(file, path, **kwargs)
        if not fd.check_master(): # True if no duplicated
            raise Exception('See check_master')
        return fd.data_tickers[col_name].to_dict()
        

    def _get_tickers_file(self, path=None, col_ticker='ticker', col_name='name', **kw):
        """
        tickers: file for names of tickers
        """
        file = self.tickers # file of tickers
        path = self._check_var(path, self.path)
        df = pd.read_csv(f'{path}/{file}')
        return df.set_index(col_ticker)[col_name].to_dict()


    def _get_tickers_yahoo(self, col_name='longName', **kw):
        """
        tickers: subset of self.tickers or None
        """
        tkrs_uv = self.tickers # universe as tickers
        if tkrs_uv is None:
            print('ERROR: Set tickers for names')
            return dict()
        yft = yf.Tickers(' '.join(tkrs_uv))
        return {x:yft.tickers[x].info[col_name] for x in tkrs_uv}
        

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
            tickers = [tickers] if isinstance(tickers, str) else tickers
            security_names = {k:v for k,v in security_names.items() if k in tickers}
            _ = self._check_security_names(security_names)
        return security_names
        

    def _download_universe(self, *args, **kwargs):
        """
        return df of price history if multiple tickers set, series if a single ticker set
        args, kwargs: for DataManager.download_*
        """
        universe = self.universe
        kwargs = {**self.kwargs_universe, **kwargs, 
                  'file':self.tickers, 'path':self.path} # see download_fund
        return DataManager.download_universe(universe, *args, **kwargs)

    @staticmethod
    def download_universe(universe, *args, **kwargs):
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
    def download_fdr(tickers, start_date, end_date, col_price1='Adj Close', col_price2='Close', **kw):
        """
        kw: dummy
        """
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
        # set ticker name for downloading a single ticker
        if isinstance(df_data, pd.Series):
            if len(tickers) == 1:
                df_data = df_data.to_frame(tickers[0])
            else:
                return print('ERROR: Check failed tickers')
        return df_data.rename_axis('date')
        
    @staticmethod
    def download_krx(tickers, start_date, end_date,
                      interval=5, pause_duration=1, msg=False, **kw):
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
                      file=None, path='.', freq='daily', batch_size=100, **kw):
        """
        file: master file of fund data
        """
        fd = FundDownloader(file, path=path, check_master=True, 
                            freq=freq, batch_size=batch_size, msg=False)
        fd.set_tickers(tickers)
        _ = fd.download(start_date, end_date, save=False, msg=msg,
                        interval=interval, pause_duration=pause_duration)
        return fd.df_prices

    @staticmethod
    def download_yahoo(tickers, start_date, end_date, col_price='Close', **kw):
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

    
    def get_names(self, tickers=None, reset=False):
        """
        tickers: None, 'selected' or list of tickers
        reset: True to get security_names aftre resetting first
        """
        security_names = self.security_names
        if reset or (security_names is None):
            security_names = self._get_tickers(tickers)
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
            out = [x for x in security_names.keys() if x not in tickers]
            n_out = len(out)
            if n_out > 0:
                print(f'WARNING: Update price data as {n_out} tickers missing in universe')
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
                              .unstack(0).sort_index().ffill()) # sort date index before ffill
            print(f'REMINDER: {len(tickers)} equities converted to daily')
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


    def plot(self, tickers, reload=True, **kwargs):
        visualization = self.get_visualizer(reload)
        return visualization.plot(tickers, **kwargs)
        

    def performance(self, tickers=None, reload=True, **kwargs):
        visualization = self.get_visualizer(reload)
        return visualization.performance(tickers=tickers, **kwargs)
        

    def get_visualizer(self, reload=False):
        if reload or (self.visualization is None):
            self.visualization = DataVisualizer(self.df_prices, self.security_names)
        return self.visualization
            


class DataVisualizer():
    """
    helper class for DataManager or DataMultiverse
    """
    def __init__(self, df_prices, security_names=None):
        self.df_prices = df_prices
        self.security_names = security_names
             
    def performance(self, tickers=None, metrics=None, 
                    sort_by=None, start_date=None, end_date=None,
                    cost=None, period_fee=3, percent_fee=True, transpose=True):
        df_prices = self._get_prices(tickers=tickers, start_date=start_date, 
                                      end_date=end_date, n_max=-1)
        if df_prices is None:
            return None

        if cost is not None:
            df_p = self._get_prices_after_fee(df_prices, cost=cost, 
                                              period=period_fee, percent=percent_fee)
            df_prices = df_prices if df_p is None else df_p

        if transpose:
            return self._performance(df_prices, metrics=metrics, sort_by=sort_by)
        else:
            return performance_stats(df_prices, metrics=metrics, sort_by=sort_by, align_period=False)

    
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
             base=1000, cost=None, period_fee=3, percent_fee=True,
             length=20, ratio=1, lw=0.5,
             figsize=(12,4), ratios=(7, 3)):
        """
        plot total returns of tickers and bar chart of metric
        """
        kw_tkrs = dict(tickers=tickers, start_date=start_date, end_date=end_date)
        kw_fees = dict(cost=cost, period_fee=period_fee, percent_fee=percent_fee)
        # create gridspec
        ax1, ax2 = create_split_axes(figsize=figsize, ratios=ratios, vertical_split=False)
        
        # plot total returns
        kw = dict(base=base, compare_fees=compare_fees[0], length=length, ratio=ratio, lw=lw)
        ax1 = self.plot_return(ax=ax1, **kw_tkrs, **kw_fees, **kw)
        if ax1 is None:
            return
        
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
             base=-1, n_max=-1, cost=None, period_fee=3, percent_fee=True, compare_fees=True,
             **kwargs):
        """
        compare tickers by plot
        tickers: list of tickers to plot
        base: set value for adjusting price so the starting values are identical
        n_max: max num of tickers to plot
        """
        df_tickers = self._get_prices(tickers=tickers, start_date=start_date, 
                                      end_date=end_date, base=base, n_max=n_max)
        if df_tickers is None:
            return None

        title = 'Total returns'
        if cost is None:
            df_tf = None
        else: # df_tf is None if fee of dict or series missing any ticker
            df_tf = self._get_prices_after_fee(df_tickers, cost=cost, period=period_fee, 
                                               percent=percent_fee)
        if df_tf is None:
            compare_fees = False # force to False as no fee provided for comparison
        else:
            if not compare_fees:
                df_tickers = df_tf.copy()
                df_tf = None
                title = 'Total returns after fees'
        
        ax = self._plot_return(df_tickers, df_tf, security_names=self.security_names, **kwargs)
            
        if base > 0:
            title = f'{title} (adjusted for comparison)'
            ax.axhline(base, c='grey', lw=0.5)
        ax.set_title(title)       
        return ax
    

    def _plot_return(self, df_prices, df_prices_compare=None, security_names=None,
              ax=None, figsize=(8,5), lw=1, loc='upper left', length=20, ratio=1):
        """
        df_prices: price date of selected tickers
        df_prices_compare: additional data to compare with df_prices such as price after fees
         whose legend assumed same as df_prices
        length, ratio: args for xtick labels 
        """
        if security_names is not None:
            # rename legend if security_names exists
            clip = lambda x: string_shortener(x, n=length, r=ratio)
            # add number to distinguish similar names such as those of same class fund
            df_prices.columns = [f'{i+1}.{clip(security_names[x])}' for i,x in enumerate(df_prices.columns)]
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
        legend = [re.sub(r'^\d+\.\s*', '', x) for x in legend] # remove numbers added
        if compare_fees: 
            colors = {x: ax.get_lines()[i].get_color() for i,x in enumerate(df_prices.columns)}
            _ = df_prices_compare.apply(lambda x: x.plot(c=colors[x.name], ls='--', lw=lw, ax=ax))
        ax.legend(legend, loc=loc) # remove return after fees from legend
        return ax


    def plot_bar(self, tickers=None, start_date=None, end_date=None, metric='cagr', n_max=-1, 
                 cost=None, period_fee=3, percent_fee=True, compare_fees=True,
                 **kwargs):
        df_tickers = self._get_prices(tickers=tickers, start_date=start_date, 
                                      end_date=end_date, n_max=n_max)
        if df_tickers is None:
            return None

        if cost is None:
            df_tf = None
        else:
            df_tf = self._get_prices_after_fee(df_tickers, cost=cost, 
                                               period=period_fee, percent=percent_fee)
        label = metric.upper()
        if df_tf is None:
            labels = [label]
        else:
            if compare_fees:
                labels = [label, f'{label} after fees']
            else:
                df_tickers = df_tf.copy()
                df_tf = None
                labels = [f'{label} after fees']
                
        df_stat = self._performance(df_tickers, metrics=None, sort_by=None)
        try:
            df_stat = df_stat[metric]
            df_stat = df_stat.to_frame(labels[0]) # for bar loop
        except KeyError:
            return print(f'ERROR: No metric such as {metric}')

        if df_tf is not None:
            df_stat_f = self._performance(df_tf, metrics=None, sort_by=None)
            df_stat_f = df_stat_f[metric].to_frame(labels[1])
            df_stat = df_stat.join(df_stat_f)
            
        return self._plot_bar(df_stat, security_names=self.security_names, 
                                      metric=metric, **kwargs)


    def _plot_bar(self, df_stat, security_names=None, metric='cagr', 
                   ax=None, figsize=(6,4), length=20, ratio=1,
                   colors=None, alphas=[0.4, 0.8]):
        if security_names is not None:
            clip = lambda x: string_shortener(x, n=length, r=ratio)
            df_stat.index = [f'{i+1}.{clip(security_names[x])}' for i,x in enumerate(df_stat.index)]
    
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        cols = df_stat.columns.size
        alphas = [max(alphas)] if cols == 1 else alphas
        x = df_stat.index.to_list()
        _ = [ax.bar(x, df_stat.iloc[:, i], color=colors, alpha=alphas[i]) for i in range(cols)]
        #ax.tick_params(axis='x', labelrotation=45)
        plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
        ax.set_title(f'{metric.upper()}')
        legend = df_stat.columns.to_list()
        legend = [re.sub(r'^\d+\.\s*', '', x) for x in legend] # remove numbers added
        ax.legend(legend)
        ax.axhline(0, c='grey', lw=0.5)
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
        tickers = tickers or df_prices.columns.to_list()
        if len(tickers) > n_max > 0:
            tickers = random.sample(tickers, n_max)
        
        try:
            df_tickers = df_prices[tickers]
        except KeyError:
            return print('ERROR: Check tickers')

        dts = df_tickers.apply(lambda x: x.dropna().index.min()) # start date of each tickers
        if base > 0: # relative price in common period
            dt_adj = df_tickers.index.min()
            dt_max = dts.max() # min start date where all tickers have data 
            dt_adj = dt_max if dt_adj < dt_max else dt_adj
            df_tickers = df_tickers.apply(lambda x: x / x.loc[dt_adj] * base)
        elif base == 0: # relative price only
            df_tickers = df_tickers.apply(lambda x: x.dropna() / x.dropna().iloc[0])
            dt_adj = None
        else:
            dt_adj = dts.min() # drop dates of all None
        return df_tickers.loc[dt_adj:] 


    def _get_prices_after_fee(self, df_prices, cost=None, period=3, percent=True):
        """
        get df_prices after cost
        cost: dict of buy/sell commissions, fee and tax. see CostManager
        """
        cost = cost or dict()
        return CostManager.get_history_with_fee(df_prices, period=period, percent=percent, **cost)

        

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
        
        self.df_data = df_data.rename_axis(col_date).sort_index()
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
    def __init__(self, file=None, path='.', file_historical=None, check_master=True, msg=True,
                 freq='daily', batch_size=None, # recommended 150 days for daily, 24 months for monthly
                 col_ticker='ticker', 
                 cols_check=['check1_date', 'check1_price', 'check2_date', 'check2_price'],
                 cols_commissions=['buy', 'sell', 'fee'],
                 url = "https://dis.kofia.or.kr/proframeWeb/XMLSERVICES/",
                 headers = {"Content-Type": "application/xml"}):
        file = set_filename(file, 'csv')
        file_historical = set_filename(file_historical, 'csv') 
        # master file of securities with ticker, name, adjusting data, etc
        self.file_master = get_file_latest(file, path)
        # price file name
        file_historical = None if file_historical is None else get_file_latest(file_historical, path)
        self.file_historical = file_historical
        self.path = path
        self.col_ticker = col_ticker
        self.cols_check = cols_check
        self.cols_commissions = cols_commissions
        # df of master file
        self.data_tickers = self._load_master(msg)
        self.url = url
        self.headers = headers
        self.tickers = None # tickers to download rate
        self.df_prices = None
        #self.df_rates = None # rate data downloaded to calc df_prices
        self.freq = freq # price data freq
        # period of a batch for downloading. days if freq is daily, months if monthly
        self.batch_size = batch_size 
        self.interval_min = None # a list of start & end dates just enough to rate conversion
        self.failed = {} # tickers failed to download
        self.debug_fetch_data = None # for debugging. see _download_data
        # check missing data for conversion
        _ = self.check_master() if check_master else None
        if batch_size is None:
            print(f'ERROR: Set batch_size for {freq} price data')
            print(f'Check {url} first to determine the batch_size')
        # allows asyncio.run() inside an already running event loop
        nest_asyncio.apply()
        

    def _load_master(self, msg=True):
        """
        load master file of tickers and its info such as values for conversion from rate to price
        """
        file = self.file_master
        path = self.path
        col_ticker = self.col_ticker
        cols_check = self.cols_check
        cols_dates = [cols_check[i] for i in [0,2]]
        f = f'{path}/{file}'
        try:
            data_tickers = pd.read_csv(f, index_col=col_ticker, parse_dates=cols_dates)
        except Exception as e:
            return print(f'ERROR: failed to load {file} as {e}')
            
        cols = pd.Index(cols_check).difference(data_tickers.columns)
        if cols.size > 0:
            data_tickers[cols] = None
        print(f'Master data for {len(data_tickers)} funds loaded.') if msg else None
        return data_tickers
        

    def check_master(self):
        """
        check if data_tickers exists or has duplicated tickers
        """
        data_tickers = self.data_tickers
        if data_tickers is None:
            print('ERROR: no ticker data loaded yet')
            return False

        # check ticker duplication
        idx = data_tickers.index
        idx = idx[idx.duplicated()]
        if idx.size > 0:
            #print_list(idx.drop_duplicates(), 'ERROR: Duplicated {}')
            print(f'ERROR: {idx.size} tickers duplicated')
            #return idx.to_list()
            return False
        return True


    def update_master(self, save=True, overwrite=False, batch_size=None,
                      interval=5, pause_duration=.1, msg=False):
        """
        update & save the master file
        overwrite: set to False to update only update values that are NA
        batch_size: adjust if necessary
        """
        data_tickers = self.data_tickers
        if data_tickers is None:
            return print('ERROR: No data_tickers available')

        batch_size = self._check_var(batch_size, self.batch_size)
        kw = dict(interval=interval, pause_duration=pause_duration, 
                  msg=msg, overwrite=overwrite)
        # update data for conversion to price
        df_up = self._get_master_conversions(data_tickers, batch_size=batch_size, **kw)
        if df_up is not None:
            data_tickers.update(df_up, overwrite=True)

        # update commission
        df_up = self._get_master_commissions(data_tickers, **kw)
        if df_up is not None:
            data_tickers.update(df_up, overwrite=True)
            
        if df_up is not None:
            # assign before saving updated
            self.data_tickers = data_tickers 
            # create new master file with today
            self.save_master(overwrite=False) if save else None 
        return None


    def _get_master_conversions(self, data_tickers, batch_size=120, interval=5, pause_duration=.1, 
                                msg=False, overwrite=False):
        """
        download convesion data to update data_tickers
        """
        col_ticker = self.col_ticker
        cols_check = self.cols_check
        cols_check_float = ['check1_price', 'check2_price']

        if overwrite: # update all
            tickers = data_tickers.index
        else: # update nan only
            tickers = data_tickers.loc[data_tickers[cols_check].isna().any(axis=1)].index
        if tickers.size == 0:
            return None
        else:
            print('Collecting conversion data ...')
        
        data, failed = list(), list()
        tracker = TimeTracker(auto_start=True)
        for x in tqdm(tickers):
            # download settlements history to get dates for price history
            df = self.download_settlements(x, msg=msg)
            if df is None:
                failed.append(x)
                continue
            else:
                cond = (df['type'] == '결산') # "결산 및 상환" 탭의 "구분명" 컬럼값
                start = df.loc[cond, 'start'].max()
                end = df.set_index('start').loc[start, 'end']
                
            # download date & price for conversion from rate to price
            sr_p = self.download_price(x, start, end, freq=self.freq, batch_size=batch_size, msg=msg)
            if sr_p is None:
                failed.append(x)
                continue
            else:
                start = sr_p.index.min()
                end = sr_p.index.max()
                data.append([x, start, sr_p[start], end, sr_p[end]])
            tracker.pause(interval=interval, pause_duration=pause_duration, msg=msg)
        tracker.stop()
        
        if len(failed) > 0:
            #print_list(failed, 'ERROR: Failed to get conversion data for {}')
            print(f'WARNING: {len(failed)} tickers failed to get conversion data')

        if len(data) > 0:
            cols = [col_ticker, *cols_check]
            df_up = pd.DataFrame().from_records(data, columns=cols).set_index(col_ticker)
            df_up[cols_check_float] = df_up[cols_check_float].astype(float)
            return df_up
        else:
            return None


    def _get_master_commissions(self, data_tickers, interval=5, pause_duration=.1, 
                                msg=False, overwrite=False):
        """
        download commission data to update data_tickers
        """
        col_ticker = self.col_ticker
        cols_cms = self.cols_commissions

        if overwrite: # update all
            tickers = data_tickers.index
        else: # update only tickers with all commisions nan
            tickers = data_tickers.loc[data_tickers[cols_cms].isna().all(axis=1)].index
        if tickers.size == 0:
            return None
        else:
            print('Collecting commission data ...')
        
        df_cms, failed = None, list()
        tracker = TimeTracker(auto_start=True)
        for x in tqdm(tickers):
            df = self.download_commissions(x, msg=msg)
            if df is None:
                failed.append(x)
                continue
            else:
                df_cms = df if df_cms is None else pd.concat([df_cms, df])
            tracker.pause(interval=interval, pause_duration=pause_duration, msg=msg)
        tracker.stop()

        if len(failed) > 0:
            #print_list(failed, 'ERROR: Failed to get commission data for {}')
            print(f'WARNING: {len(failed)} tickers failed to get commission data')
        
        if df_cms is not None:
            df_cms[cols_cms] = df_cms[cols_cms].astype(float)
            return df_cms
        else:
            return None
        

    def set_tickers(self, tickers=None, col_ticker='ticker'):
        """
        set tickers to download prices
        tickers: tickers to download.
        """
        cols_check = self.cols_check
        data_tickers = self.data_tickers
        if data_tickers is None:
            return print('ERROR')
        else:
            tickers_all = data_tickers.index.to_list()

        # check tickers to download
        if tickers is None:
            tickers = tickers_all
        else:
            tickers = [tickers] if isinstance(tickers, str) else tickers
            n = pd.Index(tickers).difference(tickers_all).size
            if n > 0:
                print(f'WARNING: {n} funds unable to process')
                tickers = pd.Index(tickers).intersection(tickers_all).to_list()

        # set start & end dates required for rate conversion
        col_start, col_end = [cols_check[i] for i in [0,2]]
        start = data_tickers.loc[tickers, col_start].min()
        end = data_tickers.loc[tickers, col_start].max()
        self.interval_min = [start, end]
    
        print(f'{len(tickers)} tickers set to download')
        self.tickers = tickers
        return None


    def download(self, start_date, end_date, percentage=True,
                 url=None, headers=None,
                 interval=5, pause_duration=.1, msg=False, batch_size=None,
                 file=None, path=None, save=True, n_retry=3, timeout=60):
        """
        download rate and convert to price using prices in settlement info
        file/path: file/path to save price data
        n_retry: total num of retry of downloading rate
        timeout: time to wait to reserve a ticker for later retry
        """
        data_tickers = self.data_tickers
        tickers = self.tickers
        if tickers is None:
            return print('ERROR: load tickers first')

        # date check for rate conversion
        start, end = pd.DatetimeIndex([start_date, end_date])
        istart, iend = self.interval_min
        msgw = False
        if start > istart:
            start_date = istart.strftime('%Y-%m-%d')
            msgw = True
        if end < iend:
            end_date = iend.strftime('%Y-%m-%d')
            msgw = True
        if msgw:
            print(f'WARNING: Download period set to {start_date} ~ {end_date} for rate conversion')
        
        batch_size = self._check_var(batch_size, self.batch_size)
        kwargs = dict(freq=self.freq, batch_size=batch_size, 
                      url=url, headers=headers, interval=interval, 
                      pause_duration=pause_duration, msg=msg,
                      n_retry=n_retry, timeout=timeout)
        df_rates = asyncio.run(self._get_rate(tickers, start_date, end_date, **kwargs))
        
        # convert to price
        df_prices, sr_err = self._get_prices(df_rates, data_tickers, percentage=percentage, msg=msg)
        if sr_err is not None:
            self.df_prices = df_prices.round(1)
            self.save(file, path) if save else None
            _ = [print(f'{len(v)} tickers failed for {k}') for k, v in self.failed.items() if len(v) > 0]    
        return sr_err


    async def _get_rate(self, tickers, start_date, end_date, msg=False, n_retry=3, **kwargs):
        """
        Asynchronously downloads rate data for a list of tickers within a given date range.
        Handles retries and progress tracking.
        kwargs: see _get_rate_batch
        """
        tracker = TimeTracker(auto_start=True)
        df_rates, failed, tkrs_tout = None, [], []
        tkrs = tickers
        i_try = 0
        
        while i_try <= n_retry:
            res = await self._get_rate_batch(tkrs, start_date, end_date, 
                                             df_rates=df_rates, failed=failed, tickers_tout=tkrs_tout, tracker=tracker,
                                             msg=msg, **kwargs)
            df_rates, failed, tkrs_tout = res
            if tkrs_tout:  
                print(f'Retry {len(tkrs_tout)} tickers')
            else: # If no tickers timed out, exit retry loop
                break
            tkrs = tkrs_tout  # Retry only failed tickers
            tkrs_tout = []  # Reset timeout list
            i_try += 1
        tracker.stop(msg=msg)

        if df_rates is None:
            return print('ERROR: Set msg to True to see error messages')
        else:
            df_rates = df_rates.sort_index()
            n = len(failed)
            print(f'WARNING: {n} tickers failed to download') if n>0 else None
        self.failed = {'downloading': failed}  # Reset failure tracking
        return df_rates


    async def _get_rate_batch(self, tickers, start_date, end_date, 
                              df_rates=None, failed=None, tickers_tout=None, tracker=None,
                              freq='monthly', batch_size=24, url=None, headers=None, interval=5, 
                              pause_duration=0.1, msg=False, progress_meter=True, timeout=60):
        """
        Downloads ticker data asynchronously with timeout handling.
        """
        url = self._check_var(url, self.url)
        headers = self._check_var(headers, self.headers)
        kw = dict(freq=freq, batch_size=batch_size, msg=msg, url=url, headers=headers)
        iterator = tqdm(tickers, total=len(tickers)) if progress_meter else tickers
        for x in iterator:
            try:
                sr_tkr = await asyncio.wait_for(
                    self.download_rate(x, start_date, end_date, **kw), timeout=timeout)
                if sr_tkr is None:
                    failed.append(x)
                else:
                    df_rates = sr_tkr.to_frame() if df_rates is None else pd.concat([df_rates, sr_tkr], axis=1)
            except asyncio.TimeoutError:
                tickers_tout.append(x)
            tracker.pause(interval=interval, pause_duration=pause_duration, msg=msg)
        return df_rates, failed, tickers_tout


    def _get_prices(self, df_rates, data_tickers, percentage=True, msg=True):
        df_prices = None
        failed = [] # tickers failed to convert
        # convert nan & NaT to None for to_dict
        data_tickers = data_tickers.map(lambda x: 0 if pd.isna(x) else x).replace(0, None)
        errors, index_errors = list(), list()
        for x in df_rates.columns:
            sr_n_err = self._convert_rate(x, data_tickers, df_rates, percentage=percentage, msg=msg)
            if sr_n_err is None: # see _convert_rate for err msg
                failed.append(x)
            else: 
                sr, err = sr_n_err
                df_prices = sr.to_frame() if df_prices is None else pd.concat([df_prices, sr], axis=1)
                index_errors.append(x)
                errors.append(err)
        if len(errors) > 0:
            print(f'Max error of conversions: {max(errors):.2e}') if msg else None
            df_prices = df_prices.sort_index()
            sr_err = pd.Series(errors, index=index_errors, name='error')
        else:
            sr_err = None
        self.failed['conversion'] = failed # reset old conversion error
        return df_prices, sr_err


    async def download_rate(self, ticker, start_date, end_date, freq='monthly', batch_size=24, 
                       percentage=True, **kwargs):
        """
        download rate data of ticker
        kwargs: additional args for _download_rate
        """
        start_date, end_date = pd.DatetimeIndex([start_date, end_date])
        kw_offset = {'months' if freq=='monthly' else 'days': batch_size}
        unit = 100 if percentage else 1
        end = end_date
        sr_rates = None
        while end >= start_date:
            start = max(end - pd.DateOffset(**kw_offset), start_date)
            sr_p = self._download_rate(ticker, start, end, freq=freq, **kwargs)
            if (sr_p is None) or sr_p.sum() == 0:
                break
    
            if sr_rates is None:
                sr_rates = sr_p
            else:
                if sr_p.index.min() == sr_rates.index.min():
                        break
                else:
                    sr_p = sr_p.sort_index()
                    # convert sr_rates based on sr_p
                    ft = (1 + sr_p.dropna().iloc[-1] / unit)
                    sr_rates = ft * (unit + sr_rates)  - unit
                    # remove last date of sr_p before concat
                    sr_p = sr_p.iloc[:-1] if sr_p.index.max() == sr_rates.index.min() else sr_p
                    sr_rates = pd.concat([sr_rates, sr_p], axis=0)
            end = sr_rates.index.min() # include prv start date
        return sr_rates.sort_index().round(2) if sr_rates is not None else None


    def download_price(self, ticker, start_date, end_date, freq='daily', batch_size=120, 
                       percentage=True, **kwargs):
        """
        download price data of ticker
        kwargs: additional args for _download_price
        """
        start_date, end_date = pd.DatetimeIndex([start_date, end_date])
        kw_offset = {'months' if freq=='monthly' else 'days': batch_size}
        end = end_date
        sr_prices = None
        while end >= start_date:
            start = max(end - pd.DateOffset(**kw_offset), start_date)
            sr_p = self._download_price(ticker, start, end, freq=freq, **kwargs)
            if sr_p is None:
                break
            else:
                sr_p = sr_p['price']
    
            if sr_prices is None:
                sr_prices = sr_p
            else:
                if sr_p.index.min() == sr_prices.index.min():
                    break
                else:
                    sr_p = sr_p.sort_index()
                    # remove last date of sr_p before concat
                    sr_p = sr_p.iloc[:-1] if sr_p.index.max() == sr_prices.index.min() else sr_p
                    sr_prices = pd.concat([sr_prices, sr_p], axis=0)
            end = sr_prices.index.min() # include prv start date
        return sr_prices.sort_index().round(1) if sr_prices is not None else None
        

    def save(self, file=None, path=None, overwrite=False):
        """
        save price data
        """
        df_prices = self.df_prices
        if df_prices is None:
            return print('ERROR')
        
        file = self._check_var(file, self.file_historical)
        path = self._check_var(path, self.path)
        self._save(df_prices, file, path, overwrite=overwrite)
        return None


    def save_master(self, file=None, path=None, overwrite=False):
        """
        save master data
        """
        data_tickers = self.data_tickers
        if data_tickers is None:
            return print('ERROR')
        file = self._check_var(file, self.file_master)
        path = self._check_var(path, self.path)
        self._save(data_tickers, file, path, overwrite=overwrite)
        return None


    def _save(self, df_result, file, path, date=None, date_format='%y%m%d', overwrite=False):
        """
        overwrite: set to False to save df_result to new file of date
        """
        if not overwrite:
            if date is None:
                date = datetime.now()
            if not isinstance(date, str):
                date = date.strftime(date_format)
            file = get_filename(file, f'_{date}', r"_\d+(?=\.\w+$)")
        _ = save_dataframe(df_result, file, path, overwrite=overwrite,
                           msg_succeed=f'{file} saved',
                           msg_fail=f'ERROR: failed to save as {file} exists')
        return None


    def _download_rate(self, ticker, start_date, end_date, freq='m',
                       msg=False, url=None, headers=None, date_format='%Y%m%d',
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
        
        df = self._download_data(payload, tag_iter, tags, url=url, headers=headers, msg=msg, **kwargs)
        if df is not None:
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df = df['rate'].astype('float').rename(ticker)
        return df
        

    def download_settlements(self, ticker, 
                             msg=False, url=None, headers=None,
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
        df = self._download_data(payload, tag_iter, tags, url=url, headers=headers, msg=msg, ticker=ticker)
        if df is not None:
            df['price'] = df['price'].astype('float')
            df['amount'] = df['amount'].astype('int')
            df['start'] = pd.to_datetime(df['start'])
            df['end'] = pd.to_datetime(df['end'])
        return df


    def _download_price(self, ticker, start_date, end_date, freq='m',
                        msg=False, url=None, headers=None, date_format='%Y%m%d',
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
        df = self._download_data(payload, tag_iter, tags, url=url, headers=headers, msg=msg, **kwargs)
        if df is not None:
            df = df.set_index('date')
            df.index = pd.to_datetime(df.index)
            df['price'] = df['price'].astype('float')
            df['amount'] = df['amount'].astype('int')
        return df


    def download_commissions(self, ticker, msg=False, url=None, headers=None,
                             payload="""<?xml version="1.0" encoding="utf-8"?>
                                        <message>
                                          <proframeHeader>
                                            <pfmAppName>FS-COM</pfmAppName>
                                            <pfmSvcName>COMFundUnityBasInfoSO</pfmSvcName>
                                            <pfmFnName>fundBasInfoSrch</pfmFnName>
                                          </proframeHeader>
                                          <systemHeader></systemHeader>
                                            <COMFundUnityInfoInputDTO>
                                            <standardCd>{ticker:}</standardCd>
                                            <companyCd></companyCd>
                                            <standardDt></standardDt>
                                        </COMFundUnityInfoInputDTO>
                                        </message>""",
                               tag_iter='COMFundBasInfoOutDTO', 
                               tags={'fee':'rewSum', 'buy':'frontendCmsRate', 'sell':'backendCmsRate'}
                              ):
        """
        get fee, buy/sell commissions of a ticker
        """
        # convert inputs for request
        kwargs = dict(ticker=ticker)
        df = self._download_data(payload, tag_iter, tags, url=url, headers=headers, msg=msg, **kwargs)
        if df is not None:
            df.index = [ticker]
        return df


    def _download_data(self, payload, tag_iter, tags, 
                       msg=True, url=None, headers=None, **kwargs_payload):

        url = self._check_var(url, self.url)
        headers = self._check_var(headers, self.headers)
        payload = payload.format(**kwargs_payload)
        self.debug_fetch_data = dict(url=url, headers=headers, data=payload) # for debugging
        xml = FundDownloader.fetch_data(url, headers, payload, msg=msg)
        return None if xml is None else FundDownloader.parse_xml(xml, tag_iter, tags, msg=msg)

    
    @staticmethod
    def fetch_data(url, headers, payload, msg=False):
        # Sending the POST request
        try:
            response = requests.post(url, headers=headers, data=payload)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            return response.text
        except requests.exceptions.RequestException as e:
            return print(f"An error occurred: {e}") if msg else None

    
    @staticmethod
    def parse_xml(xml, tag_iter, tags, msg=True):
        """
        tags: dict of column name to tag in list or list of tags
        """
        if isinstance(tags, dict):
            cols = list(tags.keys())
            tags = list(tags.values())
        elif isinstance(tags, (list, tuple)):
            cols = tags
        else:
            return print('ERROR from parse_xml') if msg else None
            
        root = ET.fromstring(xml)
        data = list()
        try:
            for itr in root.iter(tag_iter):
                d = [itr.find(x).text for x in tags]
                data.append(d)
        except Exception as e:
            return print(f'ERROR: {e}') if msg else None
    
        if len(data) == 0:
            return print('ERROR from parse_xml') if msg else None
        
        return pd.DataFrame().from_records(data, columns=cols)
    

    def _convert_rate(self, ticker, data_tickers, df_rates, 
                      percentage=True, msg=False, price_init=1000):
        """
        calc price from rate of return
        """
        data = data_tickers.loc[ticker].to_dict()
        sr_rate = df_rates[ticker].dropna()
 
        unit = 100 if percentage else 1
        dt1, prc1, dt2, prc2 = [data[x] for x in self.cols_check]
        if dt1 is None: # reset all conversion data for the ticker
            rat1 = sr_rate.iloc[0]
            prc1 = price_init
            dt2 = sr_rate.index.max()
            # set price to get zero for conversion error
            prc2 = prc1 * (sr_rate.loc[dt2] + unit) / unit
        else:
            try:
                rat1 = sr_rate.loc[dt1]
            except KeyError as e: # df_rates has no date for conversion
                return print(f'ERROR: Check data for {ticker}') if msg else None
                
        # calc price from rate
        price_base = prc1 / (rat1 + unit)
        sr_price = (sr_rate + unit) * price_base 

        # calc conversion error
        try:
            e = sr_price.loc[dt2]/prc2 - 1
        except KeyError as e:
            print(f'WARNING: Failed to calc error for {ticker}')
            e = 0
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
    def create(file, path='.', **kwargs):
        """
        file: master file or instance of DataManager
        """
        if isinstance(file, DataManager):
            path = file.path
            kwargs = {**file.kwargs_universe, **kwargs}
            # get file name at the end
            file = file.tickers
        return FundDownloader(file, path, **kwargs)

    
    @staticmethod
    def export_master(file, path='.'):
        """
        get df of fund list (master)
        """
        fd = FundDownloader.create(file, path=path)
        return fd.data_tickers

    
    def export_cost(self, universe, file=None, path='.', update=True,
                    cols_cost=['buy', 'sell', 'fee', 'tax'],
                    col_uv='universe', col_ticker='ticker', universes=UNIVERSES):
        """
        universe: universe name. see keys of UNIVERSES
        update: True to update the file with new cost data
        """
        data_tickers = self.data_tickers
        if data_tickers is None:
            return print('ERROR: no ticker data loaded yet')

        cols = [col_ticker, *cols_cost]
        # get universe name for cost data
        univ = PortfolioData.get_universe(universe, universes=universes) if universes else universe
        df_cost = (data_tickers.reset_index().loc[:, cols]
                   .fillna(0)
                   .assign(universe=univ)
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
                df_cost = pd.concat([df.reset_index(), df_cost]).sort_index()
                # save as new file name
                dt = datetime.today().strftime('%y%m%d')
                file = get_filename(file, f'_{dt}', r"_\d+(?=\.\w+$)")
            save_dataframe(df_cost, file, path, 
                           msg_succeed=f'Cost data saved to {file}',
                           index=False)
            return None
        else:
            return df_cost


    def load_price(self, file=None, path=None):
        """
        load data to check result
        """
        file = self._check_var(file, self.file_historical)
        path = self._check_var(path, self.path)
        try:
            df_prices = pd.read_csv(f'{path}/{file}', parse_dates=[0], index_col=[0])
        except FileNotFoundError as e:
            return print('ERROR: No price data to load')
        self.df_prices = df_prices
        return None


    def util_check_price(self, tickers=None):
        """
        return rate converted from price to check if price conversion works or not
        """
        df_prices = self.df_prices
        if df_prices is None:
            return None
        if tickers is None:
            df_prc = df_prices
        else:
            try:
                df_prc = df_prices[tickers]
            except KeyError as e:
                return print('ERROR: Check tickers')
        return df_prc.apply(lambda x: 1 - x.dropna().iloc[0]/x.dropna()).mul(100).round(2)
            


class PortfolioBuilder():
    def __init__(self, df_universe, file=None, path='.', name='portfolio',
                 method_select='all', sort_ascending=False, n_tickers=0, lookback=0, lag=0, tickers=None, 
                 method_weigh='Equally', weights=None, lookback_w=None, lag_w=None, weight_min=0,
                 df_additional=None, security_names=None, unit_fund=False,
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
        n_tickers: num of tickers to select
        security_names: dict of ticker to name
        cols_record: all the data fields of transaction file
        """
        self.df_universe = df_universe.rename_axis(cols_record['date'])
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
        self.df_additional = df_additional # addtional data except for price
        self.security_names = security_names
        self.unit_fund = unit_fund # True for managed fund. see allocate
        self.name = name # portfolio name
        self.cols_record = cols_record
        self.date_format = date_format # date str format for record & printing
        self.cost = cost # dict of buy/sell commissions, fee and tax (%). see CostManager
        
        self.selected = None # data for select, weigh and allocate
        self.df_rec = None # record updated with new transaction
        # TradingHalts instance to save existing record before new transaction. 
        # see import_record for its init
        self.tradinghalts = None 
        # records of all trading except for halted
        self.record = self.import_record(return_on_fail=True) 
        # record of halted after new transaction
        self.record_halt = None if self.tradinghalts is None else self.tradinghalts.record_halt
        # price data not in universe but in record. see _update_universe
        self.df_prices_missing = None 
        _ = self.check_universe(msg=True)
            

    def import_record(self, record=None, halt=True, msg=True, return_on_fail=True):
        """
        read record from file and update transaction dates
        halt: set to False before saving
        """
        if record is None:
            record = self._load_transaction(self.file, self.path, print_msg=msg)
        
        if record is None:
            return print('REMINDER: make sure this is 1st transaction as no records provided')
        
        # run _check_record instead of _check_result as self.record not yet set
        if not self._check_record(record):
            return record if return_on_fail else None

        # check record is amount-based
        if record[self.cols_record['prc']].notna().any(): 
            print('WARNING: Run update_record first after editing record') if msg else None
            return record
        else:
            # init TradingHalts instance only with amount-based
            self.tradinghalts = TradingHalts(record, self.cols_record)
            # use record w/o tickers of halt if halt true
            return self.tradinghalts.record if halt else record


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

        # check tickers
        if tickers is not None:
            # check duplicates
            tickers = pd.Index(tickers)
            if tickers.duplicated().any():
                tickers = tickers.drop_duplicates()
                print('WARNING: Duplicate tickers removed')
            # check universe
            dup = tickers.difference(self.df_universe.columns)
            if dup.size > 0:
                dup = ', '.join(dup)
                return print(f'ERROR: {dup} not in universe') 
            # check portfolio size
            if (n_tickers is not None) and n_tickers > tickers.size:
                return print(f'ERROR: n_tickers greater than ticker size {tickers.size}')
            else: # back to list
                tickers = tickers.to_list()
            
        
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
        else: # all or selected
            if cond(method, 'selected'):
                if tickers is None:
                    record = self.record
                    if record is None: # force to all
                        tickers = df_data.columns
                        method = 'All'
                    else: # select tickers in the latest transaction
                        col_date, col_tkr = [self.cols_record[x] for x in ['date','tkr']]
                        date_lt = record.index.get_level_values(col_date).max()
                        tickers = record.loc[date_lt].index.to_list()
                        method = 'Selected'
                else:
                    method = 'Selected'
            else:
                tickers = df_data.columns
                method = 'All'
            rank = pd.Series(1, index=tickers)
            n_tickers = rank.count()
                
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
            # return equal weights if weights is str or list
            w = self.check_weights(weights, df_data, none_weight_is_error=True)
            if w is None:
                return None
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
        
        self.selected['weights'] = weights # weights is series
        print(f'Weights of tickers determined by {method}.')
        return weights
        

    def allocate(self, capital=10000000, int_nshares=True):
        """
        calc amount of each security for net on the transaction date 
        capital: amount of cash for rebalance. int if money, float if ratio to portfolio values, 
                 postitive to add, negative to reduce
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
        if self.record is None:
            if capital == 0:
                return print('ERROR: Neither capital nor tickers to rebalance exists')
        else: # determine total cash for rebalance
            if self.check_new_transaction(date):
                self.df_rec = None # reset df_rec to calc capital
                sr = self.valuate(date, total=True, exclude_cost=True, 
                                  int_to_str=False, print_msg=False)
                val = sr['value']
                st = 'contribution' if capital > 0 else 'residual'
                if abs(capital) > 1:
                    print(f'Rebalancing with {st} {abs(capital):,}')
                    capital += val 
                else:
                    x = round(capital* val)
                    print(f'Rebalancing with {st} {abs(capital):.0%} of the portfolio value ({x:,})')
                    capital = (1 + capital) * val

        # calc amount of each security by weights and capital
        df_prc = self.df_universe # no _update_universe to work on tickers in the universe
        wi = pd.Series(weights, name=col_wgt).rename_axis(col_tkr) # ideal weights
        sr_net = wi * capital # weighted security value
        
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
               # set errot zero if denominator is zero
               .apply(lambda x: 0 if x[col_wgt] == 0 else x[col_wgta]/x[col_wgt] - 1, axis=1)
               .abs().mean() * 100)
        print(f'Mean absolute error of weights: {mae:.0f} %')
        
        df_net = (sr_net.to_frame().join(wi.round(4)) # round weights after finishing calc
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


    def transaction(self, df_net, date_actual=None):
        """
        add new transaction to records
        df_net: output of self.allocate
        date_actual: set actual transaction date
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
        date_actual = date if date_actual is None else date_actual
        record = self.record
        if record is None: # no transation record saved
            # allocation is same as transaction for the 1st time
            df_rec = df_net.assign(**{col_trs: df_net[col_net], col_dttr:date_actual})
        else:
            # check input record by self.cols_record
            if not self._check_record(record):
                return None 
            # check if new transaction added
            if self.check_new_transaction(date):
                # confine tickers on transaction date
                date_lt = record.index.get_level_values(col_date).max()
                tickers_lt = record.loc[date_lt].index
                tickers_lt = tickers_lt.union(df_net.index.get_level_values(col_tkr))
                # add new to record after removing additional info except for cols_all in record
                df_rec = pd.concat([record[cols_all], df_net]).sort_index()
                # update universe by adding tickers not in the universe but in the past transactions
                df_prc = self._update_universe(df_rec, msg=True)
            else: # return None if no new transaction
                return None
            
            # get assets of zero net and concat to df_rec
            lidx = [df_rec.index.get_level_values(i).unique() for i in range(2)]
            midx = pd.MultiIndex.from_product(lidx).difference(df_rec.index)
            df_m = pd.DataFrame({col_rat:1, col_net:0, col_wgt:0}, index=midx)
            # add security names
            if self.security_names is not None: 
                df_m = df_m.join(pd.Series(self.security_names, name=col_name), on=col_tkr)
            df_rec = pd.concat([df_rec, df_m]).sort_index()

            # get num of shares for transaction & net with price history
            # where num of shares is ratio of value to close from latest data
            df_nshares = self._get_nshares(df_rec, df_prc, cols_record, int_nshares=False)
            # get transaction amount for the transaction date
            df_trs = (df_nshares.loc[date, col_net]
                      .sub(df_nshares.groupby(col_tkr)[col_trs].sum())
                      .mul(df_prc.loc[date]).dropna() # get amount by multiplying price
                      .round() # round very small transaction to zero for the cond later
                      .to_frame(col_trs).assign(**{col_date:date, col_dttr:date_actual})
                      .set_index(col_date, append=True).swaplevel())
            # confine tickers on the transaction date
            df_trs = df_trs.loc[df_trs.index.get_level_values(1).isin(tickers_lt)]
            df_rec.update(df_trs)
            # drop new tickers before the date
            df_rec = df_rec.dropna(subset=cols_val) 
            # drop rows with neither transaction nor net 
            cond = (df_rec[col_trs] == 0) & (df_rec[col_net] == 0)
            df_rec = df_rec.loc[~cond]

        df_rec = df_rec[cols_all]
        #df_rec.loc[:, cols_int] = df_rec.loc[:, cols_int].astype(int).sort_index(level=[0,1])
        df_rec[cols_int] = df_rec[cols_int].astype(int)

        # print Invested capital or Residual cash
        date_lt = df_rec.index.get_level_values(col_date).max()
        invested = df_rec.loc[date_lt, col_trs].sum()
        st = 'Deployed capital' if invested > 0 else 'Residual cash'
        print(f'{st}: {abs(invested):,}')
        
        # overwrite existing df_rec with new transaction
        self.df_rec = df_rec
        # print portfolio value and profit/loss after self.df_rec updated
        _ = self.valuate(total=True, int_to_str=True, print_summary_only=True)
        return df_rec


    def valuate(self, date=None, total=True, exclude_cost=False, exclude_sold=True,
                print_msg=False, print_summary_only=False,
                sort_by='ugl', int_to_str=True, join_str=False):
        """
        calc date, buy/sell prices & portfolio value from self.record or self.df_rec
        date: date, None, 'all'
        exclude_sold: set to True to exclude assets that are no longer held when calculating individual asset performance.
            effective when date!='all' & total=False
        sort_by: sort value of date (total=False) by one of 'start', 'end', 'buy', 'sell', 
                'value', 'ugl', 'roi' in descending order
        int_to_str: applied only if date != 'all'
        join_str: applied only if date != 'all' and total==True
        """
        if print_summary_only:
            print_msg = False # force no print except print_summary_only case
    
        # get latest record
        df_rec = self._check_result(print_msg)
        if df_rec is None:
            return None
    
        if isinstance(date, str) and date.lower() == 'all':
            date = None
            history = True
        else:
            history = False
    
        date_format = self.date_format
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
        col_net = cols_record['net']
        col_roi = 'roi'
        col_ugl = 'ugl'
        col_start, col_end = 'start', 'end'
        col_val = 'value'
    
        # check date by price data
        df_prices = self._update_universe(df_rec, msg=print_msg, download_missing=True)
        if df_prices is None:
            return None
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
        # calc value
        sr_val = self._calc_value_history(df_rec, df_prices, date, self.name, msg=False, 
                                          total=total, exclude_cost=exclude_cost)
        # buy & sell prices to date.
        df_cf = self._calc_cashflow_history(df_rec, total=total, exclude_cost=exclude_cost)
    
        # calc profit
        df_pnl = self._calc_profit(sr_val, df_cf, result='all')
        
        if total:
            df_m = df_cf.join(sr_val.rename('value'), how='right').join(df_pnl).ffill()
        else:
            df_m = self._join_cashflow_by_ticker(sr_val, df_cf, df_pnl)
        df_m = df_m.dropna(how='all')
    
        if not history:
            if total:
                start = df_m.index.get_level_values(col_date).min().strftime(date_format)
                sr = pd.Series([start, date], index=[col_start, col_end])
                sr_m = pd.concat([sr, df_m.loc[date]]) # concat data range
                df_m = sr_m.apply(format_price, digits=0) if int_to_str else sr_m
                if print_msg or print_summary_only:
                    if join_str:
                        sr = sr_m.apply(format_price, digits=0, int_to_str=False)
                        idx = ', '.join(sr.index)
                        to_print = ', '.join(map(str, sr.to_list()))
                        to_print = f'{idx}\n{to_print}'
                    else:
                        to_print = PortfolioBuilder.get_title_pnl(df_m[col_roi], df_m[col_ugl], date)
                    print(to_print) 
            else:
                # get periods of holding for each assets
                df_r = df_m.groupby(col_tkr).apply(lambda x: pd.Series(get_date_minmax(x.dropna()), index=[col_start, col_end]))
                # update end date of assets liquidated on the latest transaction
                date_lt = df_rec.index.get_level_values(col_date).max()
                sr_net = df_rec.loc[date_lt, col_net]
                tkrs = sr_net.loc[sr_net == 0].index
                if tkrs.size > 0:
                    df_r.loc[tkrs, col_end] = date_lt
                # add periods to performance data
                df_m = pd.concat([df_r, df_m.loc[date]], axis=1)
                df_m = df_m.sort_values(sort_by, ascending=False) if sort_by else df_m
                df_m = df_m.map(format_price, digits=0) if int_to_str else df_m
                # remove past (disposed) assets if exclude_sod = True
                df_m = df_m.loc[df_m[col_val].notna()] if exclude_sold else df_m
                # add asset name column if given
                if self.security_names is not None: 
                    col_name = cols_record['name']
                    cols = df_m.columns
                    df_m = df_m.join(pd.Series(self.security_names, name=col_name))
                    df_m = df_m[cols.insert(0, col_name)]
        return df_m


    def transaction_pipeline(self, date=None, capital=10000000, 
                             save=False, nshares=False, date_actual=None):
        """
        nshares: set to True if saving last transaction as num of shares for the convenience of trading
        capital: int, float if adding capital. 
                 dict of ticker to money to buy where all assets not in dict to sell 
        """        
        rank = self.select(date=date)
        if rank is None:
            return None # rank is not None even for static portfolio (method_select='all')
        
        if not self.check_new_transaction(date=None, msg=True):
            if self._check_result() is None:
                return None # return if record is not amount-based
            # calc profit at the last transaction
            dt = self.selected['date'] # selected defined by self.select
            _ = self.valuate(dt, total=True, int_to_str=True, print_msg=True)
            # add tickers of halt to recover original record
            return self.tradinghalts.recover(self.record, self.record_halt)  

        # check capital if given as asset to capital to buy
        # all assets not in dict will be sold
        if isinstance(capital, dict):
            method = 'specified'
            ttl = sum(capital.values())
            weights = {k:v/ttl for k,v in capital.items()}
            capital = ttl # set capital for allocate
        else:
            method, weights = None, None
            
        weights = self.weigh(method=method, weights=weights)
        if weights is None:
            return None
        
        df_net = self.allocate(capital=capital, int_nshares=not self.unit_fund)
        if df_net is None:
            return None
            
        df_rec = self.transaction(df_net, date_actual=date_actual)
        if df_rec is not None: # new transaction updated
            # recover record with halt before saving or converting to record with num of shares
            df_rec = df_rec if self.tradinghalts is None else self.tradinghalts.recover(df_rec, self.record_halt)
            if save:
                # save transaction as num of shares for the convenience of trading
                if nshares:
                    df_prc = self._update_universe(df_rec, msg=False)
                    # DO NOT SAVE both of transaction & net as int. Set int_nshares to False
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


    def transaction_halt(self, date=None, save=False, **kw_halt):
        """
        create transaction with TradingHalts instance
        kw_halt: kwargs for tradinghalts.transaction
        """
        record = self.record
        if record is None:
            # run transaction_pipeline if first transaction with buy only
            if 'buy' in kw_halt and {'sell', 'resume', 'halt'}.isdisjoint(kw_halt):
                kw = dict(
                    capital = kw_halt['buy'],
                    date_actual = kw_halt.pop('date_actual', None)
                )
                return self.transaction_pipeline(date=date, save=save, nshares=False, **kw)
            else:
                return print('ERROR: No transaction record exits')
        else:
            self.df_rec = None # reset prv transaction if any
    
        date = self._get_data(0, 0, date=date).index.max()
        # get values of assets on the date for tradinghalts
        sr_net = self.valuate(total=False, date=date, exclude_cost=True, int_to_str=False)
        sr_net = sr_net['value']
        recs = self.tradinghalts.transaction(date, date_format=self.date_format, 
                                             values_on_date=sr_net, **kw_halt) 
        if recs is not None: # new transaction created
            df_rec, record_halt = recs
            df_rec = self._update_ticker_name(df_rec) # update name for buy case.
            # save before recover
            self.df_rec = df_rec
            # recover record with halt before saving or converting to record with num of shares
            df_rec = self.tradinghalts.recover(df_rec, record_halt)
            if save:
                self.save_transaction(df_rec) # where self.record updated
            else:
                print('Set save=True to save transaction record')
            _ = self.valuate(total=True, int_to_str=True, print_summary_only=True)
            return df_rec
        else:
            return print('Nothing to save')


    def get_value_history(self, total=True, exclude_cost=True):
        """
        get history of portfolio value
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
        else:
            df_prices = self._update_universe(df_rec, msg=True, download_missing=True)
            if df_prices is None:
                return None
            else:
                return self._calc_value_history(df_rec, df_prices, name=self.name, msg=True,
                                                total=total, exclude_cost=exclude_cost)


    def get_cash_history(self, exclude_cost=True, cumsum=True, date_actual=False):
        """
        get history of buy and sell prices
        cumsum: set to False to check buy & sell for each transaction date
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None
        else:
            df_cf = self._calc_cashflow_history(df_rec, exclude_cost=exclude_cost)
    
        if not cumsum:
            df = df_cf.diff(1)
            df.update(df_cf, overwrite=False) # fill first date
            df_cf = df.astype(int)

        if date_actual:
            col_dttr = self.cols_record['dttr']
            col_tkr = self.cols_record['tkr']
            df_cf[col_dttr]=df_rec[col_dttr].droplevel(col_tkr).drop_duplicates()
            df_cf = df_cf.set_index(col_dttr)
        
        return df_cf


    def get_profit_history(self, result='ROI', total=True, exclude_cost=True, 
                           roi_log=False, msg=True):
        """
        get history of profit/loss
        result: 'ROI', 'UGL' or 'all'
        """
        df_all = self.valuate(date='all', total=total, exclude_cost=exclude_cost)
        if df_all is None:
            return None
        cols_pnl = ['ugl', 'roi']
    
        result = result.lower()
        if result in cols_pnl:
            return df_all[result]
        else:
            return df_all[cols_pnl]

    
    def plot(self, start_date=None, end_date=None, total=True, exclude_cost=False,
             figsize=(10,6), legend=True, height_ratios=(3,1), loc='upper left',
             roi=True, roi_log=False, cashflow=True):
        """
        plot total, net and profit histories of portfolio
        """
        df_all = self.valuate(date='all', total=total, exclude_cost=exclude_cost)
        if df_all is None:
            return None
        else: # necessary to get tickers of end_date and plot vlines for trasaction dates
            # unnecessary to check record as df_all must have passed the check_record
            df_rec = self._check_result(False, check_record=False)
    
        # reset start & end dates
        func = lambda x: x.loc[start_date:end_date]
        df_all = func(df_all)
        start_date, end_date = get_date_minmax(df_all, self.date_format)
    
        col_tkr = self.cols_record['tkr']
        col_date = self.cols_record['date']
        col_val = 'value'
        col_buy = 'buy'
        col_sell = 'sell'
        col_roi = 'roi'
        col_ugl = 'ugl'
    
        ax1, ax2 = self._plot_get_axes(figsize=figsize, height_ratios=height_ratios)
        # determine plot type: total or individuals
        col_pnl = col_roi if roi else col_ugl
        str_pnl = 'Return On Investment (%)' if roi else 'Unrealized Gain/Loss'
        mlf_pnl = 100 if roi else 1
        
        if total:
            # data for plot
            sr_ttl = df_all.apply(lambda x: x[col_sell] + x[col_val], axis=1)
            sr_val = df_all[col_val]
            sr_pnl = df_all[col_pnl].mul(mlf_pnl)
    
            # get title
            sr_end = df_all.loc[end_date]
            title = PortfolioBuilder.get_title_pnl(sr_end[col_roi], sr_end[col_ugl], end_date)
    
            # plot
            line_ttl = {'c':'darkgray', 'ls':'--'}
            _ = sr_ttl.plot(ax=ax1, label='Total', title=title, **line_ttl)
            _ = sr_val.plot(ax=ax1, c=line_ttl['c'])
            ax1.fill_between(sr_ttl.index, sr_ttl, ax1.get_ylim()[0], facecolor=line_ttl['c'], alpha=0.1)
            ax1.fill_between(sr_val.index, sr_val, ax1.get_ylim()[0], facecolor=line_ttl['c'], alpha=0.2)
            ax1.set_ylabel('Value')
            ax1.margins(0)
            
            # plot profit history
            ax1t = sr_pnl.plot(ax=ax1.twinx(), label=col_pnl.upper(), lw=1, color='orange')
            ax1t.set_ylabel(str_pnl)
            ax1t.margins(0)
            
            # set env for the twins
            _ = set_matplotlib_twins(ax1, ax1t, legend=legend, loc=loc)
        else:
            # data for plot
            # get tickers of the latest transaction
            dt = df_rec.loc[:end_date].index.get_level_values(col_date).max()
            tickers = df_rec.loc[dt].index
            idx = pd.IndexSlice
            df_tkr = df_all.loc[idx[:, tickers], :]
            
            df_pnl = df_tkr[col_pnl].unstack(col_tkr).mul(mlf_pnl)
            if self.security_names is not None:
                df_pnl.columns = [self.security_names[x] for x in df_pnl.columns]
            sr_pnl_ttl = df_tkr.groupby(col_date).sum()
            sr_pnl_ttl = sr_pnl_ttl[col_ugl].div(sr_pnl_ttl[col_buy].div(mlf_pnl) if roi else 1).rename('Portfolio')
    
            # get title
            sr_end = df_all.loc[end_date]
            sr_end = sr_end.sum()
            sr_end[col_roi] = sr_end[col_ugl]/sr_end[col_buy]
            title = PortfolioBuilder.get_title_pnl(sr_end[col_roi], sr_end[col_ugl], end_date)
    
            # plot profit history
            line_ttl = {'c':'darkgray', 'ls':'--'}
            _ = sr_pnl_ttl.plot(ax=ax1, lw=1, title=title, **line_ttl)
            ax1.fill_between(sr_pnl_ttl.index, sr_pnl_ttl, ax1.get_ylim()[0], facecolor=line_ttl['c'], alpha=0.1)
            _ = df_pnl.plot(ax=ax1, lw=1)
            yl = ax1.set_ylabel(str_pnl)
            ax1.yaxis.tick_right()
            ax1.yaxis.set_label_position("right")
            #yl.set_rotation(-90)
            ax1.margins(0)
            ax1.set_xlabel('')
            leg = ax1.legend(title=None, bbox_to_anchor=(1.05, 1))
            leg.set_visible(False) if not legend else None
            #cashflow = False
    
        # plot cashflow
        # you might use df_all to plot cashflow if transaction vlines not necessary 
        if cashflow:
            # plot vline for transaction dates
            dates_trs = func(df_rec).index.get_level_values(0).unique()
            ax1.vlines(dates_trs, 0, 1, transform=ax1.get_xaxis_transform(), lw=0.5, color='gray')
            # set slice for record with a single transaction
            ax2 = self.plot_cashflow(df_rec=df_rec, start_date=start_date, end_date=end_date, 
                                     exclude_cost=exclude_cost, ax=ax2)
        return None


    def plot_cashflow(self, df_rec=None, start_date=None, end_date=None, exclude_cost=False,
                      ax=None, figsize=(8,2), alpha=0.4, colors=('r', 'g'),
                      labels=['Buy', 'Sell'], loc='upper left'):
        if df_rec is None:
            df_rec = self._check_result()
        if df_rec is None:
            return None

        df_cf = self._calc_cashflow_history(df_rec, exclude_cost=exclude_cost)
        df_cf = self._plot_cashflow_slice(df_cf, start_date, end_date)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        kw = lambda i: {'label':labels[i], 'color':colors[i]}
        _ = [self._plot_cashflow(ax, df_cf[x], end_date, **kw(i)) for i, x in enumerate(df_cf.columns)]
        ax.legend(loc=loc)
        return ax  

    
    def plot_assets(self, date=None, roi=True, sort_by='roi', figsize=None, label=True, 
                    exclude_cost=False, exclude_sold=True):
        """
        exclude_sold: set to False to include all historical assets
        """
        df_val = self.valuate(date=date, total=False, int_to_str=False, 
                              exclude_cost=exclude_cost, exclude_sold=exclude_sold)
        if df_val is None:
            return None
        df_val = df_val.sort_values(sort_by, ascending=True) if sort_by in df_val.columns else df_val
        _= PortfolioBuilder._plot_assets(df_val, roi=roi, figsize=figsize, label=label)
        return df_val


    def performance_stats(self, date=None, metrics=METRICS2, sort_by=None, exclude_cost=False):
        """
        calc performance stats of a portfolio with 2 different methods
        date: str for date for fixed weights of simulated performance
              None for the latest date
              int for index to slice transaction dates
        """
        col_date = self.cols_record['date']
        col_val = 'value'
        col_roi = 'roi'
        
        # Portfolio value from actual returns
        df_all = self.valuate(date='all', total=True, exclude_cost=exclude_cost)
        if df_all is None:
            return None
        
        df_val = df_all[col_roi]
        # portfolio returns from cumulative roi
        df_val = (1 + df_val) / (1 + df_val.shift(1)) - 1
        # portfolio values from returns
        df_res = df_val.apply(lambda x: (1 + x)).cumprod().dropna()
    
        # get price history of assets
        df_rec = self._check_result()
        df_prices = self._update_universe(df_rec, msg=True, download_missing=True)
        if df_prices is None:
            return None
    
        if isinstance(date, int): # date is index to slice transaction dates
            dates = df_rec.index.get_level_values(col_date).unique().sort_values(ascending=True)
            if date > 0:
                dates = dates[:date]
            elif date < 0:
                dates = dates[date:]
            else:
                pass
        elif date is None:
            dates = [df_all.index.get_level_values(col_date).max()]
        else: # date is string
            dates = [datetime.strptime(date, self.date_format)]
    
        df_sim = None
        # Portfolio value by assuming the end-date weights were held from the start
        for date in dates:
            df_all = self.valuate(date=date, total=False, int_to_str=False, exclude_cost=exclude_cost)
            if df_all is None:
                return None
            df_val = df_prices.loc[:date, df_all.index].dropna(how='all')
            date = date.strftime('%y%m%d') # cast to str for column name
            df_val = (df_all[col_val].div(df_all[col_val].sum()) # weights
                     .div(df_val.iloc[0]) # unit price of each asset
                     .mul(df_val).sum(axis=1) # total value
                     .rename(f'Simulated ({date})'))
            df_sim = df_val if df_sim is None else pd.concat([df_sim, df_val], axis=1)
    
        df_res = df_res.to_frame('Realized').join(df_sim, how='outer')
        return performance_stats(df_res, metrics=metrics, sort_by=sort_by, align_period=False)


    def diversification_history(self, start_date=None, end_date=None, 
                                metrics=None, exclude_cost=False, 
                                plot=True, figsize=(8,4), ylim=None):
        """
        Compute history of three key diversification metrics for a portfolio:
        - Diversification Ratio (DR)
        - HHI-based Diversification Score
        - Effective Number of Bets (ENB)
        start_date: None for history since rebalance (end_date ignored)
        """
        df_val = self.valuate(date='all', total=False, exclude_cost=exclude_cost)
        if df_val is None:
            return None
        else:  # get latest record to update price history
            # df_rec is not None if df_val is not None
            df_rec = self._check_result()
   
        col_tkr = self.cols_record['tkr']
        col_date = self.cols_record['date']
        col_val = 'value'
        col_roi = 'roi'
    
        # history since rebalance
        dt = df_val.index.get_level_values(col_date).max()
        if (start_date is not None) and datetime.strptime(start_date, self.date_format) >= dt:
            start_date = None
        if start_date is None:
            df = df_val.loc[dt]
            tickers = df.loc[df[col_val] > 0].index
            idx = pd.IndexSlice
            df_val = df_val.loc[idx[:, tickers], :]
        else:
            df_val = df_val.loc[start_date:end_date]
    
        # check num of assets
        if df_val.index.get_level_values(col_tkr).nunique() < 2:
            return None
        
        # get weight history
        df_wgt = df_val[col_val].unstack(col_tkr)
        df_wgt = df_wgt if start_date else df_wgt.dropna()
        df_wgt = (df_wgt.replace(0, None) # replace to None for next apply
                  .apply(lambda x: x.dropna() / sum(x.dropna()), axis=1))
    
        # get asset returns
        df_ret = self._update_universe(df_rec, download_missing=True)
        df_ret = df_ret.pct_change()
        
        # check metrics
        options = ['HHI', 'DR', 'ENB']
        metrics = [metrics] if isinstance(metrics, str) else metrics
        metrics = [x.upper() for x in metrics] if metrics else options
        if len(set(options) - set(metrics)) == len(options):
            return print('ERROR')
        else:
            dates = df_wgt.index
            df_div = pd.DataFrame(index=dates)
    
        # calc metrics history
        if 'HHI' in metrics:
            df_div['HHI'] = df_wgt.apply(lambda x: diversification_score(x.dropna()), axis=1)
    
        for mtr, func in zip(options[1:], [diversification_ratio, effective_number_of_risk_bets]):
            if mtr in metrics:
                res = []
                for dt in dates:
                    sr_tkr = df_wgt.loc[dt].dropna()
                    ret = df_ret.loc[:dt, sr_tkr.index]
                    x = func(sr_tkr.to_list(), ret)
                    res.append(x)
                df_div[mtr] = pd.Series(res, index=dates)
    
        if plot:
            # add portfolio value plot
            df_ttl = self.valuate(date='all', total=True, exclude_cost=exclude_cost)
            start, end = get_date_minmax(df_div)
            df_ttl = df_ttl.loc[start:end, col_roi].mul(100)
            ax = df_ttl.plot(label='ROI', figsize=figsize, color='grey', ls='--', lw=1)
            # plot metrics
            axt = ax.twinx()
            _ = df_div.plot(ax=axt, title='Portfolio Diversification')
            if ylim is None:
                ylim = (df_div.min().min()*0.9, df_div.max().max()*1.1)
            axt.set_ylim(ylim)
            ax.set_ylabel('Return On Investment (%)')
            axt.set_ylabel('Diversification')
            axt.grid(axis='y', alpha=0.3)
            ax.margins(x=0)
            ax.set_xlabel('')
            _ = set_matplotlib_twins(ax, axt, legend=True, loc='upper left')
            return None
        else:
            return df_div

    
    def check_new_transaction(self, date=None, msg=True):
        """
        check if new transaction date is after the last date of transactions 
        """
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
        """
        DO NOT use _check_record for df_rec as saving nshare allowed for convenience
        df_rec: all transaction record w/ tickers of halt
        """
        file, path = self.file, self.path
        self.file = self._save_transaction(df_rec, file, path)
        if self.file is not None:
            self.record = self.import_record(df_rec, return_on_fail=True)
            self.record_halt = None if self.tradinghalts is None else self.tradinghalts.record_halt
        return df_rec
        

    def update_record(self, security_names=None, save=True, update_var=True):
        """
        update amount-based & ticker names with the saved record
        save: overwrite record file if True
        """
        # reload record w/ full transaction history first
        record = self.import_record(halt=False, msg=False, return_on_fail=False)
        if record is None:
            return None
        else:
            df_rec = record.copy()

        # update col_rat and convert record from num of shares to amount
        df_prc = self._update_universe(df_rec, msg=False)
        df_rec = self._update_price_ratio(df_rec, df_prc)
        df_rec = self._convert_to_amount(df_rec, df_prc)
        
        # update ticker name
        df_rec = self._update_ticker_name(df_rec, security_names)
        # save or remind        
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
            df_rec = self._check_result(msg, nshares=int_nshares)
        if df_rec is None: # record is None or nshares-based to edit
            if self.tradinghalts is None:
                return None
            else:
                return self.tradinghalts.recover(self.record, self.record_halt) # see _check_result for err msg

        if weight_actual:# add actual weights
            df_rec = self.insert_weight_actual(df_rec)
        
        if nshares or value:
            df_prc = self._update_universe(df_rec, msg=False)

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
            
        #df_rec.loc[:, cols_int] = df_rec.loc[:, cols_int].astype(int)
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


    def insert_weight_actual(self, df_rec, decimals=3):
        cols_record = self.cols_record
        col_net = cols_record['net']
        col_wgt = cols_record['wgt']
        col_wgta = 'weight*'
        cols = df_rec.columns
        i = cols.get_loc(col_wgt)
        return (df_rec.join(self._calc_weight_actual(df_rec[col_net], decimals=decimals))
                .loc[:, cols.insert(i+1, col_wgta)])


    def check_weights(self, *args, **kwargs):
        """
        return equal weights if weights is str or list
        """
        return BacktestManager.check_weights(*args, **kwargs)
        

    def check_additional(self, start_date=None, df_additional=None):
        """
        compare df_additional with df_universe from start_date to latest
        """
        df_add = self._check_var(df_additional, self.df_additional)
        if df_add is None:
            return print('ERROR: no df_additional to check')
        else:
            df_add = df_add.loc[start_date:]
    
        # get prices
        df_rec = self._check_result(False) # no err msg as df_rec is just for df_prc
        if df_rec is None:
            df_prc = self.df_universe
        else:
            df_prc = self._update_universe(df_rec, msg=False)
        df_prc = df_prc.loc[start_date:]
    
        # check dates
        dates = df_prc.index.difference(df_add.index)
        n = dates.size
        if n > 0:
            print(f'WARNING: Missing {n} dates in the additional data')
            result = (df_prc, df_add)
            msg = 'Returning price and additional'
        else:
            result = None
    
        # check tickers
        tickers = df_prc.columns.difference(df_add.columns)
        n = tickers.size
        if n > 0:
            print(f'WARNING: Missing {n} tickers in the additional data')
            if result is None: 
                result = tickers.to_list()
                msg = 'Returning missing tickers'
    
        print(msg) if result is not None else None
        return result


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


    @staticmethod
    def get_title_pnl(roi, ugl, date, **kwargs):
        title = f'ROI: {roi:.1%}, UGL: {format_price(ugl, **kwargs)}'
        return f"{title} ({date})"

    
    @staticmethod
    def _plot_assets(df_val, roi=True, figsize=None, label=True,
                     col_name='name', col_value='value', col_roi='roi', cpl_ugl='ugl',
                     height_bar=0.3, height_padding=0):
        """
        Bar chart displaying the performance of individual assets within the portfolio
        df_val: output of self.valuate(date=None, total=False, int_to_str=False)
        """
        kw = dict(kind='barh', legend=False)
        fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=figsize)
        _ = df_val.plot(col_name, col_value, ax=ax1, title='Value', **kw)
        if roi:
            _ = (df_val.assign(roi=df_val[col_roi].mul(100))
                 .plot(col_name, col_roi, ax=ax2, color='orange', title='ROI(%)', **kw))
        else:
            _ = df_val.plot(col_name, cpl_ugl, ax=ax2, color='orange', title='UGL', **kw)
        _ = ax1.set_ylabel(None)
        _ = ax2.axvline(0, lw=0.5, c='gray')

        # Update the figure height
        n_bars = df_val[col_name].nunique()
        height_tmp = n_bars * height_bar + height_padding
        width, height = fig.get_size_inches()
        if height_tmp > height:
            fig.set_size_inches(width, height_tmp)
    
        if label:
            _= ax1.bar_label(ax1.containers[0], label_type='center', fmt='{:,.3g}')
            _= ax2.bar_label(ax2.containers[0], label_type='center', fmt='{:.1f}' if roi else '{:,g}')
        
        plt.subplots_adjust(wspace=0.05)
        return (ax1, ax2)


    def util_plot_additional(self, tickers=None, start_date=None, **kwargs):
        """
        tickers: set to None to plot additional data of held assets
        kwargs: kwargs for plot
        """
        df_add = self.df_additional
        if df_add is None:
            return print('ERROR: no df_additional to check')
        
        col_start = 'start'
        if isinstance(tickers, str):
            tickers = [tickers]
    
        if tickers is None: # retrieve tickers of held assets
            df = self.valuate(total=False)
            if df is None:
                return print('ERROR: Set tickers to plot')
            else:
                tickers = df.index
            if start_date is None:
                start_date = df[col_start].min()
    
        tickers_add = df_add.columns.intersection(tickers)
        if tickers_add.size == 0:
            return print('ERROR')
    
        m = pd.Index(tickers).difference(df_add.columns)
        if m.size > 0:
            print(f'WARNING: {m.size} tickers not in additional data')

        df_add = df_add.loc[start_date:, tickers_add]
        if len(df_add) > 1:
            title = 'Additional data of Portfolio assets'
            return df_add.plot(**{'title':title, **kwargs})
        else: # no plot if there's just one date for data
            print('REMINDER: More additional data required to generate the plot.')
            return df_add


    def util_get_prices(self, tickers, update_security_names=True):
        """
        util to get price history of additional tickers
        """
        if update_security_names:
            try:
                ticker_names = DataManager.get_tickers_krx('krx')
                ticker_names = {k:v for k,v in ticker_names.items() if k in tickers}
                self.security_names.update(ticker_names)
                print('security_names updated')
            except Exception as e:
                print(f'ERROR: Failed to update security names as {e}')
    
        start, end = get_date_minmax(self.df_universe, self.date_format)
        return DataManager.download_fdr(tickers, start, end)


    def util_check_entry_turnover(self, date=None):
        """
        Calculate the entry turnover ratio of the portfolio over time.
        """
        df_rec = self._check_result()
        if df_rec is None:
            return None # see msg from view_record
    
        df = df_rec.loc[date:]
        if len(df) > 0:
            df_rec = df
    
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_trs = cols_record['trs']
        col_net = cols_record['net']
        
        return df_rec.groupby(col_date).apply(
            lambda x: pd.Series({
                'New': sum(x[col_net] == x[col_trs]),
                'Total': sum(x[col_net] > 0),
                'Ratio': round(sum(x[col_net] == x[col_trs])/sum(x[col_net] > 0), 3)
            })
        )
    

    def _update_universe(self, df_rec, msg=False, download_missing=False):
        """
        update universe with missing tickers
        df_rec: transaction record with amount
        download_missing: set to True to download missing tickers in the universe
        """
        df_prices = self.df_universe.copy()
        col_tkr = self.cols_record['tkr']
        
        # tickers not in the universe
        tkr_m = df_rec.index.get_level_values(col_tkr).unique().difference(df_prices.columns)
        # guess close price from transaction history
        if tkr_m.size > 0:
            if download_missing: # download prices of missing tickers
                # check saved before downloading
                df_m = self.df_prices_missing
                if df_m is None or tkr_m.difference(df_m.columns).size > 0:
                    try:
                        df_m = self.util_get_prices(tkr_m)
                    except:
                        tkrs = ', '.join(tkr_m)
                        return print(f'ERROR: Failed to download {tkrs}')
                    self.df_prices_missing = df_m
                    print_list(tkr_m, 'Data of tickers {} downloaded')
                df_prices = pd.concat([df_prices, df_m], axis=1)
            else: # guess prices of missing tickers
                idx = pd.IndexSlice
                df_m = df_rec.loc[idx[:, tkr_m], :]
                df_m = self._calc_price_from_transactions(df_m, price_start=1000)
                df_m = df_m.unstack(col_tkr)
                df_prices = pd.concat([df_prices, df_m], axis=1)
                df_prices[tkr_m] = df_prices[tkr_m].ffill()
            print_list(tkr_m.to_list(), 'Tickers {} added to universe') if msg else None
        return df_prices


    def check_universe(self, all_transaction=True, msg=False):
        """
        check if assets in record missing in universe. 
         ex) delisted security
        all_transaction: set to True to track exact value histories of sold assets delisted from universe
        """
        # no err msg as df_rec is just for date & tickers
        df_rec = self._check_result(False)
        if df_rec is None:
            return None
    
        df_prices = self.df_universe
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
    
        if all_transaction:
            date = None
        else:
            date = df_rec.index.get_level_values(col_date).max()
        tickers = df_rec.loc[date:].index.get_level_values(col_tkr).unique().difference(df_prices.columns)
        n = tickers.size
        if n > 0:
            print(f'WARNING: Missing {n} assets in the universe')
            print('Run check_universe to get the list of missing assets') if msg else None
            return tickers.to_list()
            

    def _calc_periodic_value(self, df_rec, df_prices, date=None, msg=False,
                             col_val='value', col_end='end'):
        """
        get record of transactions with values by asset 
         which is for CostManager._calc_fee_annual
        df_rec: error if value column
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
        if df_val is None:
            return
        
        sr_prc = df_val[col_prc] # buy/sell price
        df_val = df_val[col_net].to_frame(col_nshares) 
        # transaction date to calc period for fee calc
        df_val[col_stt] = df_val.index.get_level_values(col_date) 
        # get end date before next transaction
        date = df_prices.index.max() if date is None else date
        if df_val.index.get_level_values(col_tkr).nunique() > 1:
            df_val[col_end] = (df_val.groupby(col_tkr, group_keys=False)
                              .apply(lambda x: x[col_stt].shift(-1)).fillna(date))
        else: # for single asset portfolio
            df_val[col_end] = df_val[col_stt].shift(-1).fillna(date)
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
        df_rec.loc[:, cols_int] = df_rec[cols_int].mul(df_rec[col_prc], axis=0)
        df_rec[cols_int] = df_rec[cols_int].astype(int) # ensure casting works by not using loc?
        df_rec.loc[:, col_prc] = None # set col_prc to None as flag
        return df_rec


    def _get_trading_price(self, df_rec, df_universe):
        """
        calc buy/sell price from ratio and close price for transaction record
        """
        cols_record = self.cols_record
        col_rat = cols_record['rat']
        col_prc = cols_record['prc']
        
        sr_rat = df_rec[col_rat]
        if sr_rat.isna().any(): # all col_rat must be filled to get trading price
            return print('ERROR: Missing ratio\'s exist')
            
        # stack df_universe to div by df_rec
        idx = sr_rat.index.names
        sr_close = df_universe.stack().rename_axis(idx)
        
        # return buy/sell pirces for transactions
        return sr_close.div(sr_rat).rename(col_prc).dropna(axis=0)


    def _calc_price_from_transactions(self, df_rec, price_start=1000):
        """
        guess close price history from df_rec w/o price data
        """
        cols_record = self.cols_record
        col_tkr = cols_record['tkr']
        col_rat = cols_record['rat']
        col_trs = cols_record['trs']
        col_net = cols_record['net']
        
        def func(df_tkr, price_start=price_start):
            sr_prc = pd.Series(index=df_tkr.index, dtype=float)  # Initialize output series
            sr_prc.iloc[0] = price_start
            sr_trs = df_tkr[col_rat] * df_tkr[col_trs]
            sr_net = df_tkr[col_rat] *  df_tkr[col_net]
            sum_t_p = 0  # This keeps track of the summation term
            for i in range(1, len(df_tkr)):  # Start from i=2 (index 1 in pandas)
                sum_t_p += sr_trs.iloc[i-1] / sr_prc.iloc[i-1]  # Accumulate sum from previous terms
                if sum_t_p == 0:
                    sr_prc.iloc[i] = None
                else:
                    sr_prc.iloc[i] = (sr_net.iloc[i] - sr_trs.iloc[i]) / sum_t_p 
            return sr_prc.ffill()
    
        return df_rec.groupby(col_tkr, group_keys=False).apply(lambda x: func(x)).sort_index()
        

    def _get_nshares(self, df_rec, df_universe, cols_record, 
                     int_nshares=False, add_price=True):
        """
        calc number of shares for amount net & transaction
        """
        cols = [cols_record[x] for x in ['prc','trs','net']]
        col_prc, col_trs, col_net = cols
        try:
            if df_rec[col_prc].notna().any():
                return print(f'ERROR: {col_prc} is not None')
        except KeyError:
            return print('ERROR: record has no price columns')
    
        # get buy/sell price to calc num of shares
        sr_prc = self._get_trading_price(df_rec, df_universe)
        if sr_prc is None:
            return None # see _get_trading_price for err msg
    
        df_nshares = df_rec[[col_trs, col_net]].div(sr_prc, axis=0)
        df_nshares = df_nshares.join(sr_prc) if add_price else df_nshares
        return df_nshares.map(np.rint).astype(int) if int_nshares else df_nshares


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
        index = [self.cols_record[x] for x in ['date','tkr']]
        sr_close = df_universe.stack().rename(col_close).rename_axis(index)
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


    def _calc_value_history(self, df_rec, df_universe, end_date=None, name=None, msg=False, 
                            total=True, exclude_cost=True):
        """
        calc historical of portfolio value from transaction
        end_date: calc value from 1st transaction of df_rec to end_date.
        name: name of output series
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_rat = cols_record['rat']
        col_net = cols_record['net']
        end = datetime.today() if end_date is None else end_date
        sr_ttl = None
        dates_trs = df_rec.index.get_level_values(0).unique()
        # get number of shares
        sr_nshares = self._get_nshares(df_rec, df_universe, cols_record, int_nshares=False)
        if sr_nshares is None:
            return
        else:
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
                        .apply(lambda x: x*n_tickers.loc[x.name])) # x.name: index name
            # concat histories        
            sr_ttl = sr_i if sr_ttl is None else pd.concat([sr_ttl, sr_i])
            end = start - pd.DateOffset(days=1)
        
        if sr_ttl is None:
            return print('ERROR: no historical')
        else:
            sr_ttl = sr_ttl.sort_index()

        if not exclude_cost:
            cost = self.cost or dict()
            kw = dict(period=3, percent=True, **cost)
            sr_ttl = CostManager.get_history_with_fee(sr_ttl, **kw)
        
        if total:
            sr_ttl = sr_ttl.fillna(0).sum(axis=1).astype(int)
        else:
            sr_ttl = sr_ttl.stack().dropna().astype(int)
        return sr_ttl if name is None else sr_ttl.rename(name)
        

    def _calc_cashflow_history(self, record, total=True, exclude_cost=True):
        """
        Returns df of cumulative buy and sell prices at each transaction.
        """
        # add value to record to calc year-fee
        df_rec = self.view_record(0, df_rec=record, nshares=False, value=True, 
                                  weight_actual=False, msg=False, int_nshares=False)
        cm = CostManager(df_rec, self.cols_record, self.date_format)
        cost = None if exclude_cost else self.cost
        return cm.calc_cashflow_history(cost=cost, total=total, percent=True)


    def _join_cashflow_by_ticker(self, sr_val, df_cf, df_pnl):
        """
        Merge cashflow history (based on transaction records) with value history (based on price changes).
        sr_val, df_cf, df_pnl: data by date & ticker
        """
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
        col_val = 'value'
        
        index = None
        dates_cf = df_cf.index.get_level_values(col_date).unique()
        end = sr_val.index.get_level_values(col_date).max()
        for start in dates_cf.sort_values(ascending=False):
            dts = sr_val.loc[start:end].index.get_level_values(col_date).unique()
            if dts.size == 0: # end date is smaller tha start
                continue
            tkrs = df_cf.loc[start].index
            idx = pd.MultiIndex.from_product([dts, tkrs])
            index = idx if index is None else index.append(idx)
            end = start - pd.DateOffset(days=1)
       
        return (pd.DataFrame(index=index).join(df_cf).join(sr_val.rename(col_val))
                .join(df_pnl).groupby(col_tkr).ffill()
                # DO NOT drop col_val of None to include the cashflow of all tickers in history
                #.dropna(subset=col_val) 
                .sort_index())


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


    def _calc_profit(self, sr_val, df_cashflow_history, result='ROI', roi_log=False,
                     col_val='value', col_sell='sell', col_buy='buy'):
        """
        calc history of roi or unrealized gain/loss
        sr_val: output of _calc_value_history. supporting total or by tickers
        df_cashflow_history: output of _calc_cashflow_history. supporting total or by tickers
        result: ROI, UGL or 'all'
        """
        col_tkr = self.cols_record['tkr']
        
        conds = [col_tkr in df.index.names for df in [sr_val, df_cashflow_history]]
        if sum(conds) == 1: # both of sr_val & df_cashflow_history be total or by tickers
            return print('ERROR') 
        else: 
            df_his = sr_val.to_frame(col_val).join(df_cashflow_history, how='outer')
        
        df_his = df_his.groupby(col_tkr) if sum(conds) == 2 else df_his
        df_his = df_his.ffill().fillna(0)
    
        return PortfolioBuilder.calc_profit(df_his, result=result, roi_log=roi_log,
                           col_val=col_val, col_sell=col_sell, col_buy=col_buy)

        
    @staticmethod
    def calc_profit(df_his, result='ROI', roi_log=False,
                    col_val='value', col_sell='sell', col_buy='buy'):
        """
        calc ROI/UGL from df_his of col_val, col_sell and col_buy
        """
        if pd.Index([col_buy, col_sell, col_val]).difference(df_his.columns).size > 0:
            return print("ERROR")
        
        col_roi = 'roi'
        col_ugl = 'ugl'
        
        # calc ROI
        ratio = lambda x: (x[col_val] + x[col_sell]) / x[col_buy]
        if roi_log:
            sr_roi = df_his.apply(lambda x: np.log(ratio(x)), axis=1)
        else:
            sr_roi = df_his.apply(lambda x: ratio(x) - 1, axis=1)
        sr_roi = sr_roi.rename(col_roi)
        
        # calc unrealized gain/loss
        sr_ugl = (df_his.apply(lambda x: x[col_val] + x[col_sell] - x[col_buy], axis=1)
                        .rename(col_ugl))
        
        result = result.lower()
        if result == col_roi:
            return sr_roi
        elif result == col_ugl:
            return sr_ugl
        else:
            return sr_ugl.to_frame().join(sr_roi)
        

    def _check_result(self, msg=True, check_record=True, nshares=False):
        if self.df_rec is None:
            if self.record is None:
                return print('ERROR: No transaction record') if msg else None
            else:
                df_res = self.record
        else:
            df_res = self.df_rec
        # check record of df_rec which must be not None now
        if check_record:
            if not self._check_record(df_res):
                return None

        col_prc = self.cols_record['prc']
        if df_res[col_prc].notna().any(): # print error regardless of the arg msg 
            # seems like record saved as nshares for editing
            print(f'ERROR: Run update_record first after editing record')
            if not nshares:
                return None # return None to terminate following process
        
        # self.df_rec or self.record could be modified if not copied
        return df_res.copy() 


    def _check_record(self, df_rec, msg=True):
        """
        check if df_rec follows transaction format
        """
        # check columns
        cols_record = self.cols_record
        cols = df_rec.columns.union(df_rec.index.names).difference(cols_record.values())
        if cols.size > 0:
            print('ERROR: Record is not default form') if msg else None
            return False

        # check if the assets of no transaction written as well with new transaction 
        col_date = cols_record['date']
        col_net = cols_record['net']
        dates = df_rec.index.get_level_values(col_date).unique() # all dates of transaction
        prv = dates[0]
        for date in dates[1:]:
            df_p = df_rec.loc[prv]
            tkrs = df_p.loc[df_p[col_net] > 0].index.difference(df_rec.loc[date].index)
            if tkrs.size > 0:
                dt = date.strftime(self.date_format)
                print(f'ERROR: Held assets missing in Transaction on {dt}') if msg else None
                return False
            else:
                prv = date
        return True
    

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
            #print(f'Transaction record {file} (~ {dt}) loaded')
            print(f'Transaction record {file} loaded')
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
        get price data for select or weigh
        """
        df_data = self.df_universe
        if date is not None:
            df_data = df_data.loc[:date]
        if tickers is not None:
            df_data = df_data[tickers]
            
        # setting tradinghalts
        if self.tradinghalts is not None:
            df_data = self.tradinghalts.set_price(df_data, self.record_halt)
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
            self.record_halt = self.tradinghalts.record_halt
            print(f'self.record updated')
        return None



class CostManager():
    def __init__(self, df_rec, cols_record, date_format='%Y-%m-%d'):
        """
        df_rec: transaction record from PortfolioBuilder
        cols_record: dict of var name to col name of df_rec
        """
        self.df_rec = df_rec
        self.cols_record = cols_record
        self.date_format = date_format
        
    
    def calc_cashflow_history(self, date=None, percent=True,
                              cost=dict(buy=0.00363960, sell=0.00363960, tax=0.18, fee=0),
                              total=True):
        """
        calc net buy & sell prices at each transaction date
        cost: dict of cost items
        total: set to False to get histories of individual tickers
        """
        df_rec = self.df_rec
        if df_rec is None:
            return None
        
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
        col_buy, col_sell, col_tax, col_fee = 'buy', 'sell', 'tax', 'fee'
        col_bcost = 'cost_buy'
        col_scost = 'cost_sell'
    
        # force to total to calc df_net
        df_cf = CostManager._calc_cashflow_history(df_rec, cols_record, total=False,
                                                   col_buy=col_buy, col_sell=col_sell)
        df_cf = df_cf.loc[:date]
        if cost is None: # cashflow without cost
            return df_cf.groupby(col_date).sum() if total else df_cf
    
        # calc cost
        df_cost = self.calc_cost(date=date, percent=percent, **cost)
        df_cost = df_cost.rename(columns={col_buy:col_bcost})
        df_cost[col_scost] = df_cost[col_sell] + df_cost[col_fee] + df_cost[col_tax]
        df_cost = df_cost[[col_bcost, col_scost]]
        df_cost = df_cost.unstack(col_tkr).cumsum().ffill().stack()
        
        # calc net cashflow
        df_net = (df_cf.join(df_cost, how='outer')
                  # ffill cost for the date of no new transaction from halt
                  .groupby(col_tkr).ffill()) 
        df_net[col_buy]  = df_net[col_buy] - df_net[col_bcost]
        df_net[col_sell]  = df_net[col_sell] - df_net[col_scost]
        df_net = df_net[[col_buy, col_sell]]
    
        if total: 
            df_net = df_net.groupby(col_date).sum()
    
        return df_net


    def calc_cost(self, date=None, buy=0, sell=0, tax=0, fee=0, percent=True):
        """
        buy, sell, tax, fee: float, series or dict of ticker to cost
        """
        df_rec = self.df_rec
        if df_rec is None:
            return None
        else:    
            cols_record = self.cols_record
            date = datetime.today().strftime(self.date_format) if date is None else date
            df_rec = df_rec.loc[:date]

        m = 0.01 if percent else 1
        # convert dict to series to multiply m
        buy, sell, tax, fee = [pd.Series(x) if isinstance(x, dict) else x for x in [buy, sell, tax, fee]]
        sr_buy = CostManager._calc_fee_trading(df_rec, cols_record, m*buy, transaction='buy')
        sr_sell = CostManager._calc_fee_trading(df_rec, cols_record, m*sell, transaction='sell')
        sr_tax = CostManager._calc_fee_trading(df_rec, cols_record, m*tax, transaction='tax')
        sr_fee = CostManager._calc_fee_annual(df_rec, cols_record, m*fee, date)
        return (sr_buy.to_frame().join(sr_sell, how='outer')
                .join(sr_fee, how='outer').join(sr_tax, how='outer')
                .fillna(0))


    def calc_fee_trading(self, commission, date=None, transaction='all', percent=True):
        """
        wrapper for CostManager._calc_fee_trading
        """
        df_rec = self.df_rec
        if df_rec is None:
            return None
        else:
            cols_record = self.cols_record
            df_rec = df_rec.loc[:date]

        commission = pd.Series(commission) if isinstance(x, dict) else commission
        commission = commission * (0.01 if percent else 1)
        return CostManager._calc_fee_trading(df_rec, cols_record, commission, transaction=transaction)


    def calc_tax(self, tax, date=None, percent=True):
        """
        calc_fee_trading with transaction 'tax'
        """
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
    def _calc_cashflow_history(df_rec, cols_record, col_buy='buy', col_sell='sell', total=True):
        """
        Returns df of cumulative buy and sell prices at each transaction.
        total: set to False to get histories of individual tickers
        """
        col_trs = cols_record['trs']
        col_tkr = cols_record['tkr']
        col_date = cols_record['date']
        col_keep = 'keep' # for the case of date of no transaction except for halt
        col_cf = 'cf'
    
        sr_tr = df_rec.apply(lambda x: col_buy if x[col_trs]>0 else 
                             (col_sell if x[col_trs]<0 else col_keep), axis=1)
        df_cf = (df_rec.assign(**{col_cf: sr_tr})
                       .pivot(columns=col_cf, values=col_trs)
                       .drop(col_keep, axis=1, errors='ignore')
                       .abs().sort_index())
        
        if col_sell not in df_cf.columns:
            df_cf[col_sell] = 0
        
        if total: 
            return df_cf.groupby(col_date).sum().cumsum() 
        else:
            return df_cf.unstack(col_tkr).cumsum().ffill().stack().fillna(0)


    @staticmethod
    def _calc_fee_trading(df_rec, cols_record, fee, transaction='all'):
        """
        calc trading fee
        fee: sell/buy commissions. rate of float or seires or dict
        transaction: 'all', 'buy', 'sell', 'tax'
        """
        col_tkr = cols_record['tkr']
        col_trs = cols_record['trs']
        
        sr_val = df_rec[col_trs]
        # now sr_fee series or float
        sr_fee = pd.Series(fee) if isinstance(fee, dict) else fee
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
    def _calc_fee_annual(df_rec, cols_record, fee, date, name='fee',
                         col_val='value', col_end='end'):
        """
        calc annual fee
        df_rec: should have col_val & col_end. see _calc_periodic_value
        fee: number, dict or series of ticker to annual fee. rate
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
        
        if isinstance(fee, dict):
            sr_fee = pd.Series(fee)
        elif isinstance(fee, Number):
            sr_fee = pd.Series(fee, index=df_rec.index.get_level_values(col_tkr).unique())
        else: # fee is series
            sr_fee = fee
        sr_fee = sr_fee.rename_axis(col_tkr).rename(name)
    
        df_val[col_rate] = (df_val.join(sr_fee)
                            # year fee converted to fee for period of x[col_prd] days
                           .apply(lambda x: -1 + (1 + x[name]) ** (x[col_prd]/365), axis=1)
                           .fillna(0)) # fillna for missing tickers in sr_fee
        return (df_val.apply(lambda x: x[col_val] * x[col_rate], axis=1) # amount of fee for period
                .rename(name).swaplevel().sort_index())


    @staticmethod
    def get_history_with_fee(df_val, buy=0, sell=0, tax=0, fee=0, period=3, percent=True):
        """
        df_val: history of single value or price to apply fee. ex) DataManager.df_prices
        buy: float, series or dict of ticker to cost to buy
        sell: float, series or dict of ticker to cost to sell
        tax: float, series or dict of ticker to tax
        fee: float, series or dict of ticker to annual fee
        period: add fee every period of months
        """
        # calc fee every period
        def calc_fee(df, sr_fee, period=period):
            sr_fee = sr_fee.apply(lambda x: -1 + (1+x)**(period/12)) # get equivalent rate of fee for period
            days = check_days_in_year(df, msg=False) # get days fo a year
            days = days.mul(period/12).round().astype(int) # get dats for a period
            return df.apply(lambda x: x.dropna().iloc[::days[x.name]] * sr_fee[x.name] 
                            if x.count() >= period else 0).fillna(0)

        # convert cost data to series
        def convert_to_series(cost, percent=percent):
            if isinstance(cost, dict):
                cost = pd.Series(cost)
            elif isinstance(cost, Number):
                cost = pd.Series(cost, index=df_val.columns)
            n = df_val.columns.difference(cost.index).size
            if n > 0:
                return print(f'ERROR: Missing cost data for {n} tickers')
            else:
                return cost/100 if percent else cost

        converted = []
        for x in [buy, sell, tax, fee]:
            c = convert_to_series(x)
            if c is None:
                return df_val
            else:
                converted.append(c)
        else:
            fee = converted[-1]
            cost = sum(converted[:-1])
        
        # calc buy + sell + tax
        df_cost = df_val.apply(lambda x: x * cost[x.name])
        # calc annual fee
        df_fee = df_val.copy()
        df_fee.loc[:,:] = None
        df_fee.update(calc_fee(df_val, fee)) # get fee for every period
        df_fee = df_fee.fillna(0).cumsum() # get history of fees
        # sub total cost from value history
        return df_val.sub(df_cost).sub(df_fee)


    @staticmethod
    def get_value_after_cost(df_val):
        pass # testing

    
    @staticmethod
    def load_cost(file, path='.', col_universe='universe', col_ticker='ticker'):
        """
        load cost data of strategy, universe & ticker
        """
        try:
            file = get_file_latest(file, path)
            df_cost = pd.read_csv(f'{path}/{file}', dtype={col_ticker:str}, comment='#')
        except FileNotFoundError:
            return print('ERROR: Failed to load')
    
        if df_cost[[col_universe, col_ticker]].duplicated().any():
            return print('ERROR: Check cost data for duplicates')
        else:
            print(f'Cost data {file} loaded')
            return df_cost

    
    @staticmethod
    def get_cost(universe, file, path='.', 
                 cols_cost=['buy', 'sell', 'fee', 'tax'],
                 col_uv='universe', col_ticker='ticker', universes=UNIVERSES):
        """
        load cost file and get dict of commission for the universe
        """
        df_kw = CostManager.load_cost(file, path)
        if df_kw is None:
            #return print('ERROR: Load cost file first')
            return None
        # get universe name for cost data
        universe = PortfolioData.get_universe(universe, universes=universes) if universes else universe
        df_kw = df_kw.loc[df_kw[col_uv] == universe]
        if (len(df_kw) == 1) and df_kw[col_ticker].isna().all(): # same cost for all tickers
            return df_kw[cols_cost].to_dict('records')[0]
        elif len(df_kw) > 1: # cost items are series of ticker to cost
            return df_kw.set_index(col_ticker)[cols_cost].to_dict('series')
        else:
            return print('WARNING: No cost data available')


    @staticmethod
    def check_cost(file, path='.', universe=None,
                   col_uv='universe', col_ticker='ticker', universes=UNIVERSES):
        """
        check cost file and cost data for the universe
        """
        # load cost file
        df_cst = CostManager.load_cost(file, path)
        if df_cst is None:
            return None
    
        # check cost data for given univese
        if universe:
            # get universe name in cost data
            univ = PortfolioData.get_universe(universe, universes=universes) if universes else universe
            # check if universe in cost data
            df_cst_uv = df_cst.loc[df_cst[col_uv] == univ]
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


    @staticmethod
    def save_cost(df_cost, file, path='.'):
        save_dataframe(df_cost, file, path, 
                      msg_succeed=f'Cost data saved to {file}',
                      index=False)
        return None



class TradingHalts():
    """
    split transaction record into record of trading stocks and record of halt
    """
    def __init__(self, record=None, cols_record=None, prefix_halt='HLT_'):
        """
        record: PortfolioBuilder.record
                record is None if first transaction of a portfolio
        cols_record: PortfolioBuilder.cols_record
        """
        self.record = record
        self.cols_record = cols_record
        self.prefix_halt = prefix_halt
        # record of halt before new transaction, not updated except by set
        self.record_halt = None 
        self.date_lt = None
        # set record, record_halt and date_lt
        self.set()


    def set(self):
        """
        init record, record_halt and date_lt by moving assets of halt from record to record_halt
            which is necessary even for jobs before transaction. ex) valuate
        """
        record = self.record
        if record is None:
            return None
        col_date = self.cols_record['date']
        date_lt = record.index.get_level_values(col_date).max()
        
        # create record_halt from record
        tickers, record_halt = None, None
        record, record_halt = self._set_to_halt(tickers, record, record_halt)

        self.record = record
        self.record_halt = record_halt
        self.date_lt = date_lt
        return None


    def transaction(self, date, buy=None, sell=None, resume=None, halt=None, 
                    date_actual=None, values_on_date=None, date_format='%Y-%m-%d'):
        """
        make new transaction from the latest transaction without price data
        buy: dict of tickers to total buy price
        halt: list of tickers to halt
        sell: dict of tickers to total sell price or list 
        resume: dict, list, 'all' 
        values_on_date: dict/series of asset to value on the new transaction date
        """
        record = None if self.record is None else self.record.copy()
        if record is None:
            return print('REMINDER: No record to start with')
        else:
            record_halt = None if self.record_halt is None else self.record_halt.copy()
    
        date_lt = self.date_lt
        cols_record = self.cols_record
        col_date = cols_record['date']
        col_tkr = cols_record['tkr']
        col_trs = cols_record['trs']
        col_net = cols_record['net']
        col_rat = cols_record['rat']
        col_dttr = cols_record['dttr']
        idx = pd.IndexSlice
        print_reminder = lambda x: print(f'REMINDER: For the {x} price, use the total amount, not the unit price.')
    
        # set transaction date
        date = pd.to_datetime(date)
        if date <= date_lt:
            dt1, dt2 = [x.strftime(date_format) for x in [date, date_lt]]
            return print(f'ERROR: set date ({dt1}) after the latest transaction date {dt2}')
    
        # move tickers to sell first to record if in record_halt 
        if sell is not None: 
            tickers = sell.keys() if isinstance(sell, dict) else sell
            record, record_halt = self._free_halt(tickers, record, record_halt, msg=False)
    
        # update record for resume
        if resume is not None:
            if resume == 'all':
                tickers = None # resume all in halt
                resume = None # set to None to skip later net update
            else:
                if isinstance(resume, list):
                    tickers = resume
                    resume = None # set to None to skip later update
                else:
                    tickers = resume.keys()
            record, record_halt = self._free_halt(tickers, record, record_halt)
    
        # update record for halt
        if halt is not None:
            record, record_halt = self._set_to_halt(halt, record, record_halt)
        
        # copy record of date_lt to date
        # cast date_actual of new transaction to datetime like existing transactions 
        date_actual = date if date_actual is None else pd.to_datetime(date_actual)
        kw = {col_date:date, col_trs:0, col_rat:1, col_dttr:date_actual}
        record_date_lt = record.loc[date_lt, :]
        record_date = (record_date_lt.loc[record_date_lt[col_net] > 0].assign(**kw)
                           .set_index(col_date, append=True).reorder_levels([col_date, col_tkr]))
        
        # update value of assets on the date
        if isinstance(values_on_date, dict):
            values_on_date = pd.Series(values_on_date)
        if isinstance(values_on_date, pd.Series):
            if record_date.index.get_level_values(col_tkr).difference(values_on_date.index).size > 0:
                print('WARNING: No update of net as missing assets in values_on_date')
            else:
                values_on_date = (values_on_date.rename_axis(col_tkr).to_frame(col_net)
                                  .assign(**{col_date:date}) # date is datetime
                                  .set_index(col_date, append=True).swaplevel())
                record_date.update(values_on_date)
        else:
            print(f'WARNING: No update of net on {date.strftime(date_format)}')
        record = pd.concat([record, record_date]).sort_index()
    
        if buy is not None:
            print_reminder('buy')
            # check if assets to buy in halted
            if record_halt is not None:
                tkr = record_halt.loc[date_lt].index.map(self.toggle_prefix).intersection(buy.keys())
                if tkr.size > 0:
                    return self._print_tickers(tkr, 'ERROR: Resume {} first to buy')

            # update existing assets 
            tkr = record.loc[date].index.intersection(buy.keys())
            if tkr.size > 0:
                sr_buy = pd.Series(buy).rename_axis(col_tkr)
                for x in [col_trs, col_net]:
                    record.loc[idx[date, tkr], x] += sr_buy
                # remove exising assets from buy
                buy = {k:v for k,v in buy.items() if k not in tkr}

            if len(buy) > 0: # add new assets
                index = pd.MultiIndex.from_product([[date], buy.keys()], names=[col_date, col_tkr])
                kw = {col_rat:1, col_dttr:date_actual}
                df_buy = (pd.DataFrame([buy, buy], index=[col_trs, col_net]).T
                          .set_index(index).assign(**kw))
                record = pd.concat([record, df_buy])

            record = record.sort_index()
    
        if sell is not None:
            tkr = sell.keys() if isinstance(sell, dict) else sell
            tkr = pd.Index(tkr).difference(record.loc[date].index)
            if tkr.size > 0:
                return self._print_tickers(tkr, 'ERROR: {} to sell not in the latest transaction')
            
            if isinstance(sell, list):
                # get sell price from net if not spec after the size check
                sell = {x: record.loc[idx[date, x], col_net] for x in sell}
            else: # assuming sell is dict
                print_reminder('sell')
            index = pd.MultiIndex.from_product([[date], sell.keys()], names=[col_date, col_tkr])
            kw = {col_net:0, col_rat:1, col_dttr:date_actual}
            df_sell = (pd.DataFrame(sell, index=[col_trs]).mul(-1).T
                       .set_index(index).assign(**kw))
            record.update(df_sell, overwrite=True)

        # update net of resumed if net given in resume dict
        if resume is not None: # resume is dict
            tkr = pd.Index(resume.keys()).difference(record.loc[date].index)
            if tkr.size > 0:
                return self._print_tickers(tkr, 'ERROR: {} to resume not in the latest transaction')
            else:
                index = pd.MultiIndex.from_product([[date], resume.keys()], names=[col_date, col_tkr])
                kw = {col_trs:0, col_rat:1, col_dttr:date_actual}
                df_resume = (pd.DataFrame(resume, index=[col_net]).T
                           .set_index(index).assign(**kw))
                record.update(df_resume, overwrite=True)

        dt = date.strftime(date_format)
        print(f'Updated with transaction on {dt}')
        return (record, record_halt)


    def recover(self, record, record_halt):
        """
        add to record the transaction of tickres to halt 
        record: record with new transaction
        """
        if record_halt is None:
            return record
        
        date_lt = self.date_lt
        cols_record = self.cols_record
        col_date = cols_record['date']
        # get new transaction date before concat
        date_nt = record.index.get_level_values(col_date).max()
        # concat record with halt in kept
        record = pd.concat([record, record_halt])
        if date_nt > date_lt:
            col_tkr = cols_record['tkr']
            col_name = cols_record['name']
            col_net = cols_record['net']
            col_trs = cols_record['trs']
            col_rat = cols_record['rat']
            col_dttr = cols_record['dttr']
            # copy record_halt on date_lt to new transaction date
            kw = {col_date:date_nt, col_trs:0, col_rat:1, col_dttr:date_nt}
            record_halt_new = (record_halt.loc[date_lt, [col_name, col_net]].assign(**kw)
                               .set_index(col_date, append=True).reorder_levels([col_date, col_tkr]))
            record = pd.concat([record, record_halt_new])
        return record.sort_index()
        

    def _set_to_halt(self, tickers, record, record_halt):
        """
        move transaction history of tickers to halt from record to record_halt
        """
        prefix_halt = self.prefix_halt
        date_lt = self.date_lt
        col_tkr, col_date, col_net = [self.cols_record[x] for x in ['tkr', 'date', 'net']]
        if tickers is None: # init record_halt
            if record_halt is None: # make sure no init before
                cond = record.index.get_level_values(col_tkr).str.startswith(prefix_halt)
                if sum(cond) > 0:
                    record_halt = record.loc[cond]
                    record = record.loc[~cond]
                    tickers = record_halt.index.get_level_values(col_tkr).unique()
                    self._print_tickers(tickers, 'Trading of assets {} to halt')
        else:
            # check if all tickers in the latest transaction
            tkr_u = self._check_latest(tickers, record)
            if tkr_u.size == 0:
                # check if tickers to halt in hold
                df = record.loc[date_lt].loc[tickers]
                tkr_u = df.loc[df[col_net] == 0].index
                if tkr_u.size == 0:
                    # get all transactions to halt
                    index_halt = self._get_halt(tickers, record, col_date, col_tkr, col_net)
                    record_halt_new = record.loc[index_halt]
                    # add prefix
                    record_halt_new.index = record_halt_new.index.map(lambda x: (x[0], self.toggle_prefix(x[1])))
                    if record_halt is None:
                        record_halt = record_halt_new
                    else:
                        record_halt = pd.concat([record_halt, record_halt_new]).sort_index()
                    record = record.drop(index_halt)
                    self._print_tickers(tickers, 'Trading of assets {} to halt')
            if tkr_u.size > 0:
                self._print_tickers(tkr_u, 'ERROR: Check {} to halt in the latest transaction')
        return (record, record_halt)


    def _free_halt(self, tickers, record, record_halt, msg=True):
        """
        set record & record_halt by moving tickers in record_halt to record after removing prefix
        msg: set to False to print no msg if preliminary before selling
        """
        if record_halt is None:
            return record, None # nothing to free
            
        prefix_halt = self.prefix_halt
        col_tkr = self.cols_record['tkr']
        # check assets to halt
        halted = record_halt.index.get_level_values(col_tkr).unique() 
        if tickers is None:
            freed = halted # free all in halted
        else:
            freed = [self.toggle_prefix(x) for x in tickers] # add prefix_halt
        dff = pd.Index(freed).difference(halted)
        if dff.size == 0:
            cond = record_halt.index.get_level_values(col_tkr).isin(freed)
            record_freed = record_halt.loc[cond]
            # remove prefix before concat
            record_freed.index = record_freed.index.map(lambda x: (x[0], self.toggle_prefix(x[1])))
            record = pd.concat([record, record_freed]).sort_index()
            record_halt = record_halt.loc[~cond]
            record_halt = record_halt if len(record_halt) > 0 else None
            rsm = [self.toggle_prefix(x) for x in freed] # remove prefix_halt
            self._print_tickers(rsm, 'Trading of assets {} resumed') if msg else None
        else:
            self._print_tickers(dff, 'ERROR: Assets {} not halted before') if msg else None
        return (record, record_halt)


    def _check_latest(self, tickers, record):
        """
        check tickers to sell or halt in the latest transaction
        """
        date_lt = self.date_lt
        col_date = self.cols_record['date']
        # return tickers not in the latest transaction
        return pd.Index(tickers).difference(record.loc[date_lt].index)


    def _get_halt(self, tickers, record, col_date, col_tkr, col_net):
        """
        get index of date & ticker of all transactions of tickers to halt
        """
        # Extract unique sorted dates
        dates = record.index.get_level_values(col_date).unique().sort_values()
        # Create a MultiIndex with all date-ticker combinations
        index = pd.MultiIndex.from_product([dates, tickers], names=record.index.names)
        # Initialize a Series with zeros and update with available values from `record`
        sr = pd.Series(0, index=index, name=col_net)
        idx = pd.IndexSlice
        sr.update(record.loc[idx[:, tickers], col_net])
        # Identify indices where the previous value is zero
        cond = (sr.groupby(col_tkr)
                # conditional for record with just one date 
                # ... else x > 0 to keep consistent multiindex
                .apply(lambda x: x*x.shift(-1).ffill() > 0 if len(x) > 1 else x > 0)
                .droplevel(0))
        return sr.loc[cond].index


    def set_price(self, df_prices, record_halt):
        """
        remove tickers of halt from df_prices for PortfolioBuilder.select or weigh process
        df_prices: df_universe for select
        """
        if record_halt is None:
            return df_prices

        col_tkr = self.cols_record['tkr']
        tickers = record_halt.index.get_level_values(col_tkr).unique()
        tickers = [self.toggle_prefix(x) for x in tickers]
        # set error to ignore for the case tickers be already delisted from universe
        return df_prices.drop(tickers, axis=1, errors='ignore')


    def _print_tickers(self, tickers, print_str='Trading of assets {} to halt'):
        tickers = [self.toggle_prefix(x, True) for x in tickers]
        return print_list(tickers, print_str)


    def toggle_prefix(self, ticker, remove_only=False):
        """
        Adds or removes self.prefix_halt from ticker
        remove_only: set to True to not add but to remove prefix_halt
        """
        prefix_halt = self.prefix_halt
        if ticker.startswith(prefix_halt) or remove_only:
            return ticker.removeprefix(prefix_halt)  
        else:
            return f"{prefix_halt}{ticker}"



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
        dfs: price data such as DataManager.df_prices
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
                return print_list(cols, 'ERROR: No {} in the data')
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
                return print_list(cols, 'ERROR: No {} in the data')
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
        # a simple risk parity that equalizes asset risk contributions only when correlations are ignored.
        elif cond(weigh, 'InvVol'): 
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
        df_cv = df_cv.sort_index()

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
                print(f'WARNING: No sorting as {e}')

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
                return print_list(tickers.keys(), 'ERROR: Set ticker or name from {}')
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
    def __init__(self, df_prices, days_in_year=252, metrics=METRICS, security_names=None):
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
        self.security_names = security_names


    @staticmethod
    def create(file, path='.', **kwargs):
        """
        create instance from sampled
        kwargs: kwargs of __init__
        """
        bayesian_data = BayesianEstimator.load(file, path)
        if bayesian_data is None:
            return None
        df_prices = bayesian_data['data']
        be = BayesianEstimator(df_prices, **kwargs)
        be.bayesian_data = bayesian_data
        return be
        

    def get_stats(self, metrics=None, sort_by=None, align_period=False, idx_dt=['start', 'end']):
        metrics = [metrics] if isinstance(metrics, str) else metrics
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
        

    def get_freq_days(self, freq='1Y'):
        """
        freq: str or int
        """
        if isinstance(freq, str):
            # split freq to int & unit
            n_t = BacktestManager.split_int_n_temporal(freq, 'M') # default month
        else: # return int regardless of unit
            return freq
        if n_t is None:
            return
        else:
            n, temporal = n_t        
            
        days_in_year = self.days_in_year
        cond = lambda x, y: False if x is None else x[0].lower() == y[0].lower()
        if cond(temporal, 'W'):
            n *= round(days_in_year / WEEKS_IN_YEAR)
        elif cond(temporal, 'M'):
            n *= round(days_in_year / 12)
        elif cond(temporal, 'Q'):
            n *= round(days_in_year / 4)
        elif cond(temporal, 'Y'):
            n *= days_in_year
        return n


    def _check_var(self, arg, arg_self):
        return arg_self if arg is None else arg

        
    def _calc_mean_return(self, df_prices, periods):
        return df_prices.apply(lambda x: x.pct_change(periods).dropna().mean())
        

    def _calc_volatility(self, df_prices, periods):
        return df_prices.apply(lambda x: x.pct_change(periods).dropna().std())
        

    def _calc_sharpe(self, df_prices, periods, rf=0):
        mean = self._calc_mean_return(df_prices, periods)
        std = self._calc_volatility(df_prices, periods)
        return (mean - rf) / std


    def get_ref_val(self, freq='1y', rf=0, align_period=False):
        """
        get ref val for plot_posterior
        """
        df_prices = self.df_prices
        if align_period:
            df_prices = self.align_period(df_prices, axis=0, fill_na=True)
        periods = self.get_freq_days(freq)
        args = [df_prices, periods]
        return {
            #'mean': self._calc_mean_return(*args).to_dict(),
            #'std': self._calc_volatility(*args).to_dict(),
            'total_return': self._calc_mean_return(*args).to_dict(),
            'sharpe': self._calc_sharpe(*args).to_dict(),
            'cagr': self._calc_mean_return(df_prices, self.days_in_year).to_dict()
        }

    @print_runtime
    def bayesian_sample(self, size_batch=50, file=None, path='.', **kwargs):
        """
        batch process for _bayesian_sample
        """
        df_prices = self.df_prices
        tickers = df_prices.columns.to_list()
        dim_ticker = 'ticker'
        size_batch = size_batch if size_batch > 0 else len(tickers)
        trace = None
        
        for i in range(0, len(tickers), size_batch):
            j = i + size_batch
            print(f'Running batch {j//size_batch} ...') if len(tickers) > size_batch else None
            tkrs_i = tickers[i:j]
            df_prc = df_prices[tkrs_i]
            res_dict = self._bayesian_sample(df_prc, dim_ticker=dim_ticker, **kwargs)
            trace_i = res_dict['trace']
            if trace is None:
                trace = trace_i
            else:
                trace = BayesianEstimator.combine_inference_data(trace, trace_i, dim=dim_ticker)
        
        # align_period, freq, rf are same for all batches
        res_dict.update({'trace':trace, 'coords':{dim_ticker: tickers}, 'data':df_prices})
        self.bayesian_data = res_dict
        if file:
            self.save(file, path)
        return None


    def _bayesian_sample(self, df_prices, dim_ticker='ticker', freq='1y',rf=0, align_period=False,
                        sample_draws=1000, sample_tune=1000, target_accept=0.9,
                        multiplier_std=1000, rate_nu = 29, normality_sharpe=True):
        """
        normality_sharpe: set to True if 
         -. You are making comparisons to Sharpe ratios calculated under the assumption of normality.
         -. You want to account for the higher variability due to the heavy tails of the t-distribution.
        """
        days_in_year = self.days_in_year
        periods = self.get_freq_days(freq)
        tickers = list(df_prices.columns)
        
        if align_period:
            df_prices = self.align_period(df_prices, axis=0, fill_na=True)
            df_ret = df_prices.pct_change(periods).dropna()
            mean_prior = df_ret.mean()
            std_prior = df_ret.std()
            std_low = std_prior / multiplier_std
            std_high = std_prior * multiplier_std
        else:
            ret_list = [df_prices[x].pct_change(periods).dropna() for x in tickers]
            mean_prior = [x.mean() for x in ret_list]
            std_prior = [x.std() for x in ret_list]
            std_low = [x / multiplier_std for x in std_prior]
            std_high = [x * multiplier_std for x in std_prior]
            ror = dict()
        
        coords={dim_ticker: tickers}
        with pm.Model(coords=coords) as model:
            # nu: degree of freedom (normality parameter)
            nu = pm.Exponential('nu_minus_two', 1 / rate_nu, initval=4) + 2.
            mean = pm.Normal('mu', mu=mean_prior, sigma=std_prior, dims=dim_ticker)
            std = pm.Uniform('sig', lower=std_low, upper=std_high, dims=dim_ticker)
            
            if align_period:
                _ = pm.StudentT('ror', nu=nu, mu=mean, sigma=std, observed=df_ret)
            else:
                func = lambda x: dict(mu=mean[x], sigma=std[x], observed=ret_list[x])
                _ = {i: pm.StudentT(f'ror[{x}]', nu=nu, **func(i)) for i, x in enumerate(tickers)}
    
            #pm.Deterministic('mean', mean, dims=dim_ticker)
            #pm.Deterministic('std', std, dims=dim_ticker)
            std_sr = std * pt.sqrt(nu / (nu - 2)) if normality_sharpe else std
            tret = pm.Normal('total_return', mu=mean, sigma=std_sr, dims=dim_ticker)
            sharpe = pm.Deterministic('sharpe', (mean-rf) / std_sr, dims=dim_ticker)

            years = periods/days_in_year
            cagr = pm.Deterministic('cagr', (tret+1) ** (1/years) - 1, dims=dim_ticker)
            yearly_sharpe = pm.Deterministic('yearly_sharpe', sharpe * np.sqrt(1/years), dims=dim_ticker)
    
            trace = pm.sample(draws=sample_draws, tune=sample_tune,
                              #chains=chains, cores=cores,
                              target_accept=target_accept,
                              #return_inferencedata=False, # TODO: what's for?
                              progressbar=True)
            
        return {'trace':trace, 'coords':coords, 'align_period':align_period, 
                'freq':freq, 'rf':rf, 'data':df_prices}
        

    def save(self, file, path='.'):
        """
        save bayesian_data of bayesian_sample 
        """
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
    
        file = set_filename(file, 'pkl')
        f = os.path.join(path, file)
        if os.path.exists(f):
            return print(f'ERROR: {f} exists')
        with open(f, 'wb') as handle:
            pickle.dump(self.bayesian_data, handle)
        return print(f'{f} saved')

                
    @staticmethod
    def load(file, path='.'):
        """
        load bayesian_data of bayesian_sample 
        """
        file = set_filename(file, 'pkl')
        files = get_file_list(file, path)
        if len(files) == 0:
            return print(f'ERROR: {file} does not exist')

        bayesian_data = None
        for f in files:
            f = os.path.join(path, f)
            with open(f, 'rb') as handle:
                bdata = pickle.load(handle)
            if bayesian_data is None:
                bayesian_data = bdata
            else:
                try:
                    bayesian_data = BayesianEstimator.combine_bayesian_data(bayesian_data, bdata)
                except ValueError:
                    return None
        file = f'{file}*' if len(files) > 1 else files[0]
        print(f'{file} loaded')
        return bayesian_data
        

    def bayesian_summary(self, var_names=None, filter_vars='like', **kwargs):
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
            df = az.summary(trace, var_names=var_names, filter_vars=filter_vars, **kwargs)
            # split index to metric & ticker to make them new index
            index = ['metric', 'ticker']
            def func(x):
                match = re.match(r"(.*)\[(.*)\]", x)
                if match:
                    return match.groups()
                else: # some var_name happen to have no ticker
                    return (x, None)
            df[index] = df.apply(lambda x: func(x.name), axis=1, result_type='expand')
            return df.loc[df['ticker'].notna()].set_index(index)


    def plot_posterior(self, *args, plotly=True, **kwargs):
        if plotly:
            return self._plot_posterior_plotly(*args, **kwargs)
        else:
            return self._plot_posterior(*args, **kwargs)
    

    def _plot_posterior(self, var_names=None, tickers=None, ref_val=None, 
                       length=20, ratio=1, textsize=9, **kwargs):
        """
        ref_val: None, float or 'default'
        kwargs: additional kwa for az.plot_posterior
        """
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
            coords = self.bayesian_data['coords']
            freq = self.bayesian_data['freq']
            rf = self.bayesian_data['rf']
            align_period = self.bayesian_data['align_period']
            security_names = self.security_names
    
        if tickers is not None:
            tickers = [tickers] if isinstance(tickers, str) else tickers
            coords = {'ticker': tickers}
    
        if ref_val == 'default':
            ref_val = self.get_ref_val(freq=freq, rf=rf, align_period=align_period)
            col_name = list(coords.keys())[0]
            ref_val = {k: [{col_name:at, 'ref_val':rv} for at, rv in v.items()] for k,v in ref_val.items()}
        #ref_val.update({'ror': [{'ref_val': 0}], 'cagr': [{'ref_val': 0}]})
    
        axes = az.plot_posterior(trace, var_names=var_names, filter_vars='like', coords=coords,
                                ref_val=ref_val, textsize=textsize, **kwargs)
        if len(axes.shape) == 1:
            n_r, n_c = 1, axes.shape[0] # axes.shape is tuple such as (5,)
            axes = [axes]
        else:
            n_r, n_c = axes.shape
        for i in range(n_r):
            for j in range(n_c):
                ax = axes[i][j]
                t = ax.get_title()
                if t == '':
                    continue
                else:
                    title = t.split('\n')[1]
                if security_names is not None:
                    #func = lambda x: string_shortener(x, n=length, r=ratio)
                    # break title every length. ratio is depricated then
                    func = lambda x: '\n'.join([x[i:i+length] for i in range(0, len(x), length)])
                    title = func(security_names[title])
                ax.set_title(title, fontsize=textsize)
        #return ref_val
        return None


    def _plot_posterior_plotly(self, var_name='total_return', tickers=None, 
                               n_points=200, hdi_prob=0.94, error=0.9999):
        """
        plot density with plotly
        """
        if self.bayesian_data is None:
            print('ERROR: run bayesian_sample first')
        else:
            posterior = self.bayesian_data['trace']
            coords = self.bayesian_data['coords']
            posterior = posterior.posterior
            security_names = self.security_names
        
        if tickers is not None:
            tickers = [tickers] if isinstance(tickers, str) else tickers
            coords = {'ticker': tickers}
        
        # Average over the chain dimension, keep the draw dimension
        averaged_data = posterior[var_name].sel(**coords).mean(dim="chain")
        
        # Convert to a DataFrame for Plotly
        df_dst = (averaged_data.stack(sample=["draw"])  # Combine draw dimension into a single index
                  .to_pandas()  # Convert to pandas DataFrame
                  .T)
        
        # Example: KDE computation for the DataFrame
        kde_data = []  # To store results
        x_values = np.linspace(df_dst.min().min(), df_dst.max().max(), n_points)  # Define global x range
        
        for ticker in df_dst.columns:
            ticker_samples = df_dst[ticker].values  # Extract samples for the ticker
            
            # Compute KDE
            kde = gaussian_kde(ticker_samples)
            density = kde(x_values)  # Compute density for the range
            
            # Store results in a DataFrame
            kde_data.append(pd.DataFrame({ticker: density}, index=x_values))
        
        # Combine all KDE data into a single DataFrame
        df_dst = pd.concat(kde_data, axis=1)
        
        # Calculate the HDI for each ticker
        hdi_lines = calculate_hdi(df_dst, hdi_prob)
        # remove small number of density
        xlims = calculate_hdi(df_dst, error)
        cond = lambda x: (x.index > xlims[x.name]['x'][0]) & (x.index < xlims[x.name]['x'][1])
        df_dst = df_dst.apply(lambda x: x.loc[cond(x)])
        
        # Plot using Plotly
        title=f"Density of {var_name.upper()} (with {hdi_prob:.0%} Interval)"
        fig = px.line(df_dst, title=title)
        fig.update_layout(
            xaxis=dict(title=var_name),
            yaxis=dict(
                title='',             # Remove y-axis title (label)
                showticklabels=False  # Hide y-tick labels
            ),
            hovermode = 'x unified',
            legend=dict(title='')
        )
        
        # Get the color mapping of each ticker from the plot
        colors = {trace.name: trace.line.color for trace in fig.data}
        # update trace name after colors creation
        if security_names is not None:
            fig.for_each_trace(lambda x: x.update(name=security_names[x.name]))
        
        # Add horizontal hdi_lines as scatter traces with line thickness, transparency, and markers
        for tkr, vals in hdi_lines.items():
            fig.add_trace(go.Scatter(
                x=vals['x'], y=vals['y'],
                mode="lines+markers",            # Draw lines with markers
                line=dict(color=colors[tkr], width=5),  # Adjust thickness, dash style, and transparency
                marker=dict(size=10, symbol='line-ns-open', color=colors[tkr]),  # Customize marker style
                opacity=0.3, 
                legendgroup=tkr,                 # Group with the corresponding data
                showlegend=False                 # Do not display in the legend
            ))
            
        #fig.update_traces(hovertemplate="%{fullData.name}<extra></extra>")
        for trace in fig.data:
            if trace.showlegend:
                trace.update(hovertemplate="%{fullData.name}<extra></extra>")  # Keep trace name
            else:
                trace.update(hoverinfo='skip')  # Exclude from hover text
        
        # Show plot
        fig.show()
    
        
    def plot_returns(self, tickers=None, num_samples=None, var_names=['total_return', 'sharpe'],
                     figsize=(10,3), xlims=None, length=20, ratio=1, max_legend=99):
        """
        var_names: ['total_return', 'sharpe'] or ['cagr', 'yearly_sharpe']
        xlims: list of xlim for ax1 & ax2. ex) [(-1,1),None]
        """
        security_names = self.security_names
        axes = create_split_axes(figsize=figsize, vertical_split=False, 
                                 ratios=(1, 1), share_axis=False, space=0.05)
        
        axes = self._plot_compare(var_names, tickers=tickers, num_samples=num_samples, 
                                  figsize=figsize, axes=axes)
        if axes is None:
            return None # see _plot_compare for err msg
            
        _ = [ax.set_xlim(x) for ax, x in zip(axes, xlims)] if isinstance(xlims, list) else None
        ax1, ax2 = axes
        _ = ax1.set_title(var_names[0].upper())
        _ = ax1.axvline(0, c='grey', lw=1, ls='--')
        _ = ax1.get_legend().remove()
        _ = ax2.set_title(var_names[1].upper())

        legend = ax2.get_legend_handles_labels()[1]
        if security_names is not None:
            clip = lambda x: string_shortener(x, n=length, r=ratio)
            legend = [clip(security_names[x]) for x in legend]
        _ = ax2.legend(legend[:max_legend], bbox_to_anchor=(1.0, 1.0), loc='upper left')
        
        _ = [ax.set_yticks([]) for ax in axes]
        _ = [ax.set_ylabel(None) for ax in axes]
        return axes

    
    def _plot_compare(self, var_names, tickers=None, num_samples=None, figsize=(6,5), axes=None):
        if self.bayesian_data is None:
            return print('ERROR: run bayesian_sample first')
        else:
            trace = self.bayesian_data['trace']
    
        if isinstance(var_names, str):
            var_names = [var_names]
        if (tickers is not None) and isinstance(tickers, str):
            tickers = [tickers]
            
        stacked = az.extract(trace, num_samples=num_samples)
        vn = [x for x in var_names if x not in stacked.keys()]
        if len(vn) > 0:
            return print_list(var_names, 'ERROR: Check if {} exit')

        if axes is None:
            fig, axes = plt.subplots(1, len(var_names), figsize=figsize)
        for i, v in enumerate(var_names):
            df = stacked[v].to_dataframe()
            df = (df[v].droplevel(['chain','draw'])
                       .reset_index().pivot(columns="ticker")
                       .droplevel(0, axis=1))
            df = df[tickers] if tickers is not None else df
            _ = df.plot.kde(ax=axes[i])
        return axes
        

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


    @staticmethod
    def combine_inference_data(idata1, idata2, dim='ticker'):
        """
        Combines two InferenceData objects, concatenating along the dim 
        for all variables in the dataset.
        """
        # Check if both InferenceData objects have the same groups
        cond1 = set(idata1.groups()) == set(idata2.groups())
        # check only coords of group posterior for convenience
        cond2 = set(idata1['posterior'].coords) == set(idata1['posterior'].coords)
        if not (cond1 and cond2) :
            raise ValueError("InferenceData objects have different components, cannot combine.")
        
        # Initialize an empty dictionary to hold the combined data variables
        combined_data = {}
        # Loop through all data variables in the first InferenceData object
        for var in idata1.groups():
            # Concatenate the corresponding variables along the 'ticker' dimension
            combined_data[var] = xr.concat([idata1[var], idata2[var]], dim=dim)
        # Create the new InferenceData object with the combined data
        return az.InferenceData(**combined_data)

    
    @staticmethod
    def combine_bayesian_data(data1, data2):
        """
        Combines two bayesian_data from bayesian_sample
        """
        # conditions to check if two data can be combined
        cond1 = set(data1.keys()) == set(data2.keys())
        keys_same = ['align_period', 'freq', 'rf']
        cond2 = all(data1.get(k) == data2.get(k) for k in keys_same)
        dim_ticker = list(data1['coords'].keys())[0]
        cond3 = dim_ticker == list(data2['coords'].keys())[0]
        cond4 = data1['data'].columns.intersection(data2['data'].columns).size == 0
        if not (cond1 and cond2 and cond3 and cond4):
            #return print('ERROR: Data cannot combine')
            raise ValueError('Bayesian data have different structures, cannot combine')
        trace = BayesianEstimator.combine_inference_data(data1['trace'], data2['trace'], dim=dim_ticker)
        tickers = data1['coords'][dim_ticker] + data2['coords'][dim_ticker]
        df_prices = pd.concat([data1['data'], data2['data']], axis=1)
        data1.update({'trace':trace, 'coords':{dim_ticker: tickers}, 'data':df_prices})
        return data1
    

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
                     .swaplevel().sort_index())
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
        Converts price data from DataManager to f-ratios in FinancialRatios format, or vice versa 
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
    Compares performance metrics across multiple portfolios, each built using a separate
    PortfolioBuilder instance
    """
    def __init__(self, *pf_names, col_ticker='ticker', col_portfolio='portfolio', 
                 verbose=True, **kwargs):
        """
        pf_names: list of portfolio names
        kwargs: see create_portfolio
        """
        self.pf_data = PortfolioData()
        self.col_ticker = col_ticker
        self.col_portfolio = col_portfolio
        self.portfolios = dict() # dict of name to PortfolioBuilder instance
        self.load(*pf_names, verbose=verbose, **kwargs)
        self.df_category = None # see import_category
        self.names_vals = dict(
            date='date', ttl='TOTAL', start='start', end='end', 
            buy='buy', sell='sell', value='value', ugl='ugl', roi='roi')

    
    def load(self, *pf_names, reload=False, 
             verbose=True, default_name='Portfolio', **kwargs):
        """
        loading multiple portfolios (no individual args except for PortfolioData)
        pf_names: list of portfolio names
        """
        # split pf list to names and instances
        pf_str, pf_inst = [], []
        for pf in pf_names:
            if isinstance(pf, str):
                pf_str.append(pf)
            else:
                pf_inst.append(pf)

        # check pf names
        if len(pf_str) > 0:
            pf_str = self.check_portfolios(*pf_str, loading=True)
            if len(pf_str) == 0:
                return None
            
        if reload:
            pf_dict = dict()
        else:
            pf_dict = self.portfolios

        # create pf instances from pf names
        for name in pf_str:
            if name in pf_dict.keys():
                print(f'{name} already exists')
            else:
                print(f'{name}:', end='\n' if verbose else ' ')
                with SuppressPrint(not verbose):
                    pf = PortfolioManager.create_portfolio(name, **kwargs)
                if pf.record is None:
                    print(f'WARNING: Portfolio {name} not loaded')
                else:
                    pf_dict[name] = pf
                print() if verbose else print('imported')

        # add pf instances from input to portfolio
        if len(pf_inst) > 0:
            for pf in pf_inst:
                name = pf.name if pf.name else add_suffix(default_name, pf_dict.keys())
                if name in pf_dict.keys():
                    return print(f'ERROR: Duplicate portfolios {name}')
                else:
                    pf_dict[name] = pf
                    print(f'{name}: imported')
        
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
        name: name of universe for DataManager input
        args, kwargs: args & kwargs to replace DataManager input
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
    def create_portfolio(name, *args, universe=None, strategy=None, df_additional=None, **kwargs):
        """
        name: portfolio name defined in PortfolioData if both universe & strategy are None
        args, kwargs: additional args & kwargs for PortfolioBuilder
        df_additional: explicit set to exlcude from kwargs
        universe & strategy: set to create portfolio not predefined in pf_data
        """
        # removal for comparison with strategy_data
        security_names = kwargs.pop('security_names', None)
        _ = kwargs.pop('name', None) # drop name if it's in kwargs

        # get transaction file & path if given in kwargs
        kwa_tran = ['file', 'path']
        kwa_tran = {x: kwargs.pop(x, None) for x in kwa_tran}

        # get kwarg sets of portfolios
        pfd = PortfolioData()
        if not (universe and strategy):
            kwa_pf = pfd.review_portfolio(name, strategy=False, universe=False)
            if kwa_pf is None:
                return None
            else:
                universe = kwa_pf['universe']
                strategy = kwa_pf['strategy']
                # update transaction file
                kwa_tran = {k:kwa_pf[k] if v is None else v for k,v in kwa_tran.items()}
        else: # both of universe and strategy given
            if None in kwa_tran.values():
                return print('ERROR: Set transaction file & path')

        # create portfolio_data with transaction data of the portfolio
        portfolio_data = {**kwa_tran}

        # get universe
        dm = PortfolioManager.create_universe(universe) # instance of DataManager
        if dm is None:
            return None
        security_names = dm.get_names() if security_names is None else security_names # update security_names
        df_universe = dm.df_prices
        portfolio_data.update(dm.portfolio_data)
            
        # get kwargs of PortfolioBuilder
        strategy_data = pfd.review_strategy(strategy)
        if strategy_data is None:
            return None # see review_strategy for err msg 
            
        # update strategy_data if input kwargs given
        tmp = [k for k in kwargs.keys() if k in strategy_data.keys()]
        strategy = None if len(tmp) > 0 else strategy # reset name if strategy updated
        strategy_data = {**strategy_data, **kwargs}
        # set portfolio_data for ref
        portfolio_data['strategy'] = {'data':strategy_data, 'name':strategy}

        # create cost if its file given
        cost = strategy_data.pop('cost', None)
        if isinstance(cost, str): # cost is file name
            path = kwa_tran['path'] # cost file in the same dir with transaction file
            cost = PortfolioManager.get_cost(universe, cost, path=path)
        
        kws = {**strategy_data, 'name':name, 'security_names':security_names, 
               'cost':cost, **kwa_tran}
        pb = PortfolioBuilder(df_universe, *args, df_additional=df_additional, **kws)
        pb.portfolio_data = portfolio_data
        return pb

    @staticmethod
    def get_cost(name, file, path='.'):
        """
        name: universe name. see keys of UNIVERSES
        """
        return CostManager.get_cost(name, file, path=path)

    @staticmethod
    def check_cost(name, file, path='.'):
        """
        name: universe name. set to None to check cost data regardless of universe
        """
        return CostManager.check_cost(file, path=path, universe=name)


    def check_portfolios(self, *pf_names, loading=False):
        """
        check if portfolios exist 
        pf_names: list of portfolios as name or prefix 
        loading: set to False to retrieve names of the portfolios loaded 
        """
        if loading:
            pf_all = self.pf_data.portfolios.keys()
        else:
            pf_all = self.portfolios.keys()
        if len(pf_names) == 0:
            pf_names = pf_all
        else:
            out = [x for x in pf_names for y in pf_all if y.startswith(x)]
            out = set(pf_names)-set(out)
            if len(out) > 0:
                print_list(out, 'ERROR: No portfolio such as {}')
                print_list(pf_all, 'Portfolios available: {}')
                pf_names = list()
            else:
                pf_names = [y for x in pf_names for y in pf_all if y.startswith(x)]
        return pf_names

    
    def plot(self, *pf_names, start_date=None, end_date=None, roi=True, exclude_cost=False, 
             figsize=(10,5), legend=True, colors=plt.cm.Spectral):
        """
        start_date: date of beginning of the return plot
        end_date: date to calc return
        roi: ROI plot if True, UGL plot if False
        """
        # check portfolios
        pf_names = self.check_portfolios(*pf_names)
        if len(pf_names) == 0:
            return None
    
        nms_v = self.names_vals
        nm_val = nms_v['value']
        nm_buy = nms_v['buy']
        nm_ttl = nms_v['ttl']
        nm_ugl = nms_v['ugl']
        nm_roi = nms_v['roi']
        nm_date = nms_v['date']
        nm_end = nms_v['end']
        
        # get data
        df_all = self._valuate(*pf_names, date='all', category=None, exclude_cost=exclude_cost)
        df_all = df_all.loc[start_date:end_date]
        _, end_date = get_date_minmax(df_all)
    
        # set plot title
        df = self.summary(*pf_names, date=end_date, int_to_str=False)
        sr = df[nm_ttl]
        title = PortfolioBuilder.get_title_pnl(sr[nm_roi], sr[nm_ugl], sr[nm_end])
    
        # total value
        line_ttl = {'c':'black', 'alpha':0.3, 'ls':'--', 'lw':1}
        # rename for legend
        sr_val = df_all[nm_val].unstack().ffill().sum(axis=1).rename('Total Value')
        ax1 = sr_val.plot(title=title, figsize=figsize, **line_ttl)
        ax1.set_ylabel('Total Value')
        ax1.set_xlabel('')
    
        # roi or ugl total
        if roi:
            result_indv = nm_roi
            ylabel_indv = 'Return On Investment (%)'
            calc = lambda x: x[nm_ugl] / x[nm_buy] * 100
            sr_pnl = (df_all.unstack().ffill() # ffill dates of no value with the last value
                      .stack().groupby(nm_date).sum().apply(calc, axis=1)) # calc roi
        else:
            result_indv = nm_ugl
            ylabel_indv = 'Unrealized Gain/Loss'
            sr_pnl = df_all[nm_ugl].unstack().ffill().sum(axis=1)
        ax2 = ax1.twinx()
        _ = sr_pnl.rename(f'Total {result_indv.upper()}').plot(ax=ax2, lw=1)
    
        # roi or ugl individuals
        df_all[result_indv].unstack().ffill().mul(100 if roi else 1).plot(ax=ax2, alpha=0.5, lw=1)
        ax2.set_prop_cycle(color=colors(np.linspace(0,1,len(pf_names))))
        ax2.set_ylabel(ylabel_indv)
        _ = set_matplotlib_twins(ax1, ax2, legend=legend)
    
        # fill total roi/ugl
        ax2.fill_between(sr_pnl.index, sr_pnl, ax2.get_ylim()[0], 
                         facecolor=ax2.get_lines()[0].get_color(), alpha=0.1)
        ax1.margins(0)
        ax2.margins(0)
        return None
        

    def summary(self, *pf_names, date=None, 
                int_to_str=True, category=None, exclude_cost=False, sort_by=None, 
                plot=False, roi=True, figsize=None):
        """
        get cashflow & pnl of groups in category (default portfolio) on date
        """       
        pf_names = self.check_portfolios(*pf_names)
        if len(pf_names) == 0:
            return None
        # history not supported in summary
        date = None if date == 'all' else date
        nms_v = self.names_vals
        nm_ttl = nms_v['ttl']
        nm_start = nms_v['start']
        nm_end = nms_v['end']
        nm_roi = nms_v['roi']
        nm_ugl = nms_v['ugl']
        nm_buy = nms_v['buy']
    
        df_val = self._valuate(*pf_names, date=date, category=category, exclude_cost=exclude_cost)
        if plot:
            category = df_val.index.name # reset category according to result df_val
            df_val = df_val.reset_index()
            df_val = df_val.sort_values(sort_by, ascending=True) if sort_by in df_val.columns else df_val
            axes = PortfolioBuilder._plot_assets(df_val, col_name=category, roi=roi, figsize=figsize)
            return None
        else:
            # set total
            df_val = df_val.T
            df_val[nm_ttl] = [df_val.loc[nm_start].min(), df_val.loc[nm_end].max(), 
                                 *df_val.iloc[2:].sum(axis=1).to_list()]
            df_ttl = df_val[nm_ttl]
            df_val.loc[nm_roi, nm_ttl] = df_ttl[nm_ugl] / df_ttl[nm_buy]
            return df_val.map(format_price, digits=0) if int_to_str else df_val


    def performance_stats(self, *pf_names, date=None, column='Realized',
                          metrics=METRICS, sort_by=None, exclude_cost=False):
        """
        compare performance stats of portfolios with 2 different methods
        date: date for fixed weights of simulated performance
        column: 'Realized' for stats of actual portfolio,
                 see ProtfolioBuilder.performance_stats for details
        """
        # check portfolios
        pf_names = self.check_portfolios(*pf_names)
        if len(pf_names) == 0:
            return None
            
        # get data from each portfolio
        df_all = None
        no_res = []
        for name in pf_names:
            pf = self.portfolios[name]
            df = pf.performance_stats(date=date, metrics=metrics, sort_by=sort_by, exclude_cost=exclude_cost)
            if df is None:
                no_res.append(name)
            else:
                # add portfolio name
                df = df[column].rename(name)
                df_all = df if df_all is None else pd.concat([df_all, df], axis=1) 
        print(f"WARNING: Check portfolios {', '.join(no_res)}") if len(no_res) > 0 else None
        return df_all


    def diversification_history(self, *pf_names, start_date=None, end_date=None, 
                                metrics=None, exclude_cost=False, min_dates=20,
                                plot=True, figsize=(8,4), ylim=None):
        """
        Compute history of three key diversification metrics for a group of portfolios:
        - Diversification Ratio (DR)
        - HHI-based Diversification Score
        - Effective Number of Bets (ENB)
        """
        # check portfolios
        pf_names = self.check_portfolios(*pf_names)
        if len(pf_names) < 2:
            return None
    
        nms_v = self.names_vals
        nm_val = nms_v['value']
        nm_buy = nms_v['buy']
        nm_ugl = nms_v['ugl']
        nm_roi = nms_v['roi']
        nm_date = nms_v['date']
        col_portfolio = self.col_portfolio
    
        df_all = self._valuate(*pf_names, date='all', category=None, exclude_cost=exclude_cost)
        df_val = df_all.loc[start_date:end_date]
    
        # get weight history
        df_wgt = df_val[nm_val].unstack(col_portfolio)
        df_wgt = (df_wgt.replace(0, None) # replace to None for next apply
                  .apply(lambda x: x.dropna() / sum(x.dropna()), axis=1))
    
        # portfolio returns from cumulative roi
        df_ret = df_all[nm_roi].unstack(col_portfolio) 
        df_ret = (1 + df_ret) / (1 + df_ret.shift(1)) - 1
        df_ret = df_ret.dropna(how='all')
        
        # check metrics
        options = ['HHI', 'DR', 'ENB']
        metrics = [metrics] if isinstance(metrics, str) else metrics
        metrics = [x.upper() for x in metrics] if metrics else options
        if len(set(options) - set(metrics)) == len(options):
            return print('ERROR')
        else:
            dates = df_wgt.index
            df_div = pd.DataFrame(index=dates)
            # reset dates depending on the return size for 'DR' & 'ENB'
            n = df_ret.index.min() + timedelta(days=min_dates) - dates.min()
            dates = dates[n.days:] if n.days > 0 else dates
        
        # calc metrics history
        if 'HHI' in metrics:
            df_div['HHI'] = df_wgt.apply(lambda x: diversification_score(x.dropna()), axis=1)
    
        for mtr, func in zip(options[1:], [diversification_ratio, effective_number_of_risk_bets]):
            if mtr in metrics:
                res = []
                for dt in dates:
                    sr_tkr = df_wgt.loc[dt].dropna()
                    ret = df_ret.loc[:dt, sr_tkr.index]
                    x = func(sr_tkr.to_list(), ret)
                    res.append(x)
                df_div[mtr] = pd.Series(res, index=dates)
        #df_div = df_div.interpolate()
    
        if plot:
            # add total roi plot
            calc = lambda x: x[nm_ugl] / x[nm_buy] * 100
            df_ttl = (df_val.unstack().ffill() # ffill dates of no value with the last value
                      .stack().groupby(nm_date).sum().apply(calc, axis=1)) # calc roi
            ax = df_ttl.plot(label='ROI', figsize=figsize, color='grey', ls='--', lw=1)
            # plot metrics
            axt = ax.twinx()
            _ = df_div.plot(ax=axt, title='Portfolio Diversification')
            if ylim is None:
                ylim = (df_div.min().min()*0.9, df_div.max().max()*1.1)
            axt.set_ylim(ylim)
            ax.set_ylabel('Return On Investment (%)')
            axt.set_ylabel('Diversification')
            axt.grid(axis='y', alpha=0.3)
            ax.margins(x=0)
            ax.set_xlabel('')
            _ = set_matplotlib_twins(ax, axt, legend=True, loc='upper left')
            return None
        else:
            return df_div


    def import_category(self, file, path='.', col_ticker='ticker', col_portfolio='portfolio', exclude=None):
        """
        get df of category
        file: file name, dict, series, df
        """
        if isinstance(file, str): # file is file of category
            try:
                df_cat = (pd.read_csv(f'{path}/{file}')
                            # rename cols to join with tickers in portfolios
                            .rename(columns={col_ticker:self.col_ticker})
                            .set_index(self.col_ticker))
            except Exception as e:
                return print('ERROR:', e)
        elif isinstance(file, dict): # file is dict of category to ticker list
            df_cat = pd.Series({v: k for k, vals in file.items() for v in vals})
        elif isinstance(file, pd.Series):
            if file.name is None:
                file.name = 'Category1'
            df_cat = file.to_frame()
        elif isinstance(file, pd.DataFrame):
            df_cat = file
        else:
            return print('ERROR: Input must be file name, dict, series or dataframe of category')
    
        if exclude is not None: # remove category set in exclude
            exclude = [exclude] if isinstance(exclude, str) else exclude
            cols = df_cat.columns.difference(exclude)
            df_cat = df_cat[cols]
    
        # add portfolio name as index if given in new cat
        df_all = self.util_performance_by_asset(date=None, exclude_cost=True)
        if col_portfolio in df_cat.columns:
            df_cat = (df_cat.rename(columns={col_portfolio:self.col_portfolio})
                      .set_index(self.col_portfolio, append=True))
            # update index of df_all as well for following checks
            df_all = df_all.set_index(self.col_portfolio, append=True)

        # check duplicate assets in multiple groups of new category
        if df_cat.index.has_duplicates:
            x = ', '.join(df_cat.index.names)
            return print(f'ERROR: Duplicate {x} in the category')
        
        # check missing groups of a category for assets in pfs
        if df_all.index.difference(df_cat.index).size > 0:
            x = ', '.join(df_cat.index.names)
            return print(f'ERROR: Check category as missing {x}')
    
        # check duplicate category
        cats = df_cat.columns.intersection(df_all.columns)
        if cats.size > 0:
            cats = ', '.join(cats)
            return print(f'ERROR: Failed to import custom category. Rename column {cats}')
        
        cats = ', '.join(df_cat.columns)
        print(f'Custom category loaded: {cats}')
        self.df_category = df_cat
        return None
        

    def util_print_summary(self, *pf_names, date=None):
        """
        print summary for bookkeeping
        """
        df_s = self.summary(*pf_names, date=date, int_to_str=False)
        df = df_s.drop(columns=['TOTAL'])
        for p in df.columns:
            sr = df[p].apply(format_price, digits=0, int_to_str=False)
            values = sr.iloc[2:].astype(str).tolist()
            print(f"{sr['end']}, {', '.join(p.split('_'))}, , , , 평가, , {', '.join(values)}")


    def util_performance_by_asset(self, *pf_names, date=None, exclude_cost=False):
        """
        get ticker list of assets in all portfolios
        """
        pf_names = self.check_portfolios(*pf_names)
        if len(pf_names) == 0:
            return None
    
        df_all = self._performance_by_asset(*pf_names, date=date, exclude_cost=exclude_cost)
        return df_all


    def _valuate(self, *pf_names, date=None, category=None, exclude_cost=False, format_date='%Y-%m-%d'):
        """
        return evaluation summary df the portfolios in pf_names
        pf_names: list of portfolio names
        date: date for values on date, None for values on last date, 'all' for history
        """
        df_cat = self.df_category
        col_portfolio = self.col_portfolio
        col_ticker = self.col_ticker
        nms_v = self.names_vals
        nm_val = nms_v['value']
        nm_sell = nms_v['sell']
        nm_buy = nms_v['buy']
        nm_start = nms_v['start']
        nm_end = nms_v['end']
        nm_date = nms_v['date']
    
        # get data from each portfolio
        df_all = self._performance_by_asset(*pf_names, date=date, exclude_cost=exclude_cost)
    
        # set custom category
        if date != 'all':
            # check category
            category = category or col_portfolio
            if category not in df_all.columns:
                if (df_cat is not None) and category in df_cat.columns:
                    if len(df_cat.index.names) > 1: # index of df_cat is (ticker, portfolio)
                        df_all = (df_all.set_index(self.col_portfolio, append=True)
                                  .join(df_cat[category]).reset_index(self.col_portfolio))
                    else:
                        df_all = df_all.join(df_cat[category])
                else:
                    print(f'WARNING: Reset category to {col_portfolio} as no {category} exists in the category')
                    category = col_portfolio
            # construct result according to category
            sr_start = df_all.groupby(category)[nm_start].min().dt.strftime(format_date)
            sr_end = df_all.groupby(category)[nm_end].max().dt.strftime(format_date)
            cols = [nm_buy, nm_sell, nm_val]
            df_all = sr_start.to_frame().join(sr_end).join(df_all.groupby(category)[cols].sum())
            # add profit columns
            df_prf = PortfolioBuilder.calc_profit(df_all, result='both', 
                                         col_val=nm_val, col_sell=nm_sell, col_buy=nm_buy)
            return pd.concat([df_all, df_prf], axis=1)
        else:
            return df_all.set_index(col_portfolio, append=True).sort_index()


    def _performance_by_asset(self, *pf_names, date=None, exclude_cost=False):
        """
        pf_names: list of portfolio names
        date: date for values on date, None for values on last date, 'all' for history
        """
        # get data from each portfolio
        df_all = None
        # custom category not supported for history
        total = True if date == 'all' else False
        no_res = []
        for name in pf_names:
            pf = self.portfolios[name]
            df = pf.valuate(date=date, total=total, exclude_cost=exclude_cost, 
                            exclude_sold=False, # Including all historical assets to calculate total profit
                            int_to_str=False, print_msg=False) 
            if df is None:
                no_res.append(name)
            else:
                # add portfolio name
                df = df.assign(**{self.col_portfolio:name})
                df_all = df if df_all is None else pd.concat([df_all, df], axis=0) 
        print(f"WARNING: Check portfolios {', '.join(no_res)}") if len(no_res) > 0 else None
        return df_all



class DataMultiverse:
    """
    Manages and compares data across multiple independent data universes,
    each handled by its own DataManager instance.
    """
    def __init__(self, *universes):
        """
        universes: list of universe names or DataManager instances
        """
        self.multiverse = dict() # dict of universe name to instance
        self.cost = None # cost of tickers in multiverse
        self.pf_data = PortfolioData()
        self.load(*universes) # load price history across universes
        self.tickers_in_multiverse = self.map_tickers()

    
    def load(self, *universes, reload=False, verbose=True, default_name='UV', **kwargs):
        """
        load instances of universes
        universes: list of universe names, DataManager instances or tuple of name & instance
        kwargs: keyword args for create_universe
        """        
        # split universe list to names and instances
        uv_str, uv_inst, cnt = [], {}, 0
        for uv in universes:
            if isinstance(uv, str):
                uv_str.append(uv)
            else: # uv assumed as DataManager instance or tuple
                if isinstance(uv, tuple) and isinstance(uv[0], str):
                    uv_inst[uv[0]] = uv[1]
                elif isinstance(fund, DataManager): # define name for the instance
                    cnt += 1
                    uv_inst[f'{default_name}{cnt}'] = uv
                else:
                    return print('ERROR')

        # check uv names
        if len(uv_str) > 0:
            uv_str = self.check_universes(uv_str, loading=True)
            if len(uv_str) == 0:
                return None
            
        if reload:
            multiverse = dict()
        else:
            multiverse = self.multiverse

        # create uv instances from uv names
        for name in uv_str:
            if name in multiverse.keys():
                print(f'{name} already exists')
            else:
                print(f'{name}:', end='\n' if verbose else ' ')
                with SuppressPrint(not verbose):
                    uv = PortfolioManager.create_universe(name, **kwargs)
                if uv is None:
                    print(f'WARNING: Portfolio {name} not loaded')
                else:
                    multiverse[name] = uv
                print() if verbose else print('imported')

        # add uv instances from input to multiverse
        if len(uv_inst) > 0:
            for name, uv in uv_inst.items():
                if name in multiverse.keys():
                    return print(f'ERROR: Duplicate universe {name}')
                else:
                    multiverse[name] = uv
                    print(f'{name}: imported')

        self.multiverse = multiverse
        return None


    def check_universes(self, universes=None, loading=False, exact=True):
        """
        check if universe exist 
        universes: list of universes as name or prefix
        loading: set to False to retrieve list of sets of name & universe instance loaded 
        exact: set to False if universes is list of pattern to search
        """
        if loading:
            uv_all = self.pf_data.universes.keys()
        else:
            if len(self.multiverse) == 0:
                return print('ERROR: Load data first')
            else:
                uv_all = self.multiverse.keys()
        
        if universes is None:
            universes = uv_all
        else:
            cond = lambda pattern, target: pattern == target if exact else pattern in target
            out = [x for x in universes for y in uv_all if cond(x, y)]
            out = set(universes)-set(out)
            if len(out) > 0:
                print_list(out, 'ERROR: No universe such as {}')
                print_list(uv_all, 'Universes available: {}')
                universes = list()
            else:
                universes = [y for x in universes for y in uv_all if cond(x, y)]
        # return list of sets of name & universe instance loaded if loading=False
        return universes if loading else {k:v for k,v in self.multiverse.items() if k in universes} 


    def map_tickers(self, fmt="{ticker}({universe})"):
        """
        create mapper of tickers in universes to ones in multiverse
        """
        multiverse = self.check_universes(loading=False)
        if len(multiverse) == 0:
            return None

        mapping = dict()
        for name, uv in multiverse.items():
            tkrs = uv.get_names(reset=False)
            if tkrs is None:
                continue
            tkrs = {x: fmt.format(ticker=x, universe=name) for x in tkrs.keys()}
            mapping = {**mapping, **tkrs}
        return mapping
            

    def get_names(self, tickers=None, universes=None, search=None, reset=False):
        """
        tickers: None, a ticker, list of tickers or 'selected'
        search: word to search in ticker names
        universes: list of universes to search tickers
        """
        multiverse = self.check_universes(universes, loading=False)
        if len(multiverse) == 0:
            return None

        # find universe for each ticker
        if isinstance(tickers, str):
            tickers = tickers if tickers.lower() == 'selected' else [tickers]
        security_names = dict()
        for name, uv in multiverse.items():
            sname = uv.get_names(tickers, reset)
            if sname is None:
                continue
            #sname = {f'{k}({name})': v for k,v in sname.items()}
            sname = {self.tickers_in_multiverse[k]: v for k,v in sname.items()}
            security_names = {**security_names, **sname}

        # search word in name
        if search is not None:
            security_names = {k:v for k,v in security_names.items() if search in v}
        return SecurityDict(security_names, names=security_names)
        

    def get_prices(self, universes=None):
        """
        merge price data from universes by adding universe name to column names 
        """
        multiverse = self.check_universes(universes, loading=False)
        if len(multiverse) == 0:
            return None

        df_prices = None
        for name, uv in multiverse.items():
            df_p = uv.df_prices.copy() # use copy not to contaminate uv.df_prices
            if df_p is None:
                continue
            # update column names with universe name
            df_p.columns = [self.tickers_in_multiverse[x] for x in df_p.columns]
            df_prices = df_p if df_prices is None else pd.concat([df_prices, df_p], axis=1)
        return df_prices


    def set_cost(self, cost_multiverse, path=None):
        """
        cost_multiverse: dict of universe name and its cost data from CostManager.get_cost
            or cost file for all universes
        """
        multiverse = self.check_universes(loading=False)
        if len(multiverse) == 0:
            return None
    
        # load cost for each universe
        if isinstance(cost_multiverse, str): # arg cost is file name
            cost_mv = dict()
            for name in multiverse.keys():
                cost_mv[name] = PortfolioManager.get_cost(name, cost_multiverse, path=path)
        else:
            cost_mv = cost_multiverse
    
        if len(cost_mv) == 0:
            return print('ERROR: No cost loaded')
        
        # fill zero cost for missing universe
        loaded, failed = [], []
        for name in multiverse.keys():
            if name not in cost_mv.keys():
                cost_mv[name] = {x: 0 for x in list(cost_mv.values())[0].keys()}
                failed.append(name)
            else:
                loaded.append(name)
        
        # map ticker to multiverse
        cost = {x: dict() for x in list(cost_mv.values())[0].keys()}
        mapping = self.tickers_in_multiverse
        for name, _cost in cost_mv.items():
            cost_uv = dict()
            cv0 = list(_cost.values())[0]
            if isinstance(cv0, Number):
                # retrieve tickers in universe to set cost for tickers
                uv = self.multiverse[name]
                tkrs = uv.get_names(reset=False)
                for ct, cv in _cost.items():
                    cost_uv[ct] = {mapping[x]: cv for x in tkrs}
            elif isinstance(cv0, (dict, pd.Series)):
                for ct, cv in _cost.items():
                    cost_uv[ct] = {mapping[t]: c for t,c in cv.items()}
            else:
                return print('ERROR')
            # add to multiverse cost
            cost = {x: {**cost[x], **cost_uv[x]} for x in cost}
    
        if len(list(cost.values())[0]) > 0:
            self.cost = cost
            if len(loaded) > 0:
                print(f"Cost of {', '.join(loaded)} loaded")
            if len(failed) > 0:
                print(f"Cost of {', '.join(failed)} set to zero")
            return None
        else:
            return print('ERROR: No cost loaded')


    def plot(self, tickers, universes=None, reload=True, **kwargs):
        mapping = self.tickers_in_multiverse
        if mapping is None:
            return print('ERROR')
        
        vs = self.get_visualizer(universes=universes, reload=reload)
        try:
            tickers = self.get_names('selected').keys() if tickers is None else [mapping[x] for x in tickers]
        except KeyError:
            return print('ERROR: Check tickers')
        return vs.plot(tickers, cost=self.cost, **kwargs)
        

    def performance(self, tickers=None, universes=None, reload=True, 
                    metrics=METRICS2, **kwargs):
        """
        kwargs: extra args for DataVisualizer.performance
        """
        mapping = self.tickers_in_multiverse
        if mapping is None:
            return print('ERROR')
    
        vs = self.get_visualizer(universes=universes, reload=reload)
        try:
            tickers = self.get_names('selected').keys() if tickers is None else [mapping[x] for x in tickers]
        except KeyError:
            return print('ERROR: Check tickers')
        return vs.performance(tickers=tickers, metrics=metrics, 
                              cost=self.cost, transpose=False, **kwargs)
        

    def get_visualizer(self, universes=None, reload=False):
        if reload or (self.visualization is None):
            df_prices = self.get_prices(universes=universes)
            security_names = self.get_names(universes=universes)
            self.visualization = DataVisualizer(df_prices, security_names)
        return self.visualization
   
    

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
        name: strategy name
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

    @staticmethod
    def get_universe(universe, universes=UNIVERSES):
        try:
            return universes[universe]['universe']
        except KeyError:
            return None