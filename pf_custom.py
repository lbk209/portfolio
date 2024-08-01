from bt.core import Algo, AlgoStack
from bt.algos import SelectN

import pandas as pd
import numpy as np
import statsmodels.api as sm


def calc_kratio(ret):
    """
    ret: pandas series
    """
    ret_cs = np.log(1 + ret).cumsum() 
    X = list(range(len(ret)))
    Y = ret_cs
    try:
        reg = sm.OLS(Y, X).fit()
        coef = reg.params.values[0]
        std_err = reg.bse.values[0]
        if std_err == 0:
            return None
        else:
            return coef / std_err
    except ValueError as e:
        return print(f'ERROR: {e}')


def calc_information_discreteness(ret):
    """
    ret: daily return of prices, pandas series
    """
    #ret = ret.dropna()
    get_sign = lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
    sign_cnt = pd.Series(0, index=[1,-1,0])
    sign_cnt.update(ret.apply(get_sign).value_counts())
    
    tret = np.log(1 + ret).cumsum() 
    tret = np.exp(tret[-1]) - 1
    
    return get_sign(tret) * (sign_cnt[-1] - sign_cnt[1])/sum(sign_cnt) 


class AlgoSelectKRatio(AlgoStack):
    """
    Sets temp['selected'] based on a k-ratio momentum filter.
    """
    def __init__(
        self,
        n,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        sort_descending=True,
        all_or_none=False,
    ):
        super(AlgoSelectKRatio, self).__init__(
            AlgoStatKRatio(lookback=lookback, lag=lag),
            SelectN(n=n, sort_descending=sort_descending, all_or_none=all_or_none),
        )


class AlgoStatKRatio(Algo):
    """
    Sets temp['stat'] with k-ratio over a given period.
    """
    def __init__(self, lookback=pd.DateOffset(months=3), lag=pd.DateOffset(days=0)):
        super(AlgoStatKRatio, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = target.temp["selected"]
        t0 = target.now - self.lag
        if target.universe[selected].index[0] > t0:
            return False
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        
        if prc.iloc[0].notna().sum() > 0:
            kratio = prc.pct_change(1).apply(lambda x: calc_kratio(x.dropna()))
        else:
            kratio = prc.iloc[0]
        
        target.temp["stat"] = kratio
        
        return True


class AlgoSelectIDiscrete(AlgoStack):
    """
    Sets temp['selected'] based on a k-ratio momentum filter.
    """
    def __init__(
        self,
        n,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        sort_descending=False,
        all_or_none=False,
    ):
        super(AlgoSelectIDiscrete, self).__init__(
            AlgoStatIDiscrete(lookback=lookback, lag=lag),
            SelectN(n=n, sort_descending=sort_descending, all_or_none=all_or_none),
        )


class AlgoStatIDiscrete(Algo):
    """
    Sets temp['stat'] with k-ratio over a given period.
    """
    def __init__(self, lookback=pd.DateOffset(months=3), lag=pd.DateOffset(days=0)):
        super(AlgoStatIDiscrete, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = target.temp["selected"]
        t0 = target.now - self.lag
        if target.universe[selected].index[0] > t0:
            return False
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        
        if prc.iloc[0].notna().sum() > 0:
            id = prc.pct_change(1).apply(lambda x: calc_information_discreteness(x.dropna()))
        else:
            id = prc.iloc[0]
        
        target.temp["stat"] = id
        
        return True
        
        
class AlgoRunAfter(Algo):
    def __init__(self, lookback=pd.DateOffset(months=0), lag=pd.DateOffset(days=0)):
        """
        start trading after date offset by lookback and lag
        """
        super(AlgoRunAfter, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        t0 = target.now - self.lag - self.lookback
        if t0 in target.universe.index:
            return True
        else:
            return False



