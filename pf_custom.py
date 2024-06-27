from bt.core import Algo, AlgoStack
from bt.algos import SelectN

import pandas as pd
import statsmodels.api as sm
import numpy as np


class SelectKRatio(AlgoStack):
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
        super(SelectKRatio, self).__init__(
            StatKRatio(lookback=lookback, lag=lag),
            SelectN(n=n, sort_descending=sort_descending, all_or_none=all_or_none),
        )


class StatKRatio(Algo):
    """
    Sets temp['stat'] with k-ratio over a given period.
    """
    def __init__(self, lookback=pd.DateOffset(months=3), lag=pd.DateOffset(days=0)):
        super(StatKRatio, self).__init__()
        self.lookback = lookback
        self.lag = lag

    def __call__(self, target):
        selected = target.temp["selected"]
        t0 = target.now - self.lag
        if target.universe[selected].index[0] > t0:
            return False
        prc = target.universe.loc[t0 - self.lookback : t0, selected]

        #print(prc.iloc[0].notna().sum())
        if prc.iloc[0].notna().sum() > 0:
            kratio = prc.pct_change(1).apply(lambda x: calc_kratio(x.dropna()))
        else:
            kratio = prc.iloc[0]
        
        target.temp["stat"] = kratio
        
        #kratio = prc.pct_change(1).apply(lambda x: calc_kratio(x.dropna()))
        #target.temp["stat"] = kratio
        #print(kratio)

        #tret = (prc.iloc[-1] / prc.iloc[0]) - 1
        #target.temp["stat"] = tret
        #print(tret)
        #print(prc.iloc[0])
        
        return True



def calc_kratio(ret):
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
        return None
