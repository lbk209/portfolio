from bt.core import Algo, AlgoStack
from bt.algos import StatTotalReturn

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


def redistribute_weights(weights, threshold=0.1, n_min=1, none_if_fail=False):
    """
    Discards elements below the threshold and redistributes their weights 
    to the remaining elements proportionally based on their original weights.

    Args:
        weights (Series or dict): A series or dictionary describing the weights.
        threshold (float): Minimum threshold for keeping elements.

    Returns:
        Series: Adjusted weights after discarding and redistributing.
    """
    if isinstance(weights, dict):
        weights = pd.Series(weights)

    if np.round(weights.sum(), 1) != 1.0:
        raise ValueError("Expecting weights (that sum to 1) - sum is %s" % weights.sum())

    # Separate weights below the threshold
    above_threshold = weights[weights >= threshold]
    below_threshold = weights[weights < threshold]

    if len(above_threshold) < n_min:
        #print("WARNING:No weights above the limit. All weights have been discarded.")
        return None if none_if_fail else weights.to_dict()

    # Redistribute the sum of discarded weights to remaining weights proportionally
    discarded_sum = below_threshold.sum()
    redistributed_weights = above_threshold + (above_threshold / above_threshold.sum()) * discarded_sum

    # Normalize weights to ensure they sum to 1
    result = np.round(redistributed_weights / redistributed_weights.sum(), 4).to_dict()
    if discarded_sum > 0:
        result = {**result, **{k:0 for k in below_threshold.keys()}}
    return result


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
    Sets temp['selected'] based on information discreteness
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
    Sets temp['stat'] with id over a given period.
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
            idsc = prc.pct_change(1).apply(lambda x: calc_information_discreteness(x.dropna()))
            idsc = idsc.loc[idsc < 0]
        else:
            idsc = prc.iloc[0]
        
        target.temp["stat"] = idsc
        
        return True


class AlgoSelectIDRank(AlgoStack):
    """
    Sets temp['selected'] based on rank with a momentum & information discreteness.
    """
    def __init__(
        self,
        n,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        sort_descending=False,
        all_or_none=False,
        scale = 1
    ):
        super(AlgoSelectIDRank, self).__init__(
            AlgoStatIDRank(lookback=lookback, lag=lag, scale=scale),
            SelectN(n=n, sort_descending=sort_descending, all_or_none=all_or_none),
        )


class AlgoStatIDRank(Algo):
    def __init__(self, lookback=pd.DateOffset(months=3), lag=pd.DateOffset(days=0), scale=1):
        super(AlgoStatIDRank, self).__init__()
        self.lookback = lookback
        self.lag = lag
        self.scale = scale

    def __call__(self, target):
        selected = target.temp["selected"]
        t0 = target.now - self.lag
        if target.universe[selected].index[0] > t0:
            return False
        prc = target.universe.loc[t0 - self.lookback : t0, selected]
        
        if prc.iloc[0].notna().sum() > 0:
            rank_mt = prc.apply(lambda x: x.dropna().iloc[-1]/x.dropna().iloc[0]-1)
            rank_mt = rank_mt.loc[rank_mt > 0].rank(ascending=False)
            rank_id = prc.pct_change(1).apply(lambda x: calc_information_discreteness(x.dropna()))
            rank_id = rank_id.loc[rank_id < 0].rank(ascending=True)
            rank = rank_mt + self.scale * rank_id
            rank = rank.dropna()
            if len(rank) == 0:
                rank = prc.iloc[0]
        else:
            rank = prc.iloc[0]
        
        target.temp["stat"] = rank
        
        return True


class AlgoSelectFinRatio(AlgoStack):
    """
    Sets temp['selected'] based on a financial ratio filter.
    """
    def __init__(
        self,
        df_ratio, # df of financial ratio such as PER
        n, # number of elements to select
        lookback_days=pd.DateOffset(days=0),
        sort_descending=False,
        all_or_none=False,
    ):
        super(AlgoSelectFinRatio, self).__init__(
            AlgoStatFinRatio(df_ratio, lookback_days=lookback_days),
            SelectN(n=n, sort_descending=sort_descending, all_or_none=all_or_none),
        )


class AlgoStatFinRatio(Algo):
    """
    Sets temp['stat'] with financial ratio
    """
    def __init__(self, df_ratio, lookback_days=pd.DateOffset(days=0)):
        super(AlgoStatFinRatio, self).__init__()
        self.df_ratio = df_ratio
        self.lookback_days = lookback_days

    def __call__(self, target):
        t0 = target.now
        # f-ratios for a date range
        stat = self.df_ratio.loc[t0 - self.lookback_days: t0]
        if stat.size == 0:
            return False
            
        stat = stat.mean()
        stat = stat.loc[stat > 0]
        if len(stat) == 0:
            return False

        target.temp["stat"] = stat
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
        if t0 < target.universe.index[0]:
            return False
        else:
            return True


class SelectN(Algo):
    """
    Custom SelectN with an additional arg threshold
    """

    def __init__(self, n, sort_descending=True, all_or_none=False, filter_selected=False, 
                 threshold=None):
        super(SelectN, self).__init__()
        if n < 0:
            raise ValueError("n cannot be negative")
        self.n = n
        self.ascending = not sort_descending
        self.all_or_none = all_or_none
        self.filter_selected = filter_selected
        self.threshold = threshold

    def __call__(self, target):
        stat = target.temp["stat"].dropna()
        if self.filter_selected and "selected" in target.temp:
            stat = stat.loc[stat.index.intersection(target.temp["selected"])]
        if self.threshold is not None:
            stat = stat.loc[stat > self.threshold]
        stat.sort_values(ascending=self.ascending, inplace=True)

        # handle percent n
        keep_n = self.n
        if self.n < 1:
            keep_n = int(self.n * len(stat))

        sel = list(stat[:keep_n].index)

        if self.all_or_none and len(sel) < keep_n:
            sel = []

        target.temp["selected"] = sel

        return True


class SelectMomentum(AlgoStack):
    """
    Custom SelectMomentum with an additional arg threshold
    """

    def __init__(
        self,
        n,
        lookback=pd.DateOffset(months=3),
        lag=pd.DateOffset(days=0),
        sort_descending=True,
        all_or_none=False,
        threshold=None
    ):
        super(SelectMomentum, self).__init__(
            StatTotalReturn(lookback=lookback, lag=lag),
            SelectN(n=n, sort_descending=sort_descending, all_or_none=all_or_none, threshold=threshold),
        )


class RedistributeWeights(Algo):
    """
    Modifies temp['weights'] based on weight threshold.
    discards elements below the threshold, and redistributes the sum of discarded weights 
    to the remaining elements proportionally based on their original weights.
    """

    def __init__(self, threshold=0.1, n_min=1, false_if_fail=True):
        super(RedistributeWeights, self).__init__()
        self.threshold = threshold
        self.n_min = n_min
        self.false_if_fail = false_if_fail

    def __call__(self, target):
        if "weights" not in target.temp:
            return True

        tw = target.temp["weights"]
        if len(tw) == 0:
            return True

        tw = redistribute_weights(tw, self.threshold, self.n_min, self.false_if_fail)
        if tw is None:
            return False
        else:
            target.temp["weights"] = tw

        return True
