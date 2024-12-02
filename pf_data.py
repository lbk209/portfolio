path_data = 'data'
path_tran = 'transaction'

# Universe: equity pool, price file, daily/monthly, ticker
kwargs_dm = ['universe', 'file', 'tickers', 'daily'] # kwargs of DataManager
UNIVERSES = dict(
    UV_K200 = ['kospi200', 'kospi200_prices', 'KRX/INDEX/STOCK/1028', True],
    UV_KRX = ['krx', 'krx_prices', 'KOSPI,KOSDAQ', True],
    UV_LIQ = ['krx', 'krx_liq_prices', 'KOSPI,KOSDAQ', True],
    UV_WTR = ['etf', 'etfs_weather', 'ETF/KR', True],
    UV_ETF = ['etf', 'etfs_all', 'ETF/KR', True],
    UV_FUND = ['fund', 'funds_prices', 'funds_info', False],
    #UV_FUND = ['fund', 'test_funds_prices', 'test_funds_info', False], # for testing
    UV_FCTR = ['yahoo', 'etfs_factors', None, True]
)
UNIVERSES = {k: {**dict(zip(kwargs_dm, v)), 'path':path_data} for k,v in UNIVERSES.items()}


# Portfolio strategy (kwargs of PotfolioManager)
STRATEGIES = dict(
    MMT = dict(method_select='Momentum', method_weigh='Equally', sort_ascending=False, n_tickers=5, lookback='1y', lag='1w'),
    PER = dict(method_select='F-ratio', method_weigh='Equally', sort_ascending=True, n_tickers=20, lookback='2m', lag=0),
    # 'Selected' works with additional ticker list
    WTR = dict(method_select='Selected', method_weigh='Equally'), # freq6m
    LIQ = dict(method_select='Selected', method_weigh='Equally'),
    TDF = dict(method_select='Selected', method_weigh='Equally'),
    HANA= dict(method_select='Selected', method_weigh='InvVol', lookback='2y', lag=0), # freq2y
    FCTR= dict(method_select='Selected', method_weigh='MeanVar', lookback='1q', lag=0), # freq1q
    KRX = dict(method_select='Momentum', method_weigh='Equally', sort_ascending=False, n_tickers=5, lookback='1y', lag='1m')
)

# Transaction file
RECORDS = dict(
    MMT = 'pf_k200_momentum',
    PER = 'pf_k200_per',
    WTR = 'pf_wtr_static',
    LIQ = 'pf_liq_static',
    TDF = 'pf_tdf_static',
    #TDF = 'test_pf_tdf_static', # for testing
    HANA= 'pf_hana_static',
    FCTR= 'pf_fctr_static',
    KRX = 'test_pf_krx_momentum'
)
STRATEGIES = {k: {**STRATEGIES[k], 'file':RECORDS[k], 'path':path_tran} for k,v in STRATEGIES.items()}


# kwargs of PortfolioData
PORTFOLIOS = [
    ('MMT', 'UV_K200'), 
    ('PER', 'UV_K200'), 
    ('WTR', 'UV_WTR'), # modified all weather
    ('LIQ', 'UV_LIQ'), 
    ('TDF', 'UV_FUND'), 
    ('HANA', 'UV_FUND'), 
    ('FCTR', 'UV_FCTR'), # factor intesting with etf
    ('KRX', 'UV_KRX') # for testing
]
PORTFOLIOS = {x[0]: {'strategy':x[0], 'universe':x[1]} for x in PORTFOLIOS}
