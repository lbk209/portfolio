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


# Portfolio strategy: strategy name to data
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

# Transactions: portfolio to transaction file & path
TRANSACTIONS = dict(
    MMT_2407 = dict(file='pf_k200_momentum'),
    PER_2410 = dict(file='pf_k200_per'),
    WTR_2407 = dict(file='pf_wtr_static'),
    WTR_2412 = dict(file='pf_wtr2412_static'),
    LIQ = dict(file='pf_liq_static'),
    TDF_2406 = dict(file='pf_tdf_static'),
    #TDF = dict(file='test_pf_tdf_static'), # for testing
    HANA_2408= dict(file='pf_hana_static'),
    FCTR= dict(file='pf_fctr_static'),
    KRX = dict(file='test_pf_krx_momentum'),
    TEST = dict(file='test') # for testing
)
TRANSACTIONS = {k: {**v, 'path':path_tran} for k,v in TRANSACTIONS.items()}


# kwargs of PortfolioData
PORTFOLIOS = {
    'MMT_2407': {'strategy': 'MMT', 'universe': 'UV_K200'},
    'PER_2410': {'strategy': 'PER', 'universe': 'UV_K200'},
    'WTR_2407': {'strategy': 'WTR', 'universe': 'UV_WTR'}, # modified all weather
    'WTR_2412': {'strategy': 'WTR', 'universe': 'UV_WTR'},
    'LIQ': {'strategy': 'LIQ', 'universe': 'UV_LIQ'},
    'TDF_2406': {'strategy': 'TDF', 'universe': 'UV_FUND'},
    'HANA_2408': {'strategy': 'HANA', 'universe': 'UV_FUND'},
    'FCTR': {'strategy': 'FCTR', 'universe': 'UV_FCTR'},  # factor intesting with etf
    'KRX': {'strategy': 'KRX', 'universe': 'UV_KRX'}, # for testing
    'TEST': {'strategy': 'WTR', 'universe': 'UV_WTR'}, # for testing
}
PORTFOLIOS = {k: {**v, **TRANSACTIONS[k]} for k,v in PORTFOLIOS.items()}
