path_data = 'data'
path_tran = 'transaction'

# Universe: data for DataManager
UNIVERSES = dict(
    UV_K200 = dict(universe='kospi200', file='kospi200_prices', tickers='KRX/INDEX/STOCK/1028'),
    UV_KRX = dict(universe='krx', file='krx_prices', tickers='KOSPI,KOSDAQ'),
    UV_LIQ = dict(universe='krx', file='krx_liq_prices', tickers='KOSPI,KOSDAQ'),
    UV_WTR = dict(universe='etf', file='etfs_weather', tickers='ETF/KR'),
    UV_ETF = dict(universe='etf', file='etfs_all', tickers='ETF/KR'),
    UV_FUND = dict(universe='fund', file='funds_prices', tickers='funds_info',
                   freq='daily', batch_size=100, check_master=True),
    UV_FNDM = dict(universe='fund', file='fundm_prices', tickers='fundm_info',
                   freq='monthly', batch_size=12, check_master=True), # fund data of monthly bassis
    UV_FCTR = dict(universe='yahoo', file='etfs_factors', tickers=None) # universe defined by tickers each time
)
UNIVERSES = {k: {**v, 'path':path_data} for k,v in UNIVERSES.items()}

# Portfolio strategy: strategy name to data
STRATEGIES = dict(
    MMT = dict(method_select='Momentum', method_weigh='Equally', sort_ascending=False, n_tickers=5, lookback='1y', lag='1w'),
    PER = dict(method_select='F-ratio', method_weigh='Equally', sort_ascending=True, n_tickers=20, lookback='2m', lag=0),
    # 'Selected' works with additional ticker list
    WTR = dict(method_select='Selected', method_weigh='Equally'), # freq6m
    LIQ = dict(method_select='Selected', method_weigh='Equally'),
    TDF = dict(method_select='Selected', method_weigh='Equally', unit_fund=True),
    HANA= dict(method_select='Selected', method_weigh='InvVol', lookback='2y', lag=0, unit_fund=True), # freq2y
    SAVE= dict(method_select='Selected', method_weigh='Equally', lookback='6m', lag=0, unit_fund=True),
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
    HANA_2408 = dict(file='pf_hana_static'),
    SAVE_2503 = dict(file='pf_save_static'),
    SAVE_2504 = dict(file='pf_save2_static'),
    FISA_2504 = dict(file='pf_fisa_static'),
    FCTR= dict(file='pf_fctr_static'),
    #KRX = dict(file='test_pf_krx_momentum'),
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
    'SAVE_2503': {'strategy': 'SAVE', 'universe': 'UV_FUND'},
    'SAVE_2504': {'strategy': 'SAVE', 'universe': 'UV_FUND'},
    'FISA_2504': {'strategy': 'SAVE', 'universe': 'UV_FUND'},
    'FCTR': {'strategy': 'FCTR', 'universe': 'UV_FCTR'},  # factor investing with etf
    #'KRX': {'strategy': 'KRX', 'universe': 'UV_KRX'}, # for testing
}
PORTFOLIOS = {k: {**v, **TRANSACTIONS[k]} for k,v in PORTFOLIOS.items()}
