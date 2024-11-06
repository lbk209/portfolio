path_data = 'data'
path_tran = 'transaction'

# Universe: equity pool, file, price/rate
kwargs_dm = ['universe', 'file', 'upload_type'] # kwargs of DataManager
UNIVERSE = dict(
    UV_K200 = ['kospi200', 'kospi200_prices', 'price'],
    UV_KRX = ['krx', 'krx_prices', 'price'],
    UV_LIQ = ['krx', 'krx_liq_prices', 'price'],
    UV_ETF = ['etf', 'etfs_all', 'price'],
    UV_IRP = ['file', 'funds_irp', 'rate'],
    UV_HANA = ['file', 'funds_kebhana', 'rate'],
    UV_FACTOR = ['yahoo', 'etfs_factors', 'price']
)
UNIVERSE = {k: {**dict(zip(kwargs_dm, v)), 'path':path_data} for k,v in UNIVERSE.items()}

MONTHLY = ['UV_IRP', 'UV_HANA'] # universe of monthly price
UNIVERSE = {k: {**v, 'daily':False} if k in MONTHLY else v for k,v in UNIVERSE.items()}


# Portfolio strategy (kwargs of Static/Dynamic Potfolio)
STRATEGY = dict(
    MOM = dict(static=False, method_select='Simple', n_assets=5, lookback='1y', lag='1w'),
    PER = dict(static=False, method_select='F-ratio', n_assets=20, lookback='2m', align_axis=None, sort_ascending=True),
    ETF = dict(align_axis=None),
    LIQ = dict(align_axis=None),
    IRP = dict(align_axis=None),
    HANA= dict(align_axis=None),
    KRX = dict(static=False, method_select='Simple', n_assets=5, lookback='1y', lag='1m')
)

# Transaction file
RECORD = dict(
    MOM = 'pf_k200_momentum',
    PER = 'pf_k200_per',
    ETF = 'pf_etf_static',
    LIQ = 'pf_liq_static',
    IRP = 'pf_irp_static',
    HANA= 'pf_hana_static',
    KRX = 'test_pf_krx_momentum'
)
STRATEGY = {k: {**STRATEGY[k], 'file':RECORD[k], 'path':path_tran} for k,v in STRATEGY.items()}


# kwargs of PotfolioManager
PORTFOLIOS = [
    ('MOM', 'UV_K200'), 
    ('PER', 'UV_K200'), 
    ('ETF', 'UV_ETF'), 
    ('LIQ', 'UV_LIQ'), 
    ('IRP', 'UV_IRP'), 
    ('HANA', 'UV_HANA'), 
    ('KRX', 'UV_KRX') # for testing
]
PORTFOLIOS = {x[0]: {'strategy':STRATEGY[x[0]], 'universe':UNIVERSE[x[1]]} for x in PORTFOLIOS}