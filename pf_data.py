path_data = 'data'
path_tran = 'transaction'

# Universe: equity pool, file, price/rate
kwargs_dm = ['universe', 'file', 'upload_type'] # kwargs of DataManager
UNIVERSES = dict(
    UV_K200 = ['kospi200', 'kospi200_prices', 'price'],
    UV_KRX = ['krx', 'krx_prices', 'price'],
    UV_LIQ = ['krx', 'krx_liq_prices', 'price'],
    UV_ETF = ['etf', 'etfs_all', 'price'],
    UV_IRP = ['file', 'funds_irp', 'rate'],
    UV_HANA = ['file', 'funds_kebhana', 'rate'],
    UV_FACTOR = ['yahoo', 'etfs_factors', 'price']
)
UNIVERSES = {k: {**dict(zip(kwargs_dm, v)), 'path':path_data} for k,v in UNIVERSES.items()}

MONTHLY = ['UV_IRP', 'UV_HANA'] # universes of monthly price
UNIVERSE = {k: {**v, 'daily':False} if k in MONTHLY else v for k,v in UNIVERSES.items()}


# Portfolio strategy (kwargs of PotfolioManager)
STRATEGIES = dict(
    MOM = dict(static=False, method_select='Simple', n_assets=5, lookback='1y', lag='1w'),
    PER = dict(static=False, method_select='F-ratio', n_assets=20, lookback='2m', align_axis=None, sort_ascending=True),
    ETF = dict(align_axis=None),
    LIQ = dict(align_axis=None),
    IRP = dict(align_axis=None),
    HANA= dict(align_axis=None),
    KRX = dict(static=False, method_select='Simple', n_assets=5, lookback='1y', lag='1m')
)

# Transaction file
RECORDS = dict(
    MOM = 'pf_k200_momentum',
    PER = 'pf_k200_per',
    ETF = 'pf_etf_static',
    LIQ = 'pf_liq_static',
    IRP = 'pf_irp_static',
    HANA= 'pf_hana_static',
    KRX = 'test_pf_krx_momentum'
)
STRATEGIES = {k: {**STRATEGIES[k], 'file':RECORDS[k], 'path':path_tran} for k,v in STRATEGIES.items()}


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
PORTFOLIOS = {x[0]: {'strategy':x[0], 'universe':x[1]} for x in PORTFOLIOS}


class PortfolioData():
    def __init__(self, portfolios=PORTFOLIOS, strategies=STRATEGIES, universes=UNIVERSES):
        """
        portfolios: dict of portfolios (portfolio name to tuple of strategy and universe)
        """
        self.portfolios = portfolios
        self.strategies = strategies
        self.universes = universes

    def get(self, name, strategy=False, universe=False):
        """
        name: portfolio name
        """
        result = self._get_item(name, self.portfolios)
        if result is None:
            return None

        res_s = self.get_strategy(result['strategy'])
        res_u = self.get_universe(result['universe'])
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

    def _get_item(self, name, data):
        """
        name: universe name
        """
        try:
            return data[name]
        except KeyError as e:
            return print(f'ERROR: No {e}')

    def get_strategy(self, name):
        """
        name: universe name
        """
        return self._get_item(name, self.strategies)

    def get_universe(self, name):
        """
        name: universe name
        """
        return self._get_item(name, self.universes)
