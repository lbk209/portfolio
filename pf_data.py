path_data = 'data'
path_tran = 'transaction'

# Universe: equity pool, file, price/rate
kwargs_dm = ['universe', 'file', 'upload_type'] # kwargs of DataManager
UNIVERSES = dict(
    UV_K200 = ['kospi200', 'kospi200_prices', 'price'],
    UV_KRX = ['krx', 'krx_prices', 'price'],
    UV_LIQ = ['krx', 'krx_liq_prices', 'price'],
    UV_WTR = ['etf', 'etfs_weather', 'price'],
    UV_ETF = ['etf', 'etfs_all', 'price'],
    UV_IRP = ['file', 'funds_irp', 'rate'],
    UV_HANA = ['file', 'funds_kebhana', 'rate'],
    UV_FCTR = ['yahoo', 'etfs_factors', 'price']
)
UNIVERSES = {k: {**dict(zip(kwargs_dm, v)), 'path':path_data} for k,v in UNIVERSES.items()}

MONTHLY = ['UV_IRP', 'UV_HANA'] # universes of monthly price
UNIVERSES = {k: {**v, 'daily':False} if k in MONTHLY else v for k,v in UNIVERSES.items()}


# Portfolio strategy (kwargs of PotfolioManager)
STRATEGIES = dict(
    MOM = dict(method_select='Momentum', method_weigh='Equally', sort_ascending=False, n_assets=5, lookback='1y', lag='1w'),
    PER = dict(method_select='F-ratio', method_weigh='Equally', sort_ascending=True, n_assets=20, lookback='2m', lag=0),
    WTR = dict(method_select='All', method_weigh='Equally'),
    LIQ = dict(method_select='Selected', method_weigh='Equally'),
    IRP = dict(method_select='Selected', method_weigh='Equally'),
    HANA= dict(method_select='Selected', method_weigh='Equally'),
    FCTR= dict(method_select='Selected', method_weigh='Equally'),
    KRX = dict(method_select='Momentum', method_weigh='Equally', sort_ascending=False, n_assets=5, lookback='1y', lag='1m')
)

# Transaction file
RECORDS = dict(
    MOM = 'pf_k200_momentum',
    PER = 'pf_k200_per',
    WTR = 'pf_wtr_static',
    LIQ = 'pf_liq_static',
    IRP = 'pf_irp_static',
    HANA= 'pf_hana_static',
    FCTR= 'pf_fctr_static',
    KRX = 'test_pf_krx_momentum'
)
STRATEGIES = {k: {**STRATEGIES[k], 'file':RECORDS[k], 'path':path_tran} for k,v in STRATEGIES.items()}


# kwargs of PotfolioManager
PORTFOLIOS = [
    ('MOM', 'UV_K200'), 
    ('PER', 'UV_K200'), 
    ('WTR', 'UV_WTR'), # modified all weather
    ('LIQ', 'UV_LIQ'), 
    ('IRP', 'UV_IRP'), 
    ('HANA', 'UV_HANA'), 
    ('FCTR', 'UV_FCTR'), # factor intesting with etf
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

    def review(self, space=None):
        """
        get list of portfolios, strategies or universes
        """
        space = 'P' if space is None else space[0].upper()
        if space == 'U':
            args = [self.universes, 'Universe']
        elif space == 'S':
            args = [self.strategies, 'Strategy']
        else: # default portfolio names
            args = [self.portfolios, 'Portfolio']
        return self._print_items(*args)

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

    def _print_items(self, items, space):
        if items is None:
            return print(f'ERROR: No {space} set')
        else:
            return print(f"{space}: {', '.join(items.keys())}")
        
    def _get_item(self, name, data):
        """
        name: universe name
        """
        try:
            return data[name]
        except KeyError as e:
            return print(f'ERROR: No {e}')