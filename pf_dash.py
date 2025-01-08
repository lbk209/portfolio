from dash import Dash, html, dcc, Output, Input
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import plotly.express as px
import random

# Initialize the Dash app
external_stylesheets = [dbc.themes.CERULEAN, 
                        #dbc.themes.BOOTSTRAP,
                        dbc.icons.FONT_AWESOME,
                        dbc.icons.BOOTSTRAP]


def get_data(df, start=None, base=1000):
    """
    Preprocess data to make it JSON-serializable and store it in a JavaScript variable
    """
    default = {
        'price': df.to_dict('records'),
        'index': df.index.tolist(),
    }
    start = df.apply(lambda x: x[x.notna()].index.min()).max() if start is None else start
    normalized_df = df.apply(lambda x: x / x.loc[start] * base).loc[start:]
    compare = {
        'price': normalized_df.to_dict('records'),
        'index': normalized_df.index.tolist(),
    }
    return {'default': default, 'compare': compare}, start


def get_title(title, compare, cost):
    if compare:
        title_comp = '상대 가격'
    else:
        title_comp = '각 최종 결산 기준'

    if cost:
        title_cost = '비용 고려'
    else:
        title_cost = None

    title = f'{title} ({title_comp}'
    title = f'{title}, {title_cost})' if title_cost else f'{title})'
    return title


def update_tickers(tickers, options, option_all='all'):
    if option_all in tickers:
        options = [{**x, 'disabled':True} for x in options]
    else:
        options = [{**x, 'disabled':False} for x in options]
    return options


def update_price_data(tickers, data_prc, base=1000, option_all='all', n_default=5):
    """
    process data and save to dcc.Store
    """
    fees = list(data_prc.keys())
    data_p = {k:v for k,v in data_prc.items()}
    # update data_p with tickers
    if len(tickers) == 0:
        tickers = random.sample(data_prc[fees[0]].columns.to_list(), n_default)
    
    if option_all not in tickers:
        for fee in fees:
            df = data_prc[fee].loc[:, tickers]
            data_p[fee] = df.loc[df.notna().any(axis=1)]

    data = {'default':dict(), 'compare':dict(), 'fees':fees}
    start = None
    for fee, df in data_p.items():
        p_data, start = get_data(df, start)
        data['default'][fee] = p_data['default']
        data['compare'][fee] = p_data['compare']
    return data
    

def update_price_plot(data, cost, compare):
    if data is None:
        return px.line()
        
    fees = data['fees']
    fee = fees[1] if cost else fees[0]
    kind = 'compare' if compare else 'default'
    dat = data[kind][fee]
    df = pd.DataFrame(dat['price'], index=dat['index'])
    title = get_title('펀드 가격 추이', compare, cost)
    return px.line(df, title=title, height=300)


def create_app(data_prc, options, option_all='All', 
               n_default=5, base=1000,
               title="Managed Funds",
               external_stylesheets=external_stylesheets,
               debug=True):

    dropdown_option = [{'label':option_all, 'value':option_all}]
    dropdown_option += options

    app = Dash(__name__, title=title, external_stylesheets=external_stylesheets)
    
    # tabs
    tabs_contents = [
        dbc.Tab(dcc.Graph(id='price-plot'), label='가격'),
        dbc.Tab(dcc.Graph(id='return-plot'), label='수익률'),
    ]
    tabs = dbc.Tabs(tabs_contents)
    
    # layout
    app.layout = dbc.Container([
        html.Br(),
        dbc.Row(tabs),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='ticker-dropdown',
                    options=dropdown_option,
                    value=[dropdown_option[0]['value']],
                    multi=True,
                    placeholder="Select tickers",
                ),
                #width=3
            ),
            dbc.Col(
                daq.BooleanSwitch(
                    id='compare-boolean-switch',
                    on=False
                ),
                width="auto"),
            dbc.Col(
                daq.BooleanSwitch(
                    id='cost-boolean-switch',
                    on=False
                ),
                width="auto"),
        ],
            justify="center",
            align="center",
            className="mb-3"
        ),
        dcc.Store(id='price-data'),
        dbc.Tooltip(
            '상대 비교',
            target='compare-boolean-switch',
            placement='bottom'
        ),
        dbc.Tooltip(
            '수수료 적용',
            target='cost-boolean-switch',
            placement='bottom'
        )
    ])

    @app.callback(
        Output('ticker-dropdown', 'options'),
        Input('ticker-dropdown', 'value'),
    )
    def _update_tickers(tickers):
        return update_tickers(tickers, dropdown_option, option_all)
    
    
    @app.callback(
        Output('price-data', 'data'),
        Input('ticker-dropdown', 'value'),
    )
    def _update_price_data(tickers):
        """
        process data and save to dcc.Store
        """
        return update_price_data(tickers, data_prc, base=base, option_all=option_all, n_default=n_default)
        
    
    @app.callback(
        Output('price-plot', 'figure'),
        Input('price-data', 'data'),
        Input('cost-boolean-switch', 'on'),
        Input('compare-boolean-switch', 'on')
    )
    def _update_price_plot(data, cost, compare):
        return update_price_plot(data, cost, compare)
        
    return app.run_server(debug=debug)