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
        tickers = data_prc[fees[0]].columns.to_list()
        if len(tickers) > n_default:
            tickers = random.sample(tickers, n_default)
    
    for fee in fees:
        df = data_prc[fee] if option_all in tickers else data_prc[fee].loc[:, tickers]
        data_p[fee] = df.loc[df.notna().any(axis=1)]

    data = {'default':dict(), 'compare':dict(), 'fees':fees}
    start = None
    for fee, df in data_p.items():
        p_data, start = get_data(df, start)
        data['default'][fee] = p_data['default']
        data['compare'][fee] = p_data['compare']
    return data
    

def update_price_plot(data, cost, compare, fund_name=None,
                      height=300, legend=True, length=20):
    if data is None:
        return px.line()
    # get data    
    fees = data['fees']
    fee = fees[1] if cost else fees[0]
    kind = 'compare' if compare else 'default'
    dat = data[kind][fee]
    # build df
    df = pd.DataFrame(dat['price'], index=dat['index'])
    df = df.round()
    if fund_name is not None:
        df.columns = [fund_name[x] for x in df.columns]
    # plot
    title = get_title('펀드 가격 추이', compare, cost)
    fig = px.line(df, title=title, height=height)
    
    # update hoover text
    for i in range(len(fig.data)):
        s = fig.data[i].name
        fig.data[i].name = s[:length]
    fig.update_traces(hovertemplate='%{y:.0f}')
    
    fig.update_layout(**{
        'hovermode': 'x',
        'yaxis': {'title': '가격'},
        'xaxis': {
            'rangeselector': {
                'buttons':[{
                    'count': 3,
                    'label': "3y",
                    'step': "year",
                    'stepmode': "backward"
                },{
                    'step': "all",
                    'label': "All"
                }]
            },
            'rangeslider': {
                'visible': True
            },
            'type': "date",
            'title': None
        },
        'legend':{'title':{'text':''}},
        'showlegend': legend
    })
    return fig


def update_return_plot(data, cost, compare, months_in_year=12, 
                       date_format='%Y-%m-%d', fund_name=None,
                       height=500, length=20):
    if data is None:
        return px.bar()
    fees = data['fees']
    fee = fees[1] if cost else fees[0]
    # reset opacity of selected price data
    if compare:
        dat = data['compare']
        dates = pd.Index(dat[fee]['index'])
        dates = pd.to_datetime(dates)
        dt0 = dates.min().strftime(date_format)
        dt1 = dates.max().strftime(date_format)
        title = f'펀드 수익률 ({dt0} ~ {dt1})'
    else:
        dat = data['default'] 
        title = '펀드 수익률 (각 설정일 기준)'

    df_ret = pd.DataFrame()
    for f in fees:
        prc = dat[f]['price']
        df = pd.DataFrame(prc)
        
        #df = df.apply(lambda x: x.dropna().iloc[-1]/x.dropna().iloc[0]-1).to_frame(f)

        sr_n = df.apply(lambda x: x.dropna().count()) # num of months for each ticker
        df = (df.apply(lambda x: x.dropna().iloc[-1]/x.dropna().iloc[0]-1) # total return
                .to_frame('ttr').join(sr_n.rename('n'))
                .apply(lambda x: (1+x['ttr']) ** (months_in_year/x['n']) - 1, axis=1) # CAGR
                .mul(100).to_frame(f))
        
        df_ret = pd.concat([df_ret, df], axis=1)
    if fund_name is not None:
        df_ret.index = [fund_name[x] for x in df_ret.index]
    
    fig = px.bar(df_ret, title=title, barmode='group', opacity=0.3, height=height)

    # update x-tick labels
    func = lambda x: '<br>'.join([x[i:i+length] for i in range(0, len(x), length)])
    for i in range(len(fig.data)):
        xtl = fig.data[i].x
        #fig.data[i].x = [x[:length] for x in xtl] 
        fig.data[i].x = [func(x) for x in xtl]
        
    
    fig.update_traces(hovertemplate='%{y:.2f}')
    fig.update_traces(marker_opacity=0.8, marker_line_color="black", selector={"name": fee})
    fig.update_layout(**{
        'hovermode': 'x',
        'barmode': 'group',
        'xaxis': {'title':None},
        'yaxis': {'title': '연평균 수익률 (%)' },
        'legend':{'title':{'text':''}},
    })
    
    return fig


def create_app(data_prc, options, option_all='All', fund_name=None,
               n_default=5, base=1000,
               title="Managed Funds", height=500, legend=False, length=20,
               external_stylesheets=external_stylesheets,
               debug=False):

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
                    on=True
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
        return update_price_plot(data, cost, compare, fund_name=fund_name,
                                 height=height, legend=legend, length=length)

    @app.callback(
        Output('return-plot', 'figure'),
        Input('price-data', 'data'),
        Input('cost-boolean-switch', 'on'),
        Input('compare-boolean-switch', 'on')
    )
    def _update_return_plot(data, cost, compare):
        return update_return_plot(data, cost, compare, date_format='%Y-%m-%d', 
                                  fund_name=fund_name, height=height, length=length)

    
    return app.run_server(debug=debug)