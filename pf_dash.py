from dash import Dash, html, dcc, Output, Input
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from scipy.stats import gaussian_kde

from pf_utils import calculate_hdi


# Initialize the Dash app
external_stylesheets = [dbc.themes.CERULEAN, 
                        #dbc.themes.BOOTSTRAP,
                        dbc.icons.FONT_AWESOME,
                        dbc.icons.BOOTSTRAP]


def get_data(df, start=None, base=1000):
    """
    Preprocess data to make it JSON-serializable and to store it in a JavaScript variable by update_price_data
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


def update_price_data(tickers, data_prc, base=1000, option_all='all'):
    """
    process data and save to dcc.Store
    """
    fees = list(data_prc.keys())
    data_p = {k:v for k,v in data_prc.items()}
    if len(tickers) == 0:
        return None
    
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


def get_inference(file, path, var_name='total_return', tickers=None,
                  n_points=200, hdi_prob=0.94, error=0.999):
    """
    file: inference data file
    """
    f = f'{path}/{file}'
    with open(f, 'rb') as handle:
        data = pickle.load(handle)

    posterior = data['trace'].posterior
    if tickers is None:
        coords = data['coords']
    else:
        coords = {'ticker': tickers}
    
    # Average over the chain dimension, keep the draw dimension
    averaged_data = posterior[var_name].sel(**coords).mean(dim="chain")
    
    # Convert to a DataFrame for Plotly
    df_dst = (averaged_data.stack(sample=["draw"])  # Combine draw dimension into a single index
              .to_pandas()  # Convert to pandas DataFrame
              .T)
    
    # Example: KDE computation for the DataFrame
    kde_data = []  # To store results
    x_values = np.linspace(df_dst.min().min(), df_dst.max().max(), n_points)  # Define global x range
    
    for ticker in df_dst.columns:
        ticker_samples = df_dst[ticker].values  # Extract samples for the ticker
        
        # Compute KDE
        kde = gaussian_kde(ticker_samples)
        density = kde(x_values)  # Compute density for the range
        
        # Store results in a DataFrame
        kde_data.append(pd.DataFrame({ticker: density}, index=x_values))
    
    # Combine all KDE data into a single DataFrame
    df_dst = pd.concat(kde_data, axis=1)
    # remove small number of density
    xlims = calculate_hdi(df_dst, error)
    cond = lambda x: (x.index > xlims[x.name]['x'][0]) & (x.index < xlims[x.name]['x'][1])
    df_dst = df_dst.apply(lambda x: x.loc[cond(x)])
    
    # Calculate the HDI for each ticker
    hdi_lines = calculate_hdi(df_dst, hdi_prob)

    #return df_dst, hdi_lines
    return {
        'density': df_dst.to_dict('records'),
        'x': df_dst.index.tolist(),
        'interval': hdi_lines,
        'hdi_prob': hdi_prob,
        'var_name': var_name
    }
        

def update_inference_data(tickers, data_inf, option_all='all'):
    df_dst = pd.DataFrame(data_inf['density'], index=data_inf['x'])
    hdi_lines = data_inf['interval']
    if len(tickers) == 0:
        return None

    if option_all not in tickers: 
        df_dst = df_dst[tickers]
        hdi_lines = {k:v for k,v in hdi_lines.items() if k in tickers}
    
    return {
        'density': df_dst.to_dict('records'),
        'x': df_dst.index.tolist(),
        'interval': hdi_lines,
        'hdi_prob': data_inf['hdi_prob'],
        'var_name': data_inf['var_name']
    }


def update_inference_plot(data, fund_name=None):
    if data is None:
        return px.line()
    
    var_name = data['var_name']
    hdi_prob = data['hdi_prob']
    hdi_lines = data['interval']
    df_dst = pd.DataFrame(data['density'], index=data['x'])
    
    title=f"Density of {var_name.upper()} (with {hdi_prob:.0%} Interval)"
    fig = px.line(df_dst, title=title)
    fig.update_layout(
        xaxis=dict(title=var_name),
        yaxis=dict(
            title='',             # Remove y-axis title (label)
            showticklabels=False  # Hide y-tick labels
        ),
        hovermode = 'x unified',
        legend=dict(title='')
    )
    
    # Get the color mapping of each ticker from the plot
    colors = {trace.name: trace.line.color for trace in fig.data}
    hover_text = {trace.name: hdi_lines[trace.name]['x'] for trace in fig.data}
    hover_text = {k: f'{v[0]:.3f} ~ {v[1]:.3f}' for k,v in hover_text.items()}
    
    # update trace name after colors creation
    if fund_name is not None:
        fig.for_each_trace(lambda x: x.update(name=fund_name[x.name]))
    
    # Add horizontal hdi_lines as scatter traces with line thickness, transparency, and markers
    for tkr, vals in hdi_lines.items():
        fig.add_trace(go.Scatter(
            x=vals['x'], y=vals['y'],
            mode="lines+markers",            # Draw lines with markers
            line=dict(color=colors[tkr], width=5),  # Adjust thickness, dash style, and transparency
            marker=dict(size=10, symbol='line-ns-open', color=colors[tkr]),  # Customize marker style
            opacity=0.3, 
            legendgroup=tkr,                 # Group with the corresponding data
            showlegend=False                 # Do not display in the legend
        ))
        
    for trace in fig.data:
        if trace.showlegend:
            text = hover_text[trace.legendgroup]
            trace.update(hovertemplate=f"{text} {trace.name}<extra></extra>")
        else:
            trace.update(hoverinfo='skip')  # Exclude from hover text
    
    return fig


def create_app(df_prices, df_prices_fees, tickers=None, fund_name=None,
               option_all='All', base=1000,
               title="Managed Funds", height=500, legend=False, length=20,
               external_stylesheets=external_stylesheets,
               debug=False):

    if tickers is None:
        tickers = df_prices.columns.to_list()
    data_prc = {
        'before fees':df_prices[tickers], 
        'after fees':df_prices_fees[tickers]
    }

    # create dropdown options
    options = [{'label':v, 'value':v} for v in tickers]
    if fund_name is not None:
        options = [{**x, 'title':fund_name[x['value']], 'search':fund_name[x['value']]} for x in options]
    dropdown_option = [{'label':option_all, 'value':option_all}]
    dropdown_option += options

    app = Dash(__name__, title=title, external_stylesheets=external_stylesheets)

   # tabs
    tabs_contents = [
        dbc.Tab(dcc.Graph(id='price-plot'), label='가격', tab_id='tab-1'),
        dbc.Tab(dcc.Graph(id='return-plot'), label='수익률', tab_id='tab-2'),
    ]
    tabs = dbc.Tabs(tabs_contents, id='tabs')
    
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
        return update_price_data(tickers, data_prc, base=base, option_all=option_all)
        
    
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


    def add_tab(new_tab):
        for row in app.layout.children:
            if isinstance(row, dbc.Row):
                if isinstance(row.children, dbc.Tabs):
                    labels = [x.label for x in row.children.children]
                    if new_tab.label in labels:
                        print(f"ERROR: tab '{new_tab.label}' already exits")
                        return False
                    else:
                        new_tab.tab_id = f'tab-{len(labels)+1}'
                        row.children.children.append(new_tab)
                        return True
    app.add_tab = add_tab
    
    return app


def add_density_plot(app, file=None, path=None, tickers=None, fund_name=None,
                     n_points=500, error=0.999, option_all='All'):
    data_inf = get_inference(file, path, tickers=tickers, n_points=n_points, error=error)

    # update layout of the app
    new_tab = dbc.Tab(dcc.Graph(id='density-plot'), label='추정', 
                      label_class_name="tab-label new-badge-label") # add new badge

    # Locate the Row containing Tabs and append the new Tab
    if not app.add_tab(new_tab):
        return None # see add_tab for err msg
        
    # Add density-data Store to the layout
    app.layout.children.append(
        dcc.Store(id='density-data')
    )

    @app.callback(
        Output('cost-boolean-switch', 'on'),
        Output('compare-boolean-switch', 'on'),
        Input("tabs", "active_tab"),
        Input('cost-boolean-switch', 'on'),
        Input('compare-boolean-switch', 'on')
        
    )
    def switch_tab(at, cost, compare):
        if at == new_tab.tab_id:
            return (True, False)
        else:
            return (cost, compare)

    
    @app.callback(
        Output('density-data', 'data'),
        Input('ticker-dropdown', 'value')
    )
    def _update_inference_data(tickers):
        """
        process data and save to dcc.Store
        """
        return update_inference_data(tickers, data_inf, option_all=option_all)
        
    
    @app.callback(
        Output('density-plot', 'figure'),
        Input('density-data', 'data')
    )
    def _update_inference_plot(data):
        return update_inference_plot(data, fund_name)
        