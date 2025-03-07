from dash import Dash, html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle, re
from scipy.stats import gaussian_kde

from pf_utils import calculate_hdi, BayesianEstimator


# Initialize the Dash app
external_stylesheets = [dbc.themes.CERULEAN, 
                        #dbc.themes.BOOTSTRAP,
                        dbc.icons.FONT_AWESOME,
                        dbc.icons.BOOTSTRAP]


def import_categories(file, path, 
                      cols=['거래처','구분1 (계좌)', '코드'], 
                      cols_new=['seller', 'account', 'ticker'], 
                      rename = {'일반':'일반계좌'},
                      cols_intersection=['seller','account'], prefix_intersection='&'):
    """
    import dataframe for options
    """
    df_cat = pd.read_csv(f'{path}/{file}', header=1)
    cols = df_cat.columns if cols is None else cols
    cols_new = cols if cols_new is None else cols_new
    df_cat = df_cat[cols].rename(columns=dict(zip(cols, cols_new))).dropna().reset_index(drop=True)
    if isinstance(rename, dict):
        df_cat = df_cat.map(lambda x: rename[x] if x in rename.keys() else x)
    if isinstance(cols_intersection, list):
        df_cat[cols_intersection] = df_cat[cols_intersection].map(lambda x: f'{prefix_intersection}{x}')
    return df_cat


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


def update_price_data(tickers, data_prc, base=1000):
    """
    process data and save to dcc.Store
    values: option values selected
    data_prc: dict of dfs. see create_app
    """
    fees = list(data_prc.keys())
    data_p = {k:v for k,v in data_prc.items()}
    if len(tickers) == 0:
        return None
    
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
    fig.update_yaxes(automargin=True)
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
    fig.update_yaxes(automargin=True)
    
    return fig


def get_inference(file, path, var_name='total_return', tickers=None,
                  n_points=500, hdi_prob=0.94, error=0.999):
    """
    file: inference data file
    """
    data = BayesianEstimator.load(file, path)
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
        

def update_inference_data(tickers, data_inf):
    if len(tickers) == 0:
        return None

    df_dst = pd.DataFrame(data_inf['density'], index=data_inf['x'])
    hdi_lines = data_inf['interval']
        
    tickers = df_dst.columns.intersection(tickers)
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
    fig.update_yaxes(automargin=True)
    
    return fig


def get_hdi(file, path, var_name='total_return', tickers=None, to_dict=True):
    """
    file: inference data file
    """
    be = BayesianEstimator.create(file, path)
    df_hdi = be.bayesian_summary(var_name).droplevel(0)
    if tickers is not None:
        tkr = pd.Index(tickers).difference(df_hdi.index)
        if tkr.size > 0:
            print(f'WARNING: Tickers set to None as {tkr.size} missing tickers')
        else:
            df_hdi = df_hdi.loc[tickers]
    return df_hdi.to_dict() if to_dict else df_hdi


def update_hdi_data(tickers, data, sort_by='mean', ascending=False):
    if len(tickers) == 0:
        return None
        
    df_hdi = pd.DataFrame().from_dict(data)
    if sort_by:
        df_hdi = df_hdi.sort_values(sort_by, ascending=ascending)
        
    tickers = df_hdi.index.intersection(tickers)
    df_hdi = df_hdi.loc[tickers]
    
    return df_hdi.to_dict()


def update_hdi_plot(data, fund_name=None, sort_by='mean', ascending=False, line_width=10,
                    cols_hdi = ['hdi_3%', 'hdi_97%']):
    if data is None:
        return px.line()
    
    df_hdi = pd.DataFrame().from_dict(data)
    if sort_by:
        df_hdi = df_hdi.sort_values(sort_by, ascending=ascending)
    col_l, col_u = cols_hdi
    # get intervals for each ticker
    hdi = df_hdi.apply(lambda x: [x[col_l], x[col_u]], axis=1)
    sr_mean = df_hdi['mean']
    
    fig = go.Figure()
    for ticker in df_hdi.index:
        # Plot the HDI range (hdi_3% to hdi_97%)
        fig.add_trace(go.Scatter(
            x=[ticker, ticker], 
            y=hdi[ticker],
            mode='lines', 
            uid=ticker,
            name=ticker if fund_name is None else fund_name[ticker],  
            line=dict(width=line_width),
            legendgroup=ticker,  # Group with mean marker
            showlegend=True
        ))
    
        # Plot the mean marker, grouped with its HDI range but hidden from legend
        fig.add_trace(go.Scatter(
            x=[ticker, ticker], 
            y=[sr_mean[ticker], sr_mean[ticker]],
            mode='markers', 
            uid=ticker, # ok to set uid same as hdi
            name=ticker if fund_name is None else fund_name[ticker], 
            marker=dict(color="gray", size=round(line_width*1.5), symbol='line-ew-open'),
            legendgroup=ticker,  # Same group as HDI range
            showlegend=False  # Hide from legend
        ))
    
    fig.update_layout(
        title="94% Interval of 3-Year Return",
        xaxis=dict(
            title='',             # Remove x-axis title (label)
            showticklabels=False  # Hide x-tick labels
        ),
        hoverlabel=dict(
            bgcolor="rgba(255, 255, 255, 0.5)", # opacity not works
            #font_size=16,
            #font_family="Rockwell"
        )
    )
    
    for trace in fig.data:
        if trace.showlegend:
            text = [str(x) for x in hdi[trace.uid]]
            text = '~'.join(text)
            trace.update(hovertemplate=f"Interval: {text}<br>{trace.name}<extra></extra>")
        else:
            text = sr_mean[trace.uid]
            trace.update(hovertemplate=f"Mean: {text}<br>{trace.name}<extra></extra>")
    fig.update_yaxes(automargin=True)
            
    return fig


def update_scatter_plot(data, category, n_quantiles=3):
    if data is None:
        return px.scatter()
    
    df = pd.DataFrame().from_dict(data)
    fig = px.scatter(df, x='mean', y='sd',
                 custom_data=['name', 'hdi_3%', 'hdi_97%'],
                 #hover_data='name',
                 #color='g1'
                 #color='seller'
                 color=category, symbol=category,
                 #error_x="error_x",
                 size='sharpe'
                )
    # add quantile lines
    q = np.linspace(0, 1, n_quantiles+1)[1:-1].tolist()
    kw = dict(line_width=0.5)
    _ = [fig.add_vline(x=x, **kw) for x in df['mean'].quantile(q)]
    _ = [fig.add_hline(y=x, **kw) for x in df['sd'].quantile(q)]
    
    fig.update_xaxes(autorange='reversed')
    fig.update_layout(legend = {'title':{'text':''}},
                      #width=1000, height=500, 
                      title_text='3년 평균 수익률 순위 (94% 확률 추정)',
                      xaxis=dict(
                            title=dict(
                                text="평균 순위"
                            )
                      ),
                      yaxis=dict(
                            title=dict(
                                text="편차 순위"
                            )
                      ),
    )
    fig.update_traces(
        hovertemplate =
                    "%{customdata[0]}<br>" +
                    "수익률 순위: 평균 %{x}, 편차 %{y}<br>"
                    "수익률 구간: %{customdata[1]} ~ %{customdata[2]}<extra></extra>"
    )

    return fig


def create_app(df_prices, df_prices_fees, df_categories,
               fund_name=None, tickers=None,
               title="Managed Funds", height=500, legend=False, length=20,
               base=1000,
               external_stylesheets=external_stylesheets,
               debug=False):
    """
    df_prices/df_prices: df of timeindex and col tickers
    df_categories: df of index ticker and col categories. all columns to be options for catetory
    """
    tickers_all = df_prices.columns.intersection(df_prices_fees.columns)
    tickers_all = tickers_all.intersection(df_categories.index)
    if tickers is None:
        tickers = tickers_all.to_list()
    else:
        tickers = tickers_all.intersection(tickers).to_list()

    data_prc = {
        'before fees':df_prices[tickers], 
        'after fees':df_prices_fees[tickers]
    }
    fund_name = {x:x for x in tickers} if fund_name is None else fund_name
    df_categories = df_categories.loc[tickers]
    
    # create dropdown options
    dropdown_category = [{'label':x, 'value':x, 'title':x, 'search':x} for x in df_categories.columns]
    dm = DropdownManager(tickers, fund_name)
    dm.create_all()
    dm.create_tickers()
    dropdown_group = dm.get_options()
    group_previous = '&previous_selection'
    
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
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='category-dropdown',
                    options=dropdown_category,
                    value=dropdown_category[0]['value'],
                    clearable=False,
                    placeholder="Select Category",
                ),
                #width=3
            ),
            dbc.Col(
                dcc.Dropdown(
                    id='group-dropdown',
                    options=dropdown_group,
                    value=[dropdown_group[0]['value']],
                    multi=True,
                    placeholder="Select Group",
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
        ),
        dbc.Row(tabs),
    ])


    @app.callback(
        Output('group-dropdown', 'options'),
        Output('group-dropdown', 'value'),
        Input('category-dropdown', 'value'),
        State('group-dropdown', 'value'),
    )
    def _set_category(category, group):
        """
        make selections in old category into new group in new category
        """
        if len(group) > 0 and dm.option_all not in group:
            previous = dm.merge(*group, value=group_previous, add=False, msg=False)
        else:
            previous = None
        
        dm.reset(keep_all=True)
        #dm.create_order(options_order) if options_order is not None else None
        dm.create_from_dict(df_categories[category])

        values = [dm.option_all] # default selection of group for new category
        if previous is not None:
            value_to_ticker, options = previous
            dm.add_options(value_to_ticker, options)
            values = [group_previous] + values
            
        dm.create_tickers()
        dropdown_group = dm.get_options()
        return (dropdown_group, values)

    
    @app.callback(
        #Output('group-dropdown', 'options', allow_duplicate=True),
        Output('group-dropdown', 'value', allow_duplicate=True),
        Input('group-dropdown', 'value'),
        prevent_initial_call=True
    )
    def _update_options(values):
        return dm.update_options(values)

    
    @app.callback(
        Output('price-data', 'data'),
        Input('group-dropdown', 'value'),
    )
    def _update_price_data(values):
        """
        process data and save to dcc.Store
        """
        tickers = dm.get_tickers(values)
        #print(f'tickers of {values}: {len(tickers)}') # testing
        return update_price_data(tickers, data_prc, base=base)
        
    
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
    
    return (app, dm.get_tickers)


def add_density_plot(app, get_tickers, 
                     file=None, path=None, tickers=None, fund_name=None,
                     n_points=500, error=0.999):
    """
    get_tickers: function to get tickers from selected option values
    """
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
        Input('group-dropdown', 'value')
    )
    def _update_inference_data(values):
        """
        process data and save to dcc.Store
        """
        tickers = get_tickers(values)
        return update_inference_data(tickers, data_inf)
        
    
    @app.callback(
        Output('density-plot', 'figure'),
        Input('density-data', 'data')
    )
    def _update_inference_plot(data):
        return update_inference_plot(data, fund_name)


def add_hdi_plot(app, get_tickers, 
                 file=None, path=None, tickers=None, fund_name=None,
                 badge_new=False, **kwargs):
    """
    get_tickers: function to get tickers from selected option values
    kwargs: kwargs for update_hdi_plot
    """
    data_hdi = get_hdi(file, path, tickers=tickers)

    # update layout of the app
    label_class_name = "tab-label new-badge-label" if badge_new else None # add new badge
    new_tab = dbc.Tab(dcc.Graph(id='hdi-plot'), label='HDI', label_class_name=label_class_name)

    # Locate the Row containing Tabs and append the new Tab
    if not app.add_tab(new_tab):
        return None # see add_tab for err msg
        
    # Add hdi-data Store to the layout
    app.layout.children.append(
        dcc.Store(id='hdi-data')
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
        Output('hdi-data', 'data'),
        Input('group-dropdown', 'value')
    )
    def _update_hdi_data(values):
        """
        process data and save to dcc.Store
        """
        tickers = get_tickers(values)
        return update_hdi_data(tickers, data_hdi)
        
    
    @app.callback(
        Output('hdi-plot', 'figure'),
        Input('hdi-data', 'data')
    )
    def _update_hdi_plot(data):
        return update_hdi_plot(data, fund_name, **kwargs)



def add_scatter_plot(app, get_tickers, 
                     file=None, path=None, tickers=None, fund_name=None, df_cat=None,
                     badge_new=False, **kwargs):
    """
    get_tickers: function to get tickers from selected option values
    kwargs: kwargs for update_scatter_plot
    """
    # prepare data for plot
    data_hdi = get_hdi(file, path, tickers=tickers, to_dict=False)
    data_hdi = data_hdi if fund_name is None else data_hdi.join(pd.Series(fund_name, name='name'))
    data_hdi = data_hdi if df_cat is None else data_hdi.join(df_cat)
    xlabel, ylabel = 'mean', 'sd'
    # add sharpe rank for marker size
    df_s = data_hdi.apply(lambda x: x[xlabel]/ x[ylabel], axis=1).rank().rename('sharpe')
    data_hdi = data_hdi.join(df_s)
    # convert mean/sd into respective ranks
    data_hdi[xlabel] = data_hdi[xlabel].rank(ascending=False)
    data_hdi[ylabel] = data_hdi[ylabel].rank()
    cols = ['name', 'mean', 'sd', 'hdi_3%', 'hdi_97%', 'sharpe'] + df_cat.columns.to_list()
    data_hdi = data_hdi[cols].to_dict()
    
    # update layout of the app
    label_class_name = "tab-label new-badge-label" if badge_new else None  # add new badge
    new_tab = dbc.Tab(dcc.Graph(id='scatter-plot'), label='순위', label_class_name=label_class_name)

    # Locate the Row containing Tabs and append the new Tab
    if not app.add_tab(new_tab):
        return None # see add_tab for err msg
        
    # Add hdi-data Store to the layout
    app.layout.children.append(
        dcc.Store(id='scatter-data')
    )

    @app.callback(
        Output('cost-boolean-switch', 'on', allow_duplicate=True),
        Output('compare-boolean-switch', 'on', allow_duplicate=True),
        Input("tabs", "active_tab"),
        Input('cost-boolean-switch', 'on'),
        Input('compare-boolean-switch', 'on'),
        prevent_initial_call=True
    )
    def switch_tab(at, cost, compare):
        if at == new_tab.tab_id:
            return (True, False)
        else:
            return (cost, compare)

    
    @app.callback(
        Output('scatter-data', 'data'),
        Input('group-dropdown', 'value')
    )
    def _update_hdi_data(values):
        """
        process data and save to dcc.Store
        """
        tickers = get_tickers(values)
        return update_hdi_data(tickers, data_hdi, sort_by=False)
        
    
    @app.callback(
        Output('scatter-plot', 'figure'),
        Input('scatter-data', 'data'),
        State('category-dropdown', 'value')
    )
    def _update_scatter_plot(data, cat):
        return update_scatter_plot(data, cat, **kwargs)
        


class DropdownManager():
    def __init__(self, tickers, fund_name=None, prefix_intersection='&'):
        self.tickers = tickers
        self.fund_name = fund_name # dict of ticker to name
        self.prefix_intersection = prefix_intersection
        self.options = list()
        self.value_to_ticker = dict() # option value to tickers
        self.option_all = None
        self.data_order = None
        
    # TODO: check data_order
    def reset(self, *keep, keep_all=True):
        """
        keep: list of values to keep
        """
        keep = list(keep) + [self.option_all] if keep_all else keep
        if len(keep) == 0:
            self.options = list()
            self.value_to_ticker = dict() # option value to tickers
        else: # reset options except for values in keep
            self.options = [x for x in self.options if x['value'] in keep]
            self.value_to_ticker = {k:v for k,v in self.value_to_ticker.items() if k in keep}

    def _check_tickers(self, tickers):
        """
        check tickers by returning tickers only in self.tickers
        """
        return pd.Index(tickers).intersection(self.tickers).to_list()

    def _add_options(self, value_to_ticker, options=None):
        """
        add new options without duplicates
        """
        if options is None:
            prefix = self.prefix_intersection
            options = [self._set_option(x, prefix=prefix) for x in value_to_ticker.keys()]
        else: # check input consistency
            v1 = value_to_ticker.keys()
            v2 = [x['value'] for x in options]
            if set(v1) != set(v2):
                return print('ERROR: Check values in both value_to_ticker and options')
        # set value_to_ticker. existing values updated with new ones     
        self.value_to_ticker = {**self.value_to_ticker, **value_to_ticker}
        # set options
        opt_old = {x['value']:x for x in self.options}
        options = {**opt_old, **{x['value']:x for x in options}}
        self.options = list(options.values())

    def _set_option(self, value, label=None, title=None, search=None, prefix=None):
        """
        create option item for dropdown list
        prefix: prefix of intersection or order
        """
        # prefix sign kept only for value
        label, title, search = [value.lstrip(prefix) if x is None else x for x in [label, title, search]]
        if isinstance(prefix, str) and value.startswith(prefix):
            label = f'{label} ({prefix})'
        search = search.lower()
        return {'label':label, 'value':value, 'title':title, 'search':search}

    def create_from_dict(self, sr_values):
        """
        set option and its ticker list specifically
        sr_values: dict or series of index ticker to value or df of index to categories
        """
        col_tkr, col_grp = 'ticker', 'group'
        # unstack df to series
        sr_values = sr_values.unstack().dropna().droplevel(0) if isinstance(sr_values, pd.DataFrame) else sr_values
        # convert dict to series
        sr_values = pd.Series(sr_values) if isinstance(sr_values, dict) else sr_values
        # create option values to lists of tickers
        value_to_ticker = (sr_values.rename(col_grp).rename_axis(col_tkr).reset_index()
                              .groupby(col_grp)[col_tkr].apply(list).to_dict())
        # add to existig options
        self._add_options(value_to_ticker)

    def create_all(self, option_all='All'):
        """
        make option to select all tickers
        """
        options = [{'label':option_all, 'value':option_all, 
                    'title':option_all, 'search':option_all.lower()}]
        value_to_ticker = {option_all: self.tickers}
        self._add_options(value_to_ticker, options)
        self.option_all = option_all

    def create_tickers(self):
        """
        set tickers to options
        """
        fund_name = self.fund_name
        tkr_dict = {x:x for x in self.tickers}
        if fund_name is not None:
            tkr_dict.update(fund_name)
        
        options = [{'label':k, 'value':k, 'title':v, 'search':v} for k,v in tkr_dict.items()]
        tickers = {x['value']:[x['value']] for x in options}
        self._add_options(tickers, options)

    def create_from_name(self, names):
        """
        make options of words from fund names
        names: list of words to make options
        """
        fund_name = self.fund_name
        if fund_name is None:
            return None
            
        if isinstance(names, str):
            names = [names]
        options, value_to_ticker = list(), dict()
        for value in names:
            option = self._set_option(value, prefix=self.prefix_intersection)
            title = option['title']
            # get ticker list for title
            tickers = [k for k,v in fund_name.items() if title.lower() in v.lower()]
            if len(tickers) > 0:
                options.append(option)
                value_to_ticker[value] = tickers
        if len(options) > 0:
            self._add_options(value_to_ticker, options)

    def create_order(self, sr_rank,
                     value_top='Top20', value_bottom='Bottom20', value_random='Random20',
                     rex_order = r'([^0-9]+)(\d+)', prefix='№'):
        """
        create options from kwargs value_*
        """
        options = list()
        value_to_ticker = dict()
        values_order = dict()  
        # add prefix
        values = [value_top, value_bottom, value_random]
        values = [f'{prefix}{x}' for x in values]
        value_top, value_bottom, value_random = values
        # create options for values
        for value in values:
            if value is None:
                continue
            match = re.match(rex_order, value)
            if match is None:
                print(f'ERROR: Failed to create option {value} as {e}')
                continue
            else:
                _, num = match.groups()
                values_order[value] = int(num)
            option = self._set_option(value, prefix=prefix)
            options.append(option)
            value_to_ticker[value] = list() # set to avoid keyerror in get_tickers
            
        if len(values_order) > 0: # set vars after creating options
            self._add_options(value_to_ticker, options)
            
            def select_tickers(tickers, value):
                # remove tickres not in sr_rank
                tickers = sr_rank.index.intersection(tickers) 
                sr_tkr = sr_rank.loc[tickers].sort_values()
                num = values_order[value]
                if value == value_top:
                    sr_tkr = sr_tkr[:num]
                elif value == value_bottom:
                    sr_tkr = sr_tkr[-num:]
                elif value == value_random:
                    sr_tkr = sr_tkr.sample(num)
                return sr_tkr.index.to_list()
            
            self.data_order = dict(
                prefix = prefix,
                values = values_order,   
                select = select_tickers
            )

    def merge(self, *values_to_merge, value='merged', add=False, msg=True):
        """
        create new value by merging existing values
        """
        tickers = self.get_tickers(values_to_merge)
        if len(tickers) > 0:
            value_to_ticker = {value: tickers}
            option = self._set_option(value, prefix=self.prefix_intersection)
            options = [option]
        else:
            v = ', '.join(values_to_merge)
            return print(f'ERROR: Failed to merge {v}') if msg else None
        
        if add:
            self._add_options(value_to_ticker, options)
        else:
            return (value_to_ticker, options)

    def get_options(self):
        """
        export dropdown options created
        """
        if len(self.options) == 0:
            return list()
        else:
            return self.options

    def update_options(self, values):
        """
        update option value by checking option_all
        """
        options = self.get_options()
        option_all = self.option_all
        prefix_list = [self.prefix_intersection]
        data_order = self.data_order
        if data_order is not None:
            prefix_list.append(data_order['prefix'])

        # return True if value starts with any of prefix_list
        cond = lambda value: pd.Series([value.startswith(x) for x in prefix_list]).any()
        vlist = [x for x in values if not cond(x)] # remove options in prefix_list for if-statement
        if (option_all in vlist) and len(vlist) > 1: 
            if vlist[0] == option_all: # remove option_all if new options added
                values.remove(option_all)
            elif vlist[-1] == option_all: # remove options if option_all added 
                values = [x for x in values if x==option_all or cond(x)] # keep just option_all & options in prefix_list
        return values

    def select_by_order(self, tickers, values):
        data_order = self.data_order
        if data_order is None:
            return tickers
        values_order = data_order['values']
        vals = [x for x in values if x in values_order.keys()]
        if len(vals) == 1:
            value = vals[0]
            select = data_order['select']
            return select(tickers, value)
        else:
            return tickers
        
    def get_tickers(self, values):
        """
        get union/intersection of tickers from list of option values
        values: list of values
        """
        if len(self.options) == 0:
            return list()
        
        # split options of union and intersection
        prefix = self.prefix_intersection
        values_intersection = [x for x in values if x.startswith(prefix)]
        values_union = list(set(values) - set(values_intersection))

        # get tickers from union options
        tickers = list()
        for v in values_union:
            try: # tickers is empty list for options from create_order
                tickers += self.value_to_ticker[v]
            except KeyError:
                continue
        tickers = set(tickers)
        if len(tickers) == 0:
            return list()

        # get tickers from intersection options
        for v in values_intersection:
            try:
                tickers = tickers & set(self.value_to_ticker[v])
            except KeyError:
                continue
        tickers = list(tickers)

        # check if options order exists
        tickers = self.select_by_order(tickers, values_union) 
            
        return self._check_tickers(tickers) # remove tickers not in self.tickers

    def add_options(self, value_to_ticker=None, options=None):
        if value_to_ticker is not None:
            return self._add_options(value_to_ticker, options)