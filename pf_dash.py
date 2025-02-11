from dash import Dash, html, dcc, Output, Input
import dash_bootstrap_components as dbc
import dash_daq as daq
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
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
    
    return fig


def get_hdi(file, path, var_name='total_return'):
    """
    file: inference data file
    """
    be = BayesianEstimator.create(file, path)
    df_hdi = be.bayesian_summary(var_name).droplevel(0)
    return df_hdi.to_dict()


def update_hdi_data(tickers, data, sort_by='mean', ascending=False):
    if len(tickers) == 0:
        return None
        
    df_hdi = pd.DataFrame().from_dict(data)
    if sort_by:
        df_hdi = df_hdi.sort_values(sort_by, ascending=ascending)
        
    tickers = df_hdi.index.intersection(tickers)
    df_hdi = df_hdi.loc[tickers]
    
    return df_hdi.to_dict()


def update_hdi_plot(data, fund_name=None, sort_by='mean', ascending=False):
    if data is None:
        return px.line()
    
    df_hdi = pd.DataFrame().from_dict(data)
    if sort_by:
        df_hdi = df_hdi.sort_values(sort_by, ascending=ascending)
    
    fig = go.Figure()
    for ticker in df_hdi.index:
        # Plot the HDI range (hdi_3% to hdi_97%)
        fig.add_trace(go.Scatter(
            x=[ticker, ticker], 
            y=[df.loc[ticker, 'hdi_3%'], df.loc[ticker, 'hdi_97%']],
            mode='lines', 
            name=ticker if fund_name is None else fund_name[ticker],  
            line=dict(width=3),
            legendgroup=ticker,  # Group with mean marker
            showlegend=True
        ))
    
        # Plot the mean marker, grouped with its HDI range but hidden from legend
        fig.add_trace(go.Scatter(
            x=[ticker, ticker], 
            y=[df.loc[ticker, 'mean'], df.loc[ticker, 'mean']],
            mode='markers', 
            name=ticker if fund_name is None else fund_name[ticker], 
            marker=dict(color="gray", size=5, symbol='line-ew-open'),
            legendgroup=ticker,  # Same group as HDI range
            showlegend=False  # Hide from legend
        ))
    
    fig.update_layout(
        title="HDI Range and Mean",
        xaxis=dict(
            title='',             # Remove x-axis title (label)
            showticklabels=False  # Hide x-tick labels
        ),
        hoverlabel_bgcolor="white",
    )

            
    for trace in fig.data:
        if trace.showlegend:
            trace.update(hoverinfo='skip')
            #text = hover_text[trace.legendgroup]
            #trace.update(hovertemplate=f"{text} {trace.name}<extra></extra>")
            #trace.update(hovertemplate=f"{trace.name}<extra></extra>")
        else:
            trace.update(hovertemplate=f"{trace.name}<extra></extra><br>test")
            
    return fig



def create_app(df_prices, df_prices_fees, tickers=None, fund_name=None,
               options_word=['&TDF', 'IBK', 'KB', '미래에셋', '삼성', '신한', '키움', '한국투자', '한화'], 
               options_df = None,
               title="Managed Funds", height=500, legend=False, length=20,
               base=1000,
               external_stylesheets=external_stylesheets,
               debug=False):

    if tickers is None:
        tickers = df_prices.columns.to_list()
    data_prc = {
        'before fees':df_prices[tickers], 
        'after fees':df_prices_fees[tickers]
    }

    if fund_name is None:
        fund_name = {x:x for x in tickers}
        
    # create dropdown options
    dm = DropdownManager(tickers, fund_name)
    dm.create_all()
    dm.create_from_name(options_word)
    dm.create_from_df(options_df) if options_df is not None else None
    dm.create_tickers()
    dropdown_option = dm.get_options()

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
        Output('ticker-dropdown', 'value'),
        Input('ticker-dropdown', 'value'),
    )
    def _update_options(values):
        return dm.update_options(values, dropdown_option)
    
    
    @app.callback(
        Output('price-data', 'data'),
        Input('ticker-dropdown', 'value'),
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
        Input('ticker-dropdown', 'value')
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



class DropdownManager():
    def __init__(self, tickers=None, fund_name=None, intersection='&'):
        self.tickers = tickers
        self.fund_name = fund_name # dict of ticker to name
        self.startswith_intersection = intersection
        self.options = list()
        self.value_to_ticker = dict() # option value to tickers
        self.option_all = None

    def _check_tickers(self, tickers):
        """
        check tickers by returning tickers only in self.tickers
        """
        if self.tickers is None:
            return tickers
        else:
            return pd.Index(tickers).intersection(self.tickers).to_list()

    def _set_option(self, value, label=None, title=None, search=None):
        intersection = self.startswith_intersection
        # intersection sign kept only for value
        label, title, search = [value.lstrip(intersection) if x is None else x for x in [label, title, search]]
        label = f'{label} ({intersection})' if value.startswith(intersection) else label
        search = search.lower()
        return {'label':label, 'value':value, 'title':title, 'search':search}

    def create_from_df(self, df_values, col_ticker='ticker'):
        """
        set option and its ticker list specifically
        df_values: df of index ticker and column values
        """
        cols = df_values.columns
        if col_ticker not in cols:
            return None
        # each col has option values. None for option skipped by the groupby
        for col in cols.difference([col_ticker]):
            value_to_ticker = df_values.groupby(col)[col_ticker].apply(list).to_dict()
            self.options += [self._set_option(x) for x in value_to_ticker.keys()]
            self.value_to_ticker = {**self.value_to_ticker, **value_to_ticker}

    def create_all(self, option_all='All'):
        """
        make option to select all tickers
        """
        fund_name = self.fund_name
        options = [{'label':option_all, 'value':option_all, 
                    'title':option_all, 'search':option_all.lower()}]
        self.options += options 
        self.value_to_ticker[option_all] = list() if fund_name is None else list(fund_name.keys())
        self.option_all = option_all

    def create_tickers(self):
        """
        set tickers to options
        """
        fund_name = self.fund_name
        if fund_name is None:
            return None
        options = [{'label':k, 'value':k, 'title':v, 'search':v} for k,v in fund_name.items()]
        tickers = {x['value']:[x['value']] for x in options}
        self.options += options 
        self.value_to_ticker = {**self.value_to_ticker, **tickers}

    def create_from_name(self, names):
        """
        make options of names in fund_name
        names: list of name to make options
        """
        fund_name = self.fund_name
        if fund_name is None:
            return None
            
        if isinstance(names, str):
            names = [namses]
        for value in names:
            option = self._set_option(value)
            title = option['title']
            # get ticker list for title
            tickers = [k for k,v in fund_name.items() if title.lower() in v.lower()]
            if len(tickers) > 0:
                self.options.append(option)
                self.value_to_ticker[value] = tickers

    def get_options(self):
        if len(self.options) == 0:
            return list()
        else:
            return self.options

    def update_options(self, values, options):
        """
        disable options if option_all selected
        """
        option_all = self.option_all
        intersection = self.startswith_intersection
        if option_all in values:
            # disable all options except for intersections
            options = [{**x, 'disabled':False if x['value'].startswith(intersection) else True} for x in options]
            # exclude all options except for intersections from values
            values = [x for x in values if x==option_all or x.startswith(intersection)]
        else:
            options = [{**x, 'disabled':False} for x in options]
        return options, values
        
    def get_tickers(self, values):
        """
        get union of tickers from list of option values
        """
        if len(self.options) == 0:
            return list()
        else:
            intersection = self.startswith_intersection
        # split options of union and intersection
        values_intersection = [x for x in values if x.startswith(intersection)]
        values_union = list(set(values) - set(values_intersection))

        # get tickers from union options
        tickers = list()
        for v in values_union:
            try:
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
        return self._check_tickers(tickers) # remove tickers not in self.tickers
