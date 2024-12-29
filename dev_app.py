from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
import dash_daq as daq


# Load data
file = 'fund_241228.csv'
path = 'pages'
df_prc = pd.read_csv(
    f'{path}/{file}',
    parse_dates=['date'],
    dtype={'ticker': str},
    index_col=['group', 'ticker', 'date']
)

groups = df_prc.index.get_level_values('group').unique()
default_group = 2030
groups = [{'label': f'TDF{x}', 'value': x} for x in groups]

# Initialize the Dash app
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(
            dcc.Dropdown(
                id='group-dropdown',
                options=groups,
                value=default_group,
                clearable=False,
            ),
            width=3  # Adjust width as needed
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
        justify="center",  # Centers the row's content horizontally
        align="center", # Centers the row's content vertically
        className="mb-3" # Bootstrap margin-bottom class
        #style={'margin-bottom': '20px'}  # Adds space below the dropdown row
    ),
    dbc.Row(dcc.Graph(id='price-plot')),
    dbc.Row(dcc.Graph(id='return-plot')),
    # Store DataFrame in JSON format
    dcc.Store(id='price-data'),
    dbc.Tooltip(
        '상대 비교',
        target='compare-boolean-switch',
        placement='bottom'
    ),
    dbc.Tooltip(
        '비용 고려',
        target='cost-boolean-switch',
        placement='bottom'
    )
])


@app.callback(
    Output('price-data', 'data'),
    Input('group-dropdown', 'value'),
)
def update_price_data(group):
    base = 1000
    cols = df_prc.columns
    data = {'columns':cols}
    start = None
    for col in cols:
        df = df_prc.loc[group, col].unstack('ticker')
        if start is None:
            start = df.apply(lambda x: x[x.notna()].index.min()).max()
            data.update({'index': df.index})
        df = df.apply(lambda x: x / x.loc[start] * base)
        data.update({col: df.to_dict('records')})
    return data


@app.callback(
    Output('return-plot', 'figure'),
    Input('price-data', 'data'),
)
def update_return_plot(data):
    df_ret = pd.DataFrame()
    for col in data['columns']:
        df = pd.DataFrame(data[col])
        df = df.apply(lambda x: x.dropna().iloc[-1]/x.dropna().iloc[0]-1).to_frame(col)
        df_ret = pd.concat([df_ret, df], axis=1)
    fig = px.bar(df_ret, barmode='group')
    return fig


@app.callback(
    Output('price-plot', 'figure'),
    Input('price-data', 'data'),
    Input('cost-boolean-switch', 'on'),
)
def update_price_plot(data, cost):
    cols = data['columns']
    if cost:
        col = cols[1]
        title = '펀드 가치 (비용 고려)'
    else:
        col = cols[0]
        title = '펀드 가치'
    df = pd.DataFrame(data[col], index=data['index'])
    fig = px.line(df, title=title)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)