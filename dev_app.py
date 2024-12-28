from dash import Dash, html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px

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
app = Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='group-dropdown',
        options=groups,
        value=default_group,  # Default value
        clearable=False,
        style={'width': '50%'}
    ),
    dcc.Graph(id='return-plot'),
    dcc.Graph(id='price-plot'),
    # Store DataFrame in JSON format
    dcc.Store(id='price-data')
])

@app.callback(
    Output('price-data', 'data'),
    Input('group-dropdown', 'value'),
)
def update_price_data(group):
    base = 1000
    col_prc = 'price'
    df = df_prc.loc[group, col_prc].unstack('ticker')
    dt = df.apply(lambda x: x[x.notna()].index.min()).max()
    df = df.apply(lambda x: x / x.loc[dt] * base)
    return df.to_dict('records')

@app.callback(
    Output('return-plot', 'figure'),
    Input('price-data', 'data'),
)
def update_return_plot(data):
    df = pd.DataFrame(data)
    df = df.apply(lambda x: x.dropna().iloc[-1]/x.dropna().iloc[0]-1)
    fig = px.bar(df)
    return fig

@app.callback(
    Output('price-plot', 'figure'),
    Input('price-data', 'data'),
)
def update_price_plot(data):
    df = pd.DataFrame(data)
    fig = px.line(df)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)