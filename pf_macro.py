from dash import Dash, html, dcc, Output, Input, State
import pandas as pd
import dash_bootstrap_components as dbc
import dash_daq as daq
import json

external_stylesheets = [dbc.themes.CERULEAN, 
                        #dbc.themes.BOOTSTRAP,
                        dbc.icons.FONT_AWESOME,
                        dbc.icons.BOOTSTRAP]

style_heading={'color':'slategray', 'font-weight':'bold'}

base_prc = 1000
date_format = '%Y-%m-%d'

# load data
file = 'macro_indicators.csv'
path = 'data'
df_macro = pd.read_csv(f'{path}/{file}', parse_dates=[0], index_col=0).rename_axis('date')

## sample data
freq = 'Q'
df = df_macro.stack().swaplevel().rename_axis(['ticker', 'date']).sort_index()
grouped = df.groupby([df.index.get_level_values('ticker'),
                     df.index.get_level_values('date').to_period(freq)])

### Get the index (MultiIndex) of the max/min per group
idx1 = grouped.idxmax()
idx2 = grouped.idxmin()
idx = pd.concat([idx1, idx2]).drop_duplicates()

df = df.loc[idx].unstack('ticker').sort_index()
df_macro = df.reindex(df.index.strftime(date_format))

# convert to json
data_macro = {}
for col in df_macro.columns:
    data_macro[col] = df_macro[col].dropna().to_dict()
data_macro_json = json.dumps(data_macro)

# dropdown options
indicator_options = [{'label':x, 'value':x} for x in df_macro.columns]
indicator_options = [{'label':'All', 'value':'All'}] + indicator_options
indicator_default = ['All']


app = Dash(__name__, title="Markets",
           external_stylesheets=external_stylesheets)

app.index_string = f"""
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        <link rel="icon" type="image/x-icon" href="/assets/favicon.ico">
        <style>
        #styled-numeric-input input:invalid {{
            border-color: #dc3545;
            padding-right: calc(1.5em + 0.75rem);
            background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 12 12' width='12' height='12' fill='none' stroke='%23dc3545'%3e%3ccircle cx='6' cy='6' r='4.5'/%3e%3cpath stroke-linejoin='round' d='M5.8 3.6h.4L6 6.5z'/%3e%3ccircle cx='6' cy='8.2' r='.6' fill='%23dc3545' stroke='none'/%3e%3c/svg%3e");
            background-repeat: no-repeat;
            background-position: right calc(0.375em + 0.1875rem) center;
            background-size: calc(0.75em + 0.375rem) calc(0.75em + 0.375rem);
            outline: none;
        }}
        </style>
        {{%css%}}
    </head>
    <body>
        <script>
            var dataMacro = {data_macro_json};
        </script>
        {{%app_entry%}}
        {{%config%}}
        {{%scripts%}}
        {{%renderer%}}
    </body>
</html>
"""


app.layout = dbc.Container([
    html.Br(),
    dbc.Stack([
        html.Div(
            #dcc.Input(
            dbc.Input(
                id='start-input', 
                type='number', min=1900, max=2100, step=1,
                size='4',
                placeholder='Start'
            ), 
            #style={'min-width':'10%'}
        ),
        html.Div(
            dbc.Input(
                id='end-input', 
                type='number', min=1900, max=2100, step=1,
                size='4',
                placeholder='End'
            ), 
        ),
        html.Div(
            dcc.Dropdown(
                id='indicator-dropdown',
                options=indicator_options,
                value=indicator_default,
                multi=True,
            ), style={'min-width':'60%'} 
        )
    ],
        direction="horizontal",
        gap=2,
        className="mb-3",
        id="styled-numeric-input",
    ),
    html.Div(
        dcc.Graph(id='macro-plot')
    ),
    html.Br(),
    dcc.Store(id='macro-data'),
    dcc.Location(id="url", refresh=False),  # To initialize the page
])


# update data based on selected indicators
app.clientside_callback(
    """
    function(indicators) {
        let data = {};

        // Check indicators
        if (!Array.isArray(indicators) || indicators.length === 0 || indicators[indicators.length - 1] === 'All') {
            data = dataMacro;
            indicators = ['All'];
        } else if (indicators.includes('All')) {
            // 'All' is selected but not last → remove it to avoid conflict
            indicators = indicators.filter(group => group !== 'All');
            // Still need to fetch selected indicators
        }

        // Retrieve selected indicators if not already set to all
        if (Object.keys(data).length === 0) {
            for (let tkr in dataMacro) {
                if (indicators.includes(tkr)) {
                    data[tkr] = dataMacro[tkr];
                }
            }
        }

        // Get start & end years
        let years = [];
        for (let tkr in data) {
            for (let dateStr in data[tkr]) {
                let year = new Date(dateStr).getFullYear();
                years.push(year);
            }
        }
        start = Math.min(...years);
        end = Math.max(...years);
        
        return [data, indicators, start, end];

    }
    """,
    Output('macro-data', 'data'),
    Output('indicator-dropdown', 'value'),
    Output('start-input', 'value'),
    Output('end-input', 'value'),
    Input('indicator-dropdown', 'value')
)



# plot indicator history
app.clientside_callback(
    """
    function(start, end, data) {
        
        // filter by year
        let data_filterd = {};
        if (start < end) {
            for (let tkr in data) {
                data_filterd[tkr] = {};

                for (let dateStr in data[tkr]) {
                    let year = new Date(dateStr).getFullYear();
                    if (year >= start && year <= end) {
                        data_filterd[tkr][dateStr] = data[tkr][dateStr];
                    }
                }
            }
        } else {
            data_filterd = data;
        }

        // plot indicator history
        let traces = [];
        for (let tkr in data_filterd) {
            traces.push({
                x: Object.keys(data_filterd[tkr]),  // Assuming keys are dates
                y: Object.values(data_filterd[tkr]).map(val => Math.round(val)),  // Assuming values are prices
                type: 'line',
                mode: 'lines',
                name: tkr
            });
        }

        let layout = {
            //title: { text: title},
            hovermode: 'x',
            //yaxis: { title: '가격' },
        }

        return {
            data: traces,
            layout: layout
        };
    }
    """,
    Output('macro-plot', 'figure'),
    Input('start-input', 'value'),
    Input('end-input', 'value'),
    State('macro-data', 'data'),
)


# Run the app
if __name__ == '__main__':
    app.run_server(debug=False)