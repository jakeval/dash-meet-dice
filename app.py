import dash
from dash import dcc
from dash import html
from dash import dash_table
import plotly.express as px
import pandas as pd
import core as core

app = dash.Dash(__name__)

data = core.get_pca_2d()

fig = px.scatter(data, x="xpos", y="ypos",
                 color="income", hover_name="age", hover_data=data.columns)

poi = core.get_printable_poi()[['age', 'race', 'sex']]
print(poi)

points = core.get_printable_points(10)

components = {
    'display': [
        html.Label('Data Display'),
        dcc.Graph(
            id='main-display',
            figure=fig
        )
    ],
    'title': [
        html.H1('DiCE:'),
        html.H2('Diverse Counterfactual Explanations for Machine Learning Classifiers'),
    ],
    'controls': [
        html.Label('Recourse Algorithm'),
        dcc.RadioItems(
            options=[
                {'label': 'DiCE', 'value': 'dice'},
                {'label': 'MRMC', 'value': 'mrmc'}
            ],
            value='dice'
        )
    ],
    'output': [
        html.Div('This is where the algorithm description and output goes'),
        html.Br(),
        html.Div("There's a lot to put here! But right now it's pretty empty."),
        html.Br(),
        html.Div("So here's some random info..."),
        dash_table.DataTable(
            id='table',
            columns=[{'name': column, 'id': column} for column in poi.columns],
            data=points.to_dict('rows'),
            # style_cell={'fontSize':20, 'font-family':'sans-serif'}
        )
    ]
}

app.layout = html.Div(children=[
    html.Div(children=[
        *components['title'],
    ], style={'padding': 1, 'flex': 1}),
    html.Div(children=[
        html.Div(children=[
            *components['controls'],
            *components['display'],
        ], style={'padding': 1, 'flex': 2}),
        html.Div(children=[
            *components['output']
        ], style={'padding': 1, 'flex': 1})
    ], style={'display': 'flex', 'flex-direction': 'row', 'padding': 1, 'flex': 5}),
], style={'display': 'flex', 'flex-direction': 'column'})

if __name__ == '__main__':
    app.run_server(debug=True)
