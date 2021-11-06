import dash
from dash import dcc
from dash import html
from dash import dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import core as core
import numpy as np
from dash.dependencies import Input, Output

def get_figure(data, displayable_columns, selected_points=None, newpoint=None):
    size = 10

    newdata = data.copy()
    newpoint_idx = None
    if newpoint is not None:
        newdata = pd.concat([newdata, newpoint])
        newpoint_idx = newdata.shape[0] - 1

    income = newdata['income'].to_numpy()
    income = np.where(income == '>50', 1., -1.)

    small_columns = ['age', 'race', 'sex']
    hover_text = ""
    for i, column in enumerate(displayable_columns):
        if column in small_columns:
            hover_text += f"{column}: " + "%{customdata[" + str(i) +"]}\n"
    fig = go.Figure(data=go.Scattergl(
        x=newdata["xpos"],
        y=newdata["ypos"],
        mode='markers',
        ids=newdata.index,
        customdata=newdata[displayable_columns],
        hovertemplate=hover_text,
        marker={
            'color': income,
            'cmax': 1,
            'cmin': -1
        }
    ))
    fig.update_layout(clickmode='event+select')
    if selected_points is not None:
        fig.update_traces(
            selectedpoints=selected_points + [newpoint_idx],
            selected={
                'marker': {
                    'size': size,
                    'color': 'rgb(255,0,0)'
                }
            },
            unselected={
                'marker': {
                    'opacity': 0.3,
                }
            })

    return fig


app = dash.Dash(__name__)

data = core.get_pca_2d()

displayable_columns = data.columns[~data.columns.isin(['id', 'xpos', 'ypos'])]

points = core.get_printable_points(5)[['age', 'race', 'sex']]

components = {
    'display': [
        html.Label('Data Display'),
        dcc.Graph(
            id='main-display',
            figure=get_figure(data, displayable_columns)
        )
    ],
    'title': [
        html.H1('DiCE:'),
        html.H2('Diverse Counterfactual Explanations for Machine Learning Classifiers'),
    ],
    'controls': [
        html.Label(id='algorithm-label'),
        dcc.RadioItems(
            options=[
                {'label': 'DiCE', 'value': 'dice'},
                {'label': 'MRMC', 'value': 'mrmc'}
            ],
            value='dice',
            id='algorithm-input'
        )
    ],
    'output': [
        html.Div('This is where the algorithm description and output goes'),
        html.Br(),
        html.Div("There's a lot to put here! But right now it's pretty empty."),
        html.Br(),
        html.Div(children=[], id='recourse-display')
    ]
}

def transform_df(df, new_columns):
    rows = []
    d = df.T.to_dict(orient='index')
    feature_col = 'Attribute'
    columns = [feature_col] + new_columns
    for feature in d:
        values = d[feature].values()
        m = {}
        m[feature_col] = feature
        for col, val in zip(new_columns, values):
            m[col] = val

        rows.append(m)

    return rows, columns

def make_recourse_display(poi, recourse, columns_to_display=None):
    if columns_to_display is None:
        columns_to_display = displayable_columns
    
    df = pd.concat([poi[columns_to_display], recourse[columns_to_display]])
    df_dict, df_columns = transform_df(df, new_columns=['Selected Point', 'Recourse Point'])
    table = dash_table.DataTable(
                id='joint-table',
                columns=[{'name': column, 'id': column} for column in df_columns],
                data=df_dict,
                style_cell={'fontSize':12, 'font-family':'sans-serif'}
            )
    return table

@app.callback(
    Output('recourse-display', 'children'),
    Output('main-display', 'figure'),
    Input('main-display', 'clickData')
)
def click_on_poi(clickData):
    if clickData is None:
        return html.Div("Try clicking a point in the display!"), get_figure(data, displayable_columns)
    poi_index = clickData['points'][0]['id']
    poi = core.da.data_df.iloc[core.da.data_df.index == poi_index,:]
    recourse = data.sample(1)
    selected_points = np.concatenate([poi.index.to_numpy(), recourse.index.to_numpy()])
    selected_points = [clickData['points'][0]['pointIndex']]
    return make_recourse_display(poi, recourse), get_figure(data, displayable_columns, selected_points, recourse)


@app.callback(
    Output(component_id='algorithm-label', component_property='children'),
    Input(component_id='algorithm-input', component_property='value')
)
def update_algorithm_label(input_value):
    return f'Recourse Algorithm ({input_value} not implemented yet)'

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
