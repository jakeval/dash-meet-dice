import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from data_handling import data_manager as dm

app = dash.Dash(__name__)

data = dm.get_pca_2d()

fig = px.scatter(data, x="xpos", y="ypos",
                 color="income", hover_name="age", hover_data=data.columns)


app.layout = html.Div(children=[
    html.H1(
        children='DiCE:',
    ),
    html.H2('Diverse Counterfactual Explanations for Machine Learning Classifiers'),

    html.Div(children='''
        This is a demo of DiCE on the Adult Income dataset.
        '''
    ),
    dcc.Graph(
        id='pca-display',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)