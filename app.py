import dash
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div(children=[
    html.H1(
        children='DiCE:',
    ),
    html.H2('Diverse Counterfactual Explanations for Machine Learning Classifiers'),

    html.Div(children='''
        This is a demo of DiCE on the Adult Income dataset.
        '''
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)