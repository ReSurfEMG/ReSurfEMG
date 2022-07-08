import dash
import dash_uploader as du
import dash_bootstrap_components as dbc
from dash import html, dcc


dash.register_page(__name__, path='/view-raw')


layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div(id='emg-graphs-container',
                className='six columns')
        ], width=6),
        dbc.Col([
            html.Div(id='ventilator-graphs-container',
                className='six columns')
        ], width=6)
    ]),
    html.Div(id='hidden-div'),
    html.Div(id='sample-req-emg'),
    html.Div(id='sample-req-vent')
])
