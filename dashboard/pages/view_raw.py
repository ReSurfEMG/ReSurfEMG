import dash
import dash_uploader as du
import dash_bootstrap_components as dbc
from dash import html, dcc


dash.register_page(__name__, path='/view-raw')


layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H1(id='emg-data-title', children='EMG data'),
                dbc.Button('Remove data', id='emg-delete-button',
                           style={'background': 'red',
                                  'border': 'transparent'})
            ],
                id='emg-header',
                hidden=True
            ),
            html.Div(id='emg-graphs-container',
                     className='six columns')
        ], width=6),

        dbc.Col([
            html.Div([
                html.H1(id='ventilator-data-title', children='Ventilator data'),
                dbc.Button('Remove data', id='ventilator-delete-button',
                           style={'background': 'red',
                                  'border': 'transparent'})
            ],
                id='ventilator-header', hidden=True
            ),
            html.Div(id='ventilator-graphs-container',
                     className='six columns')
        ], width=6)
    ]),
    html.Div(id='hidden-div'),
    html.Div(id='sample-req-emg'),
    html.Div(id='sample-req-vent')
])
