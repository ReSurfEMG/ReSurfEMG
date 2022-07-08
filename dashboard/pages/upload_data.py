import dash
import dash_bootstrap_components as dbc
import dash_uploader as du
from dash import html, dcc


dash.register_page(__name__, path='/')


def layout():
    return html.Div([

        html.H1('Upload Data'),
        dbc.Row([
            dbc.Col([html.Div([
                du.Upload(
                    id='upload-emg-data',
                    text='Drag and Drop Here to upload EMG data!',
                    text_completed='Uploaded: ',
                    filetypes=['Poly5'],
                ),
            ]),

            ], width=6),

            dbc.Col([html.Div([
                dcc.Input(
                    id='emg-sample-freq',
                    type="number",
                    placeholder="EMG sampling frequency",
                    value=2048
                ),
            ],
                style={'textAlign': 'right'}),
            ], width=2),

            dbc.Col([
                html.Div('EMG sampling frequency', style={'textAlign': 'left'})
                ], width=2),
        ]),

        dbc.Row([
            dbc.Col([html.Div([
                du.Upload(
                    id='upload-ventilator-data',
                    text='Drag and Drop Here to upload Ventilator data!',
                    text_completed='Uploaded: ',
                    filetypes=['Poly5'],
                ),
            ]),

            ], width=6),

            dbc.Col([html.Div([
                dcc.Input(
                    id='ventilator-sample-freq',
                    type="number",
                    placeholder="Ventilator sampling frequency",
                    value=100
                ),
            ],
                style={'textAlign': 'right'}),
            ], width=2),

            dbc.Col([
                html.Div('EMG sampling frequency', style={'textAlign': 'left'})
            ], width=2),
        ]),


        html.Div(children=[
            html.H1(id='out', children='')
        ]),
        # the following elements are only needed
        # to provide outputs to the callbacks
        html.Div(id='emg-uploaded-div'),
        html.Div(id='ventilator-uploaded-div'),
        html.Div(id='emg-frequency-div'),
        html.Div(id='ventilator-frequency-div')
    ])
