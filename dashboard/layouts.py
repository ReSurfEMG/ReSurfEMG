from dash import dcc, html
from enum import Enum
import base64
import plotly.express as px
import pandas as pd
import dash_uploader as du
import utils


class Pages(Enum):
    LOAD_DATA = 'load-data',
    VIEW_RAW = 'view-raw',
    PREPROCESSING = 'preprocessing',
    FEATURES = 'features',
    INTERPRETATION = 'interpretation'


####################################################################################################
# 000 - FORMATTING INFO
####################################################################################################

colors = {
    'dark-blue-grey': 'rgb(62, 64, 76)',
    'medium-blue-grey': 'rgb(77, 79, 91)',
    'superdark-green': 'rgb(41, 56, 55)',
    'dark-green': 'rgb(57, 81, 85)',
    'medium-green': 'rgb(93, 113, 120)',
    'light-green': 'rgb(186, 218, 212)',
    'pink-red': 'rgb(255, 101, 131)',
    'dark-pink-red': 'rgb(247, 80, 99)',
    'white': 'rgb(251, 251, 252)',
    'light-grey': 'rgb(208, 206, 206)'
}

navbarcurrentpage = {
    'textDecoration': 'underline',
    'textDecorationColor': colors['pink-red'],
    'textShadow': '0px 0px 1px rgb(251, 251, 252)'
}

navbardefault = {
    'textDecoration': 'none'
}

# static images
image_filename = 'resources/resurfemg.png'
# pylint: disable=consider-using-with
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")


#####################
# Header
#####################

def get_header():
    header = html.Div([

        html.Div([], className='two columns'),

        html.Div([
            html.H1(children='ReSurfEMG Dashboard',
                    style={'textAlign': 'center'}
                    )],
            className='eight columns',
            style={'paddingTop': '1%'}
        ),

        html.Div(
            className='two columns',
            style={
                'alignItems': 'center',
                'paddingTop': '1%',
                'height': 'auto'}),

        html.Div([
            html.Img(
                src=f'data:image/png;base64,{encoded_image.decode()}',
                height='70 px',
                width='auto')
        ],
            className='col-2',
            style={
                'align-items': 'center',
                'padding-top': '1%',
                'height': 'auto'})

    ],
        className='row',
        style={'height': '4%',
               'background-color': colors['superdark-green'],
               'color': colors['white']}
    )

    return header


#####################
# Navigation bar
#####################

def get_navbar(p=Pages.LOAD_DATA):
    load_data_name = 'Load data'
    load_data_href = '/apps/load-data'
    navbar_load_data_style = navbardefault

    view_raw_name = 'View raw data'
    view_raw_href = '/apps/view-raw'
    navbar_view_raw_style = navbardefault

    preprocessing_name = 'Preprocessing'
    preprocessing_href = '/apps/preprocessing'
    navbar_preprocessing_style = navbardefault

    features_name = 'Features'
    features_href = '/apps/features'
    navbar_features_style = navbardefault

    interpretation_name = 'Interpretation'
    interpretation_href = '/apps/interpretation'
    navbar_interpretation_style = navbardefault

    if p == Pages.LOAD_DATA:
        navbar_load_data_style = navbarcurrentpage
    elif p == Pages.VIEW_RAW:
        navbar_view_raw_style = navbarcurrentpage
    elif p == Pages.PREPROCESSING:
        navbar_preprocessing_style = navbarcurrentpage
    elif p == Pages.FEATURES:
        navbar_features_style = navbarcurrentpage
    elif p == Pages.INTERPRETATION:
        navbar_interpretation_style = navbarcurrentpage

    navbar = html.Div([

        html.Div([], className='three columns'),

        html.Div([
            dcc.Link(
                html.H4(children=load_data_name,
                        style=navbar_load_data_style),
                href=load_data_href
            )
        ],
            className='two columns'),

        html.Div([
            dcc.Link(
                html.H4(children=view_raw_name,
                        style=navbar_view_raw_style),
                href=view_raw_href
            )
        ],
            className='two columns'),

        html.Div([
            dcc.Link(
                html.H4(children=preprocessing_name,
                        style=navbar_preprocessing_style),
                href=preprocessing_href
            )
        ],
            className='two columns'),

        html.Div([
            dcc.Link(
                html.H4(children=features_name,
                        style=navbar_features_style),
                href=features_href
            )
        ],
            className='two columns'),

        html.Div([
            dcc.Link(
                html.H4(children=interpretation_name,
                        style=navbar_interpretation_style),
                href=interpretation_href
            )
        ],
            className='two columns'),

        html.Div([], className='three columns')

    ],
        className='row',
        style={'background-color': colors['dark-green']}
    )

    return navbar


####################################################################################################
# 001 - LOAD DATA PAGE
####################################################################################################

load_data_page = html.Div([
    get_header(),
    get_navbar(Pages.LOAD_DATA),

    html.H1('Upload Data'),
    html.Div([
        du.Upload(
            id='upload-emg-data',
            text='Drag and Drop Here to upload EMG data!',
            text_completed='Uploaded: ',
            filetypes=['Poly5'],
        ),
    ]),
    html.Div([
        du.Upload(
            id='upload-ventilator-data',
            text='Drag and Drop Here to upload Ventilator data!',
            text_completed='Uploaded: ',
            filetypes=['Poly5'],
        ),
    ]),
    html.Div(children=[
        html.H1(id='out', children='')
    ]),
    html.Div(id='original-emg'),
    html.Div(id='original-ventilator')
])

####################################################################################################
# 002 - VIEW RAW PAGE
####################################################################################################

view_raw_page = html.Div([
    get_header(),
    get_navbar(Pages.VIEW_RAW),
    html.Div([
        html.Div(id='emg-graphs-container',
                 className='six columns'),

        html.Div(id='ventilator-graphs-container',
                 className='six columns'),
    ],
        className='row'
    ),
    html.Div(id='hidden-div')
])

####################################################################################################
# 002 - PREPROCESSING PAGE
####################################################################################################

preprocessing_page = html.Div([
    get_header(),
    get_navbar(Pages.PREPROCESSING)
])

####################################################################################################
# 003 - FEATURES PAGE
####################################################################################################

features_page = html.Div([
    get_header(),
    get_navbar(Pages.FEATURES)
])

####################################################################################################
# 004 - INTERPRETATION PAGE
####################################################################################################

interpretation_page = html.Div([
    get_header(),
    get_navbar(Pages.INTERPRETATION)
])
