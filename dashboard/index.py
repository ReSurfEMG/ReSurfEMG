# Run this app with `python index.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import callbacks
import dash
import dash_bootstrap_components as dbc
from dash import html
from app import app

# static images
image_filename = 'resources/resurfemg.png'
# pylint: disable=consider-using-with
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

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

header = dbc.Row([
    dbc.Col(html.Div([
        html.H1(children='ReSurfEMG Dashboard',
                style={'textAlign': 'center'}
                )]
    ),
        width=8
    ),
    dbc.Col(html.Img(
        src=f'data:image/png;base64,{encoded_image.decode()}',
        height='100 px',
        width='auto'),
        width=4)
],
    align='center',
    style={'height': '150px',
           'background-color': colors['superdark-green'],
           'color': colors['white']}

)

nav = dbc.Row(
        [
            dbc.Col(dbc.NavLink("Load data", href="/",
                                style={'font-size': '25px'})),
            dbc.Col(dbc.NavLink("View raw data", href="/view-raw",
                                style={'font-size': '25px'})),
            dbc.Col(dbc.NavLink("Preprocessing", href="/preprocessing",
                                style={'font-size': '25px'})),
            dbc.Col(dbc.NavLink("Features", href="/features",
                                style={'font-size': '25px'})),
            dbc.Col(dbc.NavLink("Interpretation", href="/interpretation",
                                style={'font-size': '25px'}))
        ],
        align='center',
        style={'height': '75px',
               'background-color': colors['dark-green'],
               'color': colors['white']
}
    )

app.layout = dbc.Container([
    header,
    nav,
    dash.page_container
],
    fluid=True
)

if __name__ == '__main__':
    app.run_server(debug=True)
