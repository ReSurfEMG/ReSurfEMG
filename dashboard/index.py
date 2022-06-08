# Run this app with `python index.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, callback
from dash.dependencies import Input, Output
from layouts import load_data_page, view_raw_page, preprocessing_page, features_page, interpretation_page
import dash_uploader as du
from app import app
import callbacks

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/apps/load-data':
        return load_data_page
    elif pathname == '/apps/view-raw':
        return view_raw_page
    elif pathname == '/apps/preprocessing':
        return preprocessing_page
    elif pathname == '/apps/features':
        return features_page
    elif pathname == '/apps/interpretation':
        return interpretation_page
    else:
        return load_data_page


if __name__ == '__main__':
    app.run_server(debug=True)
