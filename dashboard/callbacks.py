from dash import Dash, html, dcc, Input, Output, callback
import dash_uploader as du
from app import app
import utils
import numpy as np
import base64
import sys, os
import resurfemg.converter_functions as cv

du.configure_upload(app, r"C:\tmp\Uploads", use_upload_id=True)

emg_data_raw = None
ventilator_data_raw = None


@du.callback(
    output=Output('original-emg', 'data'),
    id='upload-emg-data',
)
def parse_emg(status):
    emg_data = cv.poly5unpad(status[0])
    global emg_data_raw
    emg_data_raw = emg_data
    # children = utils.add_emg_graphs(emg_data)

    return 'set'


@du.callback(
    output=Output('original-ventilator', 'data'),
    id='upload-ventilator-data',
)
def parse_vent(status):
    vent_data = cv.poly5unpad(status[0])
    global ventilator_data_raw
    ventilator_data_raw = vent_data
    # children = utils.add_ventilator_graphs(vent_data)
    print('vent uploaded')
    return 'set'

@callback(Output('ventilator-graphs-container', 'children'),
          Output('emg-graphs-container', 'children'),
          Input('hidden-div', 'data'))
def show_raw_data(ventilator_data):
    if ventilator_data_raw is not None:
        children_vent = utils.add_ventilator_graphs(np.array(ventilator_data_raw))
    else:
        children_vent = []

    if emg_data_raw is not None:
        children_emg = utils.add_emg_graphs(np.array(emg_data_raw))
    else:
        children_emg = []
    return children_vent, children_emg
