from dash import Input, Output, callback
import dash_uploader as du
from app import app
import static_variables
import utils
import numpy as np
import resurfemg.converter_functions as cv

du.configure_upload(app, r"C:\tmp\Uploads", use_upload_id=True)

emg_data_raw = None
ventilator_data_raw = None

variables = static_variables.get_singleton()

@du.callback(
    output=Output('emg-uploaded-div', 'data'),
    id='upload-emg-data',
)
def parse_emg(status):
    emg_data = cv.poly5unpad(status[0])
    global emg_data_raw
    emg_data_raw = emg_data
    # children = utils.add_emg_graphs(emg_data)
    return 'set'


@du.callback(
    output=Output('ventilator-uploaded-div', 'data'),
    id='upload-ventilator-data',
)
def parse_vent(status):
    vent_data = cv.poly5unpad(status[0])
    global ventilator_data_raw
    ventilator_data_raw = vent_data
    # children = utils.add_ventilator_graphs(vent_data)
    print('vent uploaded')
    return 'set'


@callback(Output('emg-frequency-div', 'data'),
          Input('emg-sample-freq', 'value'))
def update_emg_frequency(freq,):
    variables.set_emg_freq(freq)
    return 'set'


@callback(Output('ventilator-frequency-div', 'data'),
          Input('ventilator-sample-freq', 'value'))
def update_ventilator_frequency(freq):
    variables.set_ventilator_freq(freq)
    return 'set'


@callback(Output('ventilator-graphs-container', 'children'),
          Output('emg-graphs-container', 'children'),
          Input('hidden-div', 'data'))
def show_raw_data(ventilator_data):
    if ventilator_data_raw is not None:
        ventilator_frequency = variables.get_ventilator_freq()
        children_vent = utils.add_ventilator_graphs(np.array(ventilator_data_raw), ventilator_frequency)
    else:
        children_vent = []

    if emg_data_raw is not None:
        emg_frequency = variables.get_emg_freq()
        children_emg = utils.add_emg_graphs(np.array(emg_data_raw), emg_frequency)
    else:
        children_emg = []
    return children_vent, children_emg