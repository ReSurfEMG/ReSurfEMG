import numpy as np
import plotly.graph_objects as go
from dash import html, dcc

# colors
colors = {
    'white': '#FFFFFF',
    'text': '#091D58',
    'blue1': '#063446',  # dark blue
    'blue2': '#0e749b',
    'blue3': '#15b3f0',
    'blue4': '#E4F3F9',  # light blue
    'yellow1': '#f0d515'
}


def blank_fig(text=None):
    """Creates a blank figure."""
    fig = go.Figure(data=go.Scatter(x=[], y=[]))
    fig.update_layout(
        paper_bgcolor=colors['blue4'],
        plot_bgcolor=colors['blue4'],
        width=300,
        height=300)

    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

    if text is not None:
        fig.update_layout(
            width=300,
            height=300,
            annotations=[
                {
                    "text": text,
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "font": {
                        "size": 14,
                        "color": colors['blue1']
                    },
                    "valign": "top",
                    "yanchor": "top",
                    "xanchor": "center",
                    "yshift": 60,
                    "xshift": 10
                }
            ]
        )

    return fig


def add_emg_graphs(emg_data, frequency):

    if emg_data is None:
        return []

    graphs = []

    for i in range(emg_data.shape[0]):
        time_array=get_time_array(emg_data.shape[1], frequency)
        fig=go.Figure(data=go.Scatter(x=time_array, y=emg_data[i], mode='lines'))
        fig.update_layout(
            title="EMG Track " + str(i),
            xaxis_title="Time [s]",
            yaxis_title="micro Volts",
            legend_title="Legend Title"
        )
        graphs.append(dcc.Graph(
            id='emg'+str(i)+'-graph',
            figure=fig
        ))

    return graphs


def add_ventilator_graphs(vent_data, frequency):

    if vent_data is None:
        return []

    graphs = []

    for i in range(vent_data.shape[0]):
        time_array=get_time_array(vent_data.shape[1], frequency)
        fig=go.Figure(data=go.Scatter(x=time_array, y=vent_data[i], mode='lines'))
        fig.update_layout(
            title="Ventilator Track " + str(i),
            xaxis_title="Time [s]",
            legend_title="Legend Title"
        )
        graphs.append(dcc.Graph(
            id='ventilator'+str(i)+'-graph',
            figure=fig
        ))

    return graphs


def get_time_array(data_size, frequency):
    time_array = np.arange(0, data_size / frequency, 1 / frequency)

    return time_array
