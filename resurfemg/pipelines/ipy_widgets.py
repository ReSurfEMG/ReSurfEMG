
"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains Jupyter widgets to perform default procedures.
"""
import numpy as np
import ipywidgets as widgets


def file_select(files, folder_levels, default_select=None):
    """
    A widget for file selection for organised/nested data.
    :param files: file paths tabled by the folder_levels
    :type files: pd.DataFrame
    :param folder_levels: data directory organisation, e.g. ['patient', 'date']
    :type folder_levels: list(str)
    :param default_select: list of default indices to select per folder_level
    :type default_select: list(int)

    :returns button_list: file paths tabled by the folder_levels
    :rtype button_list: [ipywidgets.widgets.widget_selection.Dropdown]
    """
    button_list = list()
    btn_dict = dict()
    for _, folder_level in enumerate(folder_levels[:-1]):
        _btn = widgets.Dropdown(
            description=folder_level + ':',
            disabled=False,
        )

        button_list.append(_btn)
        btn_dict[folder_level] = _btn

    @widgets.interact(**btn_dict)
    def update_select(**btn_dict):
        for idx, dict_key in enumerate(btn_dict):
            btn_idx = folder_levels.index(dict_key)
            _btn = button_list[btn_idx]

            if idx == 0:
                filter_files = files
            else:
                bool_list = list()
                for _idx in range(idx):
                    bool_list.append(
                        (files[folder_levels[_idx]] ==
                         button_list[_idx].value).values)

                filter_files = files[np.all(np.array(bool_list), 0)]

            options = list(set(filter_files[dict_key].values))
            options.sort()
            if default_select is None or default_select[btn_idx] is None:
                if _btn.value in options:
                    value = options[options.index(_btn.value)]
                elif len(options) > 0:
                    value = options[0]
            else:
                if default_select[btn_idx] in options:
                    value = options[options.index(default_select[btn_idx])]
                elif len(options) > 0:
                    value = options[0]

            _btn.options = options
            if len(options) > 0:
                _btn.value = value
    return button_list
