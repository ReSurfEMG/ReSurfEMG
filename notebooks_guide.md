# Notebooks guide

You will see various notebooks you can use and adapt alongside our package. If you read the whole document we will describe each notebook (work in progress). For the impatient, here are some quick links:

[You want to properly align your ventilator and EMG signal](#align)


[You want to understand gating](#gating)


[You want to make array-snippets for ML](#snippets)


[You want to see some ML in action](#mlaction)

### Lag lead mismatch
#### align

You can either upsample the less sampled lead or downsample the more frequently sampled lead. For us only upsampling works as shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/lead_lag_match_upsample.ipynb) but you can also try it the other way [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/lag_lead_match.ipynb)

### Gating exploration
#### gating

Our explortion of gating is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/gating_example.ipynb) 


### Machine learning snippets
#### snippets

Our snippets of arrays fed into machine learning are checked by hand, however, if you want to cut your arrays in an automated manner first, you can examine work we did on this  [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/ml_snipper_maker.ipynb) 

### Machine learning in action
#### mlaction

Our use of ML is currently around differentiating inhale and exhale segments based on extracted features, as shown  [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ML_EMG_1.ipynb) 


### Historical notebooks
#### history

Our [historical notebooks](https://github.com/ReSurfEMG/ReSurfEMG/tree/main/historical_notebooks) are basically notebooks we keep around just in case we want to look back on them. If you feel one is relevant to the entire community, let us know and we will add it to the researcher interface in a more polished form.
Included in these notebooks are:
            - [ a lag lead mismatch with downsampling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/historical_notebooks/lag_lead_match_downsample_BAD.ipynb)
            -[ another lag lead mismatch with downsampling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/historical_notebooks/lag_lead_match.ipynb)
            -
            -


Work in progress, needs to be finished for every single notebook.


