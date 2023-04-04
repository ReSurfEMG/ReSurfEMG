# Notebooks guide :rainbow:

ReSurfEMG contains various notebooks you can use and adapt alongside our package. The researcher_interface folder contains notebooks that are more or less 'standard' procedures. The open_work folder contains notebooks with more experimental work. This document is a work in progress overview of all notebooks. Below is a list providing a general overview of the notebooks and their purpose. 

## **Researcher_ interface**

> #### Basic EMG analysis

>> This notebook provides [the basic EMG analysis](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/basic_emg_analysis.ipynb): picking a data file and sample, visualization, ECG removal with ICA and usage of the working_pipeline_pre_ml from helper_functions, resulting in calculation of length of breaths and absolute and percentage of total breath location of peaks. Area under the curve under construction.

> #### Entropy

>> There are multiple entropy notebooks; this notebook provides [multiple approaches to use entropy for breath detection](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/entropy_eline_near_final.ipynb), using different thresholds and visualization on breath by breath basis, as well as a manual check of breath count. We can use it in various ways to automatically identify inhalation  [using multiple cut-offs](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/widgeted_entropy_updated_june.ipynb) or [a single cutt off](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/widgeted_entropy1.ipynb). Entropy is also used as part of some workflows to determine stegnth of respiratory efforts (listed in a seperate section).

> #### Align

>> Lag lead mismatch: You can either upsample the less sampled lead or downsample the more frequently sampled lead. For us only upsampling works as shown [in this lag-lead notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/lead_lag_match_upsample.ipynb) but you can also try it the other way i.e. downsampling the fast signal [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/lag_lead_match.ipynb).


> #### Machine learning 

>> [This notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/ml_snipper_maker.ipynb) starts equal to the [previously listed entropy notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/entropy_eline_near_final.ipynb), but can then be used to create snippets of arrays to feed in to the machine learning model.

> #### Neuromuscular efficiency

>> Our first analyses on [neuromuscular efficiency](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/neuromuscular_efficiency_3lead_for_pub.ipynb) are in this notebook. Included are the option for saving experiments, there is preprocessing pipeline with filters and ECG gating, RMS, identifying PEEP levels and occlusion pressure from ventilator data, detecting diaphragm and parasternal EMG peaks, and some plotting and visualization options.

> #### Automated breath analyses

>> Our first analyses sought to quantify how many breaths occured based on simple parameters. We do not recconed this approach, but the interfaces for them exists [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/widgeted_breath_experiments.ipynb).

> #### Information loss

>> Reading EMG requires a pre-processing pipeline, but how much information is lost in each step? We explored this using the power spectrum in a notebook on [information loss](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/widgeted_info_loss_experiments_seconds.ipynb)


## **Open_work**

> #### Preprocessing exploration 

>> Our exploration of basic preprocessing is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/basic_preprocessing.ipynb).

> #### Gating 

>> Our initial exploration of gating, one way to reduce the heart signal, is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/gating_example.ipynb). 

> #### ML action 

>> Our use of machine learning is currently around differentiating inhale and exhale segments based on extracted features, as shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ML_EMG_1.ipynb) and [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ML_EMG_1-Copy1.ipynb). :key:


> #### EMG and vent analysis (emgandvent)

>> Our final analyses can examine both ventilator and EMG signals from the same patient to examine issues like synchrony, one approach is [looking at the overlap of simplified EMG and ventilator signals](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/diagnose_emg_vent_relationship.ipynb), another uses [edit distance](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/edit_distance_emg_vent.ipynb).


> #### Historical notebooks :ghost:

> Our historical notebooks are basically notebooks we keep around just in case we want to look back on them, or that we are still working on. If you feel one of these or your own contribution is relevant to the entire community, let us know and we will add it to the researcher interface in a more polished form.

> These notebooks are:

>> * [a lag lead mismatch with downsampling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/historical_notebooks/lag_lead_match_downsample_BAD.ipynb)
>> * [another lag lead mismatch with downsampling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/historical_notebooks/lag_lead_match.ipynb)
>> * [another lag lead mismatch with downsampling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/historical_notebooks/lag_lead_match.ipynb)
>> * [amplitude exploration](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/amplitude_exploration.ipynb)
>> * [area under curve work](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/area_under_curve_work.ipynb)
>> * [breath detection variability](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/breath_detection_variability.ipynb)
>> * [another breath detection variability](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/breath_detection_variability_nb.ipynb)
>> * [distances](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/distances.ipynb)
>> * [ECG puller](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ecg_puller.ipynb)
>> * [entropy cycler](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/entropy_cycler.ipynb)
>> * [entropy testing](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/entropy_tests.ipynb)
>> * [file reading](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/file_reading.ipynb)
>> * [ica comparison](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ica_comparison-Copy1.ipynb)
>> * [ica remix function](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ica_remix_function.ipynb)
>> * [ml inf loop](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ml_inf_loop.ipynb)
>> * [ml model creation](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ml_model_creation_111.ipynb)
>> * [n array](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/n_array.ipynb)
>> * [notes on nans](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/notes_on_nans.ipynb)
>> * [synthetic ECG maker](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/synthetic_ecg_maker.ipynb)
>> * [time smooth and slope](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/time_smooth_and_slope.ipynb)
>> * [various ICA examples](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/various_ICA_examples.ipynb)
>> * [example of bad working ICA](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/why_n_ICA_is_bad.ipynb)
>> * [widgeted baseline experiments](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/widgeted_baseline_experiments.ipynb)