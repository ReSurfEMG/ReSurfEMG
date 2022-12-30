# Notebooks guide :rainbow:

You will see various notebooks you can use and adapt alongside our package. If you read the whole document we will describe each notebook (work in progress). For the impatient, here are some quick links :fast_forward: :

> * [You want to configure everything to run on your own data](#configuration)
> * [You want to properly align your ventilator and EMG signal](#align)
> * [You want to understand gating](#gating)
> * [You want to see some basic analysis](#analysis)
> * [You want to make array-snippets for ML](#snippets)
> * [You want to see some ML in action](#mlaction)

Now a *comprehensive list* (under development, missing many notebooks as of 17/09/2022 :construction:): 



> #### Configuration

>> We include a [notebook which explains how to configure your data paths](http://localhost:8888/notebooks/open_work/config_demo.ipynb).


> #### Align

>> Lag lead mismatch: You can either upsample the less sampled lead or downsample the more frequently sampled lead. For us only upsampling works as shown [in this lag-lead notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/lead_lag_match_upsample.ipynb) but you can also try it the other way i.e. downsampling the fast signal [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/lag_lead_match.ipynb).


> #### Array time matching

>> Arrays from your EMG and venilator should match in time. For an exploration of our double check of this, see our [array_time_matching notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/array_time_matching.ipynb) (we later realized zero-padding was the culprit)

> #### Preprocessing exploration 

>> Our explortion of basic preprocessing is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/basic_preprocessing.ipynb).

> #### Gating 

>> Our initial explortion of gating, one way to reduce the heart signal, is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/gating_example.ipynb). 


> #### Entropy

>> Our explortion of entropy are multiple. We can use it in various ways to automatically identify inhalation  [using multiple cut-offs](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/widgeted_entropy_updated_june.ipynb) or [a single cutt off](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/widgeted_entropy1.ipynb). Entropy is also used as part of some workflows to determine stegnth of respiratory efforts (listed in a seperate section).


> #### Analysis

>> Explortion of very basic analysis for simple parameters in a preprocessed EMG signal is showen [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/basic_emg_analysis.ipynb)


> #### Snippets 

>> Our snippets of arrays fed into machine learning are checked by hand, however, if you want to cut your arrays in an automated manner first, you can examine work we did on [snipppet making](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/ml_snipper_maker.ipynb). 


> #### Mlaction 

>> Our use of machine learning is currently around differentiating inhale and exhale segments based on extracted features, as shown  [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ML_EMG_1.ipynb). :key:


> #### EMG and vent analysis (emgandvent)

>> Our final analyses can examine both ventilator and EMG signals from the same patient to examine issues like synchrony, one approach is [looking at the overlap of simplified EMG and ventilator signals](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/diagnose_emg_vent_relationship.ipynb) , another uses [edit distance](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/edit_distance_emg_vent.ipynb).


> #### Neuromuscular efficiency

>> Our first analyses on [neuromuscular efficiency](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/neuromuscular_efficiency_3lead_for_pub.ipynb) are part of research for an upcoming paper.


> #### Automated breath analyses

>> Our first analyses sought to quantify how many breaths occured based on simple parameters. We do not recconed this approach, but the interfaces for them exists [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/widgeted_breath_experiments.ipynb).

> #### Information loss

>> Reading EMG requires a pre-processing pipeline, but how much information is lost in each step? We explored this in a notebook on [informaiton loss](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/widgeted_info_loss_experiments_seconds.ipynb)



> #### Historical notebooks :ghost:
> ##### history

>> Our [historical notebooks](https://github.com/ReSurfEMG/ReSurfEMG/tree/main/historical_notebooks) are basically notebooks we keep around just in case we want to look back on them. If you feel one is relevant to the entire community, let us know and we will add it to the researcher interface in a more polished form.
Included in these notebooks are:


>>> * [a lag lead mismatch with downsampling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/historical_notebooks/lag_lead_match_downsample_BAD.ipynb)
>>> * [another lag lead mismatch with downsampling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/historical_notebooks/lag_lead_match.ipynb)
>>> * [another lag lead mismatch with downsampling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/historical_notebooks/lag_lead_match.ipynb)


Work in progress, needs to be finished for every single notebook.


