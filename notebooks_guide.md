# Notebooks guide :rainbow:

ReSurfEMG contains various notebooks you can use and adapt alongside our package. The researcher_interface folder contains notebooks that are more or less 'standard' procedures. The open_work folder contains notebooks with more experimental work. The dev folder contains notebooks that were used for feature development and debugging. This document is a work in progress overview of all notebooks. Below is a list providing a general overview of the notebooks and their purpose. 

## **Researcher_interface**

### Getting started

> This [notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/getting.ipynb): describes in more detail why certain steps are taken in Notebooks.

### Basic EMG analysis

> This notebook provides [the basic EMG analysis](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/researcher_interface/basic_emg_pipeline.ipynb): picking a data file and sample, visualization, filter and ECG removal, envelope calculation, detection of peaks including on- and offset, feature calculation, and quality assessment.


### Synthetic ecg maker

> This [notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/researcher_interface/synthetic_ecg_maker.ipynb): provides the code to generate your own synthetic EMG data and store it to csv.

### Neuro-muscular coupling

> The analyses on [neuro-muscular coupling](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/researcher_interface/publication_2024_neuromuscular_coupling) used for the [Warnaar et al. (2024)](https://doi.org/10.1186/s13054-024-04978-0) paper are in this folder. Included are the option for saving experiments, there is preprocessing pipeline with filters, ECG gating, RMS, identifying PEEP levels and occlusion pressure from ventilator data, detecting diaphragm EMG peaks, and some plotting and visualization options.


## **Development (dev)**

### Features

#### File-reading
> How to load data. This feature is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/file_reading.ipynb).

#### Gating
> One way to reduce the heart signal, is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/gating_example.ipynb) and [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/gating_pipeline.ipynb).

#### Root-mean-square and Average-rectified (ARV)
> Methods to calculate an EMG envelope. An example is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/rms_arv_example.ipynb).

#### Data classes
> Data classes bundle EMG data storage, processing, and plotting in an object oriented way, minimizing the lines of code needed for basic processing. An example is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/pipeline_structure.ipynb).

#### Area under the baseline (AUB)
> A peak feature indicating the reliability of a peak by how much of the signal relative to the surroundings is under the baseline. This feature is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/aub_test.ipynb).

#### Bell fit
> A peak feature indicating the reliability of a peak by how much it resembles it bell curve. This feature is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/bell_fit_example.ipynb).

#### Pocc quality
> A feature indicating the reliability of an end-expiratory occlusion manoevure by how steep the release is. This feature is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/pocc_quality_test.ipynb).

#### Pseudo SNR
> A peak feature indicating the reliability of a peak by the peak amplitude relative to the baseline level.. This feature is shown [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/feature_development/pseudo_snr.ipynb).

### Debugging

#### Gating
> We encountered issues when gating windows overlapped. This resolved using this [notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/debugging/gating_adjacent_peaks_example.ipynb).

#### Root-mean-square centralization
> We encountered issues with the centralization of the RMS method. This resolved using this [notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/dev/debugging/rms_centralisation_example.ipynb).

## **Open_work**

#### Entropy

> There are multiple entropy notebooks; this notebook provides [multiple approaches to use entropy for breath detection](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/entropy_eline_near_final.ipynb), using different thresholds and visualization on breath by breath basis, as well as a manual check of breath count. We can use it in various ways to automatically identify inhalation [using multiple cut-offs](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/entropy_eline_near_final.ipynb/entropy_widgeted_updated_june.ipynb). Entropy is also used as part of some workflows to determine strength of respiratory efforts (listed in a separate section).

#### Align

> Lag lead mismatch: You can either upsample the less sampled lead or downsample the more frequently sampled lead. For us only upsampling works as shown [in this lag-lead notebook](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/lag_lead_match_upsample.ipynb), but you can also try it the other way i.e. downsampling the fast signal [here](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/lag_lead_match.ipynb).

#### Information loss

> Reading EMG requires a pre-processing pipeline, but how much information is lost in each step? We explored this using the power spectrum in a notebook on [information loss](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/information_loss_widgeted_seconds.ipynb)

#### Independent component analysis (ICA)

> Our explorations on Independent Component Analysis (ICA), a way to reduce the heart signal, are shown in
[this](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/ica_ecg_subtraction.ipynb), and
[this](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/ica_methods.ipynb)
 Notebook. The ICA functionality of ReSurfEMG is not fully functional yet.

## Historical notebooks :ghost:
Our historical notebooks are basically notebooks we keep around just in case we want to look back on them, or that we are still working on. If you feel one of these or your own contribution is relevant to the entire community, let us know and we will add it to the researcher interface in a more polished form.

These notebooks are:
### Entropy
* [entropy cycler](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/entropy_cycler_on_offset_detection.ipynb)
* [entropy testing](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/entropy_tests.ipynb)

### Independent component analysis
* [ica comparison](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/ica_comparison-Copy1.ipynb)
* [ica remix function](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/ica_remix_function.ipynb)
* [various ICA examples](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks/open_work/ica_various_examples.ipynb)
* [example of bad working ICA](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/open_work/ica_why_n_ICA_is_bad.ipynb)