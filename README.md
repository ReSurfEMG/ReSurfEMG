## ReSurfEMG
<p align="center">
    <img style="width: 10%; height: 10%" src="https://github.com/resurfemg/resurfemg/blob/main/resurfemg.png">
</p>

ReSurfEMG is an open source collaborative python library for analysis
of respiratory electromyography (EMG).  The library was initially
built for surface EMG, however many functions will also work for
respiratory EMG directly acquired (trans-esophageal).  This library
supports the ongoing research at University of Twente on respiratory
EMG.


### Folders and Notebooks

researcher_interface:
- These are a growing series of interactive notebooks that allow
  researchers to investigate questions about their own EMG data
  - ⚡ Important: in almost all data there will be a time 
  difference between EMG signals and ventilator signals. You can
  pre-process to correct for this lead or lag with the notebook
  called lead_lag_mismatch_upsample.

open_work:
- This folder contains experimental work by core members of the rsemg
  team


### Program files

The main program in this repository contains functions for analysis of
EMG.


## Data sets

The notebooks are configured to run on various datasets.  Contact
Candace Makeda Moore (c.moore@esciencecenter.nl) to discuss any
questions on data configuration.


## Getting started

How to get the notebooks running? Assuming the raw data set and
metadata is available.

1. Install all Python packages required, using conda and the
   `environment.yml` file.
   * The command for Windows/Anaconda users can be something like:
     `conda env create -f environment.yml`.
   * Linux users can create their own environment by hand.
2. Open a notebook in researcher_interface and interactively run the
   cells.


## Generating documentation

Up to date documentation can be generated in command-line as follows
(in bash terminal):

``` sh
sphinx-apidoc -o ./docs  -f --separate ./resurfemg 
rm -rf ./built_documentation
sphinx-build -b html ./docs ./built_documentation
```

If you are working in a VScode command line interface (terminal cmd):

``` sh
sphinx-apidoc -o ./docs  -f --separate ./resurfemg 
rm -rf ./built_documentation
sphinx-build -b html ./docs ./built_documentation
```

Online documentation can be found at
https://resurfemg.github.io/ReSurfEMG/
or on https://readthedocs.org/ by searching for ReSurfEMG



<!--
**ReSurfEMG/ReSurfEMG** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
