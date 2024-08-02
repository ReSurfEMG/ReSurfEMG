<p align="center">
    <img style="width: 35%; height: 35%" src="https://github.com/resurfemg/resurfemg/blob/main/Logo_rond_tekst.svg">
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6811554.svg)](https://doi.org/10.5281/zenodo.6811554)
[![PyPI](https://img.shields.io/pypi/v/resurfemg.svg)](https://pypi.python.org/pypi/resurfemg/)
[![Sanity](https://github.com/resurfemg/resurfemg/actions/workflows/on-commit.yml/badge.svg)](https://github.com/resurfemg/resurfemg/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/resurfemg/resurfemg/actions/workflows/on-tag.yml/badge.svg)](https://github.com/resurfemg/resurfemg/actions/workflows/on-tag.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6487/badge)](https://bestpractices.coreinfrastructure.org/projects/6487)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
[![status](https://joss.theoj.org/papers/5f08d1f2bb717b7d05762296e37ded3d/status.svg)](https://joss.theoj.org/papers/5f08d1f2bb717b7d05762296e37ded3d)

**ReSurfEMG** is an open source collaborative python library for analysis
of respiratory electromyography (EMG). On the same site as 
the repository for this library we keep [related resources](https://github.com/ReSurfEMG?tab=repositories). 

ReSurfEMG includes a [main code library](https://github.com/ReSurfEMG/ReSurfEMG) where the user can access the code to change various filter and analysis settings directly and/or in our [researcher interface notebooks](https://github.com/ReSurfEMG/ReSurfEMG/tree/main/researcher_interface).
In addition, ReSurfEMG has a [dashboard interface](https://github.com/ReSurfEMG/ReSurfEMG-dashboard) which contains default settings for preprocessing and analysis which can be changed through a graphical (no code) interface. We have some functionality available through a [command line interface](#command-line-interface) as well.

The library was initially built for surface EMG, however many functions will also work for
invasively measured respiratory EMG.  This library supports the ongoing research at University of Twente on respiratory EMG.


### Program files

The core functions of ReSurfEMG are in the folder [resurfemg](https://github.com/ReSurfEMG/ReSurfEMG/tree/main/resurfemg):

-   **cli:** Scripts for the command line interface
-   **config:** Configure all paths for data analysis
-   **data_connector:**  Converter functions for various hardware/software and the TMSisdk lite module
-   **helper_functions:** General functions to support the functions in this repository
-   **pre_processing:** Process the raw respiratory EMG signal
      - filtering: cutters, low-, high- and bandpass, notchfilter, computer power loss
      - ecg-removal: independent component analysis (ICA) and gating
      - envelope: root-mean-square (RMS), average rectified (ARV) and smoothers
-   **post_processing:** Aspects of pre-processed the respiratory EMG data:
      - moving baselines
      - event detection: find pneumatic and EMG breaths, on- and offset detection
      - features: area under the curve, slope, area under the baseline
      - quality assessment: signal-to-noise ratio, end-expiratory occlussion manoeuvre quality, interpeak distance, area under the baseline, consecutive manoeuvres, bell-curve error
-   **visualization:** Show powerspectrum
-   **data_classes:** Store and process EMG and ventilator data in an object-oriented way.


### Folders and Notebooks

Our [guide to notebooks](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks_guide.md) is under construction. To look around keep in mind the following distinction on folders:

dev:
- These notebooks are used in feature development and debugging by core members of the ReSurfEMG team. They can provide a basic example how to use some of the functionality.

open_work:
- This folder contains experimental work by core members of the ReSurfEMG
  team that is not deployed yet.

researcher_interface:
- These are a growing series of interactive notebooks that allow
  researchers to investigate questions about their own EMG data

### Data sets

The notebooks are configured to run on various datasets.  Contact
Dr. Eline Mos-Oppersma( 📫 e.mos-oppersma@utwente.nl) to discuss any
questions on data configuration for your datasets.

If you want to use a standardized dataset for any purpose we recommend
the data in the ReSurfEMG/synthetic_data repository

[![DOI](https://zenodo.org/badge/635680008.svg)](https://zenodo.org/badge/latestdoi/635680008)

Data there can be used with any respiratory EMG algorithms in any program. Thus that data can function as a benchmarking set to compare algorithms across different programs.

Alternatively, the data in the [test data folder](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/test_data/), which is also used in testing the ReSurfEMG functions.


### Configuring (to work with your data)

In order to preprocess and/or to train  models the code needs to be
able to locate the raw data you want it to find.

There are several ways to specify the location of the following
directories:

-   **root_emg_directory:** Special directory. The rest of the directory layout can be derived from its location.
-   **preprocessed:** The directory that will be used by preprocessing
    code to output to.

You can store this information persistently in several locations.

1.  In the same directory where you run the script (or the notebook).
    e.g. `./config.json`.
2.  In home directory, e.g. `~/.resurfemg/config.json`.
3.  In global directory, e.g. `/etc/resurfemg/config.json`.

However, we highly recommend using the home directory.
This file can have this or similar contents:
```
{
    "root_emg_directory": "/mnt/data",
    "preprocessed": "/mnt/data/preprocessed",
    "output": "/mnt/data/output",
}
```
The file is read as follows: if the files specifies `root_emg_directory`
directory, then the missing entries are assumed to be relative to
the root.  You don't need to specify all entries.

### Test data

You can get test data by extracting it from the Docker image like
this:

``` sh
mkdir -p not_pushed
cd ./not_pushed
docker create --name test-data crabbone/resurfemg-poly5-test-files:latest
docker cp test-data:/ReSurfEMG/tests/not_pushed/. .
docker rm -f test-data
```

### Supported Platforms

ReSurfEMG is a pure Python package. Below is the list of
platforms that should work. Other platforms may work, but have had less extensive testing.
Please note that where
python.org Python stated as supported, it means
that versions 3.9 and 3.10 are supported.

#### AMD64 (x86)

|                             | Linux     | Win       | OSX       |
|:---------------------------:|:---------:|:---------:|:---------:|
| ![p](etc/python-logo.png)   | Supported | Supported   | Unknown   |
| ![a](etc/anaconda-logo.png) | Supported | Supported | Supported |

### Installation for all supported platforms

Installation with Anaconda/conda and/or mamba are the preffered methods.
They are covered in [the "Getting Started" section](#Getting-Started). 
If you wish to install with pip:

1. Create and activate a virtual environment (see developer setup section for more details) 
2. Install ResurfEMG package by running `pip install resurfemg`.


## Getting Started
#### with the recommended Python venv setup

How to get the notebooks running? Assuming the raw data set and
metadata is available. Note for non-conda installations see next sections.

0. Create a virtual environment using Python

  # On Linux/OSX:
``` sh
python3 -m venv .venv
```

  **This might require the python3-venv.**
    
  # On Windows:
  ``` sh
  python -m venv .venv
  pip install -e resurfemg[dev]
  ```

1. Activate the virtual environment and install ReSurfEMG

  # On Linux/OSX:
``` sh
source .venv/bin/activate
pip install -e resurfemg[dev]
```
    
  # On Windows:
  ``` sh
  python -m venv .venv
  .venv\Scripts\activate.bat
  pip install -e resurfemg[dev]
  ```

2. Open a notebook
   Start a local Jupyter Notebook server by running the `jupyter notebook` 
   command in your terminal. This opens a browser window where you can browse, 
   open and run the notebooks. (We use [Jupyter notebooks](https://jupyter.org/try-jupyter/retro/notebooks/?path=notebooks/Intro.ipynb))
   The ReSurfEMG notebooks are located in the notebooks folder. Navigate there 
   and open a notebook of interest. The [basic_emg_analysis](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/researcher_interface/basic_emg_analysis.ipynb) 
   notebook can be used to understand how to use the package.


## Advanced contributor's setup / "Developer's setup"

We distinguish between people who want to use this library in their own code 
and/or analysis, and people who also want to develop this library who we call 
developers, be it as members of our team or independent contributors. People 
who simply want to use our library need to install the packaged version from 
one of the package indexes to which we publish released versions (eg. PyPI). 
This section of the readme is for advanced developers who want to modify the 
library code (and possibly contribute their changes back or eventually publish 
thier own modified fork). NB: You can accomplish modifications of the code, 
submit pull-requests (PRs) and soforth without a 'developer's setup' but we 
feel this setup will make advanced contributions easier.

We have transitioned to a fully Python 3.9+ environment. 
(For older instructions with `venv` please see versions below 0.2.0, and
adapt them if using Windows and/or a different Python version than
Python.org Python. e.g. you may need to use `.venv\Scripts\activate.bat` in
place of `.venv/bin/activate`) 
The instructions below are for our newer versions above 3.0.0. This will create 
a distributable package from the source code, then install it in the currently
active environment.  This will also install development tools we use
s.a. `pytest` and `codestyle` and will also install tools we use for
working with the library, such as `jupyter`.

# On Linux/OSX:
``` sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
``` 
  
# On Windows:
``` sh
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .[dev]
```

These installs differ in two ways from the regular install: 1) The `.[dev]` 
installs the library as it currently is including all your local changes, 
instead of pulling it from the PyPI repository.  2) The `-e` flag ensures an 
editable install, such that any changes to the library are automatically 
applied to your environment.

Now you should have everything necessary to start working on the
source code.


## Generating documentation

Online documentation can be found at https://resurfemg.github.io/ReSurfEMG/
or on https://readthedocs.org/ by searching for ReSurfEMG. Up-to-date 
documentation can be generated in command-line as follows in terminal:

# On Linux/OSX:
``` sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[docs]
python setup.py apidoc
python setup.py build_sphinx
``` 
  
# On Windows:
``` sh
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .[docs]
python setup.py apidoc
python setup.py build_sphinx
```


## Automation

The project comes with several modifications to the typical
default `setup.py`.

The project has a sub-project of a related dashboard. Dashboard is a GUI that
exposes some of the project's functionality. In the past, we kept a a legacy 
dashboard in the same repository with ReSurfEMG code but we have deleted it. 
The current version of the dashboard into it's own repository:
https://github.com/ReSurfEMG/ReSurfEMG-dashboard


### New commands

* `isort resurfemg --check --diff` checks that the imports are properly 
  formatted and sorted.
* `setup.py apidoc` generates RST outlines necessary to generate
  documentation.

### Testing

The project includes testing data. This test data is synthetic, not acquired 
from human, and certainly not a patient.

# On Linux/OSX:
``` sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[tests]
pytest
``` 
  
# On Windows:
``` sh
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .[tests]
pytest
```


## Command-Line Interface

You will be able to preprocess data using command-line interface. You can also,
in some cases, create files in the correct format for our Dashboard in a per 
folder batch process.

You can make synthetic data. To explore this start with
    `python -m resurfemg synth --help`
You can also make from horizontally formated csv files 
that can be read by the dashboard. To explore this start with
    `python -m resurfemg save_np --help`

All long options have short aliases.


✨Copyright 2022 Netherlands eScience Center and U. Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.✨
