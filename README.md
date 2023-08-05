<p align="center">
    <img style="width: 35%; height: 35%" src="https://github.com/resurfemg/resurfemg/blob/main/Logo_rond_tekst.svg">
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6811554.svg)](https://doi.org/10.5281/zenodo.6811554)
[![PyPI](https://img.shields.io/pypi/v/resurfemg.svg)](https://pypi.python.org/pypi/resurfemg/)
[![Anaconda-Server Badge](https://anaconda.org/resurfemg/resurfemg/badges/version.svg)](https://anaconda.org/resurfemg/resurfemg)
[![Sanity](https://github.com/resurfemg/resurfemg/actions/workflows/on-commit.yml/badge.svg)](https://github.com/resurfemg/resurfemg/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/resurfemg/resurfemg/actions/workflows/on-tag.yml/badge.svg)](https://github.com/resurfemg/resurfemg/actions/workflows/on-tag.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6487/badge)](https://bestpractices.coreinfrastructure.org/projects/6487)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
[![status](https://joss.theoj.org/papers/5f08d1f2bb717b7d05762296e37ded3d/status.svg)](https://joss.theoj.org/papers/5f08d1f2bb717b7d05762296e37ded3d)

**ReSurfEMG** is an open source collaborative python library for analysis
of respiratory electromyography (EMG).  On the same site as 
the repository for this library we keep [related resources](https://github.com/ReSurfEMG?tab=repositories). 

Most important to know before using ReSurfEMG is that we have a [main code library](https://github.com/ReSurfEMG/ReSurfEMG) where the user can access the code to change various filter and analysis settings directly and/or in our [researcher interface notebooks](https://github.com/ReSurfEMG/ReSurfEMG/tree/main/researcher_interface), and a [dashboard interface](https://github.com/ReSurfEMG/ReSurfEMG-dashboard) which contains default settings for preprocessing and analysis which can be changed through a graphical (no code) interface. We have some functionality available through a [command line interface](#command-line-interface) as well.

The library was initially
built for surface EMG, however many functions will also work for
respiratory EMG directly acquired (trans-esophageal).  This library
supports the ongoing research at University of Twente on respiratory
EMG.


### Program files

The main program in this repository (made of the modules in the resurfemg folder) contains functions for analysis of EMG and other electrophysiological signals. This analysis often includes
analysis of signals from machines i.e. ventilators as well.

### Folders and Notebooks

Our [guide to notebooks](https://github.com/ReSurfEMG/ReSurfEMG/blob/main/notebooks_guide.md) is under construction. To look around keep in mind the following distinction on folders:

researcher_interface:
- These are a growing series of interactive notebooks that allow
  researchers to investigate questions about their own EMG data
  - ⚡ Important: in almost all data there will be a time 
  difference between EMG signals and ventilator signals. You can
  pre-process to correct for this lead or lag with the notebook
  called lead_lag_mismatch_upsample.

open_work:
- This folder contains experimental work by core members of the ReSurfEMG
  team (Dr. Eline Mos-Oppersma, Rob Warnaar, Dr. Walter Baccinelli and Dr. Candace Makeda Moore)


### Data sets

The notebooks are configured to run on various datasets.  Contact
Dr. Eline Mos-Oppersma( 📫 e.mos-oppersma@utwente.nl) to discuss any
questions on data configuration for your datasets.

If you want to use a standardized dataset for any purpose we reccomend
the data in the ReSurfEMG/synthetic_data repository

[![DOI](https://zenodo.org/badge/635680008.svg)](https://zenodo.org/badge/latestdoi/635680008)

Data there can be used with any respiratory EMG algorithms in any program. Thus that data can function as a benchmarking set to compare algorithms across different programs.


### Configuring (to work with your data)

In order to preprocess and/or to train  models the code needs to be
able to locate the raw data you want it to find.

There are several ways to specify the location of the following
directories:

-   **root_emg_directory:** Special directory.  The rest of the directory layout can
    be derived from its location.
-   **preprocessed:** The directory that will be used by preprocessing
    code to output to.
-   **models:** The directory to output trained models to.

You can store this information persistently in several locations.

1.  In the same directory where you run the script (or the notebook).
    e.g. `./config.json`.
2.  In home directory, e.g. `~/.resurfemg/config.json`.
3.  In global directory, e.g. `/etc/resurfemg/config.json`.

However, we highly recommend you use the home directory.
This file can have this or similar contents:

    {
 
        'root_emg_directory': '/mnt/data',
        'preprocessed': '/mnt/data/preprocessed',
        'models': '/mnt/data/models',
        'output': '/mnt/data/output',
    }

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


## Getting Started
#### with the reccomended Conda setup

How to get the notebooks running?  Assuming the raw data set and
metadata is available. Note for non-conda installations see next sections.

0. Assuming you are using conda for package management:    
  * Make sure you are in no environment:

      ```sh
      conda deactivate
      ```

      _(optional repeat if you are in the base environment)_

      You can build on your
      base environment if you want, or if you want to not use option A, you can go below it (no environment)


1. Option A: Fastest option:
  In a base-like environment with mamba installed, you can install all Python packages required, using `mamba` and the `environment.yml` file. 

  If you do not have mamba installed you can follow instructions [here](https://anaconda.org/conda-forge/mamba)
  


   * The command for Windows/Anaconda/Mamba users can be something like:

     ```sh
     mamba env create -f environment.yml
     ```

Option B: To work with the most current versions with the possibility for development:
  Install all Python packages required, using `conda` and the `environment.yml` file. 


   * The command for Windows/Anaconda users can be something like:

     ```sh
     conda env create -f environment.yml
     ```

   * Linux users can create their own environment by hand (use
     install_dev as in setup).
    
  Make sure to enter your newly created environment.

Option C: In theory if you want to work, but never develop (i.e. add code), as a conda user
   with a stable (released) version create an empty environment, and install
   there: 


   * Create a blank environment with python pinned to 3.8 (assuming version < 0.1.0):

     ```sh
     conda create -n blank python=3.8
     ```

   * Install within the blank environment
      ```sh
        conda activate blank
        conda install -c conda-forge -c resurfemg resurfemg jupyter ipympl
        ```

2. Open a notebook (we use [Jupyter
   notebooks](https://jupyter.org/try-jupyter/retro/notebooks/?path=notebooks/Intro.ipynb))
   in researcher_interface folder and interactively run the cells.
   You can use the command `jupyter notebook` to open a browser window
   on the folders of notebooks.  Note, if you run with an installed
   library import appropriately.

### Supported Platforms

ReSurfEMG is a pure Python package. Below is the list of
platforms that should work. Other platforms may work, but have had less extensive testing.
Please note that where
python.org Python or Anaconda Python stated as supported, it means
that versions 3.8 or 3.9 (depending on the release) are supported.

#### AMD64 (x86)

|                             | Linux     | Win       | OSX       |
|:---------------------------:|:---------:|:---------:|:---------:|
| ![p](etc/python-logo.png)   | Supported | Unknown   | Unknown   |
| ![a](etc/anaconda-logo.png) | Supported | Supported | Supported |

### Installation for all supported platforms

Installation with Anaconda/conda and/or mamba are covered in [the
"Getting Started" section](#Getting-Started). If you wish to install
with pip:

1. Create and activate virtual environment.
2. Install ResurfEMG package by running `pip install resurfemg`.


## Developer's setup

We are currently transitioning to a fully Python 3.9 environment. The
instructions below are for our versions below 0.1.0:

After checking out the source code, create virtual environment.  Both
`conda` and `venv` environments are supported, however, if you are on
Windows, we reccomend using `conda`. 
For instructions with `venv` please
see versions below 0.1.0, and adapt them if using Windows and/or a different Python version
than Python.org Python e.g. you may need to use `.venv/Scripts/activate` in place of
`.venv/bin/activate`.
   This will create a distributable package from the your source code,
   then install it in the currently active environment.  This will
   also install development tools we use s.a. `pytest` and
   `codestyle` and  will also install tools we use for working with
   the library, s.a. `jupyter`.

1. Using Anaconda Python

   ```sh
   conda create -n resurfemg python=3.8
   conda activate resurfemg
   python setup.py anaconda_gen_meta
   python setup.py install_dev
   ```

   Note, you will need to run `anaconda_gen_meta`.  This generates
   `meta.yaml` which is necessary to create `conda` package.  In the
   future this will probably be called automatically by `install_dev`.

2. Using PyPI Python

   ```sh
   python3.9 -m venv .venv3.9
   # On Linux:
   . .venv3.9/bin/activate
   # On Windows:
   .venv3.9/Scripts/activate
   python setup.py install_dev
   ```

Now you should have everything necessary to start working on the
source code.  Whenever you make any changes, re-run `install_dev` to
see them applied in your environment.  It's possible to find a
shortcut sometimes to running the entire cycle (`install_dev` takes a
long time to run).  Look at the source of `bdist_conda` and
`sdist_conda` commands in `setup.py` for more info.


## Generating documentation

Online documentation can be found at
https://resurfemg.github.io/ReSurfEMG/
or on https://readthedocs.org/ by searching for ReSurfEMG.
Up to date documentation can be generated in command-line as follows
(in bash terminal):

``` sh
python3 -m venv .venv
. ./.venv/bin/activate
pip install wheel sphinx
./setup.py install
./setup.py apidoc
./setup.py build_sphinx
```

If you are working in a VScode command line interface (terminal cmd)
should be more or less something like the following:

This is given with `cmd.exe` in mind:

``` sh
python3 -m venv .venv
.venv/bin/activate
pip install wheel sphinx
python setup.py install
python setup.py apidoc
python setup.py build_sphinx
```


## Automation

The project comes with several modifications to the typical
default `setup.py`.

At the moment, the support for Anaconda Python is lacking.  The
instructions and the commands listed below will work in Anaconda
installation but due to the difference in management of installed
packages, the project will be installed into base environment
regardless of the currently active one.  We are planning to integrate
with Anaconda in near future.

The project has a sub-project of a related dashboard.  Dashboard is a GUI that
exposes some of the project's functionality. In the past we kept a a legacy dashboard
in the same repository with ReSurfEMG code but we have deleted it. The
current version of the dashboard into it's own repository:
https://github.com/ReSurfEMG/ReSurfEMG-dashboard


### New commands

Commands that perform repeating tasks have a `--fast` option.  Use
this if you ran `setup.py install_dev`, and you are sure the
dependencies are up to date.  Otherwise, these commands will create a
new virtual environment and install necessary dependencies there
before execution.  This is primarily intended for use in CI to create
controlled environment.

Please note that documentation is built using `sphinx` command
for `setuptools`: `setup.py build_sphinx`, but `sphinx` is not
installed as part of development dependencies, rather it is declared
as a dependency of `setup.py` itself.  There are cases when `setup.py`
will not install its own dependencies.  You are advised to install
them manually.

* `setup.py lint` checks that the source code is formatted according to
  PEP-8 recommendations.
* `setup.py isort` checks that the imports are properly formatted and
  sorted.
* `setup.py apidoc` generates RST outlines necessary to generate
  documentation.
* `setup.py install_dev` installs dependencies necessary for
  development.
  
### Modified commands

* `setup.py test` the original command is overloaded with a new one.
  Similarly to most of the new commands, this command takes `--fast`
  as an option with the same meaning.  Unlike its predecessor, this
  command will create a clean environment and install all necessary
  dependencies before running the test (the original did install
  dependencies, but ran the test in the source directory, this made it
  impossible to test code that relied on being installed).
  
  
### Installing from source

The traditional way to install from source is to run `setup.py
install` or `setup.py develop`.  Both of these will
work... sort of.  This is because of the default behavior inherited from
`setuptools`.  The problem here is that instead of creating a
distributable package and installing that, `setuptools` does it
in the other order: it installs the package in order to create a distributable
one.

As was already mentioned earlier, Anaconda Python will need better
support, at which point, `setup.py install` will have to change to
enable that.

We are not planning on patching `setup.py develop` as we don't believe
it is a good practice to use this command.  It is not removed,
however, and should work in the same way it would work with a typical
`setuptools` project.  

Note that `pip install -e .` and `pip install -e '.[dev]'` are
discouraged by association (since that is just a wrapper around
`setup.py develop`.)  Similarly to `setup.py develop` they might work,
but you have to be careful with interpreting the results.  If those
don't work, it's on you.


### Testing

The project doesn't include testing data.  It was packaged into a Docker
image which can be found at `crabbone/resurfemg-poly5-test-files:latest`.
This test data was created by taking a signal from equipment, not a human,
and certainly not a patient.

It is possible to run tests in container created from this image.
Alternatively, you may download the image and extract directory
`/ReSurfEMG/tests/not_pushed` into `not_pushed` in the root of the
project.

Below is a snippet that may help you to run the tests in a container:

``` sh
docker run --rm -v $(pwd):/ci \
    --cap-add=SYS_ADMIN \
    --privileged=true \
    crabbone/resurfemg-poly5-test-files \
    sh -c 'set -xe
        cd /ci
        mkdir -p ./not_pushed/
        mount --bind /ReSurfEMG/tests/not_pushed/ ./not_pushed/
        python setup.py test'
```

## Command-Line Interface

You will be able to preprocess, train and use models using command-line interface.
You can also, in some cases, create files in the correct format for our Dashboard
in a per folder batch process.

Below is an example of how to do that:

This will pre-process (with the alternative_a_pipeline_multi algorithm) the
 Poly5 files in the
`/mnt/data/originals` directory, and output leads 1 and 2 preprocessed.
(Note the \ symbol is simply a line-break and not meant to be included.)

    python -m resurfemg acquire \
           --input /mnt/data/originals \
           --lead 1 --lead 2 \
           --output /mnt/data/preprocessed \
           --preprocessing alternative_a_pipeline_multi \
           

The following will run an ML model over all files:

    python -m resurfemg ml |
            --input /mnt/data/preprocessed \
            --output /mnt/data/ml_output \
            --model  ml_models/finalized_lr_model_in_111.sav \

You can also make synthetic data. To explore this start with
    `python -m resurfemg synth --help`
You can also make from horizontally formated csv files 
that can be read by the dashboard. To explore this start with
    `python -m resurfemg save_np --help`
The help commandis also available for ml and acquire.

All long options have short aliases.


✨Copyright 2022 Netherlands eScience Center and U. Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.✨
