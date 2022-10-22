<p align="center">
    <img style="width: 35%; height: 35%" src="https://github.com/resurfemg/resurfemg/blob/main/resurfemg_long.svg">
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6811554.svg)](https://doi.org/10.5281/zenodo.6811554)
[![PyPI](https://img.shields.io/pypi/v/resurfemg.svg)](https://pypi.python.org/pypi/resurfemg/)
[![Anaconda-Server Badge](https://anaconda.org/resurfemg/resurfemg/badges/version.svg)](https://anaconda.org/resurfemg/resurfemg)
[![Sanity](https://github.com/resurfemg/resurfemg/actions/workflows/on-commit.yml/badge.svg)](https://github.com/resurfemg/resurfemg/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/resurfemg/resurfemg/actions/workflows/on-tag.yml/badge.svg)](https://github.com/resurfemg/resurfemg/actions/workflows/on-tag.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6487/badge)](https://bestpractices.coreinfrastructure.org/projects/6487)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)

ReSurfEMG is an open source collaborative python library for analysis
of respiratory electromyography (EMG).  On the same site as 
the repository for this library we keep [related resources](https://github.com/ReSurfEMG?tab=repositories). 

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

However, we highly reccomend you use the home directory.
This file can have this or similar contents:

    {
 
        'root_emg_directory': '/mnt/data',
        'preprocessed': '/mnt/data/preprocessed',
        'models': '/mnt/data/models',
        'output': '/mnt/data/output',
    }

The file is read as follows: if the files specifies `root_emg_directory`
directory, then the missing entires are assumed to be relative to
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


## Getting started

How to get the notebooks running? Assuming the raw data set and
metadata is available.

0. If you want to work with the stable version create an empty
    environment, and install there:
    * Make sure you are in no environment:
      `conda deactivate` (repeat if you are in the base environment)
      You should be in no environment now 

    * Create a blank environment
      `conda create -n blank`

    * Install within the blank environment:
      `conda activate blank`
      `conda install -c conda-forge -c resurfemg resurfemg jupyter`

1. To work with the most current versions:
    Install all Python packages required, using conda and the
    `environment.yml` file.
   * The command for Windows/Anaconda users can be something like:
     `conda env create -f environment.yml`.
   * Linux users can create their own environment by hand.

2. Open a notebook in researcher_interface and interactively run the
   cells. Note, if you run with an installed library import appropriately


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

If you are working in a VScode command line interface (terminal cmd):

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
exposes some of the project's fucntionality. In the past we kept a a legacy dashboard
in the same repository with ReSurfEMG code but we have deleted it. The
current version of the dashboard into it's own repository:
https://github.com/ReSurfEMG/ReSurfEMG-dashboard

As of today, we use out own version of the package TMSiSDK together with our
project.  Hopefully, this is a temporary measure, and, eventually,
TMSiSDK will become just another independent project we may depend on.
The TMSiSDK project (which has an Apache 2 license) can be added as a submodule
containing our patched fork of the original code.
You will need to pull Git submodules in order to be able to properly
install and run the project.

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
as a dependancy of `setup.py` itself.  There are cases when `setup.py`
will not install its own dependencies.  You are advised to install
them manually.

* `setup.py lint` checks that the source code is formatted accoring to
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
discouranged by association (since that is just a wrapper around
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


✨Copyright 2022 Netherlands eScience Center and U. Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.✨
