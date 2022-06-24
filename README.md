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

Online documentation can be found at
https://resurfemg.github.io/ReSurfEMG/
or on https://readthedocs.org/ by searching for ReSurfEMG


## Automation

The project comes with several modifications to the default `setup.py`.

At the moment, the support for Anaconda Python is lacking.  The
instructions and the commands listed below will work in Anaconda
installation but due to the difference in management of installed
packages, the project will be installed into base environment
regardless of the currently active one.  We are planning to integrate
with Anaconda in near future.

The project has a sub-project that is as of yet not properly
integrated with the rest of the source code.  Dashboard is a GUI that
exposes some of the project's fucntionality.  We are planning to make
it installable and configurable from `setup.py` too.  As the project
was developoed using Anaconda Python, the integration with the later
needs to happen first.

As of today, we had no choice but to package TMSiSDK together with our
project.  Hopefully, this is a temporary measure, and, eventually,
TMSiSDK will become an independent project we may depend on.  The
project is added as a submodule containing our patched fork of the
original code.  You will need to pull Git submodules in order to be
able to properly install and run the project.

### New commands

Commands that perform repeating tasks have a `--fast` option.  Use
this if you ran `setup.py install_dev`, and you are sure the
dependencies are up to date.  Otherwise, these commands will create a
new virtual environment and install necessary dependencies there
before execution.  This is primarily intended for use in CI to create
controlled environment.

while not new, note that documentation is built using `sphinx` command
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
work... incorrectly.  This is the default behavior inherited from
`setuptools`.  The problem here is that instead of creating a
distributable package and installing that, `setuptools` does it
backwards: it installs the package in order to create a distributable
one.

As was already mentioned earlier, Anaconda Python will need better
support, at which point, `setup.py install` will have to change to
enable that.

We are not planning on patching `setup.py develop` as we don't believe
it is a good practice to use this command.  It is not removed,
however, and should work in the same way it would work with a typical
`setuptools` project.  If you are feeling adventurous, you may try that.

Note that `pip install -e .` and `pip install -e '.[dev]'` are
discouranged by association (since that is just a wrapper around
`setup.py develop`.)  Similarly to `setup.py develop` they might work,
but you have to be careful with interpreting the results.  If those
don't work, it's on you.


### Testing

The project doesn't include testing data.  It was packaged into a Docker
image which can be found at `crabbone/resurfemg-poly5-test-files:latest`.

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
