=================
Developer's Guide
=================

Contributing
============

We accept pull requests made through GitHub. As is usual,
we request that the changes be rebased
on the branch they are to be integrated into.  We also request that
you pre-lint and test anything you send.

We'll try our best to attribute
your work to you, however, you need to release your work under
compatible license for us to be able to use it.

.. warning::

   We don't use `git-merge` command, and if your submission has merge
   commits, we'll have to remove them.  This means that in such case
   commit hashes will be different from those in your original
   submission.


Setting up development environment
==================================


The way we do it
^^^^^^^^^^^^^^^^

If you want to develop using Anaconda Python, you would:

Follow the instructions on the readme.

This will create a virtual environment, build `conda` package, install
it and then add development dependencies to what was installed.



The traditional ways
^^^^^^^^^^^^^^^^^^^^

Regardless of the downsides of this approach, we try to support more
common ways to work with Python projects.  It's a common practice to
"install" a project during development by either using `pip install
--editable` command, or by using `conda` environment files.

We provide limited support for approaches not based on Anaconda right
now.  For instance, if you want to work on the project using `pip`,
you could try it, and contact us: resurfemg@gmail.com

.

The environment files are generated using:


.. code-block:: bash

   conda env create -f ./environment.yml



Rationale
^^^^^^^^^

There are several problems with the traditional way Python programmers are
taught to organize their development environment.  The way a typical
Python project is developed, it is designed to support a single
version of Python, rarely multiple Python distributions or operating
systems. We are working to support multiple Pythons. Pending. But for
now we are doing what is simple and fast.

Run requirements
^^^^^^^^^^^^^^^^

The ReSurfEMG package has the following dependencies:

Data handling packages

- pyxdf: XDF-file format importer (tmsidk_lite dependency)
- h5py: H5PY-file format importer


Signal analysis packages

- mne: Exploring, visualizing, and analyzing human neurophysiological data such as MEG, EEG, sEEG, ECoG
- pandas: data analysis and statistics library
- scipy: Advanced signal analysis library
- textdistance: Algorithm for comparing signal/text similarity

Visualization packages

- matplotlib: Plotting library

Machine learning packages

- scikit-learn: Machine learning and data mining library


Development requirements
^^^^^^^^^^^^^^^^^^^^^^^^

The ReSurfEMG package has the following dependencies for developing:

Coding style conventions: 

- codestyle: Code style checker
- isort: Sorting imports

Running Jupyter Notebooks:

- jupyter
- ipympl: matplotlib extension for Jupyter

Testing:

- pytest: Running tests

Releases:

- wheel: Building distributions


Developing
==========


Style Guide for Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have linting!

.. code-block:: bash

   python ./setup.py lint



Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

This project has `GitHub repo`_ that uses `GitHub Actions`_
platform.  


.. _GitHub repo: https://github.com/ReSurfEMG/ReSurfEMG
.. _GitHub Actions: https://github.com/ReSurfEMG/ReSurfEMG/actions


Style
^^^^^

When it comes to style, beyond linting we are trying
to conform, more or less, to the Google Python style
https://google.github.io/styleguide/pyguide.html


Releases
^^^^^^^^

Releases to PyPI and Conda are automatically managed by Github actions when 
versions are tagged in the main ReSurfEMG branch. The required steps:

1. Update the 'changelog.md' and 'CITATION.cff'
2. Tag the main branch with the newest version number:

.. code-block:: bash

  git tag v0.x.x

3. Push the tags to main:

.. code-block:: bash

  git push --tags

4. Check in `Github`_ whether the release was successful:

.. _GitHub: https://github.com/ReSurfEMG/ReSurfEMG/releases


Testing
=======

You can run tests from the setup.py file
i.e. 

.. code-block:: bash

  python setup.py test

