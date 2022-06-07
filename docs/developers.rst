=================
Developer's Guide
=================

Sending Your Work
=================

We accept pull requests made through GitHub.  Alternatively, you can
send patch files in an email (please find developers' emails on their
profile pages).  As is usual, we request that the changes be rebased
on the branch they are to be integrated into.  We also request that
you pre-lint and test anything you send

We'll try our best to attribute
your work to you, however, you need to release your work under
compatible license for us to be able to use it.

.. warning::

   We don't use `git-merge` command, and if your submission has merge
   commits, we'll have to remove them.  This means that in such case
   commit hashes will be different from those in your original
   submission.

Setting Up Development Environment
==================================


The Way We Do It
^^^^^^^^^^^^^^^^

If you want to develop using Anaconda Python, you would:

Follow the instructions on the readme

Similar to above, this will create a virtual environment, build
`conda` package, install it and then add development dependencies to
what was installed. 

The Traditional Ways
^^^^^^^^^^^^^^^^^^^^

Regradless of the downsides of this approach, we try to support more
common ways to work with Python projects.  It's a common practice to
"install" a project during development by either using `pip install
--editable` command, or by using `conda` environment files.

We provide limited support for approaches not based on Anaconda right now.  For instance, if you
want to work on the project using `pip`, you could try it, and contact us:

.

The environment files are generated using:


.. code-block:: bash

   conda env create -f ./environment.yml



Rationale
^^^^^^^^^

There are several problems with traditional way Python programmers are
taught to organize their development environment.  The way a typical
Python project is developed, it is designed to support a single
version of Python, rarely multiple Python distributions or operating
systems. We are working to support multiple Pythons. Pending.



Testing
=======

You may run:

.. code-block:: bash

  python ./tests/test.py 

Under the hood, this runs unittest:



Style Guide for Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have yet to install a linter, but we will soon change it so you can run one from the setup we need to write

.. code-block:: bash

   python ./setup.py lint



Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

This project has no extensive CI setup that uses GitHub Actions platform.
This is a templatebut it's far from
being ready yet, and we currently don't work on it.

.. warning::

   

.. _GitHub repo: https://github.com/..
.. _GitHub Actions dashboard: https://github.com/...