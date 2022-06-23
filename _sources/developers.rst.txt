=================
Developer's Guide
=================

Sending Your Work
=================

We accept pull requests made through GitHub. As is usual,
we request that the changes be rebased
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

Follow the instructions on the readme.

This will create a virtual environment, build `conda` package, install
it and then add development dependencies to what was installed.



The Traditional Ways
^^^^^^^^^^^^^^^^^^^^

Regradless of the downsides of this approach, we try to support more
common ways to work with Python projects.  It's a common practice to
"install" a project during development by either using `pip install
--editable` command, or by using `conda` environment files.

We provide limited support for approaches not based on Anaconda right
now.  For instance, if you want to work on the project using `pip`,
you could try it, and contact us:

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
systems. We are working to support multiple Pythons. Pending. But for
now we are doing what is simple and fast.



Testing
=======

You may run:

.. code-block:: bash

  python ./tests/test.py 

Under the hood, this runs unittest.

Alternatively,
you can run tests from the setup.py file
i.e. 
.. code-block:: bash

   python setup.py test



Style Guide for Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We have have linting!

.. code-block:: bash

   python ./setup.py lint



Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

This project has CI setup that uses GitHub Actions
platform.  


.. _GitHub repo: https://github.com/ReSurfEMG/ReSurfEMG
.. _GitHub Actions dashboard: https://github.com/ReSurfEMG/ReSurfEMG/actions


Style
^^^^^

When it comes to style, beyond linting we are trying
to conform, more or less, to the Google Python style
https://google.github.io/styleguide/pyguide.html
