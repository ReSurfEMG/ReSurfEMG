 #!/usr/bin/env python


import setuptools

from setuptools import setup

setup(
   name='ReSurfEMG',
   version='0.0.1',
   author='An Awesome Team from the Netherlands eScience Center!',
   author_email='c.moore@esciencecenter.nl',
   packages=['ReSurfEMG', 'ReSurfEMG.test'],
   #scripts=['bin/script1','bin/script2'],
   #url='http://pypi.python.org/pypi/ReSurfEMG/',
   license='LICENSE.md',
   description='A package that helps with analysis of respiratory EMG data',
   long_description=open('README.md').read(),
   install_requires=[
       "matplotlib",
       "numpy",
   ],
)