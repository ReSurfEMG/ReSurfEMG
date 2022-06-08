 #!/usr/bin/env python

## this setup.py is under development. please don't bother taking it seriously, this is template pasting
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


        # packages=['ReSurfEMG',
        # ],
        # cmdclass={
        #     'test': UnitTest,
        #     'lint': Pep8,
        #     'apidoc': SphinxApiDoc,
        #     'genconda': GenerateCondaYaml,
        #     'install': Install,
        #     # TODO(makeda): CI for real 
        #     'install_dev': InstallDev,
        #     'find_egg': FindEgg,
        #     'anaconda_upload': AnacondaUpload,
        #     'anaconda_gen_env': GenCondaEnv,
        # },
        # tests_require=['unittest'],
        # command_options={
        #     'build_sphinx': {
        #         'project': ('setup.py', name),
        #         'version': ('setup.py', version),
        #         'source_dir': ('setup.py', './docs'),
        #         'config_dir': ('setup.py', './docs'),
        #     },
        # },
        # setup_requires=['sphinx'],
        # install_requires=install_requires(versions),
        
        # zip_safe=False,