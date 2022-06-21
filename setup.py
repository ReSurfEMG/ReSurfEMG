#!/usr/bin/env python

import os
import shlex
import site
import subprocess
import sys
from distutils.dir_util import copy_tree
from glob import glob
from tempfile import TemporaryDirectory
from contextlib import contextmanager

from setuptools import Command, setup
from setuptools.command.easy_install import easy_install as EZInstallCommand
from setuptools.command.bdist_egg import bdist_egg as BDistEgg
from setuptools.command.install import install as InstallCommand
from setuptools.dist import Distribution
import unittest
import venv


project_dir = os.path.dirname(os.path.realpath(__file__))
name = 'resurfemg'
try:
    tag = subprocess.check_output(
        [
            'git',
            '--no-pager',
            'describe',
            '--abbrev=0',
            '--tags',
        ],
        stderr=subprocess.DEVNULL,
    ).strip().decode()
except subprocess.CalledProcessError as e:
    tag = 'v0.0.0'

version = tag[1:]

with open(os.path.join(project_dir, 'README.md'), 'r') as f:
    readme = f.read()


class ContextVenvBuilder(venv.EnvBuilder):

    def ensure_directories(self, env_dir):
        self.context = super().ensure_directories(env_dir)
        return self.context


class TestCommand(Command):

    user_options = [
        ('fast', 'f', (
            'Don\'t install dependencies, test in the current environment'
        )),
    ]

    def initialize_options(self):
        self.fast = False

    def finalize_options(self):
        self.test_args = []
        self.test_suite = True

    def sources(sefl):
        return glob(
            os.path.join(project_dir, 'resurfemg', '**/*.py'),
            recursive=True,
        ) + [os.path.join(project_dir, 'setup.py')]

    @contextmanager
    def prepare(self):
        recs = self.distribution.tests_require

        with TemporaryDirectory() as builddir:
            vbuilder = ContextVenvBuilder(with_pip=True)
            vbuilder.create(os.path.join(builddir, '.venv'))
            env_python = vbuilder.context.env_exe
            platlib = subprocess.check_output(
                (env_python,
                 '-c',
                 'import sysconfig;print(sysconfig.get_path("platlib"))'
                 ),
            ).strip().decode()

            egg = BDistEgg(self.distribution)
            egg.initialize_options()
            egg.dist_dir = builddir
            egg.keep_temp = False
            egg.finalize_options()
            egg.run()

            test_dist = Distribution()
            test_dist.install_requires = recs
            ezcmd = EZInstallCommand(test_dist)
            ezcmd.initialize_options()
            ezcmd.args = recs
            ezcmd.always_copy = True
            ezcmd.install_dir = platlib
            ezcmd.install_base = platlib
            ezcmd.install_purelib = platlib
            ezcmd.install_platlib = platlib
            sys.path.insert(0, platlib)
            os.environ['PYTHONPATH'] = platlib
            ezcmd.finalize_options()

            ezcmd.easy_install(glob(os.path.join(builddir, '*.egg'))[0])

            ezcmd.run()
            site.main()

            yield env_python

    def run(self):
        if not self.fast:
            with self.prepare() as env_python:
                self.run_tests(env_python)
        self.run_tests()


class UnitTest(TestCommand):

    description = 'run unit tests'

    def run_tests(self, env_python=None):
        if env_python is None:
            loader = unittest.TestLoader()
            suite = loader.discover('tests', pattern='test.py')
            runner = unittest.TextTestRunner()
            result = runner.run(suite)
            sys.exit(1 if result.errors else 0)

        tests = os.path.join(project_dir, 'tests', 'test.py')
        sys.exit(subprocess.call((env_python, '-m', 'unittest', tests)))


class Pep8(TestCommand):

    description = 'validate sources against PEP8'

    def run_tests(self, env_python=None):
        if env_python is None:
            from pycodestyle import StyleGuide

            style_guide = StyleGuide(paths=self.sources())
            options = style_guide.options

            report = style_guide.check_files()
            report.print_statistics()

            if report.total_errors:
                if options.count:
                    sys.stderr.write(str(report.total_errors) + '\n')
                sys.exit(1)
            sys.exit(0)

        sys.exit(
            subprocess.call(
                [env_python, '-m', 'pycodestyle'] + self.sources(),
            ))


class Isort(TestCommand):

    description = 'validate imports'

    def run_tests(self, env_python=None):
        options = ['-c', '--lai', '2', '-m' '3']

        if env_python is None:
            from isort.main import main as imain

            if imain(options + self.sources()):
                sys.exit(1)
            sys.exit(0)

        sys.exit(
            subprocess.call(
                [env_python, '-m', 'isort'] + options + self.sources(),
            ))


class SphinxApiDoc(Command):

    description = 'run apidoc to generate documentation'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.ext.apidoc import main

        src = os.path.join(project_dir, 'docs')
        special = (
            'index.rst',
            'developers.rst',
            'medical-professionals.rst',
        )

        for f in glob(os.path.join(src, '*.rst')):
            for end in special:
                if f.endswith(end):
                    os.utime(f, None)
                    break
            else:
                os.unlink(f)

        sys.exit(main([
            '-o', src,
            '-f',
            os.path.join(project_dir, 'resurfemg'),
            '--separate',
        ]))


class InstallDev(InstallCommand):

    def run(self):
        self.distribution.install_requires.extend(
            self.distribution.extras_require['dev'],
        )
        super().do_egg_install()


if __name__ == '__main__':
    setup(
        name=name,
        version=version,
        author='A team including the NLeSC and the U. of Twente',
        author_email='c.moore@esciencecenter.nl',
        package_dir={'.': '', 'TMSiSDK': '.vendor/tmsisdk/TMSiSDK'},
        packages=[
            'resurfemg',
            'TMSiSDK',
            'TMSiSDK.devices',
            'TMSiSDK.devices.saga',
            'TMSiSDK.file_formats',
            'TMSiSDK.file_readers',
            'TMSiSDK.filters',
            'TMSiSDK.plotters',
        ],
        url='https://github.com/ReSurfEMG/ReSurfEMG',
        license='LICENSE.md',
        description='A package for analysis of respiratory EMG data',
        long_description=open('README.md').read(),
        package_data={'': ('README.md',)},
        cmdclass={
            'test': UnitTest,
            'lint': Pep8,
            'isort': Isort,
            'apidoc': SphinxApiDoc,
            'install_dev': InstallDev,
        },
        test_suite='setup.my_test_suite',
        install_requires=[
            'pandas',
            'scipy',
            'matplotlib',
            'h5py',
            'sklearn',
            'pyxdf',
            'mne',
            'textdistance',
        ],
        tests_require=['pytest', 'pycodestyle', 'isort', 'wheel'],
        command_options={
            'build_sphinx': {
                'project': ('setup.py', name),
                'version': ('setup.py', version),
                'source_dir': ('setup.py', './docs'),
                'config_dir': ('setup.py', './docs'),
            },
        },
        setup_requires=['sphinx', 'wheel'],
        extras_require={
            'dev': ['pytest', 'codestyle', 'isort', 'wheel'],
        },
        zip_safe=False,
    )
