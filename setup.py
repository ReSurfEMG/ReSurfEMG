#!/usr/bin/env python

import os
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
project_url = 'https://github.com/ReSurfEMG/ReSurfEMG'
project_description = 'A package for analysis of respiratory EMG data'
project_license = 'Apache v2'
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

    def sources(self):
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


class GenerateCondaYaml(Command):

    description = 'generate metadata for conda package'

    user_options = [(
        'target-python=',
        't',
        'Python version to build the package for',
    )]

    user_options = [(
        'target-conda=',
        'c',
        'Conda version to build the package for',
    )]

    def meta_yaml(self):
        python = 'python=='.format(self.target_python)
        conda = 'conda=='.format(self.target_conda)

        return {
            'package': {
                'name': name,
                'version': version,
            },
            'source': {'git_url': '..'},
            'requirements': {
                'host': [python, conda, 'sphinx'],
                'build': ['setuptools'],
                'run': [python, conda] + self.distribution.install_requires,
            },
            'test': {
                'requires': [python, conda],
                'imports': [name],
            },
            'about': {
                'home': project_url,
                'license': project_license,
                'summary': project_description,
            },
        }

    def initialize_options(self):
        self.target_python = None
        self.target_conda = None

    def finalize_options(self):
        if self.target_python is None:
            self.target_python = '.'.join(sys.version_info[:2])
        if self.target_conda is None:
            conda_exe = os.environ.get('CONDA_EXE', 'conda')
            self.target_conda = subprocess.check_output(
                [conda_exe, '--version'],
            ).split()[-1].decode()

    def run(self):
        meta_yaml_path = os.path.join(project_dir, 'conda-pkg', 'meta.yaml')
        with open(meta_yaml_path, 'w') as f:
            json.dump(self.meta_yaml(), f)


class AnacondaUpload(Command):

    description = 'upload packages for Anaconda'

    user_options = [
        ('token=', 't', 'Anaconda token'),
        ('package=', 'p', 'Package to upload'),
    ]

    def initialize_options(self):
        self.token = None
        self.package = None

    def finalize_options(self):
        if (self.token is None) or (self.package is None):
            sys.stderr.write('Token and package are required\n')
            raise SystemExit(2)

    def run(self):
        env = dict(os.environ)
        env['ANACONDA_API_TOKEN'] = self.token
        upload = glob(self.package)[0]
        sys.stderr.write('Uploading: {}\n'.format(upload))
        args = ['upload', '--force', '--label', 'main', upload]
        try:
            proc = subprocess.Popen(
                ['anaconda'] + args,
                env=env,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            for elt in os.environ.get('PATH', '').split(os.pathsep):
                found = False
                sys.stderr.write('Searching for anaconda: {!r}\n'.format(elt))
                base = os.path.basename(elt)
                if base == 'condabin':
                    # My guess is conda is adding path to shell
                    # profile with backslashes.  Wouldn't be the first
                    # time they do something like this...
                    sub = os.path.join(os.path.dirname(elt), 'conda', 'bin')
                    sys.stderr.write(
                        'Anacondas hiding place: {}\n'.format(sub),
                    )
                    sys.stderr.write(
                        '    {}: {}\n'.format(elt, os.path.isdir(elt)),
                    )
                    sys.stderr.write(
                        '    {}: {}\n'.format(sub, os.path.isdir(sub)),
                    )
                    if os.path.isdir(sub):
                        elt = sub
                    executable = os.path.join(elt, 'anaconda')
                    exists = os.path.isfile(executable)
                    sys.stderr.write(
                        '    {}: {}\n'.format(executable, exists),
                    )
                    sys.stderr.write('    Possible matches:\n')
                    for g in glob(os.path.join(elt, '*anaconda*')):
                        sys.stderr.write('        {}\n'.format(g))
                elif base == 'miniconda':
                    # Another thing that might happen is that whoever
                    # configured our environment forgot to add
                    # miniconda/bin messed up the directory name somehow
                    minibin = os.path.join(elt, 'bin')
                    if os.path.isdir(minibin):
                        sys.stderr.write(
                            'Maybe anaconda is here:{}\n'.format(minibin),
                        )
                        elt = minibin
                for p in glob(os.path.join(elt, 'anaconda')):
                    sys.stderr.write('Found anaconda: {}'.format(p))
                    anaconda = p
                    found = True
                    break
                if found:
                    proc = subprocess.Popen(
                        [anaconda] + args,
                        env=env,
                        stderr=subprocess.PIPE,
                    )
                    break
            else:
                import traceback
                traceback.print_exc()
                raise

        _, err = proc.communicate()
        if proc.returncode:
            sys.stderr.write('Upload to Anaconda failed\n')
            sys.stderr.write('Stderr:\n')
            for line in err.decode().split('\n'):
                sys.stderr.write(line)
                sys.stderr.write('\n')
            raise SystemExit(1)


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
        url=project_url,
        license=project_license,
        license_files=('LICENSE.md',),
        description=project_description,
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
