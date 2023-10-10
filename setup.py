#!/usr/bin/env python

import importlib
import os
import subprocess
import sys
from glob import glob
from contextlib import contextmanager
from urllib import parse as urlparse
from urllib import request as urlrequest

from setuptools import Command, setup
from setuptools.command.bdist_egg import bdist_egg as BDistEgg
from setuptools.command.install import install as InstallCommand


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


def find_conda():
    conda_exe = os.environ.get('CONDA_EXE', 'conda')
    return subprocess.check_output(
        [conda_exe, '--version'],
    ).split()[-1].decode()


def run_and_log(cmd, **kwargs):
    sys.stderr.write('> {}\n'.format(' '.join(cmd)))
    return subprocess.call(cmd, **kwargs)


def translate_reqs(packages, double_equals=False):
    re = importlib.import_module('re')
    tr = {
        'codestyle': 'pycodestyle',
        # Apparently, there isn't mne-base on PyPI...
        'mne': 'mne-base',
    }
    result = []

    for p in packages:
        p = re.sub(r'\s+', '', p)
        if not double_equals:
            p = re.sub('=+', '=', p)
        parts = re.split(r'[ <>=]', p, maxsplit=1)
        name = parts[0]
        version = p[len(name):]
        if name in tr:
            result.append(tr[name] + version)
        else:
            result.append(p)

    return result


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
        tf = importlib.import_module('tempfile')
        site = importlib.import_module('site')
        st = importlib.import_module('setuptools.dist')
        venv = importlib.import_module('venv')
        ei = importlib.import_module('setuptools.command.easy_install')
        EZInstallCommand = ei.easy_install

        class ContextVenvBuilder(venv.EnvBuilder):

            def ensure_directories(self, env_dir):
                self.context = super().ensure_directories(env_dir)
                return self.context

        recs = self.distribution.tests_require

        with tf.TemporaryDirectory() as builddir:
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

            test_dist = st.Distribution()
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
        unittest = importlib.import_module('unittest')
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


class SphinxDoc(Command):

    description = 'generate documentation'

    user_options = [('wall', 'W', ('Warnings are errors'))]

    def initialize_options(self):
        self.wall = True

    def finalize_options(self):
        pass

    def run(self):
        from sphinx.util.console import nocolor
        from sphinx.util.docutils import docutils_namespace, patch_docutils
        from sphinx.application import Sphinx
        from sphinx.cmd.build import handle_exception

        nocolor()
        confoverrides = {}
        confoverrides['project'] = name
        confoverrides['version'] = version
        confdir = os.path.join(project_dir, 'docs')
        srcdir = confdir
        builder = 'html'
        build = self.get_finalized_command('build')
        build_dir = os.path.join(os.path.abspath(build.build_base), 'sphinx')
        builder_target_dir = os.path.join(build_dir, builder)
        app = None

        try:
            with patch_docutils(confdir), docutils_namespace():
                app = Sphinx(
                    srcdir,
                    confdir,
                    builder_target_dir,
                    os.path.join(build_dir, 'doctrees'),
                    builder,
                    confoverrides,
                    sys.stdout,
                    freshenv=False,
                    warningiserror=self.wall,
                    verbosity=self.distribution.verbose - 1,
                    keep_going=False,
                )
                app.build(force_all=False)
                if app.statuscode:
                    sys.stderr.write(
                        'Sphinx builder {} failed.'.format(app.builder.name),
                    )
                    raise SystemExit(8)
        except Exception as e:
            handle_exception(app, self, e, sys.stderr)
            raise


class InstallDev(InstallCommand):

    def local_repo(self):
        return urlparse.urljoin('file:', urlrequest.pathname2url(
            os.path.join(project_dir, 'dist')
        ))

    def run(self):
        if os.environ.get('CONDA_DEFAULT_ENV'):
            bdist_conda = BdistConda(self.distribution)
            bdist_conda.run()
            cmd = [
                'conda',
                'install',
                '--strict-channel-priority',
                '--override-channels',
                '-c', 'conda-forge',
                '-c', self.local_repo(),
                '--update-deps',
                '--force-reinstall',
                '-y',
                name,
                'python=={}'.format('.'.join(map(str, sys.version_info[:2]))),
                'conda=={}'.format(find_conda()),
            ] + translate_reqs(self.distribution.extras_require['dev'], True)
            if run_and_log(cmd):
                sys.stderr.write('Couldn\'t install {} package\n'.format(name))
                raise SystemExit(6)
        else:
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
        python = 'python=={}'.format(self.target_python)
        conda = 'conda=={}'.format(self.target_conda)

        return {
            'package': {
                'name': name,
                'version': version,
            },
            'source': {'path': '..'},
            'requirements': {
                'host': [python, conda, 'sphinx'],
                'build': ['setuptools'],
                'run': [python] + translate_reqs(
                    self.distribution.install_requires,
                    True,
                )
            },
            'test': {
                'requires': [python],
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
            self.target_python = '.'.join(map(str, sys.version_info[:2]))
        if self.target_conda is None:
            self.target_conda = find_conda()

    def run(self):
        json = importlib.import_module('json')

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
        if run_and_log(['anaconda'] + args, env=env):
            sys.stderr.write('Upload to Anaconda failed\n')
            raise SystemExit(7)


class SdistConda(Command):

    description = 'Helper for conda-build to make it work on Windows'

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self, install_dir=None, record='record.txt'):
        sysconfig = importlib.import_module('sysconfig')
        ei = importlib.import_module('setuptools.command.easy_install')
        EZInstallCommand = ei.easy_install

        if install_dir is None:
            install_dir = sysconfig.get_path('platlib')

        self.distribution.install_requires = translate_reqs(
            self.distribution.install_requires,
            True,
        )
        self.distribution.tests_require = translate_reqs(
            self.distribution.tests_require,
            True,
        )
        bdist_egg = BDistEgg(self.distribution)
        bdist_egg.initialize_options()
        bdist_egg.finalize_options()
        bdist_egg.run()
        egg = glob(os.path.join(project_dir, 'dist/*.egg'))[0]
        sys.stderr.write('Finished building {}'.format(egg))

        ezcmd = EZInstallCommand(self.distribution)
        ezcmd.initialize_options()
        ezcmd.no_deps = True
        ezcmd.record = record
        ezcmd.args = [egg]
        ezcmd.install_dir = install_dir
        ezcmd.install_base = install_dir
        ezcmd.install_purelib = install_dir
        ezcmd.install_platlib = install_dir
        ezcmd.finalize_options()
        ezcmd.run()


class PopenWrapper:

    distribution = None

    def __init__(self, *args, **kwargs):
        self.elapsed = 0
        self.disk = 0
        self.processes = 0
        self.cpu_user = 0
        self.cpu_sys = 0
        self.rss = 0
        self.vms = 0

        if args[0][-1].endswith('build.sh'):
            self.run(*args, **kwargs)
        else:
            self.run_in_subprocess(*args, **kwargs)

    def run(self, *args, **kwargs):
        io = importlib.import_module('io')
        sc = SdistConda(self.distribution)
        sc.run(
            kwargs['env']['SP_DIR'],
            os.path.join(kwargs['env']['SRC_DIR'], 'record.txt'),
        )
        self.returncode = 0
        self.err = self.out = io.StringIO()

    def run_in_subprocess(self, *args, **kwargs):
        proc = subprocess.Popen(*args, **kwargs)
        while proc.returncode is None:
            proc.poll()
        self.returncode = proc.returncode
        self.out = proc.stdout
        self.err = proc.stderr


class BdistConda(BDistEgg):

    description = 'Helper for conda-build to make it work on Windows'

    user_options = [
        (
            'optimize-low-memory',
            'o',
            'Optimize for low memory environment (Github Actions CI)',
        ),
    ]
    boolean_options = [
        'optimize-low-memory',
    ]

    def initialize_options(self):
        self.optimize_low_memory = False

    def finalize_options(self):
        pass

    def patch_conda_build(self):
        conda_index = importlib.import_module('conda_build.index')
        conda_utils = importlib.import_module('conda_build.utils')
        ds = importlib.import_module('distutils.spawn')

        conda_index.update_index.__defaults__ = (
            False,
            None,
            None,
            1,                  # This is the number of threads
            False,
            False,
            None,
            None,
            True,
            None,
            False,
            None,
        )
        conda_index.ChannelIndex.__init__.__defaults__ = None, 1, False, False
        conda_utils.PopenWrapper = PopenWrapper
        PopenWrapper.distribution = self.distribution
        ds.spawn = lambda *args: None

    def run(self):
        shutil = importlib.import_module('shutil')

        frozen = '.'.join(map(str, sys.version_info[:2]))
        conda = find_conda()
        cmd = [
            'conda',
            'install', '-y',
            '--strict-channel-priority',
            '--override-channels',
            '-c', 'conda-forge',
            '-c', 'anaconda',
            'conda-build',
            'conda-verify',
            'anaconda-client',
            'python=={}'.format(frozen),
        ]
        if run_and_log(cmd):
            sys.stderr.write('Failed to install conda-build\n')
            raise SystemExit(3)
        shutil.rmtree(
            os.path.join(project_dir, 'dist'),
            ignore_errors=True,
        )
        shutil.rmtree(
            os.path.join(project_dir, 'build'),
            ignore_errors=True,
        )

        conda_build = importlib.import_module('conda_build.cli.main_build')

        cmd = [
            '--no-anaconda-upload',
            '--override-channels',
            '--output-folder', os.path.join(project_dir, 'dist'),
            '-c', 'conda-forge',
            '--no-locking',
            os.path.join(project_dir, 'conda-pkg'),
        ]

        if self.optimize_low_memory:
            cmd.insert(0, '--no-test')
            self.patch_conda_build()

        rc = conda_build.execute(cmd)
        sys.stderr.write('Built package: {}'.format(rc[0]))


if __name__ == '__main__':
    setup(
        name=name,
        version=version,
        author='A team including the NLeSC and the U. of Twente',
        author_email='c.moore@esciencecenter.nl',
        packages=['resurfemg'],
        url=project_url,
        license=project_license,
        license_files=('LICENSE.md',),
        description=project_description,
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        package_data={'': ('README.md',)},
        cmdclass={
            'test': UnitTest,
            'lint': Pep8,
            'isort': Isort,
            'apidoc': SphinxApiDoc,
            'install_dev': InstallDev,
            'anaconda_upload': AnacondaUpload,
            'anaconda_gen_meta': GenerateCondaYaml,
            'bdist_conda': BdistConda,
            'sdist_conda': SdistConda,
            'build_sphinx': SphinxDoc,
        },
        test_suite='setup.my_test_suite',
        install_requires=[
            'pyxdf',
            'mne==0.23.4',
            'textdistance',
            'pandas',
            'scipy',
            'matplotlib',
            'h5py',
            'scikit-learn==1.1.1',
        ],
        tests_require=['pytest', 'pycodestyle', 'isort', 'wheel'],
        setup_requires=['wheel'],
        extras_require={
            'dev': [
                'pytest',
                'codestyle',
                'isort',
                'wheel',
                'jupyter',
                'ipympl',
            ],
        },
        zip_safe=False,
    )
