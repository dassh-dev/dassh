########################################################################
# Copyright 2021, UChicago Argonne, LLC
#
# Licensed under the BSD-3 License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a
# copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
########################################################################
"""
date: 2021-11-12
author: Milos Atz
"""
########################################################################
from setuptools import setup
from setuptools._vendor.packaging import version


DESCRIPTION = "DASSH: Ducted Assembly Steady-State Heat Transfer software"
LONG_DESCRIPTION = """DASSH performs steady-state thermal fluids
calculations for hexagonal, ducted fuel assemblies typical of liquid
metal fast reactors."""
DISTNAME = 'DASSH'
MAINTAINER = 'Milos Atz'
MAINTAINER_EMAIL = 'matz@anl.gov'
URL = 'https://github.com/dassh-dev/dassh'
LICENSE = 'https://opensource.org/licenses/BSD-3-Clause'
DOWNLOAD_URL = ''
# FROM OPENMC: Get version information from __init__.py. This is ugly,
# but more reliable than using an import.
with open('dassh/__init__.py', 'r') as f:
    VERSION = f.readlines()[-1].split()[-1].strip("'")


###############################################################################


def check_dependencies():
    install_requires = []
    # Just make sure dependencies exist, I haven't rigorously
    # tested what are the minimal versions that will work
    try:
        import numpy
        assert version.parse(numpy.__version__) >= version.parse('1.18')
    except (ImportError, AssertionError):
        install_requires.append('numpy>=1.18')
    try:
        import matplotlib
        assert version.parse(matplotlib.__version__) >= version.parse('3.2')
    except (ImportError, AssertionError):
        install_requires.append('matplotlib>=3.2')
    try:
        import configobj
    except ImportError:
        install_requires.append('configobj')
    try:
        import pandas
    except ImportError:
        install_requires.append('pandas')
    try:
        import pytest
    except ImportError:
        install_requires.append('pytest')

    return install_requires


###############################################################################


if __name__ == "__main__":
    install_requires = check_dependencies()
    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=LICENSE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          python_requires='>=3.5',  # FULL FUNCTIONALITY REQUIRES 3.7+!!
          install_requires=install_requires,
          packages=['dassh'],
          package_dir={'dassh': 'dassh'},
          package_data={'dassh': ['./*.txt',
                                  './correlations/*.py',
                                  './_se2anl/*.py',
                                  './py4c/*.py',
                                  './varpow_osx.x',
                                  './varpow_linux.x',
                                  './data/*.csv',
                                  'tests/*.py']},
          include_package_data=True,
          classifiers=['Intended Audience :: Science/Research',
                       # 'Programming Language :: Python :: 3.5',
                       # 'Programming Language :: Python :: 3.6',
                       'Programming Language :: Python :: 3.7+',
                       # 'License :: OSI Approved :: MIT License',
                       'Topic :: Scientific/Engineering :: Nuclear Energy',
                       'Operating System :: Unix',
                       'Operating System :: MacOS'],
          entry_points={'console_scripts': [
              'dassh = dassh.__main__:main',
              'dassh_plot = dassh.__main__:plot']}
          )


###############################################################################
