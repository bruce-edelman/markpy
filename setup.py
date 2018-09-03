#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2018  Bruce Edelman
#
# This file is part of markPy
#
# markPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# markPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with markPy.  If not, see <http://www.gnu.org/licenses/>.

from setuptools import setup, find_packages
from os import path
from io import open
import re

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

version_re = re.compile("__version__ = \"(.*?)\"")
with open(path.join(path.dirname(path.abspath(__file__)), "markpy", "__init__.py")) as inp:
    r = inp.read()
version = version_re.findall(r)[0]


# -- dependencies -------------------------------------------------------------
setup(

    name='markpy',  # Required
    version=version,  # Required
    description='Python MCMC library',  # Required
    url='https://git.ligo.org/bruce.edelman/markpy',  # Optional
    author='Bruce Edelman',  # Optional
    author_email='bedelman@uoregon.edu',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=['numpy', 'scipy', 'matplotlib', 'corner'],  # Optional

)
