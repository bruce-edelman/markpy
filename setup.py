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
    install_requires=['numpy', 'scipy', 'matplotlib'],  # Optional

)
