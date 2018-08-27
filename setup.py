from setuptools import setup, find_packages
from os import path
from io import open
import versioneer

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.


# -- versioning ---------------------------------------------------------------

cmdclass = versioneer.get_cmdclass()
__version__ = versioneer.get_version()

# -- dependencies -------------------------------------------------------------
setup(

    name='markpy',  # Required
    version=__version__,  # Required
    description='Python MCMC library',  # Required
    url='https://git.ligo.org/bruce.edelman/markpy',  # Optional
    author='Bruce Edelman',  # Optional
    author_email='bedelman@uoregon.edu',  # Optional
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),  # Required
    install_requires=['numpy', 'scipy', 'matplotlib'],  # Optional

)
