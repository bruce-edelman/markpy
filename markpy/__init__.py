from markpy import sampler
from .sampler import *
from .convergence_tests import *

__author__ = 'Bruce Edelman <bruce.edelman@ligo.org>'

__version__ = '0.1.0'

__likes__ = {cls.name: cls for cls in (
    sampler.Liklie_Base, sampler.Liklie_Norm)
}
