# Copyright (C) 2018  Bruce Edelman
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""
markPy is a python package developed by Bruce Edelman to implement MCMC sampling among other things
"""

from .sampler import *
from .convergence_tests import *
from .models import *
from .corner_plot import *
from .steppers import *
import numpy as np

__author__ = 'Bruce Edelman <bruce.edelman@ligo.org>'

__version__ = '0.1.0'

__models__ = {cls.name: cls for cls in (
    models.NormModelInfer, models.RosenbrockAnalytic, models.BaseModel,
    models.EggBoxAnalytic, models.NormModelAnalytic, models.BaseInferModel)
}

__chains__ = {cls.name: cls for cls in (
    sampler.MarkChain, sampler.ParallelMarkChain)
}

__steppers__ = {cls.name: cls for cls in (
    steppers.BaseStepper, steppers.BaseStepper
)}

