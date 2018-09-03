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

__author__ = 'Bruce Edelman <bruce.edelman@ligo.org>'

__version__ = '0.1.0'

__likes__ = {cls.name: cls for cls in (
    sampler.LiklieBase, sampler.LiklieNorm)
}
