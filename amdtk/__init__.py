
"""Acoustic Model Discovery Toolkit (AMDTK) module.
Set of tools to do Bayesian clustering of raw acoustic
features to automatically discover phone-like units.

"""

# Configure the logging system.
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s %(asctime)s [%(module)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

from .io import *
from .densities import *
from .models import *
from .interface import *

