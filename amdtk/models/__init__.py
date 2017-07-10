
"""Acoustic Model Discovery Toolkit (AMDTK) module.
Set of tools to do Bayesian clustering of raw acoustic
features to automatically discover phone-like units.

"""

from .mixture import Mixture
from .phone_loop import PhoneLoop
from .vae import MLPClassifier, SVAE

