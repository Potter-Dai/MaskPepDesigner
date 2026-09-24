"""Legacy import; implementation lives in MPDesigner.losses.auxillary."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.losses.auxillary")
