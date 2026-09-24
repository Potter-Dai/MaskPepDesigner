"""Legacy import; implementation lives in MPDesigner.diffusion.scipy_so3."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.diffusion.scipy_so3")
