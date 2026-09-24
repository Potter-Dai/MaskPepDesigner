"""Legacy import; implementation lives in MPDesigner.diffusion.transition."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.diffusion.transition")
