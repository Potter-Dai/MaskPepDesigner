"""Legacy import; implementation lives in MPDesigner.diffusion.cos_angle."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.diffusion.cos_angle")
