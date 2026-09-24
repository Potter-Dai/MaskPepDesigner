"""Legacy import; implementation lives in MPDesigner.inference.job."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.inference.job")
