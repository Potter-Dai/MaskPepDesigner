"""Legacy import; implementation lives in MPDesigner.data.process."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.data.process")
