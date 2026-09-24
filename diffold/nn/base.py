"""Legacy import; implementation lives in MPDesigner.nn.base."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.nn.base")
