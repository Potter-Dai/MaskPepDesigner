"""Legacy import; implementation lives in MPDesigner.nn.dfv6_attempt."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.nn.dfv6_attempt")
