"""Legacy import; implementation lives in MPDesigner.inference.utils_cutpock."""
from importlib import import_module
import sys
sys.modules[__name__] = import_module("MPDesigner.inference.utils_cutpock")
