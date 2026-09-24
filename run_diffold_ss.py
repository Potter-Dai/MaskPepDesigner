"""Legacy entry point; use run_MPDesigner_ss.py."""
import runpy
from pathlib import Path
from run_MPDesigner_ss import *

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("run_MPDesigner_ss.py")), run_name="__main__")
