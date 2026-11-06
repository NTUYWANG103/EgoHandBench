"""Run the native run_opt.py entry point in this interpreter."""

from pathlib import Path
import os
import runpy
import sys


class NativeStageRuntime:

    def __init__(self, source):
        self.source = Path(source)

    def run(self, command):
        previous = sys.argv
        previous_directory = Path.cwd()
        try:
            os.chdir(self.source / "dyn-hamr")
            sys.argv = command[1:]
            runpy.run_module("baselines.dyn_hamr.dyn-hamr.run_opt", run_name="__main__")
        finally:
            sys.argv = previous
            os.chdir(previous_directory)
