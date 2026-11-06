"""Thin CLI for a single official Dyn-HaMR stage."""

import argparse
from baselines.dyn_hamr.worker import DynStageRunner

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--assets", required=True)
parser.add_argument("--work", required=True)
parser.add_argument("--clip", required=True)
parser.add_argument("--frames", type=int, required=True)
parser.add_argument("--fps", type=float, required=True)
parser.add_argument("--device", default="cuda:0")
DynStageRunner(**vars(parser.parse_args())).run()
