"""Bridge: distill a trained Genesis policy into Booster behavioral-rule constants.

This package is the "桥接蒸馏" link chosen for the AMD hackathon <-> Booster T1
project. Genesis trains soccer skills on the AMD Radeon cloud; this bridge records
the exact observables Booster's rules consume, then fits simple heuristics and
emits paste-ready constants for Booster's param.py.

Pure-python (no torch/genesis) so it imports and self-tests on macOS.
"""
