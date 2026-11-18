"""Protennoia Protocol v0.1 -- an operational, substrate-agnostic measurement
protocol for AI cognitive candidacy.

numpy + standard library only. Deterministic. No network, no accelerator.
"""
from .protocol import PROTOCOL_VERSION, IMPLEMENTATION_VERSION, evaluate_system, run_matrix
from .systems import system_names

__all__ = ["PROTOCOL_VERSION", "IMPLEMENTATION_VERSION", "evaluate_system",
           "run_matrix", "system_names"]
__version__ = IMPLEMENTATION_VERSION