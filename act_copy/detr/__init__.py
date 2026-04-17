"""DETR subpackage used by act_copy.

This file makes `act_copy.detr` a regular Python subpackage so that
imports like `act_copy.detr.util.misc` work reliably when act_copy is
used from different entrypoints (training and IsaacLab playback).
"""

__all__: list[str] = []
