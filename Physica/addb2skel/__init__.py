"""
addb2skel: AddBiomechanics to SKEL Conversion Pipeline

A production-quality pipeline for converting AddBiomechanics (AddB) joint
trajectories to SKEL joint trajectories with robust handling of shoulder/scapula
differences and subject-specific proportion offsets.

References:
- HSMR SKELify: https://github.com/IsshikiHugh/HSMR
- AddBiomechanics: https://nimblephysics.org/docs/working-with-addbiomechanics-data.html
"""

__version__ = "0.1.0"
__author__ = "Kwonjoon Lee"

from .pipeline import convert_addb_to_skel
from .joint_definitions import ADDB_JOINTS, SKEL_JOINTS, ADDB_TO_SKEL_MAPPING

__all__ = [
    "convert_addb_to_skel",
    "ADDB_JOINTS",
    "SKEL_JOINTS",
    "ADDB_TO_SKEL_MAPPING",
]
