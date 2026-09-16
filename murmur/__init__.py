"""Murmur: computational ethology for ant-mimicking jumping spiders.

Core pipeline:
    video -> detection -> short-term tracking -> movement episodes
         -> trajectory kinematics -> movement representation (embedding)
         -> motifs -> behavioral space -> comparative ethology.

Every derived datum carries provenance back to source video and frames.
"""

__version__ = "0.1.0"

from .schema import Episode, TrajectoryQC, Provenance  # noqa: F401
