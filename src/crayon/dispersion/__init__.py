"""
Models for dispersion tensor including methods for calculating eigenvalues and
polarisations of wave modes and their derivatives.
"""

__all__ = [
    "BesselIExp",
    "ColdDispersion",
    "DispersionModel",
    "DispersionType",
    "FullyRelativisticDispersion",
    "NonRelativisticDispersion",
    "PlasmaZ",
    "SusceptibilityCache",
    "calculate_harmonic_range",
    "determinant",
    "determinant_dx",
    "determinant_dx2",
    "eigenvalue",
    "eigenvalue_dx",
    "eigenvalue_dx2",
    "eigenvector_dx",
    "polarisation",
    "polarisation_phase_convention_factor",
    "vacuum_dispersion_tensor",
    "vacuum_dispersion_tensor_dq",
    "vacuum_dispersion_tensor_dq2",
    "vacuum_stix_polarisation",
]

from crayon.dispersion.base import (
    DispersionModel,
    DispersionType,
    SusceptibilityCache,
    calculate_harmonic_range,
    determinant,
    determinant_dx,
    determinant_dx2,
    eigenvalue,
    eigenvalue_dx,
    eigenvalue_dx2,
    eigenvector_dx,
    polarisation,
    polarisation_phase_convention_factor,
    vacuum_dispersion_tensor,
    vacuum_dispersion_tensor_dq,
    vacuum_dispersion_tensor_dq2,
    vacuum_stix_polarisation,
)
from crayon.dispersion.cold import ColdDispersion
from crayon.dispersion.fully_relativistic import FullyRelativisticDispersion
from crayon.dispersion.non_relativistic import (
    BesselIExp,
    NonRelativisticDispersion,
    PlasmaZ,
)
