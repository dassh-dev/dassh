########################################################################
# Copyright 2021, UChicago Argonne, LLC
#
# Licensed under the BSD-3 License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a
# copy of the License at
#
#     https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
########################################################################
"""
date: 2023-02-08
author: matz
Bare rod turbulent mixing correlation from Kim and Chung (2001)
"""
########################################################################
import numpy as np
from . import nusselt_db


# Kim-Chung empirical constants
alpha = 0.18
beta = 0.2
gamma = 20
b = 0.666666667
Pr_T = 0.9


def calculate_mixing_params(rr):
    """Calculate turbulent mixing parameters for bare rod bundle based
    on correlation by Kim and Chung (2001)

    Parameters
    ----------
    rr : DASSH RoddedRegion object
        Contains geometry and flow parameters

    Returns
    -------
    tuple
        Tuple of two floats: (1) the diffusivity enhancement due to
        turbulent mixing, and (2) 0.0, because there is no swirl

    Notes
    -----
    The diffusivity enhancement produced by this function needs to be
    equivalent to the eddy diffusivity from the SE2 / CTD correlations.
    The 2016 ANTEO+ paper by Lodi provides the necessary information.

    The eddy diffusivity is used to define a "mixing exchange rate"
    [kg / m / s] in Equation 42 as follows:

        W_ij = density * eddy * (P-D) / eta
            eddy = eddy diffusivity [m / s^2]
            eta = centroid-centroid distance [m]

    The correlations for eddy diffusivity produce a dimensionless
    value which is dimensionalized by multiplying by the centroid-
    centroid distance and the axial velocity.

        eddy = eddy_dimless * velocity * eta

    The "mixing exchange rate" parameter is defined for bare rod
    bundles experiencing turbulent mixing in Equation 36 as:

        W_ij = density * velocity * St_g * (P-D)
            St_g = gap Stanton number, per Equation 37

    Relating these two values and solving for eddy diffusivity:
        density * eddy * (P-D) / eta = density * velocity * St_g * (P-D)
        eddy / eta = velocity * St_g
        eddy = velocity * eta * St_g

    This shows how St_g is actually equivalent to the dimensionless
    eddy diffusivity and can be treated the same way.

    """
    # Try to pull in pre-calculated constants
    try:
        C1, C2, C3, C4 = rr.corr_constants['mix']
    except (KeyError, AttributeError):
        rr.corr_constants['mix'] = calc_constants(rr)
        C1, C2, C3, C4 = rr.corr_constants['mix']

    # Calculate the turbulent mixing parameter based on Pr and Re
    Pr = nusselt_db._calc_prandtl(rr.coolant)
    # Re = rr.coolant_int_params['Re']
    Re = rr.coolant_int_params['Re_sc'][0]
    Stg = _calculate_stg(Pr, Re, C1, C2, C3, C4)
    #
    return Stg * rr.L[0][0], 0.0


def _calculate_stg(Pr, Re, C1, C2, C3, C4):
    """Calculate the gap Stanton number"""
    return (C1 / (Pr * Re**C2) + C3) * Re**C4


def calc_constants(rr, use_simple=True):
    """Calculate and store constants"""
    # Get/calculate geometric parameters
    # DH = rr.bundle_params['de']
    DH = rr.params['de'][0]
    g = rr.pin_pitch - rr.pin_diameter
    eta = rr.L[0][0]
    Lx = b * eta
    Ly = g
    lambd = Ly / Lx
    if use_simple:
        ax = 1 - 2 * lambd ** 2 / np.pi
    else:
        ax = (2 / np.pi) * np.sqrt(1 / (1 - lambd**4))
        ax *= np.arcsin(np.sqrt(1 - lambd**2))
    z_FP_over_D = 2 * b * eta / rr.pin_diameter * \
        (1 + 0.5 * (-np.log(lambd) + np.log(4) - 0.5) * lambd**2)
    Str = 1 / (0.822 * (g / rr.pin_diameter) + 0.144)
    # Calculate constants
    C1 = 2 / gamma**2 * np.sqrt(alpha / 8) * DH / g
    C2 = gamma**2 / 2
    C3 = 1 - 0.5 * beta
    C4 = np.sqrt(8 / alpha)
    C5 = 1 / Pr_T
    C6 = g / b / eta
    C7 = ax * z_FP_over_D * Str
    C8 = -0.5 * beta
    # Combine constants
    constants = [C1 * C2 * C4 * C6,
                 C3,
                 C1 * C5 * C6 + C1 * C7,
                 C8]
    return constants
