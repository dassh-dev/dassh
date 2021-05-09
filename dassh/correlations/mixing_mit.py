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
date: 2021-03-31
author: matz
MIT (Chiu-Rohsenow-Todreas) mixing correlations: eddy diffusivity and
swirl velocity (1978)
"""
########################################################################
import numpy as np
from . import corr_utils


# Application ranges of correlations (No info given in SE2 manual)
applicability = {}
applicability['Nr'] = np.array([19, 900])
applicability['bare rod'] = False


def calculate_mixing_params(asm_obj, shortcut=True):
    """Calculate the eddy diffusivity and swirl velocity based
    on the MIT (Chiu-Rohsenow-Todreas) 1978 correlations.

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometry and flow parameters

    Returns
    -------
    tuple
        tuple of two floats, the eddy diffusivity and the swirl
        velocity for the assembly geometry and flow regime

    Notes
    -----

    """
    if shortcut:
        try:
            return (asm_obj.corr_constants['mix']['eddy'],
                    asm_obj.corr_constants['mix']['swirl'])
        except (KeyError, AttributeError):
            pass  # continue onward and do the calculation

    # Eddy diffusivity
    # Just spell these out, this is only called once anyway
    P = asm_obj.pin_pitch
    D = asm_obj.pin_diameter
    Dw = asm_obj.wire_diameter
    H = asm_obj.wire_pitch
    AS = corr_utils.calculate_bare_rod_sc_area('mit', P, D, Dw)
    AR = corr_utils.calculate_wproj('mit', P, D, Dw)
    # AS1 = np.sqrt(3) * P**2 / 4 - np.pi * D**2 / 8
    # AR1 = np.pi * (P - 0.5 * D)**2 / 6 - np.pi * D**2 / 24
    e = (0.128
         * np.sqrt(P / (P - D))
         * np.sqrt(AR[0] / AS[0])
         * (D + Dw) * P**2
         / AS[0] / np.sqrt(np.pi**2 * (D + Dw)**2 + H**2))
    e *= asm_obj.params['de'][0]

    # Swirl velocity
    # AS2 = P * (0.5 * D + Dw) - 0.125 * np.pi * D**2
    # AR2 = np.pi * (0.25 * (0.5 * D + Dw)**2 - 0.0625 * D**2)
    s = (10.5
         * ((P - D) / P)**0.35
         * (AR[1] / AS[1])**0.5
         * (D + Dw)
         / np.sqrt(np.pi**2 * (D + Dw)**2 + H**2))
    return e, s


def calc_constants(asm_obj):
    """Calculate and store constants for MIT mixing parameters
    so I don't have to recalculate them at every step"""
    c = {}
    c['eddy'], c['swirl'] = calculate_mixing_params(asm_obj)
    return c
