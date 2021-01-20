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
date: 2020-06-12
author: matz
MIT (Chiu-Rohsenow-Todreas) mixing correlations: eddy diffusivity and
swirl velocity (1978)
"""
########################################################################
import numpy as np


# Application ranges of correlations (No info given in SE2 manual)
applicability = {}
applicability['Nr'] = np.array([19, 900])
applicability['bare rod'] = False


def calculate_mixing_params(asm_obj):
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
    # Eddy diffusivity
    # Just spell these out, this is only called once anyway
    P = asm_obj.pin_pitch
    D = asm_obj.pin_diameter
    Dw = asm_obj.wire_diameter
    H = asm_obj.wire_pitch
    AS1 = np.sqrt(3) * P**2 / 4 - np.pi * D**2 / 8
    AR1 = np.pi * (P - 0.5 * D)**2 / 6 - np.pi * D**2 / 24
    e = (0.128
         * np.sqrt(P / (P - D))
         * np.sqrt(AR1 / AS1)
         * (D + Dw) * P**2
         / AS1 / np.sqrt(np.pi**2 * (D + Dw)**2 + H**2))
    # e = (0.128
    #      * np.sqrt(asm_obj.pin_pitch
    #                / (asm_obj.pin_pitch - asm_obj.pin_diameter))
    #      * np.sqrt(asm_obj.params['wproj'][0]
    #                / asm_obj.bare_params['area'][0])
    #      * (asm_obj.pin_diameter + asm_obj.wire_diameter)
    #      * asm_obj.pin_pitch**2
    #      / np.sqrt((np.pi**2 * (asm_obj.pin_diameter
    #                             + asm_obj.wire_diameter)**2
    #                 + asm_obj.wire_pitch**2))
    #      / asm_obj.bare_params['area'][0])
    e *= asm_obj.params['de'][0]

    # Swirl velocity
    s = (10.5 * (asm_obj.pin_pitch - asm_obj.pin_diameter)**0.35
         * (asm_obj.pin_diameter + asm_obj.wire_diameter)
         * np.sqrt(asm_obj.params['wproj'][1]))
    s /= (asm_obj.pin_pitch**0.35
          * np.sqrt(asm_obj.params['area'][1]
                    * (np.pi**2 * (asm_obj.pin_diameter
                                   + asm_obj.wire_diameter)**2
                       + asm_obj.wire_pitch**2)))
    return e, s


def calc_constants(asm_obj):
    """Calculate and store constants for MIT mixing parameters
    so I don't have to recalculate them at every step"""
    c = {}
    c['eddy'], c['swirl'] = calculate_mixing_params(asm_obj)
    return c
