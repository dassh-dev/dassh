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
date: 2020-04-24
author: matz
MIT correlation for flow split (1980)
"""
########################################################################
import numpy as np


# Application ranges of correlations
applicability = {}
# P/D upper bound is from SE2 manual; 1980 paper says 1.28
applicability['P/D'] = np.array([1.063, 1.4])
applicability['H/D'] = np.array([4.0, 52.0])
applicability['Nr'] = np.array([37, 900])  # SE2 man: >37; no max found
# applicability['regime'] = ['turbulent', 'transition', 'laminar']
# applicability['Re'] = np.array([50, 1e6])
applicability['bare rod'] = False


def calculate_flow_split(asm):
    """Calculate the flow split into the different types of
    subchannels based on the MIT (Chiu 1980) model

    Parameters
    ----------
    asm : DASSH Assembly object
        Contains the geometric description of the assembly

    Returns
    -------
    numpy.ndarray
        Flow split between interior, edge, and corner coolant
        subchannels
    """
    # If you've already loaded the constant, skip the calculation
    try:
        return asm.corr_constants['fs']['fs']
    except (KeyError, AttributeError):
        pass  # continue onward and do the calculation

    na1 = (asm.subchannel.n_sc['coolant']['interior']
           * asm.params['area'][0])
    na2 = (asm.subchannel.n_sc['coolant']['edge']
           * asm.params['area'][1])
    na3 = (asm.subchannel.n_sc['coolant']['corner']
           * asm.params['area'][2])
    c1 = 2200.0  # correlated constant
    c2 = 1.9  # correlated constant
    c3 = 1.2  # correlated constant
    # Ugly constants
    n = (asm.pin_pitch * (asm.pin_pitch - asm.pin_diameter) / 2
         / ((asm.pin_pitch - 0.5 * asm.pin_diameter) * asm.pin_pitch / 2
            - np.pi * asm.pin_diameter**2 / 16))
    hyp = np.sqrt(np.pi**2 * asm.pin_pitch**2 + asm.wire_pitch**2)
    # hyp1 = np.sqrt(np.pi**2 * (asm.pin_diameter + asm.wire_diameter)**2
    #                + asm.wire_pitch**2)
    vtv2_gap = (10.5 * (asm.wire_diameter / asm.pin_pitch)**0.35
                * np.sqrt(asm.params['wproj'][1]
                          / asm.bare_params['area'][1])
                * asm.pin_pitch / hyp)
    # vtv2_gap = (10.5 * ((asm.pin_pitch - asm.pin_diameter)
    #                     / asm.pin_pitch)**0.35
    #             * np.sqrt(asm.params['wproj'][1]
    #                       / asm.bare_params['area'][1])
    #             * (asm.pin_diameter + asm.wire_diameter) / hyp1)
    lol = (c1 * (asm.params['de'][0] / asm.wire_pitch)
           * (asm.params['wproj'][0] / asm.bare_params['area'][0])
           * (asm.pin_pitch / hyp)**2 + 1)
    lol = lol / c3 / (1 + (c2 * n * vtv2_gap)**2)**1.375
    # combine into flow split relationships
    x1 = ((na1 + na2 + na3)
          / (na1 + ((na2 + na3) * lol**0.571
                    * (asm.params['de'][1]
                       / asm.params['de'][0])**0.714)))
    x2 = ((na1 + na2 + na3)
          / ((na2 + na3) + (na1 * lol**-0.571
                            * (asm.params['de'][0]
                               / asm.params['de'][1])**0.714)))
    return np.array([x1, x2, x2])


def calc_constants(asm_obj):
    """Calculate constants needed by the MIT flowsplit calculation

    Notes
    -----
    In other flow split methods, the parameters can vary axially, so
    these constants minimize the redundancy of the calculation; here,
    the flow split is totally geometric, so the constant is just the
    flow split itself.

    """
    c = {}
    c['fs'] = calculate_flow_split(asm_obj)
    return c
