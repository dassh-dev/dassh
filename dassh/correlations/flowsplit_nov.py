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
date: 2021-07-02
author: matz
Novendstern correlation (1972) for flow split
"""
########################################################################
import numpy as np


applicability = {}
applicability['P/D'] = np.array([1.06, 1.42])
applicability['H/D'] = np.array([8.0, 96.0])
applicability['Nr'] = np.array([19, 217])
applicability['regime'] = ['transition', 'turbulent']
applicability['Re'] = np.array([2600, 1e5])
applicability['bare rod'] = False


def calculate_flow_split(asm_obj, shortcut=True):
    """Calculate the flow split into the different types of
    subchannels based on the Novendstern correlation

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometric description of the assembly

    Returns
    -------
    numpy.ndarray
        Flow split between interior, edge, and corner coolant
        subchannels

    """
    # If you've already loaded the constant, skip the calculation
    if shortcut:
        try:
            return asm_obj.corr_constants['fs']['fs']
        except (KeyError, AttributeError):
            pass  # continue onward and do the calculation

    na1 = (asm_obj.subchannel.n_sc['coolant']['interior']
           * asm_obj.params['area'][0])
    na2 = (asm_obj.subchannel.n_sc['coolant']['edge']
           * asm_obj.params['area'][1])
    na3 = (asm_obj.subchannel.n_sc['coolant']['corner']
           * asm_obj.params['area'][2])

    x1 = (asm_obj.bundle_params['area']
          / (na1 + na2 * (asm_obj.params['de'][1]
                          / asm_obj.params['de'][0])**0.714
             + na3 * (asm_obj.params['de'][2]
                      / asm_obj.params['de'][0])**0.714))
    x2 = (asm_obj.bundle_params['area']
          / (na2 + na1 * (asm_obj.params['de'][0]
                          / asm_obj.params['de'][1])**0.714
             + na3 * (asm_obj.params['de'][2]
                      / asm_obj.params['de'][1])**0.714))
    x3 = (asm_obj.bundle_params['area']
          / (na3 + na1 * (asm_obj.params['de'][0]
                          / asm_obj.params['de'][2])**0.714
             + na2 * (asm_obj.params['de'][1]
                      / asm_obj.params['de'][2])**0.714))
    return np.array([x1, x2, x3])


def calc_constants(asm_obj):
    """Calculate constants for the Novendstern flowsplit calculation

    Notes
    -----
    In other flow split methods, the parameters can vary axially, so
    these constants minimize the redundancy of the calculation; here,
    the flow split is totally geometric, so the constant is just the
    flow split itself.

    """
    c = {}
    c['fs'] = calculate_flow_split(asm_obj, shortcut=False)
    return c
