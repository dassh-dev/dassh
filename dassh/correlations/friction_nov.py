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
date: 2020-04-01
author: matz
Novendstern correlation (1972) for friction factor
"""
########################################################################
import numpy as np
from . import flowsplit_nov


applicability = {}
applicability['P/D'] = np.array([1.06, 1.42])
applicability['H/D'] = np.array([8.0, 96.0])
applicability['Nr'] = np.array([19, 217])
applicability['regime'] = ['transition', 'turbulent']
applicability['Re'] = np.array([2600, 1e5])
applicability['bare rod'] = False


def calculate_bundle_friction_factor(asm_obj, flow_split=None):
    """Calculate the bundle-average friction factor

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the assembly geometric details
    flow_split : numpy.ndarray
        Flow split parameter for the three coolant subchannel types

    Returns
    -------
    float
        Bundle-average friction factor based on assembly geometry
        and flow regime

    """
    # if flow_split is None:
    #     flow_split = flowsplit_nov.calculate_flow_split(asm_obj,
    #                                                     shortcut=False)
    # Re1 = (asm_obj.coolant_int_params['Re'] * flow_split[0]
    #        * asm_obj.params['de'][0] / asm_obj.bundle_params['de'])
    # pd = asm_obj.pin_pitch / asm_obj.pin_diameter
    # hd = asm_obj.wire_pitch / asm_obj.pin_diameter
    # M = (1.034 / pd**0.124
    #      + 29.7 * pd**6.94 * Re1**0.086 / hd**2.239)**0.885
    # f_smooth = (2 * np.log10(-5.028 * np.log10(16.76 / Re1) / Re1))**-2
    # # f_smooth = 0.316 / asm_obj.coolant_int_params['Re']**0.25
    # print(flow_split[0])
    # print(Re1)
    # print(M)
    # print(f_smooth)
    # return (f_smooth * M * flow_split[0]**2
    #         * asm_obj.bundle_params['de'] / asm_obj.params['de'][0])
    # If you've already loaded the constant, skip the calculation
    try:
        cc = asm_obj.corr_constants['ff']
    except (KeyError, AttributeError):
        cc = calc_constants(asm_obj)

    if flow_split is None:
        if cc is not None and 'fs' in cc.keys():
            flow_split = cc['fs']
        else:
            flow_split = flowsplit_nov.calculate_flow_split(
                asm_obj, shortcut=False)
    else:
        pass

    Re1 = asm_obj.coolant_int_params['Re'] * flow_split[0] / cc['de_ratio']
    M = (cc['C1'] + cc['C2'] * Re1**0.086 / cc['C3'])**0.885
    f_smooth = (2 * np.log10(-5.028 * np.log10(16.76 / Re1) / Re1))**-2
    # f_smooth = 0.316 / asm_obj.coolant_int_params['Re']**0.25
    return (f_smooth * M * flow_split[0]**2 * cc['de_ratio'])


def calc_constants(asm_obj):
    """Calculate constants for the Novendstern friction factor

    Notes
    -----
    In other methods, the parameters can vary axially, so these
    constants minimize the redundancy of the calculation
    """
    c = {}
    c['fs'] = flowsplit_nov.calculate_flow_split(asm_obj, shortcut=False)
    pd = asm_obj.pin_pitch / asm_obj.pin_diameter
    hd = asm_obj.wire_pitch / asm_obj.pin_diameter
    c['C1'] = 1.034 / pd**0.124
    c['C2'] = 29.7 * pd**6.94
    c['C3'] = hd**2.239
    c['de_ratio'] = asm_obj.bundle_params['de'] / asm_obj.params['de'][0]
    return c
