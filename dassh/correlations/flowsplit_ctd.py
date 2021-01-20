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
date: 2020-10-14
author: matz
Cheng-Todreas correlation for flow split (1986)
"""
########################################################################
import numpy as np
from . import friction_ctd as ctd


applicability = ctd.applicability


def calculate_flow_split(asm_obj, regime=None, beta=0.05):
    """Calculate the flow split into the different types of
    subchannels based on the Cheng-Todreas model

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometric description of the assembly
    regime : str or NoneType
        Indicate flow regime for which to calculate flow split
        {'turbulent', 'laminar', None}; default = None

    Returns
    -------
    numpy.ndarray
        Flow split between interior, edge, and corner coolant
        subchannels
    """
    try:
        Re_bnds = asm_obj.corr_constants['fs']['Re_bnds']
    except (KeyError, AttributeError):
        Re_bnds = ctd.calculate_Re_bounds(asm_obj)

    try:
        Cf = asm_obj.corr_constants['fs']['Cf_sc']
    except (KeyError, AttributeError):
        Cf = ctd.calculate_subchannel_friction_factor_const(asm_obj)

    if regime is not None:
        return _calculate_flow_split(asm_obj, Cf, regime, beta)
    elif asm_obj.coolant_int_params['Re'] <= Re_bnds[0]:
        return _calculate_flow_split(asm_obj, Cf, 'laminar', beta)
    elif asm_obj.coolant_int_params['Re'] >= Re_bnds[1]:
        return _calculate_flow_split(asm_obj, Cf, 'turbulent', beta)
    else:
        return _calculate_flow_split(
            asm_obj, Cf, 'transition', Re_bnds, beta)


def _calculate_flow_split(asm_obj, Cf_dict, regime, Re_bnds=None, beta=1.0):
    """Worker function to calculate the flow split into the
    different types of subchannels based on the Cheng-Todreas
    model.

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometric description of the assembly
    Cf_dict : dict
        Dictionary containing subchannel friction factor constants;
        keys: ['laminar', 'turbulent']
    regime : str {'laminar', 'turbulent', 'transition'}
        Flow regime with which to evaluate flow split ratios
    Re_bnds : list (optional)
        Reynolds number flow regime boundaries for calculating
        intermittency factor in transition regime

    Returns
    -------
    numpy.ndarray
        Flow split between interior, edge, and corner coolant
        subchannels

    Notes
    -----
    This method is imported by the flow split model in the
    Upgraded Cheng-Todreas correlation (flowsplit_uctd)

    """
    try:
        na = asm_obj.corr_constants['fs']['na']
    except (KeyError, AttributeError):
        na = [asm_obj.subchannel.n_sc['coolant']['interior']
              * asm_obj.params['area'][0],
              asm_obj.subchannel.n_sc['coolant']['edge']
              * asm_obj.params['area'][1],
              asm_obj.subchannel.n_sc['coolant']['corner']
              * asm_obj.params['area'][2]]

    flow_split = np.zeros(3)
    if regime == 'transition':
        # beta = 0.05
        # beta = 1.0
        gamma = 1 / 3.0
        m = ctd._m['turbulent']
        _exp2 = 1 / (2 - ctd._m['turbulent'])
        intf_b = ctd.calc_intermittency_factor(
            asm_obj, Re_bnds[0], Re_bnds[1])
        xratio_t1 = (Cf_dict['laminar']
                     * asm_obj.bundle_params['de']
                     * (1 - intf_b)**gamma
                     / asm_obj.params['de']**2
                     / asm_obj.coolant_int_params['Re'])
        xratio_t2 = (Cf_dict['turbulent']
                     * asm_obj.bundle_params['de']**m
                     * intf_b**gamma
                     / asm_obj.params['de']**(m + 1)
                     / asm_obj.coolant_int_params['Re']**m)**_exp2
        # xratio = xratio_t1 + beta * xratio_t2
        xratio = xratio_t1 + beta * xratio_t2
        x1x2 = xratio[1] / xratio[0]
        x3x2 = xratio[1] / xratio[2]

    else:
        _exp1 = (1 + ctd._m[regime]) / (2 - ctd._m[regime])
        _exp2 = 1 / (2 - ctd._m[regime])
        # Ratio between subchannel type 1 and 2 (idx 0 and 1)
        x1x2 = ((asm_obj.params['de'][0]
                 / asm_obj.params['de'][1])**_exp1
                * (Cf_dict[regime][1] / Cf_dict[regime][0])**_exp2)
        # Ratio between subchannel type 3 and 2 (idx 2 and 1)
        x3x2 = ((asm_obj.params['de'][2]
                 / asm_obj.params['de'][1])**_exp1
                * (Cf_dict[regime][1] / Cf_dict[regime][2])**_exp2)

    # Flow split to subchannel type 2
    flow_split[1] = (asm_obj.bundle_params['area']
                     / (na[1] + x1x2 * na[0] + x3x2 * na[2]))
    flow_split[0] = x1x2 * flow_split[1]
    flow_split[2] = x3x2 * flow_split[1]
    return flow_split


def calc_constants(asm_obj):
    """Calculate constants needed by the CTD flowsplit calculation"""
    constants = ctd.calc_constants(asm_obj)
    del constants['Cf_b']
    constants['na'] = [asm_obj.subchannel.n_sc['coolant']['interior']
                       * asm_obj.params['area'][0],
                       asm_obj.subchannel.n_sc['coolant']['edge']
                       * asm_obj.params['area'][1],
                       asm_obj.subchannel.n_sc['coolant']['corner']
                       * asm_obj.params['area'][2]]
    return constants
