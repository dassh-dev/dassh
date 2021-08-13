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
date: 2021-08-12
author: matz
Cheng-Todreas correlation for flow split (1986)
"""
########################################################################
import numpy as np
from . import friction_ctd as ctd


applicability = ctd.applicability


########################################################################


# MODULE-WIDE CONSTANTS
_GAMMA = 1 / 3.0
_M = ctd._m
_EXP1 = {}
_EXP2 = {}
for regime in ctd._m.keys():
    _EXP1[regime] = (1 + ctd._m[regime]) / (2 - ctd._m[regime])
    _EXP2[regime] = 1 / (2 - ctd._m[regime])


########################################################################


def calculate_flow_split(asm_obj, regime=None, beta=1.0):
    """Calculate the flow split into the different types of
    subchannels based on the Cheng-Todreas model

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometric description of the assembly
    regime : str or NoneType
        Indicate flow regime for which to calculate flow split
        {'turbulent', 'laminar', None}; default = None
    beta : float
        Beta is a factor used to combine the laminar and turbulent
        flowpslit terms in the transition region. It comes from
        Cheng's 1984 thesis in which he recommends a value of
        0.05. There, Figure 4.19 shows the edge flowsplit assuming
        beta=0.05. However, in reality beta=0.05 gives weird results
        and beta=1.0 matches what's shown in the figure. Therefore,
        it'set to 1.0 here by default.

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
        return _calculate_flow_split(asm_obj, Cf, regime, Re_bnds, beta=beta)
    elif asm_obj.coolant_int_params['Re'] <= Re_bnds[0]:
        return _calculate_flow_split(asm_obj, Cf, 'laminar')
    elif asm_obj.coolant_int_params['Re'] >= Re_bnds[1]:
        return _calculate_flow_split(asm_obj, Cf, 'turbulent')
    else:
        return _calculate_flow_split(asm_obj, Cf, 'transition', Re_bnds, beta)


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
    beta : float
        Beta is a factor used to combine the laminar and turbulent
        flowpslit terms in the transition region. It comes from
        Cheng's 1984 thesis in which he recommends a value of
        0.05. There, Figure 4.19 shows the edge flowsplit assuming
        beta=0.05. However, in reality beta=0.05 gives weird results
        and beta=1.0 matches what's shown in the figure. Therefore,
        it'set to 1.0 here by default.

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

    if regime == 'transition':
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
        intf_b = ctd.calc_intermittency_factor(
            asm_obj, Re_bnds[0], Re_bnds[1])
        xratio_t = asm_obj.corr_constants['fs']['xr']['transition'].copy()
        xratio_t[0] = (xratio_t[0]
                       * (1 - intf_b)**_GAMMA
                       / asm_obj.coolant_int_params['Re'])
        xratio_t[1] = (xratio_t[1]
                       * intf_b**_GAMMA
                       / asm_obj.coolant_int_params['Re']**_M['turbulent']
                       )**_EXP2['turbulent']
        # xratio = xratio_t1 + beta * xratio_t2
        xratio = xratio_t[0] + beta * xratio_t[1]
        x1x2 = xratio[1] / xratio[0]  # Equation 4.51 in Cheng 1984
        x3x2 = xratio[1] / xratio[2]  # Equation 4.51 in Cheng 1984
        flow_split[1] = (asm_obj.bundle_params['area']
                         / (na[1] + x1x2 * na[0] + x3x2 * na[2]))
        flow_split[0] = x1x2 * flow_split[1]
        flow_split[2] = x3x2 * flow_split[1]
    else:
        flow_split = asm_obj.corr_constants['fs']['fs'][regime]
    #     x1x2 = asm_obj.corr_constants['fs']['xr'][regime][0]
    #     x3x2 = asm_obj.corr_constants['fs']['xr'][regime][1]
    #
    # # Flow split to subchannel type 2
    # flow_split[1] = (asm_obj.bundle_params['area']
    #                  / (na[1] + x1x2 * na[0] + x3x2 * na[2]))
    # flow_split[0] = x1x2 * flow_split[1]
    # flow_split[2] = x3x2 * flow_split[1]
    return flow_split


def calc_constants(asm_obj):
    """Calculate constants needed by the CTD flowsplit calculation"""
    const = ctd.calc_constants(asm_obj)
    del const['Cf_b']

    # Total subchannel area for each subchannel type
    const['na'] = [asm_obj.subchannel.n_sc['coolant']['interior']
                   * asm_obj.params['area'][0],
                   asm_obj.subchannel.n_sc['coolant']['edge']
                   * asm_obj.params['area'][1],
                   asm_obj.subchannel.n_sc['coolant']['corner']
                   * asm_obj.params['area'][2]]

    # REGIME RATIO CONSTANTS
    const['xr'] = _calc_regime_ratio_constants(asm_obj, const['Cf_sc'])

    # # Transition regime
    # const['xr'] = {}
    # const['xr']['transition'] = np.array([
    #     (const['Cf_sc']['laminar']
    #      * asm_obj.bundle_params['de']
    #      / asm_obj.params['de']**2),
    #     (const['Cf_sc']['turbulent']
    #      * asm_obj.bundle_params['de']**_M['turbulent']
    #      / asm_obj.params['de']**(_M['turbulent'] + 1))
    # ])
    #
    # # Laminar/turbulent regime
    # for k in ['laminar', 'turbulent']:
    #     const['xr'][k] = np.array([
    #         ((asm_obj.params['de'][0] / asm_obj.params['de'][1])**_EXP1[k]
    #          * (const['Cf_sc'][k][1] / const['Cf_sc'][k][0])**_EXP2[k]),
    #         ((asm_obj.params['de'][2] / asm_obj.params['de'][1])**_EXP1[k]
    #          * (const['Cf_sc'][k][1] / const['Cf_sc'][k][2])**_EXP2[k])
    #     ])

    # Laminar/turbulent: constant flow split!
    const['fs'] = _calc_constant_flowsplits(asm_obj, const)
    # const['fs'] = {}
    # for k in ['laminar', 'turbulent']:
    #     const['fs'][k] = np.zeros(3)
    #     const['fs'][k][1] = (asm_obj.bundle_params['area']
    #                          / (const['na'][1]
    #                             + const['xr'][k][0] * const['na'][0]
    #                             + const['xr'][k][1] * const['na'][2]))
    #     const['fs'][k][0] = const['xr'][k][0] * const['fs'][k][1]
    #     const['fs'][k][2] = const['xr'][k][1] * const['fs'][k][1]
    return const


def _calc_regime_ratio_constants(asm_obj, Cf_sc):
    """Constant ratios for laminar, turbulent, and transition regimes"""
    xr = {}
    xr['transition'] = np.array([
        (Cf_sc['laminar']
         * asm_obj.bundle_params['de']
         / asm_obj.params['de']**2),
        (Cf_sc['turbulent']
         * asm_obj.bundle_params['de']**_M['turbulent']
         / asm_obj.params['de']**(_M['turbulent'] + 1))
    ])
    # Laminar/turbulent regime
    for k in ['laminar', 'turbulent']:
        xr[k] = np.array([
            ((asm_obj.params['de'][0] / asm_obj.params['de'][1])**_EXP1[k]
             * (Cf_sc[k][1] / Cf_sc[k][0])**_EXP2[k]),
            ((asm_obj.params['de'][2] / asm_obj.params['de'][1])**_EXP1[k]
             * (Cf_sc[k][1] / Cf_sc[k][2])**_EXP2[k])
        ])
    return xr


def _calc_constant_flowsplits(asm_obj, const):
    """Laminar and turbulent flowsplits are constant"""
    fs = {}
    for k in ['laminar', 'turbulent']:
        fs[k] = np.zeros(3)
        fs[k][1] = (asm_obj.bundle_params['area']
                    / (const['na'][1]
                       + const['xr'][k][0] * const['na'][0]
                       + const['xr'][k][1] * const['na'][2]))
        fs[k][0] = const['xr'][k][0] * fs[k][1]
        fs[k][2] = const['xr'][k][1] * fs[k][1]
    return fs
