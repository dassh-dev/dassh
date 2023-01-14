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
date: 2023-01-10
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


def calculate_flow_split(asm_obj, regime=None):
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

    Re_bundle = asm_obj.coolant_int_params['Re']
    if regime == 'laminar' or Re_bundle <= Re_bnds[0]:
        return asm_obj.corr_constants['fs']['fs']['laminar']
    elif regime == 'turbulent' or Re_bundle >= Re_bnds[1]:
        return asm_obj.corr_constants['fs']['fs']['turbulent']
    else:
        return _calc_transition_flowsplit(asm_obj)


def _calc_transition_flowsplit(asm_obj, _lambda=None):
    """Calculate the flowsplit in transition conditions

    Notes
    -----
    This is an iterative method (method of successive
    approximations) to determine X1, X2, and X3 based on
    Equation 27 and Equation 30 in the 1986 Cheng-Todreas
    paper. In each iteration, Equation 27 is used to obtain
    relationships for X1/X2 and X3/X2 as a function of f1,
    f2, and f3 (which depend on X1, X2, and X3). Then, X2
    is determined by Equation 30. The process is repeated
    until convergence is reached.

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
    # Subchannel constants
    s = [na_x / asm_obj.bundle_params['area'] for na_x in na]
    Re_iL = asm_obj.corr_constants['ff']['Re_bnds'][0] \
        * asm_obj.params['de'] \
        / asm_obj.bundle_params['de'] \
        * asm_obj.corr_constants['fs']['fs']['laminar']
    Re_iT = asm_obj.corr_constants['ff']['Re_bnds'][1] \
        * asm_obj.params['de'] \
        / asm_obj.bundle_params['de'] \
        * asm_obj.corr_constants['fs']['fs']['turbulent']
    # Do iterations
    try:
        x = _iterate(asm_obj.coolant_int_params['Re'],
                     s,
                     asm_obj.params['de'],
                     asm_obj.bundle_params['de'],
                     Re_iL,
                     Re_iT,
                     asm_obj.corr_constants['ff']['Cf_sc']['laminar'],
                     asm_obj.corr_constants['ff']['Cf_sc']['turbulent'],
                     _lambda=None)
        x = np.array(x)
    except StopIteration:
        x = _calc_transition_flowsplit_APPROX(asm_obj, beta=5.0)
    return x


def _iterate(Re, s, De_i, De_b, Re_iL, Re_iT, Cf_iL, Cf_iT, _lambda=None):
    Re_i = np.array([Re, Re, Re])
    x1 = 1
    x2 = 1
    x3 = 1
    Dei_over_Deb = De_i / De_b
    log10_ReiT_over_ReiL = np.log10(Re_iT / Re_iL)
    # ITERATE
    for iteration in range(100):
        Re_i = Re * np.array([x1, x2, x3]) * Dei_over_Deb
        INT_i = np.log10(Re_i / Re_iL) / log10_ReiT_over_ReiL
        INT_i[INT_i > 1.0] = 1.0
        INT_i[INT_i < 0.0] = 0.0
        ff_iL = Cf_iL / Re_i
        ff_iT = Cf_iT / Re_i**_M['turbulent']
        ff_1, ff_2, ff_3 = _calc_ffb_tr(ff_iL, ff_iT, INT_i, _GAMMA, _lambda)
        # ff_1, ff_2, ff_3 =ff_iL * (1 - INT_i)**_GAMMA + ff_iT * INT_i**_GAMMA
        x1x2 = np.sqrt((ff_2 / De_i[1]) / (ff_1 / De_i[0]))
        x3x2 = np.sqrt((ff_2 / De_i[1]) / (ff_3 / De_i[2]))
        x2_new = 1 / (s[1] + s[0] * x1x2 + s[2] * x3x2)
        x1_new = x1x2 * x2_new
        x3_new = x3x2 * x2_new
        if abs(x2_new - x2) < 1e-5:
            return x1_new, x2_new, x3_new
        else:
            x1 = x1_new
            x2 = x2_new
            x3 = x3_new
    raise StopIteration("CTD transition flow split iteration limit reached")


def _calc_ffb_tr(ff_iL, ff_iT, INT_i, gam=_GAMMA, lam=None):
    """Calculate the transition regime bundle friction factor

    Parameters
    ----------
    ff_iL : numpy.ndarray
        Friction factor for each subchannel in the laminar regime
    ff_iT : numpy.ndarray
        Friction factor for each subchannel in the turbulent regime
    INT_i : numpy.ndarray
        Intermittency factors for each subchannel
    gam (optional) : float
        Exponent fitted from data (by default, equals 1/3)
    lam (optional) : float
        Exponent fitted from data (default=None)

    Notes
    -----
    If lam (lambda) is None, the transition regime friction factor
    is calculated based on Eq. 9 in the Cheng-Todreas 1986 paper.
    If lam is not None, then the friction factor is calculated
    using the updated formulation in Eq. 4 in the Chen-Todreas
    2018 "Upgrade" paper.

    """
    ffb_tr = ff_iL * (1 - INT_i)**gam
    if lam:
        ffb_tr *= (1 - INT_i**lam)
    ffb_tr += ff_iT * INT_i**gam
    return ffb_tr


def _calc_transition_flowsplit_APPROX(asm_obj, beta=5.0):
    """Approximate the transition regime flow split parameters
    using Equation 4.51 from the 1984 Cheng thesis

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometric description of the assembly
    beta : float
        Factor to combine laminar and turbulent flowsplit terms

    Notes
    -----
    Cheng's 1984 thesis recommends a value of 0.05 for beta. There,
    Figure 4.19 shows the edge flowsplit and indicates that beta=0.05.
    In Figure 4.19, as Re increases, the edge flow split quickly
    decreases and levels out as it approaches the constant turbulent
    value. This indicates that the turbulent term is weighted more
    than the laminar term. However, I have observed that beta=0.05
    drastically overweights the laminar term and results in the
    opposite behavior (as Re increases, the edge flow split remains
    level at the laminar constant value, eventually falling sharply
    to reach the turbulent value).

    Upon inspection of Equation 4.51, beta appears to have units
    related to length, in order for the two terms in the numerator
    (or denominator) to have the same units. Therefore, the value
    of 0.05 might be related to the units used. I have found that
    beta=5.0 gives a value very close to that obtained via iteration
    and matches the behavior shown in the figure. Therefore, the
    default value is set to 5.0.

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
    intf_b = ctd.calc_intermittency_factor(
        asm_obj,
        asm_obj.corr_constants['ff']['Re_bnds'][0],
        asm_obj.corr_constants['ff']['Re_bnds'][1])
    xratio_t = asm_obj.corr_constants['fs']['xr']['transition'].copy()
    xratio_t[0] = (xratio_t[0]
                   * (1 - intf_b)**_GAMMA
                   / asm_obj.coolant_int_params['Re'])
    xratio_t[1] = (xratio_t[1]
                   * intf_b**_GAMMA
                   / asm_obj.coolant_int_params['Re']**_M['turbulent']
                   )**_EXP2['turbulent']
    xratio = xratio_t[0] + beta * xratio_t[1]
    x1x2 = xratio[1] / xratio[0]  # Equation 4.51 in Cheng 1984
    x3x2 = xratio[1] / xratio[2]  # Equation 4.51 in Cheng 1984
    flow_split[1] = (asm_obj.bundle_params['area']
                     / (na[1] + x1x2 * na[0] + x3x2 * na[2]))
    flow_split[0] = x1x2 * flow_split[1]
    flow_split[2] = x3x2 * flow_split[1]
    return flow_split


def _calc_intermittency_factor(Re_i, Re_il, Re_it):
    """Calculate the bundle intermittency factor used to
    determine the transition regime friction factor

    Parameters
    ----------
    asm_obj : DASSH Assembly object
    Re_bl : float
        Laminar-transition boundary Reynolds number
    Re_bt : float
        Transition-turbulent boundary Reynolds number

    """
    return np.log10(Re_i / Re_il) / np.log10(Re_it / Re_il)


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
    # Laminar/turbulent: constant flow split!
    const['fs'] = _calc_constant_flowsplits(asm_obj, const)
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
