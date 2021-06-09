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
date: 2021-06-09
author: matz
Cheng-Todreas Detailed original correlations (1986)
"""
########################################################################
import numpy as np
from . import corr_utils


# Cheng-Todreas Reynolds number exponent
_m = {'laminar': 1.0, 'turbulent': 0.18}


# Application ranges of friction factor correlations
applicability = {}
applicability['P/D'] = np.array([1.0, 1.42])
applicability['H/D'] = np.array([4.0, 52.0])
applicability['Nr'] = np.array([19, 217])
applicability['regime'] = ['turbulent', 'transition', 'laminar']
applicability['Re'] = np.array([50, 1e6])
applicability['bare rod'] = True


########################################################################
# CONSTANTS - based only on assembly geometry
########################################################################


def calc_constants(asm_obj):
    """Calculate and store constants for friction factor calculation

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains geometric parameters on which constants are based

    Returns
    -------
    dict
        Constants used in friction factor calculation:
        1. Reynolds number flow regime bounds {list}
            (laminar-transition-turbulent)
        2. Subchannel friction factor constants {dict}
            (turbulent, laminar; one for each type of subchannel)
        3. Bundle friction factor constants {dict}
            (turbulent, laminar)

    """
    c = {}
    c['Re_bnds'] = calculate_Re_bounds(asm_obj)
    c['Cf_sc'] = calculate_subchannel_friction_factor_const(asm_obj)
    c['Cf_b'] = calculate_bundle_friction_factor_const(asm_obj)
    return c


########################################################################
# REYNOLDS NUMBER BOUNDARIES: laminar-transition, transition-turbulent
########################################################################


def calculate_Re_bounds(asm_obj):
    """Calculate the Reynolds numbers for the boundaries of the
    laminar, transition, and turbulent flow regimes

    Parameters
    ----------
    asm_obj : DASSH assembly object
        Required attributes: pin diameter (m) and pin pitch (m)

    Returns
    -------
    tuple
        Reynolds numbers (float) for the boundaries between the
        laminar-transition and transition-turbulent flow regimes

    """
    re_bt = 1e4 * 10**(0.7 * (asm_obj.pin_pitch
                              / asm_obj.pin_diameter - 1.0))
    re_bl = 3e2 * 10**(1.7 * (asm_obj.pin_pitch
                              / asm_obj.pin_diameter - 1.0))
    return (re_bl, re_bt)


########################################################################
# SUBCHANNEL FRICTION FACTOR CONSTANTS
########################################################################


def calculate_subchannel_friction_factor_const(asm):
    """Calculate the Cheng-Todreas Detailed friction factor
    constant (Cf)

    Parameters
    ----------
    asm : DASSH Assembly object
        Contains the assembly geometry data

    Returns
    -------
    dict
        Friction factor constants (float) for different coolant
        subchannels in different flow regimes (keys: ["laminar",
        "turbulent"])
    """
    # Get the Cheng-Todreas friction constant polynomial factors
    # and other constants required for the correlation
    wire_sweep = calculate_wire_sweeping_const(asm.wire_pitch
                                               / asm.pin_diameter)
    wire_drag = calculate_wire_drag_const(asm.wire_diameter
                                          / asm.pin_diameter,
                                          asm.wire_pitch
                                          / asm.pin_diameter)
    # Calculate friction factor constants
    return _calc_sc_ff_const(asm, wire_drag, wire_sweep)


def _calc_sc_ff_const(asm, wd, ws):
    """Worker function to calculate subchannel friction factors
    based on assembly geometry and correlation-specific constants

    Parameters
    ----------
    asm : DASSH Assembly object
        Contains geometry parameters
    wd : dict
        Dictionary of turbulent and laminar wire drag constants
    ws : dict
        Dictionary of turbulent and laminar wire sweeping constants

    """
    # Calculate some assembly-specific parameters to keep it clean
    p2d_m1 = asm.pin_pitch / asm.pin_diameter - 1  # p/d - 1
    w2d_m1 = asm.edge_pitch / asm.pin_diameter - 1  # edge_pitch / d - 1

    # Calculate the bare rod friction factor polynomial constants
    # Interior subchannels
    a1 = get_ff_poly_constants(asm.pin_pitch / asm.pin_diameter)
    # Edge, corner subchannels
    a23 = get_ff_poly_constants(asm.edge_pitch / asm.pin_diameter)

    # Calculate bare rod friction factors
    Cfb = {}
    for r in ['laminar', 'turbulent']:
        # Cfb[r] = np.zeros(3)
        # Cfb[r][0] = np.sum(a1[r][0]
        #                    * np.array([1.0, p2d_m1, p2d_m1**2]))
        # Cfb[r][1:] = np.sum(a23[r][1:]
        #                     * np.array([1.0, w2d_m1, w2d_m1**2]),
        #                     axis=1)
        Cfb[r] = np.zeros(3)
        Cfb[r][0] = (a1[r][0, 0] + a1[r][0, 1] * p2d_m1
                     + a1[r][0, 2] * p2d_m1**2)
        Cfb[r][1] = (a23[r][1, 0] + a23[r][1, 1] * w2d_m1
                     + a23[r][1, 2] * w2d_m1**2)
        Cfb[r][2] = (a23[r][2, 0] + a23[r][2, 1] * w2d_m1
                     + a23[r][2, 2] * w2d_m1**2)

    # Calculate wire-wrapped friction factors
    Cf = {}
    if asm.wire_diameter == 0.0:
        Cf = Cfb
    else:
        bwp = corr_utils.calculate_bare_rod_wp(
            asm.pin_pitch, asm.pin_diameter, asm.edge_pitch)
        wproj = corr_utils.calculate_wproj(
            'ctd', asm.pin_pitch, asm.pin_diameter, asm.wire_diameter)
        b_area = corr_utils.calculate_bare_rod_sc_area(
            'ctd', asm.pin_pitch, asm.pin_diameter, asm.wire_diameter,
            asm.edge_pitch)
        for r in ['laminar', 'turbulent']:
            Cf[r] = np.zeros(3)
            # Interior
            Cf[r][0] = (Cfb[r][0] * (bwp[0] / asm.params['wp'][0])
                        + (wd[r] * (3 * wproj[0] / b_area[0])
                           * (asm.params['de'][0] / asm.wire_pitch)
                           * (asm.params['de'][0]
                              / asm.wire_diameter)**_m[r]))
            # Edge
            _exp = (3.0 - _m[r]) / 2
            Cf[r][1] = Cfb[r][1]
            Cf[r][1] *= (1 + (ws[r] * (wproj[1] / b_area[1])
                              * np.tan(asm.params['theta'])**2))**_exp
            # Corner

            Cf[r][2] = Cfb[r][2]
            Cf[r][2] *= (1 + (
                ws[r] * (wproj[2] / b_area[2])
                * np.tan(asm.params['theta'])**2))**_exp
    return Cf


def get_ff_poly_constants(pin_or_edge_pitch_to_diam):
    """Return the quadratic polynomial constants used to fit the bare
    rod friction factor constant used in the Cheng-Todreas calculations
    as a function of P/D or W/D (where W/D applies to the edge and
    corner subchannels and W is the "edge-pitch" parameter)."""
    a = {}
    if pin_or_edge_pitch_to_diam <= 1.1:
        a['laminar'] = np.array([[26.00, 888.2, -3334.0],
                                 [26.18, 554.5, -1480.0],
                                 [26.98, 1636.0, -10050.0]])
        a['turbulent'] = np.array([[0.09378, 1.398, -8.664],
                                   [0.09377, 0.8732, -3.341],
                                   [0.1004, 1.625, -11.85]])
    else:
        a['laminar'] = np.array([[62.97, 216.9, -190.2],
                                 [44.40, 256.7, -267.6],
                                 [87.26, 38.59, -55.12]])
        a['turbulent'] = np.array([[0.1458, 0.03632, -0.03333],
                                   [0.1430, 0.04199, -0.04428],
                                   [0.1499, 0.006706, -0.009567]])
    return a


def calculate_wire_sweeping_const(h_over_d):
    """Calculate the wire-sweeping constant for the Cheng-Todreas (or
    Upgraded Cheng-Todreas) friction factor constant calculation"""
    ws = {}
    if h_over_d == 0.0:
        ws['turbulent'] = 0.0
        ws['laminar'] = 0.0
    else:
        ws['turbulent'] = 20.0 * np.log10(h_over_d) - 7.0
        ws['laminar'] = 0.3 * ws['turbulent']
    return ws


def calculate_wire_drag_const(wd_over_d, h_over_d):
    """Calculate the wire-drag constant for the Cheng-Todreas (or
    Upgraded Cheng-Todreas) friction factor constant calculation"""
    wd = {}
    if wd_over_d == 0.0:
        wd['turbulent'] = 0.0
        wd['laminar'] = 0.0
    else:
        wd['turbulent'] = ((29.5 - 140.0 * wd_over_d
                            + 401.0 * wd_over_d**2) / h_over_d**0.85)
        wd['laminar'] = 1.4 * wd['turbulent']
    return wd


########################################################################
# BUNDLE FRICTION FACTOR CONSTANT
########################################################################


def calculate_bundle_friction_factor_const(asm_obj):
    """Calculate the bundle (assembly avg) friction factor constant

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains geometric and flow parameters
    subchannel_cf : dict
        dict (keys: ['laminar', 'turbulent']) of numpy.ndarray
        (length = 3), each containing friction factor constants
        for each coolant subchannel

    Notes
    -----
    This is a wrapper function used to apply CTD (1986) parameters
    to the bundle average friction factor constant calculation

    """
    try:
        cf_sc = asm_obj.corr_constants['ff']['Cf_sc']
    except (KeyError, AttributeError, TypeError):
        cf_sc = calculate_subchannel_friction_factor_const(asm_obj)
    return _calc_cfb(asm_obj, cf_sc)


def _calc_cfb(asm_obj, subchannel_cf):
    """Calculate the bundle (assembly avg) friction factor constant

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains geometric and flow parameters
    subchannel_cf : dict
        dict (keys: ['laminar', 'turbulent']) of numpy.ndarray
        (length = 3), each containing friction factor constants
        for each coolant subchannel

    Notes
    -----
    This is a worker function to calculate the bundle friction
    factor constant for correlations that provide different
    parameters (occurs through the subchannel friction factor
    constants)

    """
    # Calculate friction factor constants for each subchannel in
    # each possible flow regime (laminar or turbulent)
    cfb = {}
    # Combine constants into bundle-avg friction factor constant
    for r in subchannel_cf.keys():  # flow regimes: laminar, turbulent
        cfb[r] = 0.0
        _exp1 = _m[r] / (2 - _m[r])
        _exp2 = 1 / (_m[r] - 2)
        _exp3 = _m[r] - 2
        for i in range(3):
            sci = ['interior', 'edge', 'corner'][i]
            NAi = (asm_obj.subchannel.n_sc['coolant'][sci]
                   * asm_obj.params['area'][i]
                   / asm_obj.bundle_params['area'])
            cfb[r] += (NAi
                       * (asm_obj.params['de'][i]
                          / asm_obj.bundle_params['de'])**_exp1
                       * (subchannel_cf[r][i]
                          / asm_obj.params['de'][i])**_exp2)
        cfb[r] = asm_obj.bundle_params['de'] * cfb[r]**_exp3
    return cfb


########################################################################
# BUNDLE FRICTION FACTOR
########################################################################


def calculate_bundle_friction_factor(asm_obj):
    """Calculate the bundle-average friction factor using the Cheng-
    Todreas Detailed (1986) correlation.

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the assembly geometric details and bundle Re

    Returns
    -------
    float
        Bundle-average friction factor at given flow conditions

    Notes
    -----
    This is a wrapper function around _calc_bundle_ff, which can be
    used to calculate the friction factors by both the CTD and CTS
    correlations.

    """
    try:
        cfb = asm_obj.corr_constants['ff']['Cf_b']
    except (KeyError, AttributeError):
        cfb = calculate_bundle_friction_factor_const(asm_obj)

    return _calc_bundle_ff(asm_obj, cfb)


def _calc_bundle_ff(asm_obj, cfb):
    """Calculate the bundle-average friction factor

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the assembly geometric details and bundle Re
    cfb : dict
        Bundle friction factor constants for laminar and turbulent
        flow regimes

    Returns
    -------
    float
        Bundle-average friction factor based on assembly geometry
        and flow regime

    Notes
    -----
    This is a worker function that can calculate the friction factor
    for both the CTD and CTS correlations; the difference between
    them arises in the bundle friction factor constant, which is
    passed as an argument.

    """
    f = {}
    # Calculate friction factor for laminar and turbulent regimes
    for r in cfb.keys():
        f[r] = cfb[r] / asm_obj.coolant_int_params['Re']**_m[r]

    # If transition region, combine laminar and turbulent friction
    # factors using intermittency fxn; otherwise, return value
    try:
        Re_bl, Re_bt = asm_obj.corr_constants['ff']['Re_bnds']
    except (KeyError, AttributeError):
        Re_bl, Re_bt = calculate_Re_bounds(asm_obj)

    if asm_obj.coolant_int_params['Re'] <= Re_bl:
        return f['laminar']
    elif asm_obj.coolant_int_params['Re'] >= Re_bt:
        return f['turbulent']
    else:
        # Transition regime intermittency factor
        x = calc_intermittency_factor(asm_obj, Re_bl, Re_bt)
        # print(x)
        return (f['laminar'] * (1 - x)**(1 / 3.0)
                + f['turbulent'] * x**(1 / 3.0))


def calc_intermittency_factor(asm_obj, Re_bl, Re_bt):
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
    return((np.log10(asm_obj.coolant_int_params['Re']) - np.log10(Re_bl))
           / (np.log10(Re_bt) - np.log10(Re_bl)))
