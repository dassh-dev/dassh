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
Upgraded Cheng-Todreas Detailed correlations (2018)
"""
########################################################################
import numpy as np
from . import friction_ctd as ctd


# Application ranges of friction factor correlations
applicability = {}
applicability['P/D'] = np.array([1.0, 1.42])
applicability['H/D'] = np.array([8.0, 52.0])
applicability['Nr'] = np.array([7, 217])
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
    re_bl = 320.0 * 10**(asm_obj.pin_pitch
                         / asm_obj.pin_diameter - 1.0)
    return (re_bl, re_bt)


########################################################################
# SUBCHANNEL FRICTION FACTOR CONSTANTS
########################################################################


def calculate_subchannel_friction_factor_const(asm):
    """Calculate the Upgraded Cheng-Todreas Detailed correlation
    subchannel friction factor constants (Cf)

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
    return ctd._calc_sc_ff_const(asm, wire_drag, wire_sweep)


def calculate_wire_sweeping_const(h_over_d):
    """Calculate the wire-sweeping constant for the Upgraded
    Cheng-Todreas friction factor constant calculation"""
    ws = {}
    if h_over_d == 0.0:
        ws['turbulent'] = 0.0
        ws['laminar'] = 0.0
    else:
        ws['turbulent'] = -11.0 * np.log10(h_over_d) + 19.0
        ws['laminar'] = ws['turbulent']
    return ws


def calculate_wire_drag_const(wd_over_d, h_over_d):
    """Calculate the wire-drag constant for the Upgraded
    Cheng-Todreas friction factor constant calculation"""
    wd = {}
    if wd_over_d == 0.0:
        wd['turbulent'] = 0.0
        wd['laminar'] = 0.0
    else:
        wd['turbulent'] = ((19.56 - 98.71 * wd_over_d + 303.47 * wd_over_d**2)
                           / h_over_d**0.541)
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
    This is a wrapper function used to apply UCTD (2018) parameters
    to the bundle average friction factor constant calculation

    """
    try:
        cf_sc = asm_obj.corr_constants['ff']['Cf_sc']
    except (KeyError, AttributeError):
        cf_sc = calculate_subchannel_friction_factor_const(asm_obj)

    return ctd._calc_cfb(asm_obj, cf_sc)


########################################################################
# BUNDLE FRICTION FACTOR
########################################################################


def calculate_bundle_friction_factor(asm_obj):
    """Calculate the bundle-average friction factor

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the assembly geometric details and bundle Re

    Returns
    -------
    float
        Bundle-average friction factor based on assembly geometry
        and flow regime

    """
    try:
        cfb = asm_obj.corr_constants['ff']['Cf_b']
    except (KeyError, AttributeError):
        cfb = calculate_bundle_friction_factor_const(asm_obj)

    f = {}
    # Calculate friction factor for laminar and turbulent regimes
    for r in cfb.keys():
        f[r] = cfb[r] / asm_obj.coolant_int_params['Re']**ctd._m[r]
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
        x = ctd.calc_intermittency_factor(asm_obj, Re_bl, Re_bt)
        # Different correlation for transition region than O.G. CTD
        return (f['laminar'] * (1 - x)**(1 / 3.0) * (1 - x**7.0)
                + f['turbulent'] * x**(1 / 3.0))
