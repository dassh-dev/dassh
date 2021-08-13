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
Cheng-Todreas Detailed mixing correlations: eddy diffusivity and
swirl velocity (1986)
"""
########################################################################
import numpy as np
from . import friction_ctd as ctd_ff
from . import flowsplit_ctd as ctd_fs
from . import corr_utils


applicability = {}
applicability['P/D'] = np.array([1.067, 1.35])
applicability['H/D'] = np.array([4.0, 52.0])
applicability['Nr'] = np.array([7, 217])
applicability['regime'] = ['turbulent', 'transition', 'laminar']
applicability['Re'] = np.array([400, 1e6])
applicability['bare rod'] = False


def calculate_mixing_params(asm_obj):
    """Calculate the dimensionless eddy diffusivity and swirl velocity
    based on the Cheng-Todreas 1986 correlation.

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
    To be tested against Tables 1 and 2 of the 1986 Cheng-Todreas paper

    """
    # Get params in laminar and turbulent regimes - either access from
    # stored location or recalculate
    try:
        eddy = asm_obj.corr_constants['mix']['eddy']
        swirl = asm_obj.corr_constants['mix']['swirl']
    except (KeyError, AttributeError):
        eddy, swirl = calculate_laminar_turbulent_params(asm_obj)

    # Calculate parameters based on subchannel Reynolds number
    try:
        Re_bl, Re_bt = asm_obj.corr_constants['mix']['Re_bnds']
    except (KeyError, AttributeError):
        Re_bl, Re_bt = ctd_ff.calculate_Re_bounds(asm_obj)

    if asm_obj.coolant_int_params['Re'] <= Re_bl:
        eddy = eddy['laminar']
        swirl = swirl['laminar']
    elif asm_obj.coolant_int_params['Re'] >= Re_bt:
        eddy = eddy['turbulent']
        swirl = swirl['turbulent']
    else:  # Transition regime; use intermittency factor
        x = calc_sc_intermittency_factors(asm_obj, Re_bl, Re_bt)
        eddy = ((eddy['turbulent']
                 - eddy['laminar']) * x[0]**(2 / 3.0)
                + eddy['laminar'])
        swirl = ((swirl['turbulent']
                  - swirl['laminar']) * x[1]**(2 / 3.0)
                 + swirl['laminar'])
    return eddy * asm_obj.L[0][0], swirl


def calc_sc_intermittency_factors(asm_obj, Re_bL, Re_bT):
    """Calculate the intermittency factors for the interior and edge
    coolant subchannels; required to find the mixing parameters in
    the transition region

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometry and flow parameters
    Re_bL : float
        Reynolds number boundary between laminar-transition regimes
    Re_bT : float
        Reynolds number boundary between transition-turbulent regimes

    Notes
    -----
    See Equations 10, 11, and 30-32 in the Cheng-Todreas 1986 paper

    """
    y = np.zeros(3)
    fs = ctd_fs.calculate_flow_split(asm_obj)
    v_bundle = (asm_obj.int_flow_rate / asm_obj.coolant.density
                / asm_obj.bundle_params['area'])
    try:
        fs_L = asm_obj.corr_constants['fs']['fs']['laminar']
        fs_T = asm_obj.corr_constants['fs']['fs']['turbulent']
    except (KeyError, AttributeError):
        fs_L = ctd_fs.calculate_flow_split(asm_obj, 'laminar')
        fs_T = ctd_fs.calculate_flow_split(asm_obj, 'turbulent')
    Re = (asm_obj.coolant.density * v_bundle * fs
          * asm_obj.params['de'] / asm_obj.coolant.viscosity)
    Re_iL = (Re_bL * fs_L * asm_obj.params['de']
             / asm_obj.bundle_params['de'])
    Re_iT = (Re_bT * fs_T * asm_obj.params['de']
             / asm_obj.bundle_params['de'])
    y = ((np.log10(Re) - np.log10(Re_iL))
         / (np.log10(Re_iT) - np.log10(Re_iL)))
    return y


def calculate_laminar_turbulent_params(asm_obj):
    """Calculate laminar and turbulent regime mixing params"""
    eddy = {}
    swirl = {}
    cm, cs = calculate_mixing_param_constants(asm_obj)
    # Calculate params in laminar and turbulent regimes
    AS = corr_utils.calculate_bare_rod_sc_area('ctd',
                                               asm_obj.pin_pitch,
                                               asm_obj.pin_diameter,
                                               asm_obj.wire_diameter,
                                               asm_obj.edge_pitch)
    AR = corr_utils.calculate_wproj('ctd',
                                    asm_obj.pin_pitch,
                                    asm_obj.pin_diameter,
                                    asm_obj.wire_diameter)
    for r in ['laminar', 'turbulent']:
        eddy[r] = (cm[r] * np.tan(asm_obj.params['theta'])
                   * np.sqrt(AR[0] / AS[0]))
        swirl[r] = (cs[r] * np.tan(asm_obj.params['theta'])
                    * np.sqrt(AR[1] / AS[1]))
    return eddy, swirl


def calculate_mixing_param_constants(asm_obj):
    """Calculate the constants Cs and Cm required for the
    determination of the Cheng-Todreas mixing parameters

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometry and flow parameters

    Returns
    -------
    tuple
        tuple of two dicts, each containing the laminar and turbulent
        constants for the calculation of eddy diffusivity and the swirl
        velocity for the assembly

    Notes
    -----
    Implemented as a separate method so that it can be tested against
    the results in Tables 1 and 2 of the Cheng-Todreas 1986 paper.

    """
    try:
        c_over_d = (asm_obj.d['pin-pin'] / asm_obj.pin_diameter)**-0.5
    except ZeroDivisionError:  # single pin, d['pin-pin'] = 0
        c_over_d = 0.0

    h_over_d = (asm_obj.wire_pitch / asm_obj.pin_diameter)**0.3
    cm = {}
    cs = {}
    if asm_obj.n_pin >= 19:
        # Laminar
        cm['laminar'] = 0.077 * c_over_d
        cs['laminar'] = 0.413 * h_over_d
        # Turbulent
        cm['turbulent'] = 0.14 * c_over_d
        cs['turbulent'] = 0.75 * h_over_d
    else:
        # Laminar
        cm['laminar'] = 0.055 * c_over_d
        cs['laminar'] = 0.33 * h_over_d
        # Turbulent
        cm['turbulent'] = 0.1 * c_over_d
        cs['turbulent'] = 0.6 * h_over_d

    return cm, cs


def calc_constants(asm_obj):
    """Calculate and store constants for CTD mixing parameters so I
    don't have to recalculate them at every step"""

    c = {}
    c['Re_bnds'] = ctd_ff.calculate_Re_bounds(asm_obj)
    c['eddy'], c['swirl'] = calculate_laminar_turbulent_params(asm_obj)
    return c
