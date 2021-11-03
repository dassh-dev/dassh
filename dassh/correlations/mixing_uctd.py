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
date: 2021-09-21
author: matz
Upgraded Cheng-Todreas Detailed mixing correlations: eddy
diffusivity and swirl velocity; these are the same as in the
1986 correlation but rely on the upgraded correlations (2018)
for friction factor and flow split.
"""
########################################################################
import numpy as np
from . import mixing_ctd as ctd_mix
from . import friction_uctd as uctd_ff
from . import flowsplit_uctd as uctd_fs


applicability = ctd_mix.applicability


def calculate_mixing_params(asm_obj):
    """Calculate the eddy diffusivity and swirl velocity based
    on the Cheng-Todreas 1986 correlation with flow regime boundaries
    and flow split parameters from the 2018 upgrade.

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
    To be tested?

    """
    # Get params in laminar and turbulent regimes - either access from
    # stored location or recalculate
    try:
        eddy = asm_obj.corr_constants['mix']['eddy']
        swirl = asm_obj.corr_constants['mix']['swirl']
    except (KeyError, AttributeError):
        eddy, swirl = ctd_mix.calculate_laminar_turbulent_params(asm_obj)

    # Calculate parameters based on subchannel Reynolds number
    try:
        Re_bl, Re_bt = asm_obj.corr_constants['mix']['Re_bnds']
    except (KeyError, AttributeError):
        Re_bl, Re_bt = uctd_ff.calculate_Re_bounds(asm_obj)

    # Calculate parameters based on subchannel Reynolds number
    Re_bl, Re_bt = uctd_ff.calculate_Re_bounds(asm_obj)
    if asm_obj.coolant_int_params['Re'] <= Re_bl:
        eddy_diffusivity = eddy['laminar']
        swirl_velocity = swirl['laminar']
    elif asm_obj.coolant_int_params['Re'] >= Re_bt:
        eddy_diffusivity = eddy['turbulent']
        swirl_velocity = swirl['turbulent']
    else:  # Transition regime; use intermittency factor
        x = calc_sc_intermittency_factors(asm_obj, Re_bl, Re_bt)
        eddy_diffusivity = eddy['turbulent'] - eddy['laminar']
        eddy_diffusivity *= x[0]**(2 / 3.0)
        eddy_diffusivity += eddy['laminar']
        swirl_velocity = swirl['turbulent'] - swirl['laminar']
        swirl_velocity *= x[1]**(2 / 3.0)
        swirl_velocity += swirl['laminar']
    return eddy_diffusivity * asm_obj.L[0][0], swirl_velocity


def calc_sc_intermittency_factors(asm_obj, Re_bL, Re_bT):
    """Calculate the intermittency factors for the interior and edge
    coolant subchannels; required to find the mixing parameters in
    the transition regime

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
    # Need CTD flowsplit: if already calculaing CTD flowsplit, just take it
    # from the values you calculated previously within the RoddedRegion obj.
    if asm_obj.corr_names['fs'] == 'uctd':
        try:
            fs = asm_obj.coolant_int_params['fs']
        except(KeyError, AttributeError):
            fs = uctd_fs.calculate_flow_split(asm_obj)
    else:
        fs = uctd_fs.calculate_flow_split(asm_obj)

    # Pull bundle average axial velocity and subchannel Re from object if they
    # exist and are not zero; otherwise, calculate here. The only time they'd
    # be zero is at the very beginning of the problem if they've not yet been
    # initialized for some reason
    try:
        v_bundle = asm_obj.coolant_int_params['vel']
        Re = asm_obj.coolant_int_params['Re_sc']
        if v_bundle == 0.0:
            v_bundle = (asm_obj.int_flow_rate / asm_obj.coolant.density
                        / asm_obj.bundle_params['area'])
            Re = (asm_obj.coolant.density * v_bundle * fs
                  * asm_obj.params['de'] / asm_obj.coolant.viscosity)
    except(KeyError, AttributeError):
        v_bundle = (asm_obj.int_flow_rate / asm_obj.coolant.density
                    / asm_obj.bundle_params['area'])
        Re = (asm_obj.coolant.density * v_bundle * fs
              * asm_obj.params['de'] / asm_obj.coolant.viscosity)

    # Get laminar/turbulent flow splits from correlation constants, or by
    # recalculating if necessary.
    try:
        fs_L = asm_obj.corr_constants['fs']['fs']['laminar']
        fs_T = asm_obj.corr_constants['fs']['fs']['turbulent']
    except (KeyError, AttributeError):
        fs_L = uctd_fs.calculate_flow_split(asm_obj, 'laminar')
        fs_T = uctd_fs.calculate_flow_split(asm_obj, 'turbulent')

    # Calculate subchannel laminar/turbulent factors and use these to get
    # the subchannel intermittency factors.
    Re_iL = (Re_bL * fs_L * asm_obj.params['de']
             / asm_obj.bundle_params['de'])
    Re_iT = (Re_bT * fs_T * asm_obj.params['de']
             / asm_obj.bundle_params['de'])
    y = np.log10(Re / Re_iL) / np.log10(Re_iT / Re_iL)
    y[y < 0] = 0.0  # Clip any negative values to zero!
    return y


def calc_constants(asm_obj):
    """Calculate and store constants for UCTD mixing parameters so
    I don't have to recalculate them at every step"""
    c = uctd_fs.calc_constants(asm_obj)
    c['eddy'], c['swirl'] = \
        ctd_mix.calculate_laminar_turbulent_params(asm_obj)
    return c
