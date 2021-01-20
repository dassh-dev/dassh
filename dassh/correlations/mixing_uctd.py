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
date: 2020-06-12
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
        eddy = eddy['laminar']
        swirl = swirl['laminar']
    elif asm_obj.coolant_int_params['Re'] >= Re_bt:
        eddy = eddy['turbulent']
        swirl = swirl['turbulent']
    else:  # Transition regime; use intermittency factor
        x = calc_sc_intermittency_factors(asm_obj, Re_bl, Re_bt)
        eddy = (eddy['laminar'] + (eddy['turbulent']
                                   - eddy['laminar']) * x**(2 / 3.0))
        swirl = (swirl['laminar'] + (swirl['turbulent']
                                     - swirl['laminar']) * x**(2 / 3.0))
    return eddy * asm_obj.L[0][0], swirl


def calc_sc_intermittency_factors(asm_obj, Re_bL, Re_bT):
    """Calculate the intermittency factors for the interior and edge
    coolant subchannels; required to find the mixing parameters in
    the transition region

    Notes
    -----
    See Equations 10, 11, and 30-32 in the Cheng-Todreas 1986 paper

    """
    y = np.zeros(2)
    fs_L = uctd_fs.calculate_flow_split(asm_obj, 'laminar')
    fs_T = uctd_fs.calculate_flow_split(asm_obj, 'turbulent')
    for i in range(len(y)):
        Re_iL = (Re_bL * fs_L[i] * asm_obj.params['de'][i]
                 / asm_obj.bundle_params['de'])
        Re_iT = (Re_bT * fs_T[i] * asm_obj.params['de'][i]
                 / asm_obj.bundle_params['de'])
        y[i] = ((np.log10(asm_obj.params['Re'][i]) - np.log10(Re_iL))
                / (np.log10(Re_iT) - np.log10(Re_iL)))
    return y


def calc_constants(asm_obj):
    """Calculate and store constants for UCTD mixing parameters so
    I don't have to recalculate them at every step"""
    c = uctd_fs.calc_constants(asm_obj)
    c['eddy'], c['swirl'] = \
        ctd_mix.calculate_laminar_turbulent_params(asm_obj)
    return c
