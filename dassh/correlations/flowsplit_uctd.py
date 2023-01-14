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
Upgraded Cheng-Todreas correlation for flow split (2018)
"""
########################################################################
from . import friction_uctd as fr_uctd
from . import flowsplit_ctd as fs_ctd


applicability = fr_uctd.applicability


def calculate_flow_split(asm_obj, regime=None):
    """Calculate the flow split into the different types of
    subchannels based on the Upgraded Cheng-Todreas model

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

    Notes
    -----
    This correlation differs from the original Cheng-Todreas flow
    split by the calculation of the subchannel friction factors.
    However, it relies on the hidden worker function in that module,
    which takes those friction factors as input.

    """
    try:
        Re_bnds = asm_obj.corr_constants['fs']['Re_bnds']
    except (KeyError, AttributeError):
        Re_bnds = fr_uctd.calculate_Re_bounds(asm_obj)

    Re_bundle = asm_obj.coolant_int_params['Re']
    if regime == 'laminar' or Re_bundle <= Re_bnds[0]:
        return asm_obj.corr_constants['fs']['fs']['laminar']
    elif regime == 'turbulent' or Re_bundle >= Re_bnds[1]:
        return asm_obj.corr_constants['fs']['fs']['turbulent']
    else:
        return fs_ctd._calc_transition_flowsplit(asm_obj, _lambda=7)


def calc_constants(asm_obj):
    """Calculate constants needed by the UCTD flowsplit calculation"""
    const = fr_uctd.calc_constants(asm_obj)
    del const['Cf_b']

    # Total subchannel area for each subchannel type
    const['na'] = [asm_obj.subchannel.n_sc['coolant']['interior']
                   * asm_obj.params['area'][0],
                   asm_obj.subchannel.n_sc['coolant']['edge']
                   * asm_obj.params['area'][1],
                   asm_obj.subchannel.n_sc['coolant']['corner']
                   * asm_obj.params['area'][2]]

    # REGIME RATIO CONSTANTS
    const['xr'] = fs_ctd._calc_regime_ratio_constants(asm_obj, const['Cf_sc'])

    # Laminar/turbulent: constant flow split!
    const['fs'] = fs_ctd._calc_constant_flowsplits(asm_obj, const)

    return const
