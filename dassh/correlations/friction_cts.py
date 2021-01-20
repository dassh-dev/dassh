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
date: 2020-04-22
author: matz
Cheng-Todreas Simple correlations (1986)
"""
########################################################################
import numpy as np
from . import friction_ctd as ctd


# Application ranges of friction factor correlations
applicability = {}
applicability['P/D'] = np.array([1.025, 1.42])
applicability['H/D'] = np.array([8.0, 50.0])
applicability['Nr'] = np.array([19, 217])
applicability['regime'] = ['turbulent', 'transition', 'laminar']
applicability['Re'] = np.array([50, 1e6])
applicability['bare rod'] = False


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
    c['Re_bnds'] = ctd.calculate_Re_bounds(asm_obj)
    c['Cf_b'] = calculate_bundle_friction_factor_const(asm_obj)
    return c

########################################################################
# BUNDLE FRICTION FACTOR CONSTANT
########################################################################


def calculate_bundle_friction_factor_const(asm_obj):
    """Calculate the laminar and turbulent bundle-average friction
    factor coefficients using the Cheng-Todreas Simple correlation"""
    pd = asm_obj.pin_pitch / asm_obj.pin_diameter
    hd = asm_obj.wire_pitch / asm_obj.pin_diameter
    cfb = {}
    cfb['turbulent'] = (0.8063 - 0.9022 * np.log10(hd)
                        + 0.3526 * np.log10(hd)**2)
    cfb['turbulent'] *= pd**9.7 * hd**(1.78 - 2.0 * pd)
    cfb['laminar'] = (-974.6 + 1612.0 * pd - 598.5 * pd**2)
    cfb['laminar'] *= hd**(0.06 - 0.085 * pd)
    return cfb


########################################################################
# BUNDLE FRICTION FACTOR
########################################################################


def calculate_bundle_friction_factor(asm_obj):
    """Calculate the bundle-average friction factor with constants
    calculated from the Cheng-Todreas Simple correlation

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
    This is a wrapper function around ctd._calc_bundle_ff, which can
    be used to calculate the friction factors by both the CTD and CTS
    correlations.

    """
    try:
        cfb = asm_obj.corr_constants['ff']['Cf_b']
    except (KeyError, AttributeError):
        cfb = calculate_bundle_friction_factor_const(asm_obj)

    return ctd._calc_bundle_ff(asm_obj, cfb)
