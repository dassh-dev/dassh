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
date: 2020-05-12
author: matz
Rehme friction factor correlation (1973)
"""
########################################################################
import numpy as np


# Application ranges of friction factor correlations
applicability = {}
applicability['P/D'] = np.array([1.1, 1.42])
applicability['H/D'] = np.array([8.0, 50.0])
applicability['Nr'] = np.array([7, 217])
applicability['regime'] = ['turbulent', 'transition']
applicability['Re'] = np.array([1000, 3e5])
applicability['bare rod'] = False


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
    pd = asm_obj.pin_pitch / asm_obj.pin_diameter
    # Geometry factor F
    F = np.sqrt(pd)
    F += (7.6 * pd**2 * (asm_obj.pin_diameter + asm_obj.wire_diameter)
          / asm_obj.wire_pitch)**2.16
    # friction factor f
    return ((np.sqrt(F) * 64 / asm_obj.coolant_int_params['Re']
             + 0.0816 * F**0.9335 / asm_obj.coolant_int_params['Re']**0.133)
            * (asm_obj.pin_diameter + asm_obj.wire_diameter)
            * asm_obj.n_pin * np.pi / asm_obj.bundle_params['wp'])
