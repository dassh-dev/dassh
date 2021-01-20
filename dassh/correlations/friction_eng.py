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
date: 2020-03-19
author: matz
Engel friction factor correlation (1979)
"""
########################################################################
import numpy as np


# Application ranges of friction factor correlations
applicability = {}
applicability['P/D'] = np.array([1.067, 1.082])
applicability['H/D'] = np.array([7.7, 8.3])
applicability['Nr'] = np.array([19, 61])
applicability['regime'] = ['turbulent', 'transition', 'laminar']
applicability['Re'] = np.array([50, 1e5])
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
    if asm_obj.coolant_int_params['Re'] < 400.0:
        f = 110 / asm_obj.coolant_int_params['Re']
    elif asm_obj.coolant_int_params['Re'] > 5000.0:
        f = 0.55 / asm_obj.coolant_int_params['Re']**0.25
    else:  # transition
        x = (asm_obj.coolant_int_params['Re'] - 400) / 4600
        f = (110 * np.sqrt(1 - x) / asm_obj.coolant_int_params['Re']
             + 0.55 * np.sqrt(x) / asm_obj.coolant_int_params['Re']**0.25)
    return f
