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
date: 2022-07-06
author: matz
Cheng-Todreas correlation for conduction shape factor (1984)
"""
########################################################################


def calculate_shape_factor(rr):
    """Calculate conduction shape factor

    Parameters
    ----------
    rr : DASSH RoddedRegion object

    Returns
    -------
    float
        Scalar by which conduction between subchannels is enhanced

    """
    P_over_D = rr.pin_pitch / rr.pin_diameter
    s_over_D = (rr.pin_pitch - rr.pin_diameter) / rr.pin_diameter
    return 0.66 * P_over_D * s_over_D**-0.3
