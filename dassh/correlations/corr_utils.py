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
date: 2021-03-09
author: matz
Correlation utility functinos
"""
########################################################################
import numpy as np


def calculate_wproj(corr, P, D, Dw):
    """Calculate area projection of wire wrap into subchannel"""
    wproj = np.zeros(3)
    wproj[2] = (np.pi * (D + Dw) * Dw / 6)
    if corr == 'mit':
        wproj[0] = np.pi * (P - 0.5 * D)**2 / 6 - np.pi * D**2 / 24
        wproj[1] = np.pi * (0.25 * (0.5 * D + Dw)**2 - 0.0625 * D**2)
    else:
        wproj[0] = np.pi * (D + Dw) * Dw / 6
        wproj[1] = np.pi * (D + Dw) * Dw / 4
    return wproj


def calculate_bare_rod_sc_area(corr, P, D, Dw, ep=None):
    """Calculate subchannel area without wire wrap"""
    a_bare = np.zeros(3)
    a_bare[0] = np.sqrt(3) * 0.25 * P**2 - 0.125 * np.pi * D**2
    if corr == 'mit':
        a_bare[1] = P * (0.5 * D + Dw) - 0.125 * np.pi * D**2
    else:
        assert ep is not None
        a_bare[1] = P * (ep - 0.5 * D) - np.pi * D**2 / 8
        a_bare[2] = (ep - 0.5 * D)**2 / np.sqrt(3) - np.pi * D**2 / 24
    return a_bare


def calculate_bare_rod_wp(P, D, ep):
    """Calculate subchannel wetted perimeter without wire wrap"""
    wp = np.zeros(3)
    wp[0] = np.pi * D / 2
    wp[1] = P + np.pi * D / 2
    wp[2] = np.pi * D / 6 + 2 * (ep - 0.5 * D) / np.sqrt(3)
    return wp
