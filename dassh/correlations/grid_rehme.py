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
date: 2022-11-01
author: matz
Grid spacer pressure drop via Rehme correlation
"""
########################################################################
import numpy as np


# Modified loss coefficient from Rehme paper - obtained visually
# from Figure 3 for triangular array. Meets the stated observation
# that for Re > 5e4, Cv between 6 and 7.
Cv = np.array([
    [4.00E+03, 13.0],
    [6.00E+03, 10.0],
    [8.00E+03, 9.0],
    [1.00E+04, 8.5],
    [2.00E+04, 7.6],
    [4.00E+04, 7.1],
    [6.00E+04, 6.8],
    [8.00E+04, 6.7],
    [1.00E+05, 6.5],
    [2.00E+05, 6.2]])


def calc_loss_coeff(Re_b, solidity, corr_coeff=None):
    """Calculate the loss coefficient using linear interpolation on
    array for modified loss coeff. vs. bundle Re

    Parameters
    ----------
    Re_b : float
        Bundle-average Reynolds number
    solidity : float
        Ratio of the grid projected cross section area to
        to unrestricted flow area (A_grid / A_flow)
    c : NoneType
        Unused input; including for continuity with
        equivalent CDD correlation function

    Returns
    -------
    Pressure loss coefficient due to grid spacer in rod bundle

    Notes
    -----
    From Rehme (1973) Fig. 3 for triangular array

    """
    Cv_interp = np.interp(Re_b, Cv[:, 0], Cv[:, 1])
    return Cv_interp * solidity**2