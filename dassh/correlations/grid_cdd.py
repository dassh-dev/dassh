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
date: 2022-10-31
author: matz
Grid spacer pressure drop via Cigarini and Dalle Donne correlation
"""
########################################################################


_DEFAULT_COEFFS = [3.5, 73.14, -0.264, 2.79e10, -2.79, 2.0, 2.0]


def calc_loss_coeff(Re_b, solidity, c=_DEFAULT_COEFFS):
    """Calculate the pressure loss coefficient from a grid spacer
    using the Cigarini and Dalle Donne correlation (1988)

    Parameters
    ----------
    Re_b : float
        Bundle-average Reynolds number
    solidity : float
        Ratio of the cross-sectional area of the grid spacer to the
        coolant flow area without the grid
    c (optional) : list
        List of floats containing the coefficients for evaluating
        the CDD correlation

    Returns
    -------
    float
        Loss coefficient for the grid spacer

    """
    Cv = c[0] + c[1] * Re_b**c[2] + c[3] * Re_b**c[4]
    eps_raised_c6 = solidity**c[6]
    Cs = Cv * eps_raised_c6
    if Cs > c[5]:
        Cs = c[5]
    return Cs


def _DEFAULT_SOLIDITY(gap_between_pins):
    """Use the CDD emprical relation for solidity as a function of
    pin pitch and diameter based on "practical grid design"

    Parameters
    ----------
    gap_between_pins : float
        Gap between pins in bundle (= pin_pitch - pin_diameter) [m]

    Returns
    -------
    float
        Grid spacer solidity based on empirical relationship

    """
    return 0.6957 - 162.8 * gap_between_pins
