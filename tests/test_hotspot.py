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
date: 2022-08-23
author: matz
comment: Unit tests for hot spot analysis methods
"""
########################################################################
import os
import pytest
import numpy as np
from dassh import hotspot


def test_two_sigma_clad_temp():
    """Test 2-sigma clad temperature calculation based on LMR hot
    spot subfactors from FRA-TM-152 (See Table II on page 23 and
    Table III on page 24)"""
    hcf = {}
    hcf['direct'] = np.array([
        [[1.000, 1.000, 1.000],   # Measurement
         [1.060, 1.060, 1.060],   # Rx physics models
         [1.020, 1.020, 1.020],   # Control rod banking
         [1.020, 1.000, 1.000],   # Inter-asm flow dist
         [1.030, 1.006, 1.000],   # Intra-asm flow dist
         [1.000, 1.140, 1.140],   # Pellet clad eccentricity
         [1.000, 1.800, 0.800],   # Clad circumferential temp dist
         [1.000, 1.000, 1.050]],  # Clad cond/thickness
        [[1.000, 1.000, 1.000],   # ...these are the same, but the
         [1.060, 1.060, 1.060],   # code expects a subarray for each
         [1.020, 1.020, 1.020],   # assembly
         [1.020, 1.000, 1.000],
         [1.030, 1.006, 1.000],
         [1.000, 1.140, 1.140],
         [1.000, 1.800, 0.800],
         [1.000, 1.000, 1.050]]])
    hcf['statistical'] = np.array([
        [[1.052, 1.052, 1.052],   # Fissile fuel dist
         [1.020, 1.020, 1.020],   # Nuclear data
         [1.140, 1.000, 1.000],   # Balance of plant
         [1.000, 1.000, 1.000],   # Loop temp imbalance
         [1.050, 1.000, 1.000],   # Inter-asm flow dist
         [1.058, 1.005, 1.000],   # Intra-asm flow dist
         [1.010, 1.000, 1.000],   # Subchannel flow area
         [1.000, 1.000, 1.000],   # Wire wrap orientation
         [1.013, 1.000, 1.000],   # Coolant properties
         [1.000, 1.120, 1.000],   # Film HT coefficient
         [1.000, 1.174, 1.174]],  # Pellet clad eccentricity
        [[1.052, 1.052, 1.052],   # ...these are the same, but the
         [1.020, 1.020, 1.020],   # code expects a subarray for each
         [1.140, 1.000, 1.000],   # assembly
         [1.000, 1.000, 1.000],
         [1.050, 1.000, 1.000],
         [1.058, 1.005, 1.000],
         [1.010, 1.000, 1.000],
         [1.000, 1.000, 1.000],
         [1.013, 1.000, 1.000],
         [1.000, 1.120, 1.000],
         [1.000, 1.174, 1.174]]])
    T_in = 850
    dT = np.array([[309, 7, 6],
                   [309, 7, 6]])
    ans = 1263  # sig fig to "ones" place
    res = hotspot.calculate_temps(T_in, dT, hcf)
    msg = f"Ans (F): {ans}\nRes (F): {res}"
    assert np.all(np.abs(ans - res) < 1.0), msg


def test_two_sigma_fuel_temp():
    """Test 2-sigma fuel temperature calculation based on LMR hot
    spot subfactors from FRA-TM-152 (See Table IV on page 27 and
    Table V on page 30)"""
    hcf = {}
    hcf['direct'] = np.array([[
        [1.000, 1.000, 1.000, 1.000, 1.000],    # Measurement
        [1.060, 1.060, 1.060, 1.060, 1.060],    # Rx physics models
        [1.020, 1.020, 1.020, 1.020, 1.020],    # Control rod banking
        [1.020, 1.000, 1.000, 1.000, 1.000],    # Inter-asm flow dist
        [1.030, 1.006, 1.000, 1.000, 1.000],    # Intra-asm flow dist
        [1.000, 1.000, 1.000, 1.000, 1.000],    # Pellet clad eccentricity
        [1.000, 1.000, 1.000, 1.000, 1.000],    # Clad circ. temp. dist.
        [1.000, 1.000, 1.050, 1.000, 1.000]]])  # Clad cond., thickness
    hcf['statistical'] = np.array([[
        [1.052, 1.052, 1.052, 1.052, 1.052],    # Fissile fuel dist.
        [1.020, 1.020, 1.020, 1.020, 1.020],    # Nuclear data
        [1.140, 1.000, 1.000, 1.000, 1.000],    # Balance of plant
        [1.000, 1.000, 1.000, 1.000, 1.000],    # Loop temp. imbalance
        [1.050, 1.000, 1.000, 1.000, 1.000],    # Inter-asm flow dist
        [1.058, 1.005, 1.000, 1.000, 1.000],    # Intra-asm flow
        [1.010, 1.000, 1.000, 1.000, 1.000],    # Subchannel flow area
        [1.000, 1.000, 1.000, 1.000, 1.000],    # Wire wrap orientation
        [1.013, 1.000, 1.000, 1.000, 1.000],    # Coolant properties
        [1.000, 1.120, 1.000, 1.000, 1.000],    # Film HT coeff
        [1.000, 1.000, 1.000, 1.000, 1.000],    # Pellet clad eccentricity
        [1.000, 1.000, 1.000, 1.480, 1.000],    # Gap conductance
        [1.000, 1.000, 1.000, 1.000, 1.100]]])  # Fuel conductivity ***
    T_in = 850
    dT = np.array([[160, 25, 43, 265, 928]])
    # NOTE: The answer given in the Table V (2523 F) is incorrect.
    # The 3-sigma uncertainties should be obtained by multiplying
    # the 0-sigma delta temps by the statistical subfactors. The
    # fuel conductivity 3-sigma uncertainty is instead calculated
    # using the nominal fuel delta temp.
    ans = 2523  # As given in Table V; this is WRONG!
    ans = 2526  # As calculated using the correct values.
    res = hotspot.calculate_temps(T_in, dT, hcf)
    msg = f"Ans (F): {ans}\nRes (F): {res}"
    assert abs(ans - res) < 1.0, msg


def test_read_hcf_csv(testdir):
    """Test that code can read HCF CSV and evaluate expressions"""
    dT = np.array([[151, 12, 4], [148, 14, 5]])
    ans_direct = np.array([
        [[1.02, 1, 1],
         [1.03, 1.006, 1],
         [1.02, 1.02, 1.02],
         [1.02, 1.02, 1.02]],
        [[1.02, 1, 1],
         [1.03, 1.006, 1],
         [1.02, 1.02, 1.02],
         [1.02, 1.02, 1.02]]])
    ans_stat = np.array([
        [[1.21356415, 1, 1],
         [1.059, 1.016, 1],
         [1.0272259, 1, 1],
         [1.01, 1, 1],
         [1.019, 1, 1],
         [1, 1.12, 1],
         [1.017, 1, 1],
         [1.058, 1.005, 1],
         [1.07, 1.07, 1.07],
         [1.01, 1.01, 1.01],
         [1.052, 1.052, 1.052],
         [1, 1, 1],
         [1, 1, 1]],
        [[1.21702902, 1, 1],
         [1.059, 1.016, 1],
         [1.02777778, 1, 1],
         [1.01, 1, 1],
         [1.019, 1, 1],
         [1, 1.12, 1],
         [1.017, 1, 1],
         [1.058, 1.005, 1],
         [1.07, 1.07, 1.07],
         [1.01, 1.01, 1.01],
         [1.052, 1.052, 1.052],
         [1, 1, 1],
         [1, 1, 1]]])
    fpath = os.path.join(testdir, 'test_data', 'hcf_input_test.csv')
    tmp, expr = hotspot._read_hcf_table(fpath)
    res = hotspot._evaluate_hcf_expr(tmp, expr, dT)
    assert np.allclose(res['direct'], ans_direct)
    assert np.allclose(res['statistical'], ans_stat)


def test_read_and_evaluate(testdir):
    """Test that hotspot subfactors are properly read and 2-sigma
    temperatures are correctly calculated"""
    T_in = 623.15
    dT = np.array([[151, 12, 4], [148, 14, 5]])
    fpath = os.path.join(testdir, 'test_data', 'hcf_input_test.csv')
    hcf, expr = hotspot._read_hcf_table(fpath)
    hcf = hotspot._evaluate_hcf_expr(hcf, expr, dT)
    two_sig_temps = hotspot.calculate_temps(T_in, dT, hcf)
    ans = np.array([832.662166, 832.381719])
    assert np.allclose(two_sig_temps, ans)


def test_csv_incorrect_header(testdir, caplog):
    """Catch errors in user-prepared HCF CSV - incorrect header row"""
    # Changed one of the headers: "Film" --> "FilmLOL"
    fpath = os.path.join(testdir, 'test_inputs', 'x_hcf_header.csv')
    with pytest.raises(SystemExit):
        hotspot._read_hcf_table(fpath)
    msg = 'ERROR: Incorrect header row in HCF table'
    assert msg in caplog.text


def test_csv_incorrect_ncol(testdir, caplog):
    """Catch errors in user-prepared HCF CSV - incorrect num cols"""
    # Deleted one of the necessary columns from a working version
    fpath = os.path.join(testdir, 'test_inputs', 'x_hcf_ncols.csv')
    with pytest.raises(SystemExit):
        hotspot._read_hcf_table(fpath)
    msg = 'ERROR: Incorrect number of columns in HCF table'
    assert msg in caplog.text


def test_csv_invalid_expr(testdir, caplog):
    """Catch errors in user-prepared HCF CSV - invalid expression"""
    # Changed one of the "dT" variables to just be "T"
    fpath = os.path.join(testdir, 'test_inputs', 'x_hcf_expr.csv')
    with pytest.raises(SystemExit):
        hotspot._read_hcf_table(fpath)
    msg = 'ERROR: Invalid expression! '
    msg += 'Found: "1 + (3 / T) * np.sqrt(0.002304 '
    assert msg in caplog.text
