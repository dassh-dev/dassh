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
date: 2022-10-26
author: matz
comment: Unit tests for hot spot analysis methods
"""
########################################################################
import os
import pytest
import numpy as np
import dassh
from dassh import hotspot
from .test_reactor import cleanup


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
    assert res.shape == (2, 3)
    res = res[:, -1]
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
    # the 0-sigma delta temps by the statistical subfactors. In the
    # table, the fuel conductivity 3-sigma uncertainty is instead
    # calculated using the nominal fuel delta temp.
    # ans = 2523  # As given in Table V; this is WRONG!
    ans = 2526  # As calculated using the correct values.
    res = hotspot.calculate_temps(T_in, dT, hcf)
    assert res.shape == (1, 5)
    res = res[:, -1]
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
    fpath = os.path.join(testdir, 'test_data', 'hcf_input_clad.csv')
    tmp, expr = hotspot._read_hcf_table(fpath)
    res = hotspot._evaluate_hcf_expr(tmp, expr, dT)
    assert np.allclose(res['direct'], ans_direct)
    assert np.allclose(res['statistical'], ans_stat)


def test_read_and_evaluate(testdir):
    """Test that hotspot subfactors are properly read and 2-sigma
    temperatures are correctly calculated"""
    T_in = 623.15
    dT = np.array([[151, 12, 4], [148, 14, 5]])
    fpath = os.path.join(testdir, 'test_data', 'hcf_input_clad.csv')
    hcf, expr = hotspot._read_hcf_table(fpath)
    hcf = hotspot._evaluate_hcf_expr(hcf, expr, dT)
    two_sig_temps = hotspot.calculate_temps(T_in, dT, hcf)
    assert two_sig_temps.shape == (2, 3)
    two_sig_temps = two_sig_temps[:, -1]
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


def test_clad_subfactors_for_fuel_calc(testdir, caplog):
    """Test that code captures error when working HCF for clad
    calculation are used for fuel calculation (not enough cols)"""
    # Read input
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'x_hcf_ncols-2.txt'))
    # Run setup method - should pass
    test = dassh.hotspot._setup_postprocess(inp)
    # Now see what happens when you try to read the table
    with pytest.raises(SystemExit):
        sf, expr = dassh.hotspot._read_hcf_table(
            test['fuel']['fuel_cl']['subfactors'], cols_needed=7)
    assert 'Incorrect number of columns in HCF table' in caplog.text


def test_csv_invalid_expr(testdir, caplog):
    """Catch errors in user-prepared HCF CSV - invalid expression"""
    # Changed one of the "dT" variables to just be "T"
    fpath = os.path.join(testdir, 'test_inputs', 'x_hcf_expr.csv')
    with pytest.raises(SystemExit):
        hotspot._read_hcf_table(fpath)
    msg = 'ERROR: Invalid expression! '
    msg += 'Found: "1 + (3 / T) * np.sqrt(0.002304 '
    assert msg in caplog.text


def test_load_builtin_subfactors_from_input(testdir):
    """Test that input can handle user input for builtin subfactors"""
    # Read input
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'input_builtin_hotspot.txt'))
    # Run setup method - should pass
    test = dassh.hotspot._setup_postprocess(inp)
    # Now make sure you can read the subfactors - should pass
    sf, expr = dassh.hotspot._read_hcf_table(
        test['fuel']['clad_mw']['subfactors'])
    sf, expr = dassh.hotspot._read_hcf_table(
        test['fuel']['fuel_cl']['subfactors'])


def test_read_all_builtin_subfactors(testdir):
    """Test that all builtin subfactor tables can be read"""
    path_to_builtins = os.path.abspath(
        os.path.join(testdir, '..', 'dassh', 'data'))
    fnames = ('hcf_crbr_blanket_clad_mw.csv',
              'hcf_crbr_fuel_clad_mw.csv',
              'hcf_ebrii_markv_fuel_cl.csv',
              'hcf_fftf_clad_mw.csv',
              'hcf_fftf_fuel_cl.csv')
    for f in fnames:
        sf, expr = dassh.hotspot._read_hcf_table(
            os.path.join(path_to_builtins, f))


def test_rx_hotspot_analysis_and_table_gen(testdir):
    """Test hotspot analysis and output table generation
    from mock Reactor object"""
    # Read input and create Reactor object
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir, 'test_results', 'silly_hotspot')
    cleanup(outpath)
    inp = dassh.DASSH_Input(
        os.path.join(inpath, 'input_silly_hotspot.txt'))
    r = dassh.Reactor(inp, path=outpath, write_output=True)

    # Assign fake "_peak" data to assemblies
    # Items: Pin ID, Height, Power, T_cool, T_clad_od,
    #        T_clad_mw, T_clad_id, T_fuel_od, T_fuel_cl
    pin_data = [0.0, 3.0, 0.0, 787.0, 795.4, 805.0, 815.2, 815.2, 1000]
    keys = ['clad_od', 'clad_mw', 'clad_id', 'fuel_od', 'fuel_cl']
    for i in range(len(r.assemblies)):
        r.assemblies[i]._peak = {'pin': {}}
        for j in range(len(keys)):
            # tmp = np.array(pin_data)
            # tmp[3:] *= np.random.uniform(low=0.995, high=1.005)
            # tmp = list(tmp)
            r.assemblies[i]._peak['pin'][keys[j]] = \
                [pin_data[j + 4], j + 4, pin_data]

    # Run the analysis calculation and check the result
    hotspot_results = hotspot.analyze(r)
    two_sig_temps, asm_ids = hotspot_results
    # *** check the result ***
    ans_clad_mw = np.array([830.53, 839.56, 849.78])
    for row in two_sig_temps['clad_mw']:
        assert np.all(np.abs(row - ans_clad_mw) < 0.1)
    ans_fuel_cl = np.array([830.53, 839.56, 849.78, 860.65, 860.65, 1060.52])
    for row in two_sig_temps['fuel_cl']:
        assert np.all(np.abs(row - ans_fuel_cl) < 0.1)

    # Generate output tables - this is the code used in the
    # DASSH Reactor object.
    out = ''
    include = [('clad', 'mw'), ('fuel', 'cl')]
    if 'clad_od' in hotspot_results[0].keys():
        # Put clad OD at the beginning
        include.insert(0, ('clad', 'od'))
    if 'clad_id' in hotspot_results[0].keys():
        # Put clad ID right before fuel CL
        include.insert(-1, ('clad', 'id'))
    if 'fuel_od' in hotspot_results[0].keys():
        # Put fuel OD right before fuel CL
        include.insert(-1, ('fuel', 'od'))
    if any(['pin' in a._peak.keys() for a in r.assemblies]):
        for k in include:
            peak_pin = dassh.table.PeakPinTempTable(k[0], k[1])
            out += peak_pin.generate(r, hotspot_results)

    # *** check the result ***
    refpath = os.path.join(testdir, 'test_data', 'ref_hotspot_datatable.txt')
    with open(refpath, 'r') as f:
        ref = f.read()
    if out != ref:  # Write to disk so you can look into it
        with open(os.path.join(outpath, 'table_test.txt'), 'w') as f:
            f.writelines(out)
    assert out == ref
