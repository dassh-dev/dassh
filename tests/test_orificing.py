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
date: 2022-01-19
author: matz
Unit tests for orificing optimization execution
"""
########################################################################
import os
import numpy as np
import subprocess
import dassh


def test_orificing_fuel_single_timestep(testdir, wdir_setup):
    """Test orificing optimization against hand calculated result"""
    datapath = os.path.join(testdir, 'test_data', 'orificing-1')
    inpath = os.path.join(datapath, 'input.txt')
    outpath = os.path.join(testdir, 'test_results', 'orificing-1')
    path_to_tmp_infile = wdir_setup(inpath, outpath)

    # Link other directories to skip DASSH calculation
    # for dir in ('cccc', '_power', '_parametric'):  # , '_iter1', '_iter2'):
    for dir in ('cccc', '_power', '_parametric', '_iter1', '_iter2'):
        dassh.utils._symlink(os.path.join(datapath, dir),
                             os.path.join(outpath, dir))

    # Run DASSH
    return_code = subprocess.call(['dassh', path_to_tmp_infile])
    assert return_code == 0

    # Import results
    resfile = os.path.join(outpath, 'orificing_result_assembly.csv')
    with open(resfile, 'r') as f:
        res = f.read()
    res = res.splitlines()
    tmp = [[], [], [], [], []]
    for l in res:
        ll = l.split(',')
        tmp[0].append(ll[0])   # assembly ID
        tmp[1].append(ll[4])   # orifice group
        tmp[2].append(ll[5])   # flow rate
        tmp[3].append(ll[8])   # bulk coolant temp
        tmp[4].append(ll[-1])  # peak fuel temp
    res = np.array(tmp, dtype=float).T

    # Check bulk coolant temperature
    res_t_bulk = np.sum(res[:, 2] * res[:, 3]) / np.sum(res[:, 2])
    res_dt_bulk = res_t_bulk - 623.15
    assert abs(res_dt_bulk - 150.0) / 150 < 1e-4

    # Check peak fuel temperature
    group_max = np.zeros(3)
    for g in range(3):
        group_max[g] = np.max(res[res[:, 1] == g, 4])
    avg = np.average(group_max)
    diff = group_max - avg
    reldiff = np.abs(diff / (avg - 623.15))
    assert np.all(reldiff < 0.001)
    assert abs(avg - 869.75) < 0.02


def test_orificing_dp_limit_abort(testdir, wdir_setup):
    """Ensure optimization aborts if pressure drop limit incurred"""
    datapath = os.path.join(testdir, 'test_data', 'orificing-1')
    inpath = os.path.join(datapath, 'input.txt')
    outpath = os.path.join(testdir, 'test_results', 'orificing-2')
    path_to_tmp_infile = wdir_setup(inpath, outpath)

    # Modify temporary input file to insert pressure drop constraint
    with open(path_to_tmp_infile, 'r') as f:
        inp = f.read()
    tag = inp.find('[Orificing]')
    tag = inp.find('\n', tag)
    inp = inp[:tag] + '\n    pressure_drop_limit = 0.75' + inp[tag:]
    with open(path_to_tmp_infile, 'w') as f:
        f.write(inp)

    # Link other directories to skip DASSH calculation
    for dir in ('cccc', '_power', '_parametric', '_iter1', '_iter2'):
        dassh.utils._symlink(os.path.join(datapath, dir),
                             os.path.join(outpath, dir))

    # Run DASSH
    return_code = subprocess.call(['dassh', path_to_tmp_infile])
    assert return_code == 0

    # Check message in log file
    msg = ('Breaking optimization iteration due to '
           'incurred pressure drop limit')
    with open(os.path.join(outpath, 'dassh.log'), 'r') as f:
        log = f.read()
    assert msg in log


def test_orificing_dp_limit_error(testdir, wdir_setup):
    """Ensure optimization aborts if pressure drop limit incurred in
    multiple groups (this means that the pressure drop limit is too
    strict)"""
    datapath = os.path.join(testdir, 'test_data', 'orificing-1')
    inpath = os.path.join(datapath, 'input.txt')
    outpath = os.path.join(testdir, 'test_results', 'orificing-3')
    path_to_tmp_infile = wdir_setup(inpath, outpath)

    # Modify temporary input file to insert pressure drop constraint
    with open(path_to_tmp_infile, 'r') as f:
        inp = f.read()
    tag = inp.find('[Orificing]')
    tag = inp.find('\n', tag)
    inp = inp[:tag] + '\n    pressure_drop_limit = 0.5' + inp[tag:]
    with open(path_to_tmp_infile, 'w') as f:
        f.write(inp)

    # Link other directories to skip DASSH calculation
    for dir in ('cccc', '_power', '_parametric', '_iter1', '_iter2'):
        dassh.utils._symlink(os.path.join(datapath, dir),
                             os.path.join(outpath, dir))

    # Run DASSH
    return_code = subprocess.call(['dassh', path_to_tmp_infile])
    assert return_code == 1

    # Check message in log file
    msg = 'Multiple groups constrained by pressure drop limit'
    with open(os.path.join(outpath, 'dassh.log'), 'r') as f:
        log = f.read()
    assert msg in log


def test_orificing_peak_coolant(testdir, wdir_setup):
    """Test orificing optimization for peak coolant temperature;
    full DASSH execution"""
    datapath = os.path.join(testdir, 'test_data', 'orificing-1')
    infile = 'input_orificing_peak_cool.txt'
    inpath = os.path.join(testdir, 'test_inputs', infile)
    outpath = os.path.join(testdir, 'test_results', 'orificing-4')
    path_to_tmp_infile = wdir_setup(inpath, outpath)

    # Link other directories to skip DASSH calculation
    dassh.utils._symlink(os.path.join(datapath, 'cccc'),
                         os.path.join(outpath, 'cccc'))

    # Run DASSH
    return_code = subprocess.call(['dassh', path_to_tmp_infile])
    assert return_code == 0

    # Import results
    resfile = os.path.join(outpath, 'orificing_result_assembly.csv')
    with open(resfile, 'r') as f:
        res = f.read()
    res = res.splitlines()
    tmp = [[], [], [], [], []]
    for l in res:
        ll = l.split(',')
        tmp[0].append(ll[0])   # assembly ID
        tmp[1].append(ll[4])   # orifice group
        tmp[2].append(ll[5])   # flow rate
        tmp[3].append(ll[8])   # bulk coolant temp
        tmp[4].append(ll[10])  # peak coolant temp
    res = np.array(tmp, dtype=float).T

    # Check bulk coolant temperature
    res_t_bulk = np.sum(res[:, 2] * res[:, 3]) / np.sum(res[:, 2])
    res_dt_bulk = res_t_bulk - 623.15
    assert abs(res_dt_bulk - 150.0) / 150 < 1e-4

    # Check peak coolant temperature
    group_max = np.zeros(3)
    for g in range(3):
        group_max[g] = np.max(res[res[:, 1] == g, -1])
    avg = np.average(group_max)
    diff = group_max - avg
    reldiff = np.abs(diff / (avg - 623.15))
    assert np.all(reldiff < 0.005)
    assert abs(avg - 817.5) < 0.1


def test_regrouping(testdir, wdir_setup, caplog):
    """Test that regrouping method properly identifies assembly too hot
    for Group 2 and moves it to Group 1, showing improvement"""
    datapath = os.path.join(testdir, 'test_data', 'orifice_regrouping')
    infile = 'input_orifice_regrouping.txt'
    inpath = os.path.join(testdir, 'test_inputs', infile)
    outpath = os.path.join(testdir, 'test_results', 'orifice_regrouping')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    dassh.utils._symlink(os.path.join(datapath, '_parametric'),
                         os.path.join(outpath, '_parametric'))
    dassh.utils._symlink(os.path.join(datapath, 'pin_power.csv'),
                         os.path.join(outpath, 'pin_power.csv'))

    # Set up DASSH Orificing object and all of the attributes it needs to
    # be able to regroup.
    dassh_logger = dassh.logged_class.init_root_logger(outpath, 'dassh')
    dassh_input = dassh.DASSH_Input(path_to_tmp_infile)
    orifice_obj = dassh.orificing.Orificing(dassh_input, dassh_logger)
    with dassh.logged_class.LoggingContext(40):
        orifice_obj.group_by_power()
    orifice_obj._parametric = {}
    with dassh.logged_class.LoggingContext(40):
        orifice_obj.run_parametric()  # With recycled results

    # Check initial grouping
    initial_grouping = np.ones(19, dtype=int)
    initial_grouping[1:7] = 0
    assert np.allclose(orifice_obj.group_data[:, 2], initial_grouping)

    # Don't need to run an iteration: use data stored in the datapath
    # Import "iteration 1" data
    i1_datapath = os.path.join(datapath, 'iter1_data_regrouping.csv')
    iter1_data = np.loadtxt(i1_datapath, delimiter=',')

    # Do a regrouping on it. This should result in moving Assembly 1
    # from Group 2 to Group 1
    orifice_obj.regroup(iter1_data, verbose=True)
    print(orifice_obj.group_data[:, 2])

    # Check result
    ans = initial_grouping.copy()
    ans[0] = 0
    assert np.allclose(orifice_obj.group_data[:, 2], ans)

    # Check logs
    msg = 'Moved Assembly 1 from Group 2 to Group 1'
    assert msg in caplog.text


def test_regrouping_tolerance(testdir, wdir_setup, caplog):
    """Test that regrouping method does not regroup asm when their
    temperatures fall within the tolerance relative to the group"""
    datapath = os.path.join(testdir, 'test_data', 'orifice_regrouping')
    infile = 'input_orifice_regrouping.txt'
    inpath = os.path.join(testdir, 'test_inputs', infile)
    outpath = os.path.join(testdir, 'test_results', 'orifice_regrouping')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    dassh.utils._symlink(os.path.join(datapath, '_parametric'),
                         os.path.join(outpath, '_parametric'))
    dassh.utils._symlink(os.path.join(datapath, 'pin_power.csv'),
                         os.path.join(outpath, 'pin_power.csv'))

    # Set up DASSH Orificing object and all of the attributes it needs to
    # be able to regroup.
    dassh_logger = dassh.logged_class.init_root_logger(outpath, 'dassh')
    dassh_input = dassh.DASSH_Input(path_to_tmp_infile)
    orifice_obj = dassh.orificing.Orificing(dassh_input, dassh_logger)
    with dassh.logged_class.LoggingContext(40):
        orifice_obj.group_by_power()
    orifice_obj._parametric = {}
    with dassh.logged_class.LoggingContext(40):
        orifice_obj.run_parametric()  # With recycled results

    # Check initial grouping
    initial_grouping = np.ones(19, dtype=int)
    initial_grouping[1:7] = 0
    assert np.allclose(orifice_obj.group_data[:, 2], initial_grouping)

    # Don't need to run an iteration: use data stored in the datapath
    # Import "iteration 1" data
    i1_datapath = os.path.join(datapath, 'iter1_data_no_regrouping.csv')
    iter1_data = np.loadtxt(i1_datapath, delimiter=',')

    # Do a regrouping on it. No values should change because the
    # max/avg ratio in Group 2 is within the given tolerance (~1.049)
    orifice_obj.regroup(iter1_data, verbose=True)
    assert np.allclose(orifice_obj.group_data[:, 2], initial_grouping)


def test_regrouping_multiple_asm(testdir, wdir_setup, caplog):
    """Test that regrouping method properly regroups multiple
    assemblies if necessary"""
    datapath = os.path.join(testdir, 'test_data', 'orifice_regrouping')
    infile = 'input_orifice_regrouping.txt'
    inpath = os.path.join(testdir, 'test_inputs', infile)
    outpath = os.path.join(testdir, 'test_results', 'orifice_regrouping')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    dassh.utils._symlink(os.path.join(datapath, '_parametric'),
                         os.path.join(outpath, '_parametric'))
    dassh.utils._symlink(os.path.join(datapath, 'pin_power.csv'),
                         os.path.join(outpath, 'pin_power.csv'))

    # Set up DASSH Orificing object and all of the attributes it needs to
    # be able to regroup.
    dassh_logger = dassh.logged_class.init_root_logger(outpath, 'dassh')
    dassh_input = dassh.DASSH_Input(path_to_tmp_infile)
    orifice_obj = dassh.orificing.Orificing(dassh_input, dassh_logger)

    with dassh.logged_class.LoggingContext(40):
        orifice_obj.group_by_power()
    orifice_obj._parametric = {}
    with dassh.logged_class.LoggingContext(40):
        orifice_obj.run_parametric()  # With recycled results

    # Check initial grouping
    initial_grouping = np.ones(19, dtype=int)
    initial_grouping[1:7] = 0
    assert np.allclose(orifice_obj.group_data[:, 2], initial_grouping)

    # Don't need to run an iteration: use data stored in the datapath
    # Import "iteration 1" data
    i1_datapath = os.path.join(datapath, 'iter1_data_regrouping.csv')
    iter1_data = np.loadtxt(i1_datapath, delimiter=',')
    # Make it so that another assembly in Group 2 has hecka high temps
    iter1_data[-1, -1] = 1100.0

    # Do a regrouping on it. This should result in moving Assemblies
    # 1 and 19 from Group 2 to Group 1
    orifice_obj.regroup(iter1_data, verbose=True)

    # Check result
    ans = initial_grouping.copy()
    ans[[0, 18]] = 0
    assert np.allclose(orifice_obj.group_data[:, 2], ans)

    # Check logs
    msg = 'Moved Assembly 1 from Group 2 to Group 1'
    assert msg in caplog.text
    msg = 'Moved Assembly 19 from Group 2 to Group 1'
    assert msg in caplog.text
