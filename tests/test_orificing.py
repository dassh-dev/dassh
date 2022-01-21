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
date: 2022-01-20
author: matz
Unit tests for orificing optimization execution
"""
########################################################################
import os
import numpy as np
import subprocess
import pytest
import dassh


def test_initial_grouping(testdir):
    """Test that grouping algorithm properly assigns asm to groups"""
    infile = 'input_orifice_grouping.txt'
    dassh_input = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', infile),
        empty4c=True)
    orificing_obj = dassh.Orificing(dassh_input)

    # Manually specify the power to group by
    power = np.array([
        [0, 510000],
        [1, 750000],
        [2, 745000],
        [3, 730000],
        [4, 725000],
        [5, 735000],
        [6, 740000],
        [7, 508000],
        [8, 508000],
        [9, 506000],
        [10, 504000],
        [11, 502000],
        [12, 500000],
        [13, 498000],
        [14, 498000],
        [15, 500000],
        [16, 502000],
        [17, 504000],
        [18, 506000]
    ])
    ans = np.zeros((power.shape[0], 3))
    ans[:, :2] = power[np.argsort(power[:, 1])][::-1]
    ans[6:, 2] = 1
    result = orificing_obj._group(power)
    assert np.allclose(result, ans)


def test_grouping_fail(testdir, caplog):
    """Test that proper error message is raised when grouping
    algorithm cannot converge"""
    infile = 'input_orifice_grouping.txt'
    dassh_input = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', infile),
        empty4c=True)
    orificing_obj = dassh.Orificing(dassh_input)

    # Manually specify the power to group by - give it something
    # crazy that there's no way it can put into just two groups
    power = np.zeros((19, 2))
    power[:, 0] = np.arange(19)
    power[:, 1] = [1e3, 1e3, 1e3, 1e3, 1e3, 1e3,
                   1e4, 1e4, 1e4, 1e4, 1e4, 1e4,
                   1e5, 1e5, 1e5, 1e5, 1e5, 1e5,
                   1e6]

    # Try the grouping - it won't converge and will hit the
    # iteration limit
    with pytest.raises(SystemExit):
        orificing_obj._group(power)

    # Check the logs for the error message you expect
    msg = 'Grouping not converged; please adjust "group_cutoff"'  # ...
    assert msg in caplog.text


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


def test_regrouping1(testdir, wdir_setup, caplog):
    """Test that regrouping method properly identifies assembly too hot
    for Group 2 and moves it to Group 1, showing improvement"""
    datapath = os.path.join(testdir, 'test_data', 'orifice_regrouping')
    infile = 'input_orifice_regrouping.txt'
    inpath = os.path.join(testdir, 'test_inputs', infile)
    outpath = os.path.join(testdir, 'test_results', 'orifice_regrouping-1')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    dassh.utils._symlink(os.path.join(datapath, '_parametric'),
                         os.path.join(outpath, '_parametric'))
    dassh.utils._symlink(os.path.join(datapath, 'pin_power.csv'),
                         os.path.join(outpath, 'pin_power.csv'))

    # Run DASSH
    return_code = subprocess.call(['dassh', path_to_tmp_infile])
    assert return_code == 0

    # Check for outcomes in iteration data
    iter1_datapath = os.path.join(outpath, '_iter1', 'data.csv')
    iter1_data = np.loadtxt(iter1_datapath, delimiter=',')
    iter1_fr = iter1_data[:, 3]
    assert iter1_fr[0] != iter1_fr[1]  # Asm 1 in Group 2
    iter2_datapath = os.path.join(outpath, '_iter2', 'data.csv')
    iter2_data = np.loadtxt(iter2_datapath, delimiter=',')
    iter2_fr = iter2_data[:, 3]
    assert iter2_fr[0] == iter2_fr[1]  # Asm 1 now in Group 1

    # Check for outcomes in the logs
    with open(os.path.join(outpath, 'dassh.log'), 'r') as f:
        logfile = f.read()
    msg_to_find = 'Moved Assembly 1 from Group 2 to Group 1'
    assert msg_to_find in logfile


def test_regrouping_tolerance(testdir, wdir_setup, caplog):
    """Test that regrouping method does not regroup asm when their
    temperatures fall within the tolerance relative to the group"""
    datapath = os.path.join(testdir, 'test_data', 'orifice_regrouping')
    infile = 'input_orifice_regrouping.txt'
    inpath = os.path.join(testdir, 'test_inputs', infile)
    outpath = os.path.join(testdir, 'test_results', 'orifice_regrouping-2')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    dassh.utils._symlink(os.path.join(datapath, '_parametric'),
                         os.path.join(outpath, '_parametric'))
    dassh.utils._symlink(os.path.join(datapath, 'pin_power.csv'),
                         os.path.join(outpath, 'pin_power.csv'))

    # Link "fake" iteration 1 data - DASSH will "recycle" it,
    # skipping iteration 1 and that should lead it to skip
    # regrouping.
    os.makedirs(os.path.join(outpath, '_iter1'))
    dassh.utils._symlink(
        os.path.join(datapath, 'iter1_data_no_regrouping.csv'),
        os.path.join(outpath, '_iter1', 'data.csv'))

    # Run DASSH
    return_code = subprocess.call(['dassh', path_to_tmp_infile])
    assert return_code == 0

    # Check for outcomes in iteration data
    iter1_datapath = os.path.join(outpath, '_iter1', 'data.csv')
    iter1_data = np.loadtxt(iter1_datapath, delimiter=',')
    iter1_fr = iter1_data[:, 3]
    assert iter1_fr[0] != iter1_fr[1]  # Asm 1 in Group 2
    iter2_datapath = os.path.join(outpath, '_iter2', 'data.csv')
    iter2_data = np.loadtxt(iter2_datapath, delimiter=',')
    iter2_fr = iter2_data[:, 3]
    assert iter2_fr[0] != iter2_fr[1]  # Asm 1 still in Group 2

    # Check for outcomes in the logs
    with open(os.path.join(outpath, 'dassh.log'), 'r') as f:
        logfile = f.read()
    msg_to_find = 'Moved Assembly 1 from Group 2 to Group 1'
    assert msg_to_find not in logfile


def test_regrouping_multiple_asm(testdir, wdir_setup, caplog):
    """Test that regrouping method properly regroups multiple
    assemblies if necessary"""
    datapath = os.path.join(testdir, 'test_data', 'orifice_regrouping')
    infile = 'input_orifice_regrouping.txt'
    inpath = os.path.join(testdir, 'test_inputs', infile)
    outpath = os.path.join(testdir, 'test_results', 'orifice_regrouping-3')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    dassh.utils._symlink(os.path.join(datapath, '_parametric'),
                         os.path.join(outpath, '_parametric'))
    dassh.utils._symlink(os.path.join(datapath, 'pin_power.csv'),
                         os.path.join(outpath, 'pin_power.csv'))

    # Link "fake" iteration 1 data - DASSH will "recycle" it,
    # skipping iteration 1 and that should lead it to skip
    # regrouping.
    i1_datapath = os.path.join(datapath, 'iter1_data_regrouping.csv')
    iter1_data = np.loadtxt(i1_datapath, delimiter=',')
    # Make it so that another assembly in Group 2 has hecka high temps
    iter1_data[-1, -1] = 1100.0
    # Put fake data in working directory
    os.makedirs(os.path.join(outpath, '_iter1'))
    np.savetxt(
        os.path.join(outpath, '_iter1', 'data.csv'),
        iter1_data,
        delimiter=',')

    # Run DASSH
    return_code = subprocess.call(['dassh', path_to_tmp_infile])
    assert return_code == 0

    # Check for outcomes in iteration data
    iter1_datapath = os.path.join(outpath, '_iter1', 'data.csv')
    iter1_data = np.loadtxt(iter1_datapath, delimiter=',')
    iter1_fr = iter1_data[:, 3]
    assert iter1_fr[0] != iter1_fr[1]  # Asm 1 in Group 2
    iter2_datapath = os.path.join(outpath, '_iter2', 'data.csv')
    iter2_data = np.loadtxt(iter2_datapath, delimiter=',')
    iter2_fr = iter2_data[:, 3]
    assert iter2_fr[0] == iter2_fr[1]  # Asm 1 now in Group 1
    assert iter2_fr[-1] == iter2_fr[1]  # Asm 19 now also in Group 1

    # Check for outcomes in the logs
    with open(os.path.join(outpath, 'dassh.log'), 'r') as f:
        logfile = f.read()
    msg_to_find = 'Moved Assembly 1 from Group 2 to Group 1'
    assert msg_to_find in logfile
    msg_to_find = 'Moved Assembly 19 from Group 2 to Group 1'
    assert msg_to_find in logfile
