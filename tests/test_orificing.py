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
date: 2021-11-02
author: matz
Unit tests for orificing optimization execution
"""
########################################################################
import os
import numpy as np
import dassh
import subprocess
# import pytest
# import shutil


def test_orificing_fuel_single_timestep(testdir, wdir_setup):
    """Test orificing optimization against hand calculated result"""
    datapath = os.path.join(testdir, 'test_data', 'orificing-1')
    inpath = os.path.join(datapath, 'input.txt')
    outpath = os.path.join(testdir, 'test_results', 'orificing-1')
    path_to_tmp_infile = wdir_setup(inpath, outpath)

    # Link other directories to skip DASSH calculation
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
        tmp[4].append(ll[11])  # peak fuel temp
    res = np.array(tmp, dtype=float).T

    # Check bulk coolant temperature
    res_t_bulk = np.sum(res[:, 2] * res[:, 3]) / np.sum(res[:, 2])
    res_dt_bulk = res_t_bulk - 623.15
    assert abs(res_dt_bulk - 150.0) / 150 < 1e-4

    # Check peak fuel temperature
    group_max = np.zeros(3)
    for g in range(3):
        group_max[g] = np.max(res[res[:, 1] == g, -1])
    avg = np.average(group_max)
    diff = group_max - avg
    reldiff = np.abs(diff / (avg - 623.15))
    assert np.all(reldiff < 0.001)


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
    inpath = os.path.join(testdir, 'test_inputs', 'input_orificing_peak_cool.txt')
    outpath = os.path.join(testdir, 'test_results', 'orificing-4')
    path_to_tmp_infile = wdir_setup(inpath, outpath)

    # Link other directories to skip DASSH calculation
    for dir in ('cccc', '_power'):
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
        tmp[4].append(ll[11])  # peak coolant temp
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
