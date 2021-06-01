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
date: 2021-02-25
author: matz
Test container for general execution
"""
########################################################################
import os
import numpy as np
import dassh
import subprocess
import pytest
import shutil


def dassh_setup(infile_path, outpath):
    # Remove the DASSH reactor object, if it exists
    if os.path.exists(outpath):
        shutil.rmtree(outpath)
    os.makedirs(outpath, exist_ok=True)
    infile_name = os.path.split(infile_path)[1]
    shutil.copy(infile_path, os.path.join(outpath, infile_name))
    return os.path.join(outpath, infile_name)


def test_1conservation_simple_1(testdir):
    """Test that an adiabatic, single-assembly sweep results in energy
    being conserved; power delivered only to pins"""
    inpath = os.path.join(testdir, 'test_inputs', 'input_conservation-1.txt')
    outpath = os.path.join(testdir, 'test_results', 'conservation-1')
    path_to_tmp_infile = dassh_setup(inpath, outpath)
    return_code = subprocess.call(['dassh',
                                   path_to_tmp_infile,
                                   '--save_reactor'])
    assert return_code == 0

    # Check that energy was conserved
    r = dassh.reactor.load(os.path.join(outpath, 'dassh_reactor.pkl'))
    dT_result = r.assemblies[0].avg_coolant_int_temp - r.inlet_temp
    cp = r.assemblies[0].rodded.coolant.heat_capacity  # constant
    m = r.assemblies[0].flow_rate
    q_perfect = r.total_power
    q_delivered = np.sum(list(r.assemblies[0]._power_delivered.values()))
    dT_perfect = q_perfect / m / cp
    dT_answer = q_delivered / m / cp
    print(dT_perfect)
    print(dT_answer)
    print(dT_result)
    assert dT_answer == pytest.approx(dT_result, 1e-6)
    print(q_delivered)
    # print(r.assemblies[0].rodded.ebal)
    ebal_tot = (r.assemblies[0].rodded.ebal['power']
                + np.sum(r.assemblies[0].rodded.ebal['duct']))
    diff = ebal_tot - dT_result * cp * m
    assert diff < 1e-6


def test_conv_approx_interior(testdir):
    """Test convection approximation for duct wall connection gives
    similar results to the original implementation"""
    r_obj = {}
    for x in ['on', 'off']:
        inpath = os.path.join(testdir, 'test_inputs',
                              f'input_conv_approx_{x}.txt')
        outpath = os.path.join(testdir, 'test_results', 'conv_approx', x)
        path_to_tmp_infile = dassh_setup(inpath, outpath)
        return_code = subprocess.call(['dassh',
                                       path_to_tmp_infile,
                                       '--save_reactor'])
        assert return_code == 0
        r_obj[x] = dassh.reactor.load(
            os.path.join(outpath, 'dassh_reactor.pkl'))

    # -----------------------------------------------------------------
    # Average differences
    diff_avg_int = (r_obj['on'].assemblies[0].avg_coolant_int_temp
                    - r_obj['off'].assemblies[0].avg_coolant_int_temp)
    print('Avg. interior temp diff:', diff_avg_int)

    # Interior subchannel differences
    diff = (r_obj['on'].assemblies[-1].region[0].temp['coolant_int']
            - r_obj['off'].assemblies[-1].region[0].temp['coolant_int'])
    print('Max (+) interior temp diff: ', np.max(diff))
    print('Max (-) interior temp diff: ', np.min(diff))

    overall_diff = 150.0
    max_rel_diff = np.max(np.abs(diff)) / overall_diff
    print('Max (abs) diff / dT:', max_rel_diff)
    assert max_rel_diff < 0.0025


def test_conv_approx_byp(testdir):
    """Test low flow approximation for duct wall connection gives
    similar results to the original implementation"""
    # temp_int = {}
    # temp_byp = {}
    r_obj = {}
    for x in ['on', 'off']:
        inpath = os.path.join(testdir, 'test_inputs',
                              f'input_conv_approx_byp_{x}.txt')
        outpath = os.path.join(testdir, 'test_results',
                               'input_conv_approx_byp', x)
        path_to_tmp_infile = dassh_setup(inpath, outpath)
        return_code = subprocess.call(['dassh', path_to_tmp_infile,
                                       '--save_reactor'])
        assert return_code == 0
        # temp_int[x] = np.loadtxt(
        #     os.path.join(outpath, 'temp_coolant_int.csv'),
        #     delimiter=',')
        # temp_byp[x] = np.loadtxt(
        #     os.path.join(outpath, 'temp_coolant_int.csv'),
        #     delimiter=',')
        r_obj[x] = dassh.reactor.load(
            os.path.join(outpath, 'dassh_reactor.pkl'))

    # -----------------------------------------------------------------
    # Average differences
    diff_avg_int = (r_obj['on'].assemblies[0].avg_coolant_int_temp
                    - r_obj['off'].assemblies[0].avg_coolant_int_temp)
    diff_avg_byp = (r_obj['on'].assemblies[0].rodded.avg_coolant_byp_temp
                    - r_obj['off'].assemblies[0].rodded.avg_coolant_byp_temp)
    print('Avg. interior temp diff:', diff_avg_int)
    print('Avg. bypass temp diff:', diff_avg_byp[0])
    # assert np.abs(diff_avg_int) < 1.0
    # assert np.abs(diff_avg) < 1.0

    # Interior subchannel differences
    diff_int = (r_obj['on'].assemblies[0].rodded.temp['coolant_int']
                - r_obj['off'].assemblies[0].rodded.temp['coolant_int'])
    print('Max (+) interior temp diff: ', np.max(diff_int))
    print('Max (-) interior temp diff: ', np.min(diff_int))

    overall_diff = 150.0
    max_rel_diff = np.max(np.abs(diff_int)) / overall_diff
    print('Max (abs) int diff / dT:', max_rel_diff)
    assert max_rel_diff < 0.0025

    # Bypass subchannel differences
    diff_byp = (r_obj['on'].assemblies[0].rodded.temp['coolant_byp']
                - r_obj['off'].assemblies[0].rodded.temp['coolant_byp'])
    print('Max (+) bypass temp diff: ', np.max(diff_byp))
    print('Max (-) bypass temp diff: ', np.min(diff_byp))

    overall_diff = 150.0
    max_rel_diff = np.max(np.abs(diff_byp)) / overall_diff
    print('Max (abs) byp diff / dT:', max_rel_diff)
    assert max_rel_diff < 0.0025


def test_ebal_with_ur(testdir):
    """Test that energy balance is achieved with unrodded assemblies"""
    inpath = os.path.join(testdir, 'test_inputs', 'input_ebal_w_unrodded.txt')
    outpath = os.path.join(testdir, 'test_results', 'ebal_w_unrodded')
    path_to_tmp_infile = dassh_setup(inpath, outpath)
    return_code = subprocess.call(['dassh',
                                   path_to_tmp_infile,
                                   '--save_reactor'])
    assert return_code == 0

    # Check that energy was conserved
    r = dassh.reactor.load(os.path.join(outpath, 'dassh_reactor.pkl'))
    dt = np.zeros(8)
    q_in = np.zeros(8)
    q_duct = np.zeros(8)
    mfr = np.zeros(8)
    for i in range(len(r.assemblies)):
        dt[i] = r.assemblies[i].avg_coolant_int_temp - r.inlet_temp
        q_in[i] = sum(r.assemblies[i]._power_delivered.values())
        mfr[i] = r.assemblies[i].flow_rate
        q_duct[i] = np.sum(r.assemblies[i].active_region.ebal['duct'])

    dt[-1] = r.core.avg_coolant_gap_temp - r.inlet_temp
    mfr[-1] = r.core.gap_flow_rate
    q_duct[-1] = np.sum(r.core.ebal['asm'])
    cp = r.assemblies[0].active_region.coolant.heat_capacity  # constant
    q_dt = mfr * cp * dt
    assert np.abs(np.sum(q_in) - np.sum(q_dt)) < 2e-8
