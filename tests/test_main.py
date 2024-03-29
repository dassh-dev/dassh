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
date: 2022-12-19
author: matz
Test container for general execution
"""
########################################################################
import os
import numpy as np
import pytest
import dassh
from .conftest import execute_dassh


def test_1conservation_simple_1(testdir, wdir_setup):
    """Test that an adiabatic, single-assembly sweep results in energy
    being conserved; power delivered only to pins"""
    inpath = os.path.join(testdir, 'test_inputs', 'input_conservation-1.txt')
    outpath = os.path.join(testdir, 'test_results', 'conservation-1')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    execute_dassh([path_to_tmp_infile, '--save_reactor'])

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


def test_conv_approx_interior(testdir, wdir_setup):
    """Test convection approximation for duct wall connection gives
    similar results to the original implementation"""
    r_obj = {}
    for x in ['on', 'off']:
        inpath = os.path.join(testdir, 'test_inputs',
                              f'input_conv_approx_{x}.txt')
        outpath = os.path.join(testdir, 'test_results', 'conv_approx', x)
        path_to_tmp_infile = wdir_setup(inpath, outpath)
        execute_dassh([path_to_tmp_infile, '--save_reactor'])
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


def test_conv_approx_byp(testdir, wdir_setup):
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
        path_to_tmp_infile = wdir_setup(inpath, outpath)
        execute_dassh([path_to_tmp_infile, '--save_reactor'])

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
    assert max_rel_diff < 0.003

    # Bypass subchannel differences
    diff_byp = (r_obj['on'].assemblies[0].rodded.temp['coolant_byp']
                - r_obj['off'].assemblies[0].rodded.temp['coolant_byp'])
    print('Max (+) bypass temp diff: ', np.max(diff_byp))
    print('Max (-) bypass temp diff: ', np.min(diff_byp))

    overall_diff = 150.0
    max_rel_diff = np.max(np.abs(diff_byp)) / overall_diff
    print('Max (abs) byp diff / dT:', max_rel_diff)
    assert max_rel_diff < 0.003


def test_conv_approx_interior_shield(testdir, wdir_setup):
    """Test convection approximation for duct wall connection gives
    similar results to the original implementation

    Note: inter-asm gap model is no_flow; both cases use the same axial
    step size because the step size change """
    r_obj = {}
    for x in ['on', 'off']:
        inpath = os.path.join(testdir, 'test_inputs',
                              f'input_conv_approx_shield_{x}.txt')
        outpath = os.path.join(testdir, 'test_results', 'conv_approx', x)
        path_to_tmp_infile = wdir_setup(inpath, outpath)
        execute_dassh([path_to_tmp_infile, '--save_reactor'])
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

    overall_diff = 4e3 / 0.02 / 1275
    max_rel_diff = np.max(np.abs(diff)) / overall_diff
    print('Max (abs) diff / dT:', max_rel_diff)
    assert max_rel_diff < 0.0025


def test_ebal_with_ur(testdir, wdir_setup):
    """Test that energy balance is achieved with unrodded assemblies"""
    inpath = os.path.join(testdir, 'test_inputs', 'input_ebal_w_unrodded.txt')
    outpath = os.path.join(testdir, 'test_results', 'ebal_w_unrodded')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    execute_dassh([path_to_tmp_infile, '--save_reactor'])

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


def test_dasshpower_exec(testdir, wdir_setup):
    """Test energy conservation in 'dassh_power' execution"""
    inpath = os.path.join(
        testdir,
        'test_inputs',
        'input_seven_asm_dasshpower_exec.txt')
    outpath = os.path.join(
        testdir,
        'test_results',
        'dasshpower_exec')
    path_to_tmp_infile = wdir_setup(inpath, outpath)
    args = [path_to_tmp_infile, '--save_reactor']
    execute_dassh(args, entrypoint="dassh_power")

    # Check that energy was conserved
    # 1. Total power stored in DASSH Reactor object
    r = dassh.reactor.load(os.path.join(outpath, 'dassh_reactor.pkl'))
    assert r.total_power == pytest.approx(1e7)

    # 2. Sum of powers in all DASSH Assembly objects
    assert sum(a.total_power for a in r.assemblies) == pytest.approx(1e7)

    # 3. Sum of pin powers written to output CSV
    with open(os.path.join(outpath, 'total_pin_power.csv'), 'r') as f:
        pin_power = np.loadtxt(f, delimiter=',')
    assert np.sum(pin_power[1:]) == pytest.approx(1e7)
