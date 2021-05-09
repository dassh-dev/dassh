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

#
# def test_lowflow_byp_approximation(testdir):
#     """Test low flow approximation for duct wall connection gives
#     similar results to the original implementation"""
#     path = os.path.join(testdir, 'test_inputs', 'lowflow_duct_byp_approx')
#
#     # -----------------------------------------------------------------
#     # Case without approximation
#     path2dir = os.path.join(path, 'approx_off')
#     path2input = os.path.join(path2dir, 'input.txt')
#
#     # Remove the DASSH reactor object, if it exists
#     try:
#         os.remove(os.path.join(path2dir, 'dassh_reactor.pkl'))
#         os.remove(os.path.join(path2dir, 'temp_coolant_int.csv'))
#         os.remove(os.path.join(path2dir, 'temp_coolant_byp.csv'))
#     except FileNotFoundError:
#         pass
#
#     # Run DASSH!
#     return_code = subprocess.call(['dassh', path2input, '--save_reactor'])
#     assert return_code == 0
#
#     # Read results
#     temp_byp_off = np.loadtxt(
#         os.path.join(path2dir, 'temp_coolant_byp.csv'),
#         delimiter=',')
#     temp_int_off = np.loadtxt(
#         os.path.join(path2dir, 'temp_coolant_int.csv'),
#         delimiter=',')
#
#     # -----------------------------------------------------------------
#     # Case with approximation
#     path2dir = os.path.join(path, 'approx_on')
#     path2input = os.path.join(path2dir, 'input.txt')
#
#     # Remove the DASSH reactor object, if it exists
#     try:
#         os.remove(os.path.join(path2dir, 'dassh_reactor.pkl'))
#         os.remove(os.path.join(path2dir, 'temp_coolant_int.csv'))
#         os.remove(os.path.join(path2dir, 'temp_coolant_byp.csv'))
#     except FileNotFoundError:
#         pass
#
#     # Run DASSH!
#     return_code = subprocess.call(['dassh', path2input, '--save_reactor'])
#     assert return_code == 0
#
#     # Read results
#     temp_byp_on = np.loadtxt(
#         os.path.join(path2dir, 'temp_coolant_byp.csv'),
#         delimiter=',')
#     temp_int_on = np.loadtxt(
#         os.path.join(path2dir, 'temp_coolant_int.csv'),
#         delimiter=',')
#
#     # -----------------------------------------------------------------
#     # Average differences
#     r_on = dassh.reactor.load(
#         os.path.join(path, 'approx_on', 'dassh_reactor.pkl'))
#     r_off = dassh.reactor.load(
#         os.path.join(path, 'approx_off', 'dassh_reactor.pkl'))
#     diff_avg_int = (r_on.assemblies[0].avg_coolant_int_temp
#                     - r_off.assemblies[0].avg_coolant_int_temp)
#     diff_avg_byp = (r_on.assemblies[0].rodded.avg_coolant_byp_temp
#                     - r_off.assemblies[0].rodded.avg_coolant_byp_temp)
#
#     print('Avg. interior temp diff:', diff_avg_int)
#     print('Avg. bypass temp diff:', diff_avg_byp[0])
#     # assert np.abs(diff_avg_int) < 1.0
#     # assert np.abs(diff_avg) < 1.0
#
#     # Interior subchannel differences
#     diff_int = temp_int_on[-1, 4:] - temp_int_off[-1, 4:]
#     print('Max (+) interior temp diff: ', np.max(diff_int))
#     print('Max (-) interior temp diff: ', np.min(diff_int))
#
#     overall_diff = 150.0
#     max_rel_diff = np.max(np.abs(diff_int)) / overall_diff
#     print('Max (abs) int diff / dT:', max_rel_diff)
#     assert max_rel_diff < 0.0025
#
#     # Bypass subchannel differences
#     diff_byp = temp_byp_on[-1, 5:] - temp_byp_off[-1, 5:]
#     print('Max (+) bypass temp diff: ', np.max(diff_byp))
#     print('Max (-) bypass temp diff: ', np.min(diff_byp))
#
#     overall_diff = 150.0
#     max_rel_diff = np.max(np.abs(diff_byp)) / overall_diff
#     print('Max (abs) byp diff / dT:', max_rel_diff)
#     assert max_rel_diff < 0.0025


#
# @pytest.mark.skip(reason='Does not work at the moment')
# def test_2conservation(testdir):
#     """Test that an adiabatic, single-assembly sweep results in energy
#     being conserved; power delivered to pins, duct, and coolant"""
#     path_to_input_dir = os.path.join(testdir, 'test_inputs',
#                                      'conservation-2')
#     path_to_input = os.path.join(path_to_input_dir, 'input.txt')
#
#     # Remove the DASSH reactor object, if it exists
#     try:
#         os.remove(os.path.join(path_to_input_dir, 'dassh_reactor.pkl'))
#     except FileNotFoundError:
#         pass
#
#     # Run DASSH!
#     return_code = subprocess.call(['dassh', path_to_input,
#                                    '--save_reactor'])
#     assert return_code == 0
#
#     # Check that energy was conserved
#     r = dassh.reactor.load(os.path.join(path_to_input_dir,
#                                         'dassh_reactor.pkl'))
#     dT_result = r.assemblies[0].avg_coolant_int_temp - r.inlet_temp
#     cp = r.assemblies[0].rodded.coolant.heat_capacity  # constant
#     m = r.assemblies[0].flow_rate
#     q_perfect = r.total_power
#     q_delivered = 0.0
#     for i in range(len(r.dz)):
#         p = r.assemblies[0].power.get_power(r.z[i + 1])
#         p = np.sum(p['pins']) + np.sum(p['duct']) + np.sum(p['cool'])
#         q_delivered += p * r.dz[i]
#     dT_perfect = q_perfect / m / cp
#     dT_answer = q_delivered / m / cp
#     print('q_perf / q_delivered', q_perfect / q_delivered)
#     print('dT perfect', dT_perfect)
#     print('dT answer', dT_answer)
#     print('dT result', dT_result)
#     assert dT_answer == pytest.approx(dT_result, 1e-6)

#
#
# def test_main_execution(testdir):
#     """Test that no exceptions are raised when main thing runs"""
#     # Clean up from previous runs
#     if 'dassh.log' in os.listdir('.'):
#         os.remove('dassh.log')
#     testdir_contents = os.listdir(os.path.join(testdir, 'xxxx'))
#     for f in ['varpow_MatPower.out', 'varpow.MonoExp.out',
#               'varpow_stdout.txt', 'VARPOW.out']:
#         if f in testdir_contents:
#             os.remove(os.path.join(testdir, 'xxxx', f))
#
#     # Run the new case
#     path_to_input = os.path.join(testdir, 'xxxx', 'xxxx.txt')
#     return_code = subprocess.call(['dassh', path_to_input])
#     assert return_code == 0
#

# def test_main_total_power(testdir):
#     """Test the total power produced in the main example"""
#
# def test_interasm_ht(testdir):
#     """Test that interassembly heat transfer behavior is as expected
#     in a small, seven-assembly core
#
#
#     Core map
#     --------
#         A3*      A2
#
#     A4*      A1*     A7
#
#         A5*     A6
#
#     """
#     cwd = os.getcwd()
#     os.chdir(os.path.join(testdir, 'test_inputs', 'interasm_ht'))
#     inp = dassh.DASSH_Input('input.txt')
#     print(inp.data['Setup']['Options']['dif3d_indexing'])
#     r = dassh.Reactor(inp, calc_power=False)
#     print([a.name for a in r.assemblies])
#     print([a.id for a in r.assemblies])
#     print([a.loc for a in r.assemblies])
#     print([a.dif3d_id for a in r.assemblies])
#     print([a.dif3d_loc for a in r.assemblies])
#     print('rerarrange')
#     poop = copy.deepcopy(r.assemblies)
#     poop.sort(key=lambda x: x.dif3d_id)
#     print([x.name for x in poop])
#     print([x.id for x in poop])
#     print([x.loc for x in poop])
#     print([x.dif3d_id for x in poop])
#     print([x.dif3d_loc for x in poop])
#
#     for i in range(len(r.power.power)):
#         print(i, np.sum(r.power.power[i]))
#
#     os.chdir(cwd)
#     assert 0
#     # with open('dassh_stdout.txt', 'w') as f:
#     #     subprocess.call(['dassh', 'input.txt'], stdout=f)
#
#     # Read in temperatures from DASSH
#     temps = np.genfromtxt('temp_duct_mw.csv',
#                           delimiter=',',
#                           filling_values=0)
#
#
#     os.chdir(cwd)
