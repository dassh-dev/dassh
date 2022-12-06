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
date: 2022-12-06
author: matz
Test various aspects of the DASSH pressure drop calculations
"""
########################################################################
# NOTE: These are organized here because each test can use a variety
# of DASSH modules and it seems better to keep them here rather than
# distribute them throughout the DASSH unit tests.


import os
import numpy as np
import dassh


########################################################################
# RODDED REGION PRESSURE DROP TESTS
########################################################################


def test_rodded_reg_dp(c_shield_rr_params):
    """Test pressure drop calculation in RR object"""
    input, mat = c_shield_rr_params
    mat['coolant'] = dassh.Material('sodium')
    fr = 0.50
    rr = dassh.region_rodded.make_rr_asm(input, 'testboi', mat, fr)
    T_in = 623.15
    rr._init_static_correlated_params(T_in)
    dz = 0.01
    dp = 0.0
    ff = []
    rho = []
    vel = []
    for i in range(50):
        T = T_in + i
        rr._update_coolant_int_params(T)
        dp += rr.calculate_pressure_drop(dz)
        ff.append(rr.coolant_int_params['ff'])
        rho.append(rr.coolant.density)
        vel.append(rr.coolant_int_params['vel'])

    # Calculate the answer using analytical Darcy Weisbach - should be
    # pretty dang close because the "answer" calculation is the same,
    # just with an "average" friction factor and material properties
    ff_avg = np.average(ff)
    v_avg = np.average(vel)
    rho_avg = np.average(rho)
    dH = rr.bundle_params['de']
    ans = ff_avg * (dz * 50) * rho_avg * v_avg**2 / dH / 2
    print('dp_rr:', dp)
    print('dp_ans:', ans)
    diff = dp - ans
    print('diff:', diff)
    rdiff = diff / ans
    print('rel diff:', rdiff)
    assert np.abs(rdiff) < 0.0001


def test_pressure_drop_w_spacer_grid(testdir):
    """Test a sweep with both friction and spacer grid losses"""
    inpath = os.path.join(testdir, 'test_inputs', 'input_single_spacer.txt')
    outpath = os.path.join(testdir, 'test_results', 'single_spacer')
    inp = dassh.DASSH_Input(inpath)
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    r.temperature_sweep()
    r.save()

    # Check results using Reactor object attributes
    dh = r.assemblies[0].rodded.bundle_params['de']
    vel = r.assemblies[0].rodded.coolant_int_params['vel']
    rho = r.assemblies[0].rodded.coolant.density
    dz = r.core_length
    # Friction
    ff = r.assemblies[0].rodded.coolant_int_params['ff']
    ans = ff * dz * rho * vel**2 / dh / 2.0
    res = r.assemblies[0].rodded._pressure_drop['friction']
    diff = ans - res
    rdiff = diff / ans
    if abs(rdiff) > 1e-9:
        print('Friction')
        print('--------')
        print('Ans: ', ans)
        print('Res: ', res)
        print('Rel diff: ', rdiff)
    assert abs(rdiff) < 1e-9
    # Spacer
    Re = r.assemblies[0].rodded.coolant_int_params['Re']
    solidity = 0.6957 - 162.8 * r.assemblies[0].rodded.d['pin-pin']
    loss_coeff = solidity**2 * np.interp(
        Re,
        dassh.correlations.grid_rehme.Cv[:, 0],
        dassh.correlations.grid_rehme.Cv[:, 1])
    ans = loss_coeff * rho * vel**2 / 2.0
    ans *= 4  # four spacer grids
    res = r.assemblies[0].rodded._pressure_drop['spacer_grid']
    diff = ans - res
    rdiff = diff / ans
    if abs(rdiff) > 1e-9:
        print('Spacer Grid')
        print('-----------')
        print('Ans: ', ans)
        print('Res: ', res)
        print('Rel diff: ', rdiff)
    assert abs(rdiff) < 1e-9


def test_pressure_drop_output_table(testdir):
    """Test output table generation"""
    outpath = os.path.join(testdir, 'test_results', 'single_spacer')
    r = dassh.reactor.load(os.path.join(outpath, 'dassh_reactor.pkl'))
    r.postprocess()
    with open(os.path.join(outpath, 'dassh.out'), 'r') as f:
        out = f.read()
    to_find = """Asm.        Name        Loc.       Total    Friction  SpacerGrid    Region 1
----------------------------------------------------------------------------
   1      driver     ( 1, 1)  2.4332E-01  1.8604E-01  5.7281E-02  2.4332E-01"""
    assert to_find in out


########################################################################
# UNRODDED REGION PRESSURE DROP TESTS
########################################################################


def test_unrodded_reg_dp(c_shield_rr_params):
    """Test pressure drop calculation in UR object"""
    input, mat = c_shield_rr_params
    mat['coolant'] = dassh.Material('sodium')
    fr = 0.50
    # Make unrodded region; manually set UR params and sweep
    input['use_low_fidelity_model'] = True
    ur = dassh.region_unrodded.make_ur_asm('testboi', input, mat, fr)
    T_in = 623.15
    ur._init_static_correlated_params(T_in)
    dz = 0.01
    dp_ur = 0.0
    ff = []
    rho = []
    vel = []
    for i in range(50):
        T = T_in + i
        ur._update_coolant_params(T)
        dp_ur += ur.calculate_pressure_drop(dz)
        ff.append(ur.coolant_params['ff'])
        rho.append(ur.coolant.density)
        vel.append(ur._rr_equiv.coolant_int_params['vel'])

    # Calculate the answer using analytical Darcy Weisbach - should be
    # pretty dang close because the "answer" calculation is the same,
    # just with an "average" friction factor and material properties
    ff_avg = np.average(ff)
    v_avg = np.average(vel)
    rho_avg = np.average(rho)
    dH = ur._rr_equiv.bundle_params['de']
    ans = ff_avg * (dz * 50) * rho_avg * v_avg**2 / dH / 2
    print('dp_ur:', dp_ur)
    print('dp_ans:', ans)
    diff = dp_ur - ans
    print('diff:', diff)
    rdiff = diff / ans
    print('rel diff:', rdiff)
    assert np.abs(rdiff) < 0.0001


def test_dp_agreement_between_unrodded_rodded_regs(c_shield_rr_params):
    """Test that the pressure drop calculation gives the same result
    in RR and UR objects

    Notes
    -----
    In this test, the friction factor in the UR object should be the
    same as that in the RR object. Therefore, the two should have the
    same pressure drop.

    """
    input, mat = c_shield_rr_params
    mat['coolant'] = dassh.Material('sodium')  # get dynamic proeprties
    fr = 0.50
    T_in = 623.15
    dz = 0.01

    # Make rodded region
    rr = dassh.region_rodded.make_rr_asm(input, 'dummy', mat, fr)
    rr._init_static_correlated_params(T_in)

    # Make unrodded region; manually set UR params
    input['use_low_fidelity_model'] = True
    ur = dassh.region_unrodded.make_ur_asm('testboi', input, mat, fr)
    ur._init_static_correlated_params(T_in)

    dp_rr = 0.0
    dp_ur = 0.0
    for i in range(50):
        T = T_in + i
        rr._update_coolant_int_params(T)
        ur._update_coolant_params(T)
        dp_rr += rr.calculate_pressure_drop(dz)
        dp_ur += ur.calculate_pressure_drop(dz)

    print('dp_rr:', dp_rr)
    print('dp_ur:', dp_ur)
    diff = dp_rr - dp_ur
    print(diff)
    assert np.abs(diff) < 1e-8


def test_dp_agreement_between_unrodded_rodded_equiv_regs(testdir):
    """Test that the RR equivalent UR returns the same pressure drop

    Notes
    -----
    This test is similar to the previous in that the UR object is using
    a friction factor obtained from an RR object; therefore, the pressure
    drop difference between the UR and RR should be low.

    """
    # Get answer for comparison
    path_ans = os.path.join(
        testdir, 'test_data', 'test_single_asm', 'dassh_reactor.pkl')
    if os.path.exists(path_ans):
        r_ans = dassh.reactor.load(path_ans)
    else:
        inpath = os.path.join(testdir, 'test_inputs', 'input_single_asm.txt')
        outpath = os.path.join(testdir, 'test_results', 'test_single_asm')
        inp = dassh.DASSH_Input(inpath)
        r_ans = dassh.Reactor(inp, path=outpath, write_output=True)
        r_ans.temperature_sweep()
    ans = np.zeros(4)
    for i in range(len(r_ans.assemblies[0].region)):
        ans[i] = r_ans.assemblies[0].region[i].pressure_drop
    ans[-1] = r_ans.assemblies[0].pressure_drop

    # Get result to compare
    inpath = os.path.join(testdir, 'test_inputs', 'input_single_asm_lf.txt')
    outpath = os.path.join(testdir, 'test_results', 'test_single_asm_lf')
    inp = dassh.DASSH_Input(inpath)
    r_res = dassh.Reactor(inp, path=outpath, write_output=True)
    r_res.temperature_sweep()
    res = np.zeros(4)
    for i in range(len(r_res.assemblies[0].region)):
        res[i] = r_res.assemblies[0].region[i].pressure_drop
    res[-1] = r_res.assemblies[0].pressure_drop

    # Compare them
    diff = (res - ans) / ans
    assert np.max(np.abs(diff)) < 1e-4
