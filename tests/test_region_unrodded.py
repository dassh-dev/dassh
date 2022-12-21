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
date: 2022-11-30
author: matz
Test the behavior and attributes of unrodded DASSH Region instances
"""
########################################################################
import copy
import os
import dassh
import numpy as np
import pytest


def test_simple_unrodded_reg_instantiation(c_lrefl_simple):
    """Test that the unrodded region has all the right stuffs"""

    assert c_lrefl_simple.vf['coolant'] == 0.25
    assert c_lrefl_simple.vf['struct'] == 0.75
    assert c_lrefl_simple.duct_ftf[1] == 0.116
    assert len(c_lrefl_simple.temp['coolant_int']) == 1
    assert c_lrefl_simple.temp['duct_mw'].shape == (1, 6)
    # If it don't fail, it pass
    c_lrefl_simple.temp['coolant_int'] *= 623.15
    c_lrefl_simple._update_coolant_params(623.15)


def test_ur_reg_instantiation_fancy(testdir):
    """Make sure a fancy unrodded region can be instantiated"""
    inp = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_ur_conv_factor.txt'),
        empty4c=True)
    mat = {'coolant': dassh.Material('sodium'),
           'duct': dassh.Material('ht9')}

    # Test fully unrodded assembly
    ur1 = dassh.region_unrodded.make_ur_asm(
        'testboi', inp.data['Assembly']['fuel'], mat, 1.0)
    print(ur1.mratio)
    print(ur1._mratio)
    print(inp.data['Assembly']['fuel']['convection_factor'])
    assert ur1.mratio is not None
    assert ur1.mratio != 1.0

    # Test default in unrodded axial regions
    ur2 = dassh.region_unrodded.make_ur_axialregion(
        inp.data['Assembly']['control'], 'empty_cr', mat, 1.0)
    assert ur2.mratio == 1.0

    # Test nondefault in unrodded axial regions
    ur2 = dassh.region_unrodded.make_ur_axialregion(
        inp.data['Assembly']['control'], 'upper_cr', mat, 1.0)
    assert ur2.mratio == 0.8


def test_unrodded_reg_clone_shallow(c_lrefl_simple):
    """Test that region attributes are properly copied"""
    clone = c_lrefl_simple.clone(15.0)
    non_matches = []
    # Shallow copies
    for attr in ['z', 'duct_ftf', 'duct_thickness', 'duct_perim',
                 'vf', 'area', 'total_area', '_params', 'x_pts']:
        id_clone = id(getattr(clone, attr))
        id_original = id(getattr(c_lrefl_simple, attr))
        if id_clone == id_original:  # They should be the same
            continue
        else:
            non_matches.append(attr)
            print(attr, id_clone, id_original)
    assert len(non_matches) == 0


def test_unrodded_reg_clone_deep(c_lrefl_simple):
    """Test that region attributes are properly copied"""
    clone = c_lrefl_simple.clone(15.0)
    non_matches = []
    # Shallow copies
    for attr in ['temp', 'flow_rate', 'coolant_params']:
        id_clone = id(getattr(clone, attr))
        id_original = id(getattr(c_lrefl_simple, attr))
        if id_clone != id_original:  # They should be different
            continue
        else:
            non_matches.append(attr)
            print(attr, id_clone, id_original)
    assert len(non_matches) == 0


def test_simple_unrodded_reg_zero_power(c_lrefl_simple):
    """Test that no power temp calc returns no change"""
    in_temp = c_lrefl_simple.temp['coolant_int']
    t_gap = np.ones(6) * c_lrefl_simple.avg_duct_mw_temp
    c_lrefl_simple.calculate(
        0.1, {'refl': 0.0}, t_gap, 0.0, adiabatic_duct=True)
    assert c_lrefl_simple.temp['coolant_int'] == pytest.approx(in_temp)
    assert c_lrefl_simple.pressure_drop > 0.0


def test_simple_unrodded_reg_none_power(c_lrefl_simple):
    """Test that giving power=None returns no change in temps"""
    in_temp = c_lrefl_simple.temp['coolant_int']
    t_gap = np.ones(6) * c_lrefl_simple.avg_duct_mw_temp
    c_lrefl_simple.calculate(
        0.1, {'refl': None}, t_gap, 0.0, adiabatic_duct=True)
    assert c_lrefl_simple.temp['coolant_int'] == pytest.approx(in_temp)
    assert c_lrefl_simple.pressure_drop > 0.0


def test_simple_unrodded_reg_qmcdt(c_lrefl_simple):
    """Test that simple coolant calc returns proper result"""
    # Set up some stuff
    c_lrefl_simple.temp['coolant_int'] *= 623.15
    c_lrefl_simple.temp['duct_mw'] *= 623.15
    c_lrefl_simple.temp['duct_surf'] *= 623.15
    in_temp = copy.deepcopy(c_lrefl_simple.temp['coolant_int'])
    power = 10000.0
    dz = 0.1
    qlin = power / dz

    # Calculate dT and estimate Q
    c_lrefl_simple._update_coolant_params(623.15)
    dT = c_lrefl_simple._calc_coolant_temp(dz, {'refl': qlin})
    q_est = (c_lrefl_simple.coolant.heat_capacity *
             c_lrefl_simple.flow_rate * dT)

    print('m =', c_lrefl_simple.flow_rate)
    print('Cp =', c_lrefl_simple.coolant.heat_capacity)
    # print('dT =', c_lrefl_simple.temp['coolant_int'] - in_temp)
    print('dT =', dT)
    print('q (est) = ', q_est)
    assert power == pytest.approx(q_est)
    assert c_lrefl_simple.temp['coolant_int'] == in_temp


def test_simple_unrodded_reg_duct(c_lrefl_simple):
    """Test that simple homog duct calc returns proper result"""
    # Set up some stuff
    c_lrefl_simple.temp['coolant_int'] *= 633.15
    # Calculate dT and estimate Q
    gap_temp = np.ones(6) * 623.15
    gap_htc = np.ones(6) * 7.5e4  # made this up
    # print(c_lrefl_simple.temp['duct_mw'][0])
    c_lrefl_simple._update_coolant_params(633.15)
    c_lrefl_simple._calc_duct_temp(gap_temp, gap_htc)
    print('inner', c_lrefl_simple.temp['duct_surf'][0, 0])
    print('midwall', c_lrefl_simple.temp['duct_mw'][0])
    print('outer', c_lrefl_simple.temp['duct_surf'][0, 1])
    assert all([623.15 < x < 633.15 for x in
                c_lrefl_simple.temp['duct_mw'][0]])
    # Coolant temp is greater than inner duct surface temp, which
    # is greater than duct midwall temp
    assert all([633.15
                > c_lrefl_simple.temp['duct_surf'][0, 0, i]
                > c_lrefl_simple.temp['duct_mw'][0, i]
                for i in range(6)])
    # Duct midwall temp is greater than outer duct surface temp,
    # which is greater than gap coolant temp
    assert all([c_lrefl_simple.temp['duct_mw'][0, i]
                > c_lrefl_simple.temp['duct_surf'][0, 1, i]
                > 623.15
                for i in range(6)])


def test_mnh_ur_ebal_adiabatic(shield_ur_mnh):
    """Test multi-node homogeneous unrodded region energy balance
    with adiabatic duct wall"""
    n_steps = 100
    dz = 0.001
    power = {'refl': 100.0}
    gap_temp = np.arange(625, 775, 25)  # [625, 650, 675, 700, 725, 750]
    fake_htc = np.ones(6) * 2e4
    for i in range(n_steps):
        shield_ur_mnh.calculate(dz, power, gap_temp, fake_htc,
                                ebal=True, adiabatic_duct=True)
    assert np.sum(shield_ur_mnh.ebal['duct']) == 0.0
    # Check power added real quick
    tot_power_added = n_steps * dz * power['refl']
    assert shield_ur_mnh.ebal['power'] - tot_power_added <= 1e-12
    print('ENERGY ADDED (W): ', shield_ur_mnh.ebal['power'])
    print('ENERGY FROM DUCT (W)', np.sum(shield_ur_mnh.ebal['duct']))
    total = (np.sum(shield_ur_mnh.ebal['duct'])
             + shield_ur_mnh.ebal['power'])
    print('TOTAL ENERGY INPUT (W)', total)
    e_temp_rise = (shield_ur_mnh.flow_rate
                   * shield_ur_mnh.coolant.heat_capacity
                   * (shield_ur_mnh.avg_coolant_temp - 623.15))
    print('ENERGY COOLANT DT (W):', e_temp_rise)
    bal = total - e_temp_rise
    print('DIFFERENCE (W)', bal)
    assert bal <= 1e-7


def test_mnh_ur_ebal(shield_ur_mnh):
    """Test multi-node homogeneous unrodded region energy balance"""
    dz = dassh.region_unrodded.calculate_min_dz(
        shield_ur_mnh, 623.15, 773.15)
    n_steps = 100
    dz = 0.001  # less than dz calculated above
    power = {'refl': 0.0}
    gap_temp = np.arange(625, 775, 25)  # [625, 650, 675, 700, 725, 750]
    fake_htc = np.ones(6) * 2e4
    for i in range(n_steps):
        shield_ur_mnh.calculate(dz, power, gap_temp, fake_htc, ebal=True)
    # Check power added real quick
    tot_power_added = n_steps * dz * power['refl']
    assert shield_ur_mnh.ebal['power'] - tot_power_added <= 1e-12
    print('ENERGY ADDED (W): ', shield_ur_mnh.ebal['power'])
    print('ENERGY FROM DUCT (W):', np.sum(shield_ur_mnh.ebal['duct']))
    total = (np.sum(shield_ur_mnh.ebal['duct'])
             + shield_ur_mnh.ebal['power'])
    print('TOTAL ENERGY INPUT (W):', total)
    e_temp_rise = (shield_ur_mnh.flow_rate
                   * shield_ur_mnh.coolant.heat_capacity
                   * (shield_ur_mnh.avg_coolant_temp - 623.15))
    print('ENERGY COOLANT DT (W):', e_temp_rise)
    bal = total - e_temp_rise
    print('DIFFERENCE (W):', bal)
    assert bal <= 1e-7


def test_ur_asm_pressure_drop(c_shield_rr_params):
    """Test that the pressure drop calculation gives the same result
    in RR and UR objects"""
    input, mat = c_shield_rr_params
    mat['coolant'] = dassh.Material('sodium')  # get dynamic proeprties
    fr = 0.50

    # Make rodded region
    rr = dassh.region_rodded.make_rr_asm(input, 'dummy', mat, fr)
    rr._init_static_correlated_params(623.15)

    # Make unrodded region; manually set UR params
    input['use_low_fidelity_model'] = True
    input['convection_factor'] = 'calculate'
    ur = dassh.region_unrodded.make_ur_asm('testboi', input, mat, fr)
    ur._init_static_correlated_params(623.15)

    T_in = 623.15
    dz = 0.01
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


def test_ur_dp_rr_equiv(testdir):
    """Test that the RR equivalent UR returns the same pressure drop
    as a regular RR object"""
    # Get answer to compare with
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
    assert np.max(np.abs(diff)) < 1e-3
