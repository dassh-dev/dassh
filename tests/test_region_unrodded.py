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
        inp.data['Assembly']['fuel'], mat, 1.0)
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

    # Make unrodded region; manually set UR params
    input['use_low_fidelity_model'] = True
    input['convection_factor'] = 'calculate'
    ur = dassh.region_unrodded.make_ur_asm(input, mat, fr)

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
    """Test that the RR equivalent UR returns the same pressure drop"""
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


def test_ur_dp(testdir):
    """Test that the pressure drop calculation for the unrodded region
    is similar to that of the pin bundle when comparable parameters
    are used"""
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
    # Just want pressure drop per unit length of rod bundle region
    asm = r_ans.assemblies[0]
    ans = asm.rodded.pressure_drop
    ans /= asm.region_bnd[2] - asm.region_bnd[1]

    # Get result to compare
    inpath = os.path.join(testdir, 'test_inputs', 'input_single_asm_lf.txt')
    outpath = os.path.join(testdir, 'test_results', 'test_single_asm_lf-2')
    inp = dassh.DASSH_Input(inpath)
    k = ('Assembly', 'fuel', 'AxialRegion', 'lower_refl')
    inp.data[k[0]][k[1]][k[2]][k[3]]['hydraulic_diameter'] = \
        asm.rodded.bundle_params['de']
    inp.data[k[0]][k[1]][k[2]][k[3]]['vf_coolant'] = \
        (asm.rodded.bundle_params['area']
         / (0.5 * np.sqrt(3) * asm.rodded.duct_ftf[0][0]**2))
    # print('de', inp.data[k[0]][k[1]][k[2]][k[3]]['hydraulic_diameter'])
    # print('vfc', inp.data[k[0]][k[1]][k[2]][k[3]]['vf_coolant'])
    r_res = dassh.Reactor(inp, path=outpath, write_output=True)
    r_res.temperature_sweep()
    asm = r_res.assemblies[0]
    res = asm.region[0].pressure_drop
    res /= asm.region_bnd[1] - asm.region_bnd[0]
    print('ans', ans)
    print('res', res)
    # Compare them
    diff = (res - ans) / ans
    print('rel diff', diff)
    assert abs(diff) < 0.05  # 5 % difference is tolerable


@pytest.mark.skip(reason='toy problem for milos')
def test_ur_ctrl_asm_sweep(simple_ctrl_params):
    """Test the simple model approximation on a double-duct assembly"""
    input, mat = simple_ctrl_params
    mat = {'coolant': dassh.Material('sodium_se2anl_425'),
           'duct': dassh.Material('ht9_se2anl_425')}
    fr = 1.0

    # Make rodded region
    rr = dassh.region_rodded.make_rr_asm(input, 'dummy', mat.copy(), fr)

    # Make unrodded region; manually set UR params
    input['use_low_fidelity_model'] = True
    input['convection_factor'] = "calculate"
    ur = dassh.region_unrodded.make_ur_asm(input, mat.copy(), fr)

    # Manual activation
    for k in rr.temp.keys():
        rr.temp[k] *= 623.15
        try:
            ur.temp[k] *= 623.15
        except KeyError:
            continue

    # Calculate mesh size
    dz_rr = dassh.region_rodded.calculate_min_dz(rr, 623.15, 773.15)
    dz_ur = dassh.region_unrodded.calculate_min_dz(ur, 623.15, 773.15)
    dz = min([dz_rr[0], dz_ur[0]])
    print('dz_rr', dz_rr)
    print('dz_ur (simple)', dz_ur)

    print(rr.coolant.thermal_conductivity * rr._sf)
    print(rr.coolant.density * rr.coolant.heat_capacity
          * rr.coolant_int_params['eddy'])
    print(1 - rr.pin_diameter / rr.pin_pitch)
    assert 0
    # Sweep
    length = 1.0
    n_steps = np.ceil(length / dz)
    print(n_steps)
    p_lin = 0.15e6
    power_ur = {'refl': p_lin}
    power_rr = make_rr_power(rr, power_ur)
    gap_temp_ur = np.ones(6) * (350.0 + 273.15)
    gap_temp_rr = make_rr_gap_temps_rr(rr, gap_temp_ur)
    fake_htc = np.ones(2) * 2e4
    for i in range(int(n_steps)):
        ur._update_coolant_params(ur.avg_coolant_int_temp)
        ur.calculate(dz, power_ur, gap_temp_ur, fake_htc, ebal=True)
        rr._update_coolant_int_params(rr.avg_coolant_int_temp)
        rr._update_coolant_byp_params(rr.avg_coolant_byp_temp)
        rr.calculate(dz, power_rr, gap_temp_rr, fake_htc, ebal=True)

    cp = ur.coolant.heat_capacity
    print()
    print('UR ENERGY FROM DUCT (W):', ur.ebal['from_duct'])
    print('RR ENERGY FROM DUCT (W):', rr.ebal['from_duct'])
    print()
    print('UR COOLANT DT (C): ', ur.avg_coolant_temp - 623.15)
    print('RR COOLANT DT (C): ', rr.avg_coolant_temp - 623.15)
    print()
    print('UR EBAL PER HEX SIDE')
    print(ur.ebal['per_hex_side'])
    print('RR EBAL PER HEX SIDE')
    print(rr.ebal['per_hex_side'])
    print()
    print('UR EBAL')
    print('added:', ur.ebal['power'])
    print('from duct:', ur.ebal['from_duct'])
    tot = ur.ebal['power'] + ur.ebal['from_duct']
    print('sum:', tot)
    dT = ur.avg_coolant_temp - 623.15
    print('coolant rise:', dT * ur.flow_rate * cp)
    print('bal:', tot - dT * ur.flow_rate * cp)
    print()

    print('RR EBAL')
    print('added:', rr.ebal['power'])
    print('from duct:', rr.ebal['from_duct'])
    print('to byp:', rr.ebal['from_duct_byp'])
    tot = rr.ebal['power'] + rr.ebal['from_duct_byp'] + rr.ebal['from_duct']
    print('sum:', tot)
    dT = rr.avg_coolant_temp - 623.15
    print('coolant rise:', dT * rr.total_flow_rate * cp)
    print('bal:', tot - dT * rr.total_flow_rate * cp)
    print()

    print('UR AVG COOLANT // DUCT TEMP')
    print(ur.temp['coolant_int'])
    print(ur.avg_coolant_temp - 273.15)
    print(ur.avg_duct_mw_temp[0] - 273.15)
    print(np.average(ur.temp['duct_surf'][-1, -1]) - 273.15)
    print('RR AVG COOLANT // DUCT TEMP')
    print(rr.avg_coolant_int_temp - 273.15)
    print(rr.avg_coolant_temp - 273.15)
    print(rr.avg_duct_mw_temp[0] - 273.15)
    print(np.average(rr.temp['duct_surf'][-1, -1]) - 273.15)
    print()
    # print(c_shield_rr.temp['coolant_int'])
    assert 0


@pytest.mark.skip(reason='lol')
def test_ur_vs_rr_ebal(shield_ur_simple, shield_ur_mnh, c_shield_rr,
                       c_shield_simple_rr):
    """Compare energy balance in rodded and un-rodded regions"""
    c_shield_rr._conv_approx = True
    c_shield_simple_rr._conv_approx = True

    # shield_ur_mnh._params['xhtc'] = shield_ur_mnh.vf['coolant']
    # shield_ur_simple._params['xhtc'] = shield_ur_mnh.vf['coolant']
    # shield_ur_mnh._params['xhtc'] = 0.577442107490257
    # shield_ur_simple._params['xhtc'] = 0.577442107490257
    # shield_ur_mnh._params['xhtc'] = 0.12
    # shield_ur_simple._params['xhtc'] = 0.12
    shield_ur_mnh._params['lowflow'] = True
    shield_ur_simple._params['lowflow'] = True
    # print(c_shield_rr.params['area'][0]
    #       * c_shield_rr.subchannel.n_sc['coolant']['interior'])
    # print(c_shield_rr.params['area'][1]
    #       * c_shield_rr.subchannel.n_sc['coolant']['edge']
    #       + c_shield_rr.params['area'][2]
    #       * c_shield_rr.subchannel.n_sc['coolant']['corner'])

    # print(c_shield_rr._sf)
    c_shield_rr._sf = 1.0
    dz_rr = dassh.region_rodded.calculate_min_dz(
        c_shield_rr, 623.15, 773.15)
    # dz_rr2 = dassh.region_rodded.calculate_min_dz(
    #     c_shield_simple_rr, 623.15, 773.15)
    dz_ur1 = dassh.region_unrodded.calculate_min_dz(
        shield_ur_simple, 623.15, 773.15)
    dz_ur2 = dassh.region_unrodded.calculate_min_dz(
        shield_ur_mnh, 623.15, 773.15)
    dz = min([dz_rr[0], dz_ur1[0], dz_ur2[0]])
    print('dz_rr (m)', dz_rr)
    # print('dz_rr_7pin (m)', dz_rr2)
    print('dz_ur (simple)', dz_ur1)
    print('dz_ur (6 node)', dz_ur2)
    n_steps = 100
    # p_lin = 1000.0
    p_lin = 0.0
    power_ur = {'refl': p_lin}
    power_rr = {'pins': np.ones(61) * p_lin / 61,
                'duct': np.zeros(
                    c_shield_rr.subchannel.n_sc['duct']['total']),
                'cool': np.zeros(
                    c_shield_rr.subchannel.n_sc['coolant']['total'])
                }
    power_rr2 = {'pins': np.ones(7) * p_lin / 7,
                 'duct': np.zeros(c_shield_simple_rr
                                  .subchannel.n_sc['duct']['total']),
                 'cool': np.zeros(c_shield_simple_rr
                                  .subchannel.n_sc['coolant']['total'])}
    # gap_temp_ur = np.linspace(625, 750, 6)  # [625, 650, 675, 700, 725, 750]
    # gap_temp_rr = np.linspace(625, 750, (c_shield_rr.subchannel
    #                                      .n_sc['duct']['total']))
    # gap_temp_ur = 623.15 * np.ones(6)
    # gap_temp_rr = 623.15 * np.ones((c_shield_rr.subchannel
    #                                 .n_sc['duct']['total']))
    gap_temp_ur = np.ones(6) * 700.0
    # gap_temp_ur = np.array([623.15 + 10, 623.15 - 10, 623.15 - 20,
    #                         623.15 - 10, 623.15 + 10, 623.15 + 20])
    duct_per_side = int(c_shield_rr.subchannel.n_sc['duct']['total'] / 6)
    gap_temp_rr = np.linspace(np.roll(gap_temp_ur, 1),
                              gap_temp_ur,
                              duct_per_side + 1)
    gap_temp_rr = gap_temp_rr.transpose()
    gap_temp_rr = gap_temp_rr[:, 1:]
    gap_temp_rr = np.hstack(gap_temp_rr)

    duct_per_side = int(c_shield_simple_rr.subchannel.n_sc['duct']['total'] / 6)
    print(duct_per_side)
    gap_temp_rr2 = np.linspace(np.roll(gap_temp_ur, 1),
                               gap_temp_ur,
                               duct_per_side + 1)

    gap_temp_rr2 = gap_temp_rr2.transpose()
    print(gap_temp_rr2.shape)
    gap_temp_rr2 = gap_temp_rr2[:, 1:]
    gap_temp_rr2 = np.hstack(gap_temp_rr2)

    fake_htc = np.ones(2) * 2e4

    # shield_ur_mnh._params['hde'] /= 2
    # wp_ur = shield_ur.duct_perim
    total_area = np.sqrt(3) * 0.5 * shield_ur_mnh.duct_ftf[0]**2
    struct_area = shield_ur_mnh.vf['struct'] * total_area
    struct_r = np.sqrt(struct_area / np.pi)
    struct_perim = 2 * np.pi * struct_r
    # print('ORIGINAL DE:', shield_ur._params['de'])
    # print('ORIGINAL WP:', wp_ur)
    # print('ADDED WP:', struct_perim)
    # print('INCREASE:', struct_perim / wp_ur)
    # shield_ur._params['de'] = (4 * shield_ur.total_area['coolant_int']
    #                            / (2 * (struct_perim + wp_ur)))

    for i in range(n_steps):
        # gap_temp_ur = np.linspace(
        #     shield_ur_mnh.avg_coolant_temp,
        #     shield_ur_mnh.avg_coolant_temp - 10.0,
        #     6)
        # gap_temp_rr = np.linspace(
        #     c_shield_rr.avg_coolant_temp,
        #     c_shield_rr.avg_coolant_temp - 10.0,
        #     c_shield_rr.subchannel.n_sc['duct']['total'])
        shield_ur_mnh.calculate(
            dz, power_ur, gap_temp_ur, fake_htc, ebal=True)
        shield_ur_simple.calculate(
            dz, power_ur, gap_temp_ur, fake_htc, ebal=True)
        c_shield_rr.calculate(
            dz, power_rr, gap_temp_rr, fake_htc, ebal=True)
        c_shield_simple_rr.calculate(
            dz, power_rr2, gap_temp_rr2, fake_htc, ebal=True)

    print('UNRODDED (MNH)')
    # print('AREA:', shield_ur_mnh.total_area['coolant_int'])
    # print('DUCT PERIM', shield_ur_mnh.duct_perim)
    # print('STRUCT PERIM', struct_perim)
    print('DE:', shield_ur_mnh._params['de'])
    # print('RE:', shield_ur_mnh.coolant_params['Re'])
    # print('HTC:', shield_ur_mnh.coolant_params['htc'])
    print('ENERGY ADDED (W): ', shield_ur_mnh.ebal['power'])
    print('ENERGY FROM DUCT (W):', shield_ur_mnh.ebal['from_duct'])
    total = (shield_ur_mnh.ebal['from_duct']
             + shield_ur_mnh.ebal['power'])
    print('TOTAL ENERGY INPUT (W):', total)
    print('COOLANT DT (C): ', shield_ur_mnh.avg_coolant_temp - 623.15)
    e_temp_rise = (shield_ur_mnh.flow_rate
                   * shield_ur_mnh.coolant.heat_capacity
                   * (shield_ur_mnh.avg_coolant_temp - 623.15))
    print('ENERGY COOLANT DT (W):', e_temp_rise)
    bal = total - e_temp_rise
    print('DIFFERENCE (W):', bal)
    # print(shield_ur_mnh.temp['coolant_int'])
    print(shield_ur_mnh.ebal['per_hex_side'])

    # print()
    # print('UNRODDED (SIMPLE)')
    # print('ENERGY ADDED (W): ', shield_ur_simple.ebal['power'])
    # print('ENERGY FROM DUCT (W):', shield_ur_simple.ebal['from_duct'])
    # total = (shield_ur_simple.ebal['from_duct']
    #          + shield_ur_simple.ebal['power'])
    # print('TOTAL ENERGY INPUT (W):', total)
    # print('COOLANT DT (C): ', shield_ur_simple.avg_coolant_temp - 623.15)
    # e_temp_rise = (shield_ur_simple.flow_rate
    #                * shield_ur_simple.coolant.heat_capacity
    #                * (shield_ur_simple.avg_coolant_temp - 623.15))
    # print('ENERGY COOLANT DT (W):', e_temp_rise)
    # bal = total - e_temp_rise
    # print('DIFFERENCE (W):', bal)
    # print(shield_ur_simple.ebal['per_hex_side'])

    # print()
    # print('RODDED 7')
    # print('AREA:', c_shield_simple_rr.params['area'])
    # print('BUNDLE AREA:', c_shield_simple_rr.bundle_params['area'])
    # print('BUNDLE WP:', c_shield_simple_rr.bundle_params['wp'])
    # # print('DE:', c_shield_rr.params['de'])
    # print('BUNDLE DE:', c_shield_simple_rr.bundle_params['de'])
    # # print('RE:', c_shield_rr.coolant_int_params['Re'])
    # # print('RE_sc:', c_shield_rr.coolant_int_params['Re_sc'])
    # # print('HTC:', c_shield_rr.coolant_int_params['htc'])
    # print('ENERGY ADDED (W): ', c_shield_simple_rr.ebal['power'])
    # print('ENERGY FROM DUCT (W):', c_shield_simple_rr.ebal['from_duct'])
    # total = (c_shield_simple_rr.ebal['from_duct']
    #          + c_shield_simple_rr.ebal['power'])
    # print('TOTAL ENERGY INPUT (W):', total)
    # print('COOLANT DT (C): ', c_shield_simple_rr.avg_coolant_temp - 623.15)
    # e_temp_rise = (c_shield_simple_rr.int_flow_rate
    #                * c_shield_simple_rr.coolant.heat_capacity
    #                * (c_shield_simple_rr.avg_coolant_temp - 623.15))
    # print('ENERGY COOLANT DT (W):', e_temp_rise)
    # bal = total - e_temp_rise
    # print('DIFFERENCE (W):', bal)
    # print(c_shield_simple_rr.temp['coolant_int'])
    # print(c_shield_simple_rr.ebal['per_hex_side'])

    print()
    print('RODDED 61')
    # print('AREA:', c_shield_rr.params['area'])
    # print('BUNDLE AREA:', c_shield_rr.bundle_params['area'])
    # print('BUNDLE WP:', c_shield_rr.bundle_params['wp'])
    print('DE:', c_shield_rr.params['de'])
    print('BUNDLE DE:', c_shield_rr.bundle_params['de'])
    # print('RE:', c_shield_rr.coolant_int_params['Re'])
    # print('RE_sc:', c_shield_rr.coolant_int_params['Re_sc'])
    # print('HTC:', c_shield_rr.coolant_int_params['htc'])
    print('ENERGY ADDED (W): ', c_shield_rr.ebal['power'])
    print('ENERGY FROM DUCT (W):', c_shield_rr.ebal['from_duct'])
    total = (c_shield_rr.ebal['from_duct']
             + c_shield_rr.ebal['power'])
    print('TOTAL ENERGY INPUT (W):', total)
    print('COOLANT DT (C): ', c_shield_rr.avg_coolant_temp - 623.15)
    e_temp_rise = (c_shield_rr.int_flow_rate
                   * c_shield_rr.coolant.heat_capacity
                   * (c_shield_rr.avg_coolant_temp - 623.15))
    print('ENERGY COOLANT DT (W):', e_temp_rise)
    bal = total - e_temp_rise
    # print('DIFFERENCE (W):', bal)
    print(c_shield_rr.temp['coolant_int'][:4])
    # print(c_shield_rr.temp['coolant_int'][-4:])
    # print(c_shield_rr.subchannel.n_sc['coolant'])
    print(c_shield_rr.temp['coolant_int'][93: 100])
    print(c_shield_rr.ebal['per_hex_side'])
    assert 0


def make_rr_gap_temps_rr(rr, gap_temp_ur):
    duct_per_side = int(rr.subchannel.n_sc['duct']['total'] / 6)
    gap_temp_rr = np.linspace(np.roll(gap_temp_ur, 1),
                              gap_temp_ur,
                              duct_per_side + 1)
    gap_temp_rr = gap_temp_rr.transpose()
    gap_temp_rr = gap_temp_rr[:, 1:]
    gap_temp_rr = np.hstack(gap_temp_rr)
    return gap_temp_rr


def make_rr_power(rr, power_ur):
    n_pin = rr.pin_lattice.n_pin
    power_rr = {}
    power_rr['pins'] = np.ones(n_pin) * power_ur['refl'] / n_pin
    power_rr['duct'] = np.zeros(rr.n_duct *
                                rr.subchannel.n_sc['duct']['total'])
    power_rr['cool'] = np.zeros(rr.subchannel.n_sc['coolant']['total'])
    return power_rr


@pytest.mark.skip(reason='lol')
def test_ur_vs_rr_yoyoyo1(c_shield_rr_params):
    """x"""
    asm_params, mat_dict = c_shield_rr_params
    # asm_params['shape_factor'] = 10.0
    fr = 0.05
    # asm_params['interior_mfr_frac'] = 0.06775
    # asm_params['interior_mfr_frac'] = 0.0001
    for x in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]:
        # Make rodded region
        rr = dassh.region_rodded.make_rr_asm(
            asm_params, 'dummy', mat_dict, fr)
        print(rr.pin_pitch)
        print(rr.pin_diameter)
        print(rr.duct_ftf[0][0])
        print((rr.pin_pitch - rr.pin_diameter) / rr.pin_pitch)
        assert 0
        # Make unrodded region; manually set UR params
        asm_params['use_low_fidelity_model'] = True
        asm_params['convection_factor'] = x
        ur = dassh.region_unrodded.make_ur_asm(asm_params, mat_dict, fr)

        for k in rr.temp.keys():
            rr.temp[k] *= 623.15
            ur.temp[k] *= 623.15

        # Calculate mesh size
        dz_rr = dassh.region_rodded.calculate_min_dz(rr, 623.15, 773.15)
        dz_ur = dassh.region_unrodded.calculate_min_dz(ur, 623.15, 773.15)
        dz = min([dz_rr[0], dz_ur[0]])
        # print('dz_rr', dz_rr)
        # print('dz_ur (simple)', dz_ur)
        length = 1.0
        n_steps = np.ceil(length / dz)
        # print(n_steps)
        # print(ur._params['htc'])
        # print(ur.coolant_params['htc'])
        # # nu = nusselt_db.calculate_bundle_Nu(ur.coolant,
        # #                                     ur.coolant_params['Re'],
        # #                                     ur._params['htc'])
        # # print(nu)
        # # print(rr.coolant_int_params['htc'])
        # # print((rr.pin_pitch - rr.pin_diameter) / rr.pin_pitch)
        # ur._params['htc'][-1] *= (rr.pin_pitch - rr.pin_diameter) / rr.pin_pitch
        # nu = nusselt_db.calculate_bundle_Nu(ur.coolant,
        #                                     ur.coolant_params['Re'],
        #                                     ur._params['htc'])
        # print(nu * ur.coolant.thermal_conductivity / ur._params['de'])
        # assert 0
        p_lin = 1e3
        power_ur = {'refl': p_lin}
        power_rr = make_rr_power(rr, power_ur)
        gap_temp_ur = np.ones(6) * (400.0 + 273.15)
        gap_temp_rr = make_rr_gap_temps_rr(rr, gap_temp_ur)
        fake_htc = np.ones(2) * 2e4
        z = 0.0
        for i in range(int(n_steps)):
            z += dz
            ur.calculate(dz, power_ur, gap_temp_ur, fake_htc, ebal=True)
            rr.calculate(dz, power_rr, gap_temp_rr, fake_htc, ebal=True)
        print(x, ur.avg_coolant_temp, rr.avg_coolant_temp)

    # print()
    # print(rr.coolant_int_params['fs'])
    # print(rr.coolant_int_params['fs']
    #       * rr.params['area']
    #       * fr
    #       / rr.bundle_params['area'])
    # cp = ur.coolant.heat_capacity
    # print()
    # print('UR ENERGY FROM DUCT (W):', np.sum(ur.ebal['duct']))
    # print('RR ENERGY FROM DUCT (W):', np.sum(rr.ebal['duct']))
    # print()
    # print('UR COOLANT DT (C): ', ur.avg_coolant_temp - 623.15)
    # print('RR COOLANT DT (C): ', rr.avg_coolant_temp - 623.15)
    # print()
    # # print('UR EBAL PER HEX SIDE')
    # # print(ur.ebal['per_hex_side'])
    # # print('RR EBAL PER HEX SIDE')
    # # print(rr.ebal['per_hex_side'])
    # # print()
    # print('UR EBAL')
    # print('added:', ur.ebal['power'])
    # print('from duct:', np.sum(ur.ebal['duct']))
    # tot = ur.ebal['power'] + np.sum(ur.ebal['duct'])
    # print('sum:', tot)
    # dT = ur.avg_coolant_temp - 623.15
    # print('coolant rise:', dT * ur.flow_rate * cp)
    # print('bal:', tot - dT * ur.flow_rate * cp)
    # print()
    #
    # print(ur.ebal)
    # print()
    # print('UR AVG COOLANT // DUCT TEMP')
    # print(ur.temp['coolant_int'])
    # print(ur.avg_coolant_temp - 273.15)
    # print(ur.avg_duct_mw_temp[0] - 273.15)
    # print(np.average(ur.temp['duct_surf'][-1, -1]) - 273.15)
    # print('RR AVG COOLANT // DUCT TEMP')
    # print(rr.avg_coolant_temp - 273.15)
    # print(rr.avg_duct_mw_temp[0] - 273.15)
    # print(np.average(rr.temp['duct_surf'][-1, -1]) - 273.15)
    # print()
    # print(c_shield_rr.temp['coolant_int'])
    assert 0


@pytest.mark.skip(reason='lol')
# def test_ur_vs_rr_yoyoyo2(c_fuel_params):
def test_ur_vs_rr_yoyoyo2(c_shield_rr_params):
    """x"""
    # aparams, mat = c_fuel_params
    aparams, mat = c_shield_rr_params
    fr = 0.05
    # Make rodded region
    rr = dassh.region_rodded.make_rr_asm(aparams, 'dummy', mat.copy(), fr)
    # Make unrodded region; manually set UR params
    aparams['use_low_fidelity_model'] = True
    aparams['convection_factor'] = 1.0  #'calculate'
    ur = dassh.region_unrodded.make_ur_asm(aparams, mat.copy(), fr)

    for k in rr.temp.keys():
        rr.temp[k] *= 623.15
        ur.temp[k] *= 623.15

    # Calculate mesh size
    dz_rr = dassh.region_rodded.calculate_min_dz(rr, 623.15, 773.15)
    dz_ur = dassh.region_unrodded.calculate_min_dz(ur, 623.15, 773.15)
    dz = min([dz_rr[0], dz_ur[0]])
    print('dz_rr', dz_rr)
    print('dz_ur (simple)', dz_ur)
    length = 1.0
    n_steps = np.ceil(length / dz)
    print(n_steps)

    p_lin = 20000.0
    power_ur = {'refl': p_lin}
    power_rr = make_rr_power(rr, power_ur)
    gap_temp_ur = np.ones(6) * 623.15  # (450.0 + 273.15)
    # gap_temp_ur = np.ones(6) * (450.0 + 273.15)
    gap_temp_rr = make_rr_gap_temps_rr(rr, gap_temp_ur)
    fake_htc = np.ones(2) * 2e4
    for i in range(int(n_steps)):
        ur.calculate(dz, power_ur, gap_temp_ur, fake_htc, ebal=True)
        rr.calculate(dz, power_rr, gap_temp_rr, fake_htc, ebal=True)
        print(ur.avg_coolant_temp, rr.avg_coolant_temp)
        # print(shield_ur_simple.avg_coolant_temp,
        #       shield_ur_simple.avg_duct_mw_temp[0],
        #       c_shield_rr.avg_coolant_temp,
        #       c_shield_rr.avg_duct_mw_temp[0])
    assert 0
    cp = ur.coolant.heat_capacity
    print()
    print('UR ENERGY FROM DUCT (W):', ur.ebal['from_duct'])
    print('RR ENERGY FROM DUCT (W):', rr.ebal['from_duct'])
    print()
    print('UR COOLANT DT (C): ', ur.avg_coolant_temp - 623.15)
    print('RR COOLANT DT (C): ', rr.avg_coolant_temp - 623.15)
    print()
    print('UR EBAL PER HEX SIDE')
    print(ur.ebal['per_hex_side'])
    print('RR EBAL PER HEX SIDE')
    print(rr.ebal['per_hex_side'])
    print()
    print('UR EBAL')
    print('added:', ur.ebal['power'])
    print('from duct:', ur.ebal['from_duct'])
    tot = ur.ebal['power'] + ur.ebal['from_duct']
    print('sum:', tot)
    dT = ur.avg_coolant_temp - 623.15
    print('coolant rise:', dT * ur.flow_rate * cp)
    print('bal:', tot - dT * ur.flow_rate * cp)
    print()

    print(ur.ebal)
    print()
    print('UR AVG COOLANT // DUCT TEMP')
    print(ur.avg_coolant_temp - 273.15)
    print(ur.avg_duct_mw_temp[0] - 273.15)
    print(np.average(ur.temp['duct_surf'][-1, -1]) - 273.15)
    print('RR AVG COOLANT // DUCT TEMP')
    print(rr.avg_coolant_temp - 273.15)
    print(rr.avg_duct_mw_temp[0] - 273.15)
    print(np.average(rr.temp['duct_surf'][-1, -1]) - 273.15)
    print()
    # print(c_shield_rr.temp['coolant_int'])
    print('DT_UR = ', ur.avg_coolant_temp - 723.15)
    print('DT_RR = ', rr.avg_coolant_temp - 723.15)
    assert 0
