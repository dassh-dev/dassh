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
date: 2021-11-24
author: matz
Test the DASSH Assembly object
"""
########################################################################
import os
import numpy as np
import copy
import pytest
import dassh


# Use "print_option" to print temperatures and parameters
# for use in Excel spreadsheet for verification
print_option = False


def test_int_coolant_verification(simple_asm):
    """Test that the method to calculate interior and bypass coolant
    temperatures performs as expected"""
    # Set up some stuff
    inlet_temp = 623.15
    gap_t = np.ones(simple_asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    simple_asm.rodded._update_coolant_int_params(inlet_temp)
    # gap_htc = np.ones(simple_asm.rodded.subchannel.n_sc['duct']['total'])
    # gap_htc *= simple_asm.rodded.coolant_int_params['htc'][1:]
    gap_htc = simple_asm.rodded.coolant_int_params['htc'][1:]
    gap_htc = gap_htc[
        simple_asm.rodded.subchannel.type[
            simple_asm.rodded.subchannel.n_sc['coolant']['interior']:
            simple_asm.rodded.subchannel.n_sc['coolant']['total']] - 1]
    # ans = np.array([6.4496451619E+02, 6.4283014996E+02, 6.4478668013E+02,
    #                 6.4915222308E+02, 6.5162946544E+02, 6.4939828872E+02,
    #                 6.4334361185E+02, 6.6798001104E+02, 6.3973505298E+02,
    #                 6.6586474966E+02, 6.4202848578E+02, 6.7929001807E+02,
    #                 6.4829348703E+02, 6.9723737663E+02, 6.5254123430E+02,
    #                 7.0016988643E+02, 6.4988479191E+02, 6.8433778925E+02])
    #
    ans = np.array([
        6.2361894147E+02, 6.2357888738E+02, 6.2361746368E+02,
        6.2370276753E+02, 6.2374961923E+02, 6.2370448287E+02,
        6.2358481199E+02, 6.2412338768E+02, 6.2351685219E+02,
        6.2408479054E+02, 6.2356204267E+02, 6.2434522339E+02,
        6.2368527386E+02, 6.2469859277E+02, 6.2376607109E+02,
        6.2474451781E+02, 6.2371183773E+02, 6.2443032335E+02
    ])
    # print(asm_tables.print_specs(simple_asm, 6, 72))
    # print(asm_tables.print_subchannel(simple_asm, 6, 72))
    # simple_asm._z = 1.29
    # print(simple_asm.region)
    # print(simple_asm.region_idx)
    # print(simple_asm.region_bnd)
    # print(simple_asm.active_region_idx)
    # print(simple_asm.active_region.name)

    # print('{:.12f}'.format(simple_asm.rodded.L[0][0]))
    # print('{:.12f}'.format(simple_asm.rodded.L[0][1]))
    # print('{:.12f}'.format(simple_asm.rodded.L[1][1]))
    # print('{:.12f}'.format(simple_asm.rodded.L[1][2]))
    # print('{:.12f}'.format(simple_asm.rodded.d['wcorner'][0, 1]))
    z = 1.29
    dz = 0.001
    for i in range(200):
        # Calculate coolant and duct temperatures at the current level
        simple_asm.calculate(z, dz, gap_t, gap_htc)

        # Collect data to print for verification if test is not passed
        z_power = simple_asm.power.get_power(z)
        print_list = [z]
        print_list += list(simple_asm.temp_coolant)
        print_list += [simple_asm.active_region.coolant.heat_capacity,
                       simple_asm.active_region.coolant.thermal_conductivity,
                       simple_asm.active_region.coolant.density,
                       simple_asm.active_region.coolant_int_params['fs'][0],
                       simple_asm.active_region.coolant_int_params['fs'][1],
                       simple_asm.active_region.coolant_int_params['fs'][2],
                       simple_asm.active_region.coolant_int_params['htc'][1],
                       simple_asm.active_region.coolant_int_params['htc'][2],
                       simple_asm.active_region.coolant_int_params['eddy'],
                       simple_asm.active_region.coolant_int_params['swirl'][1]]
        print_list += list(z_power['pins'])
        print_list += list(z_power['cool'])
        print_list += list(simple_asm.temp_duct_surf[0, 0])
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))
        z += dz

    # print(simple_asm.temp_coolant - ans)
    assert np.allclose(simple_asm.temp_coolant, ans)


def test_pin_only_int_coolant_verification(testdir):
    """Test that the method to calculate interior coolant temperatures
    performs as expected in the simplest case: adiabatic duct wall,
    power delivered only to pins"""
    rpath = os.path.join(testdir,
                         'test_results',
                         'conservation-1',
                         'dassh_reactor.pkl')
    if os.path.exists(rpath):
        r = dassh.reactor.load(rpath)
    else:
        pytest.skip('Cannot load necessary reactor object')

    T_ans = copy.deepcopy(r.assemblies[0].rodded.temp['coolant_int'])
    dT_ans = T_ans - r.inlet_temp
    r.reset()
    asm = copy.deepcopy(r.assemblies[0])
    # # Set up some stuff
    inlet_temp = 623.15
    gap_t = np.ones(asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    asm.rodded._update_coolant_int_params(inlet_temp)
    gap_htc = asm.rodded.coolant_int_params['htc'][1:]
    gap_htc = gap_htc[
        asm.rodded.subchannel.type[
            asm.rodded.subchannel.n_sc['coolant']['interior']:
            asm.rodded.subchannel.n_sc['coolant']['total']] - 1]

    # print(simple_asm_pin_only.rodded.pin_pitch)
    # print(simple_asm_pin_only.rodded.pin_diameter)
    # for i in range(3):
    #     print('{:.16f}'.format(simple_asm_pin_only.rodded.params['area'][i]))
    #     print('{:.16f}'.format(simple_asm_pin_only.rodded.coolant_int_params['fs'][i]))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.bundle_params['area']))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.d['pin-pin']))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.d['pin-wall']))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.d['wcorner'][0, 0]))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.coolant_int_params['htc'][1]))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.coolant_int_params['htc'][2]))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.coolant_int_params['eddy']))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.coolant_int_params['swirl'][1]))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.coolant.heat_capacity))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.coolant.thermal_conductivity))
    # print('{:.16f}'.format(simple_asm_pin_only.rodded.coolant.density))
    # assert 0
    ans = np.array([8.013751882556e+02, 8.013751882556e+02,
                    8.013751882556e+02, 8.013751882556e+02,
                    8.013751882556e+02, 8.013751882556e+02,
                    7.828658939007e+02, 7.909089870977e+02,
                    7.829086787407e+02, 7.828658939007e+02,
                    7.909089870977e+02, 7.829086787407e+02,
                    7.828658939007e+02, 7.909089870977e+02,
                    7.829086787407e+02, 7.828658939007e+02,
                    7.909089870977e+02, 7.829086787407e+02,
                    7.828658939007e+02, 7.909089870977e+02,
                    7.829086787407e+02, 7.828658939007e+02,
                    7.909089870977e+02, 7.829086787407e+02,
                    7.732092711018e+02, 7.733564281780e+02,
                    7.730218622350e+02, 7.732092711018e+02,
                    7.733564281780e+02, 7.730218622350e+02,
                    7.732092711018e+02, 7.733564281780e+02,
                    7.730218622350e+02, 7.732092711018e+02,
                    7.733564281780e+02, 7.730218622350e+02,
                    7.732092711018e+02, 7.733564281780e+02,
                    7.730218622350e+02, 7.732092711018e+02,
                    7.733564281780e+02, 7.730218622350e+02])
    for i in range(len(r.dz)):
        z = r.z[i + 1]
        dz = r.dz[i]
        # Calculate coolant and duct temperatures at the current level
        asm.calculate(z, dz, gap_t, gap_htc, adiabatic=True)

        # Collect data to print for verification if test is not passed
        z_power = asm.power.get_power(z - 0.5 * dz)
        print_list = [z]
        print_list += list(asm.temp_coolant)
        print_list += [z_power['pins'][0]]
        print_list += list(asm.temp_duct_surf[0, 0])
        if print_option:
            print(' '.join(['{:.12e}'.format(v) for v in print_list]))

    # assert 0
    dT_ss = ans - r.inlet_temp
    dT_res = asm.rodded.temp['coolant_int'] - r.inlet_temp
    print(asm.rodded._adj_sw)
    print(dT_res - dT_ss)
    assert np.allclose(dT_res, dT_ans)
    assert np.allclose(dT_res, dT_ss)
    assert np.allclose(asm.rodded.temp['coolant_int'], ans)


def test_duct_verification(simple_asm):
    """Test that the method to calculate interior and bypass coolant
    temperatures performs as expected"""
    # Set up some stuff
    inlet_temp = 623.15
    gap_t = np.ones(simple_asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    simple_asm.rodded._update_coolant_int_params(inlet_temp)
    gap_htc = simple_asm.rodded.coolant_int_params['htc'][1:]
    ans = {
        's_in': np.array([
            6.2331273999E+02, 6.2366950290E+02, 6.2328786212E+02,
            6.2365882272E+02, 6.2330652850E+02, 6.2381576163E+02,
            6.2335518811E+02, 6.2401663962E+02, 6.2338521129E+02,
            6.2402946267E+02, 6.2336258389E+02, 6.2383945365E+02]),
        'mw': np.array([
            6.2327050326E+02, 6.2346223552E+02, 6.2325629088E+02,
            6.2345679790E+02, 6.2326736000E+02, 6.2356179829E+02,
            6.2331509061E+02, 6.2370635253E+02, 6.2333667513E+02,
            6.2371288115E+02, 6.2331883317E+02, 6.2357386063E+02]),
        's_out': np.array([
            6.2315376531E+02, 6.2316285191E+02, 6.2315338826E+02,
            6.2316265687E+02, 6.2315369029E+02, 6.2316729174E+02,
            6.2315536805E+02, 6.2317394004E+02, 6.2315603219E+02,
            6.2317417422E+02, 6.2315545737E+02, 6.2316772442E+02])
    }
    simple_asm._z = 1.29
    z = 1.29
    dz = 0.001
    for i in range(100):
        htc1 = simple_asm.active_region.coolant_int_params['htc'][1]
        htc2 = simple_asm.active_region.coolant_int_params['htc'][2]
        start = simple_asm.active_region.subchannel.n_sc['coolant']['interior']
        coolant_temps = list(simple_asm.temp_coolant[start:])
        simple_asm.calculate(z, dz, gap_t, gap_htc)

        # Collect data to print for verification if test is not passed
        z_power = simple_asm.power.get_power(z - dz * 0.5)
        print_list = [z]
        print_list += list(simple_asm.temp_duct_surf[0, 0])
        print_list += list(simple_asm.temp_duct_mw[0])
        print_list += list(simple_asm.temp_duct_surf[0, 1])
        print_list += [htc1, htc2]
        print_list.append(simple_asm.active_region.duct.thermal_conductivity)
        print_list += list(z_power['duct'])
        print_list += coolant_temps
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        z += dz

    assert np.allclose(ans['s_in'], simple_asm.temp_duct_surf[0, 0])
    assert np.allclose(ans['mw'], simple_asm.temp_duct_mw[0])
    assert np.allclose(ans['s_out'], simple_asm.temp_duct_surf[0, 1])


def test_bypass_gap_verification(simple_ctrl_asm):
    """Test that the method to calculate interior and bypass coolant
    temperatures performs as expected"""
    # Set up some stuff
    inlet_temp = 623.15
    gap_t = np.ones(simple_ctrl_asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    simple_ctrl_asm.rodded._update_coolant_int_params(inlet_temp)
    simple_ctrl_asm.rodded._update_coolant_byp_params([inlet_temp])
    gap_htc = simple_ctrl_asm.rodded.coolant_byp_params['htc'][0]
    if print_option:
        keys = ['d_bypass', 'Flow area (edge)', 'Flow area (corner)',
                'Flow area (total)', 'WP in (edge)', 'WP out (edge)',
                'WP in (corner)', 'WP out (corner)', 'L67']
        vals = [simple_ctrl_asm.rodded.d['bypass'][0],
                simple_ctrl_asm.rodded.bypass_params['area'][0, 0],
                simple_ctrl_asm.rodded.bypass_params['area'][0, 1],
                simple_ctrl_asm.rodded.bypass_params['total area'][0],
                simple_ctrl_asm.rodded.L[5][5][0],
                simple_ctrl_asm.rodded.L[5][5][0],
                2 * simple_ctrl_asm.rodded.d['wcorner'][0, 1],
                2 * simple_ctrl_asm.rodded.d['wcorner'][1, 1],
                simple_ctrl_asm.rodded.L[5][6][0]]
        for i in range(len(keys)):
            print(keys[i] + ': ' + '{:.15e}'.format(vals[i]))

    ans = np.array([6.234377846069E+02, 6.236242077736E+02,
                    6.234377827967E+02, 6.236242078352E+02,
                    6.234377846069E+02, 6.236242077736E+02,
                    6.234377827967E+02, 6.236242078352E+02,
                    6.234377846069E+02, 6.236242077736E+02,
                    6.234377827967E+02, 6.236242078352E+02])
    simple_ctrl_asm._z = 1.29
    z = 1.29
    dz = 0.001
    for i in range(100):
        simple_ctrl_asm.calculate(z, dz, gap_t, gap_htc)

        # Print things to see what's going on in the bypass gap
        print_list = [z]
        print_list += list(simple_ctrl_asm.temp_bypass[0])
        print_list += \
            [simple_ctrl_asm.active_region.coolant.heat_capacity,
             simple_ctrl_asm.active_region.coolant.thermal_conductivity,
             simple_ctrl_asm.active_region.byp_flow_rate[0],
             simple_ctrl_asm.active_region.coolant_byp_params['htc'][0, 0],
             simple_ctrl_asm.active_region.coolant_byp_params['htc'][0, 1]]
        print_list += list(simple_ctrl_asm.temp_duct_surf[0, 1])
        print_list += list(simple_ctrl_asm.temp_duct_surf[1, 0])

        if print_option:
            print(' '.join(['{:.12e}'.format(v) for v in print_list]))
        z += dz

    assert np.allclose(ans, simple_ctrl_asm.temp_bypass[0])


def test_interasm_gap_flow_model_verification(three_asm_core):
    """Test that the method to calculate inter-assembly gap coolant
    temperatures (flowing gap model) performs as expected"""
    # Set up some stuff
    asm_list, core_obj = three_asm_core
    inlet_temp = 623.15
    dz = 0.001  # should be sufficient
    n_zpts = 20
    r = np.random.RandomState(seed=42)  # set for reproducibility

    ans = np.array([6.4430257093E+02, 6.4133750612E+02,
                    6.4464137575E+02, 6.4204271522E+02,
                    6.4378541819E+02, 6.4345130150E+02,
                    6.4357382799E+02, 6.4400775157E+02,
                    6.4345297922E+02, 6.4337305191E+02,
                    6.4317890562E+02, 6.4074841996E+02,
                    6.3492735383E+02, 6.3384866583E+02,
                    6.3386094328E+02, 6.3326827778E+02,
                    6.3417287934E+02, 6.3349425161E+02,
                    6.3428529964E+02, 6.3624922416E+02,
                    6.3399980086E+02, 6.3267989605E+02,
                    6.3384325575E+02, 6.3344587959E+02,
                    6.3414989863E+02, 6.3341650255E+02,
                    6.3430409517E+02, 6.3349615942E+02])
    # 2021-04-29: ANSWER FOR OLD DASSH GAP MESHING
    # Spreadsheet for this answer still in file
    # ans = np.array([6.45037183080E+02, 6.41980972740E+02,
    #                 6.45358606160E+02, 6.42969608370E+02,
    #                 6.44963090990E+02, 6.39769145330E+02,
    #                 6.44748592010E+02, 6.40273536950E+02,
    #                 6.44595385000E+02, 6.39742980240E+02,
    #                 6.44317033130E+02, 6.41608817810E+02,
    #                 6.35587316080E+02, 6.31907150960E+02,
    #                 6.34474448330E+02, 6.31444749440E+02,
    #                 6.34788828880E+02, 6.31676407570E+02,
    #                 6.34928995790E+02, 6.36683571690E+02,
    #                 6.34538836150E+02, 6.33215033410E+02,
    #                 6.31916035520E+02, 6.34023142120E+02,
    #                 6.34780302460E+02, 6.31590910040E+02,
    #                 6.34916559090E+02, 6.31607081410E+02])
    # Create fake duct wall surface temperature array
    core_obj._update_coolant_gap_params(inlet_temp)
    duct_temps = []
    for asm in asm_list:
        tmp = np.zeros((n_zpts, asm.rodded.subchannel.n_sc['duct']['total']))
        tmp[0] = asm.duct_outer_surf_temp
        duct_temps.append(tmp)

    for zi in range(n_zpts):
        # Assemble information used to calculate gap temperatures
        core_obj._update_coolant_gap_params(core_obj.avg_coolant_gap_temp)
        print_list = [core_obj.gap_coolant.heat_capacity,
                      core_obj.gap_coolant.thermal_conductivity,
                      core_obj.gap_flow_rate,
                      core_obj.coolant_gap_params['htc'][0],
                      core_obj.coolant_gap_params['htc'][1],
                      core_obj.coolant_gap_params['htc'][5]]
        print_list += list(duct_temps[0][zi])
        print_list += list(duct_temps[1][zi])
        print_list += list(duct_temps[2][zi])

        # Calculate and print gap subchannel temperatures
        tduct = np.array([d[zi] for d in duct_temps])
        core_obj.calculate_gap_temperatures(dz, tduct)
        print_list = [dz * zi] + list(core_obj.coolant_gap_temp) + print_list
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        # Update duct temperatures
        if zi + 1 < n_zpts:
            for d in duct_temps:
                dT = r.rand(len(d[zi])) + 0.5  # range = [0.5, 1.5]
                d[zi + 1] = d[zi] + dT
            duct_temps[0][zi + 1] += 1.0

    assert np.allclose(ans, core_obj.coolant_gap_temp)


def test_interasm_gap_noflow_model_verification(three_asm_core):
    """Test no-flow (conduction) model for inter-assembly gap coolant"""
    asm_list, core_obj = three_asm_core
    core_obj.model = 'no_flow'
    core_obj.gap_flow_rate = 0.0
    core_obj.load(asm_list)

    # Set up some stuff
    inlet_temp = 623.15
    dz = 0.001  # should be sufficient
    n_zpts = 20
    r = np.random.RandomState(seed=42)  # set for reproducibility
    ans = np.array([
        6.5150426488E+02, 6.4752164488E+02, 6.5138793687E+02,
        6.5338567527E+02, 6.6059116544E+02, 6.6205940071E+02,
        6.6076180758E+02, 6.6202876454E+02, 6.5974523677E+02,
        6.6106137785E+02, 6.5953737318E+02, 6.5148242142E+02,
        6.4385701672E+02, 6.4355821554E+02, 6.4283016740E+02,
        6.4219931914E+02, 6.4205760814E+02, 6.4148057698E+02,
        6.4369322289E+02, 6.4000978899E+02, 6.4087071644E+02,
        6.3993620392E+02, 6.4299347495E+02, 6.4133422169E+02,
        6.4319991462E+02, 6.4239624871E+02, 6.4249846862E+02,
        6.4345278609E+02])

    # Create fake duct wall surface temperature array
    core_obj._update_coolant_gap_params(inlet_temp)
    duct_temps = []
    for asm in asm_list:
        tmp = np.zeros((n_zpts, asm.rodded.subchannel.n_sc['duct']['total']))
        tmp[0] = asm.duct_outer_surf_temp
        duct_temps.append(tmp)

    for zi in range(n_zpts):
        # Assemble information used to calculate gap temperatures
        core_obj._update_coolant_gap_params(core_obj.avg_coolant_gap_temp)
        print_list = [core_obj.gap_coolant.thermal_conductivity]
        print_list += list(duct_temps[0][zi])
        print_list += list(duct_temps[1][zi])
        print_list += list(duct_temps[2][zi])

        # Calculate and print gap subchannel temperatures
        tduct = np.array([d[zi] for d in duct_temps])
        core_obj.calculate_gap_temperatures(dz, tduct)
        print_list = [dz * zi] + list(core_obj.coolant_gap_temp) + print_list
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        # Update duct temperatures
        if zi + 1 < n_zpts:
            for d in duct_temps:
                dT = r.rand(len(d[zi])) + 0.5  # [0.5, 1.5]
                d[zi + 1] = d[zi] + dT
            duct_temps[0][zi + 1] += 1.0

    assert np.allclose(ans, core_obj.coolant_gap_temp)


def test_interasm_gap_ductavg_model_verification(three_asm_core):
    """Test no-flow (duct-avg) model for inter-assembly gap coolant"""
    asm_list, core_obj = three_asm_core
    core_obj.model = 'duct_average'
    core_obj.gap_flow_rate = 0.0

    # Set up some stuff
    inlet_temp = 623.15
    dz = 0.001  # should be sufficient
    n_zpts = 20
    r = np.random.RandomState(seed=42)  # set for reproducibility
    ans = np.array([651.511243165, 647.525353140, 651.392401735,
                    653.399834485, 660.609908130, 662.082117640,
                    660.764436810, 662.053100090, 659.745021560,
                    661.084043170, 659.557681360, 651.491705490,
                    643.869481620, 643.566222600, 642.834433430,
                    642.203403990, 642.061278200, 641.475761430,
                    643.682222870, 640.004054625, 640.870416540,
                    639.927400840, 643.014265190, 641.336761630,
                    643.189862100, 642.399881550, 642.500626920,
                    643.471961030])

    # Create fake duct wall surface temperature array
    core_obj._update_coolant_gap_params(inlet_temp)
    duct_temps = []
    for asm in asm_list:
        tmp = np.zeros((n_zpts, asm.rodded.subchannel.n_sc['duct']['total']))
        tmp[0] = asm.duct_outer_surf_temp
        duct_temps.append(tmp)

    for zi in range(n_zpts):
        # Assemble information used to calculate gap temperatures
        core_obj._update_coolant_gap_params(core_obj.avg_coolant_gap_temp)
        print_list = list(duct_temps[0][zi])
        print_list += list(duct_temps[1][zi])
        print_list += list(duct_temps[2][zi])

        # Calculate and print gap subchannel temperatures
        tduct = np.array([d[zi] for d in duct_temps])
        core_obj.calculate_gap_temperatures(dz, tduct)
        print_list = [dz * zi] + list(core_obj.coolant_gap_temp) + print_list
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        # Update duct temperatures
        if zi + 1 < n_zpts:
            for d in duct_temps:
                dT = r.rand(len(d[zi])) + 0.5  # [0.5, 1.5]
                d[zi + 1] = d[zi] + dT
            duct_temps[0][zi + 1] += 1.0

    assert np.allclose(ans, core_obj.coolant_gap_temp)


def print_bypass_gap_energy_cons_verification(simple_ctrl_asm_pins_cmat):
    """Test that method to calculate bypass coolant conserves energy;
    not actually a test, only run to print data"""
    # Notes:
    # - Power to pins only (no duct wall heating)
    # - Interior coolant temperature taken as given: will use to get
    #   inner duct temperatures and then bypass gap temperatures
    # - Will be calculating heat lost from interior coolant to duct
    #   and transferred from inner duct to bypass coolant
    # - Assuming adiabatic outer duct: no heat transfer there.

    # Set up some stuff
    asm = simple_ctrl_asm_pins_cmat
    inlet_temp = 623.15
    gap_t = np.ones(asm.rodded.subchannel.n_sc['duct']['total'])
    gap_t *= inlet_temp
    asm.rodded._update_coolant_int_params(inlet_temp)
    asm.rodded._update_coolant_byp_params([inlet_temp])
    gap_htc = asm.rodded.coolant_byp_params['htc'][0]

    # Print all the parameters we'll need
    asm_params = {
        'flow_rate': asm.flow_rate,
        'pin_pitch': asm.rodded.pin_pitch,
        'dwc00': asm.rodded.d['wcorner'][0, 0],
        'dwc01': asm.rodded.d['wcorner'][0, 1],
        'dwc10': asm.rodded.d['wcorner'][1, 0],
        'dwc11': asm.rodded.d['wcorner'][1, 1],
        'dthickness0': asm.rodded.duct_params['thickness'][0],
        'byp_thickness': asm.rodded.d['bypass'][0],
        'a_byp_edge': asm.rodded.bypass_params['area'][0, 0],
        'a_byp_corn': asm.rodded.bypass_params['area'][0, 1],
        'htc_int_edge': asm.rodded.coolant_int_params['htc'][1],
        'htc_int_corn': asm.rodded.coolant_int_params['htc'][2],
        'htc_byp_edge': asm.rodded.coolant_byp_params['htc'][0, 0],
        'htc_byp_corn': asm.rodded.coolant_byp_params['htc'][0, 1],
        'coolant_cp': asm.rodded.coolant.heat_capacity,
        'coolant_k': asm.rodded.coolant.thermal_conductivity,
        'duct_k': asm.rodded.duct.thermal_conductivity,
        'int_fr': asm.rodded.int_flow_rate,
        'byp_fr': asm.rodded.byp_flow_rate[0]
    }
    if print_option:
        for k in asm_params.keys():
            print(k, asm_params[k])
    asm._z = 1.29
    z = 1.29
    dz = 0.001
    for i in range(20):
        asm.calculate(z, dz, gap_t, gap_htc, adiabatic=True, ebal=True)

        # Collect data to print for verification if test is not passed
        print_list = [z]
        # Interior edge/corner coolant channels
        start = asm.active_region.subchannel.n_sc['coolant']['interior']
        print_list += list(asm.temp_coolant[start:])
        # Duct wall inner surface, midwall, and outer surface temps
        print_list += list(asm.temp_duct_surf[0, 0])
        print_list += list(asm.temp_duct_mw[0])
        print_list += list(asm.temp_duct_surf[0, 1])
        # Coolant bypass temperatures
        print_list += list(asm.temp_bypass[0])
        # Ignore outer duct: adiabatic boundary so it should be
        # effectively the same temp as the bypass coolant.
        if print_option:
            print(' '.join(['{:.10e}'.format(v) for v in print_list]))

        z += dz
    if print_option:
        assert 0


# def test_porous_media_method(simple_asm, conceptual_core):
#     """Test that the method to calculate interior and bypass coolant
#     temperatures performs as expected"""
#     # Set up some stuff
#     inlet_temp = 623.15
#     z = 0.0
#     # z_end = 1.281
#
#     from dassh.correlations import nusselt_db
#     conceptual_core.gap_coolant.update(inlet_temp)
#     Nu = nusselt_db.calculate_interasm_gap_sc_Nu(conceptual_core)
#     gap_htc = np.ones(2) * (conceptual_core.gap_coolant.thermal_conductivity
#                             * Nu / conceptual_core.gap_params['de'])
#     # print(gap_htc)
#     dz = dassh.axial_constraint.calculate_asm_min_dz(simple_asm,
#                                                          inlet_temp,
#                                                          inlet_temp + 150.0)
#     gap_temps = (np.ones(simple_asm.subchannel.n_sc['duct']['total'])
#                  * simple_asm.avg_coolant_int_temp)
#     simple_asm.update_coolant_int_params(inlet_temp)
#     simple_asm.duct.update(inlet_temp)
#     while z < 3.862:
#         simple_asm.calculate_temperatures(z, dz, gap_temps, gap_htc)
#         z_power = simple_asm.power.get_power(z)
#         print_list = [z,
#                       simple_asm.coolant.heat_capacity,
#                       simple_asm.coolant.thermal_conductivity,
#                       simple_asm.coolant.density,
#                       simple_asm.duct.thermal_conductivity,
#                       simple_asm.duct.heat_capacity,
#                       gap_htc[0],
#                       simple_asm.porous_media['area'] * dz,
#                       simple_asm.porous_media['R'],
#                       simple_asm.avg_coolant_int_temp_j,
#                       simple_asm.avg_duct_mw_temp_j[0]]
#         if z < 1.281 or z > 2.1233:
#             print_list.append(z_power['refl'])
#         else:
#             print_list.append(np.sum(z_power['pins']))
#             print_list.append(np.sum(z_power['duct']))
#             print_list.append(np.sum(z_power['cool']))
#         print(' '.join(['{:.10e}'.format(v) for v in print_list]))
#         z += dz
#     assert 0
#
#
# def test_calc_pm_temps(simple_asm):
#     # power = 4.0061173027e-04
#     dz = 1.1725749335e-02
#     z = 0.0
#     for i in range(30):
#         # z = 3.4004673071e-01
#         power = simple_asm.power.get_power(z)['refl']
#         porosity = 0.25
#         temp_gap = 623.15
#         htc_gap = [2.5e4]
#         dT = simple_asm.calculate_porous_media_temps(dz, power, porosity,
#                                                      temp_gap, htc_gap)
#         print(dT)
#         z += dz
#     assert 0
