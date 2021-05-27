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
date: 2021-05-27
author: matz
Test the DASSH Assembly object
"""
########################################################################
import numpy as np
import pytest
import copy
import dassh


def test_rr_flowsplit_conservation(textbook_rr):
    """Test flowsplit mass conservation requirement"""
    textbook_rr._update_coolant_int_params(700.0)
    total = 0.0
    total += (textbook_rr.coolant_int_params['fs'][0]
              * textbook_rr.params['area'][0]
              * textbook_rr.subchannel.n_sc['coolant']['interior'])
    total += (textbook_rr.coolant_int_params['fs'][1]
              * textbook_rr.params['area'][1]
              * textbook_rr.subchannel.n_sc['coolant']['edge'])
    total += (textbook_rr.coolant_int_params['fs'][2]
              * textbook_rr.params['area'][2]
              * textbook_rr.subchannel.n_sc['coolant']['corner'])
    total /= textbook_rr.bundle_params['area']
    assert np.abs(total - 1.0) <= 1e-6


def mock_AssemblyPower(rregion):
    """Fake a dictionary of linear power like that produced by the
    DASSH AssemblyPower object"""
    n_pin = rregion.n_pin
    n_duct_sc = rregion.subchannel.n_sc['duct']['total']
    n_cool_sc = rregion.subchannel.n_sc['coolant']['total']
    return {'pins': np.random.random(n_pin) * 50 + 50,
            'duct': np.random.random(n_duct_sc) * 2 + 3,
            'cool': np.random.random(n_cool_sc) + 1}


@pytest.mark.skip(reason='answers use non-costheta geom')
def test_rr_geometry_params(textbook_rr):
    """Test the attributes of a RoddedRegion object; numerical values
    taken from Nuclear Systems II textbook (Todreas); Table 4-3
    page 159"""
    # print(dir(textbook_rr))
    # for x in textbook_rr.ht_consts:
    #     print(x)
    # Do the geometric attributes exist?
    # assert hasattr(textbook_rr, 'bare_params')
    assert hasattr(textbook_rr, 'params')
    assert hasattr(textbook_rr, 'bundle_params')

    # Note here that in this method I'm subtracting a little more area
    # because I'm accounting for the wire wrap angle. We're going to
    # delete that extra contribution in the test
    # cos_theta = np.cos(textbook_rr.params['theta'])
    # x = np.array([np.pi * textbook_rr.wire_diameter**2 / 8,
    #               np.pi * textbook_rr.wire_diameter**2 / 8,
    #               np.pi * textbook_rr.wire_diameter**2 / 24])
    # # Check the values for the individual coolant subchannels
    # # Subchannel area
    # ans = np.array([10.4600, 20.9838, 7.4896])
    # ans -= x * (1 / cos_theta - 1)
    # res = textbook_rr.params['area']
    # diff = res * 1e6 - ans
    # assert diff == pytest.approx(0, abs=1e-4)
    # # Wetted perimeter
    # ans = np.array([12.4690, 20.4070, 9.6562])
    # ans -= 4 * x
    # res = textbook_rr.params['wp'] + x * (1 / cos_theta - 1)
    # diff = res * 1e3 - ans
    # assert diff == pytest.approx(0, abs=1e-4)
    # # Hydraulic diameter
    # res = textbook_rr.params['de'] + x * (1 / cos_theta - 1)
    #
    # ans = np.array([3.3555, 4.1131, 3.1025])
    # diff = textbook_rr.params['de'] * 1e3 - ans
    # assert diff == pytest.approx(0, abs=1e-4)

    # Check the bulk values
    # Bundle flow area
    ans = 1552.7
    print(textbook_rr.bundle_params['area'] * 1e6, ans)
    print(np.sqrt(3) * textbook_rr.duct_ftf[0][0]**2 / 2
          - 61 * np.pi * (textbook_rr.pin_diameter**2
                          + textbook_rr.wire_diameter**2) / 4)
    diff = textbook_rr.bundle_params['area'] * 1e6 - ans
    assert diff == pytest.approx(0, abs=1e-1)  # <-- only given 1 dec.
    # Bundle de
    ans = 3.56
    diff = textbook_rr.bundle_params['de'] * 1e3 - ans
    assert diff == pytest.approx(0, abs=1e-3)  # only given 2 dec


def test_rr_sc_areas(textbook_rr):
    """Test that the individual subchannel areas sum to the total"""
    tot = 0.0
    for i in range(textbook_rr.subchannel.n_sc['coolant']['total']):
        # sc_type = textbook_rr.subchannel.type[i] - 1
        sc_type = textbook_rr.subchannel.type[i]
        tot += textbook_rr.params['area'][sc_type]
    assert pytest.approx(tot) == textbook_rr.bundle_params['area']


def test_bypass_sc_areas(c_ctrl_rr):
    """Test that individual bypass subchannel areas sum to total"""
    tot = 0.0
    for i in range(c_ctrl_rr.n_bypass):
        start = (c_ctrl_rr.subchannel.n_sc['coolant']['total']
                 + c_ctrl_rr.subchannel.n_sc['duct']['total']
                 + i * (c_ctrl_rr.subchannel.n_sc['bypass']['total']
                        + c_ctrl_rr.subchannel.n_sc['duct']['total']))
        for j in range(c_ctrl_rr.subchannel.n_sc['bypass']['total']):
            # sc_type = c_ctrl_rr.subchannel.type[start + j] - 6
            sc_type = c_ctrl_rr.subchannel.type[start + j] - 5
            tot += c_ctrl_rr.bypass_params['area'][i][sc_type]
    assert pytest.approx(tot) == c_ctrl_rr.bypass_params['total area']


def test_double_duct_rr_geometry(c_ctrl_rr):
    """Test the geometry specifications of the double-ducted assembly
    (inner/outer ducts, bypass region)"""
    tol = 1e-9
    # Duct wall-corner lengths
    for d in range(c_ctrl_rr.n_duct):
        d_ftf = c_ctrl_rr.duct_ftf[d]
        for di in range(len(d_ftf)):
            hex_side = d_ftf[di] / np.sqrt(3)
            hex_perim = 6 * hex_side
            hex_edge_perim = \
                (c_ctrl_rr.subchannel.n_sc['duct']['edge']
                 * c_ctrl_rr.L[1][1])
            hex_corner_perim = hex_perim - hex_edge_perim
            msg = 'duct: ' + str(d) + '; wall: ' + str(di)
            assert hex_corner_perim / 12 == \
                pytest.approx(c_ctrl_rr.d['wcorner'][d][di], tol), msg

    # duct areas
    for d in range(c_ctrl_rr.n_duct):
        assert c_ctrl_rr.duct_params['total area'][d] == \
            pytest.approx(c_ctrl_rr.subchannel.n_sc['duct']['edge']
                          * c_ctrl_rr.duct_params['area'][d][0]
                          + 6 * c_ctrl_rr.duct_params['area'][d][1],
                          tol)

    # bypass subchannel params
    assert c_ctrl_rr.bypass_params['total area'][0] == \
        pytest.approx(c_ctrl_rr.subchannel.n_sc['bypass']['edge']
                      * c_ctrl_rr.bypass_params['area'][0][0]
                      + 6 * c_ctrl_rr.bypass_params['area'][0][1],
                      tol)


def test_single_pin_fail(coolant, structure):
    """Test that single pin assembly fails instantiation"""
    n_ring = 1
    pin_diameter = 0.085  # one big fat pin
    clad_thickness = 0.132 * 1.005 / 1e2  # cm -> m
    wire_pitch = 0.02
    wire_diameter = 0.0001
    duct_ftf = [0.09952, 0.10554, 0.11154, 0.11757]  # m
    pin_pitch = 0.0
    inlet_flow_rate = 5.0  # kg /s
    # rr = dassh.RoddedRegion('ctrl', n_ring, pin_pitch, pin_diameter,
    #                         wire_pitch, wire_diameter, clad_thickness,
    #                         duct_ftf, inlet_flow_rate, coolant, structure,
    #                         None, 'CTD', 'CTD', 'CTD', 'DB', byp_ff=0.1)
    with pytest.raises(SystemExit):
        dassh.RoddedRegion('ctrl', n_ring, pin_pitch, pin_diameter,
                           clad_thickness, wire_pitch, wire_diameter,
                           duct_ftf, inlet_flow_rate, coolant, structure,
                           None, None, 'CTD', 'CTD', 'CTD', 'DB')


@pytest.mark.skip(reason='No single pin functionality at the moment')
def test_double_duct_single_pin_rr():
    """This is just to see if a special case will break things"""
    loc = (5, 1)
    n_ring = 1
    pin_diameter = 0.085  # one big fat pin
    clad_thickness = 0.132 * 1.005 / 1e2  # cm -> m
    wire_pitch = 0.02
    wire_diameter = 0.0001
    duct_ftf = [0.09952, 0.10554, 0.11154, 0.11757]  # m
    pin_pitch = 0.0
    inlet_flow_rate = 5.0  # kg /s
    inlet_temp = 273.15 + 350.0  # K
    coolant_obj = dassh.Material('sodium')
    duct_obj = dassh.Material('ss316')
    asm = dassh.Assembly('ctrl', loc, n_ring, pin_pitch,
                         pin_diameter, clad_thickness, wire_pitch,
                         wire_diameter, duct_ftf, coolant_obj,
                         duct_obj, inlet_temp, inlet_flow_rate,
                         'CTD', 'CTD', 'CTD')

    # asm._cleanup_1pin()
    assert all([np.all(asm.d[key] >= 0.0) for key in asm.d.keys()])
    assert all([np.all(asm.params[key] >= 0.0)
                for key in asm.params.keys()])
    # assert all([np.all(asm.bare_params[key] >= 0.0)
    #             for key in asm.bare_params.keys()])
    assert all([np.all(asm.bundle_params[key] >= 0.0)
                for key in asm.bundle_params.keys()])
    assert all([np.all(asm.bypass_params[key] >= 0.0)
                for key in asm.bypass_params.keys()])
    assert all([np.all(asm.duct_params[key] >= 0.0)
                for key in asm.duct_params.keys()])
    for i in range(len(asm.L)):
        for j in range(len(asm.L[i])):
            assert np.all(np.array(asm.L[i][j]) >= 0.0)
            # assert np.all(np.array(asm.ht_consts[i][j]) >= 0.0)


def test_rr_duct_areas(textbook_rr):
    """."""
    tol = 1e-9
    assert textbook_rr.duct_params['total area'][0] == \
        pytest.approx(textbook_rr.subchannel.n_sc['duct']['edge']
                      * textbook_rr.duct_params['area'][0][0]
                      + 6 * textbook_rr.duct_params['area'][0][1], tol)


def test_thesis_rr_hydraulic_diam(thesis_asm_rr):
    """Test the MIT thesis assembly used for friction factor tests"""
    # Subchannel equivalent hydraulic diameter
    ans = [5.792, 7.383, 5.288]
    res = thesis_asm_rr.params['de'] * 1e3
    print('ans', ans)
    print('result', res)
    for i in range(len(res)):
        assert abs(100 * (res[i] - ans[i]) / ans[i]) < 0.01

    # Bundle average hydraulic diameter
    de_err = (thesis_asm_rr.bundle_params['de'] * 1e3 - 6.298) / 6.298
    assert abs(100 * de_err) < 0.01


def test_error_pins_fit_in_duct(c_fuel_rr, caplog):
    """Test that the RoddedRegion object throws an error if the pins
    won't fit in the duct"""
    duct_ftf = [item for sublist in c_fuel_rr.duct_ftf
                for item in sublist]
    with pytest.raises(SystemExit):
        dassh.RoddedRegion(
            'conceptual_fuel',
            c_fuel_rr.n_ring,
            c_fuel_rr.pin_pitch * 2,
            c_fuel_rr.pin_diameter,
            c_fuel_rr.wire_pitch,
            c_fuel_rr.wire_diameter,
            c_fuel_rr.clad_thickness,
            duct_ftf,
            c_fuel_rr.int_flow_rate,
            c_fuel_rr.coolant,
            c_fuel_rr.duct,
            None, None,
            'CTD', 'CTD', 'CTD', 'DB')
        assert 'Pins do not fit inside duct;' in caplog.text


def test_error_correlation_assignment(c_fuel_rr, caplog):
    """Make sure RoddedRegion fails if bad correlations are specified"""
    # Gotta specify duct in the way it will be in the DASSH_Input
    duct_ftf = [item for sublist in c_fuel_rr.duct_ftf
                for item in sublist]
    kwargs = {'corr_friction': 'CTD',
              'corr_flowsplit': 'CTD',
              'corr_mixing': 'CTD',
              'corr_nusselt': 'DB'}
    passed = 0
    for corr in kwargs.keys():
        tmp = copy.deepcopy(kwargs)
        tmp[corr] = 'X'
        with pytest.raises(SystemExit):  # Bad friction factor
            dassh.RoddedRegion(
                'conceptual_driver',
                c_fuel_rr.n_ring,
                c_fuel_rr.pin_pitch,
                c_fuel_rr.pin_diameter,
                c_fuel_rr.wire_pitch,
                c_fuel_rr.wire_diameter,
                c_fuel_rr.clad_thickness,
                duct_ftf,
                c_fuel_rr.int_flow_rate,
                c_fuel_rr.coolant,
                c_fuel_rr.duct,
                None,
                **tmp)
        passed += 1

    print(caplog.text)
    assert passed == len(kwargs)


def test_rr_clone_shallow(textbook_rr):
    """Test that assembly clone has correct shallow-copied attributes"""
    clone = textbook_rr.clone()
    non_matches = []
    # Note: These attributes are immutable and therefore won't be
    # "deepcopied" to a new position:
    # 'n_ring', 'n_pin', 'pin_pitch', 'pin_diameter',
    # 'clad_thickness', 'wire_pitch', 'wire_diameter',
    # 'n_duct', 'n_bypass', 'kappa',
    # 'int_flow_rate',
    for attr in ['pin_lattice', 'subchannel', 'params', 'bundle_params',
                 'duct_params', 'L', 'd', 'duct_ftf', 'ht']:
        id_clone = id(getattr(clone, attr))
        id_original = id(getattr(textbook_rr, attr))
        if id_clone == id_original:  # they should be the same
            continue
        else:
            non_matches.append(attr)
            print(attr, id_clone, id_original)
    assert len(non_matches) == 0


def test_rr_clone_deep(textbook_rr):
    """Test that RoddedRegion clone has deep-copied attributes"""
    clone = textbook_rr.clone()
    assert id(clone) != id(textbook_rr)
    non_matches = []
    for attr in ['coolant_int_params', 'temp', 'corr']:
        id_clone = id(getattr(clone, attr))
        id_original = id(getattr(textbook_rr, attr))
        if id_clone != id_original:  # they should be different
            continue
        else:
            non_matches.append(attr)
            print(attr, id_clone, id_original)
    assert len(non_matches) == 0


def test_assembly_clone_new_fr(textbook_rr):
    """Test behavior of clone method with new flowrate spec"""
    clone = textbook_rr.clone(12.5)
    assert clone.total_flow_rate != textbook_rr.total_flow_rate
    assert clone.total_flow_rate == 12.5
    print(clone.total_flow_rate)
    print(textbook_rr.total_flow_rate)
    non_matches = []
    for attr in ['total_flow_rate', 'int_flow_rate', 'ht']:
        id_clone = id(getattr(clone, attr))
        id_original = id(getattr(textbook_rr, attr))
        if id_clone != id_original:  # They should be different
            continue
        else:
            non_matches.append(attr)
            print(attr, id_clone, id_original)
    assert len(non_matches) == 0


def test_rr_average_temperatures(textbook_active_rr):
    """Test that I can return average duct and coolant temperatures"""
    # temperature is from conftest.py
    assert textbook_active_rr.avg_coolant_int_temp == \
        pytest.approx(300.15, 1e-9)
    print(textbook_active_rr.duct_params['total area'])
    print(textbook_active_rr.subchannel.n_sc['duct'])
    print(textbook_active_rr.duct_params['area'])
    print(textbook_active_rr.d['wcorner'])
    print(textbook_active_rr.duct_ftf)
    print(textbook_active_rr.L[1][1])
    # print(textbook_asm.duct_midwall_temp_j)
    print(textbook_active_rr.avg_duct_mw_temp)
    assert textbook_active_rr.avg_duct_mw_temp == \
        pytest.approx(300.15, 1e-9)


def test_rr_overall_average_temperatures(simple_ctrl_rr):
    """Test whether I can know the overall average coolant temp"""
    # Let's mess some of them up and see what happens ...TBD, never
    # actually implemented that one....
    print(simple_ctrl_rr.avg_coolant_temp)
    print(simple_ctrl_rr.avg_coolant_int_temp)
    # Can you format it as a string? Checking to ensure not np array
    print('{:.2f}'.format(simple_ctrl_rr.avg_coolant_temp))
    # As input, all subchannels should have temp of 623.15
    assert simple_ctrl_rr.avg_coolant_temp == pytest.approx(623.15)


def test_rr_temp_properties(c_fuel_rr):
    """Test that temperature property attributes return the correct
    structures and values"""
    # Inlet temperature: 350 + 273.15 (K)
    # Coolant internal temperatures
    ans = np.ones(c_fuel_rr.subchannel.n_sc['coolant']['total']) * 623.15
    assert np.array_equal(c_fuel_rr.temp['coolant_int'], ans)

    # Duct wall temperatures
    ans = np.ones((1, c_fuel_rr.subchannel.n_sc['duct']['total'])) * 623.15
    # print(c_fuel_rr.temp['duct_mw'])
    # print(ans)
    # print(c_fuel_rr.temp['duct_mw'].shape)
    # print(ans.shape)
    # print(c_fuel_rr.temp['duct_mw'] - ans)
    print(c_fuel_rr.temp['duct_mw'][0, 0])
    print(ans[0, 0])
    np.testing.assert_array_almost_equal(
        c_fuel_rr.temp['duct_mw'], ans, decimal=12)

    # Duct outer surface temperatures
    ans = np.ones(c_fuel_rr.subchannel.n_sc['duct']['total']) * 623.15
    np.testing.assert_array_almost_equal(
        c_fuel_rr.duct_outer_surf_temp, ans, decimal=12)

    # Duct surface temperatures
    ans = 623.15 * np.ones((c_fuel_rr.n_duct, 2,
                            c_fuel_rr.subchannel.n_sc['duct']['total']))
    np.testing.assert_array_almost_equal(
        c_fuel_rr.temp['duct_surf'], ans)


def test_asm_zero_power(c_fuel_rr):
    """Test that the power put into the coolant subchannels is zero"""
    pcoolant = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
    ppin = np.zeros(c_fuel_rr.n_pin)
    res = c_fuel_rr._calc_int_sc_power(ppin, pcoolant)
    assert not np.any(res)  # all should be zero


def test_rr_none_power(c_fuel_rr):
    """Test that correct power is delivered to subchannels if pin and/or
    coolant power is None"""
    # Both pin and coolant power are None: result is zero power
    res = c_fuel_rr._calc_int_sc_power(None, None)
    zero_power = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
    assert np.array_equal(res, zero_power)
    # Pin power is None: result is coolant power
    pcool = np.random.random(c_fuel_rr.subchannel.n_sc['coolant']['total'])
    res = c_fuel_rr._calc_int_sc_power(None, pcool)
    assert np.allclose(res, pcool)
    # Coolant power is None: result is distributed pin power as if no coolant
    # power is specified; use zero coolant power to check
    ppins = np.random.random(c_fuel_rr.subchannel.n_sc['coolant']['total'])
    res = c_fuel_rr._calc_int_sc_power(ppins, None)
    ans = c_fuel_rr._calc_int_sc_power(ppins, zero_power)
    assert np.allclose(res, ans)


def test_zero_power_coolant_int_temp(c_fuel_rr):
    """Test that the internal coolant temperature calculation
    with no heat generation returns no temperature change"""
    pcoolant = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
    pin_power = np.zeros(c_fuel_rr.n_pin)
    res = c_fuel_rr._calc_coolant_int_temp(0.5, pin_power, pcoolant)
    # Temperature should be unchanged relative to the previous level
    # (res is delta T)
    assert np.allclose(res, 0.0)


def test_none_power_coolant_int_temp(c_fuel_rr):
    """Test that the internal coolant temperature calculation with None
    power for pins/coolant returns no temperature change"""
    res = c_fuel_rr._calc_coolant_int_temp(0.5, None, None)
    assert np.allclose(res, 0.0)


def test_zero_power_coolant_interior_adj_temp(c_fuel_rr):
    """Test that if only one subchannel has nonzero dT, only the
    adjacent channels are affected"""
    inlet_temp = 623.15
    outlet_temp = inlet_temp + 150.0
    perturb_temp = 1.0
    z = 1.0
    dz, sc = dassh.region_rodded.calculate_min_dz(c_fuel_rr,
                                                  inlet_temp,
                                                  outlet_temp)
    print('dz = ' + str(dz) + '\n')
    unperturbed_temperature = c_fuel_rr.temp['coolant_int']
    coolant_power = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
    pin_power = np.zeros(c_fuel_rr.n_pin)
    for sc in range(c_fuel_rr.subchannel.n_sc['coolant']['interior'] - 1):
        # adj_sc = c_fuel_rr.subchannel.sc_adj[sc] - 1
        adj_sc = c_fuel_rr.subchannel.sc_adj[sc]

        # Perturb the temperature, calculate new temperatures, then
        # unperturb the temperature
        c_fuel_rr.temp['coolant_int'][sc] += perturb_temp
        res = c_fuel_rr._calc_coolant_int_temp(z, pin_power, coolant_power)
        print(res.shape)
        c_fuel_rr.temp['coolant_int'][sc] -= perturb_temp
        assert np.allclose(c_fuel_rr.temp['coolant_int'],
                           unperturbed_temperature)

        dT = []
        m = []
        for s in range(len(res)):  # only does coolant channels
            if s in adj_sc or s == sc:
                # s_type = c_fuel_rr.subchannel.type[s] - 1
                s_type = c_fuel_rr.subchannel.type[s]
                print(s, s_type, inlet_temp, res[s])
                dT.append(res[s])
                m.append(c_fuel_rr.coolant_int_params['fs'][s_type]
                         * c_fuel_rr.int_flow_rate
                         * c_fuel_rr.params['area'][s_type]
                         / c_fuel_rr.bundle_params['area'])
                # assert res[s] != 0.0
                assert res[s] != pytest.approx(0.0, abs=1e-10)

            else:
                # assert res[s] == 0.0
                assert res[s] == pytest.approx(0.0, abs=1e-10)
        # Assert the balance
        mdT = [m[i] * dT[i] for i in range(len(dT))]
        print('dT: ' + str(dT))
        print('mdT: ' + str(mdT))
        print('bal: ' + str(sum(mdT)))
        print('\n')
        assert np.abs(sum(mdT)) == pytest.approx(0.0, abs=1e-10)


def test_coolant_pin_power(c_fuel_rr):
    """Test that the internal coolant subchannel power method reports
    the proper total power"""
    power = mock_AssemblyPower(c_fuel_rr)
    ans = np.sum(power['pins']) + np.sum(power['cool'])
    # print('cool: ' + str(power['cool']))
    # print('pins: ' + str(power['pins']))
    print('ans: ' + str(ans))
    res = c_fuel_rr._calc_int_sc_power(power['pins'], power['cool'])
    print(res)
    print('result: ' + str(np.sum(res)))
    assert np.sum(res) == pytest.approx(ans)


def test_coolant_temp_w_pin_power_indiv(c_fuel_rr):
    """Test that the internal coolant temperature calculation
    with no heat generation returns no temperature change"""
    tmp_asm = c_fuel_rr.clone()
    T_in = 623.15
    T_out = T_in + 150.0
    c_fuel_rr
    power = mock_AssemblyPower(c_fuel_rr)
    dz, sc = dassh.region_rodded.calculate_min_dz(tmp_asm, T_in, T_out)

    # Power added overall
    ans = dz * (np.sum(power['pins']) + np.sum(power['cool']))

    # Calculate new temperatures
    dT = tmp_asm._calc_coolant_int_temp(dz, power['pins'], power['cool'])

    # Calculate Q = mCdT in each channel
    Q = 0.0
    for sc in range(len(dT)):
        # sc_type = tmp_asm.subchannel.type[sc] - 1
        sc_type = tmp_asm.subchannel.type[sc]
        mfr = (tmp_asm.coolant_int_params['fs'][sc_type]
               * tmp_asm.int_flow_rate
               * tmp_asm.params['area'][sc_type]
               / tmp_asm.bundle_params['area'])
        Q += mfr * tmp_asm.coolant.heat_capacity * dT[sc]

    print('dz (m): ' + str(dz))
    print('Power added (W): ' + str(ans))
    print('Mdot (kg/s): ' + str(tmp_asm.int_flow_rate))
    print('Cp (J/kgK): ' + str(tmp_asm.coolant.heat_capacity))
    print('Power result (W): ' + str(Q))
    assert ans == pytest.approx(Q)


def test_coolant_temp_rougly_qmcdt(c_fuel_rr):
    """Test that the change in interior subchannel coolant temperature
    over one axial step roughly approximates Q = mCdT"""
    tmp_asm = c_fuel_rr.clone()
    inlet_temp = 623.15
    outlet_temp = inlet_temp + 150.0
    power = mock_AssemblyPower(c_fuel_rr)
    dz, sc = dassh.region_rodded.calculate_min_dz(tmp_asm, inlet_temp,
                                                  outlet_temp)

    # Power added overall
    ans = dz * (np.sum(power['pins']) + np.sum(power['cool']))

    # Calculate average dT
    dT = tmp_asm._calc_coolant_int_temp(dz, power['pins'], power['cool'])
    tot = 0.0
    for i in range(len(tmp_asm.temp['coolant_int'])):
        # tot += (tmp_asm.params['area'][tmp_asm.subchannel.type[i] - 1]
        #         * dT[i])
        tot += tmp_asm.params['area'][tmp_asm.subchannel.type[i]] * dT[i]
    dT = tot / tmp_asm.bundle_params['area']
    # Q = mCdT in the coolant
    res = tmp_asm.int_flow_rate * tmp_asm.coolant.heat_capacity * dT

    print('dz (m): ' + str(dz))
    print('Power added (W): ' + str(ans))
    print('Mdot (kg/s): ' + str(tmp_asm.int_flow_rate))
    print('Cp (J/kgK): ' + str(tmp_asm.coolant.heat_capacity))
    print('Tout (K): ' + str(dT))
    print('res (W): ' + str(res))
    assert ans == pytest.approx(res, 1e-2)


def test_zero_power_coolant_pertub_wall_temp(c_fuel_rr):
    """Test that if wall temperature is perturbed, only the adjacent
    edge/corner coolant subchannel has temperature change"""
    inlet_temp = 623.15
    outlet_temp = inlet_temp + 150.0
    perturb_temp = 100.0

    c_fuel_rr._update_coolant_int_params(inlet_temp)
    # print(c_fuel_rr.coolant.temperature)
    dz, sc = dassh.region_rodded.calculate_min_dz(c_fuel_rr,
                                                  inlet_temp,
                                                  outlet_temp)
    p_coolant = np.zeros(c_fuel_rr.subchannel.n_sc['coolant']['total'])
    p_pin = np.zeros(c_fuel_rr.n_pin)

    htc = c_fuel_rr.coolant_int_params['htc']
    cp = c_fuel_rr.coolant.heat_capacity
    fr = (c_fuel_rr.int_flow_rate
          * c_fuel_rr.coolant_int_params['fs']
          * c_fuel_rr.params['area']
          / c_fuel_rr.bundle_params['area'])
    OLD_HTCONSTS = dassh.region_rodded.calculate_ht_constants(c_fuel_rr)
    A = np.zeros(3)
    A[1] = c_fuel_rr.L[1][1] * dz
    # A[2] = c_fuel_rr.d['wcorner_m'][0] * 2 * dz
    A[2] = c_fuel_rr.d['wcorner'][0, 1] * 2 * dz
    # Loop over wall sc, perturb each one
    for w_sc in range(c_fuel_rr.subchannel.n_sc['duct']['total']):
        idx_sc_type = w_sc + c_fuel_rr.subchannel.n_sc['coolant']['total']
        # adj_sc = c_fuel_rr.subchannel.sc_adj[idx_sc_type] - 1
        adj_sc = c_fuel_rr.subchannel.sc_adj[idx_sc_type]

        # Perturb the duct temperature, calculate new coolant temps,
        # unperturb the duct temperature at the end
        # Index: first duct wall, inner surface
        c_fuel_rr.temp['duct_surf'][0, 0, w_sc] += perturb_temp
        res = c_fuel_rr._calc_coolant_int_temp(dz, p_pin, p_coolant)
        for s in range(len(res)):  # only does coolant channels
            if s in adj_sc:
                # s_type = c_fuel_rr.subchannel.type[s] - 1
                s_type = c_fuel_rr.subchannel.type[s]
                test = (A[s_type] * htc[s_type] * perturb_temp
                        / fr[s_type] / cp)
                if not res[s] == pytest.approx(test):
                    print('dz = ' + str(dz) + '\n')
                    print(c_fuel_rr.subchannel.n_sc)
                    print(OLD_HTCONSTS[s_type][2 + s_type])
                    print('htc expected: ' + str(htc))
                    print('cp expected: ' + str(cp))
                    print('fs expected: '
                          + str(c_fuel_rr.coolant_int_params['fs']))
                    print('wall sc: ' + str(idx_sc_type))
                    print('wall adj: ' + str(adj_sc))
                    print('wall temp: '
                          + str(c_fuel_rr.temp['duct_surf'][0, 0, w_sc]))
                    print('cool sc: ' + str(s))
                    print('cool sc type: ' + str(s_type))
                    print('cool in temp: '
                          + str(c_fuel_rr.temp['coolant_int'][s]))
                    print('cool out temp: ' + str(res[s] + inlet_temp))
                    print('test: ' + str(test))
                assert res[s] == pytest.approx(test)
            else:
                assert res[s] == pytest.approx(0.0)

        # Unperturb the temperature
        c_fuel_rr.temp['duct_surf'][0, 0, w_sc] -= perturb_temp


def test_zero_power_duct_temp(c_fuel_rr):
    """Test that the internal coolant temperature calculation
    with no heat generation returns no temperature change"""
    gap_temps = (np.ones(c_fuel_rr.subchannel.n_sc['duct']['total'])
                 * c_fuel_rr.avg_coolant_int_temp)
    c_fuel_rr._update_coolant_int_params(623.15)
    gap_htc = c_fuel_rr.coolant_int_params['htc'][1:]
    gap_htc = gap_htc[
        c_fuel_rr.subchannel.type[
            c_fuel_rr.subchannel.n_sc['coolant']['interior']:
            c_fuel_rr.subchannel.n_sc['coolant']['total']] - 1]
    duct_power = np.zeros(c_fuel_rr.n_duct
                          * c_fuel_rr.subchannel.n_sc['duct']['total'])
    c_fuel_rr._calc_duct_temp(duct_power, gap_temps, gap_htc)
    # res_mw = zero_pow_asm.duct_midwall_temp
    # res_surf = zero_pow_asm.duct_surface_temp

    print('duct k (W/mK): ' + str(c_fuel_rr.duct.thermal_conductivity))
    print('duct thickness (m): ' + str(c_fuel_rr.duct_params['thickness']))

    # Temperature should be unchanged relative to the previous level
    assert np.allclose(c_fuel_rr.temp['duct_mw'], 623.15)
    assert np.allclose(c_fuel_rr.temp['duct_surf'], 623.15)


def test_duct_temp_w_power_indiv(c_fuel_rr):
    """Test that duct temperature calculation gives reasonable result
    with power assignment and no temperature differential"""
    # gap_temps = (np.ones(c_fuel_rr.subchannel.n_sc['duct']['total'])
    #              * (c_fuel_rr.avg_coolant_int_temp - 5.0))
    gap_temps = (np.ones(c_fuel_rr.subchannel.n_sc['duct']['total'])
                 * c_fuel_rr.avg_coolant_int_temp)
    c_fuel_rr._update_coolant_int_params(623.15)
    gap_htc = c_fuel_rr.coolant_int_params['htc'][1:]
    gap_htc = gap_htc[
        c_fuel_rr.subchannel.type[
            c_fuel_rr.subchannel.n_sc['coolant']['interior']:
            c_fuel_rr.subchannel.n_sc['coolant']['total']] - 1]
    inlet_temp = 623.15
    outlet_temp = inlet_temp + 150.0
    dz, sc = dassh.region_rodded.calculate_min_dz(c_fuel_rr,
                                                  inlet_temp,
                                                  outlet_temp)

    # Power added overall
    power = mock_AssemblyPower(c_fuel_rr)
    ans = np.sum(power['duct'] * dz)

    # Calculate new temperatures
    c_fuel_rr._calc_duct_temp(power['duct'], gap_temps, gap_htc)
    dT_s = c_fuel_rr.temp['duct_surf'] - inlet_temp

    # print(c_fuel_rr.temp['duct_surf'][0, 0, 0])
    # print(c_fuel_rr.temp['duct_mw'][0, 0])
    # print(c_fuel_rr.temp['duct_surf'][0, 1, 0])
    # assert 0

    # Steady state - all heat added to ducts is leaving ducts via
    # convection to the adjacent coolant
    surface_area = np.array([[c_fuel_rr.L[1][1] * dz,
                              c_fuel_rr.L[1][1] * dz],
                             [2 * c_fuel_rr.d['wcorner'][0, 0] * dz,
                              2 * c_fuel_rr.d['wcorner'][0, 1] * dz]])

    Q = 0.0
    start = c_fuel_rr.subchannel.n_sc['coolant']['total']
    for i in range(c_fuel_rr.n_duct):
        for sc in range(c_fuel_rr.subchannel.n_sc['duct']['total']):
            # sc_type = c_fuel_rr.subchannel.type[sc + start] - 4
            sc_type = c_fuel_rr.subchannel.type[sc + start] - 3
            htc = c_fuel_rr.coolant_int_params['htc'][1:][sc_type]
            qtmp_in = htc * dT_s[0, 0, sc] * surface_area[sc_type, 0]
            qtmp_out = htc * dT_s[0, 1, sc] * surface_area[sc_type, 1]
            # if sc == 0:
            #    print(q[sc], sc_type, htc, dT_s[0, 0, sc],
            #          qtmp_in, dT_s[0, 1, sc], qtmp_out)
            Q += qtmp_in + qtmp_out
    assert ans == pytest.approx(Q)


def test_duct_temp_w_power_adiabatic(c_fuel_rr):
    """Test adiabatic flag for duct temp calculation"""
    # 2020-12-09: removed coolant parameter update from duct temp
    # method so I need to do it externally here...no biggie.
    c_fuel_rr._update_coolant_int_params(
        c_fuel_rr.avg_coolant_int_temp)
    p_duct = np.ones(c_fuel_rr.subchannel.n_sc['duct']['total']) * 1000.0
    t_gap = np.zeros(c_fuel_rr.subchannel.n_sc['duct']['total'])
    c_fuel_rr._calc_duct_temp(p_duct, t_gap, np.ones(2), True)
    # Not a "real" average but it'll do the trick
    print(np.average(c_fuel_rr.temp['duct_surf'][0, 0]))
    print(np.average(c_fuel_rr.temp['duct_mw'][0]))
    print(np.average(c_fuel_rr.temp['duct_surf'][0, 1]))
    # midwall temp should be higher than inner wall temp
    assert np.all(c_fuel_rr.temp['duct_mw'][0]
                  > c_fuel_rr.temp['duct_surf'][0, 0])
    # outer wall temp should be highest of all!
    assert np.all(c_fuel_rr.temp['duct_mw'][0]
                  < c_fuel_rr.temp['duct_surf'][0, 1])


def test_byp_coolant_temps_zero_dT(c_ctrl_rr):
    """Test that bypass coolant temperatures are unchanged when
    adjacent ducts have equal temperature"""
    print(c_ctrl_rr.n_bypass)
    print(c_ctrl_rr.temp['duct_surf'].shape)
    dT_byp = c_ctrl_rr._calc_coolant_byp_temp(1.0)
    assert np.allclose(dT_byp, 0.0)


def test_bypass_perturb_wall_temps(c_ctrl_rr):
    """Test that perturbations in adjacent wall mesh cells affect
    only adjacent bypass coolant subchannels"""
    inlet_temp = 623.15
    outlet_temp = inlet_temp + 150.0
    c_ctrl_rr._update_coolant_byp_params([inlet_temp])
    perturb_temp = 100.0
    dz, sc = dassh.region_rodded.calculate_min_dz(c_ctrl_rr,
                                                  inlet_temp,
                                                  outlet_temp)
    htc = c_ctrl_rr.coolant_byp_params['htc'][0]
    cp = c_ctrl_rr.coolant.heat_capacity
    fr = (c_ctrl_rr.byp_flow_rate[0]
          * c_ctrl_rr.bypass_params['area'][0]
          / c_ctrl_rr.bypass_params['total area'][0])
    OLD_HTCONSTS = dassh.region_rodded.calculate_ht_constants(c_ctrl_rr)
    A = np.zeros((2, 2))
    A[0, 0] = c_ctrl_rr.L[5][5][0] * dz
    # A[0, 1] = c_ctrl_rr.d['wcorner_m'][0] * 2 * dz
    A[0, 1] = c_ctrl_rr.d['wcorner'][0, 1] * 2 * dz
    A[1, 0] = A[0, 0]
    # A[1, 1] = c_ctrl_rr.d['wcorner_m'][1] * 2 * dz
    A[1, 1] = c_ctrl_rr.d['wcorner'][1, 1] * 2 * dz

    surf = {0: 1, 1: 0}
    byp_types = {6: 'edge', 7: 'corner'}
    # Loop over wall sc, perturb each one
    for i in range(c_ctrl_rr.n_duct):
        for w_sc in range(c_ctrl_rr.subchannel.n_sc['duct']['total']):
            # index of the adjacent coolant subchannel
            wsc_type_idx = \
                (w_sc
                 + c_ctrl_rr.subchannel.n_sc['coolant']['total']
                 + i * c_ctrl_rr.subchannel.n_sc['duct']['total']
                 + i * c_ctrl_rr.subchannel.n_sc['bypass']['total'])

            # find subchannels next to the wall
            # adj_sc = c_ctrl_rr.subchannel.sc_adj[wsc_type_idx] - 1
            adj_sc = c_ctrl_rr.subchannel.sc_adj[wsc_type_idx]

            # Perturb the duct temperature, calculate new coolant
            # temps; unperturb the duct temperature at the end
            # Index: i-th duct, surface (in: 0; out: 1), last position
            c_ctrl_rr.temp['duct_surf'][i, surf[i], w_sc] += perturb_temp
            res = c_ctrl_rr._calc_coolant_byp_temp(dz)

            # Loop over all bypass coolant results - if the channel
            # is adjacent to the perturbed wall, we should know its
            # temperature; otherwise, its temperature should be
            # unchanged.
            for s in range(len(res[0])):
                byp_idx = \
                    (s
                     + c_ctrl_rr.subchannel.n_sc['coolant']['total']
                     + c_ctrl_rr.subchannel.n_sc['duct']['total'])
                if byp_idx in adj_sc:
                    # Determine what type of bypass channel you have
                    # s_type = c_ctrl_rr.subchannel.type[byp_idx] - 1
                    s_type = c_ctrl_rr.subchannel.type[byp_idx]
                    # Expected dT - only conv from the one wall=
                    # test = (A[i, s_type - 5]
                    #         * htc[s_type - 5]
                    #         * perturb_temp
                    #         / fr[s_type - 5] / cp)
                    test = (A[i, s_type - 5]
                            * htc[s_type - 5]
                            * perturb_temp
                            / fr[s_type - 5] / cp)
                    if not res[0, s] == pytest.approx(test):
                        print('dz = ' + str(dz))
                        print('byp sc: ' + str(byp_idx + 1))
                        # print('byp sc type: ' + str(byp_types[byp_idx + 1]))
                        print('byp sc type: ' + str(s_type + 1)
                              + '; ' + str(byp_types[s_type + 1]))
                        print('perturbed wall sc: ' + str(wsc_type_idx + 1))
                        print('perturbed wall adj: ' + str(adj_sc + 1))
                        print('htc const: ' +
                              str(OLD_HTCONSTS[s_type][s_type - 2]))
                        print('htc expected: ' + str(htc))
                        print('cp expected: ' + str(cp))
                        print('fr expected: ' + str(fr))
                        print('area: ' + str(A[i, s_type - 5]))
                        print('wall temp: '
                              + str(c_ctrl_rr
                                    .temp['duct_surf'][i, surf[i], s]))
                        print('byp in temp: '
                              + str(c_ctrl_rr
                                    .temp['coolant_byp'][0, -1, s]))
                        print('byp out temp: ' + str(res[0, 0, s]))
                        print('test: ' + str(test))
                        assert res[0, s] == pytest.approx(test)
                        # assert res[s] != inlet_temp
                else:
                    assert res[0, s] == 0.0

            # Unperturb the temperature
            c_ctrl_rr.temp['duct_surf'][i, surf[i], w_sc] -= perturb_temp


def test_accelerated_coolant_sc_method_against_old(c_fuel_rr):
    """Confirm numpy coolant subchannel calculation gets same result
    as the old one (this let's me preserve the old just in case)"""
    OLD_HTCONSTS = dassh.region_rodded.calculate_ht_constants(c_fuel_rr)

    def calc_coolant_int_temp_old(self, dz, pin_power, cool_power):
        """Calculate assembly coolant temperatures at next axial mesh

        Parameters
        ----------
        self : DASSH RoddedRegion object
        dz : float
            Axial step size (m)
        pin_power : numpy.ndarray
            Linear power generation (W/m) for each pin in the assembly
        cool_power : numpy.ndarray
            Linear power generation (W/m) for each coolant subchannel

        Returns
        -------
        numpy.ndarray
            Vector (length = # coolant subchannels) of temperatures
            (K) at the next axial level

        """
        # Calculate avg coolant temperature; update coolant properties
        # self._update_coolant_int_params(self.avg_coolant_int_temp)

        # Power from pins and neutron/gamma reactions with coolant
        q = self._calc_int_sc_power(pin_power, cool_power)

        # PRECALCULATE CONSTANTS ---------------------------------------
        # Effective thermal conductivity
        keff = (self.coolant_int_params['eddy']
                * self.coolant.density
                * self.coolant.heat_capacity
                + self._sf * self.coolant.thermal_conductivity)

        # This factor is in many terms; technically, the mass flow
        # rate is already accounted for in constants defined earlier
        mCp = self.coolant.heat_capacity * self.coolant_int_params['fs']

        # The mass flow rate denominator hasn't been calculated yet for
        # the q term, so do that now (store this eventually)
        heat_added_denom = [self.int_flow_rate
                            * self.params['area'][i]
                            / self.bundle_params['area'] for i in range(3)]

        # Precalculate some other stuff
        conduction_consts = [[OLD_HTCONSTS[i][j] * keff
                              for j in range(3)] for i in range(3)]
        convection_consts = [OLD_HTCONSTS[i][i + 2]
                             * self.coolant_int_params['htc'][i]
                             for i in range(3)]
        swirl_consts = [self.coolant.density
                        * self.coolant_int_params['swirl'][i]
                        * self.d['pin-wall']
                        * self.bundle_params['area']
                        / self.coolant_int_params['fs'][i]
                        / self.params['area'][i]
                        / self.int_flow_rate
                        for i in range(3)]

        # Calculate the change in temperature in each subchannel
        dT = np.zeros(self.subchannel.n_sc['coolant']['total'])
        for sci in range(self.subchannel.n_sc['coolant']['total']):

            # The value of sci is the PYTHON indexing
            type_i = self.subchannel.type[sci]

            # Heat from adjacent fuel pins
            dT[sci] += q[sci] / heat_added_denom[type_i]

            for adj in self.subchannel.sc_adj[sci]:
                # if adj == 0:
                if adj == -1:
                    continue

                # Adjacent cell type in PYTHON indexing
                type_a = self.subchannel.type[adj]

                # Conduction to/from adjacent coolant subchannels
                if type_a <= 2:
                    dT[sci] += (conduction_consts[type_i][type_a]
                                * (self.temp['coolant_int'][adj]
                                   - self.temp['coolant_int'][sci]))

                # Convection to/from duct wall (type has to be 3 or 4)
                else:
                    sc_wi = sci - (self.subchannel.n_sc['coolant']
                                                       ['interior'])
                    dT[sci] += (convection_consts[type_i]
                                * (self.temp['duct_surf'][0, 0, sc_wi]
                                   - self.temp['coolant_int'][sci]))

            # Divide through by mCp
            dT[sci] /= mCp[type_i]

            # Swirl flow from adjacent subchannel; =0 for interior sc
            # The adjacent subchannel is the one the swirl flow is
            # coming from i.e. it's in the opposite direction of the
            # swirl flow itself. Recall that the edge/corner sub-
            # channels are indexed in the clockwise direction.
            # Example: Let sci == 26. The next subchannel in the clock-
            # wise direction is 27; the preceding one is 25.
            # - clockwise: use 25 as the swirl adjacent sc
            # - counterclockwise: use 27 as the swirl adjacent sc
            if type_i > 0:
                dT[sci] += \
                    (swirl_consts[type_i]
                     * (self.temp['coolant_int']
                                 [self.subchannel.sc_adj[sci]
                                                        [self._adj_sw]]
                        - self.temp['coolant_int'][sci]))
        return dT * dz

    dT = np.zeros(len(c_fuel_rr.temp['coolant_int']))
    dT_old = copy.deepcopy(dT)
    c_fuel_rr_old = copy.deepcopy(c_fuel_rr)

    dz = 0.01
    for i in range(50):
        pin_power = 2e4 + 5e3 * np.random.random(c_fuel_rr.n_pin)
        cool_power = 1.5e3 + 500 * np.random.random(len(dT))
        dT_old += calc_coolant_int_temp_old(
            c_fuel_rr_old, dz, pin_power, cool_power)
        dT += c_fuel_rr._calc_coolant_int_temp(
            dz, pin_power, cool_power)

    print(np.average(dT))
    print('max abs diff: ', np.max(np.abs(dT - dT_old)))
    assert np.allclose(dT, dT_old)


def test_accelerated_bypass_method_against_old(c_ctrl_rr):
    """Confirm that my changes to the bypass method maintain the same
    result as the old method"""
    OLD_HTCONSTS = dassh.region_rodded.calculate_ht_constants(c_ctrl_rr)

    def _calc_coolant_byp_temp_old(self, dz):
        """Calculate the coolant temperatures in the assembly bypass
        channels at the axial level j+1

        Parameters
        ----------
        self : DASSH RoddedRegion object
        dz : float
            Axial step size (m)

        Notes
        -----
        The coolant in the bypass channels is assumed to get no
        power from neutron/gamma heating (that contribution to
        coolant in the assembly interior is already small enough).

        """
        # Calculate the change in temperature in each subchannel
        dT = np.zeros((self.n_bypass,
                       self.subchannel.n_sc['bypass']['total']))
        # self._update_coolant_byp_params(self.avg_coolant_byp_temp)
        for i in range(self.n_bypass):

            # This factor is in many terms; technically, the mass flow
            # rate is already accounted for in constants defined earlier
            # mCp = self.coolant.heat_capacity

            # starting index to lookup type is after all interior
            # coolant channels and all preceding duct and bypass
            # channels
            start = (self.subchannel.n_sc['coolant']['total']
                     + self.subchannel.n_sc['duct']['total']
                     + i * self.subchannel.n_sc['bypass']['total']
                     + i * self.subchannel.n_sc['duct']['total'])

            # end = start + self.subchannel.n_sc['bypass']['total']
            for sci in range(0, self.subchannel.n_sc['bypass']['total']):

                # The value of sci is the PYTHON indexing
                # type_i = self.subchannel.type[sci + start] - 1
                type_i = self.subchannel.type[sci + start]

                # Heat transfer to/from adjacent subchannels
                for adj in self.subchannel.sc_adj[sci + start]:
                    # if adj == 0:
                    if adj == -1:
                        continue
                    # type_a = self.subchannel.type[adj - 1] - 1
                    type_a = self.subchannel.type[adj]

                    # Convection to/from duct wall
                    # if type_a in [3, 4]:
                    if 3 <= type_a <= 4:
                        if sci + start > adj:  # INTERIOR adjacent duct wall
                            byp_conv_const = \
                                OLD_HTCONSTS[type_i][type_a][i][0]
                            byp_conv_dT = \
                                (self.temp['duct_surf'][i, 1, sci]
                                 - self.temp['coolant_byp'][i, sci])
                        else:  # EXTERIOR adjacent duct wall
                            byp_conv_const = \
                                OLD_HTCONSTS[type_i][type_a][i][1]
                            byp_conv_dT = \
                                (self.temp['duct_surf'][i + 1, 0, sci]
                                 - self.temp['coolant_byp'][i, sci])

                        dT[i, sci] += \
                            (self.coolant_byp_params['htc'][i, type_i - 5]
                             * dz * byp_conv_const * byp_conv_dT
                             / self.coolant.heat_capacity)

                    # Conduction to/from adjacent coolant subchannels
                    else:
                        # sc_adj = adj - start - 1
                        sc_adj = adj - start
                        dT[i, sci] += \
                            (self.coolant.thermal_conductivity
                             * dz
                             * OLD_HTCONSTS[type_i][type_a][i]
                             * (self.temp['coolant_byp'][i, sc_adj]
                                - self.temp['coolant_byp'][i, sci])
                             / self.coolant.heat_capacity)

        return dT

    dT = np.zeros(c_ctrl_rr.temp['coolant_byp'].shape)
    dT_old = copy.deepcopy(dT)
    c_fuel_rr_old = copy.deepcopy(c_ctrl_rr)
    dz = 0.01
    start_temp = 623.15
    for i in range(50):
        duct_surf_temp = \
            (np.random.random(c_ctrl_rr.temp['duct_surf'].shape)
             + (start_temp + i * 1.0))

        c_ctrl_rr.temp['duct_surf'] = duct_surf_temp
        c_fuel_rr_old.temp['duct_surf'] = duct_surf_temp
        dT_old += _calc_coolant_byp_temp_old(c_fuel_rr_old, dz)
        dT += c_ctrl_rr._calc_coolant_byp_temp(dz)

    print(np.average(dT))
    print(np.average(dT_old))
    print('max abs diff: ', np.max(np.abs(dT - dT_old)))
    assert np.allclose(dT, dT_old)


@pytest.mark.skip(reason='milos is playing with this')
def test_bypass_iterate(c_ctrl_rr):
    """."""
    # i = 0
    # print(c_ctrl_rr.bundle_params['area'])
    # print(c_ctrl_rr.bypass_params['total area'])
    # c_ctrl_rr._update_coolant_int_params(623.15)
    # print(c_ctrl_rr.coolant_int_params['ff'])
    # c_ctrl_rr._update_coolant_byp_params([623.15])
    # print(c_ctrl_rr.coolant_byp_params['ff'])

    k = 650.0
    c_ctrl_rr.__iterate_bypass_flowrate(1.281, 2.9095, [k])
    # c_ctrl_rr.iterate_bypass_flowrate(1.281, 2.9095, [k])
    # c_ctrl_rr.iterate_bypass_flowrate(1.281, 2.9095, [k])
    # c_ctrl_rr.iterate_bypass_flowrate(1.281, 2.9095, [k])
    # c_ctrl_rr.iterate_bypass_flowrate(1.281, 2.9095, [k])
    assert 0

#
# @pytest.mark.skip(reason='because')
# def test_double_ducted_asm_dz(ctrl_asm):
#     """Test the dz requirement for a double ducted asm"""
#     ctrl_asm.update_coolant_int_params(623.15)
#     pd = ctrl_asm.pin_pitch / ctrl_asm.pin_diameter
#     d_p2p = ctrl_asm.d["pin-pin"]
#
#     sc_mfr = [ctrl_asm.int_flow_rate
#               * ctrl_asm.coolant_int_params['fs'][i]
#               * ctrl_asm.params['area'][i]
#               / ctrl_asm.bundle_params['area']
#               for i in range(len(ctrl_asm.coolant_int_params['fs']))]
#     print(sc_mfr)
#     print(ctrl_asm.coolant_int_params['fs'])
#     print(ctrl_asm.bare_params['area'])
#     # assert 0
#     # Calculate "effective" thermal conductivity
#     keff = (ctrl_asm.kappa * ctrl_asm.coolant.thermal_conductivity
#             + (ctrl_asm.coolant.density * ctrl_asm.coolant.heat_capacity
#                * ctrl_asm.coolant_int_params['eddy']))
#     min_dz = []
#     min_dz.append(dassh.assembly._cons1_111(sc_mfr[0],
#                                             ctrl_asm.L[0][0],
#                                             ctrl_asm.d['pin-pin'],
#                                             keff,
#                                             ctrl_asm.coolant.heat_capacity))
#     min_dz.append(dassh.assembly._cons1_112(sc_mfr[0],
#                                             ctrl_asm.L[0][0],
#                                             ctrl_asm.L[0][1],
#                                             ctrl_asm.d['pin-pin'],
#                                             keff,
#                                             ctrl_asm.coolant.heat_capacity))
#     min_dz.append(dassh.assembly._cons3_22(sc_mfr[2],
#                                            ctrl_asm.L[1][2],
#                                            ctrl_asm.d['pin-wall'],
#                                            ctrl_asm.d['wcorner'][0][0], keff,
#                                            ctrl_asm.coolant.heat_capacity,
#                                            ctrl_asm.coolant.density,
#                                            ctrl_asm.coolant_int_params['htc'][2],
#                                            ctrl_asm.coolant_int_params['swirl'][2]))
#     min_dz.append(dassh.assembly._cons2_123(sc_mfr[1],
#                                             ctrl_asm.L[1][0],
#                                             ctrl_asm.L[1][1],
#                                             ctrl_asm.L[1][2],
#                                             ctrl_asm.d['pin-pin'],
#                                             ctrl_asm.d['pin-wall'], keff,
#                                             ctrl_asm.coolant.heat_capacity,
#                                             ctrl_asm.coolant.density,
#                                             ctrl_asm.coolant_int_params['htc'][1],
#                                             ctrl_asm.coolant_int_params['swirl'][1]))
#     min_dz.append(dassh.assembly._cons2_122(sc_mfr[1],
#                                             ctrl_asm.L[1][0],
#                                             ctrl_asm.L[1][1],
#                                             ctrl_asm.d['pin-pin'],
#                                             ctrl_asm.d['pin-wall'], keff,
#                                             ctrl_asm.coolant.heat_capacity,
#                                             ctrl_asm.coolant.density,
#                                             ctrl_asm.coolant_int_params['htc'][1],
#                                             ctrl_asm.coolant_int_params['swirl'][1]))
#     c322 = min_dz[2]
#     butt = [ctrl_asm.pin_diameter, ctrl_asm.pin_pitch, pd,
#             d_p2p, ctrl_asm.coolant_int_params['fs'][2],
#             ctrl_asm.params['area'][2] / ctrl_asm.bundle_params['area'],
#             sc_mfr[2], c322]
#     # butt += min_dz
#     print(' '.join(['{:.10e}'.format(v) for v in butt]))
#     # ctrl_asm.update_coolant_byp_params([623.15])
#     # byp_dz = dassh.assembly._calculate_byp_dz(ctrl_asm)
#     assert 0
