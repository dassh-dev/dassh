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
date: 2022-01-05
author: matz
Test the DASSH Assembly object
"""
########################################################################
import numpy as np
import pytest
import dassh


def test_assembly_attributes(textbook_asm):
    """Make sure that the properties work right"""
    assert textbook_asm.has_rodded
    assert textbook_asm._rodded_idx is not None
    assert textbook_asm.rodded is not None


def test_instantiate_unrodded_assembly(c_shield_asm):
    """Make sure you can even make one of these"""
    assert not c_shield_asm.has_rodded
    assert c_shield_asm._rodded_idx is None
    assert c_shield_asm.rodded is None
    assert len(c_shield_asm.region) == 1

    # Volume fraction of coolant
    a_pins = 61 * np.pi * 0.013462**2 / 4
    a_hex = np.sqrt(3) * 0.11154**2 / 2
    ans = (a_hex - a_pins) / a_hex
    assert pytest.approx(
        c_shield_asm.active_region.vf['coolant'], ans)

    # Coolant area
    assert pytest.approx(
        c_shield_asm.active_region.total_area['coolant_int'],
        a_hex - a_pins)

    # Check the velocity
    density = 852.407
    mfr = 0.0088
    ans = mfr / density / (a_hex - a_pins)
    c_shield_asm.active_region._update_coolant_params(698.15)
    assert pytest.approx(
        c_shield_asm.active_region.coolant_params['vel'], ans)


def test_dz_unrodded_asm(c_shield_asm):
    """Can I calculate the axial stability requirement of an assembly
    that has only a SingleNodeHomogeneous region?"""
    # Try calculating minimum dz; shouldn't fail
    dz, sc = dassh.assembly.calculate_min_dz(c_shield_asm,
                                             623.15, 773.15)
    assert sc == 0


def test_temperature_unrodded_asm(c_shield_asm):
    """Can I do a temperature calculation in an unrodded assembly"""
    # Try calculating temperature; shouldn't fail
    dz = 0.005
    tgap = np.ones(6) * 623.15
    htc_gap = np.ones(6) * 1e5
    c_shield_asm.active_region._update_coolant_params(623.15)
    c_shield_asm.calculate(dz, dz, tgap, htc_gap)
    assert c_shield_asm.avg_coolant_temp > 623.15


def test_assembly_clone_shallow(textbook_asm):
    """Test that assembly clone has correct shallow-copied attributes"""
    clone = textbook_asm.clone((1, 1))
    non_matches = []
    # Note: These attributes are immutable and therefore won't be
    # "deepcopied" to a new position:
    # 'n_ring', 'n_pin', 'pin_pitch', 'pin_diameter',
    # 'clad_thickness', 'wire_pitch', 'wire_diameter',
    # 'n_duct', 'n_bypass', 'kappa',
    # 'int_flow_rate',
    for attr in ['pin_lattice', 'subchannel', 'params', 'bundle_params',
                 'duct_params', 'L', 'd', 'duct_ftf', 'ht']:
        id_clone = id(getattr(clone.rodded, attr))
        id_original = id(getattr(textbook_asm.rodded, attr))
        if not id_clone == id_original:  # they should be the same
            non_matches.append(attr)
            print(attr, id_clone, id_original)
    assert len(non_matches) == 0


def test_assembly_clone_deep(textbook_asm):
    """Test that assembly clone has correct deep-copied attributes"""
    clone = textbook_asm.clone((1, 1))
    assert id(clone) != id(textbook_asm)
    non_matches = []
    for attr in ['corr', 'coolant_int_params', 'temp']:
        id_clone = id(getattr(clone.rodded, attr))
        id_original = id(getattr(textbook_asm.rodded, attr))
        if not id_clone != id_original:  # they should be different
            non_matches.append(attr)
            print(attr, id_clone, id_original)
    assert len(non_matches) == 0


def test_assembly_clone_new_fr(textbook_asm):
    """Test behavior of clone method with new flowrate spec"""
    clone = textbook_asm.clone((1, 1), 12.5)
    assert clone.flow_rate != textbook_asm.flow_rate
    assert clone.flow_rate == 12.5
    print(clone.rodded.int_flow_rate)
    print(textbook_asm.rodded.int_flow_rate)
    non_matches = []
    for attr in ['int_flow_rate', 'ht']:
        id_clone = id(getattr(clone.rodded, attr))
        id_original = id(getattr(textbook_asm.rodded, attr))
        if not id_clone != id_original:  # They should be different
            non_matches.append(attr)
            print(attr, id_clone, id_original)
    assert len(non_matches) == 0


def test_assembly_clone_unchanged_og(textbook_asm):
    """Can I change a clone without changing the original?"""
    clone = textbook_asm.clone((1, 1))
    assert clone.rodded.coolant_int_params['Re'] == \
        pytest.approx(textbook_asm.rodded.coolant_int_params['Re'], 1e-9)
    assert clone.rodded.temp['coolant_int'][0] == \
        pytest.approx(textbook_asm.rodded.temp['coolant_int'][0], 1e-9)

    # Change clone attrs, make sure they're not equal
    clone.rodded.coolant_int_params['Re'] = 4e4
    clone.rodded.temp['coolant_int'][0] += 100
    assert clone.rodded.coolant_int_params['Re'] != \
        textbook_asm.rodded.coolant_int_params['Re']
    assert not np.allclose(clone.rodded.temp['coolant_int'],
                           textbook_asm.rodded.temp['coolant_int'],
                           atol=10.0)


def test_material_update_error_msg(c_shield_asm, caplog):
    """Check that material update error provides detailed message"""
    with pytest.raises(SystemExit):
        c_shield_asm.active_region._update_coolant(-50.0)
    msg = "Coolant material update failure; "
    msg += f"Asm: {c_shield_asm.id}; "
    msg += f"Loc: {c_shield_asm.loc}; "
    msg += f"Name: {c_shield_asm.active_region.name}"
    assert msg in caplog.text


def test_identify_axial_region(c_fuel_asm, small_core_asm_power):
    """Make sure I can figure out where I am in an assembly with
    user-specified unrodded regions"""
    # bounds = [[0, 1.281], [2.9095, 3.862]]
    print(c_fuel_asm.region_bnd)
    print(small_core_asm_power.z_finemesh)
    print(small_core_asm_power.rod_zbnds)
    assert len(c_fuel_asm.region) > 1
    zpts = [1.0, 1.2805, 1.281, 1.2811, 2.9094, 2.9095, 2.90951, 3.0]
    reg_id = [0, 0, 0, 1, 1, 1, 2, 2]

    for i in range(len(zpts)):
        # c_fuel_asm._z = zpts[i]
        active_region_id = c_fuel_asm._identify_active_region(zpts[i])
        if not active_region_id == reg_id[i]:
            print('z =', zpts[i],
                  '; ans =', reg_id[i],
                  '; res =', active_region_id)
            assert active_region_id == reg_id[i]

        # Check that power profile agrees with region determination
        p = small_core_asm_power.get_power(zpts[i])
        if reg_id[i] != c_fuel_asm._rodded_idx:
            print('z =', zpts[i], p['refl'])
            assert p['refl'] is not None
            assert p['pins'] is None


def test_axial_region_sweep(c_fuel_asm):
    """Sweep through axial space and ensure that you properly ID
    the proper axial region at each point"""
    gap_temp = np.random.random(54) + 623.15
    gap_htc = np.random.random(54) + 5e4
    for z in np.arange(0.001, 3.86, 0.001):
        try:
            if c_fuel_asm.check_region_update(z):
                c_fuel_asm.update_region(z, gap_temp, gap_htc, True)
        except:
            print(z, c_fuel_asm.active_region_idx)
            raise
        res = c_fuel_asm.active_region_idx
        if z <= 1.281:
            ans = 0
        elif 1.281 < z <= 2.9095:
            ans = 1
        else:
            ans = 2
        if not ans == res:
            print('z =', z, '; ans =', ans, '; res =', res)
            print(c_fuel_asm.z)
        assert ans == res


def test_axial_region_xpts(c_fuel_asm):
    """Test that assembly spits out xpts for correct axial region"""
    zpts = [1.0, 1.2805, 1.281, 1.2811, 2.9094, 2.9095, 2.90951, 3.0]
    sc_per_side = [0, 0, 0, 8, 8, 8, 0, 0]  # per hex side; no corners
    xptslen = [sc_per_side[i] + 2 for i in range(len(sc_per_side))]
    for i in range(len(zpts)):
        c_fuel_asm.check_region_update(zpts[i])
        print(zpts[i], xptslen[i], len(c_fuel_asm.x_pts))
    assert len(c_fuel_asm.x_pts) == xptslen[i]
