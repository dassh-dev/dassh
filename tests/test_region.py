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
Test the base Region class and its methods
"""
########################################################################
import numpy as np
import pytest
from tests import conftest


def test_activate_to_rr(c_fuel_params, c_lrefl_simple):
    """Test activation from simple model to pin bundle model (single
    adiabatic duct wall)

    - All subchannels in pin bundle equal simple model temperature.
    - Average coolant temperature should be maintained.
    - Duct wall temperatures are recalculated to maintain energy cons.
    - Since adiabatic, average duct temperature should be close.

    """
    # Need to activate RR
    flowrate = 1.0
    t_gap = np.ones(2)    # Making adiabatic so it doesn't matter
    htc_gap = np.ones(2)  # Making adiabatic so it doesn't matter
    rr = conftest.make_rodded_region_fixture('conceptual_fuel',
                                             c_fuel_params[0],
                                             c_fuel_params[1],
                                             flowrate)

    # SETUP THE PREVIOUS REGION: Update the flow rate on the simple region
    ur = c_lrefl_simple.clone(new_flowrate=flowrate)
    # Activate simple region manually (set temps equal to something)
    k = 'coolant_int'
    ur.temp[k] = np.random.random((ur.temp[k].shape)) * 10 + 623.15
    ur._update_coolant_params(ur.temp['coolant_int'][0])
    ur._calc_duct_temp(t_gap, htc_gap, True)

    # Now activate RR based on UR and see what happened
    rr.activate(ur, t_gap, htc_gap, True)

    # Subchannel temperatures
    msg = 'Subchannel coolant temperature error'
    diff = rr.temp['coolant_int'] - ur.temp['coolant_int'][0]
    assert np.all(np.abs(diff) < 1e-9), msg

    # Average coolant interior temperature
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_int_temp - ur.avg_coolant_int_temp
    assert np.abs(diff) < 1e-9, msg

    # Overall avg coolant temp (since no bypass, same as above)
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_temp - ur.avg_coolant_temp
    assert np.abs(diff) < 1e-9, msg

    # Average duct MW temp: for the outer duct, this is the only
    # thing that might be different.
    msg = 'Average duct midwall temperature'
    diff = rr.avg_duct_mw_temp - ur.avg_duct_mw_temp
    # Fails at tolerance 1e-9; 1e-8 K is still pretty close tho
    assert np.abs(diff) < 1e-8, msg

    # Corner duct MW and surface temperatures
    idx = np.arange(0, 6, 1, dtype=int) + 1
    idx *= int(rr.subchannel.n_sc['duct']['total'] / 6)
    idx -= 1

    msg = 'Corner duct midwall temperatures'
    rr_corner_temps = rr.temp['duct_mw'][-1, idx]
    diff = rr_corner_temps - ur.temp['duct_mw'][-1]
    assert np.all(np.isclose(diff, 0.0)), msg

    msg = 'Corner duct surface temperatures'
    rr_corner_temps = rr.temp['duct_surf'][:, :, idx]
    diff = rr_corner_temps - ur.temp['duct_surf']
    assert np.all(np.isclose(diff, 0.0)), msg


def test_activate_from_rr(c_fuel_params, c_lrefl_simple):
    """Test activation from pin bundle model to simple bundle model
    (single adiabatic duct wall)

    - Simple model coolant temperature equals average pin bundle model
      coolant temperature
    - Duct wall temperatures are recalculated to maintain energy cons.
      Therefore, duct wall temperatures may not be maintained (see
      note below).

    Note
    ----
    Average duct temperature won't be preserved because the
    new duct temperatures are calculated based on the new
    coolant temperature (which for the simple model region
    is only 1 value). The old average duct temperature was
    based on the temperatures of only the edge and corner
    subchannels immediately adjacent to it.

    """
    # Need to activate RR
    flowrate = 1.0
    rr = conftest.make_rodded_region_fixture('conceptual_fuel',
                                             c_fuel_params[0],
                                             c_fuel_params[1],
                                             flowrate)
    rr = conftest.activate_rodded_region(rr, 650.0)

    # Muss up the temperatures so it's like it did something
    t_gap = np.ones(54)    # Making adiabatic so it doesn't matter
    htc_gap = np.ones(54)  # Making adiabatic so it doesn't matter
    p_duct = np.zeros(54)  # Zero power in duct
    k = 'coolant_int'
    rr.temp[k] = np.random.random((rr.temp[k].shape)) * 10 + 650.15
    rr._update_coolant_int_params(rr.avg_coolant_int_temp)
    rr._calc_duct_temp(p_duct, t_gap, htc_gap, True)

    # Update the flow rate on the simple region and activate
    ur = c_lrefl_simple.clone(new_flowrate=flowrate)
    ur.activate(rr, t_gap, htc_gap, True)

    # Average coolant interior temperature
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_int_temp - ur.avg_coolant_int_temp
    assert np.abs(diff) < 1e-9, msg

    # Overall avg coolant temp (w/ no bypass, should be same as above)
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_temp - ur.avg_coolant_temp
    assert np.abs(diff) < 1e-9, msg


def test_activate_to_rr_dd(c_ctrl_params, c_lrefl_simple):
    """Test activation from simple model to pin bundle model (double
    duct with adiabatic boundary on outer duct wall surface)

    - All interior subchannels in pin bundle equal simple model temp.
    - All interior duct wall temperatures in pin bundle model equal
      simple model coolant temp.
    - All bypass gap temperatures in pin bundle model equal simple
      model coolant temp.
    - Outer duct wall is recalculated - may not match exactly.

    """
    # Need to activate RR
    flowrate = 1.0
    rr = conftest.make_rodded_region_fixture('conceptual_ctrl',
                                             c_ctrl_params[0],
                                             c_ctrl_params[1],
                                             flowrate)
    t_gap = np.ones(2)    # Making adiabatic so it doesn't matter
    htc_gap = np.ones(2)  # Making adiabatic so it doesn't matter

    # SETUP THE PREVIOUS REGION: Update the flow rate on the simple region
    ur = c_lrefl_simple.clone(new_flowrate=flowrate)
    # Activate simple region manually (set temps equal to something)
    k = 'coolant_int'
    ur.temp[k] = np.random.random((ur.temp[k].shape)) * 10 + 623.15
    ur._update_coolant_params(ur.temp['coolant_int'][0])
    ur._calc_duct_temp(t_gap, htc_gap, True)

    # Now activate RR based on UR and see what happened
    rr.activate(ur, t_gap, htc_gap, True)

    # Subchannel temperatures
    msg = 'Interior subchannel temperatures error'
    diff = rr.temp['coolant_int'] - ur.temp['coolant_int'][0]
    assert np.all(np.abs(diff) < 1e-9), msg

    # Interior duct wall temperatures
    msg = 'Interior duct wall temperatures error'
    diff = rr.temp['duct_mw'] - ur.temp['coolant_int'][0]
    # Note: this fails at 1e-9.
    assert np.all(np.abs(diff) < 2e-8), msg

    # Bypass gap temperatures
    msg = 'Subchannels between ducts temperature error'
    diff = rr.temp['coolant_byp'][0] - ur.temp['coolant_int'][0]
    assert np.all(np.abs(diff) < 1e-9), msg

    # Average coolant interior temperature; should be the same when
    # activated double-duct assembly because all coolant is the same
    # temp.
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_int_temp - ur.avg_coolant_int_temp
    assert np.abs(diff) < 1e-9, msg

    # Overall avg coolant temp
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_temp - ur.avg_coolant_temp
    assert np.abs(diff) < 1e-9, msg

    # Average duct MW temp
    msg = 'Average outer duct midwall temperature'
    diff = rr.avg_duct_mw_temp[-1] - ur.avg_duct_mw_temp
    # This fails at 1e-9, but 2e-8 degrees K is pretty close
    assert np.abs(diff) < 2e-8, msg

    msg = 'Average inner duct midwall temperature'
    diff = rr.avg_duct_mw_temp[0] - ur.avg_coolant_temp
    assert np.abs(diff) < 1e-9, msg

    # Corner duct MW and surface temperatures
    idx = np.arange(0, 6, 1, dtype=int) + 1
    idx *= int(rr.subchannel.n_sc['duct']['total'] / 6)
    idx -= 1

    msg = 'Corner duct midwall temperatures'
    rr_corner_temps = rr.temp['duct_mw'][-1, idx]
    diff = rr_corner_temps - ur.temp['duct_mw'][-1]
    # This fails at 1e-9, but 2e-8 degrees K is pretty close
    assert np.all(np.abs(diff) < 2e-8), msg

    msg = 'Corner duct surface temperatures'
    rr_corner_temps = rr.temp['duct_surf'][:, :, idx]
    diff = rr_corner_temps[-1] - ur.temp['duct_surf'][-1]
    # This fails at 1e-9, but 2e-8 degrees K is pretty close
    assert np.all(np.abs(diff) < 2e-8), msg


def test_activate_from_rr_dd(c_ctrl_params, c_lrefl_simple):
    """Test activation from double-ducted pin bundle model to
    simple bundle model

    - Pin bundle overall average coolant temperature (interior and
      double-duct bypass) --> simple model coolant temperature
    - Simple model outer duct wall temperature recalculated to
      maintain energy conservation; temps may not be maintained.
      coolant temperature

    Note
    ----
    Average duct temperature won't be preserved because the
    new duct temperatures are calculated based on the new
    coolant temperature (which for the simple model region
    is only 1 value). The old average duct temperature was
    based on the temperatures of only the edge and corner
    subchannels immediately adjacent to it.

    """
    # Need to activate RR
    flowrate = 1.0
    t_gap = np.ones(54)    # Making adiabatic so it doesn't matter
    htc_gap = np.ones(54)  # Making adiabatic so it doesn't matter
    rr = conftest.make_rodded_region_fixture('conceptual_ctrl',
                                             c_ctrl_params[0],
                                             c_ctrl_params[1],
                                             flowrate)
    rr = conftest.activate_rodded_region(rr, 650.0, base=False)
    # Muss up the temperatures so it's like it did something
    p_duct = np.zeros(54)  # Zero power in duct
    k = 'coolant_int'
    rr.temp[k] = np.random.random((rr.temp[k].shape)) * 10 + 650.15
    rr._update_coolant_int_params(rr.avg_coolant_int_temp)
    rr._calc_duct_temp(p_duct, t_gap, htc_gap, True)

    # Update the flow rate on the simple region and activate
    ur = c_lrefl_simple.clone(new_flowrate=flowrate)
    ur.activate(rr, t_gap, htc_gap, True)

    # Average coolant interior temperature not the same when activating
    # from double-duct assembly because it's assumed that all coolant
    # will mix. However, overall avg coolant temp should be same
    msg = 'Average coolant temperature error'
    diff = rr.avg_coolant_temp - ur.avg_coolant_temp
    assert np.abs(diff) < 1e-9, msg


def test_material_update_errors(c_fuel_params, caplog):
    """Test that material update failures return error messages"""
    flowrate = 1.0
    rr = conftest.make_rodded_region_fixture('conceptual_fuel',
                                             c_fuel_params[0],
                                             c_fuel_params[1],
                                             flowrate)
    rr = conftest.activate_rodded_region(rr, 650.0)
    with pytest.raises(SystemExit):
        rr._update_coolant(-50.0)

    msg = "Coolant material update failure; Name: conceptual_fuel"
    assert msg in caplog.text
