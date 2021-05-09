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
date: 2021-01-21
author: matz
Test the base Region class and its methods
"""
########################################################################
import numpy as np
import conftest


def test_activate_to_rr(c_fuel_params, c_lrefl_simple):
    """Test that duct temperatures can be interpolated when a new
    region is activated and when the number of ducts stays the same"""
    # Need to activate RR
    flowrate = 1.0
    rr = conftest.make_rodded_region_fixture('conceptual_fuel',
                                             c_fuel_params[0],
                                             c_fuel_params[1],
                                             flowrate)

    # Update the flow rate on the simple region
    ur = c_lrefl_simple.clone(new_flowrate=flowrate)
    # Activate simple region manually (set temps equal to something)
    for key in ur.temp:
        ur.temp[key] = np.random.random((ur.temp[key].shape))
        ur.temp[key] = ur.temp[key] * 10 + 623.15  # 623.15 to 633.15

    # Now activate RR based on UR and see what happened
    rr.activate(ur)

    # Average coolant interior temperature
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_int_temp - ur.avg_coolant_int_temp
    assert np.abs(diff) < 1e-9, msg

    # Overall avg coolant temp (w/ no bypass, should be same as above)
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_temp - ur.avg_coolant_temp
    assert np.abs(diff) < 1e-9, msg

    # Average duct MW temp
    msg = 'Average duct midwall temperature'
    diff = rr.avg_duct_mw_temp - ur.avg_duct_mw_temp
    assert np.abs(diff) < 1e-9, msg

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
    """Test that duct temperatures can be interpolated when a new
    region is activated and when the number of ducts stays the same;
    this time starting from RR and activating UR"""
    # Need to activate RR
    flowrate = 1.0
    rr = conftest.make_rodded_region_fixture('conceptual_fuel',
                                             c_fuel_params[0],
                                             c_fuel_params[1],
                                             flowrate)
    rr = conftest.activate_rodded_region(rr, 650.0)
    # Muss up the temperatures so it's like it did something
    for key in rr.temp:
        # denom = (np.random.random() + 1) / 2
        denom = 1.0
        rr.temp[key] += np.random.random(rr.temp[key].shape) / denom

    # Update the flow rate on the simple region and activate
    ur = c_lrefl_simple.clone(new_flowrate=flowrate)
    ur.activate(rr)

    # Average coolant interior temperature
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_int_temp - ur.avg_coolant_int_temp
    assert np.abs(diff) < 1e-9, msg

    # Overall avg coolant temp (w/ no bypass, should be same as above)
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_temp - ur.avg_coolant_temp
    assert np.abs(diff) < 1e-9, msg

    # Note: average duct temperature won't be preserved because by
    # adding a random set to the RR duct temperatures I'm ruining the
    # ability of the "approximate" method to capture it. But the
    # corner temperatures should be maintained, as demonstrated below.

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


def test_activate_to_rr_dd(c_ctrl_params, c_lrefl_simple):
    """Test that duct temperatures can be interpolated when a new
    region is activated and when the number of ducts stays the same"""
    # Need to activate RR
    flowrate = 1.0
    rr = conftest.make_rodded_region_fixture('conceptual_ctrl',
                                             c_ctrl_params[0],
                                             c_ctrl_params[1],
                                             flowrate)

    # Update the flow rate on the simple region
    ur = c_lrefl_simple.clone(new_flowrate=flowrate)
    # Activate simple region manually (set temps equal to something)
    for key in ur.temp:
        ur.temp[key] = np.random.random((ur.temp[key].shape))
        ur.temp[key] = ur.temp[key] * 10 + 623.15  # 623.15 to 633.15

    # Now activate RR based on UR and see what happened
    rr.activate(ur)

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
    assert np.abs(diff) < 1e-9, msg

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
    assert np.all(np.isclose(diff, 0.0)), msg

    msg = 'Corner duct surface temperatures'
    rr_corner_temps = rr.temp['duct_surf'][:, :, idx]
    diff = rr_corner_temps[-1] - ur.temp['duct_surf'][-1]
    assert np.all(np.isclose(diff, 0.0)), msg


def test_activate_from_rr_dd(c_ctrl_params, c_lrefl_simple):
    """Test that duct temperatures can be interpolated when a new
    region is activated and when the number of ducts stays the same;
    this time starting from RR and activating UR"""
    # Need to activate RR
    flowrate = 1.0
    rr = conftest.make_rodded_region_fixture('conceptual_fuel',
                                             c_ctrl_params[0],
                                             c_ctrl_params[1],
                                             flowrate)
    rr = conftest.activate_rodded_region(rr, 650.0)
    # Muss up the temperatures so it's like it did something
    for key in rr.temp:
        # denom = (np.random.random() + 1) / 2
        denom = 1.0
        rr.temp[key] += np.random.random(rr.temp[key].shape) / denom

    # Update the flow rate on the simple region and activate
    ur = c_lrefl_simple.clone(new_flowrate=flowrate)
    ur.activate(rr)

    # Average coolant interior temperature not the same when activating
    # from double-duct assembly because it's assumed that all coolant
    # will mix

    # However, overall avg coolant temp should be same
    msg = 'Average interior coolant temperature error'
    diff = rr.avg_coolant_temp - ur.avg_coolant_temp
    assert np.abs(diff) < 1e-9, msg

    # Note: average duct temperature won't be preserved because by
    # adding a random set to the RR duct temperatures I'm ruining the
    # ability of the "approximate" method to capture it. But the
    # corner temperatures should be maintained, as demonstrated below.

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
    diff = rr_corner_temps[-1] - ur.temp['duct_surf'][-1]
    assert np.all(np.isclose(diff, 0.0)), msg
