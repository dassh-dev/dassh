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
date: 2021-08-20
author: matz
Test utility methods
"""
########################################################################
import pytest
import dassh
from dassh import utils


def test_format_unit(caplog):
    """Test some crazy units and see if they get recognized"""
    # Should pass
    units = ['MeTeRs', 'fahrenheit', 'degK', 'hours', 'foot',
             'mm', 's', 'CM', 'c', 'iNCHes', 'pounds', 'kgs']
    ans = ['m', 'f', 'k', 'hr', 'ft',
           'mm', 'sec', 'cm', 'c', 'in', 'lb', 'kg']
    for i in range(len(units)):
        assert utils.format_unit(units[i]) == ans[i]
    # Should fail
    units = ['lightyear', 'Rankine', 'slugs', 'moment']
    for i in range(len(units)):
        with pytest.raises(ValueError):
            utils.format_unit(units[i])
        assert f'Do not recognize input unit: {units[i]}' in caplog.text
        caplog.clear()


def test_length_conversion():
    """Test the length conversions with obvious examples. DASSH native
    units are meters, so need to and from"""
    # Test conversions from meters
    len_in_m = 3.1415926
    assert utils._meters_to_centimeters(len_in_m) == pytest.approx(314.15926)
    assert utils._meters_to_millimeters(len_in_m) == pytest.approx(3141.5926)
    assert utils._meters_to_inches(len_in_m) == pytest.approx(123.6847480315)
    assert utils._meters_to_feet(len_in_m) == pytest.approx(10.307062335958)
    # Test conversions to meters
    assert utils._centimeters_to_meters(314.15926) == pytest.approx(len_in_m)
    assert utils._millimeters_to_meters(3141.5926) == pytest.approx(len_in_m)
    assert utils._inches_to_meters(123.6847480315) == pytest.approx(len_in_m)
    assert utils._feet_to_meters(10.307062335958) == pytest.approx(len_in_m)


def test_get_length_conversion():
    """Test that proper conversion function is returned by getter"""
    len_in_m = 3.1415926
    units = ['cm', 'mm', 'in', 'ft']
    answers = [314.15926, 3141.5926, 123.6847480315, 10.307062335958]
    # Test conversion from meters
    for i in range(len(units)):
        conv = utils.get_length_conversion('m', units[i])
        assert conv(len_in_m) == pytest.approx(answers[i])
    # Test conversions to meters
    for i in range(len(units)):
        conv = utils.get_length_conversion(units[i], 'm')
        assert conv(answers[i]) == pytest.approx(len_in_m)


def test_get_length_conv_errors(caplog):
    """Test errors when requesting length conversion"""
    # ValueError should be raised if you try to convert a unit to itself
    with pytest.raises(ValueError):
        utils.get_length_conversion('m', 'MeTeRs')
    assert 'Cannot convert unit to itself' in caplog.text

    # Converting between units other than meters
    with pytest.raises(ValueError):
        utils.get_length_conversion('ft', 'cm')
    assert 'Only convert to/from meters' in caplog.text


def test_temperature_conversion():
    """Test the temperature conversions with obvious examples. DASSH
    native units are Kelvin, so need to and from"""
    # Test conversions from Kelvin
    temp_in_K = 623.15
    assert utils._kelvin_to_celsius(temp_in_K) == pytest.approx(350.0)
    assert utils._kelvin_to_fahrenheit(temp_in_K) == pytest.approx(662.0)
    # Test conversions to Kelvin
    assert utils._celsius_to_kelvin(350.0) == pytest.approx(temp_in_K)
    assert utils._fahrenheit_to_kelvin(662.0) == pytest.approx(temp_in_K)


def test_get_temperature_conversion():
    """Test that proper conversion function is returned by getter"""
    temp_in_K = 623.15
    units = ['Celsius', 'F', 'FahrEnHeiT', 'degC']
    answers = [350.0, 662.0, 662.0, 350.0]
    # Test conversion from meters
    for i in range(len(units)):
        conv = utils.get_temperature_conversion('K', units[i])
        assert conv(temp_in_K) == pytest.approx(answers[i])
    # Test conversions to meters
    for i in range(len(units)):
        conv = utils.get_temperature_conversion(units[i], 'K')
        assert conv(answers[i]) == pytest.approx(temp_in_K)


def test_get_temperature_conv_errors(caplog):
    """Test errors when requesting temperature conversion"""
    # ValueError should be raised if you try to convert a unit to itself
    with pytest.raises(ValueError):
        utils.get_temperature_conversion('degK', 'K')
    assert 'Cannot convert unit to itself' in caplog.text

    # Converting between units other than kelvin
    with pytest.raises(ValueError):
        utils.get_temperature_conversion('C', 'F')
    assert 'Only convert to/from Kelvin' in caplog.text


def test_get_mass_conversion():
    """Test that proper conversion is returned and executed by getter"""
    mass_in_kg = 1.0
    mass_in_lb = 2.20462442
    units = ['lb', 'pounds', 'lbs', 'pound']
    for i in range(len(units)):
        # From kilograms
        conv = utils.get_mass_conversion('kg', units[i])
        assert conv(mass_in_kg) == pytest.approx(mass_in_lb)
        # To kilograms
        conv = utils.get_mass_conversion(units[i], 'kg')
        assert conv(mass_in_lb) == pytest.approx(mass_in_kg)


def test_get_mass_conv_errors(caplog):
    """Test errors when requesting mass conversion"""
    # Converting between units other than kilograms : no need for this
    # test because only one mass unit conversion available.
    # ValueError should be raised if you try to convert a unit to itself
    with pytest.raises(ValueError):
        utils.get_temperature_conversion('pounds', 'lbs')
    assert 'Cannot convert unit to itself' in caplog.text


def test_get_time_conversion():
    """Test that proper conversion is returned and executed by getter"""
    time_in_sec = 1242.420
    units = ['min', 'hrs', 'h', 'minutes']
    ans = [20.707, 0.3451166666666667, 0.3451166666666667, 20.707]
    for i in range(len(units)):
        # From seconds
        conv = utils.get_time_conversion('s', units[i])
        assert conv(time_in_sec) == pytest.approx(ans[i])
        # To seconds
        conv = utils.get_time_conversion(units[i], 's')
        assert conv(ans[i]) == pytest.approx(time_in_sec)


def test_get_time_conv_errors(caplog):
    """Test errors when requesting temperature conversion"""
    # ValueError should be raised if you try to convert a unit to itself
    with pytest.raises(ValueError):
        utils.get_time_conversion('seConDS', 'sec')
    assert 'Cannot convert unit to itself' in caplog.text

    # Converting between units other than kelvin
    with pytest.raises(ValueError):
        utils.get_time_conversion('min', 'hr')
    assert 'Only convert to/from seconds' in caplog.text


def test_parse_mfr():
    """Because mass flow rate is a complicated unit, it needs to be
    parsed - test that I can get units of mass and time from the
    compound string"""
    units = ['kg/s', 'lb/hr', 'lb/SeCoNd', 'kG per hr', 'lbpermin']
    ans = [['kg', 's'],
           ['lb', 'hr'],
           ['lb', 'second'],
           ['kg', 'hr'],
           ['lb', 'min']]
    for i in range(len(units)):
        result = utils.parse_mfr_units(units[i])
        assert result[0] == ans[i][0]
        assert result[1] == ans[i][1]


def test_q_equals_mcdT():
    """Confirm that Q=mCdT gives average estimate of either
    coolant outlet temperature or required mass flow rate"""
    p = 1000.0
    t_in = 298.15  # 25 degrees C
    coolant = dassh.Material('water')
    # Test calculation of MFR given outlet temperature
    delta_t = 50.0
    t_out = t_in + delta_t  # 75 degrees C
    coolant.update(t_in + 0.5 * delta_t)
    cp = coolant.heat_capacity
    coolant.update(42.1242)
    result = utils.Q_equals_mCdT(p, t_in, coolant, t_out=t_out)
    ans = p / cp / delta_t
    assert result == pytest.approx(ans)
    # Test calculation of outlet temperature given MFR
    mfr = 4.782972617481765  # kg/s, result from the test above
    result = utils.Q_equals_mCdT(p, t_in, coolant, mfr=mfr)
    assert result == pytest.approx(t_out)
