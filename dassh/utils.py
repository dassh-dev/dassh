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
date: 2021-10-23
author: matz
Utility methods
"""
########################################################################
import logging
import os
import errno
module_logger = logging.getLogger('dassh.utils')


########################################################################
# UNIT CONVERSION
# Not recommended; users should use SI units. Implemented and tested
# to streamline comparisons with legacy codes that do not use SI units
########################################################################


_cm = ['cm', 'centimeter', 'centimeters']
_mm = ['mm', 'millimeter', 'millimeters']
_m = ['m', 'meter', 'meters']
_in = ['in', 'inch', 'inches']
_ft = ['ft', 'foot', 'feet']
_degC = ['c', 'degc', 'celsius']
_degF = ['f', 'degf', 'fahrenheit']
_degK = ['k', 'degk', 'kelvin']
_lb = ['lb', 'lbs', 'pound', 'pounds']
_kg = ['kg', 'kgs', 'kilogram', 'kilograms']
_sec = ['s', 'sec', 'secs', 'second', 'seconds']
_min = ['m', 'min', 'mins', 'minute', 'minutes']
_hr = ['h', 'hr', 'hrs', 'hour', 'hours']


_DEFAULT_UNITS = {'temperature': _degK,
                  'length': _m,
                  'mass': _kg,
                  'time': _sec}


def format_unit(in_unit):
    """Take whatever is input and turn it into something recognizable"""
    if in_unit.lower() in _cm:
        return 'cm'
    elif in_unit.lower() in _mm:
        return 'mm'
    elif in_unit.lower() in _m:
        return 'm'
    elif in_unit.lower() in _in:
        return 'in'
    elif in_unit.lower() in _ft:
        return 'ft'
    elif in_unit.lower() in _degC:
        return 'c'
    elif in_unit.lower() in _degF:
        return 'f'
    elif in_unit.lower() in _degK:
        return 'k'
    elif in_unit.lower() in _lb:
        return 'lb'
    elif in_unit.lower() in _kg:
        return 'kg'
    elif in_unit.lower() in _sec:
        return 'sec'
    elif in_unit.lower() in _min:
        return 'min'
    elif in_unit.lower() in _hr:
        return 'hr'
    else:
        error_msg = f'Do not recognize input unit: {in_unit}'
        module_logger.log(40, error_msg)
        raise ValueError(error_msg)


def _preprocess_units(in_unit, out_unit):
    """Standardize unit formatting, check that units are not the same"""
    in_unit = format_unit(in_unit)
    out_unit = format_unit(out_unit)
    if in_unit == out_unit:
        error_msg = 'Cannot convert unit to itself'
        module_logger.log(40, error_msg)
        raise ValueError(error_msg)
    return in_unit, out_unit


def get_length_conversion(in_unit, out_unit):
    """Return the function that does the desired unit conversion to
    or from meters"""
    in_unit, out_unit = _preprocess_units(in_unit, out_unit)
    error_msg = 'Unrecognized length unit: '
    if in_unit in _m:
        if out_unit in _cm:
            return _meters_to_centimeters
        elif out_unit in _mm:
            return _meters_to_millimeters
        elif out_unit in _in:
            return _meters_to_inches
        elif out_unit in _ft:
            return _meters_to_feet
        else:
            module_logger.log(40, error_msg + out_unit)
            raise ValueError(error_msg + out_unit)
    elif out_unit in _m:
        if in_unit in _cm:
            return _centimeters_to_meters
        elif in_unit in _mm:
            return _millimeters_to_meters
        elif in_unit in _in:
            return _inches_to_meters
        elif in_unit in _ft:
            return _feet_to_meters
        else:
            module_logger.log(40, error_msg + out_unit)
            raise ValueError(error_msg + out_unit)
    else:
        error_msg = 'Only convert to/from meters'
        module_logger.log(40, error_msg)
        raise ValueError(error_msg)


def _centimeters_to_meters(length):
    """Convert length from centimeters to meters"""
    return length / 100.0


def _meters_to_centimeters(length):
    """Convert length from meters to centimeters"""
    return length * 100.0


def _millimeters_to_meters(length):
    """Convert length from centimeters to meters"""
    return length / 1000.0


def _meters_to_millimeters(length):
    """Convert length from meters to centimeters"""
    return length * 1000.0


def _inches_to_meters(length):
    """Convert length from inches to meters"""
    return length * 2.54 / 100.0


def _meters_to_inches(length):
    """Convert length from meters to inches"""
    return length * 100 / 2.54


def _feet_to_meters(length):
    """Convert length from feet to meters"""
    return length * 2.54 * 12.0 / 100.0


def _meters_to_feet(length):
    """Convert length from meters to feet"""
    return length * 100 / 2.54 / 12.0


def get_temperature_conversion(in_unit, out_unit):
    """Return the function that does the desired unit conversion to
    or from Kelvin"""
    in_unit, out_unit = _preprocess_units(in_unit, out_unit)
    error_msg = 'Unrecognized temperature unit: '
    if in_unit in _degK:
        if out_unit in _degC:
            return _kelvin_to_celsius
        elif out_unit in _degF:
            return _kelvin_to_fahrenheit
        else:
            raise ValueError(error_msg + out_unit)
    elif out_unit in _degK:
        if in_unit in _degC:
            return _celsius_to_kelvin
        elif in_unit in _degF:
            return _fahrenheit_to_kelvin
        else:
            raise ValueError(error_msg + out_unit)
    else:
        error_msg = 'Only convert to/from Kelvin'
        module_logger.log(40, error_msg)
        raise ValueError(error_msg)


def _celsius_to_kelvin(temp):
    """Convert temperature in Celsius to Kelvin"""
    return temp + 273.15


def _kelvin_to_celsius(temp):
    """Convert temperature in Kelvin to Celsius"""
    return temp - 273.15


def _fahrenheit_to_kelvin(temp):
    """Convert temperature in Fahrenheit to Kelvin"""
    return (temp - 32) * 5 / 9 + 273.15


def _kelvin_to_fahrenheit(temp):
    """Convert temperature in Kelvin to Fahrenheit"""
    return (temp - 273.15) * 9 / 5 + 32.0


def get_mass_conversion(in_unit, out_unit):
    """Return the function that does the desired unit conversion to
    or from kilograms"""
    in_unit, out_unit = _preprocess_units(in_unit, out_unit)
    error_msg = 'Unrecognized mass unit: '
    if in_unit in _kg:
        if out_unit in _lb:
            return _kilograms_to_pounds
        else:
            raise ValueError(error_msg + out_unit)
    elif out_unit in _kg:
        if in_unit in _lb:
            return _pounds_to_kilograms
        else:
            raise ValueError(error_msg + out_unit)
    else:
        error_msg = 'Only convert to/from kilograms'
        module_logger.log(40, error_msg)
        raise ValueError(error_msg)


def _pounds_to_kilograms(mass):
    """Convert mass lbs to kg"""
    return mass * 0.453592


def _kilograms_to_pounds(mass):
    """Convert mass kg to lbs"""
    return mass / 0.453592


def get_time_conversion(in_unit, out_unit):
    """Return the function that does the desired unit conversion to
    or from seconds"""
    in_unit, out_unit = _preprocess_units(in_unit, out_unit)
    error_msg = 'Unrecognized time unit: '
    if in_unit in _sec:
        if out_unit in _min:
            return _seconds_to_minutes
        if out_unit in _hr:
            return _seconds_to_hours
        else:
            raise ValueError(error_msg + out_unit)
    elif out_unit in _sec:
        if in_unit in _min:
            return _minutes_to_seconds
        if in_unit in _hr:
            return _hours_to_seconds
        else:
            raise ValueError(error_msg + in_unit)
    else:
        error_msg = 'Only convert to/from seconds'
        module_logger.log(40, error_msg)
        raise ValueError(error_msg)


def _minutes_to_seconds(time):
    """Convert time: minutes to seconds"""
    return time * 60.0


def _seconds_to_minutes(time):
    """Convert time: minutes to seconds"""
    return time / 60.0


def _hours_to_seconds(time):
    """Convert time: hours to seconds"""
    return time * 3600.0


def _seconds_to_hours(time):
    """Convert time: seconds to hours"""
    return time / 3600.0


def parse_mfr_units(in_unit):
    """Figure out the mass and time units of a mass flow rate unit"""
    in_unit = in_unit.lower()
    if '/' in in_unit:
        mass_time = in_unit.split('/')
    elif 'per' in in_unit:
        mass_time = in_unit.split('per')
    else:
        msg = f'Do not understand mass flow rate unit input: {in_unit}'
        module_logger.log(40, msg)
        raise ValueError(msg)
    return [mass_time[0].split(' ')[0], mass_time[1].split(' ')[-1]]


########################################################################


def Q_equals_mCdT(power, t_in, coolant_obj, t_out=None, mfr=None):
    """Use Q=mCdT to estimate the flow rate required to remove
    the assembly thermal power

    Parameters
    ----------
    power : float
        Power (W)
    t_in : float
        Inlet temperature (K)
    coolant_obj : DASSH Material object
        Contains coolant properties
    t_out (optional) : float
        Outlet temperature (K); if specified, calculate mass flow rate
    mfr (optional) : float
        Mass flow rate (kg/s); if specified, calculate t_out

    Returns
    -------
    float
        Depending on user input, either:
        (a) Estimated coolant mass flow rate (kg/s)
        (b) Estimated coolant outlet temperature (K)

    """
    if t_out:
        coolant_obj.update(t_in + 0.5 * (t_out - t_in))
        mfr = power / coolant_obj.heat_capacity / (t_out - t_in)
        coolant_obj.update(t_in)  # reset the temperature
        return mfr

    elif mfr:
        for i in range(3):  # Iterations to converge to a good guess
            t_out = t_in + power / coolant_obj.heat_capacity / mfr
            coolant_obj.update(t_in + 0.5 * (t_out - t_in))
        coolant_obj.update(t_in)  # reset the temperature
        return t_out

    else:
        msg = 'Need to specify mass flow rate or outlet temp'
        module_logger.log(40, msg)
        raise ValueError(msg)


########################################################################


def _get_profile_data(path='dassh_profile.out', n=50):
    """Shortcut to print DASSH profile data"""
    import pstats
    p = pstats.Stats(path)
    p.sort_stats('cumulative').print_stats(n)


########################################################################


def _symlink(target, link_name):
    """Try to symlink original file "target" to link "link_name";
    if it fails b/c of existing file, it removes it and links
    again

    From: https://stackoverflow.com/questions/8299386/
          modifying-a-symlink-in-python/55742015#55742015

    """
    try:
        os.symlink(target, link_name)
    except (OSError, FileExistsError) as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


########################################################################
