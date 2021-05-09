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
date: 20201-05-09
author: matz
Utility methods
"""
########################################################################
# import logging
# import numpy as np
# module_logger = logging.getLogger('dassh.setup')


########################################################################
# UNIT CONVERSION
########################################################################


_cm = ['cm', 'centimeter', 'centimeters']
_mm = ['mm', 'millimeter', 'millimeters']
_m = ['m', 'meter', 'meters']
_in = ['in', 'inch', 'inches']
_ft = ['ft', 'foot', 'feet']
_degC = ['c', 'degc', 'celsius']
_degF = ['f', 'degf', 'fahrenheit']
_degK = ['k', 'degK', 'kelvin']
_lb = ['lb', 'lbs', 'pound', 'pounds']
_kg = ['kg', 'kgs', 'kilogram', 'kilograms']
_sec = ['s', 'sec', 'secs', 'second', 'seconds']
_min = ['m', 'min', 'mins', 'minute', 'minutes']
_hr = ['h', 'hr', 'hrs', 'hour', 'hours']


_DEFAULT_UNITS = {'temperature': _degK,
                  'length': _m,
                  'mass': _kg,
                  'time': _sec}


def get_length_conversion(in_unit, out_unit):
    """Return the function that does the desired unit conversion to
    or from meters"""
    _cm = ['cm', 'centimeter', 'centimeters']
    _mm = ['mm', 'millimeter', 'millimeters']
    _m = ['m', 'meter', 'meters']
    _in = ['in', 'inch', 'inches']
    _ft = ['ft', 'foot', 'feet']

    assert in_unit != out_unit
    error_msg = 'Unrecognized length unit: '

    if in_unit.lower() in _m:
        if out_unit.lower() in _cm:
            return _meters_to_centimeters
        elif out_unit.lower() in _mm:
            return _meters_to_millimeters
        elif out_unit.lower() in _in:
            return _meters_to_inches
        elif out_unit.lower() in _ft:
            return _meters_to_feet
        else:
            raise ValueError(error_msg + out_unit)
    elif out_unit.lower() in _m:
        if in_unit.lower() in _cm:
            return _centimeters_to_meters
        elif in_unit.lower() in _mm:
            return _millimeters_to_meters
        elif in_unit.lower() in _in:
            return _inches_to_meters
        elif in_unit.lower() in _ft:
            return _feet_to_meters
        else:
            raise ValueError(error_msg + out_unit)
    else:
        raise ValueError('Only convert to/from meters')


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

    assert in_unit != out_unit
    error_msg = 'Unrecognized temperature unit: '

    if in_unit.lower() in _degK:
        if out_unit.lower() in _degC:
            return _kelvin_to_celsius
        elif out_unit.lower() in _degF:
            return _kelvin_to_fahrenheit
        else:
            raise ValueError(error_msg + out_unit)
    elif out_unit.lower() in _degK:
        if in_unit.lower() in _degC:
            return _celsius_to_kelvin
        elif in_unit.lower() in _degF:
            return _fahrenheit_to_kelvin
        else:
            raise ValueError(error_msg + out_unit)
    else:
        raise ValueError('Only convert to/from Kelvin')


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

    assert in_unit != out_unit
    error_msg = 'Unrecognized mass unit: '

    if in_unit.lower() in _kg:
        if out_unit.lower() in _lb:
            return _kilograms_to_pounds
        else:
            raise ValueError(error_msg + out_unit)
    elif out_unit.lower() in _kg:
        if in_unit.lower() in _lb:
            return _pounds_to_kilograms
        else:
            raise ValueError(error_msg + out_unit)
    else:
        raise ValueError('Only convert to/from kilograms')


def _pounds_to_kilograms(mass):
    """Convert mass lbs to kg"""
    return mass * 0.453592


def _kilograms_to_pounds(mass):
    """Convert mass kg to lbs"""
    return mass / 0.453592


def get_time_conversion(in_unit, out_unit):
    """Return the function that does the desired unit conversion to
    or from seconds"""

    assert in_unit != out_unit
    error_msg = 'Unrecognized time unit: '

    if in_unit.lower() in _sec:
        if out_unit.lower() in _min:
            return _seconds_to_minutes
        if out_unit.lower() in _hr:
            return _seconds_to_hours
        else:
            raise ValueError(error_msg + out_unit)
    elif out_unit.lower() in _sec:
        if in_unit.lower() in _min:
            return _minutes_to_seconds
        if in_unit.lower() in _hr:
            return _hours_to_seconds
        else:
            raise ValueError(error_msg + in_unit)
    else:
        raise ValueError('Only convert to/from seconds')


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
        raise ValueError('Do not understand mass flow rate unit '
                         f'specification: {in_unit}')
    return [mass_time[0].split(' ')[0], mass_time[1].split(' ')[-1]]


# def _kg_min_to_kg_sec(mfr):
#     """Convert mass flow rate from kg/min to kg/s"""
#     return _minutes_to_seconds(mfr)
#
#
# def _kg_hr_to_kg_sec(mfr):
#     """Convert mass flow rate from kg/h to kg/s"""
#     return _hours_to_seconds(mfr)
#
#
# def _lb_sec_to_kg_sec(mfr):
#     """Convert mass flow rate from lb/s to kg/s"""
#     return _pounds_to_kilograms(mfr)
#
#
# def _lb_min_to_kg_sec(mfr):
#     """Convert mass flow rate from lb/min to kg/s"""
#     mfr = _pounds_to_kilograms(mfr)
#     return _minutes_to_seconds(mfr)
#
#
# def _lb_hr_to_kg_sec(mfr):
#     """Convert mass flow rate from lb/h to kg/s"""
#     mfr = _pounds_to_kilograms(mfr)
#     return _hours_to_seconds(mfr)


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
        raise ValueError('Need to specify mass flow rate or outlet temp')


########################################################################


def dif3d_loc_to_dassh_loc(dif3d_loc):
    """If user specifies DIF3D indexing (counter-clockwise), convert
    to DASSH indexing (clockwise)

    Notes
    -----
    Also should work the other way (dassh_loc -> dif3d_loc)

    """
    # First assembly in ring is the same for DASSH/DIF3D indexing
    if dif3d_loc[1] == 0:
        return dif3d_loc
    # Otherwise, the ring will be the same but the position is reversed
    else:
        asm_in_ring = 6 * dif3d_loc[0]
        return (dif3d_loc[0], asm_in_ring - dif3d_loc[1])


def dassh_loc_to_dif3d_id(dassh_loc):
    """If DASSH indexing, assembly ID needs to be modified to grab the
    correct power, which is generated based on DIF3D indexing"""
    if dassh_loc[0] == 0:
        return 0
    else:
        dif3d_loc = dif3d_loc_to_dassh_loc(dassh_loc)
        return 3 * (dif3d_loc[0] - 1) * dif3d_loc[0] + dif3d_loc[1] + 1


########################################################################


def _get_profile_data(path='dassh_profile.out', n=50):
    """Shortcut to print DASSH profile data"""
    import pstats
    p = pstats.Stats(path)
    p.sort_stats('cumulative').print_stats(n)


########################################################################
