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
date: 2022-10-25
author: matz
Objects and methods to print ASCII tables in Python
"""
########################################################################
# import os
import copy
# import warnings
import numpy as np
from collections import OrderedDict

# import dassh
from dassh import utils
from dassh.logged_class import LoggedClass
_OMIT = '---'
_section_sep = '\n\n\n'
_formatted_temp_units = {'celsius': '˚C',
                         'fahrenheit': '˚F',
                         'kelvin': 'K'}


class DASSH_Table(object):
    """Base class for generating output tables in DASSH

    Parameters
    ----------
    n_col : int
        Number of data columns in the table (does not include column 0)
    col_width : int, optional
        Character width of data columns {default: 11}
    col0_width : int, optional
        Character width of column 0 {default: 23}
    divider : str, optional
        Separator between columns {default: ' '}

    Attributes
    ----------
    table : str
        The table!
    width : in
        The total width of the table
    formatter : str
        Str formatter to conform with column width requirement

    """

    def __init__(self, n_col, col_width=11, col0_width=23, divider=' '):
        """."""
        self._table = ""
        self.n_col = n_col
        self.col_width = col_width
        self.col0_width = col0_width
        self.divider = divider
        self.width = int(col0_width
                         + len(divider) * n_col
                         + n_col * col_width)
        # Build formatter with which to add rows
        self.formatter = '{:>' + f'{col0_width}.{col0_width}' + 's}'
        for i in range(n_col):
            self.formatter += \
                divider + '{:>' + f'{col_width}.{col_width}' + 's}'

    def __str__(self):
        return self.table

    def generate(self, *args):
        """Generate output table string for DASSH output file

        Parameters
        ----------
        *args : anything necessary to make table of interest;
            (specifically, what's called in the "make" method)

        Returns
        -------
        str
            ASCII output table

        """
        # Make the table
        self.make(*args)
        out = ''
        out += _section_sep
        if hasattr(self, 'title'):
            out += self.title
        if hasattr(self, 'uline'):
            out += self.uline
        if hasattr(self, 'notes'):
            out += self.notes
        out += "\n"
        out += self.table
        return out

    @property
    def table(self):
        return self._table

    @property
    def uline(self):
        """Underline the table title"""
        if hasattr(self, 'title'):
            return "-" * len(self.title) + "\n"
        else:
            return ""

    def clear(self):
        """Erase the table"""
        self._table = ""

    def add_row(self, key, values, fmt=None):
        """Add a row to the table

        Parameters
        ----------
        key : str
            Key value that describes the row
        values : list
            Values to fill in the columns
        fmt : str, optional
            Optional formatter to use on the row; else use default
        """
        if fmt is not None:
            self._table += fmt.format(key, *values) + '\n'
        else:
            self._table += self.formatter.format(key, *values) + '\n'

    def add_horizontal_line(self):
        """Add a horizontal line to the table"""
        self._table += '-' * self.width + '\n'

    def _get_len_conv(self, unit):
        """Load the conversion method to process output from DASSH
        units to user-requested units"""
        # Process units from input_obj: length
        if unit not in utils._DEFAULT_UNITS['length']:
            return utils.get_length_conversion('m', unit)
        else:
            return _echo_value

    def _get_temp_conv(self, unit):
        """Load the conversion method to process output from DASSH
        units to user-requested units"""
        # Process units from input_obj: length
        if unit not in utils._DEFAULT_UNITS['temperature']:
            return utils.get_temperature_conversion('K', unit)
        else:
            return _echo_value

    def _get_mfr_conv(self, fr_unit):
        """Load the conversion method to process output from DASSH
        units to user-requested units"""
        # Process units from input_obj: length
        m_unit, t_unit = utils.parse_mfr_units(fr_unit)
        if m_unit not in utils._DEFAULT_UNITS['mass']:
            m_conv = utils.get_mass_conversion('kg', m_unit)
        else:
            m_conv = _echo_value
        if t_unit not in utils._DEFAULT_UNITS['time']:
            # Because time is in the denominator, mult is backwards
            t_conv = utils.get_time_conversion(t_unit, 's')
        else:
            t_conv = _echo_value
        return lambda x: m_conv(t_conv(x))


def _echo_value(value):
    """To use when no unit conversion is desired"""
    return value


def _fmt_idx(idx):
    """Return assembly ID in non-Python indexing"""
    return str(idx + 1)


def _fmt_pos(pos):
    """Return position in non-Python indexing"""
    new_pos = [pi + 1 for pi in pos]
    new_pos = ['{:2d}'.format(pi) for pi in new_pos]
    return f'({new_pos[0]},{new_pos[1]})'


########################################################################


class GeometrySummaryTable(LoggedClass, DASSH_Table):
    """Create summary table of assembly characteristics

    Parameters
    ----------
    input_obj : DASSH_Input object
        Contains input path and units
    decimal : int
        Number of decimal places for rounding, where necessary

    Attributes
    ----------
    path : str
        Path to input file / working directory
    dp : int
        Number of decimal places for rounding, where necessary

    Methods
    -------
    make : Generate summary table
    len_conv : Length conversion from m to user-input unit.

    """

    title = "ASSEMBLY GEOMETRY SUMMARY" + "\n"

    def __init__(self, n_col, col_width=11, col0_width=23, sep=' '):
        """Initialize SummaryTable"""
        LoggedClass.__init__(self, 0, 'dassh.table.Summary')
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}E' + '}'
        # Number of assemblies = number of columns
        self.n_col = n_col
        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, n_col, col_width, col0_width, sep)

    def make(self, r_obj):
        """Create the summary table

        Parameters
        ----------
        r_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        # Process units from input_obj: length
        self.len_unit = r_obj.units['length']
        self.len_conv = self._get_len_conv(self.len_unit)
        assemblies = list(r_obj.asm_templates.values())
        tabdata = self._collect_asm_characteristics(assemblies)
        # Add header (method inherited from DASSH_Table)
        self.add_row(f'Parameter ({self.len_unit} // {self.len_unit})^2',
                     [f'Assembly {i + 1}' for i
                      in range(len(r_obj.asm_templates))])
        # Add horizontal line (method inherited from DASSH_Table)
        self.add_horizontal_line()

        # tabdata is OrderedDict; keys are row parameters
        for key in tabdata:
            # Format the data where necessary; some cols will have mix
            # of string/int or string/float
            # - if int: convert to string
            # - if float: round, convert to string
            # - if string: do nothing
            tmp = tabdata[key]
            for i in range(self.n_col):
                if isinstance(tabdata[key][i], int):
                    tmp[i] = str(tabdata[key][i])
                elif isinstance(tabdata[key][i], float):
                    tmp[i] = self._ffmt.format(tabdata[key][i])
                elif isinstance(tabdata[key][i], str):
                    continue
                else:
                    msg = (f'Unknown type ({type(tabdata[key][i])}) '
                           'encountered in SummaryTable.make')
                    self.log('error', msg)

            # If all values are _OMIT, then this row could use a
            # preceding line break
            if all(vi == _OMIT for vi in tmp) and not key[0].isnumeric():
                self.add_row('', ['' for j in range(len(tmp))])

            # Add the row (method inherited from DASSH_Table)
            self.add_row(key, tmp)

    def _check_line_length(self, n_asm):
        """Check that the table width will be less than 130 characters;
        if not, warn and adjust the number of decimal places"""
        raise NotImplementedError('yuh')

    def _collect_asm_characteristics(self, asm_list):
        """Collect assembly geometry specifications

        Notes
        -----
        Different parameters are printed along the rows for different
        assemblies on across the columns

        """
        # Max number of duct walls in an assembly
        ducts = [a.rodded.n_duct for a in asm_list if a.has_rodded]
        if len(ducts) == 0:
            _MAX_DUCT = 1
        else:
            _MAX_DUCT = np.max(ducts)

        dat = self._initialize(asm_list, _MAX_DUCT)
        for i in range(len(asm_list)):
            if not asm_list[i].has_rodded:
                continue

            a = asm_list[i].rodded
            dat['Name'][i] = a.name
            dat['Pins'][i] = a.n_pin
            dat['Pin rings'][i] = a.n_ring
            dat['Pin diameter'][i] = self.len_conv(a.pin_diameter)
            dat['Pin pitch'][i] = self.len_conv(a.pin_pitch)
            dat['Clad thickness'][i] = self.len_conv(a.clad_thickness)
            dat['Wire pitch'][i] = self.len_conv(a.wire_pitch)
            dat['Wire diameter'][i] = self.len_conv(a.wire_diameter)
            dat['Wire direction'][i] = a.wire_direction
            dat['Pin-pin gap'][i] = self.len_conv(a.d['pin-pin'])
            dat['Pin-wall gap'][i] = self.len_conv(a.d['pin-wall'])

            # Duct wall / bypass gap parameters
            dat['Number of duct walls'][i] = a.n_duct
            dat['Number of bypass gaps'][i] = a.n_bypass
            dat['Outside duct outer FTF'][i] = \
                self.len_conv(a.duct_ftf[-1][-1])
            dat['Outside duct thickness'][i] = \
                self.len_conv(a.d['wall'][-1])
            # Do stuff for asm with multiple ducts, bypass gaps
            # Recall that everything is initialized with _OMIT
            for j in range(1, a.n_duct):
                dat[f'Bypass {_MAX_DUCT - j} thickness'][i] = \
                    self.len_conv(a.d['bypass'][-j])
                dat[f'Duct {_MAX_DUCT - j} outer FTF'][i] = \
                    self.len_conv(a.duct_ftf[-j - 1][-1])
                dat[f'Duct {_MAX_DUCT - j} thickness'][i] = \
                    self.len_conv(a.d['wall'][-j - 1])

            # Subchannel parameters
            # dat['Subchannels'][i] = blank
            dat['Coolant'][i] = a.subchannel.n_sc['coolant']['total']
            dat['1. Interior'][i] = a.subchannel.n_sc['coolant']['interior']
            dat['2. Edge'][i] = a.subchannel.n_sc['coolant']['edge']
            dat['3. Corner'][i] = a.subchannel.n_sc['coolant']['corner']
            dat['Duct (per wall)'][i] = a.subchannel.n_sc['duct']['total']
            dat['4. Edge'][i] = a.subchannel.n_sc['duct']['edge']
            dat['5. Corner'][i] = a.subchannel.n_sc['duct']['corner']
            if a.n_duct > 1:
                dat['Bypass (per gap)'][i] = \
                    a.subchannel.n_sc['bypass']['total']
                dat['6. Edge'][i] = a.subchannel.n_sc['bypass']['edge']
                dat['7. Corner'][i] = a.subchannel.n_sc['bypass']['corner']

            # Flow area: convert twice for length squared
            # dat[f'Subchannel area'][i] = blank
            dat['1. Interior area'][i] = \
                self.len_conv(self.len_conv(a.params['area'][0]))
            dat['2. Edge area'][i] = \
                self.len_conv(self.len_conv(a.params['area'][1]))
            dat['3. Corner area'][i] = \
                self.len_conv(self.len_conv(a.params['area'][2]))
            dat['Interior total area'][i] = \
                self.len_conv(self.len_conv(a.bundle_params['area']))
            for j in range(1, a.n_duct):
                dat[f'6. Bypass {_MAX_DUCT - j} edge area'][i] = \
                    self.len_conv(self.len_conv(
                        a.bypass_params['area'][-j, 0]))
                dat[f'7. Bypass {_MAX_DUCT - j} corner area'][i] = \
                    self.len_conv(self.len_conv(
                        a.bypass_params['area'][-j, 1]))
                dat[f'Bypass {_MAX_DUCT - j} total area'][i] = \
                    self.len_conv(self.len_conv(
                        a.bypass_params['total area'][-j]))

            # Hydraulic Diameters
            dat['1. Interior De'][i] = self.len_conv(a.params['de'][0])
            dat['2. Edge De'][i] = self.len_conv(a.params['de'][1])
            dat['3. Corner De'][i] = self.len_conv(a.params['de'][2])
            dat['Bundle De'][i] = self.len_conv(a.bundle_params['de'])
            for j in range(1, a.n_duct):
                dat[f'6. Bypass {_MAX_DUCT - j} edge De'][i] = \
                    self.len_conv(a.bypass_params['de'][-j, 0])
                dat[f'7. Bypass {_MAX_DUCT - j} corner De'][i] = \
                    self.len_conv(a.bypass_params['de'][-j, 1])
                dat[f'Bypass {_MAX_DUCT - j} total De'][i] = \
                    self.len_conv(a.bypass_params['total de'][-j])

            # Centroid-centroid distances
            # dat['Centroid-centroid dist'][i] = blank
            dat['1 <--> 1'][i] = self.len_conv(a.L[0][0])
            dat['1 <--> 2'][i] = self.len_conv(a.L[0][1])
            dat['2 <--> 2'][i] = self.len_conv(a.L[1][1])
            dat['2 <--> 3'][i] = self.len_conv(a.L[1][2])
            dat['3 <--> 3'][i] = self.len_conv(a.L[2][2])
            for j in range(1, a.n_duct):
                # Edge-edge
                dat[f'Byp {_MAX_DUCT - j} 6 <--> 6'][i] = \
                    self.len_conv(a.L[5][5][-j])
                # Edge-corner
                dat[f'Byp {_MAX_DUCT - j} 6 <--> 7'][i] = \
                    self.len_conv(a.L[5][6][-j])
                # Corner-corner
                dat[f'Byp {_MAX_DUCT - j} 7 <--> 7'][i] = \
                    self.len_conv(a.L[6][6][-j])

            dat['Friction factor'][i] = a.corr_names['ff']
            dat['Flow split'][i] = a.corr_names['fs']
            dat['Mixing parameters'][i] = a.corr_names['mix']
            dat['Nusselt number'][i] = a.corr_names['nu']
            dat['Shape factor'][i] = a._sf
        return dat

    @staticmethod
    def _initialize(asm_list, _MAX_DUCT):
        """Set up empty rows to overwrite with values"""
        n_asm = len(asm_list)
        empty_row = [_OMIT for a in range(n_asm)]
        _SUMMARY_KEYS = ['Name',
                         'Pins',
                         'Pin rings',
                         'Pin diameter',
                         'Pin pitch',
                         'Clad thickness',
                         'Wire pitch',
                         'Wire diameter',
                         'Wire direction',
                         'Pin-pin gap',
                         'Pin-wall gap',
                         'Number of duct walls',
                         'Number of bypass gaps',
                         'Outside duct outer FTF',
                         'Outside duct thickness']
        for j in range(1, _MAX_DUCT):
            _SUMMARY_KEYS += [f'Bypass {_MAX_DUCT - j} thickness',
                              f'Duct {_MAX_DUCT - j} outer FTF',
                              f'Duct {_MAX_DUCT - j} thickness']

        _SUMMARY_KEYS += ['Coolant',
                          '1. Interior',
                          '2. Edge',
                          '3. Corner',
                          'Duct (per wall)',
                          '4. Edge',
                          '5. Corner',
                          'Bypass (per gap)',
                          '6. Edge',
                          '7. Corner',
                          'Subchannel area',
                          '1. Interior area',
                          '2. Edge area',
                          '3. Corner area',
                          'Interior total area']
        for j in range(1, _MAX_DUCT):
            _SUMMARY_KEYS += [f'6. Bypass {_MAX_DUCT - j} edge area',
                              f'7. Bypass {_MAX_DUCT - j} corner area',
                              f'Bypass {_MAX_DUCT - j} total area']

        _SUMMARY_KEYS += ['Hydraulic diam.',
                          '1. Interior De',
                          '2. Edge De',
                          '3. Corner De',
                          'Bundle De']
        for j in range(1, _MAX_DUCT):
            _SUMMARY_KEYS += [f'6. Bypass {_MAX_DUCT - j} edge De',
                              f'7. Bypass {_MAX_DUCT - j} corner De',
                              f'Bypass {_MAX_DUCT - j} total De']

        _SUMMARY_KEYS += ['Centroid-centroid dist',
                          '1 <--> 1',
                          '1 <--> 2',
                          '2 <--> 2',
                          '2 <--> 3',
                          '3 <--> 3']
        for j in range(1, _MAX_DUCT):
            _SUMMARY_KEYS += [f'Byp {_MAX_DUCT - j} 6 <--> 6',
                              f'Byp {_MAX_DUCT - j} 6 <--> 7',
                              f'Byp {_MAX_DUCT - j} 7 <--> 7']

        _SUMMARY_KEYS += ['Correlations',
                          'Friction factor',
                          'Flow split',
                          'Mixing parameters',
                          'Nusselt number',
                          'Shape factor']

        table = OrderedDict()
        for key in _SUMMARY_KEYS:
            table[key] = copy.deepcopy(empty_row)
        return table


########################################################################


class PositionAssignmentTable(LoggedClass, DASSH_Table):
    """Create table of assembly positional parameters

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    make : Generate table

    """

    title = "ASSEMBLY POWER AND ASSIGNED FLOW RATE" + "\n"
    notes = \
        """This table reports the total flow rate and power for each assembly.
The axial mesh size constraint required for each assembly is reported
as "dz".

"Limiting SC" refers to the type of subchannel that constrains the axial mesh
size. The subchannel ID is the first number; the subchannels to which it
is connected are the second set of numbers. Subchannel IDs are (1) interior;
(2) edge; (3) corner; (6) bypass edge; (7); bypass corner. If a bypass
channel is constraining, the third value indicates which bypass gap the
channel resides in.

Example: "3-22" identifies corner subchannel connected to two edge subchannels
Example: "7-66-0" indicates a bypass corner subchannel connected to two
    bypass edge channels in the first (innermost) bypass gap.

Gr* is the modified Grashof number, which indicates whether buoyancy effects
are important in the pin bundle. If the assembly has a pin bundle region and
Gr* >= 0.02, buoyancy effects should be important and the forced convection
representation of the flow is not accurate.
"""

    def __init__(self, col_width=9, col0_width=4, sep='  '):
        """Initialize SummaryTable"""
        LoggedClass.__init__(self, 0, 'dassh.table.Positional')
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}E' + '}'
        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 8, col_width, col0_width, sep)

    def make(self, r_obj):
        """Create the table

        Parameters
        ----------
        r_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        # Process units from input_obj: length
        fr_unit = r_obj.units['mass_flow_rate']
        len_unit = r_obj.units['length']
        mfr_conv = self._get_mfr_conv(fr_unit)
        len_conv = self._get_len_conv(len_unit)
        self.add_row('', ['', '', 'Flow rate', 'Power', 'dz',
                          'Limiting', '', 'Forced'])
        self.add_row('Asm.', ['Name',
                              'Loc.',
                              f'({fr_unit})',
                              '(W)',
                              f'({len_unit})',
                              'SC',
                              'Gr*',
                              'Conv Repr'])
        self.add_horizontal_line()
        for i in range(len(r_obj.assemblies)):
            a = r_obj.assemblies[i]
            fr = mfr_conv(a.flow_rate)
            p = a.total_power
            # Calculate Gr_star
            if a.has_rodded:
                try:
                    gr_star, gr_star_crit = \
                        self._determine_applicability(
                            a, r_obj.inlet_temp, r_obj.core_length)
                except (AttributeError, TypeError, AssertionError):
                    gr_star = _OMIT
                    gr_star_crit = _OMIT
            else:
                gr_star = _OMIT
                gr_star_crit = _OMIT
            self.add_row(
                # _fmt_idx(a.id),
                _fmt_idx(i),
                [a.name,
                 _fmt_pos(a.loc),
                 self._ffmt.format(fr),
                 self._ffmt.format(p),
                 self._ffmt.format(len_conv(r_obj.min_dz['dz'][i])),
                 str(r_obj.min_dz['sc'][i]),
                 gr_star,
                 gr_star_crit]
            )
        if r_obj.core.model == 'flow':
            core_fr = mfr_conv(r_obj.core.gap_flow_rate)
            core_dz = len_conv(r_obj.min_dz['dz'][-1])
            self.add_row(_OMIT, ['gap',
                                 _OMIT,
                                 self._ffmt.format(core_fr),
                                 _OMIT,
                                 self._ffmt.format(core_dz),
                                 str(r_obj.min_dz['sc'][-1]),
                                 _OMIT,
                                 _OMIT])

    def _determine_applicability(self, asm, t_inlet, core_len):
        """x"""
        pin_power_skew = asm.power.calculate_pin_power_skew()
        gr_star = self._calc_modified_gr(asm.rodded,
                                         t_inlet,
                                         asm._estimated_T_out,
                                         pin_power_skew,
                                         core_len)
        if gr_star >= 0.02:
            gr_star_crit = 'ERROR'
        else:
            gr_star_crit = str(u'\u2713')  # check mark
        gr_star = self._ffmt.format(gr_star)  # format for table
        return gr_star, gr_star_crit

    def _calc_modified_gr(self, rr, t_in, t_out, pskew, length):
        """Calculate modified Grashof number to evaluate importance of
        buoyancy effects on flow distribution in bundle

        Parameters
        ----------
        rr : DASSH RoddedRegion object
            Contains bundle characteristics
        t_in : float
            Bundle inlet temperature (K)
        t_out : float
            Estimated bundle outlet temperature (K)
        pskew : float
            Ratio of power (skew) between max and average pin power
        length : float
            Bundle length (m)

        Returns
        -------
        float
            Modified Grashof number to be evaluated against critical
            Grashof number (Gr*_C = 0.2)

        Notes
        -----
        For reference, see:
            1. SE2 Manual (1980); Section 4.3.1
            2. Khan et al, "A Porous Body Model For Predicting Temperature
               Distributions In Wire Wrapped Fuel and Blanket Assemblies
               of a LMFBR" (Chapter 4), COO-2245-16TR, 1975

        """
        # Update coolant properties
        rr._update_coolant_int_params(0.5 * (t_out + t_in))
        # pull out some of them
        Re = (rr.coolant.density
              * rr.coolant_int_params['vel']
              * rr.coolant_int_params['fs'][0]
              * rr.params['de'][0]
              / rr.coolant.viscosity)
        # Novendstern's multiplication factor for wire-wrapped bundles
        # If wire wrap pitch is zero (no wire wrap): treat as though it has
        # a very large wire wrap pitch (as if the wire was nearly vertical);
        # In that case, M approaches 1.0; otherwise, M is greater than 1.0
        if rr.wire_pitch == 0.0:
            M = 1.0
        else:
            M = 1.034 / (rr.pin_pitch / rr.pin_diameter)**0.124
            M += (29.7 * (rr.pin_pitch / rr.pin_diameter)**6.94 * Re**0.086
                  / (rr.wire_pitch / rr.pin_diameter)**2.239)
            M = M**0.885
        # Evaluate some other stuff
        Pr = (rr.coolant.heat_capacity
              * rr.coolant.viscosity
              / rr.coolant.thermal_conductivity)
        # Dimensionless eddy diffusivity
        eddy_dimless = rr.corr['mix'](rr)[0]
        # Gross gamma constant
        gamma = (16 * ((rr.pin_pitch / rr.pin_diameter) - 1)
                 * (rr.coolant_int_params['fs'][0] * eddy_dimless
                    + rr._sf / Re / Pr)
                 / np.pi / rr.duct_ftf[0][0])
        # kinematic viscosity
        kvisc = rr.coolant.viscosity / rr.coolant.density
        # gross chi constant
        chi = (pskew - 1) / 2 / M / gamma / length
        # evaluate Gr and then Gr_star
        ff = rr.coolant_int_params['ff']
        g0 = 9.80665
        beta = rr.coolant.beta
        Gr = g0 * beta * (t_out - t_in) * rr.params['de'][0]**3 / kvisc**2
        Gr_star = Gr * chi / ff / Re**2
        return Gr_star


########################################################################


class CoolantFlowTable(LoggedClass, DASSH_Table):
    """Create table of assembly flow parameters by position

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    make : Generate table


    """

    title = "SUBCHANNEL FLOW CHARACTERISTICS" + "\n"
    notes = \
        """Column heading definitions
    Avg. - Average coolant velocity in rod bundle or assembly
    Int. - Coolant velocity in the interior subchannel
    Edge - Coolant velocity in the edge subchannel
    Corner - Coolant velocity in the corner subchannel
    Bypass - Average coolant velocity in the bypass gap, if applicable
    Swirl - Transverse velocity in edge/corner subchannels due to wire-wrap
    Bundle RE - Average Reynolds number in rod bundle or assembly
    Friction factor - Unitless friction factor for bundle or assembly
    Eddy df. - Correlated eddy diffusivity in subchannels

Notes
- Values reported for coolant at inlet temperature
- Flow split can be obtained as ratio of subchannel and average velocities
- Average values reported for assemblies without rod bundle specification
"""

    def __init__(self, col_width=8, col0_width=4, sep='  '):
        """Instantiate flow parameters output table"""
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}E' + '}'
        self._ffmt0 = '{' + f':.{0}f' + '}'
        self._ffmt3 = '{' + f':.{3}f' + '}'
        self._ffmt5 = '{' + f':.{5}f' + '}'

        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 11, col_width, col0_width, sep)

    def make(self, reactor_obj):
        """Create the table

        Parameters
        ----------
        r_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        # Process units from input_obj: length
        len_unit = reactor_obj.units['length']
        l_conv = self._get_len_conv(len_unit)

        unit_fill = '-' * (self.col_width - len(len_unit) - 4)
        self.add_row('', ['',
                          '',
                          '|' + '-' * (self.col_width - 1),
                          '-' * self.col_width,
                          'Velocity',
                          f'({len_unit}/s) ' + unit_fill,
                          '-' * self.col_width,
                          '-' * (self.col_width - 1) + '|',
                          'Bundle',
                          'Friction',
                          'Eddy Df.'])
        self.add_row('Asm.', ['Name',
                              'Pos.',
                              'Avg.',
                              'Int.',
                              'Edge',
                              'Corner',
                              'Bypass',
                              'Swirl',
                              'RE',
                              'Factor',
                              f'({len_unit}^2/s)'])
        self.add_horizontal_line()
        for i in range(len(reactor_obj.assemblies)):
            a = reactor_obj.assemblies[i]
            if a.has_rodded:
                ar = a.rodded
                params = [a.name,
                          _fmt_pos(a.loc),
                          self._ffmt3.format(
                              l_conv(ar.coolant_int_params['vel'])),
                          self._ffmt3.format(
                              l_conv(ar.coolant_int_params['vel']
                                     * ar.coolant_int_params['fs'][0])),
                          self._ffmt3.format(
                              l_conv(ar.coolant_int_params['vel']
                                     * ar.coolant_int_params['fs'][1])),
                          self._ffmt3.format(
                              l_conv(ar.coolant_int_params['vel']
                                     * ar.coolant_int_params['fs'][2]))]
                if hasattr(ar, 'coolant_byp_params'):
                    params.append(self._ffmt3.format(
                        l_conv(ar.coolant_byp_params['vel'][0])))
                else:
                    params.append(_OMIT)
                params += [
                    self._ffmt3.format(
                        l_conv(ar.coolant_int_params['swirl'][1])),
                    self._ffmt0.format(ar.coolant_int_params['Re']),
                    self._ffmt.format(ar.coolant_int_params['ff']),
                    self._ffmt5.format(
                        l_conv(l_conv(ar.coolant_int_params['eddy'])))]
            else:
                reg = a.region[0]
                params = [a.name,
                          # _fmt_pos(asm_loc[i]),
                          _fmt_pos(a.loc),
                          self._ffmt3.format(
                              l_conv(reg.coolant_params['vel'])),
                          _OMIT,
                          _OMIT,
                          _OMIT,
                          _OMIT,
                          _OMIT,
                          self._ffmt0.format(reg.coolant_params['Re']),
                          self._ffmt.format(reg.coolant_params['ff']),
                          _OMIT]
            self.add_row(_fmt_idx(i), params)


########################################################################


class PressureDropTable(LoggedClass, DASSH_Table):
    """Create table of assembly pressure drop results by Region

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    make : Generate table


    """

    title = "PRESSURE DROP (MPa) ACROSS ASSEMBLIES" + "\n"
    notes = """Notes
- "Total" is the total pressure drop accumulated across the assembly
- "Region 1...N" is the pressure drop in each user-specified axial
    region, starting from the bottom
"""

    def __init__(self, n_reg, col_width=10, col0_width=4, sep='  '):
        """Instantiate flow parameters output table"""
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}e' + '}'
        self._ffmt5 = '{' + f':.{5}f' + '}'
        self._ffmt4e = '{' + f':.{4}E' + '}'

        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, n_reg + 3, col_width, col0_width, sep)

    def make(self, reactor_obj):
        """Create the table

        Parameters
        ----------
        r_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        # Process units from input_obj: length
        n_reg = self.n_col - 3
        header = ['Name', 'Loc.', 'Total']
        header += [f'Region {i + 1}' for i in range(n_reg)]
        self.add_row('Asm.', header)
        self.add_horizontal_line()
        for i in range(len(reactor_obj.assemblies)):
            a = reactor_obj.assemblies[i]
            params = [a.name, _fmt_pos(a.loc)]

            # Include total pressure drop
            params.append(self._ffmt4e.format(a.pressure_drop / 1e6))

            # Fill up the row with blanks; replace as applicable
            params += ['' for ri in range(n_reg)]
            for ri in range(len(a.region)):
                params[ri + 3] = self._ffmt4e.format(
                    a.region[ri].pressure_drop / 1e6)
            # self.add_row(_fmt_idx(a.id), params)
            self.add_row(_fmt_idx(i), params)


########################################################################


class AssemblyEnergyBalanceTable(LoggedClass, DASSH_Table):
    """Assembly energy-balance summary table"""

    title = "OVERALL ASSEMBLY ENERGY BALANCE" + "\n"
    notes = """Column heading definitions
    A - Heat added to coolant through pins or by direct heating (W)
    B - Heat added to duct wall (W)
    C - Heat transferred to assembly-interior coolant through duct wall (W)
    D - Heat transferred to double-duct bypass coolant through duct walls (W)
    E - Assembly coolant mass flow rate (kg/s)
    F - Assembly axially averaged heat capacity (J/kg-K)
    G - Assembly coolant temperature rise (K)
    SUM - Assembly energy balance: A + C + D - E * F * G (W)
    ERROR - SUM / (A + B)""" + "\n"
    # G - Total assembly coolant heat gain through temp. rise: E * D * F (W)

    def __init__(self, col_width=11, col0_width=4, sep='  '):
        """Instantiate assembly energy balance summary output table"""
        # Decimal places for rounding, where necessary
        self.dp = col_width - 7
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}E' + '}'

        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 9, col_width, col0_width, sep)

    def make(self, r_obj):
        """Create the table

        Parameters
        ----------
        r_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        self.add_row('Asm.', ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                              'SUM', 'ERROR'])
        self.add_horizontal_line()
        ebal = np.zeros((len(r_obj.assemblies), 9))
        for i in range(len(r_obj.assemblies)):
            asm = r_obj.assemblies[i]
            ebal[i, :7] = self._calc_asm_energy_balance(asm, r_obj)

        # Use numpy for the calculations based on the preceding data;
        # calculated energy from temp rise and the sum (ebal[:, 7])
        e_temp_rise = ebal[:, 4] * ebal[:, 5] * ebal[:, 6]
        ebal[:, 7] = np.sum(ebal[:, (0, 2, 3)], axis=1) - e_temp_rise

        # Error
        for i in range(len(ebal)):
            total_power = ebal[i, 0] + ebal[i, 1]
            if total_power == 0.0:
                ebal[i, 8] = np.nan
            else:
                ebal[i, 8] = ebal[i, 7] / (ebal[i, 0] + ebal[i, 1])

        # Gap energy balance
        gap_ebal = self._calc_gap_energy_balance(r_obj)

        # Core energy balance
        core_ebal = self._calc_core_energy_balance(ebal, gap_ebal, r_obj)

        # Now add rows to the table
        if r_obj._options['ebal']:
            for i in range(len(r_obj.assemblies)):
                self.add_row(_fmt_idx(i), [self._ffmt.format(x)
                                           for x in ebal[i]])
            # Gap energy balance; add line to differentiate
            # gap/core balances from assembly balances
            self.add_horizontal_line()
            self.add_row('GAP', [self._ffmt.format(x) for x in gap_ebal])
        else:
            for i in range(len(r_obj.assemblies)):
                row = [self._ffmt.format(x) for x in ebal[i]]
                # Omit B, C, SUM, ERROR
                row[2] = _OMIT
                row[3] = _OMIT
                row[7] = _OMIT
                row[8] = _OMIT
                # self.add_row(_fmt_idx(a.id), row)
                self.add_row(_fmt_idx(i), row)
            # Add blank line to split off core balance
            self.add_horizontal_line()
        self.add_row('CORE', core_ebal)

    def _calc_asm_energy_balance(self, asm, r_obj):
        """Get energy balance table entries for an assembly"""
        ebal_asm = np.zeros(7)
        # Energy from direct heating
        ebal_asm[0] += (asm._power_delivered['pins']
                        + asm._power_delivered['cool']
                        + asm._power_delivered['refl'])
        # Energy from duct heating
        ebal_asm[1] = asm._power_delivered['duct']
        # If energy balance was tracked, add HT term w duct
        if r_obj._options['ebal']:
            for reg in asm.region:
                reg.collect_ebal()
                # Energy from direct heating
                # ebal[asm_id[i], 0] += reg.ebal['power']
                # Energy transferred in from duct
                ebal_asm[2] += np.sum(reg.ebal['duct'])
                # Energy from duct wall to bypass coolant
                if 'duct_byp_in' in reg.ebal:
                    ebal_asm[3] += np.sum(reg.ebal['duct_byp_in'])
                    ebal_asm[3] += np.sum(reg.ebal['duct_byp_out'])

        # Assembly mass flow rate
        ebal_asm[4] = asm.flow_rate
        # Average Cp
        ebal_asm[5] = self._get_asm_avg_cp(asm, r_obj)
        # Temperature rise
        ebal_asm[6] = asm.avg_coolant_temp - r_obj.inlet_temp
        return ebal_asm

    def _calc_gap_energy_balance(self, r_obj):
        """Calculate energy balance on inter-assembly gap coolant"""
        gap_ebal = np.zeros(9)
        gap_ebal[3] = np.sum(r_obj.core.ebal['asm'])
        gap_ebal[4] = r_obj.core.gap_flow_rate
        gap_ebal[5] = self._get_gap_avg_cp(r_obj)
        gap_ebal[6] = r_obj.core.avg_coolant_gap_temp - r_obj.inlet_temp
        gap_ebal[7] = gap_ebal[3] - gap_ebal[4] * gap_ebal[5] * gap_ebal[6]
        if gap_ebal[3] != 0.0:
            gap_ebal[8] = gap_ebal[7] / gap_ebal[3]
        return gap_ebal

    def _calc_core_energy_balance(self, asm_ebal, gap_ebal, r_obj):
        """Calculate overall energy balance on core"""
        core_tot = np.zeros(9)
        core_tot[0] = np.sum(asm_ebal[:, 0])
        core_tot[1] = np.sum(asm_ebal[:, 1])
        core_tot[4] = r_obj.flow_rate

        # Flow rate- and axial-average heat capacity
        numerator = np.sum(asm_ebal[:, 4] * asm_ebal[:, 5])
        denominator = np.sum(asm_ebal[:, 4])
        if r_obj.core.model == 'flow':
            numerator += gap_ebal[4] * gap_ebal[5]
            denominator += gap_ebal[4]
        core_tot[5] = numerator / denominator

        # Flow rate- and axial-average temperature change
        numerator = np.sum(asm_ebal[:, 4] * asm_ebal[:, 6])
        denominator = np.sum(asm_ebal[:, 4])
        if r_obj.core.model == 'flow':
            numerator += gap_ebal[4] * gap_ebal[6]
            denominator += gap_ebal[4]
        core_tot[6] = numerator / denominator

        # Calculate total energy change due to temp rise
        q_dt_tot = np.sum(asm_ebal[:, 4] * asm_ebal[:, 5] * asm_ebal[:, 6])
        if r_obj.core.model == 'flow':
            q_dt_tot += gap_ebal[4] * gap_ebal[5] * gap_ebal[6]

        sum1 = (core_tot[0] + core_tot[1]
                - core_tot[4] * core_tot[5] * core_tot[6])
        # sum2 = (core_tot[0] + core_tot[1] - q_dt_tot)
        # print(sum1, sum2, sum1 - sum2)
        core_tot[7] = sum1
        total_power = core_tot[0] + core_tot[1]
        if total_power == 0.0:
            core_tot[8] = np.nan
        else:
            core_tot[8] = core_tot[7] / (core_tot[0] + core_tot[1])
        core_tot = [self._ffmt.format(x) for x in core_tot]
        core_tot[2] = _OMIT
        core_tot[3] = _OMIT
        return core_tot

    @staticmethod
    def _get_asm_avg_cp(asm, r_obj):
        asm.region[0].coolant.update(r_obj.inlet_temp)
        cp1 = asm.region[0].coolant.heat_capacity
        asm.region[0].coolant.update(asm.avg_coolant_temp)
        cp2 = asm.region[0].coolant.heat_capacity
        return np.average([cp1, cp2])

    @staticmethod
    def _get_gap_avg_cp(r_obj):
        r_obj.core.gap_coolant.update(r_obj.inlet_temp)
        cp1 = r_obj.core.gap_coolant.heat_capacity
        r_obj.core.gap_coolant.update(r_obj.core.avg_coolant_gap_temp)
        cp2 = r_obj.core.gap_coolant.heat_capacity
        return np.average([cp1, cp2])

########################################################################


class InterasmEnergyXferTable(LoggedClass, DASSH_Table):
    """Inter-assembly heat transfer summary table"""

    title = "INTER-ASSEMBLY HEAT TRANSFER" + "\n"
    notes = """Notes
- Tracks heat transfer between assemblies and inter-assembly gap coolant
- Positive values indicate heat gained by assemblies through inter-assembly gap
- Duct faces are as shown in the key below
- Adjacent assembly ID is shown in parentheses next to each value.

Duct face key                  Face 6    =   Face 1
    Face 1:  1-o'clock                =     =
    Face 2:  3-o'clock             =           =
    Face 3:  5-o'clock     Face 5  =           =  Face 2
    Face 4:  7-o'clock             =           =
    Face 5:  9-o'clock                =     =
    Face 6: 11-o'clock         Face 4    =   Face 3
"""

    def __init__(self, col_width=16, col0_width=4, sep='  '):
        """Instantiate assembly energy balance summary output table"""
        # Decimal places for rounding, where necessary
        self.dp = 3
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}E' + '}'
        self._ffmt2 = '{:.5E}'
        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 7, col_width, col0_width, sep)

    def make(self, r_obj):
        """Create the table

        Parameters
        ----------
        r_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        # Customize row 0
        row = ' ' * (self.col0_width + self.col_width) + 2 * self.divider
        row += '|-- '
        row += 'Power (W) through outer duct face; '
        row += '(adjacent assembly ID) -->'
        self._table += row + '\n'
        self.add_row('Asm.', ['Power (W)', 'Face 1', 'Face 2', 'Face 3',
                              'Face 4', 'Face 5', 'Face 6'])
        self.add_horizontal_line()

        for i in range(len(r_obj.assemblies)):
            # Get heat transfer total per side - need DASSH ID to pull
            # data from the interassembly HT array
            per_side = np.zeros(6)
            s = 0  # side index
            for sc in range(len(r_obj.core._asm_sc_types[i])):
                if r_obj.core._asm_sc_types[i][sc] == 1:
                    if s == 5:
                        sp1 = 0
                    else:
                        sp1 = s + 1
                    # Corner subchannels; need to split based on half
                    # length along each face, which can be different
                    L = (r_obj.core._geom_params['dims'][i][s][1]
                         + r_obj.core._geom_params['dims'][i][sp1][1])
                    x1 = r_obj.core._geom_params['dims'][i][s][1] / L
                    x2 = 1 - x1
                    per_side[s] += x1 * r_obj.core.ebal['asm'][i][sc]
                    per_side[sp1] += x2 * r_obj.core.ebal['asm'][i][sc]
                    # Update side index
                    s = sp1
                else:
                    per_side[s] += r_obj.core.ebal['asm'][i][sc]

            # Multiple table by (-1): energy xfer is on coolant, taken
            # as q = hA(T_wall - T_coolant); that means that when q < 0,
            # T_wall < T_coolant and the coolant is losing heat to the
            # duct, but the duct is gaining heat from the coolant. To see
            # it from the assembly perspective, need to switch the sign
            per_side *= -1.0

            # Get adjacent assemblies
            adj_id = []
            for adji in r_obj.core.asm_adj[i]:
                if adji > 0 and adji <= len(r_obj.assemblies):
                    adj_id.append(adji)
                else:
                    adj_id.append(-1)

            # Create the row
            row = []
            row.append(self._ffmt2.format(r_obj.assemblies[i].total_power))
            for col in range(6):
                entry = self._ffmt.format(per_side[col])
                if adj_id[col] < 0:
                    entry += f' ({_OMIT})'
                else:
                    entry += f' ({(adj_id[col]):03d})'
                row.append(entry)
            # self.add_row(_fmt_idx(a.id), row)
            self.add_row(_fmt_idx(i), row)


########################################################################


class CoolantTempTable(LoggedClass, DASSH_Table):
    """Create table for coolant temperatures

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    make : Generate table

    """

    title = "COOLANT TEMPERATURE SUMMARY" + "\n"
    notes = """Column heading definitions
    Power - Total assembly power
    Bulk outlet - Mixed-mean coolant temp. at the assembly outlet
    Peak outlet - Maximum coolant subchannel temp. at the assembly outlet
    Peak total - Axial-maximum coolant subchannel temp. in the assembly
    Height - Axial height at which "Peak total" temp. occurs
"""

    def __init__(self, col_width=12, col0_width=4, sep=' '):
        """Instantiate coolant temperature summary output table"""
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}e' + '}'
        self._ffmt2 = '{' + f':.{2}f' + '}'

        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 7, col_width, col0_width, sep)

    def make(self, r_obj):
        """Create the table

        Parameters
        ----------
        reactor_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        # Erase any existing data in the table
        self.clear()
        # Process units from input_obj: temp, length, mfr
        fr_unit = r_obj.units['mass_flow_rate']
        len_unit = r_obj.units['length']
        mfr_conv = self._get_mfr_conv(fr_unit)
        len_conv = self._get_len_conv(len_unit)
        temp_unit = r_obj.units['temperature']
        fmttd_temp_unit = _formatted_temp_units[temp_unit]
        self.temp_conv = self._get_temp_conv(temp_unit)
        self.add_row('', ['',
                          'Power',
                          'Flow rate',
                          'Bulk outlet',
                          'Peak outlet',
                          'Peak total',
                          'Peak ht.'])
        self.add_row('Asm.', ['Loc.',
                              '(W)',
                              f'({fr_unit})',
                              f'({fmttd_temp_unit})',
                              f'({fmttd_temp_unit})',
                              f'({fmttd_temp_unit})',
                              f'({len_unit})'])
        self.add_horizontal_line()

        for i in range(len(r_obj.assemblies)):
            a = r_obj.assemblies[i]

            # Here again, the first column in the dump file is the
            # DASSH ID, so we need to use the assembly.ID attribute
            # to locate the rows we want to search

            # Coolant temperatures
            tc_avg = self.temp_conv(a.avg_coolant_temp)
            tc_max_out = self.temp_conv(
                np.max(a.region[-1].temp['coolant_int']))
            try:
                tc_max_tot, tc_max_ht = a._peak['cool']
            except TypeError:
                tc_max_tot = a._peak['cool']
                tc_max_ht = _OMIT
            else:  # only format height as float if you get one
                tc_max_ht = self._ffmt2.format(len_conv(tc_max_ht))
            tc_max_tot = self.temp_conv(tc_max_tot)

            data = [_fmt_pos(a.loc),
                    '{:.5E}'.format(a.total_power),
                    '{:.5E}'.format(mfr_conv(a.flow_rate)),
                    self._ffmt2.format(tc_avg),
                    self._ffmt2.format(tc_max_out),
                    self._ffmt2.format(tc_max_tot),
                    tc_max_ht]
            # self.add_row(_fmt_idx(a.id), data)
            self.add_row(_fmt_idx(i), data)


########################################################################


class DuctTempTable(LoggedClass, DASSH_Table):
    """Create table for duct temperatures

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    make : Generate table

    """
    title = "DUCT TEMPERATURE SUMMARY" + "\n"
    notes = """Column heading definitions
    Average temp. - Duct mid-wall temperature per face at core outlet
    Peak temp. - Axial peak duct mid-wall temperature
    Peak Ht. - Axial position at which peak duct MW temperature occurs

Duct face key                  Face 6    =   Face 1
    Face 1:  1-o'clock                =     =
    Face 2:  3-o'clock             =           =
    Face 3:  5-o'clock     Face 5  =           =  Face 2
    Face 4:  7-o'clock             =           =
    Face 5:  9-o'clock                =     =
    Face 6: 11-o'clock         Face 4    =   Face 3
"""

    def __init__(self, col_width=9, col0_width=4, sep=' '):
        """Instantiate coolant temperature summary output table"""
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}e' + '}'
        self._ffmt2 = '{' + f':.{2}f' + '}'

        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 10, col_width, col0_width, sep)

    def make(self, r_obj):
        """Create the duct temperature summary table

        Parameters
        ----------
        reactor_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        # Erase any existing data in the table
        self.clear()
        # Process units from input_obj: temp, length
        len_unit = r_obj.units['length']
        len_conv = self._get_len_conv(len_unit)
        temp_unit = r_obj.units['temperature']
        fmttd_temp_unit = _formatted_temp_units[temp_unit]
        self.temp_conv = self._get_temp_conv(temp_unit)

        # Custom format the top row
        row = ' ' * (self.col0_width)
        row += ' ' * 2 * self.col_width
        row += self.divider * 3
        mrgd = f'Average duct MW temperature ({fmttd_temp_unit})'
        char2fill = 6 * self.col_width + 5 * len(self.divider)
        ndash = char2fill - len(mrgd) - 4
        dashes_per_side = np.array(
            [np.floor(ndash / 2), np.ceil(ndash / 2)],
            dtype=int)
        assert sum(dashes_per_side) == ndash
        row += (f' |{(dashes_per_side[0] - 1) * "-"} {mrgd}'
                f' {dashes_per_side[1] * "-"}| ')
        fmtter = '{:>' + f'{self.col_width}.{self.col_width}' + 's}'
        row += self.divider.join([fmtter.format('Peak temp'),
                                  fmtter.format('Peak ht.')])
        self._table += row + '\n'
        self.add_row('Asm.', ['Loc.',
                              'Duct ID',
                              'Face 1',
                              'Face 2',
                              'Face 3',
                              'Face 4',
                              'Face 5',
                              'Face 6',
                              f'({fmttd_temp_unit})',
                              f'({len_unit})'])
        self.add_horizontal_line()

        for i in range(len(r_obj.assemblies)):
            a = r_obj.assemblies[i]
            # nduct = len(a._peak['duct'])
            face_temps = self.temp_conv(
                self._get_avg_duct_face_temp(a))
            for d in range(len(face_temps)):
                try:
                    td_max_tot, td_max_ht = a._peak['duct'][d]
                except TypeError:
                    td_max_tot = a._peak['duct'][d]
                    td_max_ht = _OMIT
                else:  # only format height as float if you get one
                    td_max_ht = self._ffmt2.format((len_conv(td_max_ht)))
                td_max_tot = self.temp_conv(td_max_tot)
                data = [_fmt_pos(a.loc), str(d + 1)]
                data += [self._ffmt2.format(td) for td in face_temps[d]]
                data += [self._ffmt2.format(td_max_tot), td_max_ht]
                # self.add_row(_fmt_idx(a.id), data)
                self.add_row(_fmt_idx(i), data)

    @staticmethod
    def _get_avg_duct_face_temp(asm):
        """Average the duct hex face temperatures"""
        # temps = np.zeros(6)
        nduct = asm.region[-1].temp['duct_mw'].shape[0]
        ndsc = asm.region[-1].temp['duct_mw'].shape[1]
        temps = np.zeros((nduct, 6))
        if ndsc == 6:
            for d in range(nduct):
                temps[d] = np.average(
                    [asm.region[-1].temp['duct_mw'][d],
                     np.roll(asm.region[-1].temp['duct_mw'][d], 1)
                     ],
                    axis=0)
        else:
            ndps = int(ndsc / 6)
            for d in range(nduct):
                tmp = asm.region[-1].temp['duct_mw'][d].copy()
                tmp.shape = (6, ndps)
                tmp2 = np.zeros((6, ndps + 1))
                tmp2[:, 1:] = tmp
                tmp2[:, 0] = np.roll(tmp[:, -1], 1)
                temps[d] = np.average(tmp2, axis=1)
        return temps

########################################################################


class PeakPinTempTable(LoggedClass, DASSH_Table):
    """Create table for nominal and 2-sigma peak pin temperatures

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    make : Generate table

    """

    _lookup_keys = {'clad': {'od': 'clad_od',
                             'mw': 'clad_mw',
                             'id': 'clad_id'},
                    'fuel': {'od': 'fuel_od',
                             'cl': 'fuel_cl'}}

    _notes = """The table shows radial pin temperatures in the pin and at the
height where the peak {} {} temperature occurs. Nominal
temperatures are those calculated directly by DASSH.

{}""" + "\n"

    def __init__(self, component, region,
                 col_w=7, col0_w=4, sep=' ', ffmt2=1):
        """Instantiate peak pin temperature output table

        Parameters
        ----------
        component : str {'clad', 'fuel'}
            Report clad/fuel temperatures at height of peak
            temperature for this component
            - 'clad': Clad midwall
            - 'fuel': Fuel centerline
        region : str
            If component = 'clad': {'od', 'mw', 'id'}
            If component = 'fuel': {'od', 'cl'}

        """
        self._component = component
        self._region = region
        # Decimal places for rounding, where necessary
        self.dp = col_w - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}e' + '}'
        self._ffmt2 = '{' + f':.{ffmt2}f' + '}'
        # Number of data columns - does not include column 0
        if component == 'clad':
            if region == 'od':
                n_cols = 12
            elif region == 'mw':
                n_cols = 13
            else:
                n_cols = 14
        else:  # component == 'fuel'
            if region == 'od':
                n_cols = 15
            else:
                n_cols = 16
        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, n_cols, col_w, col0_w, sep)

    def make(self, r_obj, hotspot_data=None):
        """Create the table

        Parameters
        ----------
        r_obj : DASSH Reactor object
            Contains the assembly data to print
        hotspot_data (optional) : tuple
            numpy.ndarray of N-sigma hotspot temps
            list of corresponding asm IDs
            path to subfactors used in the calculation
            input and output "sigma" values

        """
        # Override the title and notes
        self.title = 'PEAK {0} {1} TEMPERATURES'.format(
            self._component.upper(), self._region.upper()) + "\n"
        if hotspot_data is not None:
            msg = self._make_hotspot_msg(r_obj)
            self._hs_lookup = self._component.lower()
            self._hs_lookup += '_' + self._region.lower()
        else:
            msg = """Two-sigma peak temperatures are not calculated."""
        self.notes = self._notes.format(self._component, self._region, msg)

        # Erase any existing data in the table
        self.clear()

        # Process units from input_obj: temp, length
        temp_unit = r_obj.units['temperature']
        fmttd_temp_unit = _formatted_temp_units[temp_unit]
        self.temp_conv = self._get_temp_conv(temp_unit)
        len_unit = r_obj.units['length']
        self.len_conv = self._get_len_conv(len_unit)

        # Custom format the top row
        row = ' ' * (self.col0_width)
        row += ' ' * self.col_width * 4
        row += self.divider * 5
        row += f'| Nominal Peak Temps ({fmttd_temp_unit}) '
        row += '----------------------'
        row += f'| N-Sigma Peak Temps ({fmttd_temp_unit}) '
        row += '-' * (self.width - len(row))
        self._table += row + '\n'

        # Custom format the second row
        fmtter = '{:>' + f'{self.col_width}.{self.col_width}' + 's}'
        row = ['Asm',  # Name
               '',  # Pin
               fmtter.format('Height'),
               fmtter.format('Power'),
               '|' + ' ' * (self.col_width - 1),  # Coolant
               fmtter.format('Clad'),  # OD
               fmtter.format('Clad'),  # MW
               fmtter.format('Clad'),  # ID
               fmtter.format('Fuel'),  # OD
               fmtter.format('Fuel'),  # CL
               '|' + ' ' * (self.col_width - 1),  # Coolant
               fmtter.format('Clad'),  # OD
               fmtter.format('Clad'),  # MW
               fmtter.format('Clad'),  # ID
               fmtter.format('Fuel'),  # OD
               fmtter.format('Fuel')]  # CL
        row = row[:(self.n_col)]
        self.add_row('Asm', row)

        # Custom format the third row
        row = [fmtter.format('Name'),
               fmtter.format('Pin'),
               f'({len_unit})',  # Height unit
               f'(W/{len_unit})',  # Power unit
               '|  Cool',
               'OD',  # Clad
               'MW',  # Clad
               'ID',  # Clad
               'OD',  # Pin
               'CL',  # Pin
               '|  Cool',
               'OD',  # Clad
               'MW',  # Clad
               'ID',  # Clad
               'OD',  # Pin
               'CL']   # Pin
        row = row[:(self.n_col)]
        self.add_row('ID', row)
        self.add_horizontal_line()

        # Get peak temperatures @ height of peak component temp
        tab = []
        for i in range(len(r_obj.assemblies)):
            a = r_obj.assemblies[i]
            row = [_fmt_idx(i)]
            if 'pin' in a._peak.keys():
                k = self._lookup_keys[self._component][self._region]
                row += self._get_nominal_temps(a, a._peak['pin'][k][2])
                row += self._get_hotspot_temps(hotspot_data, a.id)
                tab.append(row)
            else:
                continue

        # Write the data to the table
        for row in tab:
            self.add_row(row[0], row[1:])

    def _get_nominal_temps(self, asm_obj, data):
        """Get cladding/fuel temperatures at the requested height
        for the requested pin"""
        # Get pin (data[2]) and height (z = data[1])
        pin = str(int(data[2]))
        height = self._ffmt2.format(self.len_conv(data[1]))
        out = [asm_obj.name, pin, height]

        # Get power: round height to make sure no numerical error
        plin = asm_obj.power.get_power(round(data[1], 10))
        plin = plin['pins'][int(data[2])] / self.len_conv(1)
        out.append(self._ffmt2.format(plin))

        # Add temperatures and return
        out += [self._ffmt2.format(self.temp_conv(x)) for x in data[3:]]
        return out

    def _get_hotspot_temps(self, hotspot_data, a_id):
        """Get hotspot cladding/fuel temperatures"""
        # Note: hotspot_data == (t_hotspot, asm_ids)
        empty = ['-----' for i in range(self.n_col - 9)]
        if not hotspot_data:
            return empty
        elif self._hs_lookup not in hotspot_data[0].keys():
            return empty
        elif a_id not in hotspot_data[1]:
            return empty
        else:
            idx = hotspot_data[1].index(a_id)
            # tmp = hotspot_data[0][idx]
            # print(tmp)
            # print([self._ffmt2.format(self.temp_conv(x)) for x in tmp])
            t_hotspot = [self._ffmt2.format(self.temp_conv(x))
                         for x in hotspot_data[0][self._hs_lookup][idx]]
            if len(t_hotspot) > len(empty):
                t_hotspot = t_hotspot[:len(empty)]
            elif len(t_hotspot) < len(empty):
                diff = len(empty) - len(t_hotspot)
                t_hotspot += ['-----' for i in range(diff)]
            else:
                pass
            return t_hotspot

    def _make_hotspot_msg(self, r_obj):
        """ x """
        msg = """Hot spot temperatures are calculated based on:
- user input for hot channel factors (built-in or user-specified);
- the degree of uncertainty in the provided statistical factors; and
- the degree of uncertainty desired in the output hotspot temperatures.

Assembly        Input unc.   Output unc.  Hotspot subfactors
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
"""
        for a in r_obj._options['hotspot'].keys():
            line = []
            line.append(a[:14])
            line.append(" " * (14 - len(line[0])) + "  ")
            k = "input_sig_" + self._component
            line.append(f"{r_obj._options['hotspot'][a][k]}-sigma")
            line.append(" " * (11 - len(line[-1])) + "  ")
            k = "output_sig_" + self._component
            line.append(f"{r_obj._options['hotspot'][a][k]}-sigma")
            line.append(" " * (11 - len(line[-1])) + "  ")
            k = "subfactors_" + self._component
            line.append(r_obj._options['hotspot'][a][k])
            line = "".join(line)
            msg += line + "\n"
        return msg

########################################################################
