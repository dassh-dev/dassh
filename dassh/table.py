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
date: 2020-12-15
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
_formatted_temp_units = {'Celsius': '˚C',
                         'Fahrenheit': '˚F',
                         'Kelvin': 'K'}


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


def _sorted_assemblies_idx(reactor_obj):
    if reactor_obj._options['dif3d_idx']:
        return [a.dif3d_id for a in reactor_obj.assemblies]
    else:
        return [a.id for a in reactor_obj.assemblies]


def _sorted_assemblies_attr(reactor_obj):
    # Pregenerate a bunch of temporary objects: this is necessary
    # in order to get proper indexing of assemblies, which varies
    # depending on whether the user asked for DIF3D indexing or
    # DASSH indexing.
    re_index = _sorted_assemblies_idx(reactor_obj)
    assemblies = [reactor_obj.assemblies[i] for i in re_index]
    if reactor_obj._options['dif3d_idx']:
        asm_id = [a.dif3d_id for a in assemblies]
        asm_loc = [a.dif3d_loc for a in assemblies]
    else:
        assemblies = reactor_obj.assemblies
        asm_id = [a.id for a in assemblies]
        asm_loc = [a.loc for a in assemblies]
    return assemblies, asm_id, asm_loc


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
        self.add_row(f'Parameter ({self.len_unit} // sq{self.len_unit})',
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

"Desc." refers to the type of subchannel that constrains the axial mesh
size. The subchannel ID is the first number; the subchannels to which it
is connected are the second set of numbers. Subchannel IDs are (1) interior;
(2) edge; (3) corner; (6) bypass edge; (7); bypass corner. If a bypass
channel is constraining, the third value indicates which bypass gap the
channel resides in.

Example: "3-22" identifies corner subchannel connected to two edge subchannels
Example: "7-66-0" indicates a bypass corner subchannel connected to two
    bypass edge channels in the first (innermost) bypass gap.
"""

    def __init__(self, col_width=9, col0_width=4, sep='  '):
        """Initialize SummaryTable"""
        LoggedClass.__init__(self, 0, 'dassh.table.Positional')
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}E' + '}'
        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 6, col_width, col0_width, sep)

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
        self.add_row('', ['', '', 'Flow rate', 'Power', 'dz', ''])
        self.add_row('Asm.', ['Name',
                              'Loc.',
                              f'({fr_unit})',
                              '(W)',
                              f'({len_unit})',
                              'Desc.'])
        self.add_horizontal_line()

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(r_obj)
        re_index = _sorted_assemblies_idx(r_obj)
        asm_dz = [r_obj.min_dz['dz'][i] for i in re_index]
        asm_dz_sc = [r_obj.min_dz['sc'][i] for i in re_index]
        for i in range(len(assemblies)):
            a = assemblies[i]
            fr = mfr_conv(a.flow_rate)
            p = a.total_power
            dzi = len_conv(asm_dz[i])
            sci = asm_dz_sc[i]
            self.add_row(_fmt_idx(asm_id[i]), [a.name,
                                               _fmt_pos(asm_loc[i]),
                                               self._ffmt.format(fr),
                                               self._ffmt.format(p),
                                               self._ffmt.format(dzi),
                                               str(sci)])
        if r_obj.core.model == 'flow':
            core_fr = mfr_conv(r_obj.core.gap_flow_rate)
            core_dz = len_conv(r_obj.min_dz['dz'][-1])
            self.add_row(_OMIT, ['gap',
                                 _OMIT,
                                 self._ffmt.format(core_fr),
                                 _OMIT,
                                 self._ffmt.format(core_dz),
                                 str(r_obj.min_dz['sc'][-1])])


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
        # self.add_row('Asm ID', ['Name',
        #                         'Loc.',
        #                         'Average',
        #                         'Interior',
        #                         'Edge',
        #                         'Corner',
        #                         'Bypass',
        #                         'Swirl',
        #                         'RE',
        #                         'Factor',
        #                         f'({len_unit}**2/s)'])
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

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(reactor_obj)
        for i in range(len(assemblies)):
            a = assemblies[i]
            if a.has_rodded:
                ar = a.rodded
                params = [a.name,
                          _fmt_pos(asm_loc[i]),
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
                          _fmt_pos(asm_loc[i]),
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

            self.add_row(_fmt_idx(asm_id[i]), params)


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

    def __init__(self, n_reg, col_width=9, col0_width=4, sep='  '):
        """Instantiate flow parameters output table"""
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}e' + '}'
        self._ffmt5 = '{' + f':.{5}f' + '}'

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

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(reactor_obj)
        for i in range(len(assemblies)):
            a = assemblies[i]
            params = [a.name, _fmt_pos(asm_loc[i])]

            # Include total pressure drop
            params.append(self._ffmt5.format(a.pressure_drop / 1e6))

            # Fill up the row with blanks; replace as applicable
            params += ['' for ri in range(n_reg)]
            for ri in range(len(a.region)):
                params[ri + 3] = self._ffmt5.format(
                    a.region[ri].pressure_drop / 1e6)
            self.add_row(_fmt_idx(asm_id[i]), params)


########################################################################


class AssemblyEnergyBalanceTable(LoggedClass, DASSH_Table):
    """Assembly energy-balance summary table"""

    title = "OVERALL ASSEMBLY ENERGY BALANCE" + "\n"
    notes = """Column heading definitions
    A - Heat added to coolant through pins or by direct heating (W)
    B - Heat added to assembly-interior coolant through duct wall (W)
    C - Heat added to bypass coolant through duct wall (W)
    E - Assembly coolant mass flow rate (kg/s)
    D - Assembly axially averaged heat capacity (J/kg-K)
    F - Assembly coolant temperature rise (K)
    G - Total assembly coolant heat gain through temp. rise: E * D * F (W)
    SUM - Assembly energy balance: A + B + C - G (W)
    ERROR - Error for this assembly: SUM / G""" + "\n"

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
        # Process units from input_obj: length
        # fr_unit = r_obj.units['mass_flow_rate']
        # len_unit = r_obj.units['length']
        # mfr_conv = self._get_mfr_conv(fr_unit)
        # len_conv = self._get_len_conv(len_unit)
        # self.add_row('', ['', '', 'Flow rate', 'Power', 'dz', ''])
        self.add_row('Asm.', ['A', 'B', 'C', 'D', 'E', 'F', 'G',
                              'SUM', 'ERROR'])
        self.add_horizontal_line()

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(r_obj)
        ebal = np.zeros((len(assemblies), 9))
        for i in range(len(assemblies)):
            a = assemblies[i]
            for reg in a.region:
                # Energy from direct heating
                ebal[asm_id[i], 0] += reg.ebal['power_added']
                # Energy transferred in from duct
                ebal[asm_id[i], 1] += reg.ebal['from_duct']
                # Energy from duct wall to bypass coolant
                if 'from_duct_byp' in reg.ebal:
                    ebal[asm_id[i], 2] += \
                        np.sum(reg.ebal['from_duct_byp'])

            # Assembly mass flow rate
            ebal[asm_id[i], 3] = a.flow_rate
            # Average Cp
            ebal[asm_id[i], 4] = self._get_avg_cp(a, r_obj.inlet_temp,
                                                  a.avg_coolant_temp)
            # Temperature rise
            ebal[asm_id[i], 5] = a.avg_coolant_temp - r_obj.inlet_temp

        # Use numpy for the calculations based on the preceding data
        ebal[:, 6] = ebal[:, 3] * ebal[:, 4] * ebal[:, 5]
        # Sum
        ebal[:, 7] = np.sum(ebal[:, (0, 1, 2)], axis=1) - ebal[:, 6]
        # Error
        ebal[:, 8] = ebal[:, 7] / ebal[:, 6]

        # Now add rows to the table
        for i in range(len(assemblies)):
            self.add_row(_fmt_idx(asm_id[i]),
                         [self._ffmt.format(x)
                          for x in ebal[asm_id[i]]]
                         )

    @staticmethod
    def _get_avg_cp(asm, t1, t2):
        asm.region[0].coolant.update(t1)
        cp1 = asm.region[0].coolant.heat_capacity
        asm.region[0].coolant.update(t2)
        cp2 = asm.region[0].coolant.heat_capacity
        return np.average([cp1, cp2])


########################################################################


class CoreEnergyBalanceTable(LoggedClass, DASSH_Table):
    """Assembly energy-balance summary table"""

    title = "INTER-ASSEMBLY ENERGY BALANCE" + "\n"
    notes = """Notes
- Tracks energy balance on inter-assembly gap coolant
- Duct faces are as shown in the key below
- Reported values include all "side" meshes and 1/2 each corner.
- Adjacent assembly ID is shown in parentheses next to each value.

Duct face key                  Face 6    =   Face 1
    Face 1:  1-o'clock                =     =
    Face 2:  3-o'clock             =           =
    Face 3:  5-o'clock     Face 5  =           =  Face 2
    Face 4:  7-o'clock             =           =
    Face 5:  9-o'clock                =     =
    Face 6: 11-o'clock         Face 4    =   Face 3
"""

    def __init__(self, col_width=17, col0_width=4, sep='  '):
        """Instantiate assembly energy balance summary output table"""
        # Decimal places for rounding, where necessary
        self.dp = 4
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}E' + '}'

        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 6, col_width, col0_width, sep)

    def make(self, r_obj):
        """Create the table

        Parameters
        ----------
        r_obj : DASSH Reactor object
            Contains the assembly data to print

        """
        # Customize row 0
        row = ' ' * (self.col0_width) + self.divider
        row += '|-- '
        row += 'Energy through outer duct face (W); '
        row += '(adjacent assembly ID) -->'
        self._table += row + '\n'
        self.add_row('Asm.', ['Face 1', 'Face 2', 'Face 3',
                              'Face 4', 'Face 5', 'Face 6'])
        self.add_horizontal_line()

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(r_obj)
        for i in range(len(assemblies)):
            id = asm_id[i]
            a = assemblies[i]

            # Get heat transfer total per side
            tmp = r_obj.core._ebal['interasm'][id].reshape(6, -1).copy()
            tmp2 = np.zeros((6, r_obj.core._sc_per_side + 2))
            tmp2[:, 1:] = tmp
            tmp2[:, 0] = np.roll(tmp[:, -1], 1)
            tmp2[:, (0, -1)] *= 0.5
            eps = np.sum(tmp2, axis=1)

            # Get adjacent assemblies - adjacency array is in DASSH
            # indexing so need to convert if dif3d_idx is True
            if r_obj._options['dif3d_idx']:
                adj_id = [r_obj.assemblies[adji - 1].dif3d_id
                          if adji > 0 else -1
                          for adji in r_obj.core.asm_adj[a.id]]
            else:
                adj_id = [r_obj.assemblies[adji - 1].id
                          if adji > 0 else -1
                          for adji in r_obj.core.asm_adj[a.id]]

            # Create the row
            row = []
            for col in range(6):
                entry = self._ffmt.format(eps[col])
                if adj_id[col] < 0:
                    entry += f' ({_OMIT})'
                else:
                    entry += f' ({(adj_id[col] + 1):03d})'
                row.append(entry)
            self.add_row(_fmt_idx(asm_id[i]), row)

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
    Power - Total assembly power (MW)
    Bulk outlet - Mixed-mean coolant temp. at the assembly outlet (˚C)
    Peak outlet - Maximum coolant subchannel temp. at the assembly outlet (˚C)
    Peak total - Axial-maximum coolant subchannel temp. in the assembly (˚C)
    Height - Axial height at which "Peak total" temp. occurs (cm)
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

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(r_obj)

        # Get averages from reactor_obj; read maximum CSV file
        # dat_max = np.loadtxt(
        #     os.path.join(reactor_obj.path, 'temp_maximum.csv'),
        #     delimiter=',')

        for i in range(len(assemblies)):
            a = assemblies[i]

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

            data = [_fmt_pos(asm_loc[i]),
                    '{:.5E}'.format(a.total_power),
                    '{:.5E}'.format(mfr_conv(a.flow_rate)),
                    self._ffmt2.format(tc_avg),
                    self._ffmt2.format(tc_max_out),
                    self._ffmt2.format(tc_max_tot),
                    tc_max_ht]
            self.add_row(_fmt_idx(asm_id[i]), data)


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
    Power - Total assembly power (MW)
    Average temp. - Duct mid-wall temperature per face (˚C) at core outlet
    Peak temp. - Axial peak duct mid-wall temperature (˚C)
    Peak Ht. - Axial position at which peak duct MW temperature occurs (cm)
    Peak Face - Hex face in which peak duct MW temperature occurs
           If corner, adjacent two face IDs are given

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

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(r_obj)

        # for i in range(len(assemblies)):
        #     a = assemblies[i]
        #     face_temps = self.temp_conv(self._get_avg_duct_face_temp(a))
        #     try:
        #         td_max_tot, td_max_ht = a._peak['duct']
        #     except TypeError:
        #         td_max_tot = a._peak['duct']
        #         td_max_ht = _OMIT
        #     else:  # only format height as float if you get one
        #         td_max_ht = self._ffmt2.format((len_conv(td_max_ht)))
        #     td_max_tot = self.temp_conv(td_max_tot)
        #     data = [_fmt_pos(asm_loc[i])]
        #     data += [self._ffmt2.format(td) for td in face_temps]
        #     data += [self._ffmt2.format(td_max_tot), td_max_ht]
        #     self.add_row(_fmt_idx(asm_id[i]), data)
        for i in range(len(assemblies)):
            a = assemblies[i]
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
                data = [_fmt_pos(asm_loc[i]), str(d + 1)]
                data += [self._ffmt2.format(td) for td in face_temps[d]]
                data += [self._ffmt2.format(td_max_tot), td_max_ht]
                self.add_row(_fmt_idx(asm_id[i]), data)

    # @staticmethod
    # def _get_avg_duct_face_temp(asm):
    #     """Average the duct hex face temperatures"""
    #     # temps = np.zeros(6)
    #     nd = asm.region[-1].temp['duct_mw'].shape[1]
    #     if nd == 6:
    #         temps = np.average(
    #             [asm.region[-1].temp['duct_mw'][0],
    #              np.roll(asm.region[-1].temp['duct_mw'][0], 1)
    #              ],
    #             axis=0)
    #     else:
    #         ndps = int(nd / 6)
    #         tmp = asm.region[-1].temp['duct_mw'][0].copy()
    #         tmp.shape = (6, ndps)
    #         tmp2 = np.zeros((6, ndps + 1))
    #         tmp2[:, 1:] = tmp
    #         tmp2[:, 0] = np.roll(tmp[:, -1], 1)
    #         temps = np.average(tmp2, axis=1)
    #     return temps

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
    """Create table

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    make : Generate table

    """

    col_id = {'clad': 6, 'fuel': -1}
    col_id = {'clad': {'od': 5, 'mw': 6, 'id': 7},
              'fuel': {'od': 8, 'cl': 9}}

    def __init__(self, col_width=9, col0_width=4, sep=' ', ffmt2=2):
        """Instantiate flow parameters output table"""
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}e' + '}'
        self._ffmt2 = '{' + f':.{ffmt2}f' + '}'

        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 10, col_width, col0_width, sep)

    def make(self, reactor_obj, component='clad', region='mw'):
        """Create the table

        Parameters
        ----------
        reactor_obj : DASSH Reactor object
            Contains the assembly data to print
        component : str {'clad', 'fuel'}
            Report clad/fuel temperatures at height of peak
            temperature for this component
            - 'clad': Clad midwall
            - 'fuel': Fuel centerline
        region : str
            If component = 'clad': {'od', 'mw', 'id'}
            If component = 'fuel': {'od', 'cl'}

        """
        # Override the title
        self.title = 'PEAK {0} {1} TEMPERATURES'.format(
            component.upper(), region.upper()) + "\n"

        # Erase any existing data in the table
        self.clear()
        # Process units from input_obj: temp, length
        temp_unit = reactor_obj.units['temperature']
        fmttd_temp_unit = _formatted_temp_units[temp_unit]
        self.temp_conv = self._get_temp_conv(temp_unit)
        len_unit = reactor_obj.units['length']
        self.len_conv = self._get_len_conv(len_unit)

        # Custom format the top row
        row = ' ' * (self.col0_width)
        row += ' ' * self.col_width * 2
        row += self.divider * 3
        fmtter = '{:>' + f'{self.col_width}.{self.col_width}' + 's}'
        row += fmtter.format('Height')
        row += self.divider
        row += fmtter.format('Pin power')
        row += self.divider
        row += f'  Nominal Radial Temperatures ({fmttd_temp_unit}) --> '
        self._table += row + '\n'
        # self.add_row('', ['', '',
        #                   'Height',
        #                   'Pin power',
        #                   'Nominal',
        #                   'Temperatures',
        #                   f'({fmttd_temp_unit}) --> ',
        #                   '', '', ''])
        self.add_row('Asm.', ['Loc.',
                              'Pin',
                              f'({len_unit})',
                              f'(W/{len_unit})',
                              'Coolant',
                              'Clad OD',
                              'Clad MW',
                              'Clad ID',
                              'Fuel OD',
                              'Fuel CL'])
        self.add_horizontal_line()

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(reactor_obj)

        # Load pin temps; if none, skip table
        # filepath = os.path.join(reactor_obj.path, 'temp_pin.csv')
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     dat = np.loadtxt(filepath, delimiter=',')
        # if dat.size == 0:
        #     self.clear()
        #     return
        keys = {'clad': {'od': 'clad_od',
                         'mw': 'clad_mw',
                         'id': 'clad_id'},
                'fuel': {'od': 'fuel_od',
                         'cl': 'fuel_cl'}}
        # Get peak temperatures @ height of peak component temp
        tab = []
        for i in range(len(assemblies)):
            row = [_fmt_idx(asm_id[i]), _fmt_pos(asm_loc[i])]
            if 'pin' in assemblies[i]._peak.keys():
                poop = keys[component][region]
                row += self._get_temps(
                    assemblies[i],
                    assemblies[i]._peak['pin'][poop][2])
                tab.append(row)
            else:
                continue

        # Write the data to the table
        for row in tab:
            self.add_row(row[0], row[1:])

    def _get_temps(self, asm_obj, data):
        """Get cladding/fuel temperatures at the requested height
        for the requested pin"""
        # Get pin (data[3]) and height (z = data[2])
        temps = [str(int(data[3])),
                 self._ffmt2.format(self.len_conv(data[2]))]

        # Get power
        plin = asm_obj.power.get_power(data[2])
        plin = plin['pins'][int(data[3])] / self.len_conv(1)
        temps.append(self._ffmt2.format(plin))

        # Add temperatures
        temps += [self._ffmt2.format(self.temp_conv(x))
                  for x in data[4:]]

        return temps


########################################################################


class AvgTempTable(LoggedClass, DASSH_Table):
    """Create table

    Parameters
    ----------

    Attributes
    ----------

    Methods
    -------
    make : Generate table

    """
    def __init__(self, col_width=15, col0_width=6, sep=' '):
        """Instantiate flow parameters output table"""
        # Decimal places for rounding, where necessary
        self.dp = col_width - 6
        # Float formatting option
        self._ffmt = '{' + f':.{self.dp}e' + '}'
        self._ffmt2 = '{' + f':.{2}f' + '}'

        # Inherit from DASSH_Table
        DASSH_Table.__init__(self, 6, col_width, col0_width, sep)

    def make(self, reactor_obj):
        """Create the table

        Parameters
        ----------
        reactor_obj : DASSH Reactor object
            Contains the assembly data to print
        component : str {'clad', 'fuel'}
            Report clad/fuel temperatures at height of peak
            temperature for this component
            - 'clad': Clad midwall
            - 'fuel': Fuel centerline

        """
        # Erase any existing data in the table
        self.clear()
        # Process units from input_obj: temp, length
        temp_unit = reactor_obj.units['temperature']
        self.temp_conv = self._get_temp_conv(temp_unit)
        self.add_row('Asm ID', ['Name',
                                'Loc.',
                                f'Avg Coolant',
                                f'Peak Coolant',
                                f'Avg Duct MW',
                                f'Peak Duct MW'])
        self.add_horizontal_line()

        # Pregenerate a bunch of temporary objects: this is necessary
        # in order to get proper indexing of assemblies, which varies
        # depending on whether the user asked for DIF3D indexing or
        # DASSH indexing.
        assemblies, asm_id, asm_loc = _sorted_assemblies_attr(reactor_obj)

        # Get averages from reactor_obj; read maximum CSV file
        # dat_max = np.loadtxt(
        #     os.path.join(reactor_obj.path, 'temp_maximum.csv'),
        #     delimiter=',')

        for i in range(len(assemblies)):
            a = assemblies[i]

            # Here again, the first column in the dump file is the
            # DASSH ID, so we need to use the assembly.ID attribute
            # to locate the rows we want to search

            # Coolant temperatures
            tc_avg = self.temp_conv(a.avg_coolant_temp)
            tc_max = self.temp_conv(a._peak['cool'])
            # tc_pk = self.temp_conv(
            #     np.max(dat_max[dat_max[:, 0] == a.id][:, 4]))

            # Duct temperatures
            td_avg = self.temp_conv(a.avg_duct_mw_temp[-1])
            td_max = self.temp_conv(a._peak['duct'])
            # td_pk = self.temp_conv(
            #     np.max(dat_max[dat_max[:, 0] == a.id][:, 5]))
            self.add_row(str(asm_id[i]), [a.name,
                                          str(asm_loc[i]),
                                          self._ffmt2.format(tc_avg),
                                          self._ffmt2.format(tc_max),
                                          self._ffmt2.format(td_avg),
                                          self._ffmt2.format(td_max)])


########################################################################
