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
date: 2022-07-14
author: Milos Atz
This module defines the object that reads the DASSH input file
into Python data structures.
"""
########################################################################
import os
import re
import copy
import logging
import numpy as np
import configobj
from configobj import ConfigObj, flatten_errors
from validate import Validator
import matplotlib as mpl
import matplotlib.pyplot as plt
import dassh
from dassh.logged_class import LoggedClass
from dassh import utils
from dassh import py4c


_ROOT = os.path.dirname(os.path.abspath(__file__))
_ARC = ['pmatrx', 'geodst', 'ndxsrf', 'znatdn', 'labels', 'nhflux', 'ghflux']
module_logger = logging.getLogger('dassh.input')


########################################################################


class DASSH_Assignment(object):
    """Helper class to process the "Assignment" section of the DASSH
    input file; inherited by DASSH_Input"""

    def __init__(self):
        """No instance methods or attributes here"""
        pass

    def parse_assignment_section(self, assn_str):
        """Parse the Assignment section stripped from the input"""
        assn_dict = self._split_subsections(assn_str)
        assn_parsed = self._split_positions(assn_dict)
        # Make sure this section is not empty
        if len(assn_parsed) == 0:
            self.log('error', 'Assignment section is empty')

        n_ring = max(x[1] for x in assn_parsed)
        n_asm = 3 * (n_ring - 1) * n_ring + 1
        dat = {'ByPosition': [[] for i in range(n_asm)]}
        # Get assembly index (python index)
        for l in assn_parsed:
            if l[1] == 1:
                asm_start = 0
                asm_end = 1
            else:
                asm_start = 3 * (l[1] - 1 - 1) * (l[1] - 1) + l[2]
                asm_end = asm_start + l[3] - l[2] + 1
            counter = 0
            for asm in range(asm_start, asm_end):
                ring = l[1] - 1  # python indexing
                pos = l[2] + counter - 1  # python indexing
                # Check that ring/position indicator is valid
                msg0 = 'Error in ring {:d} position {:d} assignment: '
                if (ring == 0 and pos != 0):
                    msg = msg0 + 'if ring is 0, position must be 0'
                    self.log('error', msg.format(ring, pos))
                if (ring > 0 and pos > 6 * ring - 1):
                    msg = msg0 + ('position is greater than maximum '
                                  'possible on ring')
                    self.log('error', msg.format(ring, pos))
                dat['ByPosition'][asm] = copy.deepcopy(
                    [l[0], (ring, pos, asm), l[4]])
                counter += 1
        return dat

    def strip_assignment_section(self, infile):
        """Read and remove the Assignment section from the input"""
        with open(infile, 'r') as f:
            infile = f.read()
        tag0 = infile.find('[Assignment]')
        tag1 = infile.find(']', tag0)
        while True:
            tag2 = infile.find('[', tag1)
            if tag2 == -1:
                assignment = infile[tag0:]
                infile = infile[:tag0]
                break
            elif infile[tag2 + 1] == '[' or infile[tag2 - 1] == '[':
                tag1 = infile.find(']', tag2)
            elif tag1 == tag2:
                self.log('error', 'Cannot properly parse Assignmnet '
                                  'input; input file may be formatted '
                                  'incorrectly')
            else:
                assignment = infile[tag0:tag2]
                infile = infile[:tag0] + infile[tag2:]
                break
        return infile, assignment

    @staticmethod
    def _split_subsections(assn_in):
        """Split the input string into sections"""
        # Currently only one section "ByPosition" allowed
        assignment = {'ByPosition': None,
                      'ByRegion': None,
                      'ByRegionList': None,
                      'ByArea': None}
        for key in assignment.keys():
            tag1 = assn_in.find(key)
            tag2 = assn_in.find('[', tag1)
            while assn_in[tag1 - 1:tag1] == '[':
                tag1 -= 1
            if tag2 == -1:
                assignment[key] = assn_in[tag1:]
            else:
                assignment[key] = assn_in[tag1:tag2]
        return assignment

    @staticmethod
    def _split_positions(assn_dict):
        """Parse the Assignment section string and add to data dict"""
        # split up the lines to process the values inside
        tmp = assn_dict['ByPosition'].splitlines()
        parsed = []
        for l in range(1, len(tmp)):
            # Ignore comments
            if '#' in tmp[l]:
                comment_tag = tmp[l].find('#')
                tmp[l] = tmp[l][:comment_tag]

            # Split the line at the commas
            line = tmp[l].split(',')
            line = [x for x in re.split('=|, | ', tmp[l]) if x != '']
            if line != []:
                # Ring, start position, end position
                line[1] = int(line[1])
                line[2] = int(line[2])
                line[3] = int(line[3])
                # Keyword arguments: pattern is key=val, key=val, ...
                kwargs = {}
                for ki in range(4, len(line), 2):
                    kwargs[line[ki].lower()] = float(line[ki + 1])
                del line[4:]
                line.append(kwargs)
                parsed.append(line)
        return parsed


########################################################################


class DASSHPlot_Input(LoggedClass):
    """Process the DASSHPlot input"""

    def __init__(self, infile, reactor=None):
        """No instance methods here"""
        # Instantiate logger
        LoggedClass.__init__(self, 4, 'dassh.read_input.DASSHPlot_Input')
        # DASSH_Assignment.__init__(self)
        if reactor is not None:
            self.load_from_reactor(infile, reactor)
        else:
            self.load_from_dassh_input(infile)

    def _load(self, infile):
        """Read and check DASSHPlot input data against the template"""
        # Read input with ConfigObj and check it against the template
        tmp_path = os.path.join(_ROOT, 'dasshplot_input_template.txt')
        inp = _configobj_load(self, infile, tmp_path)
        _configobj_check_extra_kw(self, inp, only='Plot')
        return inp

    def load_from_reactor(self, infile, reactor):
        """Read and check DASSHPlot input data"""
        with open(infile, 'r') as f:
            infile = f.read()

        inp = self._load(infile)

        # Delete all other keys; don't need 'em
        for k in inp.keys():
            if k == 'Plot':
                continue
            elif k == 'Setup':
                for kk in inp['Setup'].keys():
                    if kk != 'Units':
                        del inp['Setup'][kk]
            else:
                del inp[k]

        # Check user-requested units
        # If no specified length unit, assume the length unit is what's
        # used in the DASSH input file and saved in the Reactor obj
        len_unit = inp['Setup']['Units']['length']
        if len_unit is None:
            len_unit = reactor.units['length']
        try:
            lconv = utils.get_length_conversion('m', len_unit)
        except ValueError:  # Assertion error raised if len_unit == 'm'
            lconv = lambda l: l  # No conversion needed, assign identity fxn

        # Check inputs for correctness and save
        self._plt_check_params = {
            'core_len': lconv(reactor.core_length),
            'n_asm': len(reactor.assemblies),
            'len_unit': reactor.units['length']}
        self.data = {}
        self.data['Plot'] = self.check_dasshplot_input(inp['Plot'])
        self.data['Setup'] = inp['Setup']

    def load_from_dassh_input(self, infile):
        """Check the DASSHPlot input against the full input file"""
        inp = self._load(infile)
        # Check inputs for correctness and save
        core_len = self.data['Core']['length']
        n_asm = len(self.data['Assignment']['ByPosition'])
        len_unit = self.data['Setup']['Units']['length']
        if len_unit is None:
            len_unit = 'm'
        self._plt_check_params = {'core_len': core_len,
                                  'n_asm': n_asm,
                                  'len_unit': len_unit}
        self.data['Plot'] = self.check_dasshplot_input(inp['Plot'])

    def check_dasshplot_input(self, all_plot_dict):
        for p in all_plot_dict.keys():
            f = f"check_{all_plot_dict[p]['type']}_input"
            all_plot_dict[p] = getattr(self, f)(all_plot_dict[p], p)
        return all_plot_dict

    def check_CorePinPlot_input(self, cpp_dict, title):
        """Check that CorePinPlot input has appropriate values"""
        cpp_dict = self._check_plot_zpts(cpp_dict, title)
        self._check_plot_pin_values(cpp_dict, title)
        self._check_plot_colorbar_bnds(cpp_dict, title)
        self._check_plot_cmap(cpp_dict, title)
        self._check_plot_units(cpp_dict, title)
        return cpp_dict

    def check_CoreSubchannelPlot_input(self, cscp_dict, title):
        """Check that CoreSubchannelPlot input has appropriate values"""
        cscp_dict = self._check_plot_zpts(cscp_dict, title)
        self._check_plot_colorbar_bnds(cscp_dict, title)
        self._check_plot_cmap(cscp_dict, title)
        self._check_plot_units(cscp_dict, title)
        return cscp_dict

    def check_SubchannelPlot_input(self, scp_dict, title):
        """Check that SubchannelPlot input has appropriate values

        Notes
        -----
        Same as for CoreSubchannelPlot, with additional check
        for assembly ID

        """
        scp_dict = self.check_CoreSubchannelPlot_input(scp_dict, title)
        scp_dict = self._check_plot_asm_id(scp_dict, title)
        return scp_dict

    def check_PinPlot_input(self, pp_dict, title):
        """Check that PinPlot input has appropriate values

        Notes
        -----
        Same as for CorePinPlot, with additional check
        for assembly ID

        """
        pp_dict = self.check_SubchannelPlot_input(pp_dict, title)
        pp_dict = self._check_plot_asm_id(pp_dict, title)
        return pp_dict

    def check_CoreHexPlot_input(self, chp_dict, title):
        """Check input parameters for the CoreHexPlot"""
        # CoreHexPlot does not necessarily require z-input
        # if ('avg' in chp_dict['value']
        #         or not all([zi is None for zi in chp_dict['z'])):
        if chp_dict['z'] is not None:
            chp_dict = self._check_plot_zpts(chp_dict, title)

        # Then check other stuff
        self._check_plot_core_hex_value(chp_dict, title)
        self._check_plot_colorbar_bnds(chp_dict, title)
        self._check_plot_cmap(chp_dict, title)
        self._check_plot_units(chp_dict, title)
        return chp_dict

    def _check_plot_zpts(self, pdict, title):
        """General check for axial height input for plotting"""
        # Check z values: must be specified, must be float
        if pdict['z'] is None:
            msg = ('At least one "z" value is required in '
                   f'"Plot" sub-block "{title}"')
            self.log('error', msg)
        else:
            try:
                pdict['z'] = [float(zi) for zi in pdict['z']]
            except ValueError:
                msg = ('Entries for "z" input in "Plot" sub-block '
                       f'"{title}" must be floats')
                self.log('error', msg)

        # Minimum z must be greater than or equal to zero; maximum
        # must be less than or equal to core height
        core_len = self._plt_check_params['core_len']
        len_unit = self._plt_check_params['len_unit']
        if min(pdict['z']) < 0:
            msg = (f'Minimum "z" input in "Plot" sub-block "{title}" '
                   'must be greater than or equal to 0.0')
            self.log('error', msg)
        if max(pdict['z']) > core_len:
            msg = (f'Maximum "z" input in "Plot" sub-block "{title}" '
                   'must be less than or equal to specified core '
                   f'height: {core_len}')
            if len_unit is not None:
                msg += '\n'
                msg += f'(Hint: DASSHPlot expected units of {len_unit})'
            self.log('error', msg)
        return pdict

    def _check_plot_cmap(self, pdict, title):
        user_cmap = pdict['cmap']
        msg = (f'Unavailable cmap "{user_cmap}" requested '
               f'in "Plot" sub-block "{title}"')
        if user_cmap not in dir(mpl.cm):
            self.log('error', msg)

    def _check_plot_units(self, pdict, title):
        user_unit = pdict['units']
        msg = (f'Unsupported temperature unit "{user_unit}" in '
               f'"Plot" sub-block "{title}"; using default (K)')
        okay = ['c', 'degc', 'celsius', 'kelvin',
                'k', 'f', 'degf', 'fahrenheit']
        if user_unit is not None and user_unit.lower() not in okay:
            self.log('error', msg)

    def _check_plot_colorbar_bnds(self, pdict, title):
        """colorbar boundaries must be arranged in increasing order"""
        # Three relationships to check:
        # 1. lower bound to midpoint
        # 2. lower bound to upper bound
        # 3. midpoint to upper bound
        lbnd = pdict['cbar_lbnd']
        mpnt = pdict['cbar_mpnt']
        ubnd = pdict['cbar_ubnd']

        if lbnd is not None:
            msg = ('Colorbar {:s} in "Plot" sub-block "{:s}" '
                   'must be greater than lower bound')
            if mpnt is not None and mpnt <= lbnd:
                self.log('error', msg.format('midpoint', title))
            if ubnd is not None and ubnd <= lbnd:
                self.log('error', msg.format('upper bound', title))

        if mpnt is not None and ubnd is not None and ubnd <= mpnt:
            msg = ('Colorbar upper bound in "Plot" sub-block "{:s}" '
                   'must be greater than midpoint')
            self.log('error', msg.format(title))

    def _check_plot_asm_id(self, pdict, title):
        """General check for assembly ID input for plotting"""
        if pdict['assembly_id'] is None:
            msg = ('At least one "assembly_id" value is required in '
                   f'"Plot" sub-block "{title}"')
            self.log('error', msg)

        else:
            try:
                a = [int(ai) for ai in pdict['assembly_id']]
            except ValueError:
                msg = ('Entries for "assembly_id" input in "Plot" '
                       f'sub-block "{title}" must be integers')
                self.log('error', msg)
            else:
                pdict['assembly_id'] = a
        # Minimum a must be greater than or equal to one; maximum
        # must be less than or equal to total number of assemblies
        if min(pdict['assembly_id']) < 1:
            msg = ('Minimum "assembly_id" input in "Plot" sub-block '
                   f'"{title}" must be greater than or equal to 1')
            self.log('error', msg)
        n_asm = self._plt_check_params['n_asm']
        if max(pdict['assembly_id']) > n_asm:
            msg = ('Maximum "assembly_id" input in "Plot" sub-block '
                   f'"{title}" must be less than or equal to given'
                   f'number of assemblies: {n_asm}')
            self.log('error', msg)
        return pdict

    def _check_plot_pin_values(self, pdict, title):
        """Check that user requests acceptable pin plot values"""
        _acceptable = ['clad_od', 'clad_mw', 'clad_id',
                       'fuel_od', 'fuel_cl']
        msg = ('Unrecognized pin value "{:s}" requested for "Plot" '
               'sub-block "{:s}"; accetable values are: "clad_od", '
               '"clad_mw", "clad_id", "fuel_od", "fuel_cl"')
        for v in pdict['value']:
            if v not in _acceptable:
                self.log('error', msg.format(v, title))

    def _check_plot_core_hex_value(self, pdict, title):
        """Check that acceptable values are given for core hex plot"""
        _acceptable = ['total_power', 'max_coolant_temp',
                       'max_duct_mw_temp', 'max_clad_mw_temp',
                       'max_fuel_cl_temp', 'avg_coolant_temp',
                       'avg_duct_mw_temp', 'avg_clad_mw_temp',
                       'avg_fuel_cl_temp']
        for v in pdict['value']:
            if v not in _acceptable:
                msg = (f'"value" input {v} in "Plot" sub-block "{title}" '
                       'must be one of the available options:')
                msg += '\n- '
                msg += '\n- '.join(_acceptable)
                self.log('error', msg)


########################################################################


class DASSH_Input(DASSHPlot_Input, DASSH_Assignment, LoggedClass):
    """Object for processing DASSH input files.

    Parameters
    ----------
    infile : str
        Path to DASSH input file
    empty4c : bool
        Testing flag indicating that 4C files are empty and should
        not be read or checked

    Notes
    -----
    DASSH input files contain 4 main sections and 2 optional sections:
    1. File paths to ARC files: [ARC]
    2. Core parameters: [Core]
    3. Assembly details: [Assembly]
    4. Assembly assignments: [Assignment]
    5. (optional) Problem setup: [Setup]
    6. (optional) Custom material properties: [Materials]

    This object reads these inputs using the ConfigObj package,
    performs additional checks on those inputs beyond what is
    built into ConfigObj, and returns the data to be used elsewhere
    in the software.

    """

    def __init__(self, infile, empty4c=False):
        """Read and check the input data"""
        LoggedClass.__init__(self, 4, 'dassh.read_input.DASSH_Input')
        DASSH_Assignment.__init__(self)
        self.path = os.path.split(infile)[0]
        self.tmp_path = self.get_template()  # path to input template

        # Check input file text that all required sections are present
        self.check_inputfile_sections(infile)

        # Remove the "Assignment" section from the input file before
        # processing into dictionary with ConfigObj
        str_infile, str_assn = self.strip_assignment_section(infile)

        # Read all sections except Assignment into ConfigObj dict,
        # stored in instances as self.data; make sure that the main
        # required sections are present
        self.data = _configobj_load(self, str_infile, self.tmp_path)
        self.check_configobj_sections()
        _configobj_check_extra_kw(self, self.data, skip='Plot')

        # Process Assignment text input; add to self.data
        self.data['Assignment'] = \
            self.parse_assignment_section(str_assn)

        # Now that self.data is complete, run consistency
        # checks on inputs in each section

        # Power input - figure out what section it's coming from and
        # make sure it exists
        self._cccc_power, self._user_power = self.determine_power_input()
        if not self._cccc_power:
            empty4c = True
        if self._cccc_power:
            if not empty4c:
                self.check_4c_input()
            self.check_ARC_fuel_specifications()
        if self._user_power:
            self.check_user_power()
        # Assembly
        self.check_unrodded_regions()
        self.check_pin()
        self.check_duct()
        self.check_dummy_pin()
        self.check_fuel_model()
        self.check_pin_model()
        self.check_correlations()
        # Assignment
        self.check_assignment_assembly_agreement()
        self.check_assignment_boundary_conditions()
        self.check_assignment_against_geodst(empty4c)
        self.convert_assn_deltaT_to_outletT()
        # Core
        self.check_core_specifications()
        self.check_htc_params()
        # Setup (optional)
        self.check_units()
        self.check_axial_plane_req()
        self.check_dump()
        self.check_setup_assembly_tables()
        # Materials (optional)
        self.check_user_spec_materials()
        self.load_materials()
        # Check orificing - set data['Orificing'] to False if no input
        self.check_orificing()

        # If the user requests plots be generated, can read them in
        # now and do cross checks against the completed input file
        # data. Need to do this last because checked with a separate
        # ConfigObj template
        DASSHPlot_Input.__init__(self, str_infile)

        # Coerce non-list input to list with the proper number of tpts
        self.timepoints = self.get_timepoints(infile)
        if not isinstance(self.data['Power']['user_power'], list):
            self.data['Power']['user_power'] = \
                [self.data['Power']['user_power']] * self.timepoints
        for f in _ARC:
            if not isinstance(self.data['Power']['ARC'][f], list):
                self.data['Power']['ARC'][f] = \
                    [self.data['Power']['ARC'][f]] * self.timepoints

        # Check user request for parallel calculation - need to do after
        # setting up self.timepoints attribute
        self.check_parallel()

        # Convert units to DASSH defaults
        self.convert_units()

        # Clean up any user-specified inputs against GEODST specs
        if not empty4c:
            self.check_geodst()

    def clone(self):
        """Create a clone of the DASSH_Input object"""
        clone = copy.copy(self)
        clone.data = copy.deepcopy(self.data)
        clone.timepoints = copy.deepcopy(self.timepoints)
        clone.path = copy.deepcopy(self.path)
        return clone

    def get_timepoints(self, infile):
        with open(infile, 'r') as f:
            txt = f.read().splitlines()
        if self._cccc_power:
            tpts_cccc = self._get_cccc_timepoints(txt)
        else:
            tpts_cccc = None
        if self._user_power:
            tpts_user = self._get_user_timepoints(txt)
        else:
            tpts_user = None
        # Now figure out which time points are relevant and assess
        if self._cccc_power and not self._user_power:
            return tpts_cccc
        elif self._user_power and not self._cccc_power:
            return tpts_user
        elif self._user_power and self._cccc_power:
            if tpts_cccc == tpts_user:
                return tpts_cccc
            else:
                msg = ('If both [ARC] and [Power] input sections are '
                       'defined, they need to have the same number '
                       'of entries')
                self.log(msg)
        else:  # neither specified
            raise ValueError('should have caught this elswhere')

    def _get_cccc_timepoints(self, txt):
        """Determine whether input contains multiple timesteps."""
        # Use GEODST as time point ref e.g. txt = 'geodst_input = ... '
        tpts = []
        for cccc_inp in ['pmatrx', 'geodst', 'ndxsrf', 'znatdn',
                         'labels', 'nhflux', 'ghflux']:
            # Find the line in the input file
            idx = [idx for idx, s in enumerate(txt) if cccc_inp in s][0]
            # Parse the line to see how many inputs there are
            subtxt = txt[idx].split('#')[0]  # eliminate comments
            subtxt = subtxt.split('=')[1]  # get input value
            tpts.append(len(subtxt.split(',')))
        if not all(x == tpts[0] for x in tpts):
            msg = ('Inconsistent number of 4C files given; the number '
                   'of PMATRX / GEODST / NDXSRF / ZNATDN / LABELS / '
                   'NHFLUX / GHFLUX files must be equal')
            self.log('error', msg)
        else:
            return tpts[0]

    def _get_user_timepoints(self, txt):
        """Determine whether input contains multiple time points."""
        idx = [idx for idx, s in enumerate(txt) if 'user_power' in s][0]
        tpts = len(txt[idx].split('=')[1].split(','))
        return tpts

    def get_template(self):
        """Get template config file for single- or multi-time input."""
        tmp_path = os.path.join(_ROOT, 'input_template.txt')
        return tmp_path

    ####################################################################
    # INPUT FILE VALIDATION
    # After ConfigObj reads input file into dictionaries, these methods
    # check that the values in each of the input blocks are physically
    # meaningful and in agreement with the rest of the input
    ####################################################################

    def check_inputfile_sections(self, inputfile):
        """Check that mandatory sections are present in the input.
        Note: RegionList is optional."""
        # Read from the input file
        with open(inputfile, 'r') as f:
            txt = f.read()
        txtlines = txt.splitlines()
        for sec in ['Assembly', 'Core', 'Assignment', 'Power']:
            tmp = [i for i, s in enumerate(txtlines) if f'[{sec}]' in s]
            if not len(tmp) == 1:  # either no entries or more than one
                # raise OSError('Missing/incorrect input section: ' + sec)
                self.log('error', f'Missing/incorrect section: {sec}')

    def check_configobj_sections(self):
        """Check the data structure contains the mandatory sections"""
        for sec in ['Core', 'Assembly', 'Power']:
            if sec not in self.data.keys():
                # raise OSError('Missing/incorrect input section: ' + sec)
                self.log('error', f'Missing/incorrect section: {sec}')
            else:
                if not self.data[sec]:  # empty dictionary entry
                    # raise OSError('Empty input section: ' + sec)
                    self.log('error', f'Empty section: {sec}')

    def determine_power_input(self):
        """Confirm input has appropriate power specification"""
        incl = [False, False]
        # If any 4C file in the ARC section is defined, count it
        if any([self.data['Power']['ARC'].get(k) for k in _ARC]):
            incl[0] = True
        # # If user power is defined, count it
        if self.data['Power'].get('user_power') is not None:
            incl[1] = True
        return incl

    def check_4c_input(self):
        """Check existence and consistency of all 4C files."""
        for ft in _ARC:  # Loop over file type
            # If None, skip; already screened for the failure if all
            # are None and no power is defined.
            if self.data['Power']['ARC'][ft] is None:
                self.log('error', f'Path not specified for binary file {ft}')
            # Entry is a list
            elif isinstance(self.data['Power']['ARC'][ft], list):
                # Note: type checking okay here because I'm expecting
                # values from ConfigObj that are either list or str
                for i in range(len(self.data['Power']['ARC'][ft])):
                    fp = self.data['Power']['ARC'][ft][i]
                    if not os.path.exists(
                        os.path.abspath(
                            os.path.join(self.path, fp))):
                        self.log('error', f'Path {fp} does not exist')
                    # Set as absolute path
                    self.data['Power']['ARC'][ft][i] = \
                        os.path.abspath(os.path.join(self.path, fp))

            else:  # Single value entry, only check file existence
                fp = self.data['Power']['ARC'][ft]
                if not os.path.exists(
                    os.path.abspath(
                        os.path.join(self.path, fp))):
                    self.log('error', f'Path {fp} does not exist')
                # Set as absolute path
                self.data['Power']['ARC'][ft] = \
                    os.path.abspath(os.path.join(self.path, fp))

    def check_ARC_fuel_specifications(self):
        """Check fuel material and alloy specification for VARPOW"""
        # Check fuel material specification:
        msg = ('\"fuel_material\" input must be one of '
               '{"metal", "oxide", "nitride"}')
        if self.data['Power']['ARC']['fuel_material'] is not None:
            if (self.data['Power']['ARC']['fuel_material'].lower() not in
                    ['metal', 'oxide', 'nitride']):
                self.log('error', msg)
        else:
            self.log('error', msg)
        # Check fuel alloy specification
        if self.data['Power']['ARC']['fuel_alloy'] is not None:
            if (self.data['Power']['ARC']['fuel_alloy'].lower() not in
                    ['zr', 'zirconium', 'al', 'aluminum']):
                self.log('error', ('\"fuel_alloy\" input must be '
                                   'either "zr" or "al"'))

    def check_user_power(self):
        """Confirm user-spec power dist input file exists, if given"""
        if not self._user_power:
            return
        for i in range(len(self.data['Power']['user_power'])):
            abs_fp = os.path.abspath(
                os.path.join(self.path,
                             self.data['Power']['user_power'][i]))
            if not os.path.exists(abs_fp):
                self.log('error', f'Path {abs_fp} does not exist')
            # Set as absolute path
            self.data['Power']['user_power'][i] = abs_fp

    def check_unrodded_regions(self):
        """Check that values describing porous media axial regions
        are nonzero."""
        for asm in self.data['Assembly']:
            pre = f'Asm: \"{asm}\"; '  # indicate asm for error msg

            # Check that user-input MFR ratio is acceptable if entire
            # assembly uses low-fidelity model
            if self.data['Assembly'][asm]['use_low_fidelity_model']:
                self.data['Assembly'][asm]['convection_factor'] = \
                    self._check_ur_convection_factor(
                        self.data['Assembly'][asm]['convection_factor'],
                        pre=pre)

            # Skip the rest if no specified axial regions
            if not any(self.data['Assembly'][asm]['AxialRegion']):
                self.data['Assembly'][asm]['AxialRegion'] = {
                    'rods': {'z_lo': 0.0,
                             'z_hi': self.data['Core']['length']}}
                continue
            else:
                for r in self.data['Assembly'][asm]['AxialRegion']:
                    for k in ['hydraulic_diameter', 'epsilon']:
                        self._check_nonnegative(
                            self.data['Assembly'][asm]['AxialRegion']
                                     [r][k], pre + k)

            # Do some checks on the region boundaries; can sort
            # them because they're supposed to agree like that
            # Check that z_hi > z_lo for each region
            z_lo = sorted([self.data['Assembly'][asm]
                                    ['AxialRegion'][r]['z_lo']
                           for r in self.data['Assembly'][asm]
                                             ['AxialRegion']])
            z_hi = sorted([self.data['Assembly'][asm]
                                    ['AxialRegion'][r]['z_hi']
                          for r in self.data['Assembly'][asm]
                                            ['AxialRegion']])
            for i in range(1, len(z_lo)):
                if z_lo[i] >= z_hi[i]:
                    msg = (f'{pre} detected an AxialRegion with '
                           'non-postive height; \"z_hi" - \"z_lo" '
                           'must be positive')
                    self.log('error', msg)

            # Confirm that there is only one rodded region
            bnds = [0.0, self.data['Core']['length']]
            rodded_regions = _find_rodded_regs(z_lo, z_hi, bnds)
            if not _check_reg_bnds(rodded_regions):
                self.log('error', (f'{pre} Problem detected with '
                                   'input specifications for '
                                   'unrodded region boundaries; '
                                   'please ensure that you have '
                                   'only one rodded region.'))
            # Add rodded region bounds to data dict
            bnds = _get_rodded_reg_bnds(
                z_lo, z_hi, rodded_regions, bnds)
            self.data['Assembly'][asm]['AxialRegion']['rods'] = \
                {'z_lo': bnds[0], 'z_hi': bnds[1]}

            # Check that user-input MFR ratio is acceptable for each
            # axial region input, if provided
            for r in self.data['Assembly'][asm]['AxialRegion']:
                if r == 'rods':
                    continue
                k = 'convection_factor'
                self.data['Assembly'][asm]['AxialRegion'][r][k] = \
                    self._check_ur_convection_factor(
                        self.data['Assembly'][asm]['AxialRegion'][r][k],
                        calc=False,
                        pre=pre + ' ' + r)

    def _check_ur_convection_factor(self, value, calc=True, pre=''):
        """Check the value of the interior MFR fraction, process
        accordingly, and raise errors if necessary"""
        do_not_understand = (f'{pre} do not understand '
                             '"convection_factor" input - must be '
                             '"calculate" (available if full-asm low-'
                             'fidelity model) or float greater than '
                             '0.0 and less than or equal to 1.0')
        if value is None:
            return 1.0
        if value == 'calculate':
            if calc:
                return 'calculate'
            else:
                self.log('error', f'{pre} "convection_factor" input '
                                  '"calculate" only available for '
                                  'full-asm low-fidelity model input')
        elif isinstance(value, (str, int, float)):
            try:
                value = float(value)
            except ValueError:
                self.log('error', do_not_understand)
            else:
                if not 0.0 < value <= 1.0:
                    msg = (f'{pre} "convection_factor" float input '
                           'must be greater than 0.0 and less than '
                           'or equal to 1.0')
                    self.log('error', msg)
                else:
                    return value
        else:
            self.log('error', do_not_understand)

    def check_pin(self):
        """Check that pin details are physically meaningful."""
        for asm in self.data['Assembly']:
            pre = f'Asm: \"{asm}\"; '  # indicate asm for error msg
            # Values must be nonzero
            for k in ['num_rings', 'pin_pitch', 'pin_diameter',
                      'clad_thickness']:
                self._check_nonzero(self.data['Assembly'][asm][k],
                                    pre + k)
            # Pin pitch must be greater than pin diameter
            msg = 'Pin pitch must be greater than pin diameter'
            if (self.data['Assembly'][asm]['pin_pitch']
                    < self.data['Assembly'][asm]['pin_diameter']):
                self.log('error', pre + msg)

            # Cladding thickness must be less than pin radius
            # (as if entire pin would be cladding)
            msg = 'Pin cladding thickness must be less than pin radius'
            if (self.data['Assembly'][asm]['pin_diameter'] / 2.0
                    < self.data['Assembly'][asm]['clad_thickness']):
                self.log('error', pre + msg)

            # Wire must fit into space between fuel pins
            msg = 'Wire diameter must be less than space between pins'
            if (self.data['Assembly'][asm]['wire_diameter']
                > (self.data['Assembly'][asm]['pin_pitch']
                   - self.data['Assembly'][asm]['pin_diameter'])):
                self.log('error', pre + msg)

            # Pins must fit inside duct
            msg1 = 'Pins do not fit inside duct; {:s} m too big.'
            msg2 = 'Rod bundle FTF calculated as:'
            msg3 = ('sqrt(3) * (N_ring - 1) * pin_pitch + pin_diameter '
                    '+ 2 * wire_diameter')
            if not self.data['Assembly'][asm]['use_low_fidelity_model']:
                dftf = min(self.data['Assembly'][asm]['duct_ftf'])
                n_ring = self.data['Assembly'][asm]['num_rings']
                pin_pitch = self.data['Assembly'][asm]['pin_pitch']
                pin_diam = self.data['Assembly'][asm]['pin_diameter']
                wire_diam = self.data['Assembly'][asm]['wire_diameter']
                clr = dftf - (np.sqrt(3) * (n_ring - 1) * pin_pitch
                              + pin_diam + 2 * wire_diam)
                if clr < 0.0:  # leave a little bit of wiggle room
                    clr = '{:0.6e}'.format(clr)
                    msg = '\n'.join([msg1.format(clr), msg2, msg3])
                    self.log('error', pre + msg)

    def check_duct(self):
        """Make sure duct details are physically meaningful; outer duct outer
        FTF must be the same for all assembly types"""
        outer_duct_oftf = []
        for a in self.data['Assembly']:
            pre = f'Asm: \"{a}\"; '  # indicate asm for error msg
            if len(self.data['Assembly'][a]['duct_ftf']) % 2 != 0:
                msg = ('Number of duct FTF values must be even; need '
                       'inner/outer FTF for each duct')
                self.log('error', pre + msg)

            # Check duct thicknesses
            for d in range(int(len(self.data['Assembly'][a]['duct_ftf']) / 2)):
                d1 = self.data['Assembly'][a]['duct_ftf'][2 * d]
                d2 = self.data['Assembly'][a]['duct_ftf'][2 * d + 1]
                # Values must be greater than zero
                msg = 'Duct FTF must be greater than zero'
                self._check_nonzero(d1, pre + msg)
                self._check_nonzero(d2, pre + msg)
                # Duct inner/outer flat-to-flat (FTF) distances cannot
                # be equal (which value is inner and which is outer is
                # inferred based on whichever is greater)
                msg = 'Duct outer FTF must be greater than inner FTF'
                if d1 == d2:
                    self.log('error', pre + msg)

            # Confirm duct FTF values are less than assembly pitch
            if any(ftf >= self.data['Core']['assembly_pitch']
                   for ftf in self.data['Assembly'][a]['duct_ftf']):
                msg = (f'Duct FTF values must be less than assembly '
                       'pitch specified in "Core" section: '
                       f'{self.data["Core"]["assembly_pitch"]}')
                self.log('error', pre + msg)

            # Add outer duct outer FTF to list
            outer_duct_oftf.append(
                np.round(self.data['Assembly'][a]['duct_ftf'][-1], 9)
            )

        # Check outer flat-to-flat agreement
        if not all(x == outer_duct_oftf[0] for x in outer_duct_oftf):
            self.log('error', 'DASSH requires that outer duct outer '
                              'flat-to-flat distances be equal for '
                              'all assemblies.')

    def check_bypass_pressure_drop_params(self):
        """Check bypass pressure drop parameters for each assembly
        with more than one duct"""
        for asm in self.data['Assembly']:
            pre = f'Asm: \"{asm}\"; '  # indicate asm for error msg
            if len(self.data['Assembly'][asm]['duct_ftf']) == 2:
                continue  # only one duct

    def check_dummy_pin(self):
        """Check that dummy pin specifications are physical.

        Notes
        -----
        In dummy_pin input, pins are indicated by their "pin number"
        position in the assembly

        """
        for asm in self.data['Assembly']:
            if self.data['Assembly'][asm]['dummy_pin'] is not None:
                pre = f'Asm: \"{asm}\"; '  # indicate asm for error msg
                # Dummy pin ID must be greater than or equal to 0
                if not all([pin >= 0 for pin in
                            self.data['Assembly'][asm]['dummy_pin']]):
                    self.log('error', (f'{pre} require nonnegative '
                                       'dummy pin ID'))
                # Dummy pin ID must be less than number of pins in asm
                n_ring = self.data['Assembly'][asm]['n_ring']
                n_pin = 3 * (n_ring - 1) * n_ring + 1
                if max(self.data['Assembly'][asm]['dummy_pin']) > n_pin:
                    self.log('error', (f'{pre} dummy pin ID cannot be '
                                       'greater than number of pins '
                                       'in assembly'))

    def check_fuel_model(self):
        """Make sure pin and fuel layer specs are physically meaningful
        and agree with other components of the Assembly input."""
        # All assemblies will have at least default FuelModel entry;
        _DEFAULT = {'r_frac': ['0.0'],
                    'pu_frac': ['0.0'],
                    'zr_frac': ['0.0'],
                    'porosity': ['0.0'],
                    'fcgap_thickness': 0.0,
                    'gap_thickness': 0.0,
                    'clad_material': None,
                    'gap_material': None,
                    'htc_params_clad': None}

        for asm in self.data['Assembly']:
            pre = f'Asm: "{asm}"; '  # indicate asm for error msg
            # For asm with default FuelModel entry: delete and continue
            if self.data['Assembly'][asm]['FuelModel'] == _DEFAULT:
                del self.data['Assembly'][asm]['FuelModel']
                continue

            # If specified, fuel-clad gap must be less than the
            # difference between the pin radius and clad thickness
            msg = ('Fuel-clad gap thickness must be less than the '
                   'clad inner radius')
            if (self.data['Assembly'][asm]['FuelModel']['gap_thickness']
                > (self.data['Assembly'][asm]['pin_diameter'] / 2.0
                   - self.data['Assembly'][asm]['clad_thickness'])):
                # raise ValueError(pre + msg)
                self.log('error', pre + msg)

            # Convert all values to float: if not possible, raise error
            # These come in as who-knows-what from "force_list", so this
            # conversion is also a setup step
            msg = 'Radial fuel pellet parameters must be of type float'
            fm_rad_keys = ['r_frac', 'pu_frac', 'zr_frac', 'porosity']
            for k in fm_rad_keys:
                try:
                    self.data['Assembly'][asm]['FuelModel'][k] = \
                        [float(x) for x in
                         self.data['Assembly'][asm]['FuelModel'][k]]
                except ValueError:
                    self.log('error', f'{pre}{msg}; key={k}')

            fm = self.data['Assembly'][asm]['FuelModel']

            # Move fcgap_thickness into gap_thickness if the user specified
            # the former but not the latter. The latter is the standard input
            # so this is just to keep the old input argument live
            if fm['gap_thickness'] == 0.0:
                if fm['fcgap_thickness'] > 0.0:
                    self.data['Assembly'][asm][
                        'FuelModel']['gap_thickness'] = fm['fcgap_thickness']
            del self.data['Assembly'][asm]['FuelModel']['fcgap_thickness']

            # Check that fractional radii are all increasing
            msg = ('Radius frations must arranged in increasing order; '
                   'if not annular fuel, the first value should be 0.0')
            for i in range(1, len(fm['r_frac'])):
                if fm['r_frac'][i] <= fm['r_frac'][i - 1]:
                    self.log('error', pre + msg)

            # Check that all inputs have entries
            msg = ('FuelModel input fields "r_frac", "pu_frac", '
                   '"zr_frac", and "porosity" are required')
            for k in fm_rad_keys:
                if len(fm[k]) == 0:
                    self.log('error', f'{pre}{msg}; missing "{k}"')

            # Check all inputs have same number of entries
            msg = 'FuelModel inputs must have equal number of nodes'
            for k in fm_rad_keys:
                if len(fm[k]) != len(fm['r_frac']):
                    self.log('error', f'{pre}{msg}; error in key={k}')

            # If FuelModel is specified, cladding material is required
            msg = '"clad_material" input required'
            if fm['clad_material'] is None:
                self.log('error', pre + msg)

            # If fuel-clad gap thickness is greater than 0, gap
            # material input is required
            msg = '"gap_material" required if gap_thickness > 0.0'
            if (fm['gap_thickness'] > 0.0 and
                    fm['gap_material'] is None):
                self.log('error', pre + msg)

            # Pu fraction must be less than 50% to ensure positive
            # thermal conductivity throughout fuel
            msg = ('Thermal conductivity correlation only guaranteed '
                   'nonnegative if Pu fraction is less than 37%')
            if any([x > 0.37037 for x in
                    self.data['Assembly'][asm]['FuelModel']['pu_frac']]):
                self.log('error', msg)

    def check_pin_model(self):
        """Make sure pin and fuel layer specs are physically meaningful
        and agree with other components of the Assembly input."""
        # All assemblies will have at least default PinModel entry;
        _DEFAULT = {'r_frac': ['0.0'],
                    'fcgap_thickness': 0.0,
                    'gap_thickness': 0.0,
                    'clad_material': None,
                    'gap_material': None,
                    'pin_material': None,
                    'htc_params_clad': None}

        for asm in self.data['Assembly']:
            pre = f'Asm: "{asm}"; '  # indicate asm for error msg
            # For asm with default PinModel entry: delete and continue
            if self.data['Assembly'][asm]['PinModel'] == _DEFAULT:
                del self.data['Assembly'][asm]['PinModel']
                continue
            else:
                if 'FuelModel' in self.data['Assembly'][asm].keys():
                    msg = 'Only one "PinModel" or "FuelModel" section allowed'
                    self.log('error', pre + msg)

            # Convert all values to float: if not possible, raise error
            # These come in as who-knows-what from "force_list", so this
            # conversion is also a setup step
            msg = 'Radial fuel pin parameters must be of type float'
            try:
                self.data['Assembly'][asm]['PinModel']['r_frac'] = \
                    [float(x) for x in
                     self.data['Assembly'][asm]['PinModel']['r_frac']]
            except ValueError:
                self.log('error', f'{pre}{msg}; key=r_frac')

            pin_model = self.data['Assembly'][asm]['PinModel']

            # Move fcgap_thickness into gap_thickness if the user specified
            # the former but not the latter. The latter is the standard input
            # so this is just to keep the old input argument live
            if pin_model['gap_thickness'] == 0.0:
                if pin_model['fcgap_thickness'] > 0.0:
                    self.data['Assembly'][asm]['PinModel']['gap_thickness'] = \
                        pin_model['fcgap_thickness']
            del self.data['Assembly'][asm]['PinModel']['fcgap_thickness']

            # Must have entries in 'pin_material'; these are checked for
            # validity in a different method
            if pin_model['pin_material'] is None:
                self.log('error', pre + 'Must specify "pin_material"')

            # If specified, fuel-clad gap must be less than the
            # difference between the pin radius and clad thickness
            msg = ('Fuel-clad gap thickness must be less than the '
                   'clad inner radius')
            r_out = 0.5 * self.data['Assembly'][asm]['pin_diameter']
            r_in = r_out - self.data['Assembly'][asm]['clad_thickness']
            if pin_model['gap_thickness'] > r_in:
                self.log('error', pre + msg)

            # Check that fractional radii are all increasing
            msg = ('Radius frations must arranged in increasing order; '
                   'if not annular fuel, the first value should be 0.0')
            for i in range(1, len(pin_model['r_frac'])):
                if pin_model['r_frac'][i] <= pin_model['r_frac'][i - 1]:
                    self.log('error', pre + msg)

            # Check that all inputs have entries
            msg = 'Inputs "r_frac" and "pin_material" are required'
            for k in ('r_frac', 'pin_material'):
                if len(pin_model[k]) == 0:
                    self.log('error', f'{pre}{msg}; missing \"{k}\"')

            # Check all inputs have same number of entries
            msg = 'PinModel inputs must have equal number of nodes'
            for k in ('r_frac', 'pin_material'):
                if len(pin_model[k]) != len(pin_model['r_frac']):
                    self.log('error', f'{pre}{msg}; error in key={k}')

            # If PinModel is specified, cladding material is required
            if pin_model['clad_material'] is None:
                self.log('error', pre + '"clad_material" input required')

            # If fuel-clad gap thickness is greater than 0, gap
            # material input is required
            msg = '"gap_material" required if gap_thickness > 0.0'
            if (pin_model['gap_thickness'] > 0.0 and
                    pin_model['gap_material'] is None):
                self.log('error', pre + msg)

    def check_correlations(self):
        """Add some hard cutoffs on assembly characteristics to avoid
        negative numbers.

        Notes
        -----
        This check is based on determining what values of P/D or W/D
        (W is edge pitch: the distance from the center of an edge pin
        to the inner duct wall) are allowable by solving for the
        quadratic roots based on the coefficients in Table 4 of the
        1986 Cheng-Todreas paper.

        There are two maximum values for P/D or W/D: one for laminar
        flow, another for turbulent flow. The laminar flow value is
        more restrictive. At the time this function is called, we
        don't know what the flow regime is. Therefore, we use the
        less restrictive value (turbulent) and will check again
        later on.

        """
        # w2d_limit = 2.10889   # LAMINAR
        # p2d_limit = 2.38024  # LAMINAR
        w2d_limit = 3.33271   # TURBULENT
        p2d_limit = 3.70617  # TURBULENT
        for asm in self.data['Assembly']:
            pre = f'Asm \"{asm}\"; '  # indicate asm for error msg
            if any(corr.lower() in ['ctd', 'uctd'] for corr in
                   [self.data['Assembly'][asm]['corr_friction'],
                    self.data['Assembly'][asm]['corr_flowsplit'],
                    self.data['Assembly'][asm]['corr_mixing']]):
                # Calculate pitch-to-diameter ratio
                p2d = (self.data['Assembly'][asm]['pin_pitch'] /
                       self.data['Assembly'][asm]['pin_diameter'])
                # Calculate edge gap-to-diameter ratio
                dftf = min(self.data['Assembly'][asm]['duct_ftf'])
                w = (dftf + self.data['Assembly'][asm]['pin_diameter']
                     - (np.sqrt(3)
                        * (self.data['Assembly'][asm]['num_rings'] - 1)
                        * self.data['Assembly'][asm]['pin_pitch']))
                w2d = w / self.data['Assembly'][asm]['pin_diameter']
                msg = 'ERROR: ' + pre
                if p2d > p2d_limit:
                    msg += ('Bundle P/D is too large to be acceptable '
                            'for CTD/UCTD correlations. Consider '
                            'modifying pin bundle design.')
                    self.log('error', msg)
                if w2d > w2d_limit:
                    msg += ('Gap between pin bundle and duct is too '
                            'large to be acceptable by CTD/UCTD '
                            'correlations. Consider modifying pin '
                            'bundle dimensions.')
                    self.log('error', msg)

    def check_assignment_assembly_agreement(self):
        """Make sure all assigned assemblies are specified"""
        # Make sure all specified assemblies are assigned to a position
        assn_pos = [a[0] for a in self.data['Assignment']['ByPosition']
                    if len(a) > 0]
        for asm in assn_pos:
            if asm not in self.data['Assembly'].keys():
                self.log('warning', (f'Assembly \"{asm}\" is specified'
                                     ' in "Assembly" input section but'
                                     ' not assigned to a position'))

        # Make sure all assigned assemblies are specified
        for i in range(len(self.data['Assignment']['ByPosition'])):
            if self.data['Assignment']['ByPosition'][i] == []:
                continue
            asm = self.data['Assignment']['ByPosition'][i][0]
            if asm not in self.data['Assembly'].keys():
                self.log('error', (f'Assembly type \"{asm}\" assigned '
                                   'to position but not specified in '
                                   '\"Assembly\" input section'))

    def check_assignment_boundary_conditions(self):
        """Confirm that all boundary conditions (flow rate or outlet
        temperature) are non-negative"""
        for i in range(len(self.data['Assignment']['ByPosition'])):
            assn = self.data['Assignment']['ByPosition'][i]
            if assn == []:
                continue

            # kwargs must have one of the following keywords
            nkwarg = 0
            for key in ['flowrate', 'outlet_temp', 'delta_temp']:
                if key in assn[2].keys():
                    bc = key
                    nkwarg += 1
            if nkwarg == 0:
                self.log('error', f'Assignment section line {i}: '
                                  'missing boundary condition')
            if nkwarg > 1:
                self.log('error', f'Assignment section line {i}: '
                                  'too many boundary conditions given')

            # Check that value is nonnegative
            msg = (f'Assignment section line {i}: \"{bc}\" must be '
                   f'positive but was given {assn[2][bc]}')
            self._check_nonzero(assn[2][bc], msg)

    def convert_assn_deltaT_to_outletT(self):
        """Convert 'delta_temp' boundary condition to 'outlet_temp'"""
        k = 'delta_temp'
        for i in range(len(self.data['Assignment']['ByPosition'])):
            if self.data['Assignment']['ByPosition'][i] == []:
                continue
            if k in self.data['Assignment']['ByPosition'][i][2].keys():
                # Get outlet temperature
                to = (self.data['Assignment']['ByPosition'][i][2][k]
                      + self.data['Core']['coolant_inlet_temp'])
                # Remove delta_temperature, save as outlet_temp
                del self.data['Assignment']['ByPosition'][i][2][k]
                self.data['Assignment']['ByPosition'][i][2]['outlet_temp'] = to

    def check_assignment_against_geodst(self, empty4c=False):
        """Make sure all assigned positions are active in GEODST"""
        if self.data['Power']['ARC']['geodst'] is None or empty4c:
            return

        for g in self.data['Power']['ARC']['geodst']:
            geodst = dassh.py4c.geodst.GEODST(g)

            msg = ('More assembly positions specified in input file '
                   f'than available in GEODST: {g}')
            n_active_pos = np.count_nonzero(geodst.reg_assignments[0])
            n_asm = len([x for x in self.data['Assignment']['ByPosition']
                         if x != []])
            if not n_asm <= n_active_pos:
                self.log('error', msg)

            # Check assignments against each reg assignment in GEODST
            geodst_map = geodst.reg_assignments[0].copy()
            n_zeros = geodst_map.size - np.count_nonzero(geodst_map)
            tmp = geodst_map.flatten()
            tmp = tmp.argsort()
            tmp = tmp.argsort() - (n_zeros - 1)
            tmp[tmp <= 0] = 0
            geodst_map = tmp.reshape(geodst_map.shape)
            _dirs = [(1, 0), (0, -1), (-1, -1), (-1, 0), (0, 1), (1, 1)]
            _origin = np.floor((np.array(geodst_map.shape) - 1) / 2)
            msg1 = ('Assembly assignment does not match region '
                    f'assignment in GEODST ({g})')
            for i in range(len(self.data['Assignment']['ByPosition'])):
                k = self.data['Assignment']['ByPosition'][i]
                if k == []:
                    continue
                else:
                    msg2 = f'Assembly: {k[1][2] + 1}; '
                    msg2 += f'Loc: ({k[1][0] + 1, k[1][1] + 1})'
                    msg = msg1 + '\n' + msg2
                    steps_per_side = k[1][0]
                    _active_pos = _origin + np.array([0, k[1][0]])
                    _idx = 0
                    _side = 0
                    _steps = 0
                    while _idx < k[1][1]:
                        _active_pos += _dirs[_side]
                        _steps += 1
                        _idx += 1
                        if _steps == steps_per_side:
                            _side += 1
                            _steps = 0
                    res = geodst_map[int(_active_pos[0]),
                                     int(_active_pos[1])]
                    if res == 0:
                        self.log('error', msg)

    def check_units(self):
        """Make sure user unit specifications are valid"""
        msg = 'Requested {:s} unit \"{:s}\" not supported'

        # Temperature
        temp_unit = self.data['Setup']['Units']['temperature'].lower()
        if temp_unit in utils._degK:
            self.data['Setup']['Units']['temperature'] = 'kelvin'
        elif temp_unit in utils._degC:
            self.data['Setup']['Units']['temperature'] = 'celsius'
        elif temp_unit in utils._degF:
            self.data['Setup']['Units']['temperature'] = 'fahrenheit'
        else:
            self.log('error', msg.format('temperature', temp_unit))

        # Length
        len_unit = self.data['Setup']['Units']['length'].lower()
        if len_unit in utils._cm:
            self.data['Setup']['Units']['length'] = 'cm'
        elif len_unit in utils._mm:
            self.data['Setup']['Units']['length'] = 'mm'
        elif len_unit in utils._m:
            self.data['Setup']['Units']['length'] = 'm'
        elif len_unit in utils._in:
            self.data['Setup']['Units']['length'] = 'in'
        elif len_unit in utils._ft:
            self.data['Setup']['Units']['length'] = 'ft'
        else:
            self.log('error', msg.format('length', len_unit))

        # Mass flow rate
        mfr_unit = self.data['Setup']['Units']['mass_flow_rate'].lower()
        # Get conversion method, then apply to all lengths
        if '/' in mfr_unit:
            mass_time = mfr_unit.split('/')
        elif 'per' in mfr_unit:
            mass_time = mfr_unit.split('per')
        else:
            msg = 'Do not understand mass flow rate unit specification: '
            self.log('error', msg + mfr_unit)
        if mass_time[0] in utils._lb:
            tmp = 'lb'
        elif mass_time[0] in utils._kg:
            tmp = 'kg'
        else:
            self.log('error', msg.format('mass flow rate (mass)', mfr_unit))
        if mass_time[1] in utils._sec:
            tmp += '/s'
        elif mass_time[1] in utils._min:
            tmp += '/min'
        elif mass_time[1] in utils._hr:
            tmp += '/hr'
        else:
            self.log('error', msg.format('mass flow rate (time)', mfr_unit))
        self.data['Setup']['Units']['mass_flow_rate'] = tmp

    def check_core_specifications(self):
        """Check the values specified in the Core section"""
        for k in ['length', 'assembly_pitch']:
            self._check_nonzero(self.data['Core'][k], k)

        if self.data['Core']['gap_model'] == 'none':
            self.data['Core']['gap_model'] = None

        if (self.data['Core']['gap_model'] == 'flow'
                and self.data['Core']['bypass_fraction'] == 0):
            self.log('error', ('\"bypass_fraction\" input must be '
                               'greater than 0 to use '
                               '\"gap_model=flow\"'))

    def check_user_spec_materials(self):
        """Check that material properties specifications make sense"""
        for m in self.data['Materials'].keys():
            # First check "from_file" key
            if self.data['Materials'][m]['from_file'] is not None:
                if not os.path.exists(
                        os.path.join(
                            self.path,
                            self.data['Materials'][m]['from_file'])):
                    msg = (f"Bad path to properties for material {m}; "
                           f"{self.data['Materials'][m]['from_file']}")
                    self.log('error', msg)
            else:
                # All properties must be lists of floats
                for p in ['heat_capacity',
                          'thermal_conductivity',
                          'density',
                          'viscosity',
                          'beta']:
                    msg = (f'Material "{m}" property "{p}" input '
                           'must be list of floats')
                    if p in self.data['Materials'][m].keys():
                        if self.data['Materials'][m][p] is not None:
                            try:
                                self.data['Materials'][m][p] = \
                                    [float(v) for v in
                                     self.data['Materials'][m][p]]
                            except ValueError:
                                self.log('error', msg)

    def check_axial_plane_req(self):
        """Check that user made appropriate requests for axial planes"""
        if self.data['Setup']['axial_plane'] is not None:
            try:
                axial = [float(x) for x in self.data['Setup']['axial_plane']]
            except ValueError:
                self.log('error', 'Setup // axial_plane input must be of '
                                  'type float (or list of float)')

            # At this point, we've either passed the try or created
            # an error, so let's make sure that none of the values in
            # the list are negative and that none are greater than
            # the given core height
            new_axial = []
            for x in axial:
                if (x > self.data['Core']['length'] or x < 0):
                    self.log('warning',
                             'Setup // axial_plane input must be greater than '
                             '0 and less than core length; ignoring ' + str(x))
                else:
                    new_axial.append(x)

            # Eliminate any duplicates, just in case
            new_axial = list(set(new_axial))

            # Now replace the old list with the new, clean one
            self.data['Setup']['axial_plane'] = new_axial

    def check_dump(self):
        """Check specifications for dumping temperatures"""
        keys = ['all', 'coolant', 'duct', 'pins', 'gap',
                'average', 'maximum', 'gap_fine']
        warn = False
        for k in self.data['Setup']['Dump'].keys():
            if k == 'interval':
                continue
            else:
                if self.data['Setup']['Dump'][k] and k not in keys:
                    warn = True
                    msg = ('Do not understand "Setup" sub-block '
                           f'"Dump" key "{k}"; ignoring...')
                    self.log('warning', msg)
                    del self.data['Setup']['Dump'][k]
        if warn:
            msg = 'Available boolean keys for "Setup" sub-block "Dump":'
            msg += '\n'
            msg += '\n'.join(['- all', '- coolant', '- duct', '- pins',
                              '- gap', '- average', '- maximum'])

    def check_parallel(self):
        """Check user request for parallel calculation"""
        if self.data['Setup']['parallel']:
            if self.timepoints == 1:
                msg = 'No parallelism for single timestep; ignoring...'
                self.log('warning', msg)
                self.data['Setup']['parallel'] = False
            else:  # self.timepoints > 1
                if self.data['Setup']['n_cpu'] == 1:
                    msg = ('Parallel execution requested but '
                           '"n_cpu"=1; ignoring...')
                    self.log('warning', msg)
                    self.data['Setup']['parallel'] = False
                else:
                    pass

    def check_setup_assembly_tables(self):
        """Check user inputs for detailed table setup"""
        pin_data_types = ('coolant_pin', 'clad_od', 'clad_mw',
                          'clad_id', 'fuel_od', 'fuel_cl')
        # Shortcut to dictionary that we're testing
        tmp = self.data['Setup']['AssemblyTables']

        # If all input fields are default None, skip
        if not any(tmp.values()):
            del self.data['Setup']['AssemblyTables']
            return

        # Otherwise, do checks for each table type
        for k in tmp.keys():

            # Check if inputs exist. If not, skip and remove
            if not any(tmp[k].values()):
                del self.data['Setup']['AssemblyTables'][k]
                continue
            # If one input field is default None, raise error - need both
            elif not all(tmp[k].values()):
                msg = (f'WARNING: Setup // AssemblyTables // {k} input '
                       'requires values for all three parameters: '
                       '"type", "assemblies", and "axial_positions". '
                       'Skipping.')
                del self.data['Setup']['AssemblyTables'][k]
                self.log('warning', msg)
                continue
            # Otherwise, do the rest of the checks
            else:
                pass

            # Check that all requested assemblies exist in assignment
            asm_to_keep = []
            for a in tmp[k]['assemblies']:
                try:
                    a = float(a)
                    assert a.is_integer()
                except:
                    msg = ('WARNING: Failed to convert Setup // '
                           f'AssemblyTables // {k} input assembly '
                           f'"{a}" to integer; skipping.')
                    self.log('warning', msg)
                    continue
                a = int(a)
                msg = (f'WARNING: Requested {k} AssemblyTable for '
                       f'assembly {a} which is not modeled; skipping.')
                if a > len(self.data['Assignment']['ByPosition']):
                    self.log('warning', msg)
                    continue
                elif len(self.data['Assignment']['ByPosition'][a - 1]) == 0:
                    self.log('warning', msg)
                    continue
                else:
                    asm_to_keep.append(a)
            if len(asm_to_keep) == 0:
                msg = ('WARNING: Did not recieve any acceptable input '
                       f'for Setup // AssemblyTables // {k} assemblies; '
                       'skipping.')
                self.log('warning', msg)
                del self.data['Setup']['AssemblyTables'][k]
                continue
            else:
                tmp[k]['assemblies'] = asm_to_keep

            # Check that axial positions are within domain
            z_to_keep = []
            for z in tmp[k]['axial_positions']:
                try:
                    z = float(z)
                except:
                    msg = ('WARNING: Failed to convert Setup // '
                           f'AssemblyTables // {k} input axial '
                           f'position "{z}" to float; skipping.')
                    self.log('warning', msg)
                    continue
                if z > self.data['Core']['length']:
                    msg = (f'WARNING: Setup // AssemblyTables // {k} '
                           f'input axial position "{z}" is greater '
                           'than specified core length; skipping.')
                    self.log('warning', msg)
                elif z < 0:
                    msg = (f'WARNING: Setup // AssemblyTables // {k} '
                           'input axial position must be greater than '
                           f'0, but was given: "{z}"; skipping.')
                    self.log('warning', msg)
                else:
                    z_to_keep.append(z)
            if len(z_to_keep) == 0:
                msg = ('WARNING: Did not recieve any acceptable input for '
                       f'Setup // AssemblyTables // {k} axial_positions '
                       '; skipping.')
                self.log('warning', msg)
                del self.data['Setup']['AssemblyTables'][k]
                continue
            else:
                tmp[k]['axial_positions'] = z_to_keep

            # If pin temperatures are requested, extra checks required:
            # 1. Make sure assembly has pin bundle model with fuel
            #    temperature calculation enabled
            # 2. Make sure axial positions are within pin bundle region
            datatype = tmp[k]['type']
            asm_to_keep = []
            z_to_keep = []
            if datatype in pin_data_types:
                for a in tmp[k]['assemblies']:
                    name = self.data['Assignment']['ByPosition'][a - 1][0]
                    asm_dict = self.data['Assembly'][name]
                    if not any([x in asm_dict.keys()
                                for x in ('FuelModel', 'PinModel')]):
                        msg = (f'WARNING: Requested "{datatype}" pin '
                               f'values in AssemblyTable "{k}" for '
                               f'assembly {a}, but no "FuelModel" or '
                               '"PinModel" input was given for that '
                               f'assembly type ("{name}"); skipping.')
                        self.log('warning', msg)
                    elif asm_dict['use_low_fidelity_model']:
                        msg = (f'WARNING: Requested "{datatype}" pin '
                               f'values in AssemblyTable "{k}" for '
                               f'assembly {a}, but cannot calculate '
                               'if "use_low_fidelity_model" option is '
                               'enabled; skipping.')
                        self.log('warning', msg)
                    else:
                        asm_to_keep.append(a)
                    for z in tmp[k]['axial_positions']:
                        bnds = [
                            asm_dict['AxialRegion']['rods']['z_lo'],
                            asm_dict['AxialRegion']['rods']['z_hi']]
                        if z < bnds[0] or z > bnds[1]:
                            msg = (f'WARNING: Requested "{datatype}" pin '
                                   f'values in AssemblyTable "{k}" for '
                                   f'assembly {a}, but cannot calculate '
                                   f'at axial_position "{z}" because it is '
                                   'outside pin bundle region; skipping.')
                            self.log('warning', msg)
                        else:
                            z_to_keep.append(z)

                if len(asm_to_keep) == 0:
                    msg = ('WARNING: Did not recieve any acceptable input '
                           f'for Setup // AssemblyTables // {k} assemblies; '
                           'skipping.')
                    self.log('warning', msg)
                    del self.data['Setup']['AssemblyTables'][k]
                    continue
                else:
                    tmp[k]['assemblies'] = asm_to_keep
                if len(z_to_keep) == 0:
                    msg = ('WARNING: Did not recieve any acceptable input for '
                           f'Setup // AssemblyTables // {k} axial_positions '
                           '; skipping.')
                    self.log('warning', msg)
                    del self.data['Setup']['AssemblyTables'][k]
                    continue
                else:
                    tmp[k]['axial_positions'] = z_to_keep

            # Assign modified dict to input data
            self.data['Setup']['AssemblyTables'][k] = tmp[k]

        # Make sure you're dumping coolant and average temperatures to CSV
        self.data['Setup']['Dump']['average'] = True
        if any(tmp[k]['type'] == 'coolant_subchannel' for k in tmp.keys()):
            self.data['Setup']['Dump']['coolant'] = True
        if any(tmp[k]['type'] == 'duct_mw' for k in tmp.keys()):
            self.data['Setup']['Dump']['duct'] = True
        if any(tmp[k]['type'] in pin_data_types for k in tmp.keys()):
            self.data['Setup']['Dump']['pins'] = True

    def check_htc_params(self):
        """Check user-specified coefficients to DB correlation"""
        msg_len = "Four HTC correlation parameters required."
        msg_neg = "HTC correlation parameters must be non-negative."
        # For each assembly
        for a in self.data['Assembly']:
            htc_params = self.data['Assembly'][a]['htc_params_duct']
            pre = f'Asm: \"{a}\"; '  # indicate asm for error msg
            if htc_params is not None:
                if not len(htc_params) == 4:
                    self.log('error', pre + msg_len)
                if any(x < 0.0 for x in htc_params):
                    self.log('error', pre + msg_neg)
        # For inter-assembly gap
        htc_params = self.data['Core']['htc_params_duct']
        if htc_params is not None:
            if not len(htc_params) == 4:
                self.log('error', 'Core; ' + msg_len)
            if any(x < 0.0 for x in htc_params):
                self.log('error', 'Core; ' + msg_neg)

    def _check_nonzero(self, val, msg):
        """If a value is not None, check that it is nonzero."""
        msg += ': Value must be greater than zero.'
        if val is not None:
            if val <= 0.0:
                self.log('error', msg)

    def _check_nonnegative(self, val, msg):
        """If a value is not None, check that it is nonnegative."""
        msg += ': Value must be greater than or equal to zero.'
        if val is not None:
            if val < 0.0:
                self.log('error', msg)

    def _check_tpts(self, list, msg):
        """Check the time-point consistency of user-inputs."""
        msg += ': Inconsistent time point input'
        if not len(list) == self.timepoints:
            self.log('error', msg)

    ####################################################################
    # UNIT CONVERSION
    ####################################################################

    def convert_units(self):
        """Convert units from user specification to those DASSH requires"""
        if (self.data['Setup']['Units']['temperature']
                not in utils._DEFAULT_UNITS['temperature']):
            self.data = convert_temperature(self.data)
        if (self.data['Setup']['Units']['length']
                not in utils._DEFAULT_UNITS['length']):
            self.data = convert_length(self.data)

        # Mass flow rate: need to split up mass/time components
        in_unit = self.data['Setup']['Units']['mass_flow_rate'].lower()
        # Get conversion method, then apply to all lengths
        if '/' in in_unit:
            mass_time = in_unit.split('/')
        elif 'per' in in_unit:
            mass_time = in_unit.split('per')
        else:
            msg = 'Do not understand mass flow rate unit specification: '
            self.log('error', msg + in_unit)
        m_unit = mass_time[0].split(' ')[0]
        t_unit = mass_time[1].split(' ')[-1]
        if (m_unit not in utils._DEFAULT_UNITS['mass']
                or t_unit not in utils._DEFAULT_UNITS['time']):
            self.data = convert_mass_flow_rate(self.data)
        return self

    ####################################################################
    # AXIAL REGION CLEANUP
    ####################################################################

    def axial_region_cleanup(self, tol=1e-6):
        """Find AxialRegion boundaries in GEODST zmesh

        Parameters
        ----------
        tol : float (optional)
            Acceptance tolerance when looking for axial boundary
            matches in GEODST files {default: 0.01 cm}

        Notes
        -----
        Because these values are static user inputs in the Assembly
        section, they should *NOT* change if multiple GEODST are
        present.

        Run this after unit conversion!!!

        """
        missing = ('Boundary {0} for AxialRegion {1} in Assembly {2} '
                   'not found in axial mesh of GEODST file {3}')
        multiple = ('Multiple matches for boundary {0} for AxialRegion '
                    '{1} in Assembly {2} found in axial mesh of GEODST '
                    'file {3}')
        # Open all geodst files
        geodst_files = []
        geodst = []
        for i in range(len(self.data['Power']['ARC']['geodst'])):
            geodst_files.append(os.path.join(self.path,
                                self.data['Power']['ARC']['geodst'][i]))
            geodst.append(py4c.geodst.GEODST(geodst_files[i]))

        # Find z_boundaries in the first GEODST file
        for asm in self.data['Assembly'].keys():

            # Skip if no specified axial regions
            if not any(self.data['Assembly'][asm]['AxialRegion']):
                continue

            for reg in self.data['Assembly'][asm]['AxialRegion'].keys():
                for k in ['z_lo', 'z_hi']:
                    z = self.data['Assembly'][asm]['AxialRegion'][reg][k]
                    z *= 100.0  # m -> cm
                    idx = np.where(np.abs(np.array(geodst[0].zmesh) - z) < tol)
                    if not len(idx[0]) > 0:
                        self.log('error',
                                 missing.format(z, reg, asm,
                                                geodst_files[0]))
                    if not len(idx[0]) == 1:
                        self.log('error',
                                 multiple.format(z, reg, asm,
                                                 geodst_files[0]))
                    # update the value
                    z1 = geodst[0].zmesh[idx[0][0]]
                    self.data['Assembly'][asm]['AxialRegion'][reg][k] = \
                        np.round(z1 * 0.01, 12)  # cm -> m

                    # Find corrected z-boundary in all other GEODST files
                    for g in range(1, len(geodst)):
                        if not np.any(np.abs(
                                np.array(geodst[g].zmesh) - z1) < tol):
                            self.log('error',
                                     missing.format(z1, reg, asm,
                                                    geodst_files[g]))

    def check_geodst(self, atol=1e-6):
        """Check GEODST files share same assembly pitch and same
        core length with DASSH input"""
        # Open all geodst files
        geodst_files = []
        geodst = []
        for i in range(len(self.data['Power']['ARC']['geodst'])):
            geodst_files.append(os.path.join(self.path,
                                self.data['Power']['ARC']['geodst'][i]))
            geodst.append(py4c.geodst.GEODST(geodst_files[i]))

        # Check assembly pitch in all files
        for i in range(len(geodst_files)):
            if not np.isclose(geodst[i].xmesh[1] / 100.0,  # cm --> m
                              self.data['Core']['assembly_pitch'],
                              atol=atol):
                self.log('error', 'Assembly pitch in GEODST file '
                                  f'{geodst_files[i]} does not match '
                                  'value given in DASSH input')
            if not np.isclose(geodst[i].zmesh[-1] / 100.0,  # cm --> m
                              self.data['Core']['length'],
                              atol=atol):
                self.log('error', 'Maximum z-mesh in GEODST file '
                                  f'{geodst_files[i]} does not match '
                                  'core length given in DASSH input')

    ####################################################################
    # LOAD MATERIALS
    ####################################################################

    def load_materials(self):
        """Set up DASSH Material objects for all coolant and
        structural materials"""
        inlet_temp = self.data['Core']['coolant_inlet_temp']
        # Expecting one coolant, multiple struct; collect all in list
        matlist = []
        matlist.append(self.data['Core']['coolant_material'])
        for a in self.data['Assembly'].keys():
            # Duct material always required
            if self.data['Assembly'][a]['duct_material'] not in matlist:
                matlist.append(self.data['Assembly'][a]['duct_material'])
            # Clad, fuel-clad gap materials optional
            if 'FuelModel' in self.data['Assembly'][a].keys():
                fm = self.data['Assembly'][a]['FuelModel']
                for k in ['gap_material', 'clad_material']:
                    if fm[k] is not None and fm[k] not in matlist:
                        matlist.append(fm[k])
            if 'PinModel' in self.data['Assembly'][a].keys():
                pm = self.data['Assembly'][a]['PinModel']
                for k in ['gap_material', 'clad_material']:
                    if pm[k] is not None and pm[k] not in matlist:
                        matlist.append(pm[k])
                for m in pm['pin_material']:
                    if m not in matlist and m is not None:
                        matlist.append(m)

        # Set up a DASSH Material for each material specified in the
        # list, importing correlations as necessary
        matdict = {}
        for m in matlist:
            if m in self.data['Materials'].keys():
                if self.data['Materials'][m]['from_file'] is not None:
                    # lookup from file - could be table/coeffs
                    file = self.data['Materials'][m]['from_file']
                    path = os.path.join(self.path, file)
                    matdict[m.lower()] = \
                        dassh.Material(m.lower(),
                                       temperature=inlet_temp,
                                       from_file=path)
                else:
                    # correlation coeffs specified as lists
                    # Filter None values out of dict
                    c = {k: v for k, v in self.data['Materials'][m].items()
                         if v is not None}
                    matdict[m.lower()] = \
                        dassh.Material(m.lower(),
                                       temperature=inlet_temp,
                                       coeff_dict=c)
            else:
                # No custom material defined, check built-in materials
                matdict[m.lower()] = \
                    dassh.Material(m.lower(), temperature=inlet_temp)

        # Check all of the materials to make sure they all have the
        # properties they need. Structure: thermal conductivity
        # Coolant: density, viscosity, thermal cond., heat capacity
        for m in matdict.keys():
            if m == self.data['Core']['coolant_material']:
                for prop in ('density', 'thermal_conductivity',
                             'viscosity', 'heat_capacity'):
                    self._check_mat(matdict[m], prop)
            else:
                self._check_mat(matdict[m], 'thermal_conductivity')

        # Assign the materials dictionary to the input object; to
        # be passed to the reactor object.
        self.materials = matdict

    def _check_mat(self, mat_obj, prop):
        """Check that material property attribute gives expected
        behavior - it has to exist and give nonnegative value"""
        # Check that material has property attribute
        if not hasattr(mat_obj, prop):
            self.log('error', (f'Material {mat_obj.name} missing '
                               f'property {prop}'))
        # Check that attribute returns feasible property value
        v = getattr(mat_obj, prop)
        if prop == 'thermal_conductivity':
            if not v >= 0.0:
                self.log('error', (f'Material {mat_obj.name}; expected '
                                   f'float >= 0 for property {prop}; '
                                   f'recieved {v}'))
        else:
            if not v > 0.0:
                self.log('error', (f'Material {mat_obj.name}; expected '
                                   f'positive float for property '
                                   f'{prop}; recieved {v}'))

    ####################################################################
    # CHECK PLOTTING INPUTS
    ####################################################################

    def check_plot_input(self):
        """General checks and input modifications; do everything req'd
        by DASSHPlot_Input and then some"""
        # Check using methods inherited from DASSHPlot_Input
        self.data['Setup']['Plot'] = \
            self.check_dasshplot_input(
                self.data['Setup']['Plot'],
                self.data['Core']['length'],
                len(self.data['Assignment']['ByPosition']))

        # Check that data required to make figures is dumped
        for k in self.data['Setup']['Plot'].keys():
            if k != 'core_hex':
                if k in ['core_subchannel', 'assembly_subchannel']:
                    self.data['Setup']['Dump']['coolant'] = True
                else:
                    pass
            else:
                tmp = self.data['Setup']['Plot'][k]  # shorter = easier
                for kk in tmp.keys():
                    if 'avg' in tmp[kk]['value']:
                        self.data['Setup']['Dump']['average'] = True
                    elif ('max' in tmp[kk]['value']
                          and tmp[kk]['z'] is not None):
                        self.data['Setup']['Dump']['maximum'] = True
                    else:
                        pass

    ####################################################################
    # CHECK ORIFICING OPTIMIZATION INPUT
    ####################################################################

    def check_orificing(self):
        """Check orificing optimization input; indicate whether
        optimization is to be performed"""
        # If orificing input not specified, then delete input section
        # and use as boolean to tell DASSH not to do any optimization
        none_keys = ['assemblies_to_group', 'n_groups',
                     'value_to_optimize', 'bulk_coolant_temp',
                     'pressure_drop_limit']
        if all(self.data['Orificing'][k] is None for k in none_keys):
            self.data['Orificing'] = False
        else:  # Otherwise, run some checks on it
            # Make sure all necessary values without defaults are given
            for k in none_keys:
                if k == 'pressure_drop_limit':
                    continue
                else:
                    if self.data['Orificing'][k] is None:
                        self.log('error', (f'Orificing input "{k}" '
                                           + 'must be specified'))
            # Check that all assemblies to be assigned to groups are
            # also found in the Assembly input block
            for a in self.data['Orificing']['assemblies_to_group']:
                if a not in self.data['Assembly'].keys():
                    self.log('error',
                             'Orificing input "assemblies_to_group" '
                             + 'must contain assembly types specified '
                             + f'in "Assembly"; do not recognize "{a}"')
            # Make sure the bulk outlet temperature requested by the
            # user is greater than the inelt temperature
            if self.data['Orificing']['bulk_coolant_temp'] <= \
                    self.data['Core']['coolant_inlet_temp']:
                self.log('error',
                         'Orificing input "bulk_coolant_temp" must be '
                         + 'greater than "Core/coolant_inlet_temp"')
            # Make sure user has supplied the necessary information to
            # perform the optimization: FuelModel input
            pin_keys = ['peak clad MW temp', 'peak clad ID temp',
                        'peak fuel temp']
            if self.data['Orificing']['value_to_optimize'] in pin_keys:
                for a in self.data['Orificing']['assemblies_to_group']:
                    if 'FuelModel' not in self.data['Assembly'][a].keys():
                        self.log('error',
                                 'Cannot perform orificing '
                                 + 'optimization on pin temperatures '
                                 + f'for Assembly "{a}": no FuelModel '
                                 + 'input section')


########################################################################
# GENERAL CONFIGOBJ METHODS
########################################################################


def _configobj_load(dassh_inp_object, infile, path_to_template):
    """Read input into dictionary using configobj.

    Parameters
    ----------
    dassh_inp_object : DASSH_Input or DASSHPlot_Input object
        DASSH input handler
    infile : str
        User-produced DASSH input file
    path_to_template : str
        File path to Configobj input template

    Returns
    -------
    dict
        Input file data

    """
    inp = ConfigObj(infile.splitlines(), configspec=path_to_template,
                    raise_errors=True, file_error=True)
    # Instantiate Validator object; check against the template
    validator = Validator()
    res = inp.validate(validator, preserve_errors=True)
    if res is not True:
        msg = ''
        for (sec_list, key, _) in flatten_errors(inp, res):
            if key is not None:
                msg += ('"%s" key in section "%s" failed validation'
                        '; check that it meets the requirements'
                        % (key, ', '.join(sec_list)) + '\n')
            else:
                msg += ('Error found in the following '
                        + 'section: %s ' % ', '.join(sec_list)
                        + '; maybe missing required input?' + '\n')
        dassh_inp_object.log('error', msg)
    # otherwise no errors, return data
    return inp


def _configobj_check_extra_kw(dassh_inp_obj, inp_data, only=None, skip=None):
    """If the user added anything funky, make sure it's known

    Parameters
    ----------
    dassh_inp_object : DASSH_Input or DASSHPlot_Input object
        DASSH input handler
    input_data : dict
        Input file data from ConfigObj
    only (optional) : str
        Check arguments for a specific section (default=None)
    skip (optional) : str
        Skip a specific section (default=None)

    Returns
    -------
    None

    """
    extra_args = configobj.get_extra_values(inp_data)
    for x in extra_args:
        msg = 'Warning: unrecognized input. '
        if len(x[0]) > 0:
            sec = '"//"'.join(x[0])
            msg += f'Section: "{sec}"'
            msg += f'; keyword: "{x[1]}"'
        else:
            msg += f'Section: "{x[1]}"'
        if only is not None and only not in msg:
            continue
        elif skip is not None and skip in msg:
            continue
        else:
            dassh_inp_obj.log('warning', msg)


########################################################################
# AXIAL REGION BOUNDARY MATCHING
########################################################################


def _find_rodded_regs(zlo, zhi, bnds):
    tmp_lo = zlo + [bnds[1]]
    tmp_hi = [bnds[0]] + zhi
    bupkis = [tmp_lo[i] - tmp_hi[i] for i in range(len(tmp_lo))]
    return bupkis


def _check_reg_bnds(rodded_regs):
    # Multiple rodded regions within the axial space (can only
    # have one rodded region)
    if len([v for v in rodded_regs if v != 0]) > 1:
        return False
    else:
        return True


def _get_rodded_reg_bnds(zlo, zhi, rr, cbnds):
    idx = np.where(np.array(rr) != 0.0)[0][0]
    # Case 1: Unrodded region(s) below the rod bundle
    if idx == len(rr) - 1:  # Rodded region is the last region
        return [zhi[idx - 1], cbnds[1]]
    # Case 2: unrodded region(s) above the rod bundle
    elif idx == 0:
        return [0.0, zlo[0]]
    # Case 3: more than one unrodded region
    else:
        return [zhi[idx - 1], zlo[idx]]


########################################################################
# UNIT CONVERSION METHODS
########################################################################


def convert_temperature(data):
    """Convert user temperature inputs from specified units to Kelvin

    Parameters
    ----------
    data : dict
        DASSH_Input data dict from ConfigObj

    Returns
    -------
    dict
        New input dict with updated temperature parameters

    """
    # Get conversion method, then apply to all temperatures
    input_unit = data['Setup']['Units']['temperature']
    conv = utils.get_temperature_conversion(input_unit, 'k')

    # Convert inlet temperature; save original just in case
    original_inlet_temp = data['Core']['coolant_inlet_temp']
    data['Core']['coolant_inlet_temp'] = \
        conv(data['Core']['coolant_inlet_temp'])

    # Convert orificing target temperature, if pressent
    if data['Orificing']:
        if data['Orificing']['bulk_coolant_temp'] is not None:
            data['Orificing']['bulk_coolant_temp'] = \
                conv(data['Orificing']['bulk_coolant_temp'])

    # Assembly outlet temperature assignments
    for i in range(len(data['Assignment']['ByPosition'])):
        if data['Assignment']['ByPosition'][i] == []:
            continue

        if 'outlet_temp' in data['Assignment']['ByPosition'][i][2].keys():
            data['Assignment']['ByPosition'][i][2]['outlet_temp'] = \
                conv(data['Assignment']['ByPosition'][i][2]['outlet_temp'])

        elif 'delta_temp' in data['Assignment']['ByPosition'][i][2].keys():
            # Get outlet temperature in original units
            t_out = (data['Assignment']['ByPosition'][i][2]['delta_temp']
                     + original_inlet_temp)
            # Just convert to outlet temp and save
            del data['Assignment']['ByPosition'][i][2]['delta_temp']
            data['Assignment']['ByPosition'][i][2]['outlet_temp'] = \
                conv(t_out)

        else:
            continue

    return data


def convert_length(data):
    """Convert user length inputs from specified units to meters

    Parameters
    ----------
    data : dict
        DASSH_Input data dict from ConfigObj

    Returns
    -------
    dict
        New input dict with updated length parameters

    """
    # Get conversion method, then apply to all lengths
    input_unit = data['Setup']['Units']['length']
    conv = utils.get_length_conversion(input_unit, 'm')

    # Convert all temperature inputs with specified conversion method
    data['Core']['length'] = conv(data['Core']['length'])
    data['Core']['assembly_pitch'] = conv(data['Core']['assembly_pitch'])

    # data['Core']['assembly_pitch'] = conv(data['Core']['assembly_pitch'])
    for a in data['Assembly'].keys():
        for p in ['pin_pitch', 'pin_diameter', 'clad_thickness',
                  'wire_pitch', 'wire_diameter']:
            data['Assembly'][a][p] = conv(data['Assembly'][a][p])

        data['Assembly'][a]['duct_ftf'] = \
            [conv(x) for x in data['Assembly'][a]['duct_ftf']]

        for k in data['Assembly'][a]['AxialRegion'].keys():
            for p in ['z_lo', 'z_hi', 'hydraulic_diameter']:
                if p in data['Assembly'][a]['AxialRegion'][k].keys():
                    data['Assembly'][a]['AxialRegion'][k][p] = \
                        conv(data['Assembly'][a]['AxialRegion'][k][p])

        if 'FuelModel' in data['Assembly'][a].keys():
            data['Assembly'][a]['FuelModel']['gap_thickness'] = \
                conv(data['Assembly'][a]['FuelModel']['gap_thickness'])

    # Convert requested axial plane solves
    if data['Setup']['axial_plane'] is not None:
        for i in range(len(data['Setup']['axial_plane'])):
            data['Setup']['axial_plane'][i] = \
                conv(data['Setup']['axial_plane'][i])

    # Convert axial step size, if given
    if data['Setup']['axial_mesh_size'] is not None:
        data['Setup']['axial_mesh_size'] = \
            conv(data['Setup']['axial_mesh_size'])

    # Convert data dumping interval, if given
    if data['Setup']['Dump']['interval'] is not None:
        data['Setup']['Dump']['interval'] = \
            conv(data['Setup']['Dump']['interval'])
    else:
        # SET DEFAULT DUMP INTERVAL: 1 cm
        data['Setup']['Dump']['interval'] = 0.01

    # Convert duct approx cutoff
    if data['Setup']['conv_approx_dz_cutoff'] is not None:
        data['Setup']['conv_approx_dz_cutoff'] = \
            conv(data['Setup']['conv_approx_dz_cutoff'])

    if 'AssemblyTables' in data['Setup'].keys():
        for k in data['Setup']['AssemblyTables'].keys():
            z = data['Setup']['AssemblyTables'][k]['axial_positions']
            z_conv = [conv(zz) for zz in z]
            data['Setup']['AssemblyTables'][k]['axial_positions'] = z_conv

    return data


def convert_mass_flow_rate(data):
    """Convert user mass flow rate inputs from specified units to kg/s

    Parameters
    ----------
    data : dict
        DASSH_Input data dict from ConfigObj

    Returns
    -------
    dict
        New input dict with updated mass flow rate parameters
    """
    # Get conversion method, then apply to all lengths
    in_unit = data['Setup']['Units']['mass_flow_rate'].lower()
    m_unit, t_unit = utils.parse_mfr_units(in_unit)

    # remove any spaces and convert
    m_conv = utils.get_mass_conversion(m_unit, 'kg')
    # Because time is in the denominator, conversion mult is backwards
    t_conv = utils.get_time_conversion('s', t_unit)

    # Assembly mass flow rate assignments
    for i in range(len(data['Assignment']['ByPosition'])):
        if data['Assignment']['ByPosition'][i] == []:
            continue
        if 'flowrate' in data['Assignment']['ByPosition'][i][2].keys():
            data['Assignment']['ByPosition'][i][2]['flowrate'] = \
                t_conv(m_conv(data['Assignment']
                                  ['ByPosition']
                                  [i][2]['flowrate']))
    return data
