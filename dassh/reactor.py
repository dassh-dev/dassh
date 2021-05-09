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
date: 2021-05-09
author: matz
Object to hold and control DASSH components and execute simulations
"""
########################################################################
import os
import copy
# import h5py
import numpy as np
import subprocess
import logging
import sys
import pickle
import datetime
import time
import multiprocessing as mp

import dassh
from dassh.logged_class import LoggedClass


_FUELS = {'metal': {'zr': 1, 'zirconium': 1, 'al': 4, 'aluminum': 4},
          'oxide': 2,
          'nitride': 3}
_COOLANTS = {'na': 1, 'sodium': 1,
             'nak': 2, 'sodium-potassium': 2,
             'pb': 3, 'lead': 3,
             'pb-bi': 4, 'lead-bismuth': 4,
             'lbe': 4, 'lead-bismuth-eutectic': 4,
             'sn': 5, 'tin': 5}


module_logger = logging.getLogger('dassh.reactor')


def load(path='dassh_reactor.pkl'):
    """Load a saved Reactor object from a file

    Parameters
    ----------
    path : str
        Path to Reactor object (default file is dassh_reactor.pkl)

    Returns
    -------
    DASSH Reactor object

    """
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


class Reactor(LoggedClass):
    """Object to hold and control DASSH Assembly and Core objects and
    perform temperature sweep calculations per user input.

    Attributes
    ----------

    Notes
    -----
    in __init__:
    - Calculate core power profile (call method)
    - Instantiate base assemblies
    - Set up assembly list via clone
    - Set up core object
    - Calculate axial constraint

    Also include in this object:
    - Subchannel temperature arrays for core object, each assembly
    - sweep method to perform whole core temperature calculation
    - oriface method to iteratively call sweep method

    """
    def __init__(self, dassh_input, path=None, calc_power=True, **kwargs):
        """Initialize Reactor object for DASSH simulation

        Parameters
        ----------
        dassh_input : DASSH_Input object
            DASSH input from read_input.DASSH_Input

        """
        LoggedClass.__init__(self, 0, 'dassh.reactor.Reactor')

        # Store user options from input/invocation
        self.units = dassh_input.data['Setup']['Units']
        self._setup_options(dassh_input, **kwargs)

        # Store general inputs
        self.inlet_temp = dassh_input.data['Core']['coolant_inlet_temp']
        self.asm_pitch = dassh_input.data['Core']['assembly_pitch']
        if path is None:
            self.path = dassh_input.path
        else:
            self.path = path
            os.makedirs(path, exist_ok=True)

        # Store DASSH materials (already loaded in DASSH_Input)
        self.materials = dassh_input.materials

        # Set up power, obtain axial region boundaries
        self.log('info', 'Setting up power distribution')
        self._setup_power(dassh_input, calc_power)
        self._setup_axial_region_bnds(dassh_input)

        # Set up DASSH Assemblies by first creating templates, then
        # cloning them into each specified position in the core
        self.log('info', 'Generating Assembly objects')
        self._setup_asm_templates(dassh_input)
        asm_power = self._setup_asm_power(dassh_input)
        est_Tout, est_fr = self._setup_asm_bc(dassh_input, asm_power)
        self._setup_asm(dassh_input, asm_power, est_Tout, est_fr)

        # Determine whether inter-assembly heat transfer is necessary,
        # then set up assembly axial mesh size requirement
        self._is_adiabatic = False
        if dassh_input.data['Core']['gap_model'] is None:
            self._is_adiabatic = True
        self._setup_asm_axial_mesh_req()

        # Set up DASSH Core object; first need to calculate inter-
        # assembly gap flow rate based on total flow rate to the
        # assemblies.
        self.log('info', 'Generating Core object')
        self.flow_rate = self._calculate_total_fr(dassh_input)
        self._setup_core(dassh_input)

        # Report some updates: total power and flow rate
        msg = 'Total power (W): {:.1f}'.format(self.total_power)
        self.log('info', msg)
        msg = 'Total flow rate (kg/s): {:.4f}'.format(self.flow_rate)
        self.log('info', msg)

        # Set up axial mesh
        self._setup_overall_axial_mesh_req()
        self.z, self.dz = self._setup_zpts()
        self.log('info', f'{len(self.z) - 1} axial steps required')
        # Warn if axial steps too small (< 0.5 mm) or too many (> 4k)
        if self.req_dz < 0.0005 or len(self.z) - 1 > 2500:
            msg = ('Your axial step size is very small so this '
                   'problem might take a while to solve;\nConsider '
                   'checking input for flow maldistribution.')
            self.log('warning', msg)

        # Raise warning if est. coolant temp will exceed extreme limit
        self._melt_warning(dassh_input, T_max=1500)

        # Generate general output file
        if self._options['write_output']:
            self.write_summary()

    def _setup_options(self, inp, **kwargs):
        """Store user options from input/invocation"""
        opt = inp.data['Setup']['Options']

        # Load defaults where they aren't be taken from input
        self._options = {}
        self._options['conv_approx_dz_cutoff'] = 0.001
        self._options['write_output'] = False
        self._options['log_progress'] = False
        self._options['parallel'] = False

        # Process user input
        self._options['axial_plane'] = opt['axial_plane']
        self._options['se2geo'] = opt['se2geo']

        if 'write_output' in kwargs.keys():  # always True in __main__
            self._options['write_output'] = kwargs['write_output']

        self._options['debug'] = opt['debug']
        if 'debug' in kwargs.keys():
            self._options['debug'] = kwargs['debug']

        self._options['axial_mesh_size'] = opt['axial_mesh_size']
        if 'axial_mesh_size' in kwargs.keys():
            self._options['axial_mesh_size'] = kwargs['axial_mesh_size']

        if opt['log_progress'] > 0:
            self._options['log_progress'] = True
            self._options['log_interval'] = opt['log_progress']
            self._stepcount = 0.0

        # Low-flow convection approximation
        self._options['conv_approx'] = opt['conv_approx']
        if opt['conv_approx_dz_cutoff'] is not None:
            self._options['conv_approx_dz_cutoff'] = \
                opt['conv_approx_dz_cutoff']

        self._options['ebal'] = opt['calc_energy_balance']
        if 'calc_energy_balance' in kwargs.keys():
            self._options['ebal'] = kwargs['calc_energy_balance']

        # DUMP FILE ARGUMENTS: collect to set up files at sweep time
        self._options['dump'] = inp.data['Setup']['Dump']
        # Overwrite with kw arguments
        for k in self._options['dump']:
            if k in kwargs.keys():
                self._options['dump'][k] = kwargs[k]
        if self._options['dump']['all']:
            for k in self._options['dump'].keys():
                if k == 'interval':
                    continue
                else:
                    self._options['dump'][k] = True
        self._options['dump']['any'] = False
        if any(self._options['dump'].values()):
            self._options['dump']['any'] = True

    def _setup_power(self, inp, calc_power_flag):
        """Create the power distributions from ARC binary files or
        user specifications

        Parameters
        ----------
        inp : DASSH_Input object
        calc_power_flag : bool
            Flag indicating whether to run VARPOW to calculate power
            distribution from ARC binary files

        """
        # NEED TO FIGURE OUT HOW TO TREAT MULTI-TIMEPOINT PROBLEMS
        # Not passing any timepoint argument to calc_power_VARIANT
        # and limiting user_power to first list entry. Could come
        # from reactor instantiation --> new reactor for every
        # timepoint? Or have a rinse/repeat method for new timepoints
        # so I can skip redefining assembly objects.
        self.power = {}

        # 1. Calculate power based on VARIANT flux
        # if True:  # This needs to be a check whether binary files exist
        if inp._cccc_power:
            if calc_power_flag:
                msg = ('Calculating core power profile from CCCC '
                       'binary files')
                self.log('info', msg)
                self.power['dif3d'] = \
                    calc_power_VARIANT(inp.data, self.path)
            else:  # Go find it in the working directory
                msg = ('Reading core power profile from VARPOW '
                       'output files')
                self.log('info', msg)
                self.power['dif3d'] = \
                    import_power_VARIANT(inp.data, self.path)

        # 2. Read user power, if given
        if inp.data['Power']['user_power'][0] is not None:
            msg = ('Reading user-specified power profiles from '
                   + inp.data['Power']['user_power'][0])
            self.power['user'] = \
                dassh.power._from_file(inp.data['Power']['user_power'][0])

    def _setup_axial_region_bnds(self, inp):
        """Get axial mesh points from ARC binary files, user-specified
        power distribution, and user input file request

        Parameters
        ----------
        inp : DASSH_Input object

        Returns
        -------
        None

        """
        # Accumulate all values in list
        ax_bnd = []

        # DIF3D binary files
        if 'dif3d' in self.power.keys():
            ax_bnd += list(self.power['dif3d'].z_finemesh * 1e-2)

        # User power specification
        if 'user' in self.power.keys():
            for ai in range(len(self.power['user'])):
                ax_bnd += list(self.power['user'][ai][1]['zfm'])

        # Axial regions in assembly specification
        for a in inp.data['Assembly'].keys():
            tmp = inp.data['Assembly'][a]['AxialRegion']
            for r in tmp.keys():
                ax_bnd.append(tmp[r]['z_lo'])
                ax_bnd.append(tmp[r]['z_hi'])

        # User axial boundary request
        if self._options['axial_plane'] is not None:
            ax_bnd += self._options['axial_plane']

        # Round values then discard duplicates
        ax_bnd = np.unique(np.around(ax_bnd, 12))
        self.axial_bnds = ax_bnd
        self.core_length = self.axial_bnds[-1]

    def _setup_asm_templates(self, inp):
        """Generate template DASSH Assembly objects based on user input

        Parameters
        ----------
        inp : DASSH_Input object
            Contains "data" attribute with user inputs

        Returns
        -------
        dict
            Dictionary of DASSH assembly objects with placeholder
            positions and coolant mass flow rates

        """
        asm_templates = {}
        mfrx = -1.0  # placeholder for mass flow rate in cloned asm
        # inlet_temp = inp_obj.data['Core']['coolant_inlet_temp']
        cool_mat = inp.data['Core']['coolant_material'].lower()
        for a in inp.data['Assembly'].keys():
            asm_data = inp.data['Assembly'][a]

            # Create materials dictionary
            mat_data = {}
            mat_data['coolant'] = self.materials[cool_mat].clone()
            mat_data['duct'] = self.materials[
                asm_data['duct_material'].lower()].clone()
            if 'FuelModel' in asm_data:
                mat_data['clad'] = self.materials[
                    asm_data['FuelModel']['clad_material'].lower()
                ].clone()
                if asm_data['FuelModel']['gap_material'] is not None:
                    mat_data['gap'] = self.materials[
                        asm_data['FuelModel']['gap_material'].lower()
                    ].clone()
                else:
                    mat_data['gap'] = None
            # make the list of "template" Assembly objects
            asm_templates[a] = dassh.assembly.Assembly(
                a, (-1, -1), asm_data, mat_data,  self.inlet_temp, mfrx,
                se2geo=self._options['se2geo'])

        # Store as attribute b/c used later to write summary output
        self.asm_templates = asm_templates

    def _setup_asm_power(self, inp):
        """Generate assembly power profiles

        Parameters
        ----------
        inp : DASSH_Input object

        Returns
        -------
        list
            List of tuples containing assembly power parameters for
            each assembly, arranged by DASSH index
            1. Power profiles for pins, duct, coolant
            2. Average power profile
            3. Total power
            4. Z-mesh that defines power profile axial boundaries

        """
        # Return list of power profiles arranged by DASSH index
        asm_power = []
        core_total_power = 0.0

        # Identify assemblies that have user power specifications;
        # convert to Python index; returns empty list of no user power
        user_power_idx = []
        if 'user' in self.power.keys():
            user_power_idx = [x[0] - 1 for x in self.power['user']]

        for i in range(len(inp.data['Assignment']['ByPosition'])):
            # If assembly in this position is undefined by DASSH:
            # leave returnables empty, and continue
            if inp.data['Assignment']['ByPosition'][i] == []:
                asm_power.append([])
                continue

            # Pull up assignment and assembly input data
            # k[0]: assembly type : str e.g. its name ("reflector")
            # k[1]: assembly loc : tuple (ring, pos, id)  all base-0
            # k[2]: dict with kwargs
            k = inp.data['Assignment']['ByPosition'][i]
            atype = k[0]

            # Calculate total power and determine component power
            # profiles, but do not assign to new assembly object.
            # Try to find in user-supplied power
            if i in user_power_idx:
                # isolate appropriate user power dictionary
                tmp = self.power['user'][user_power_idx.index(i)][1]
                avg_power_profile = tmp['avg_power']
                power_profile = tmp
                z_mesh = tmp['zfm']
                tot_power = np.sum((z_mesh[1:] - z_mesh[:-1])
                                   * avg_power_profile)

                # Need to check that user power input matches assembly
                # assignment geometry (number of pins, etc)
            else:  # Get it from DIF3D power
                power_profile, avg_power_profile = \
                    self.power['dif3d'].calc_power_profile(
                        self.asm_templates[atype], i)
                tot_power = np.sum(self.power['dif3d'].power[i])
                z_mesh = self.power['dif3d'].z_finemesh

            # Track total power
            core_total_power += tot_power

            # add to list
            asm_power.append(
                [power_profile,
                 avg_power_profile,
                 tot_power,
                 z_mesh])

        # Scale power as requested by user and assign "total_power"
        # attribute to Reactor object; return assembly power list
        asm_power, total_power = self._setup_scale_asm_power(
            asm_power,
            core_total_power,
            inp.data['Core']['total_power'],
            inp.data['Core']['power_scaling_factor'])
        self.total_power = total_power
        return asm_power

    @staticmethod
    def _setup_scale_asm_power(plist, pcalc, ptot_user, pscalar):
        """Scale assembly power according to user request

        Parameters
        ----------
        plist : list
            List of power profile information for each assembly
        pcalc : float
            Calculated total power
        ptot_user : float
            User-requested core total power
        pscalar : float
            Scaling factor to apply to core total power

        Returns
        -------
        list
            "plist" with items modified to reflect scaled power

        Notes
        -----
        If the user requests a total power normalization and applies
        a scaling factor to the power, the resulting core power will
        be equal to the product of the requested core power and the
        scaling factor.

        """
        # Normalize power to user request
        renorm = 1.0
        if ptot_user is not None:
            renorm = ptot_user / pcalc
            for i in range(len(plist)):
                if plist[i] == []:  # skip if asm is undefined
                    continue
                # Component power profiles
                for k in plist[i][0].keys():
                    plist[i][0][k] *= renorm
                # Average power profile
                plist[i][1] *= renorm
                # Total power
                plist[i][2] *= renorm

        # Scale power again if user requested
        if pscalar != 1.0:
            for i in range(len(plist)):
                if plist[i] == []:  # skip if asm is undefined
                    continue
                # Component power profiles
                for k in plist[i][0].keys():
                    plist[i][0][k] *= pscalar
                # Average power profile
                plist[i][1] *= pscalar
                # Total power
                plist[i][2] *= pscalar

        return plist, pcalc * renorm * pscalar

    def _setup_asm_bc(self, inp, power_params):
        """Estimate flow rate or outlet temperature

        Parameters
        ----------
        inp : DASSH_Input object
        power_params : list
            List of tuples generated by _setup_asm_power method

        """
        T_out = []
        flow_rate = []
        for i in range(len(inp.data['Assignment']['ByPosition'])):
            # If assembly in this position is undefined by DASSH:
            # leave returnables empty, and continue
            if inp.data['Assignment']['ByPosition'][i] == []:
                T_out.append([])
                flow_rate.append([])
                continue

            # Pull up assignment and assembly input data
            # k[0]: assembly type : str e.g. its name ("reflector")
            # k[1]: assembly loc : tuple (ring, pos, id)  all base-0
            # k[2]: dict with kwargs
            k = inp.data['Assignment']['ByPosition'][i]
            atype = k[0]

            # Pull assembly power from power parameters list
            asm_power = power_params[i][2]
            if 'flowrate' in k[2].keys():  # estimate outlet temp
                flow_rate_tmp = k[2]['flowrate']
                T_out_tmp = dassh.utils.Q_equals_mCdT(
                    asm_power,
                    self.inlet_temp,
                    self.asm_templates[atype].active_region.coolant,
                    mfr=flow_rate_tmp)
            elif 'outlet_temp' in k[2].keys():  # estimate flow rate
                T_out_tmp = k[2]['outlet_temp']
                flow_rate_tmp = dassh.utils.Q_equals_mCdT(
                    asm_power,
                    self.inlet_temp,
                    self.asm_templates[atype].active_region.coolant,
                    t_out=T_out_tmp)
            else:
                msg = ('Could not estimate flow rate / outlet temp'
                       f'for asm no. {id} ({atype}) from given inputs')
                self.log('error', msg)

            T_out.append(T_out_tmp)
            flow_rate.append(flow_rate_tmp)
        return T_out, flow_rate

    def _setup_asm(self, inp, asm_power, To, fr):
        """Generate a list of DASSH assemblies and determine the minimum
        axial mesh size required for numerical stability.

        Parameters
        ----------
        inp_obj : DASSH Input object
            User inputs to DASSH

        Returns
        -------
        list
            Assemblies in the core, ordered by position index
        float
            Minimum axial mesh size required for core-wide stability

        Notes
        -----
        1. Identify assembly index and location based on user input.
        2. Calculate total power and determine component power
           profiles but do not assign to new assembly object.
        3. Using total power, estimate outlet temperature and flow
           rate as necessary.
        4. Clone assembly object from template using flow rate and
           assign power profiles.
        5. Store outlet temperature estimate to use when determining
           axial mesh size requirement.

        """
        # List of assemblies to populate
        assemblies = []
        for i in range(len(inp.data['Assignment']['ByPosition'])):
            if inp.data['Assignment']['ByPosition'][i] == []:
                continue

            # Pull up assignment and assembly input data
            # k[0]: assembly type : str e.g. its name ("reflector")
            # k[1]: assembly loc : tuple (ring, pos, id)  all base-0
            # k[2]: dict with kwargs
            k = inp.data['Assignment']['ByPosition'][i]
            atype = k[0]
            loc = k[1][:2]
            asm_data = inp.data['Assembly'][atype]

            # Power scaling for individual assemblies: WARNING
            # This is only meant to be a developer feature to test
            # heat transfer between assemblies. It will ruin the
            # normalization of power to the fixed value requested
            # in the input
            power_scalar = 1.0
            # if self._options['debug']:
            #     if 'scale_power' in k[2].keys():
            #         power_scalar = k[2]['scale_power']
            # asm_power *= power_scalar

            # Clone assembly object from template using flow rate
            # and assign power profiles
            asm = self.asm_templates[atype].clone(loc, new_flowrate=fr[i])

            bundle_bnd = get_rod_bundle_bnds(asm_power[i][3], asm_data)
            asm.power = dassh.power.AssemblyPower(asm_power[i][0],
                                                  asm_power[i][1],
                                                  asm_power[i][3],
                                                  bundle_bnd,
                                                  scale=power_scalar)
            asm.total_power = asm_power[i][2]
            asm._estimated_T_out = To[i]
            assemblies.append(asm)

        # Sort the assemblies according to the DASSH assembly ID
        self.assemblies = assemblies

    def _setup_asm_axial_mesh_req(self):
        """Calculate the required axial mesh size for each assembly"""
        self.min_dz = {}
        self.min_dz['dz'] = []  # The step size required by each asm
        self.min_dz['sc'] = []  # Code for limiting subchannel type
        for ai in range(len(self.assemblies)):
            asm = self.assemblies[ai]
            # Calculate minumum dz (based on geometry and flow rate);
            # if min dz is constrained by edge/corner subchannel, use
            # SE2ANL model rather than DASSH model to relax constraint
            dz, sc = dassh.assembly.calculate_min_dz(
                asm, self.inlet_temp, asm._estimated_T_out,
                self._is_adiabatic)

            use_conv_approx = False
            if self._options['conv_approx']:
                if dz < self._options['conv_approx_dz_cutoff']:
                    if asm.has_rodded:
                        if sc[0] in ['2', '3', '6', '7']:
                            use_conv_approx = True
                    else:
                        use_conv_approx = True
            if use_conv_approx:
                dz_old = dz
                msg1 = ('Assembly {:d} mesh size requirement {:s} is '
                        'too small (dz = {:.2e} m);')
                msg2 = ('    Treating duct wall connection with '
                        ' modified approach that yields dz = {:.2e} m.')
                for reg in self.assemblies[ai].region:
                    reg._conv_approx = True
                dz, sc = dassh.assembly.calculate_min_dz(
                    asm, self.inlet_temp, asm._estimated_T_out,
                    self._is_adiabatic)
                self.log('info', msg1.format(asm.id, str(sc), dz_old))
                self.log('info', msg2.format(dz))
            self.min_dz['dz'].append(dz)
            self.min_dz['sc'].append(sc)

    def _calculate_total_fr(self, inp_obj):
        """Calculate core-total flow rate"""
        tot_fr = 0.0
        for a in self.assemblies:
            tot_fr += a.flow_rate
        tot_fr = tot_fr / (1 - inp_obj.data['Core']['bypass_fraction'])
        return tot_fr

    def _setup_core(self, inp_obj):
        """Set up DASSH Core object using GEODST and the parameters from
        each assembly in in the core"""
        # geodst = dassh.py4c.geodst.GEODST(
        #     os.path.join(inp_obj.path, inp_obj.data['ARC']['geodst'][0]))

        # Interassembly gap flow rate
        gap_fr = inp_obj.data['Core']['bypass_fraction'] * self.flow_rate

        # Estimate outlet temperature based on core power
        # print(t_in, self.total_power, total_fr)
        cool_mat = inp_obj.data['Core']['coolant_material'].lower()
        # print(self.total_power, self.inlet_temp, self.flow_rate)
        t_out = dassh.utils.Q_equals_mCdT(self.total_power,
                                          self.inlet_temp,
                                          self.materials[cool_mat],
                                          mfr=self.flow_rate)

        # Instantiate and load core object
        # core_obj = dassh.core.Core(
        #     geodst,
        #     gap_fr,
        #     self.materials[inp_obj.data['Core']['coolant_material'].lower()],
        #     inlet_temperature=self.inlet_temp,
        #     model=inp_obj.data['Core']['gap_model'])
        _asm = np.ones(len(inp_obj.data['Assignment']['ByPosition']))
        _asm *= np.nan
        for a in self.assemblies:
            _asm[a.id] = a.id

        core_obj = dassh.core.Core(
            _asm,
            inp_obj.data['Core']['assembly_pitch'],
            gap_fr,
            self.materials[cool_mat],
            inlet_temperature=self.inlet_temp,
            model=inp_obj.data['Core']['gap_model'])
        core_obj.load(self.assemblies)
        self.core = core_obj

        # Calculate dz required for numerical stability
        dz, sc = dassh.core.calculate_min_dz(
            core_obj, self.inlet_temp, t_out)
        if dz is not None:
            self.min_dz['dz'].append(dz)
            self.min_dz['sc'].append(sc)

        # Precalculate interpolation constants for duct --> gap and
        # gap --> duct for each assembly
        # self._setup_interpolation_params()
        self._setup_gap_mesh_params()

    def _setup_interpolation_params(self):
        """Give each assembly some precalculated constants to speed up
        the quadratic interpolation"""
        for a in self.assemblies:
            if a.has_rodded:
                a.rodded._xparams = {}
                a.rodded._yparams = {}
                # Duct --> Gap
                x = a.rodded.x_pts
                x_new = self.core.x_pts
                idx = dassh.mesh_functions.get_nearest_xy_index(x, x_new)
                a.rodded._xparams['duct2gap'] = \
                    dassh.mesh_functions.calculate_xparams(x, x_new, idx)
                a.rodded._yparams['duct2gap'] = \
                    dassh.mesh_functions.calculate_yparams(x, idx)
                # Gap --> Duct
                x = self.core.x_pts
                x_new = a.rodded.x_pts
                idx = dassh.mesh_functions.get_nearest_xy_index(x, x_new)
                a.rodded._xparams['gap2duct'] = \
                    dassh.mesh_functions.calculate_xparams(x, x_new, idx)
                a.rodded._yparams['gap2duct'] = \
                    dassh.mesh_functions.calculate_yparams(x, idx)

    def _setup_gap_mesh_params(self):
        """Pre-calculate the arrays to go back and forth between axial
        region meshes and inter-assembly gap mesh"""
        for a in range(len(self.assemblies)):
            asm = self.assemblies[a]
            for reg in asm.region:
                # map_fine2coarse, map_coarse2fine = \
                #     dassh.mesh_functions.setup_lin_interp_arrays(
                #         reg.x_pts, self.core.x_pts)
                xb_reg = reg.calculate_xbnds()
                # map_fine2coarse, poop = \
                #     dassh.mesh_functions._map_asm2gap(
                #         xb_reg, self.core.x_bnds)
                # map_fine2coarse, map_coarse2fine = \
                #     dassh.mesh_functions._identify_nearest_neighbors(
                #         reg.x_pts, self.core.x_pts)
                map_fine2coarse, map_coarse2fine = \
                    dassh.mesh_functions._map_asm2gap(
                        xb_reg, self.core._asm_sc_xbnds[a])
                # xb_reg2 = reg.calculate_xbnds2()
                # poop, map_coarse2fine = \
                #     dassh.mesh_functions._map_asm2gap2(
                #         xb_reg2, self.core.x_bnds2)

                reg._map = {}
                reg._map['gap2duct'] = map_fine2coarse
                reg._map['duct2gap'] = map_coarse2fine

    def _setup_overall_axial_mesh_req(self):
        """Evaluate axial mesh size for core and adjust based on user
        request or to ensure numerical accuracy"""
        # Take the minimum dz required; round down a little bit (this
        # just adds some buffer relative to the numerical constraint)
        self.req_dz = np.floor(np.min(self.min_dz['dz']) * 1e6) / 1e6
        self.log('info', f'Axial step size required (m): {self.req_dz}')
        if (self._options['axial_mesh_size'] is not None
                and self._options['axial_mesh_size'] <= self.req_dz):
            self.req_dz = self._options['axial_mesh_size']
            self.log('info', 'Using user-requested axial step '
                             'size (m): {:f}'.format(
                                 self._options["axial_mesh_size"]))
        else:
            if (self._options['axial_mesh_size'] is not None
                    and self._options['axial_mesh_size'] > self.req_dz):
                self.log('info', 'Ignoring user-requested axial step '
                                 'size {:f} m; too large to maintain '
                                 'numerical stability'.format(
                                     self._options["axial_mesh_size"]))
            if self.req_dz > 0.01:
                self.req_dz = 0.01
                self.log('info', 'Reducing step size to improve '
                                 'accuracy; new step size (m): '
                                 f'{self.req_dz}')

    def _melt_warning(self, inp_obj, T_max):
        """Raise error if the user has not provided enough flow to
        the reactor such that extreme temperatures are likely"""
        _MELT_MSG = ('Estimated coolant outlet temperature {0} is '
                     'greater than limit {1}')
        cool_mat = inp_obj.data['Core']['coolant_material'].lower()
        cool_obj = self.materials[cool_mat]
        T_out = dassh.utils.Q_equals_mCdT(self.total_power,
                                          self.inlet_temp,
                                          cool_obj, mfr=self.flow_rate)
        if T_out > T_max:
            self.log('warning', _MELT_MSG.format(T_out, T_max))

    def _setup_zpts(self):
        """Based on calculated dz mesh constraint and axial region
        bounds, determine points to calculate solutions"""
        z = [0.0]
        dz = []
        while z[-1] < self.core_length:
            dz.append(self._check_dz(z[-1]))
            z.append(np.around(z[-1] + dz[-1], 12))
        return np.array(z), np.array(dz)

    def _check_dz(self, z):
        """Make sure that axial step z + dz does not cross any region
        boundaries; if it does, modify dz to meet the boundary plane

        Parameters
        ----------
        z : float
            Axial mesh point

        Returns
        -------
        float
            Axial mesh size step that doesn't cross region boundary

        Notes
        -----
        This should also keep the solution from progressing beyond the
        length of the core, as that should be the last value in the
        region_bounds array.

        """
        z = np.around(z, 12)
        cross_boundary = [z < bi and z + self.req_dz > bi
                          for bi in self.axial_bnds]
        if not any(cross_boundary):
            return self.req_dz
        else:
            crossed_bound = np.where(cross_boundary)[0][0]
            return np.around(self.axial_bnds[crossed_bound] - z, 12)

    def save(self, path=None):
        """Save the Reactor object as a file for later use"""
        if path is None:
            path = self.path

        # Close the open data files
        try:
            self._data_close()
        except (KeyError, AttributeError):  # no open data
            pass

        with open(os.path.join(path, 'dassh_reactor.pkl'), 'wb') as f:
            # cPickle.dump(self, f, cPickle.HIGHEST_PROTOCOL)
            pickle.dump(self, f, protocol=pickle.DEFAULT_PROTOCOL)

    def reset(self):
        """Reset all the temperatures back to the inlet temperature"""
        self.core.coolant_gap_temp *= 0.0
        self.core.coolant_gap_temp += self.inlet_temp
        for i in range(len(self.assemblies)):
            for j in range(len(self.assemblies[i].region)):
                for k in self.assemblies[i].region[j].temp.keys():
                    self.assemblies[i].region[j].temp[k] *= 0.0
                    self.assemblies[i].region[j].temp[k] += \
                        self.inlet_temp

    def _data_setup(self):
        """Set up the data files for the temperature dumps"""
        # If no data dump requested, skip this step
        if not self._options['dump']['any']:
            return

        if self._options['dump']['interval'] is not None:
            self.log('info', 'Dumping temperatures with interval of '
                             '{:f} m and at all requested axial '
                             'positions'.format(
                                 self._options['dump']['interval']))
        else:
            self.log('info', 'Dumping temperatures at every axial step')

        self._options['dump']['dz'] = 0.0

        # Data that we're tracking
        self._options['dump']['names'] = []
        _msg = 'Dumping {:s} temperatures to \"{:s}\"'
        if self._options['dump']['coolant']:
            self._options['dump']['names'].append('coolant_int')
            self.log('info', _msg.format('interior coolant',
                                         'temp_coolant_int.csv'))
            if any([a.rodded.n_bypass > 0 for a in
                    self.assemblies if a.has_rodded]):
                self._options['dump']['names'].append('coolant_byp')
                self.log('info', _msg.format('bypass coolant',
                                             'temp_coolant_byp.csv'))
        if self._options['dump']['duct']:
            self._options['dump']['names'].append('duct_mw')
            self.log('info', _msg.format('duct mid-wall',
                                         'temp_duct_mw.csv'))
        if self._options['dump']['gap']:  # Gap temps on each asm mesh
            self._options['dump']['names'].append('coolant_gap')
            self.log('info', _msg.format('interassembly gap coolant',
                                         'temp_coolant_gap.csv'))
        if self._options['dump']['gap_fine']:  # Gap temps on fine mesh
            self._options['dump']['names'].append('coolant_gap_fine')
            self.log('info', _msg.format(
                'interassembly gap coolant (fine mesh)',
                'temp_coolant_gap_finemesh.csv'))
        if self._options['dump']['pins']:
            self._options['dump']['names'].append('pin')
            self.log('info', _msg.format('pin', 'temp_pin.csv'))
        if self._options['dump']['average']:
            self._options['dump']['names'].append('average')
            self.log('info', _msg.format('average coolant and pin',
                                         'temp_average.csv'))
        if self._options['dump']['maximum']:
            self._options['dump']['names'].append('maximum')
            self.log('info', _msg.format('maximum coolant and pin',
                                         'temp_maximum.csv'))

        # Set up dictionary of paths to data
        self._options['dump']['paths'] = {}
        for f in self._options['dump']['names']:
            name = f'temp_{f}'
            fullname = f'{name}.csv'
            if os.path.exists(os.path.join(self.path, fullname)):
                os.remove(os.path.join(self.path, fullname))
            self._options['dump']['paths'][f] = \
                os.path.join(self.path, fullname)

        # Set up data columns
        self._options['dump']['cols'] = {}
        self._options['dump']['cols']['average'] = 10
        self._options['dump']['cols']['maximum'] = 7
        self._options['dump']['cols']['coolant_int'] = 3 + max(
            [a.rodded.subchannel.n_sc['coolant']['total']
             for a in self.assemblies if a.has_rodded])
        self._options['dump']['cols']['duct_mw'] = 4 + max(
            [a.rodded.subchannel.n_sc['duct']['total']
             if a.has_rodded else 6 for a in self.assemblies])
        self._options['dump']['cols']['coolant_byp'] = 4 + max(
            [a.rodded.subchannel.n_sc['bypass']['total']
             for a in self.assemblies if a.has_rodded])
        self._options['dump']['cols']['coolant_gap'] = \
            self._options['dump']['cols']['duct_mw'] - 1
        # 1 + len(self.core.coolant_gap_temp)
        self._options['dump']['cols']['pin'] = 9

        for a in self.assemblies:
            a.setup_data_io(self._options['dump']['cols'])

    def _data_open(self):
        """Open the data files to which the temperature data will be
        dumped throughout the sweep"""
        if not self._options['dump']['any']:
            return

        self._options['dump']['files'] = {}
        for f in self._options['dump']['names']:
            self._options['dump']['files'][f] = \
                open(self._options['dump']['paths'][f], 'ab')

    def _data_close(self):
        """Close the data files"""
        for k in self._options['dump']['files'].keys():
            self._options['dump']['files'][k].close()
            self._options['dump']['files'][k] = None

    ####################################################################
    # TEMPERATURE SWEEP
    ####################################################################

    def temperature_sweep(self, verbose=False):
        """Sweep axially through the core, solving coolant and duct
        temperatures at each level

        Parameters
        ----------
        verbose (optional) : bool
            Print data from each step during sweep (default False)


        Returns
        -------
        None

        """
        # Open the CSV files to which data is dumped throughout the
        # problem; these are left open and written to at each step
        self._data_setup()
        self._data_open()

        # Initialize duct temperatures in all assemblies
        self.axial_step0()

        # Open workers, if parallel (NOT FUNCTIONAL)
        if self._options['parallel']:
            pool = mp.Pool()

        # Track the time elapsed
        self._starttime = time.time()

        for i in range(1, len(self.z)):
            # Calculate temperatures
            if self._options['parallel']:
                self.axial_step_parallel(self.z[i], self.dz[i - 1], pool)
            else:
                self.axial_step(self.z[i], self.dz[i - 1], verbose)

            # Log progress, if requested
            if self._options['log_progress']:
                self._stepcount += 1
                if self._options['log_interval'] <= self._stepcount:
                    self._print_log_msg(i)

        # Once the sweep is done close the CSV data files, if open
        if self._options['parallel']:
            pool.close()
            pool.join()
        try:
            self._data_close()
        except (AttributeError, KeyError):
            pass

        # write the summary output
        if self._options['write_output']:
            self.write_output_summary()

    def _print_log_msg(self, step):
        """Format the message to log to the screen"""
        # Format plane number and axial position
        s_fmt = str(step).rjust(4)
        z_fmt = '{:.2f}'.format(self.z[step])
        # Format time elapsed
        _end = time.time()
        hours, rem = divmod(_end - self._starttime, 3600)
        minutes, seconds = divmod(rem, 60)
        elapsed = "{:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds)
        # Print message
        msg = (f'Progress: plane {s_fmt} of '
               f'{len(self.dz)}; z = {z_fmt} m; '
               f'cumulative sweep time = {elapsed}')
        self.log('info', msg)
        self._stepcount = 0

    def axial_step(self, z, dz, verbose=False):
        """Solve temperatures at the next axial step

        Parameters
        ----------
        z : float
            Absolute axial position (m)
        dz : float
            Axial mesh size (m)
        verbose (optional) : bool
            Indicate whether to print step summary

        """
        # First, some administrative crap: figure out whether you're
        # dumping temperatures at this axial step
        dump_step = self._determine_whether_to_dump_data(z, dz)

        # 1. Calculate gap coolant temperatures at the j+1 level
        #    based on duct wall temperatures at the j level.
        if self.core.model is not None:
            # [interpolate_temps(a.x_pts,
            #                    a.duct_outer_surf_temp,
            #                    self.core.x_pts)
            #  for a in self.assemblies])
            # t_duct = np.array(
            #     [dassh.mesh_functions.interpolate_quad(
            #         a.x_pts,
            #         a.duct_outer_surf_temp,
            #         self.core.x_pts,
            #         xparams=a.xparams['duct2gap'],
            #         yparams=a.yparams['duct2gap'])
            #      for a in self.assemblies])
            # t_duct = np.array(
            #     [dassh.mesh_functions.interpolate_lin2(
            #         a.x_pts,
            #         a.duct_outer_surf_temp,
            #         self.core.x_pts)
            #      for a in self.assemblies])
            # asm_to_replace = [128, 129, 169, 172, 216,
            #                   170, 171, 217, 220, 270]
            # side_to_replace = np.array([[1, 2, 3],
            #                             [2, 0, 0],
            #                             [2, 0, 0],
            #                             [1, 2, 3],
            #                             [1, 2, 3],
            #                             [5, 0, 0],
            #                             [4, 5, 6],
            #                             [4, 5, 6],
            #                             [5, 0, 0],
            #                             [5, 0, 0]])
            t_duct = np.array(
                [dassh.mesh_functions.map_across_gap(
                    a.duct_outer_surf_temp,
                    a.active_region._map['duct2gap'])
                 for a in self.assemblies])
            self.core.calculate_gap_temperatures(dz, t_duct)
            # Dump gap temperatures
            if dump_step and self._options['dump']['gap_fine']:
                to_write = np.zeros(
                    (1, self.core.coolant_gap_temp.shape[0] + 1))
                to_write[0, 0] = z
                to_write[0, 1:] = self.core.coolant_gap_temp
                np.savetxt(
                    self._options['dump']['files']['coolant_gap_fine'],
                    to_write,
                    delimiter=',')

        # 2. Calculate assembly coolant and duct temperatures.
        #    Different treatment depending on whether in the
        #    heterogeneous or homogeneous region; varies assembly to
        #    assembly, handled in the same method.
        #    (a) Heterogeneous region:
        #        - Calculate assembly coolant temperatures at the j+1
        #          level based on coolant and duct wall temepratures
        #          at the j level
        #        - Calculate assembly duct wall temperatures at the j+1
        #          level based on assembly and gap coolant temperatures
        #          at the j+1 level
        #    (b) Homogeneous region (porous media)
        #        - Calculate assembly coolant and duct temepratures at
        #          the j+1 level based on temperatures at the j level
        for i in range(len(self.assemblies)):
            self._calculate_asm_temperatures(self.assemblies[i], i, z,
                                             dz, dump_step)

        if verbose:
            print(self._print_step_summary(z, dz))

    def axial_step0(self):
        """Update duct temperatures prior to sweep based on inlet
        coolant temperatures"""
        if self.core.model is None:
            pass
        else:
            for i in range(len(self.assemblies)):
                # gap_adj_temps = interpolate_temps(
                #     self.core.x_pts,
                #     self.core.adjacent_coolant_gap_temp(i),
                #     self.assemblies[i].x_pts)
                # gap_temp = dassh.mesh_functions.interpolate_quad(
                #     self.core.x_pts,
                #     self.core.adjacent_coolant_gap_temp(i),
                #     self.assemblies[i].x_pts,
                #     xparams=self.assemblies[i].xparams['gap2duct'],
                #     yparams=self.assemblies[i].yparams['gap2duct'])
                # gap_htc = self.core.coolant_gap_params['htc']
                # gap_adj_temps = dassh.mesh_functions.map_across_gap(
                #     self.core.adjacent_coolant_gap_temp(i),
                #     self.assemblies[i].active_region._map['gap2duct'])
                gap_htc = dassh.mesh_functions.map_across_gap(
                    self.core.adjacent_coolant_gap_htc(i),
                    self.assemblies[i].active_region._map['gap2duct'])
                gap_temp = dassh.mesh_functions.map_across_gap(
                    (self.core.adjacent_coolant_gap_htc(i)
                     * self.core.adjacent_coolant_gap_temp(i)),
                    self.assemblies[i].active_region._map['gap2duct'])
                gap_temp /= gap_htc
                self.assemblies[i].step0(gap_temp,
                                         gap_htc,
                                         self._is_adiabatic)

    def axial_step_parallel(self, z, dz, worker_pool):
        """Parallelized version of axial_step

        Parameters
        ----------
        z : float
            Absolute axial position (m)
        dz : float
            Axial mesh size (m)
        verbose (optional) : bool
            Indicate whether to print step summary
        worker_pool : multiprocessing Pool object
            Workers to perform parallel tasks

        """
        # First, some administrative crap: figure out whether you're
        # dumping temperatures at this axial step
        dump_step = self._determine_whether_to_dump_data(z, dz)

        # 1. Calculate gap coolant temperatures at the j+1 level
        #    based on duct wall temperatures at the j level.
        if self.core.model is not None:
            # worker_pool = mp.Pool()
            t_duct = []
            for asm in self.assemblies:
                t_duct.append(
                    worker_pool.apply_async(
                        approximate_temps,
                        args=(asm.x_pts,
                              asm.duct_outer_surf_temp,
                              self.core.x_pts,
                              asm._lstsq_params, )
                    )
                )
            t_duct = np.array([td.get() for td in t_duct])
            self.core.calculate_gap_temperatures(dz, t_duct)
            # worker_pool.close()
            # worker_pool.join()
        # 2. Calculate assembly coolant and duct temperatures.
        #    Different treatment depending on whether in the
        #    heterogeneous or homogeneous region; varies assembly to
        #    assembly, handled in the same method.
        updated_asm = []
        # worker_pool = mp.Pool()
        for asm in self.assemblies:
            updated_asm.append(
                worker_pool.apply_async(
                    self._calculate_asm_temperatures,
                    args=(asm, z, dz, dump_step, ),
                    error_callback=err_cb))
        self.assemblies = [a.get() for a in updated_asm]
        # worker_pool.close()
        # worker_pool.join()

        # Write the results
        if dump_step:
            for asm in self.assemblies:
                asm.write(self._options['dump']['files'], None)

    def _determine_whether_to_dump_data(self, z, dz):
        """Dump data to CSV if interval length is reached or if at
        an axial region boundary"""
        if self._options['dump']['any']:
            self._options['dump']['dz'] += dz
            # No interval given; dump at every step
            if self._options['dump']['interval'] is None:
                dump_step = True
            # Interval provided, and we need to write data and reset
            elif (np.around(self._options['dump']['dz'], 9)
                    >= self._options['dump']['interval']):
                dump_step = True
                self._options['dump']['dz'] = 0.0
            # Axial plane requested by user; write data, no reset
            elif z in self.axial_bnds:
                dump_step = True
            # Don't do anything
            else:
                dump_step = False
        else:
            dump_step = False
        return dump_step

    def _calculate_asm_temperatures(self, asm, i, z, dz, dump_step):
        """Calculate assembly coolant and duct temperatures"""
        # Update the region if necessary
        asm.check_region_update(z)
        # Find and approximate gap temperatures next to each asm
        # gap_adj_temps = interpolate_temps(
        #     self.core.x_pts,
        #     self.core.adjacent_coolant_gap_temp(i),
        #     asm.x_pts)
        if self.core.model is None:
            gap_temp = np.ones(asm.duct_outer_surf_temp.shape[0])
            gap_htc = np.ones(asm.duct_outer_surf_temp.shape[0])
        elif asm.active_region.is_rodded:
            # gap_temp = dassh.mesh_functions.interpolate_quad(
            #     self.core.x_pts,
            #     self.core.adjacent_coolant_gap_temp(i),
            #     asm.x_pts,
            #     xparams=asm.xparams['gap2duct'],
            #     yparams=asm.yparams['gap2duct'])
            # gap_temp = dassh.mesh_functions.interpolate_lin2(
            #     self.core.x_pts,
            #     self.core.adjacent_coolant_gap_temp(i),
            #     asm.x_pts)
            gap_htc = dassh.mesh_functions.map_across_gap(
                self.core.adjacent_coolant_gap_htc(i),
                asm.active_region._map['gap2duct'])
            gap_temp = dassh.mesh_functions.map_across_gap(
                (self.core.adjacent_coolant_gap_temp(i)
                 * self.core.adjacent_coolant_gap_htc(i)),
                asm.active_region._map['gap2duct'])
            gap_temp = gap_temp / gap_htc
        else:
            gap_htc = dassh.mesh_functions.map_across_gap(
                self.core.adjacent_coolant_gap_htc(i),
                asm.active_region._map['gap2duct'])
            gap_temp = dassh.mesh_functions.map_across_gap(
                self.core.adjacent_coolant_gap_temp(i),
                asm.active_region._map['gap2duct']) / gap_htc
            # poop = self.core.adjacent_coolant_gap_temp(i).reshape(6, -1)
            # gap_temp = gap_temp.reshape(6, -1)
            # gap_temp[:, -1] = poop[:, -1]
            # gap_temp = gap_temp.flatten()
            # gap_adj_temps = map_across_gap(
            #     self.core.adjacent_coolant_gap_temp(i),
            #     asm.active_region._map['gap2duct'])
        asm.calculate(z, dz, gap_temp, gap_htc,
                      self._is_adiabatic, self._options['ebal'])
        # Write the results
        if dump_step:
            asm.write(self._options['dump']['files'], gap_temp)
        return asm

    def _print_step_summary(self, z, dz):
        """Print some stuff about assembly power and coolant
        and duct temperatures at the present axial level"""
        to_print = []
        to_print.append(z)
        to_print.append(dz)
        for asm in self.assemblies:
            p = asm.power.get_power(z)
            total_power = 0.0
            for k in p.keys():
                if p[k] is not None:
                    total_power += np.sum(p[k])
            to_print.append(total_power * dz)
            to_print.append(asm.avg_coolant_temp)
            to_print += list(asm.avg_duct_mw_temp)
        to_print.append(self.core.avg_coolant_gap_temp)
        return ' '.join(['{:.10e}'.format(v) for v in to_print])

    ####################################################################
    # WRITE OUTPUT
    ####################################################################

    def write_summary(self):
        """Write the main DASSH output file"""
        # Output file preamble
        out = 'DASSH: Ducted Assembly Steady-State Heat Transfer Code\n'
        out += f'Version {dassh.__version__}\n'
        out += f'Executed {str(datetime.datetime.now())}\n'

        # Geometry summary
        geom = dassh.table.GeometrySummaryTable(len(self.asm_templates))
        out += geom.generate(self)

        # Power summary
        power = dassh.table.PositionAssignmentTable()
        out += power.generate(self)

        # Flow summary
        flow = dassh.table.CoolantFlowTable()
        out += flow.generate(self)

        # Write to output file
        with open(os.path.join(self.path, 'dassh.out'), 'w') as f:
            f.write(out)

    def write_output_summary(self):
        """Write the main DASSH output file"""
        out = ''

        # Pressure drop
        dp = dassh.table.PressureDropTable(
            max([len(a.region) for a in self.assemblies]))
        out += dp.generate(self)

        # Energy balances
        asm_ebal_table = dassh.table.AssemblyEnergyBalanceTable()
        out += asm_ebal_table.generate(self)

        # Core energy balance
        # if self._options['ebal']:
        #     interasm_ht_table = dassh.table.InterasmEnergyXferTable()
        #     out += interasm_ht_table.generate(self)

        #     asm_ebal_table = dassh.table.AssemblyEnergyBalanceTable()
        #     out += asm_ebal_table.generate(self)
        #
        #     # Core energy balance
        #     core_ebal_table = dassh.table.CoreEnergyBalanceTable()
        #     out += core_ebal_table.generate(self)

        # Coolant temperatures
        coolant_table = dassh.table.CoolantTempTable()
        out += coolant_table.generate(self)

        # Duct temperatures
        duct_table = dassh.table.DuctTempTable()
        out += duct_table.generate(self)

        # Peak pin temperatures
        if any(['pin' in a._peak.keys() for a in self.assemblies]):
            max_temps = dassh.table.PeakPinTempTable()
            out += max_temps.generate(self, 'clad', 'mw')
            out += max_temps.generate(self, 'fuel', 'cl')

        # Append to file
        with open(os.path.join(self.path, 'dassh.out'), 'a') as f:
            f.write(out)

########################################################################


def get_rod_bundle_bnds(zfm, asm_data):
    """Determine axial bounds of Assembly rod bundle

    Parameters
    ----------
    zfm : numpy.ndarray
        Axial fine mesh points for the assembly power distribution
    asm_data : dict
        Dictionary describing assembly geometry

    Returns
    -------
    List
        Rod bundle lower and upper axial bounds (cm)

    """
    if asm_data.get('use_low_fidelity_model'):
        bundle_zbnd = [100 * zfm[-1], 100 * zfm[-1]]
    else:
        assert 'rods' in asm_data['AxialRegion'].keys()
        bundle_zbnd = [100 * asm_data['AxialRegion']['rods']['z_lo'],
                       100 * asm_data['AxialRegion']['rods']['z_hi']]
    return bundle_zbnd


def match_rodded_finemesh_bnds_dif3d(power_obj, asm_data):
    """Determine bounds of core rodded region"""
    if asm_data.get('use_low_fidelity_model'):
        ck_rod_bnds = [len(power_obj.z_mesh),
                       len(power_obj.z_mesh)]
    else:
        if 'rods' in asm_data['AxialRegion'].keys():
            ck_rod_bnds = [0, 0]
            ck_rod_bnds[0] = np.where(np.isclose(
                power_obj.z_mesh,
                100 * asm_data['AxialRegion']['rods']['z_lo']))[0][0]
            ck_rod_bnds[1] = np.where(np.isclose(
                power_obj.z_mesh,
                100 * asm_data['AxialRegion']['rods']['z_hi']))[0][0]
    k_bnds = [sum(power_obj.k_fints[:ck_rod_bnds[0]]),
              sum(power_obj.k_fints[:ck_rod_bnds[1]])]
    return k_bnds


def match_rodded_finemesh_bnds(zfm, asm_data):
    """Do it for user-specified power"""
    if asm_data.get('use_low_fidelity_model'):
        kbnds = [len(zfm), len(zfm)]
    else:
        assert 'rods' in asm_data['AxialRegion'].keys()
        kbnds = [0, 0]
        zlo = asm_data['AxialRegion']['rods']['z_lo']
        zhi = asm_data['AxialRegion']['rods']['z_hi']
        kbnds[0] = np.where(np.isclose(zfm, 100 * zlo))[0][0]
        kbnds[1] = np.where(np.isclose(zfm, 100 * zhi))[0][0]
    return kbnds


def calc_power_VARIANT(input_data, working_dir, t_pt=0):
    """Calculate the power distributions from VARIANT

    Parameters
    ----------
    data : dict
        DASSH input data dictionary
    working_dir : str
        Path to current working directory

    Returns
    -------
    dict
        DASSH Power objects for each type of assembly in the problem;
        different objects are required because different assemblies
        can have different unrodded region specifications

    """
    cwd = os.getcwd()
    if working_dir != '':
        os.chdir(working_dir)

    # Identify VARPOW keys for fuel and coolant
    fuel_id = _FUELS[input_data['Core']['fuel_material'].lower()]
    if type(fuel_id) == dict:
        fuel_id = fuel_id[input_data['Core']['fuel_alloy'].lower()]

    coolant_heating = input_data['Core']['coolant_heating']
    if coolant_heating is None:
        coolant_heating = input_data['Core']['coolant_material']
    if coolant_heating.lower() not in _COOLANTS.keys():
        module_logger.error('Unknown coolant specification for '
                            'heating calculation; must choose '
                            'from options: Na, NaK, Pb, Pb-Bi')
    else:
        cool_id = _COOLANTS[coolant_heating.lower()]

    # Run VARPOW, rename output files
    path2varpow = os.path.dirname(os.path.abspath(__file__))
    if sys.platform == 'darwin':
        path2varpow = os.path.join(path2varpow, 'varpow_osx.x')
    elif 'linux' in sys.platform:
        path2varpow = os.path.join(path2varpow, 'varpow_linux.x')
    else:
        raise SystemError('DASSH currently supports only Linux and OSX')
    with open('varpow_stdout.txt', 'w') as f:
        subprocess.call([path2varpow,
                         str(fuel_id),
                         str(cool_id),
                         input_data['ARC']['pmatrx'][t_pt],
                         input_data['ARC']['geodst'][t_pt],
                         input_data['ARC']['ndxsrf'][t_pt],
                         input_data['ARC']['znatdn'][t_pt],
                         input_data['ARC']['nhflux'][t_pt],
                         input_data['ARC']['ghflux'][t_pt]],
                        stdout=f)
    subprocess.call(['mv', 'MaterialPower.out',
                     'varpow_MatPower.out'])
    subprocess.call(['mv', 'VariantMonoExponents.out',
                     'varpow_MonoExp.out'])
    subprocess.call(['mv', 'Output.VARPOW', 'VARPOW.out'])

    os.chdir(cwd)
    return import_power_VARIANT(input_data, working_dir, t_pt)


def import_power_VARIANT(data, w_dir, t_pt=0):
    """Import power distributions from VARIANT

    Parameters
    ----------
    data : dict
        DASSH input data dictionary
    w_dir : str
        Path to current working directory
    t_pt (optional) : int
        If multiple CCCC file sets provided, indicate which to use
        (default = 0)

    Returns
    -------
    dict
        DASSH Power objects for each type of assembly in the problem;
        different objects are required because different assemblies
        can have different unrodded region specifications

    """
    # Create DASSH Power object
    # core_power = dassh.power.Power(
    #     os.path.join(w_dir, 'varpow_MatPower.out'),
    #     os.path.join(w_dir, 'varpow_MonoExp.out'),
    #     os.path.join(w_dir, 'VARPOW.out'),
    #     os.path.join(w_dir, data['ARC']['geodst'][t_pt]),
    #     user_power=data['Core']['total_power'],
    #     scalar=data['Core']['power_scaling_factor'],
    #     model=data['Core']['power_model'])
    core_power = dassh.power.Power(
        os.path.join(w_dir, 'varpow_MatPower.out'),
        os.path.join(w_dir, 'varpow_MonoExp.out'),
        os.path.join(w_dir, 'VARPOW.out'),
        os.path.join(w_dir, data['ARC']['geodst'][t_pt]),
        model=data['Core']['power_model'])

    # Raise negative power warning
    # negative_power = list(core_power.values())[0].negative_power
    negative_power = core_power.negative_power
    if negative_power < 0.0:  # Note: level 30 is "warning"
        module_logger.log(30, 'Negative powers found and set equal '
                              'to zero. Check flux solution for '
                              'convergence.')
        module_logger.log(30, 'Total negative power (W): '
                              + '{:0.3e}'.format(negative_power))

    return core_power
#
#
# def map_across_gap(vector_in, map):
#     """Map values across assembly/gap mesh disagreements
#
#     Parameters
#     ----------
#     t_in : numpy.ndarray
#         Temperatures on the original basis
#     map : numpy.ndarray
#         Map to convert temperatures to the new basis
#
#     Returns
#     -------
#     numpy.ndarray
#         Temperatures on the new basis
#
#     """
#     # If no mapping required, just return the input temperatures
#     if vector_in.shape[0] == map.shape[0]:
#         return vector_in
#     else:
#         return np.dot(map, vector_in)
#
#
# def _map_asm2gap(xb_reg, xb_core, normalize=True):
#     """Make maps to convert between axial region radial mesh and radial
#     mesh of the inter-assembly gap"""
#     mapping_f2c = np.zeros((xb_reg.shape[0] - 1, xb_core.shape[0] - 1))
#     # CME: Coarse mesh element
#     # FME: Fine mesh element
#     # LBND/UBND: low-bound / upper-bound
#     for CME in range(mapping_f2c.shape[0]):
#         CME_LBND = xb_reg[CME]
#         CME_UBND = xb_reg[CME + 1]
#         FME = np.searchsorted(xb_core, xb_reg[CME]) - 1
#         FME_UBND = xb_core[FME + 1]
#         mapping_f2c[CME, FME] = min([CME_UBND - CME_LBND,
#                                      FME_UBND - CME_LBND])
#         while FME_UBND < CME_UBND:
#             FME += 1
#             FME_LBND = xb_core[FME]
#             FME_UBND = xb_core[FME + 1]
#             mapping_f2c[CME, FME] = min([CME_UBND - FME_LBND,
#                                          FME_UBND - FME_LBND])
#     # Normalize
#     dx_reg = xb_reg[1:] - xb_reg[:-1]
#     m_f2c = (mapping_f2c.T / dx_reg).T
#     dx_core = xb_core[1:] - xb_core[:-1]
#     m_c2f = (mapping_f2c / dx_core).T
#     # Combine half-corner and trim array so first entries correspond
#     # to first edge channels
#     m_f2c[-1, :] += m_f2c[0, :]
#     m_f2c[:, -1] += m_f2c[:, 0]
#     m_f2c[-1] *= 0.5
#     m_f2c = m_f2c[1:, 1:]
#     m_c2f[-1, :] += m_c2f[0, :]
#     m_c2f[:, -1] += m_c2f[:, 0]
#     m_c2f[-1] *= 0.5
#     m_c2f = m_c2f[1:, 1:]
#     return m_f2c, m_c2f
#
#
# def _map_asm2gap2(xb_reg, xb_core):
#     """x"""
#     mapping_f2c = np.zeros((xb_reg.shape[0] - 1, xb_core.shape[0] - 1))
#     # CME: Coarse mesh element
#     # FME: Fine mesh element
#     # LBND/UBND: element low-bound / upper-bound
#     for CME in range(mapping_f2c.shape[0]):
#         CME_LBND = xb_reg[CME]
#         CME_UBND = xb_reg[CME + 1]
#         FME = np.searchsorted(xb_core, xb_reg[CME]) - 1
#         FME_UBND = xb_core[FME + 1]
#         mapping_f2c[CME, FME] = min([CME_UBND - CME_LBND,
#                                      FME_UBND - CME_LBND])
#         while FME_UBND < CME_UBND:
#             FME += 1
#             FME_LBND = xb_core[FME]
#             FME_UBND = xb_core[FME + 1]
#             mapping_f2c[CME, FME] = min([CME_UBND - FME_LBND,
#                                          FME_UBND - FME_LBND])
#     # Normalize
#     dx_reg = xb_reg[1:] - xb_reg[:-1]
#     m_f2c = (mapping_f2c.T / dx_reg).T
#     dx_core = xb_core[1:] - xb_core[:-1]
#     m_c2f = (mapping_f2c / dx_core).T
#
#     # Expand to include corner duct <--> gap mapping
#     m_f2c2 = np.zeros((m_f2c.shape[0] + 1, m_f2c.shape[1] + 1))
#     m_f2c2[:-1, :-1] = m_f2c
#     m_f2c2[-1, -1] = 1
#
#     # Expand to all 6 sides
#     poop_f2c = np.zeros((6 * m_f2c2.shape[0], 6 * m_f2c2.shape[1]))
#     inds_to_fill = np.zeros((6, 2), dtype=int)
#     inds_to_fill[:, 0] = np.arange(0, poop_f2c.shape[0], m_f2c2.shape[0])
#     inds_to_fill[:, 1] = np.arange(0, poop_f2c.shape[1], m_f2c2.shape[1])
#     for inds in inds_to_fill:
#         ix2 = inds[0] + m_f2c2.shape[0]
#         iy2 = inds[1] + m_f2c2.shape[1]
#         poop_f2c[inds[0]:ix2, :][:, inds[1]:iy2] = m_f2c2
#
#     # Do the same for the other one
#     # Expand to include corner duct <--> gap mapping
#     m_c2f2 = np.zeros((m_c2f.shape[0] + 1, m_c2f.shape[1] + 1))
#     m_c2f2[:-1, :-1] = m_c2f
#     m_c2f2[-1, -1] = 1
#
#     # Expand to all 6 sides
#     poop_c2f = np.zeros((m_c2f2.shape[0] * 6, m_c2f2.shape[1] * 6))
#     inds_to_fill = np.zeros((6, 2), dtype=int)
#     inds_to_fill[:, 0] = np.arange(0, poop_c2f.shape[0], m_c2f2.shape[0])
#     inds_to_fill[:, 1] = np.arange(0, poop_c2f.shape[1], m_c2f2.shape[1])
#     for inds in inds_to_fill:
#         ix2 = inds[0] + m_c2f2.shape[0]
#         iy2 = inds[1] + m_c2f2.shape[1]
#         poop_c2f[inds[0]:ix2, :][:, inds[1]:iy2] = m_c2f2
#
#     return poop_f2c, poop_c2f

#
# def approximate_temps(x, y, x_new, asm_lstsq_params=None, order=2):
#     """Approximate a vector of temperatures to a coarser or finer mesh
#
#     Parameters
#     ----------
#     x : numpy.ndarray
#         The positions of the original mesh centroids along a hex
#         side; length must be greater than 1
#     y : numpy.ndarray
#         The original temperatures at those centroids; length must
#         be greater than 1
#     x_new : numpy.ndarray
#         The positions of the new mesh centroids to which the new
#         temperatures will be approximated
#     asm_lstsq_params : dict (optional)
#         If provided, can bypass the setup portions of the Legendre
#         polynomial fit to data
#     order : int (optional)
#         Order of the Legendre polynomial basis (default = 2)
#
#     Returns
#     -------
#     numpy.ndarray
#         The approximated temperatures at positions x_new
#
#     Notes
#     -----
#     Used in DASSH to deal with mesh disagreement in the interassembly
#     gap between assemblies with different number of pins
#
#     """
#     # If no interpolation needed, just return the original array
#     if np.array_equal(x, x_new):
#         return y
#
#     # Otherwise...
#     # Dress up the temperatures for the interpolation
#     ym = copy.deepcopy(y)
#     ym.shape = (6, int(ym.shape[0] / 6))
#     to_append = np.roll(ym[:, -1], 1)
#     to_append.shape = (6, 1)
#     ym = np.hstack((to_append, ym))
#
#     # If len(x_old) == 2 (only corners): No need for legendre fit!
#     # The approximation is just a linear fit between the two corners
#     if len(x) == 2:
#         y_new = np.linspace(ym[:, 0], ym[:, 1], len(x_new))
#         y_new = y_new.transpose()
#
#     # If len(x_new) == 2 (only corners): No need for legendre fit!
#     # Can just return the corner temperatures and be done with it.
#     elif len(x_new) == 2:
#         y_new = ym[:, (0, -1)]
#
#     # Otherwise, bummer: you have to fit polynomial and generate
#     # approximate values on the new mesh x points
#     else:
#         # y_new = np.zeros((6, len(x_new)))
#         # for side in range(6):
#         #     coeff = np.polynomial.legendre.legfit(x, ym[side], order)
#         #     y_new[side] = np.polynomial.legendre.legval(x_new, coeff)
#
#         if asm_lstsq_params is None:
#             coeff = np.polynomial.legendre.legfit(x, ym.transpose(), 2)
#         else:
#             c, resids, rank, s = np.linalg.lstsq(
#                 asm_lstsq_params['lhs_over_scl'],
#                 ym.T,
#                 asm_lstsq_params['rcond'])
#             coeff = (c.T / asm_lstsq_params['scl']).T
#         y_new = np.polynomial.legendre.legval(x_new, coeff)
#
#         # Use the exact corner temps from original array
#         y_new[:, -1] = ym[:, -1]
#
#     # Get rid of the stuff you added and return the flattened array
#     y_new = y_new[:, 1:]
#     return y_new.flatten()

#
# def interpolate_temps(x, y, x_new, lstsq_params=None):
#     """Approximate a vector of temperatures to a coarser or finer mesh
#
#     Parameters
#     ----------
#     x : numpy.ndarray
#         The positions of the original mesh centroids along a hex
#         side; length must be greater than 1
#     y : numpy.ndarray
#         The original temperatures at those centroids; length must
#         be greater than 1
#     x_new : numpy.ndarray
#         The positions of the new mesh centroids to which the new
#         temperatures will be approximated
#     lstsq_params : dict (optional)
#         If provided, can bypass the setup portions of the
#         interpolation fit to data
#
#     Returns
#     -------
#     numpy.ndarray
#         The interpolated temperatures at positions x_new
#
#     Notes
#     -----
#     Used in DASSH to deal with mesh disagreement in the interassembly
#     gap between assemblies with different number of pins
#
#     """
#     # If no interpolation needed, just return the original array
#     if np.array_equal(x, x_new):
#         return y
#
#     # Otherwise...
#     # Dress up the temperatures for the interpolation
#     ym = copy.deepcopy(y)
#     ym.shape = (6, int(ym.shape[0] / 6))
#     to_append = np.roll(ym[:, -1], 1)
#     to_append.shape = (6, 1)
#     ym = np.hstack((to_append, ym))
#
#     # If len(x_old) == 2 (only corners): No need to call interpolation
#     # fxn - the interpolation is just a linear fit between two corners
#     if len(x) == 2:
#         y_new = np.linspace(ym[:, 0], ym[:, 1], len(x_new))
#         y_new = y_new.transpose()
#
#     # If len(x_new) == 2 (only corners): No need for legendre fit!
#     # Can just return the corner temperatures and be done with it.
#     elif len(x_new) == 2:
#         y_new = ym[:, (0, -1)]
#
#     # Otherwise, bummer: you have to do the linear interpolation to
#     # get values on the new mesh x points
#     else:
#         y_new = np.zeros((6, len(x_new)))
#         for i in range(6):
#             y_new[i] = np.interp(x_new, x, ym[i])
#
#         # Use the exact corner temps from original array
#         y_new[:, -1] = ym[:, -1]
#
#     # Get rid of the stuff you added and return the flattened array
#     y_new = y_new[:, 1:]
#     return y_new.flatten()

#
# def interpolate_temps(x, y, x_new, xparams, yparams):
#     """Approximate a vector of temperatures to a coarser or finer mesh
#
#     Parameters
#     ----------
#     x : numpy.ndarray
#         The positions of the original mesh centroids along a hex
#         side; length must be greater than 1
#     y : numpy.ndarray
#         The original temperatures at those centroids; length must
#         be greater than 1
#     x_new : numpy.ndarray
#         The positions of the new mesh centroids to which the new
#         temperatures will be approximated
#     xparams : numpy.ndarray (optional)
#         If provided, can bypass the setup portions of the
#         quadratic interpolation fit to data
#     yparams : numpy.ndarray (optional)
#         Indices with which to slice y-data to shortcut some of the
#         quadratic interpolation setup
#
#     Returns
#     -------
#     numpy.ndarray
#         The interpolated temperatures at positions x_new
#
#     Notes
#     -----
#     Used in DASSH to deal with mesh disagreement in the interassembly
#     gap between assemblies with different number of pins
#
#     """
#     # return interpolation.interpolate_lin(x, y, x_new)
#     return interpolation.interpolate_lin(x, y, x_new, xparams, yparams)


def err_cb(error):
    raise error

########################################################################