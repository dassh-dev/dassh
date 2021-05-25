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
date: 2021-05-20
author: matz
Methods to describe the components of hexagonal fuel typical of liquid
metal fast reactors.
"""
########################################################################
import copy
import bisect
import numpy as np
import logging
from dassh.logged_class import LoggedClass
from dassh import region_rodded
from dassh import region_unrodded
from dassh import mesh_functions


_sqrt3 = np.sqrt(3)
_inv_sqrt3 = 1 / _sqrt3
_sqrt3over3 = np.sqrt(3) / 3
# Surface of pins in contact with each type of subchannel
q_p2sc = {1: 0.166666666666667, 2: 0.25, 3: 0.166666666666667}
module_logger = logging.getLogger('dassh.assembly')


class Assembly(LoggedClass):
    """DASSH assembly object

    Parameters
    ----------
    name : str
        Assembly type
    loc : tuple
        (Ring, position) location in the core (python indexing)
    rod_bundle_params : dict
        Parameters that describe the rod bundle (from DASSH_Input obj)
    mat_dict : dict
        Coolant, duct, structure, and fcgap materials used in the assembly
    inlet_temp : float
        Inlet temperature of the coolant entering the assembly (K)
    flow_rate : float
        User-specified bulk mass flow rate (kg/s) in the assembly
    origin : tuple
        X-Y coordinates of the assembly centroid
    se2geo : bool
        Indicate whether to use DASSH or SE2 bundle geometry definitions
        (use only when comparing DASSH and SE2)
    """

    def __init__(self, name, loc, asm_input, mat_dict, inlet_temp,
                 flow_rate, origin=(0.0, 0.0), se2geo=False):
        """Instantiate Assembly object."""
        # Instantiate Logger
        LoggedClass.__init__(self, 4, 'dassh.Assembly')

        # Assembly attributes
        self._name = name
        self._loc = loc
        if loc[0] == 0:
            self._id = 0
        else:
            self._id = 3 * (loc[0] - 1) * loc[0] + loc[1] + 1
        self._pressure_drop = 0.0
        self._z = 0.0
        self._active_region_idx = 0
        self.flow_rate = flow_rate
        self.duct_oftf = max(asm_input['duct_ftf'])

        # Create the rod bundle region - if totally unrodded, make an
        # unrodded region using the bundle parameters
        if asm_input.get('use_low_fidelity_model'):
            self.region = [region_unrodded.make_ur_asm(
                asm_input, mat_dict, flow_rate, se2geo)]
        else:
            self.region = [region_rodded.make_rr_asm(
                asm_input, self.name, mat_dict, flow_rate, se2geo)]

        # Create other requested unrodded regions
        for reg in asm_input['AxialRegion']:
            if reg != 'rods':
                self.region.append(
                    region_unrodded.make_ur_axialregion(
                        asm_input, reg, mat_dict, flow_rate))

        # Keep track of what regions are where - sort them
        sorted_region = []
        self.region_idx = []
        self.region_bnd = []
        z0 = 0.0
        for i in range(len(self.region)):
            for j in range(len(self.region)):
                if self.region[j].z[0] == z0:
                    sorted_region.append(self.region[j])
                    self.region_idx.append(i)
                    break
            # update z0
            self.region_bnd.append(z0)
            z0 = self.region[j].z[1]

        # Determine which region is the rodded region
        self._rodded_idx = None
        if self.region[0].is_rodded:
            self._rodded_idx = sorted_region.index(self.region[0])
        self.region = sorted_region

        # Activate first region manually
        self.region[0].coolant = mat_dict['coolant']
        self.region[0].coolant.update(inlet_temp)
        self.region[0].duct = mat_dict['duct']
        self.region[0].duct.update(inlet_temp)
        for key in self.region[0].temp:
            self.region[0].temp[key] *= inlet_temp

        # Get maximum x_pts (no need to do this as a property, it only
        # applies to one region and is only used once in core.load)
        self._finest_xpts = self.region[0].x_pts
        for i in range(1, len(self.region)):
            if len(self.region[i].x_pts) > len(self._finest_xpts):
                self._finest_xpts = self.region[i].x_pts

        # Save and update peak coolant, duct, and pin temperatures
        # throughout the sweep rather than reading them in at the end
        self._peak = {}
        self._peak['cool'] = (0.0, 0.0)
        self._peak['duct'] = [(0.0, 0.0) for i in range(
            max([len(reg.duct_ftf) if reg.is_rodded else 1
                 for reg in self.region]))]

        # self._peak['cool'] = 0.0
        # self._peak['duct'] = 0.0

        # Pin peak temperatures: want radial profile so need to store
        # a bit more stuff in order to seek out the right value
        if 'FuelModel' in asm_input.keys():
            self._peak['pin'] = {}
            keys = ['clad_od', 'clad_mw', 'clad_id',
                    'fuel_od', 'fuel_cl']
            for i in range(len(keys)):
                # Items in the list:
                # (1) the peak temperature value
                # (2) column in the pin temp array to go look it up
                # (3) radial pin temp data at height of the peak temp
                self._peak['pin'][keys[i]] = [0.0, i + 4, []]

        # Energy balance attributes: track total power added
        self._power_delivered = {
            'pins': 0.0,
            'duct': 0.0,
            'cool': 0.0,
            'refl': 0.0}

    ####################################################################
    # ASSEMBLY INSTANCE GENERAL METHODS
    ####################################################################

    def clone(self, new_loc, new_flowrate=None):
        """Create a clone of an assembly at a new core position with
        a new coolant mass flow rate"""

        # Create a shallow copy (stuff like the pin and subchannel
        # objects can simply be pointed to)
        clone = copy.copy(self)
        clone._loc = new_loc
        if new_loc[0] == 0:
            new_id = 0
        else:
            new_id = 3 * (new_loc[0] - 1) * new_loc[0] + new_loc[1] + 1
        clone._id = new_id

        if new_flowrate is not None:
            clone.flow_rate = new_flowrate

        new_regs = []
        for ri in range(len(self.region)):
            new_regs.append(self.region[ri].clone(new_flowrate))

        # Update pin temp array identifiers
        if self.has_rodded and hasattr(self.rodded, 'pin_model'):
            new_regs[self._rodded_idx].pin_temps[:, 0] = clone.id
            # new_regs[self._rodded_idx].pin_temps[:, 1] = clone.dif3d_id

        # Update peak temperature object
        clone._peak = copy.deepcopy(self._peak)
        clone._power_delivered = copy.deepcopy(self._power_delivered)

        # Assign copied regions
        clone.region = new_regs
        return clone

    def setup_data_io(self, ncols):
        """Pregenerate some stuff to improve data I/O performance

        Parameters
        ----------
        ncols : c

        """
        # These are the data objects that will be written to the
        # global dump file; the number of columns is determined based
        # on which assembly has the highest number of values (for
        # example, subchannels); the rest will be zeros.
        # Note: pin data handled separately.
        self._write = {}
        self._write['coolant_int'] = np.zeros((1, ncols['coolant_int']))
        self._write['duct_mw'] = np.zeros((1, ncols['duct_mw']))
        self._write['coolant_byp'] = np.zeros((1, ncols['coolant_byp']))
        self._write['maximum'] = np.zeros((1, 8))
        self._write['average'] = np.zeros((1, 11))
        # self._write['coolant_gap'] = np.zeros((1, 10))
        self._write['coolant_gap'] = np.zeros((1, ncols['coolant_gap']))

        # Fill in the assembly ID data
        for key in self._write.keys():
            self._write[key][0, 0] = self.id

        # These are the values that indicate the fill length of each
        # data field. The data produced by the Assembly object needs
        # to fill the array required by the global array. This dict
        # anticipates the length of the Assembly data arrays to speed
        # up the process of partially filling the global array
        idc = {'coolant_int': 3,
               'coolant_byp': 4,
               'duct_mw': 4,
               'average': 3,
               'maximum': 3,
               'pin': 3}
        self._fillcols = {}
        for key in self._write.keys():
            self._fillcols[key] = np.zeros(len(self.region), dtype=int)
            for i in range(len(self.region)):
                if key in self.region[i].temp.keys():
                    self._fillcols[key][i] = \
                        self.region[i].temp[key].shape[-1] + idc[key]

        # Gap columns handled separately - same number of adjacent
        # gap channels as duct midwall, so use that as setup basis
        self._fillcols['coolant_gap'] = \
            np.zeros(len(self.region), dtype=int)
        for i in range(len(self.region)):
            self._fillcols['coolant_gap'][i] = \
                self.region[i].temp['duct_mw'].shape[-1] + 3

    @property
    def name(self):
        """Assembly name/type"""
        return self._name

    @property
    def loc(self):
        """Assembly ring-position location in core"""
        return self._loc

    @property
    def id(self):
        """Assembly index (python indexing)"""
        return self._id

    @property
    def z(self):
        """Return the current axial position"""
        return self._z

    @property
    def rodded(self):
        """Return the rodded region object"""
        if self._rodded_idx is not None:
            return self.region[self._rodded_idx]

    @property
    def has_rodded(self):
        """Boolean to indicate whether assembly has a rodded region"""
        if self._rodded_idx is not None:
            return True
        else:
            return False

    @property
    def has_unrodded(self):
        """Boolean to indicate whether assembly has unrodded regions"""
        # If many regions, some must be unrodded by definition
        if len(self.region) > 1:
            return True
        else:
            if self.has_rodded:  # If only region is rodded
                return False
            else:  # Otherwise, entire thing is undrodded
                return True

    @property
    def active_region_idx(self):
        """Return the axial region index you're currently in"""
        # if self.z == 0.0:
        #     return 0
        # else:
        #     idx = bisect.bisect_left(self.region_bnd, self.z) - 1
        #     return self.region_idx[idx]
        return self._active_region_idx

    @property
    def active_region(self):
        """Return the axial region object you're in"""
        return self.region[self.active_region_idx]

    @property
    def x_pts(self):
        """Return the points on which to approximate gap temperatures
        to calculate outer duct wall temperatures (based on noding in
        current active region)"""
        return self.active_region.x_pts

    @property
    def xparams(self):
        """If available, return precalculated arrays for the least-
        squares fitting of data with Legendre polynomials"""
        if self.active_region.is_rodded:
            return self.active_region._xparams
        else:
            return {'gap2duct': None, 'duct2gap': None}

    @property
    def yparams(self):
        """If available, return precalculated arrays for the least-
        squares fitting of data with Legendre polynomials"""
        if self.active_region.is_rodded:
            return self.active_region._yparams
        else:
            return {'gap2duct': None, 'duct2gap': None}

    @property
    def pressure_drop(self):
        """Pressure drop across the assembly"""
        # self._pressure_drop is dP only over previously completed
        # regions; to catch the current region, need to add in
        # separately
        return self._pressure_drop + self.active_region.pressure_drop

    @property
    def temp_coolant(self):
        """Return the primary coolant temperature in the assembly at
        the current axial level"""
        return self.active_region.temp['coolant_int']

    @property
    def temp_bypass(self):
        if 'coolant_byp' in self.active_region.temp.keys():
            return self.active_region.temp['coolant_byp']

    @property
    def temp_duct_mw(self):
        return self.active_region.temp['duct_mw']

    @property
    def temp_duct_surf(self):
        return self.active_region.temp['duct_surf']

    @property
    def avg_coolant_int_temp(self):
        """Calculate the volume-average of the individual subchannel
        temperatures at the last axial level"""
        return self.active_region.avg_coolant_int_temp

    @property
    def avg_coolant_temp(self):
        """Calculate the volume-average of the individual subchannel
        temperatures at the last axial level"""
        return self.active_region.avg_coolant_temp

    @property
    def avg_duct_mw_temp(self):
        """Calculate the volume-average of individual duct cell
        temperatures at the last axial level"""
        return self.active_region.avg_duct_mw_temp

    @property
    def duct_outer_surf_temp(self):
        """Get the outer surface temperature of the outer duct at
        last axial level"""
        return self.active_region.temp['duct_surf'][-1, -1, :]

    @property
    def pin_temp_array(self):
        """Get the entire pin temperature array"""
        if hasattr(self.active_region, 'pin_model'):
            p = self.active_region.pin_temps
            p[:, 1] = self.z
            return p
        else:
            return None

    @property
    def _fill(self):
        """Get column fill reqs for active region to write data"""
        fill = {}
        for k in self._fillcols.keys():
            fill[k] = self._fillcols[k][self.active_region_idx]
        return fill

    ####################################################################
    # CALCULATION
    ####################################################################
    def step0(self, temp_gap, htc_gap, adiabatic=False):
        """Update duct temperatures prior to the sweep based on inlet
        coolant temperatures

        Parameters
        ----------
        temp_gap : numpy.ndarray
            Interassembly coolant gap temperatures around the assembly
            at the j+1 axial level (array len = n_sc['duct']['total'])
        htc_gap : listtype
            Heat transfer coefficient for convection between the gap
            coolant and outer duct wall based on core-average inter-
            assembly gap coolant temperature; len = 2 (edge and corner
            meshes)
        adiabatic : boolean (optional)
            Indicate whether outer duct has adiabatic BC (default False)

        Notes
        -----
        If it's the first step, the duct temperatures need to
        be determined based on the inlet coolant temperatures.
        By default, they're set equal to the inlet coolant
        temperatures, so that when inlet coolant temperatures are
        the same, this does nothing. However, if inlet coolant
        temperatures are different across a duct wall, that duct
        wall temperature needs to be precalculated before the
        sweep.

        """
        if self.active_region.is_rodded:
            p0 = np.zeros(self.active_region.temp['duct_mw'].size)
            self.active_region._calc_duct_temp(p0, temp_gap, htc_gap,
                                               adiabatic)
        else:
            self.active_region._calc_duct_temp(temp_gap, htc_gap,
                                               adiabatic)

    def calculate(self, z, dz, t_gap, h_gap, adiabatic=False, ebal=False):
        """Calculate coolant and temperatures at axial level j+1 based
        on coolant and duct wall temperatures at axial level j

        Parameters
        ----------
        z : float
            Axial mesh cell centerpoint
        dz : float
            Axial step size (m)
        t_gap : numpy.ndarray
            Interassembly coolant gap temperatures around the assembly
            at the j+1 axial level (array len = n_sc['duct']['total'])
        h_gap : listtype
            Heat transfer coefficient for convection between the gap
            coolant and outer duct wall based on core-average inter-
            assembly gap coolant temperature; len = 2 (edge and corner
            meshes)
        adiabatic : boolean (optional)
            Indicate whether outer duct has adiabatic BC (default False)
        ebal : boolean (optional)
            Indicate whether to update energy balance

        Returns
        -------
        None

        """
        self._z = z
        # Calculate power at this axial level (j), calculate
        # temperatures and pin powers (if applicable)
        # power_j = self.power.get_power(z)
        power_j = self.power.get_power(z - 0.5 * dz)
        # self._power_delivered += dz * np.sum([np.sum(x) for x in
        #                                      power_j.values()
        #                                      if x is not None])
        for k in power_j.keys():
            if power_j[k] is not None:
                self._power_delivered[k] += dz * np.sum(power_j[k])

        # Calculate coolant and duct temperatures
        self.active_region.calculate(dz, power_j, t_gap,
                                     h_gap, adiabatic, ebal)

        # Update peak coolant and duct temperatures
        self._update_peak_coolant_duct()

        # If applicable, calculate pin temperatures
        if hasattr(self.active_region, 'pin_model'):
            self.active_region.calculate_pin_temperatures(
                dz, power_j['pins'])
            self._update_peak_pin_temps()

    def check_region_update(self, z):
        """Take an axial step and update the active axial region,
        if necessary; accumulate that region's pressure drop"""
        active_region_id = self._identify_active_region(z)
        old_region_id = self.active_region_idx
        self._z = z
        # If necessary, activate new region
        if old_region_id != active_region_id:
            # Update to new region index
            self._active_region_idx = active_region_id
            # Update total pressure drop from old region
            self._pressure_drop += \
                self.region[old_region_id].pressure_drop
            # Activate new region: now that index is updated,
            # "active_region" property returns new region
            self.active_region.activate(self.region[old_region_id])

    def check_region_update2(self, z):
        """Check whether an axial step takes place in a new region

        Parameters
        ----------
        z : float
            Absolute axial position
        t_gap : numpy.ndarray
            Adjacent gap temperatures on the inter-assembly gap mesh
        h_gap : numpy.ndarray
            Adjacent gap subchannel HTC on the inter-assembly gap mesh

        Notes
        -----
        To set up a new region, this method:
            1. Averages coolant temperatures from old --> regions at
               axial level j; each subchannel in the new region gets
               the average temperature of the old region
            2. Recalculates duct temperatures at axial level j based
               on the new coolant temperatures and gap temperatures on
               the new region duct mesh.

        """
        active_region_id = self._identify_active_region(z)
        old_region_id = self.active_region_idx
        if old_region_id != active_region_id:
            return True
        else:
            return False

    def update_region(self, z, t_gap, h_gap, adiabatic=False):
        """Set up new Region objecct

        Parameters
        ----------
        z : float
            Absolute axial position
        t_gap : numpy.ndarray
            Adjacent gap temperatures on the inter-assembly gap mesh
        h_gap : numpy.ndarray
            Adjacent gap subchannel HTC on the inter-assembly gap mesh
        adiabatic : boolean
            Indicate whether adiabatic outer duct

        Notes
        -----
        To set up a new region, this method:
            1. Averages coolant temperatures from old --> regions at
               axial level j; each subchannel in the new region gets
               the average temperature of the old region
            2. Recalculates duct temperatures at axial level j based
               on the new coolant temperatures and gap temperatures on
               the new region duct mesh.

        """
        active_region_id = self._identify_active_region(z)
        old_region_id = self.active_region_idx
        if old_region_id != active_region_id:
            # Activate new region index
            self._active_region_idx = active_region_id
            # Update total pressure drop from old region
            self._pressure_drop += \
                self.region[old_region_id].pressure_drop
            # Activate new region: index is already been updated
            # so that "active_region" property returns new region
            if adiabatic:
                new_gap_htc = np.ones(self.duct_outer_surf_temp.shape[0])
                new_gap_temp = new_gap_htc
            else:
                new_gap_htc = mesh_functions.map_across_gap(
                    h_gap, self.active_region._map['gap2duct'])
                new_gap_temp = mesh_functions.map_across_gap(
                    t_gap * h_gap, self.active_region._map['gap2duct'])
                new_gap_temp = new_gap_temp / new_gap_htc
            self.active_region.activate2(self.region[old_region_id],
                                         new_gap_temp,
                                         new_gap_htc,
                                         adiabatic)

    def _identify_active_region(self, z):
        """Identify the index of the current axial region"""
        if z == 0.0:
            return 0
        else:
            return bisect.bisect_left(self.region_bnd, z) - 1

    def _update_peak_coolant_duct(self):
        """Update peak coolant and duct temperatures, if necessary"""
        max_cool = np.max(self.temp_coolant)
        if max_cool > self._peak['cool'][0]:
            self._peak['cool'] = (max_cool, self.z)
        max_duct = np.max(self.temp_duct_mw, axis=1)
        for i in range(max_duct.shape[0]):
            idx_to_write = len(self._peak['duct']) - max_duct.shape[0] + i
            if max_duct[i] > self._peak['duct'][idx_to_write][0]:
                self._peak['duct'][idx_to_write] = (max_duct[i], self.z)

    def _update_peak_pin_temps(self):
        """Check whether peak pin temperatures need to be updated
        and do so if necessary"""
        for k in self._peak['pin'].keys():
            idx = np.argmax(self.pin_temp_array[:, self._peak['pin'][k][1]])
            if (self.pin_temp_array[idx, self._peak['pin'][k][1]]
                    > self._peak['pin'][k][0]):
                self._peak['pin'][k][0] = \
                    self.pin_temp_array[idx, self._peak['pin'][k][1]]
                self._peak['pin'][k][2] = list(self.pin_temp_array[idx])

    ####################################################################
    # Write data to CSV
    ####################################################################

    def write(self, dfiles, gap_temp=None):
        """."""
        fill = self._fill
        write_step = copy.deepcopy(self._write)

        # Update z position, active region index
        for k in write_step.keys():
            write_step[k][0, 1] = self.z
            write_step[k][0, 2] = self.active_region_idx

        # Interior coolant temperatures
        if 'coolant_int' in dfiles.keys():
            write_step['coolant_int'][0, 3:fill['coolant_int']] = \
                self.temp_coolant
            np.savetxt(dfiles['coolant_int'],
                       write_step['coolant_int'],
                       delimiter=',')

        # Duct midwall
        if 'duct_mw' in dfiles.keys():
            for i in range(len(self.temp_duct_mw)):
                write_step['duct_mw'][0, 3] = i
                write_step['duct_mw'][0, 4:fill['duct_mw']] = \
                    self.temp_duct_mw[i]
                np.savetxt(dfiles['duct_mw'],
                           write_step['duct_mw'],
                           delimiter=',')

        # Bypass coolant
        if 'coolant_byp' in dfiles.keys():
            if 'coolant_byp' in self.active_region.temp.keys():
                # While you're here, update average bypass temperature
                # write_step['average'] = \
                #     self.active_region.avg_coolant_byp_temp[0]
                for i in range(len(self.temp_bypass)):
                    write_step['coolant_byp'][0, 3] = i
                    (write_step['coolant_byp']
                               [0, 4:fill['coolant_byp']]) = \
                        self.temp_bypass[i]
                    np.savetxt(dfiles['coolant_byp'],
                               write_step['coolant_byp'],
                               delimiter=',')

        # Pin cladding and fuel centerline temperatures
        if hasattr(self.active_region, 'pin_model'):
            if 'average' in dfiles.keys():
                # Average clad MW and fuel CL temperatures
                write_step['average'][0, 8] = \
                    np.average(self.active_region.pin_temps[:, 6])
                write_step['average'][0, 9] = \
                    np.average(self.active_region.pin_temps[:, -1])
            if 'maximum' in dfiles.keys():
                # Maximum clad MW and fuel CL temperatures
                write_step['maximum'][0, 5] = \
                    np.max(self.active_region.pin_temps[:, 6])
                write_step['maximum'][0, 6] = \
                    np.max(self.active_region.pin_temps[:, -1])
            if 'pin' in dfiles.keys():
                np.savetxt(dfiles['pin'],
                           self.pin_temp_array,
                           delimiter=',')

        # Update remaining average temperatures
        if 'average' in dfiles.keys():
            write_step['average'][0, 3] = self.avg_coolant_int_temp
            write_step['average'][0, 5] = self.avg_coolant_temp
            write_step['average'][0, 6] = self.avg_duct_mw_temp[0]
            write_step['average'][0, 7] = self.avg_duct_mw_temp[-1]
            np.savetxt(dfiles['average'], write_step['average'],
                       delimiter=',')

        # Update remaining maximum temperatures
        if 'maximum' in dfiles.keys():
            write_step['maximum'][0, 3] = np.max(self.temp_coolant)
            write_step['maximum'][0, 4] = np.max(self.temp_duct_mw[0])
            np.savetxt(dfiles['maximum'], write_step['maximum'],
                       delimiter=',')

        # Adjacent gap temperatures: (to do: add them to average?)
        if 'coolant_gap' in dfiles.keys() and gap_temp is not None:
            write_step['coolant_gap'][0, 3:fill['coolant_gap']] = gap_temp
            # gap_temp.shape = (6, int(len(gap_temp) / 6))
            # write_step['coolant_gap'][0, 4:] = np.average(gap_temp, axis=1)
            np.savetxt(dfiles['coolant_gap'],
                       write_step['coolant_gap'],
                       delimiter=',')

########################################################################


def calculate_min_dz(asm_obj, t1, t2, adiabatic_duct=False):
    """Calculate the minimium dz for each of the axial regions in
    the Assembly, including those without rods."""
    dz = []
    sc = []
    for r in asm_obj.region:
        if r.is_rodded:
            tmp_dz, tmp_sc = region_rodded.calculate_min_dz(
                r, t1, t2, adiabatic_duct)
        else:
            tmp_dz, tmp_sc = region_unrodded.calculate_min_dz(
                r, t1, t2, adiabatic_duct)
        dz.append(tmp_dz)
        sc.append(tmp_sc)

    min_dz = min(dz)
    return min_dz, sc[dz.index(min_dz)]


########################################################################
