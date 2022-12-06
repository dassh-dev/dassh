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
date: 2022-11-30
author: matz
Methods for unrodded axial regions; to be used within Assembly objects
"""
########################################################################
import copy
import numpy as np
from dassh import region_rodded
from dassh.correlations import nusselt_db
from dassh.region import DASSH_Region
from dassh.pin import PinLattice
from dassh.pin import count_pins
from dassh.subchannel import Subchannel
from dassh.region_rodded import RoddedRegion
from dassh.logged_class import LoggedClass


_sqrt3 = np.sqrt(3)


def make_ur_asm(name, asm_input, mat_dict, flow_rate, se2geo=False):
    """Process DASSH Assembly input to obtain un-rodded region input
    parameters; to be used when instantiating un-rodded region objects
    in DASSH Assembly object"""
    A_pins = np.pi * asm_input['pin_diameter']**2 / 4
    A_pins += np.pi * asm_input['wire_diameter']**2 / 4
    A_pins *= count_pins(asm_input['num_rings'])
    A_hex = _sqrt3 * 0.5 * min(asm_input['duct_ftf'])**2
    args = [name,
            asm_input['AxialRegion']['rods']['z_lo'],
            asm_input['AxialRegion']['rods']['z_hi'],
            asm_input['duct_ftf'],
            (A_hex - A_pins) / A_hex,
            flow_rate,
            mat_dict['coolant'],
            mat_dict['duct'],
            asm_input['htc_params_duct']]
    kwargs = {'eps': 0.0,
              'rr_equiv': _RREquivalent(asm_input, mat_dict,
                                        flow_rate, se2geo)}
    if asm_input.get('convection_factor'):
        kwargs['convection_factor'] = asm_input['convection_factor']
    else:
        kwargs['convection_factor'] = 1.0
    if asm_input.get('model') == '6node':
        return MultiNodeHomogeneous(*args, **kwargs)
    else:
        return SingleNodeHomogeneous(*args, **kwargs)


def make_ur_axialregion(asm_input, reg, mat_dict, flow_rate):
    """Process DASSH Assembly AxialRegion input to obtain un-rodded
    region input parameters; to be used when instantiating un-rodded
    region objects as axial regions in DASSH Assembly object"""
    model = asm_input['AxialRegion'][reg]['model']
    args = [reg,
            asm_input['AxialRegion'][reg]['z_lo'],
            asm_input['AxialRegion'][reg]['z_hi'],
            asm_input['duct_ftf'],
            asm_input['AxialRegion'][reg]['vf_coolant'],
            flow_rate,
            mat_dict['coolant'],
            mat_dict['duct'],
            asm_input['AxialRegion'][reg]['htc_params']]
    kwargs = {
        'eps': asm_input['AxialRegion'][reg]['epsilon'],
        'de': asm_input['AxialRegion'][reg]['hydraulic_diameter'],
        'convection_factor':
            asm_input['AxialRegion'][reg]['convection_factor']}
    if model == 'simple':
        return SingleNodeHomogeneous(*args, **kwargs)
    elif model == '6node':
        return MultiNodeHomogeneous(*args, **kwargs)
    else:
        msg = ('Only "simple" and "6node" unrodded region'
               'models available at the moment.')
        raise NotImplementedError(msg)


def _get_rr_kwargs(asm_input, mat_dict, fr, temp):
    """Instantiate a RR object and obtain from it various parameters
    to use in the UR model"""
    kwargs = {}
    rr = region_rodded.make_rr_asm(asm_input, 'dummy', mat_dict, fr)
    rr._update_coolant_int_params(temp)
    # Bundle hydraulic diameter
    kwargs['de'] = rr.bundle_params['de']
    # Bundle friction factor
    kwargs['ff'] = rr.coolant_int_params['ff']
    # Interior mass flow rate fraction
    kwargs['xm1'] = (rr.subchannel.n_sc['coolant']['interior']
                     * rr.params['area'][0]
                     * rr.coolant_int_params['fs'][0]
                     / rr.bundle_params['area'])
    return kwargs


class SingleNodeHomogeneous(LoggedClass, DASSH_Region):
    """Base reflector type; represents a simple, one-node, homogeneous
    axial region.

    Contains coolant, structure, and duct material properties and
    returns only one lumped temperature for coolant based on Q=mCdT
    along with conduction to the duct. Inherited by other classes in
    this module which overwrite the methods herein to produce more
    complicated reflector models.

    Parameters
    ----------
    z_lo : float
        Axial position at which the reflector region begins
    z_hi : float
        Axial position at which the reflector region ends
    duct_ftf : tuple, list
        Inner and outer flat-to-flat distances (m) of the assembly
        outermost duct.
    vf_cool : float
        Volume fraction of coolant in the reflector. The volume
        fraction of structure is given by (1 - vf_cool).
    flow_rate: float
        Total coolant flow rate through the assembly
    coolant : DASSH Material object
        Contains coolant material properties
    duct : DASSH Material object
        Contains duct wall material properties

    """
    def __init__(self, name, z_lo, z_hi, duct_ftf, vf_cool, flow_rate,
                 coolant_mat, duct_mat, htc_params, eps=0.0, de=0.0,
                 convection_factor=None, rr_equiv=None, lowflow=False):
        """Create a porous media region"""
        # Instantiate Logger
        LoggedClass.__init__(self, 4, 'dassh.SingleNodeHomogeneousRegion')
        self.name = name
        self.model = 'simple'
        self.z = [z_lo, z_hi]
        duct_ftf = [sorted(duct_ftf)[i:i + 2] for i in
                    range(0, len(duct_ftf), 2)]
        self.duct_ftf = duct_ftf[-1]
        self.duct_thickness = (self.duct_ftf[1] - self.duct_ftf[0]) / 2
        # self.duct_perim = self.duct_ftf[0] * 6 / _sqrt3
        self.duct_perim = self.duct_ftf[1] * 6 / _sqrt3
        # One sixth of the duct perimeter for the coolant temp calc
        # (self.duct_perimeter defined in SingleNodeHomogeneous)
        self.duct_perim_over_6 = self.duct_perim / 6
        self.flow_rate = flow_rate
        self.coolant = coolant_mat
        self.duct = duct_mat
        self._conv_approx = lowflow  # duct wall convection approx flag
        self._rr_equiv = rr_equiv  # rod bundle equivalent

        # Volume fractions of coolant and structure
        self.vf = {}
        self.vf['coolant'] = vf_cool
        self.vf['struct'] = 1 - vf_cool

        # Cross-sectional areas of coolant and structure inside assembly
        total_area = _sqrt3 * 0.5 * self.duct_ftf[0] * self.duct_ftf[0]
        coolant_area = total_area * vf_cool
        # struct_area = total_area * (1 - vf_cool)
        # Cross-sectional area of outer duct wall
        a_duct = _sqrt3 * (self.duct_ftf[1]**2 - self.duct_ftf[0]**2) / 2

        # Temperatures - get from DASSH_Region object
        DASSH_Region.__init__(self, 1, coolant_area,
                              6, np.ones((1, 6)) * a_duct / 6)

        # PARAMETER SETUP
        self.coolant_params = {}
        self._params = {}

        # Duct heat transfer coefficient correlation params
        if htc_params is not None:
            self._params['htc'] = htc_params
        else:
            self._params['htc'] = [0.023, 0.8, 0.8, 7.0]

        # Hydraulic diameter
        if de != 0.0:
            self._params['de'] = de
        else:
            wp = self.duct_perim
            a_struct = self.vf['struct'] * total_area  # incl. pins
            r_struct = np.sqrt(a_struct / np.pi)
            struct_perim = 2 * np.pi * r_struct
            wp += struct_perim
            self._params['de'] = 4 * coolant_area / wp

        # Surface roughness of the duct
        self._params['eps'] = eps

        # Set up x-points for treating mesh disagreement; this is for
        # use *along each hex side* including corners.
        self.x_pts = np.array([-1, 1])  # Only pts are the corners

        # Ratio of interior flow rate in rod bundle to total flow rate
        self._mratio = 1.0
        if convection_factor is not None:
            self._mratio = convection_factor

    def calculate_xbnds(self):
        """Calculate boundaries between duct elements

        Notes
        -----
        First array entry =0 (center of the top corner duct element)
        Second array entry is 1/2 of the top corner duct element
        Entries 3-7 are the "upper" bounds of the next 5 corner ducts
        The final entry (8) is the center of the top corner duct
            (equals duct outer perimeter)

        """
        hex_side = self.duct_ftf[1] / np.sqrt(3)
        x_bnds = np.array([0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.0])
        return hex_side * x_bnds

    def clone(self, new_flowrate=None):
        """Make a clone of the unrodded region"""
        clone = copy.copy(self)
        clone.temp = copy.deepcopy(self.temp)
        clone.ebal = copy.deepcopy(self.ebal)
        clone.coolant_params = copy.deepcopy(self.coolant_params)
        if new_flowrate is not None:
            clone.flow_rate = new_flowrate
            if self._rr_equiv is not None:
                clone._rr_equiv = self._rr_equiv.clone(new_flowrate)
        return clone

    @property
    def mratio(self):
        if self._mratio == 'calculate':
            return (
                self._rr_equiv.subchannel.n_sc['coolant']['interior']
                * self._rr_equiv.params['area'][0]
                * self._rr_equiv.corr['fs'](self._rr_equiv)[0]
                / self._rr_equiv.bundle_params['area'])
        elif self._mratio is None:
            return 1.0
        else:
            return self._mratio

    def _update_coolant_params(self, temp, use_mat_tracker=True):
        """Update correlated bundle coolant parameters based
        on current average coolant temperature

        Parameters
        ----------
        temperature : float
            Average coolant temperature
        use_mat_tracker : boolean
            Use material property tol in DASSH coolant Material object
            to limit pin bundle correlation updates in rr_equiv

        Notes
        -----
        Updates coolant_params dict attribute with keys:
            'ff': friction factor
            'fs': flow split
            'htc': heat transfer coefficient
            'eddy': eddy diffusivity
            'swirl': swirl velocity

        """
        self._update_coolant(temp)
        if self._rr_equiv is not None:
            self._rr_equiv._update_coolant_int_params(
                temp, use_mat_tracker)
            # Bundle-average velocity
            self.coolant_params['vel'] = \
                self._rr_equiv.coolant_int_params['vel']
            # Bundle RE
            self.coolant_params['Re'] = \
                self._rr_equiv.coolant_int_params['Re']
            # Friction factor
            # self.coolant_params['ff'] = \
            #     self._rr_equiv.coolant_int_params['ff']
            # Placeholder for coolant thermal conductivity
            k = self.coolant.thermal_conductivity
            k *= self._rr_equiv._sf
            k *= 1 - (self._rr_equiv.pin_diameter
                      / self._rr_equiv.pin_pitch)
            k += (self.coolant.density * self.coolant.heat_capacity
                  * self._rr_equiv.coolant_int_params['eddy'])

        else:
            # Bundle-average velocity
            self.coolant_params['vel'] = (
                self.flow_rate
                / self.coolant.density
                / self.total_area['coolant_int'])
            # Bundle Reynolds number
            self.coolant_params['Re'] = (
                self.flow_rate
                * self._params['de']
                / self.coolant.viscosity
                / self.total_area['coolant_int'])
            # # Friction factor
            # if self.coolant_params['Re'] < 2200.0:  # Laminar
            #     self.coolant_params['ff'] = \
            #         64 / self.coolant_params['Re']
            # else:  # Turbulent or transition
            #     a = self._params['eps'] / self._params['de'] / 3.7
            #     b = 4.518 / self.coolant_params['Re']
            #     c = 6.9 / self.coolant_params['Re']
            #     f = (-0.5 / np.log10(a - b * np.log10(c + a**1.11)))**2
            #     if self.coolant_params['Re'] < 3000.0:  # Turbulent
            #         f2200 = 64.0 / 2200.0
            #         x = 3.75 - 8250.0 / self.coolant_params['Re']
            #         f = f2200 + x * (f - f2200)
            #     self.coolant_params['ff'] = f
            # Placeholder for coolant thermal conductivity
            k = self.coolant.thermal_conductivity

        # Heat transfer coefficient (via Nusselt number)
        nu = nusselt_db.calculate_bundle_Nu(self.coolant,
                                            self.coolant_params['Re'],
                                            self._params['htc'])
        self.coolant_params['htc'] = k * nu / self._params['de']
        self.coolant_params['htc'] *= self.mratio

    def calculate_pressure_drop(self, dz):
        """Calculate pressure drop attribute across current step"""
        dp = (self.coolant_params['ff'] * dz * self.coolant.density
              * self.coolant_params['vel']**2 / 2)
        if self._rr_equiv is not None:
            return dp / self._rr_equiv.bundle_params['de']
        else:
            return dp / self._params['de']

    def calculate(self, dz, power, t_gap, htc_gap,
                  adiabatic_duct=False, ebal=False):
        """Calculate new coolant and duct temperatures and pressure
        drop across axial step

        Parameters
        ----------
        dz : float
            Axial step size (m)
        power : float
            Linear power delivered to mesh (W/m)
        t_gap : numpy.ndarray
            Gap temperatures in the interassembly coolant around the
            assembly (array len = number of duct meshes)
        adiabatic_duct : boolean (optional)
            Indicate whether outer duct has adiabatic BC
        ebal : boolean (optional)
            Indicate whether to update energy balance tallies

        Returns
        -------
        None

        """
        # Duct temperatures: calculate with new coolant properties
        self._calc_duct_temp(t_gap, htc_gap, adiabatic_duct)

        # Interior coolant temperatures: calculate using coolant
        # properties from previous axial step
        self.temp['coolant_int'] += self._calc_coolant_temp(
            dz, power, adiabatic_duct, ebal)

        # Update coolant properties for the duct wall calculation
        self._update_coolant_params(self.temp['coolant_int'][0])

        # # Duct temperatures: calculate with new coolant properties
        # self._calc_duct_temp(t_gap, htc_gap, adiabatic_duct)

        # Update pressure drop (now that correlations are updated)
        self._pressure_drop += self.calculate_pressure_drop(dz)

    def activate(self, previous_reg, t_gap, h_gap, adiabatic):
        """Activate region by averaging coolant temperatures from
        previous region and calculating new SS duct temperatures

        Parameters
        ----------
        previous_reg : DASSH Region
            Previous region to get average coolant temperatures to
            assign to new region
        t_gap : numpy.ndarray
            Gap temperatures on the new region duct mesh
        h_gap : numpy.ndarray
            Gap coolant HTC on the new region duct mesh
        adiabatic : boolean
            Indicate whether outer duct wall is adiabatic

        Notes
        -----

        """
        # Base method assigns coolant and duct MW temperatures
        # - Coolant temperature(s) set to previous region average
        # - Outer duct MW temperatures set to average outer duct MW
        #   temp in previous region; set MW temp of other ducts in
        #   new region to average coolant temperature
        self._activate_base(previous_reg)

        # Make new duct temperature calculation based on new coolant
        # temperatures and input gap temperatures / HTC.
        self._update_coolant_params(self.temp['coolant_int'][0])
        self._calc_duct_temp(t_gap, h_gap, adiabatic)

    def _calc_coolant_temp(self, dz, power, adiabatic=False, ebal=False):
        """Calculate single node coolant temperature with Q=mCpdT

        Parameters
        ----------
        dz : float
            Axial step size (m)
        power : dict
            Linear power (W/m) in reflector (key: 'refl')
        adiabatic : bool (optional)
            Indicate if duct wall is adiabatic; if so, shouldn't have
            any impact on temperature calculation and is neglected
        ebal : bool (optional)
            Indicate whether to update energy balance tallies

        """
        # Update coolant parameters and material properties
        # Tj = self.avg_coolant_int_temp  # <-- dont wanna calc everytim

        if power['refl'] is None:
            power['refl'] = 0

        # Calculate change in temperature from heat generation
        dT = power['refl']

        # Do some heat transfer to/from the ducts
        if not adiabatic:
            if self._conv_approx:
                R = (0.5 * self.duct_thickness
                     / self.duct.thermal_conductivity)
                R += 1 / self.coolant_params['htc']
                dT_duct = (self.duct_perim_over_6 / R
                           * (self.temp['duct_mw'][0]
                              - self.temp['coolant_int'][0]))
            else:
                dT_duct = ((self.temp['duct_surf'][0, 0]
                            - self.temp['coolant_int'][0])
                           * self.coolant_params['htc']
                           * self.duct_perim_over_6)
            # dT_duct = dT_duct * self.mratio
            dT += np.sum(dT_duct)

        if ebal:
            if adiabatic:
                self.update_ebal(power['refl'] * dz, np.zeros(6))
            else:
                self.update_ebal(power['refl'] * dz, dT_duct * dz)

        return dT * dz / self.flow_rate / self.coolant.heat_capacity

    def _calc_duct_temp(self, temp_gap, htc_gap, adiabatic=False):
        """Calculate the duct wall temperatures based on the adjacent
        coolant temperatures with no heat generation (q''' = 0)

        Parameters
        ----------
        temp_gap : numpy.ndarray
            Interassembly gap temperatures around the assembly at the
            j+1 axial level (array length = n_sc['duct']['total'])
        htc_gap : listtype
            Heat transfer coefficient for convection between the gap
            coolant and outer duct wall based on core-average inter-
            assembly gap coolant temperature; len = 2 (edge and corner
            meshes)
        adiabatic : bool
            Indicate if outer duct outer boundary cond. is adiabatic

        Returns
        -------
        None

        """
        # Because no heat generation in duct, if adiabatic, duct wall
        # temperatures are all equal to coolant temperature (no grad)
        if adiabatic:
            self.temp['duct_mw'][0, :] = self.temp['coolant_int']
            self.temp['duct_surf'][0, :, :] = self.temp['coolant_int']

        else:
            # Update coolant parameters and material properties
            # self._update_coolant_params(self.avg_coolant_int_temp)
            # self.duct.update(self.avg_duct_mw_temp[0])
            self._update_duct(self.avg_duct_mw_temp[0])
            L_over_2 = self.duct_thickness / 2
            # c1 = (self.coolant_params['htc']
            #       * (temp_gap - self.temp['coolant_int'])
            #       / (self.coolant_params['htc'] * self.duct_thickness
            #          + (self.duct.thermal_conductivity
            #             * (1 + self.coolant_params['htc'] / htc_gap[1]))))
            # c2 = (temp_gap
            #       - c1 * (L_over_2
            #               + self.duct.thermal_conductivity / htc_gap[1]))
            c1 = (self.coolant_params['htc']
                  * (temp_gap - self.temp['coolant_int'])
                  / (self.coolant_params['htc'] * self.duct_thickness
                     + (self.duct.thermal_conductivity
                        * (1 + self.coolant_params['htc'] / htc_gap))))
            c2 = (temp_gap
                  - c1 * (L_over_2 + self.duct.thermal_conductivity / htc_gap))
            self.temp['duct_mw'][0] = c2
            self.temp['duct_surf'][0, 0] = c1 * -L_over_2 + c2
            self.temp['duct_surf'][0, 1] = c1 * L_over_2 + c2

    def _init_static_correlated_params(self, t):
        """Calculate bundle friction factor at the bundle-average
        temperature

        Parameters
        ----------
        t : float
            Bundle axial average temperature ((T_in + T_out) / 2)

        Returns
        -------
        None

        Notes
        -----
        This method is called within the '_setup_asm' method
        inside the Reactor object instantiation. It is similar
        to the '_update_coolant_params' method.

        """
        # Update coolant material properties
        t_inlet = self.coolant.temperature
        self._update_coolant(t)

        if self._rr_equiv is not None:
            self._rr_equiv._init_static_correlated_params(t)
            # Bundle RE
            self.coolant_params['Re'] = \
                self._rr_equiv.coolant_int_params['Re']
            # Bundle-average velocity
            self.coolant_params['vel'] = \
                self._rr_equiv.coolant_int_params['vel']
            # Friction factor
            self.coolant_params['ff'] = \
                self._rr_equiv.coolant_int_params['ff']

        else:
            # Bundle-average velocity
            self.coolant_params['vel'] = (
                self.flow_rate
                / self.coolant.density
                / self.total_area['coolant_int'])
            # Bundle Reynolds number
            self.coolant_params['Re'] = (
                self.flow_rate
                * self._params['de']
                / self.coolant.viscosity
                / self.total_area['coolant_int'])
            # Friction factor
            if self.coolant_params['Re'] < 2200.0:  # Laminar
                self.coolant_params['ff'] = \
                    64 / self.coolant_params['Re']
            else:  # Turbulent or transition
                a = self._params['eps'] / self._params['de'] / 3.7
                b = 4.518 / self.coolant_params['Re']
                c = 6.9 / self.coolant_params['Re']
                f = (-0.5 / np.log10(a - b * np.log10(c + a**1.11)))**2
                if self.coolant_params['Re'] < 3000.0:  # Turbulent
                    f2200 = 64.0 / 2200.0
                    x = 3.75 - 8250.0 / self.coolant_params['Re']
                    f = f2200 + x * (f - f2200)
                self.coolant_params['ff'] = f

        # Reset inlet temperature
        self.coolant.temperature = t_inlet


########################################################################


class MultiNodeHomogeneous(SingleNodeHomogeneous, DASSH_Region):
    """Multi-node axial reflector with homogenized coolant and
    structure; similar to the base-type but with more refined radial
    meshing.

    Parameters
    ----------
    z_lo : float
        Axial position at which the reflector region begins
    z_hi : float
        Axial position at which the reflector region ends
    duct_ftf : tuple, list
        Inner and outer flat-to-flat distances (m) of the assembly
        outermost duct.
    vf_cool : float
        Volume fraction of coolant in the reflector. The volume
        fraction of structure is given by (1 - vf_cool).
    flow_rate: float
        Total coolant flow rate through the assembly
    coolant : DASSH Material object
        Contains coolant material properties
    duct : DASSH Material object
        Contains duct wall material properties

    Notes
    -----
    This object duplicates some of the RoddedRegion object code. It is
    simpler to double up on some of the simple code rather than further
    complicate the RoddedRegion object to handle another special case.

    """

    def __init__(self, name, z_lo, z_hi, duct_ftf, vf_cool, flow_rate,
                 coolant_mat, duct_mat, htc_params, eps=0.0, de=0.0,
                 convection_factor=1.0, rr_equiv=None, lowflow=False):
        """Create the MultiNodeHomogeneous region object"""
        # Inherit from SingleNodeHomogeneous; overwrite some methods
        SingleNodeHomogeneous.__init__(
            self, name, z_lo, z_hi, duct_ftf, vf_cool, flow_rate,
            coolant_mat, duct_mat, htc_params, eps=eps, de=de,
            convection_factor=convection_factor, rr_equiv=rr_equiv,
            lowflow=lowflow)
        self.model = '6node'

        # Use pinlattice and subchannel objects to get subchannel
        # properties
        pin_lattice = PinLattice(1, 0.0, 0.0)
        self.subchannel = Subchannel(1, 0.0, 0.0, pin_lattice.map,
                                     pin_lattice.xy, [self.duct_ftf])

        # Cross-sectional areas of coolant and duct
        total_area = _sqrt3 * 0.5 * self.duct_ftf[0] * self.duct_ftf[0]
        coolant_area = np.ones(6) * total_area * vf_cool / 6

        # Cross-sectional area of outer duct wall
        duct_area = _sqrt3 * (self.duct_ftf[1]**2 - self.duct_ftf[0]**2) / 2
        duct_area = np.ones((1, 6)) * duct_area / 6

        # Distribute flow among subchannels based on area; because
        # all have the same area, all get the same flow
        self._scfr = self.flow_rate / 6

        # Temperatures - get from DASSH_Region object
        DASSH_Region.__init__(self, 6, coolant_area, 6, duct_area)

        # Set up x-points for treating mesh disagreement; this is for
        # use *along each hex side* including corners.
        self.x_pts = np.array([-1, 1])  # Only pts are the corners

        # Set up heat transfer constants
        self._setup_ht_consts()

    def _setup_ht_consts(self):
        """Precalculate conduction/convection heat-transfer constants

        Notes
        -----
        Similar to what's done in RoddedRegion object but because we
        only have corner subchannels, it's much simpler.

        """
        # Conduction matrix: connections between subchannels
        self._cond = {}
        self._cond['adj'] = np.array([[5, 1],
                                      [0, 2],
                                      [1, 3],
                                      [2, 4],
                                      [3, 5],
                                      [4, 0]])
        # Distance between subchannel centroids: x-coordinates
        # of first two subchannels are the same so just take the
        # difference between the y-coordinates
        L = self.subchannel.xy[0, 1] - self.subchannel.xy[1, 1]
        self._cond['const'] = 0.5 * self.duct_ftf[0] / L

    def clone(self, new_flowrate=None):
        """Make a clone of the unrodded region"""
        clone = copy.copy(self)
        clone.temp = copy.deepcopy(self.temp)
        clone.ebal = copy.deepcopy(self.ebal)
        clone.coolant_params = copy.deepcopy(self.coolant_params)
        if new_flowrate is not None:
            clone.flow_rate = new_flowrate
            clone._scfr = new_flowrate / 6
        return clone

    def calculate(self, dz, power, t_gap, htc_gap,
                  adiabatic_duct=False, ebal=False):
        """Calculate new coolant and duct temperatures and pressure
        drop across axial step

        Parameters
        ----------
        dz : float
            Axial step size (m)
        power : dict
            Linear power (W/m) in homogeneous mesh (key: 'refl')
        t_gap : numpy.ndarray
            Gap temperatures in the interassembly coolant around the
            assembly (array len = number of duct meshes)
        adiabatic_duct : boolean (optional)
            Indicate whether outer duct has adiabatic BC
        ebal : boolean (optional)
            Indicate whether to update energy balance tallies

        Returns
        -------
        None

        """
        self.temp['coolant_int'] += \
            self._calc_coolant_temp(dz, power, adiabatic_duct, ebal)
        self._calc_duct_temp(t_gap, htc_gap, adiabatic_duct)

        # Update pressure drop
        self._pressure_drop += self.calculate_pressure_drop(dz)

    def _calc_coolant_temp(self, dz, power, adiabatic=False, ebal=False):
        """Calculate single node coolant temperature with Q=mCpdT

        Parameters
        ----------
        dz : float
            Axial step size (m)
        power : dict
            Linear power (W/m) in reflector (key: 'refl')
        adiabatic : bool (optional)
            Indicate if duct wall is adiabatic; if so, shouldn't have
            any impact on temperature calculation and is neglected
        ebal : bool (optional)
            Indicate whether to update energy balance tallies

        """
        # Update coolant parameters and material properties
        T_avg = self.avg_coolant_int_temp
        self._update_coolant_params(T_avg)

        # Calculate change in temperature from heat generation
        # Don't to divide by 6 because the proportion of power and
        # flow assigned to each subchannel is equal so they cancel
        dT = np.ones(6) * power['refl'] / 6

        # Calculate heat transfer between subchannels
        dT += (
            (np.sum(self.temp['coolant_int'][self._cond['adj']], axis=1)
             - 2 * self.temp['coolant_int'])
            * self._cond['const']
            * self.coolant.thermal_conductivity
        )

        # Do some heat transfer to/from the ducts
        if not adiabatic:
            if self._conv_approx:
                R = (0.5 * self.duct_thickness
                     / self.duct.thermal_conductivity)
                R += 1 / self.coolant_params['htc']
                dT_duct = (self.duct_perim_over_6 / R
                           * (self.temp['duct_mw'][0]
                              - self.temp['coolant_int']))
            else:
                dT_duct = \
                    (self.coolant_params['htc']
                     * self.duct_perim_over_6
                     * (self.temp['duct_surf'][0, 0]
                        - self.temp['coolant_int']))

            dT_duct *= self.mratio
            dT += dT_duct

        if ebal:
            if adiabatic:
                self.update_ebal(power['refl'] * dz, np.zeros(6)),
            else:
                self.update_ebal(power['refl'] * dz, dT_duct * dz)

        return dT * dz / self._scfr / self.coolant.heat_capacity


########################################################################


class _RREquivalent(RoddedRegion):
    """Container to store rod bundle parameters for use in low-fidelity
    model assemblies"""

    _attr_to_keep = ['_attr_to_keep',
                     'n_pin',
                     'L',
                     'd',
                     'pin_pitch',
                     'pin_diameter',
                     'wire_pitch',
                     'wire_diameter',
                     'params',
                     'bundle_params',
                     'subchannel',
                     'coolant_int_params',
                     'corr_constants',
                     'corr_names',
                     'corr',
                     'coolant',
                     'int_flow_rate',
                     'total_flow_rate',
                     'htc_params',
                     '_sf']

    def __init__(self, asm_input, mat_dict, fr, se2geo=False):
        """Instantiate RoddedRegion object and pull out useful attr"""
        # Instantiate RoddedRegion object
        RoddedRegion.__init__(
            self,
            'dummy',
            asm_input['num_rings'],
            asm_input['pin_pitch'],
            asm_input['pin_diameter'],
            asm_input['wire_pitch'],
            asm_input['wire_diameter'],
            asm_input['clad_thickness'],
            asm_input['duct_ftf'],
            fr,
            mat_dict['coolant'],
            mat_dict['duct'],
            asm_input['htc_params_duct'],
            asm_input['corr_friction'],
            asm_input['corr_flowsplit'],
            asm_input['corr_mixing'],
            asm_input['corr_nusselt'],
            asm_input['corr_shapefactor'],
            asm_input['bypass_gap_flow_fraction'],
            asm_input['bypass_gap_loss_coeff'],
            asm_input['wire_direction'],
            asm_input['shape_factor'],
            se2geo)

        # Delete everything we don't want
        to_delete = []
        for item in vars(self):
            if item[:2] != '__' and item not in self._attr_to_keep:
                to_delete.append(item)

        for item in to_delete:
            delattr(self, item)

    def clone(self, new_flowrate=None):
        """Clone the rodded region into another assembly object;
        shallow copy some attributes, deep copy others"""
        # Create a shallow copy (stuff like the pin and subchannel
        # objects can simply be pointed to)
        clone = copy.copy(self)
        if new_flowrate is not None:
            # Define new flow rate attribute in clone
            clone.int_flow_rate = new_flowrate
            clone.total_flow_rate = new_flowrate

        for attr in ['corr', 'coolant_int_params']:
            setattr(clone, attr, copy.deepcopy(getattr(self, attr)))
        return clone

########################################################################
#
#
# class MultiNodeHomogeneous():
#     """Multi-node axial reflector with homogenized coolant and
#     structure; similar to the base-type but with more refined radial
#     meshing.
#
#     Notes
#     -----
#     - Hex-assembly is split into 1/12 triangles, each w/ 4 radial nodes;
#         This is identical to the TCLUST reflector meshing
#     - Q=mCdT at each axial step + conduction from duct; for each node,
#         Q is the product of total power added in the overall axial
#         slice and the area of the mesh cell.
#     - Duct temperatures averaged for duct meshes adjacent to each node
#     """
#     def __init__(self, z_lo, z_hi, duct_ftf, vf_cool, flow_rate,
#                  coolant, struct, duct):
#         Reflector.__init__(z_lo, z_hi, duct_ftf, vf_cool, flow_rate,
#                            coolant, struct, duct)
#         raise NotImplementedError('lol')
#
#         self.node
#         # Noding for the 1/12 hex triangle, and how those nodes are
#         # split up to calculate the area
#         # Noding                Area calculation (notice the rectangles
#         #                       are equal to two triangles)
#         #
#         # |\  Node 1            |\  <-- "subarea_1_12_tri"
#         # |_\                   |_\
#         # |  \  Node 2          | |\  <-- 3 x subarea_1_12_tri
#         # |___\                 |_|_\
#         # |    \  Node 3        | | |\  <-- 5 x subarea_1_12_tri
#         # |_____\               |_|_|_\
#         # |      \  Node 4      | | | |\  <-- 7 x subarea_1_12_tri
#         # |_______\             |_|_|_|_\
#
#         # Node areas
#         subarea_1_12_tri = duct_ftf[0]**2 / 128.0 / _sqrt3
#         self.node['area'] = np.array([subarea_1_12_tri,
#                                       3 * subarea_1_12_tri,
#                                       5 * subarea_1_12_tri,
#                                       7 * subarea_1_12_tri])
#         self.node['area_frac'] = np.array([self.node['area'][i]
#                                            / np.sum(self.node['area'])
#                                            for i in
#                                            range(len(self.node['area']))
#                                            ])
#         # Distances between nodes; assume each node is located at the
#         # center of its cell
#         a = duct_ftf[0] / _sqrt3  # hex side
#         x = duct_ftf[0] / 8
#         self.node['d'] = np.zeros((4, 4))
#         # Distances to other nodes within the 1/12 tri: constant
#         for i in range(1, 3):
#             self.node['d'][i - 1, i] = x
#             self.node['d'][i, i - 1] = x
#             self.node['d'][i + 1, i] = x
#             self.node['d'][i, i + 1] = x
#         # Distances to other nodes on the same ring
#         for i in range(4):
#             self.node['d'][i, i] = i * a / 8 + a / 24
#
#         self.x_pts = np.array([-0.5, 0.5])  # only pts are along hex face
#         # self.x_pts = np.array([-1, -0.5, 0.5, 1])
#
#         self._temp = np.zeros((12, 4))
#
#     @property
#     def temp(self):
#         return self._temp
#
#     def calc_temp(self, dz, power, avg_duct_temp):
#         """."""
#         # Split power up into nodal quantities: power-per-node
#         ppn = self.node['area_frac'] * power / 12
#
#         # ... but how do I get the previous temperatures? can't just
#         # use the average of the subchannel temperatures. But if I
#         # define a new array of temperatures here, how do I project
#         # those back to the structure used in the Assembly object?
#         # This is pretty messy and requires a bit more care.


########################################################################


def calculate_min_dz(reg, temp_lo, temp_hi, adiabatic_duct=False):
    """Evaluate dz for the bundle at the assembly inlet and outlet
    temperatures; minimum value is taken to be the constraint

    Parameters
    ----------
    reg : DASSH unrodded region object
        Contains assembly geometry specifications
    temp_lo : float
        Assembly inlet temperature
    temp_hi : float
        Assembly outlet temperature (specified or estimated)

    Returns
    -------
    float
        Minimum required dz for stability at any temperature

    Notes
    -----
    Temperatures in KELVIN
    Currently only SingleNodeHomogeneous objects (model='simple')

    """
    min_dz = []
    # Hold the original value of the temperature to reset after
    _temp_in = reg.coolant.temperature
    for temp in [temp_lo, temp_hi]:
        # Interior coolant parameters and dz requirement
        reg._update_coolant_params(temp, use_mat_tracker=False)
        if reg.model == 'simple':
            if not adiabatic_duct:
                if reg._conv_approx:
                    R = (0.5 * reg.duct_thickness
                         / reg.duct.thermal_conductivity)
                    R += 1 / reg.coolant_params['htc']
                    dz = (reg.flow_rate
                          * reg.coolant.heat_capacity
                          * R
                          / reg.duct_perim
                          / reg.mratio)
                else:
                    # dz = (reg.flow_rate * reg.coolant.heat_capacity
                    #       / reg.coolant_params['htc'] / reg._params['xhtc']
                    #       / reg.duct_perim)
                    dz = (reg.flow_rate * reg.coolant.heat_capacity
                          / reg.coolant_params['htc'] / reg.duct_perim
                          / reg.mratio)
            else:
                dz = 1.0  # 1 meter is v big
        elif reg.model == '6node':
            term1 = (2 * reg.coolant.thermal_conductivity
                     * reg._cond['const']
                     / reg._scfr
                     / reg.coolant.heat_capacity)
            term2 = 0.0
            if not adiabatic_duct:
                if reg._conv_approx:
                    reg.duct.update(temp)
                    R = (0.5 * reg.duct_thickness
                         / reg.duct.thermal_conductivity)
                    # R += 1 / reg.coolant_params['htc'] / reg._params['xhtc']
                    R += 1 / reg.coolant_params['htc']
                    term2 = (reg.duct_perim_over_6
                             * reg.mratio
                             / R
                             / reg._scfr
                             / reg.coolant.heat_capacity)
                else:
                    term2 = (reg.duct_perim_over_6
                             * reg.coolant_params['htc']
                             * reg.mratio
                             / reg._scfr
                             / reg.coolant.heat_capacity)
            dz = 1 / (term1 + term2)
        else:
            raise NotImplementedError('yolo')
        min_dz.append(dz)

    # Reset the coolant temperature
    reg._update_coolant_params(_temp_in, use_mat_tracker=False)
    return min(min_dz), 0
