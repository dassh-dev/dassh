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
date: 2021-06-09
author: matz
Methods to describe the components of hexagonal fuel typical of liquid
metal fast reactors.
"""
########################################################################
import re
import sys
import copy
import numpy as np
# import warnings
import logging
from dassh.pin import PinLattice
from dassh.subchannel import Subchannel
from dassh.logged_class import LoggedClass
from dassh.correlations import check_correlation
from dassh.region import DASSH_Region
from dassh.fuel_pin import FuelPin


_sqrt3 = np.sqrt(3)
_inv_sqrt3 = 1 / _sqrt3
_sqrt3over3 = np.sqrt(3) / 3
# Surface of pins in contact with each type of subchannel
# q_p2sc = {1: 0.166666666666667, 2: 0.25, 3: 0.166666666666667}
q_p2sc = np.array([0.166666666666667, 0.25, 0.166666666666667])

module_logger = logging.getLogger('dassh.assembly')


def make_rr_asm(asm_input, name, mat_dict, flow_rate, se2geo=False):
    """Create RoddedRegion object within DASSH Assembly"""
    rr = RoddedRegion(name,
                      asm_input['num_rings'],
                      asm_input['pin_pitch'],
                      asm_input['pin_diameter'],
                      asm_input['wire_pitch'],
                      asm_input['wire_diameter'],
                      asm_input['clad_thickness'],
                      asm_input['duct_ftf'],
                      flow_rate,
                      mat_dict['coolant'],
                      mat_dict['duct'],
                      asm_input['htc_params_duct'],
                      asm_input['corr_friction'],
                      asm_input['corr_flowsplit'],
                      asm_input['corr_mixing'],
                      asm_input['corr_nusselt'],
                      asm_input['bypass_gap_flow_fraction'],
                      asm_input['bypass_gap_loss_coeff'],
                      asm_input['wire_direction'],
                      asm_input['shape_factor'],
                      se2geo)

    # Add z lower/upper boundaries
    rr.z = [asm_input['AxialRegion']['rods']['z_lo'],
            asm_input['AxialRegion']['rods']['z_hi']]

    # Add fuel pin model, if requested
    if 'FuelModel' in asm_input.keys():
        if asm_input['FuelModel']['htc_params_clad'] is None:
            p2d = (asm_input['pin_pitch']
                   / asm_input['pin_diameter'])
            asm_input['FuelModel']['htc_params_clad'] = \
                [p2d**3.8 * 0.01**0.86 / 3.0,
                 0.86, 0.86, 4.0 + 0.16 * p2d**5]
        rr.pin_model = FuelPin(asm_input['pin_diameter'],
                               asm_input['clad_thickness'],
                               mat_dict['clad'],
                               asm_input['FuelModel'],
                               mat_dict['gap'])
        # Only the last 6 columns are for data:
        # (local avg coolant temp, clad OD/MW/ID, fuel OD/CL);
        # The first 4 columns are for identifying crap:
        # (id, z (remains blank), pin number)
        rr.pin_temps = np.zeros((rr.n_pin, 9))
        # Fill with pin numbers
        rr.pin_temps[:, 2] = np.arange(0, rr.n_pin, 1)
    return rr


class RoddedRegion(LoggedClass, DASSH_Region):
    """DASSH assembly object

    Parameters
    ----------
    name : str
        Assembly name
    n_ring : int
        Number of fuel pin rings in the bundle
    pin_pitch : float
        Fuel pin center-to-center pin_pitch distance (m)
    pin_diam : float
        Outer diameter of the fuel pin cladding (m)
    wire_pitch : float
        Axial distance required for wire wrap to make
        one full revolution around fuel pin (m)
    wire_diam : float
        Wire wrap diameter (m)
    clad_thickness : float
        Cladding thickness (m)
    duct_ftf : list
        List of tuples containing the inner and outer duct flat-
        to-flat distances for each duct surrounding the bundle
    flow_rate : float
        User-specified bulk mass flow rate (kg/s) in the assembly
    coolant_mat : DASSH Material object
        Container for coolant material properties
    duct_mat : DASSH Material object
        Container for duct material properties
    htc_params_duct : list
        Dittus Boelter correlation parameter for duct wall Nu
    corr_friction : str {'NOV', 'REH', 'ENG', 'CTD', 'CTS', 'UCTD'}
        Correlation for bundle friction factor; "CTD" is recommended.
    corr_flowsplit : str {'NOV', 'MIT', 'CTD', 'UCTD'}
        Correlation for subchannel flow split; "CTD" is recommended
    corr_mixing : str {'MIT', 'CTD'}
        Correlation for subchannel mixing params; "CTD" is recommended
    corr_nusselt : str (optional) {'DB'}
        Correlation for Nu; "DB" (Dittus-Boelter) is recommended
    byp_ff : float (optional)
        Unused
    byp_k : float (optional)
        Unused
    wwdir : str {'clockwise', 'counterclockwise'}
        Wire wrap direction
    sf : float (optional)
        Shape factor multiplier on coolant material thermal conductivity
    se2 : bool
        Indicate whether to use DASSH or SE2 bundle geometry definitions
        (use only when comparing DASSH and SE2)

    Attributes
    ----------
    d : dict
        Distances across and around assembly components
    L : list
        Distances between subchannel centroids
    bare_params : dict
        Parameters characterizing subchannels w/o wire wrap
    params : dict
        Parameters characterizing subchannels w/ wire wrap
    bundle_params : dict
        Bundle-average subchannel parameters

    Notes
    -----
    RoddedRegion instances have many descriptive parameters. Useful
    references for the equations used here can be found in:
    [1] N. Todreas and M. Kazimi. "Nuclear Systems II - Elements of
        Thermal Hydraulic Design". Taylor & Francis (1990).
        (specifically Appendix J)
    [2] S. K. Cheng and N. Todreas. "Hydrodynamic models and
        correlations for bare and wire-wrapped hexagonal rod bundles
        - bundle friction factors, subchannel friction factors and
        mixing parameters". Nuclear Engineering and Design 92 (1986).
    """

    def __init__(self, name, n_ring, pin_pitch, pin_diam, wire_pitch,
                 wire_diam, clad_thickness, duct_ftf, flow_rate,
                 coolant_mat, duct_mat, htc_params_duct, corr_friction,
                 corr_flowsplit, corr_mixing, corr_nusselt, byp_ff=None,
                 byp_k=None, wwdir='clockwise', sf=1.0, se2=False):
        """Instantiate RoddedRegion object"""
        # Instantiate Logger
        LoggedClass.__init__(self, 4, 'dassh.RoddedRegion')

        # Disable single-pin assemblies for now; maybe revisit later
        if n_ring == 1 or pin_pitch == 0.0:
            self.log('error', 'Single-pin assemblies not supported')

        # Make sure the pins will fit inside the assembly: DASSH
        # checks for this when the inputs are read, but this is
        # useful for development and by-hand testing.
        clr = min(duct_ftf) - (_sqrt3 * (n_ring - 1) * pin_pitch
                               + pin_diam + 2 * wire_diam)
        if clr < 0.0:  # leave a little bit of wiggle room
            clr = '{:0.6e}'.format(-clr)
            self.log('error', f'RoddedRegion {name}: Pins do not fit '
                              f'inside duct; {clr} m too big')

        # --------------------------------------------------------------
        # BUNDLE ATTRIBUTES - generate pin and subchannel attributes
        # (clad thickness needed for power calculation)
        self.name = name
        self.n_ring = n_ring
        self.pin_pitch = pin_pitch
        self.pin_diameter = pin_diam
        self.wire_pitch = wire_pitch
        self.wire_diameter = wire_diam
        self.clad_thickness = clad_thickness
        duct_ftf = np.sort(duct_ftf)
        self.duct_ftf = [duct_ftf[i:i + 2] for i in
                         range(0, len(duct_ftf), 2)]
        self.n_duct = int(len(self.duct_ftf))
        self.n_bypass = self.n_duct - 1
        self.coolant = coolant_mat
        self.duct = duct_mat

        # Conductivity shape factor: magic knob input by user to boost
        # heat transfer by conduction (not by eddy diffusivity)
        self._sf = sf

        # Flag to mark whether the assembly has very low flow such that
        # the connection between edge/corner subchannels to duct wall
        # is treated differently.
        self._conv_approx = False

        # Set up heat transfer coefficient parameters
        self.htc_params = {}
        if htc_params_duct:
            self.htc_params['duct'] = htc_params_duct
        else:
            self.htc_params['duct'] = [0.023, 0.8, 0.4, 7.0]

        # Pin and subchannel objects; contain maps and adjacency arrays
        self.pin_lattice = PinLattice(n_ring, pin_pitch, pin_diam)
        self.n_pin = self.pin_lattice.n_pin
        self.subchannel = Subchannel(n_ring, pin_pitch, pin_diam,
                                     self.pin_lattice.map,
                                     self.pin_lattice.xy,
                                     self.duct_ftf)

        # Bypass flow rate parameters; need to store as
        # attributes so I can pass them to clones
        self._byp_ff = byp_ff
        self._byp_k = byp_k

        # --------------------------------------------------------------
        # BUNDLE GEOMETRY
        n_sc = np.array([self.subchannel.n_sc['coolant']['interior'],
                         self.subchannel.n_sc['coolant']['edge'],
                         self.subchannel.n_sc['coolant']['corner']])
        tmp = calculate_geometry(n_ring, pin_pitch, pin_diam,
                                 wire_pitch, wire_diam, self.duct_ftf,
                                 n_sc, se2)
        for k in tmp.keys():
            setattr(self, k, tmp[k])

        # --------------------------------------------------------------
        # Set up x-points for treating mesh disagreement; this is for
        # use *along each hex side* including corners.
        sc_per_side = int(self.subchannel.n_sc['duct']['edge'] / 6)
        self.x_pts = np.zeros(sc_per_side + 2)
        self.x_pts[1] = (np.average(self.d['wcorner'][-1])
                         + 0.5 * self.pin_pitch)
        for j in range(2, sc_per_side + 1):
            self.x_pts[j] = self.x_pts[j - 1] + self.pin_pitch
        self.x_pts[-1] = (self.x_pts[-2] + self.pin_pitch / 2
                          + np.average(self.d['wcorner'][-1]))
        # Scale the points to be on the range [-1, 1]
        self.x_pts = self.x_pts * 2.0 / self.x_pts[-1] - 1

        # Initialize xparam/yparam dicts; these are overwritten later
        self._xparams = {'duct2gap': None, 'gap2duct': None}
        self._yparams = {'duct2gap': None, 'gap2duct': None}

        # --------------------------------------------------------------
        # Set up pin to subchannel power fractions; accessed in the
        # assignment of pin power to coolant (and the determination of
        # pin-adjacent average coolant temperature)
        self._q_p2sc = q_p2sc[self.subchannel.type[
            :self.subchannel.n_sc['coolant']['total']]]

        # Swirl direction: depending on the direction the wire-wrap is
        # wound, the swirl will either flow clockwise or counter-
        # clockwise when viewed from above. This index points to the
        # adjacent subchannel from which swirl flow comes, as tracked
        # in self.subchannel.sc_adj (cols 3 and 4 are connections to
        # other edge/corners)
        self.wire_direction = wwdir
        if wwdir == 'clockwise':
            self._adj_sw = 3
        else:
            self._adj_sw = 4

        # --------------------------------------------------------------
        # Set up other parameters
        self._setup_region()
        self._setup_flowrate(flow_rate)
        self._setup_ht_constants()
        # if self.n_ring == 1:
        #     self._cleanup_1pin()
        self._setup_correlations(corr_friction, corr_flowsplit,
                                 corr_mixing, corr_nusselt)

    ####################################################################
    # SETUP METHODS
    ####################################################################

    def _setup_region(self):
        """Set up and inherit DASSH Region object"""
        # Inherit DASSH_Region
        # From DASSH_Region, RoddedRegion inherits temperature arrays,
        # methods to calculate average temperatures, and the ability
        # to activate itself within the Assembly child class.
        coolant_area = np.array([self.params['area'][i] for i in
                                 self.subchannel.type if i <= 2])
        duct_area = np.zeros((self.n_duct,
                              self.subchannel.n_sc['duct']['total']))
        for i in range(self.n_duct):
            start = (self.subchannel.n_sc['coolant']['total']
                     + 2 * i * self.subchannel.n_sc['duct']['total'])
            for j in range(self.subchannel.n_sc['duct']['total']):
                typ = self.subchannel.type[j + start]
                duct_area[i, j] = self.duct_params['area'][i, typ - 3]

        if self.n_bypass > 0:
            byp_area = np.zeros((self.n_bypass,
                                 self.subchannel.n_sc['bypass']['total']))
            for i in range(self.n_bypass):
                start = (self.subchannel.n_sc['coolant']['total']
                         + self.subchannel.n_sc['duct']['total']
                         + i * 2 * self.subchannel.n_sc['duct']['total'])
                for j in range(self.subchannel.n_sc['bypass']['total']):
                    typ = self.subchannel.type[j + start]
                    byp_area[i][j] = self.bypass_params['area'][i][typ - 5]
        else:
            byp_area = None

        DASSH_Region.__init__(self,
                              self.subchannel.n_sc['coolant']['total'],
                              coolant_area,
                              self.subchannel.n_sc['duct']['total'],
                              duct_area, self.n_bypass,
                              byp_area)

    def calculate_xbnds(self):
        """Calculate boundaries between duct elements

        Notes
        -----
        First array entry =0 (center of the top corner duct element)
        The following entries "walk" around the duct, marking the
            boundaries between elements
        The last entry is the duct perimeter (meets back up with the
            first entry); the top duct corner is "split" in half
            between the first/last entries.

        """
        x_bnds = np.zeros(self.subchannel.n_sc['duct']['total'] + 2)
        typ = self.subchannel.type[
            -self.subchannel.n_sc['duct']['total']:].copy()
        typ -= 3
        typ = np.roll(typ, 1)
        dx = np.array([self.pin_pitch, 2 * self.d['wcorner'][-1, 1]])
        x_bnds[1:-1] = np.cumsum(dx[typ]) - 0.5 * dx[1]
        x_bnds[-1] = 6 / np.sqrt(3) * self.duct_ftf[-1][1]
        return x_bnds

    def calculate_xbnds2(self):
        edge_sc_per_side = int(self.subchannel.n_sc['duct']['edge'] / 6)
        x_bnds = np.zeros(edge_sc_per_side + 1)
        typ = self.subchannel.type[
            -self.subchannel.n_sc['duct']['total']:].copy()
        typ -= 3
        typ = np.roll(typ, 1)
        dx = np.array([self.pin_pitch, 2 * self.d['wcorner'][-1, 1]])
        tmp = np.cumsum(dx[typ]) - 0.5 * dx[1]
        x_bnds = tmp[:x_bnds.shape[0]]
        P_hex = self.duct_ftf[-1][1] / np.sqrt(3)
        x_bnds /= P_hex
        x_bnds[0] = 0
        x_bnds[-1] = 1
        return x_bnds

    def _cleanup_1pin(self):
        """Clean up the parameters generated in a 1-pin assembly

        Notes
        -----
        The methods in __init__ produce some incorrect parameter values
        when applied to a 1-pin assembly. They shouldn't be accessible
        because the "subchannel" attribute counts no interior/edge-type
        subchannels, but eliminating those values that are incorrect
        will help avoid headaches and confusion later

        The following attributes are unaffected:
        - bundle_params
        - duct_params

        The affected values are generally negative/inf/nan and are set
        to zero below.

        """
        # Single values
        self.d['pin-pin'] = 0.0
        self.bypass_params['de'][:, 0] = 0.0

        # Dicts with lists: first two values should be zero
        for attr in ['bare_params', 'params']:
            tmp_attr = getattr(self, attr)
            for key in tmp_attr:
                if key != 'theta':
                    # Last value is always corner, keep it
                    for idx in range(len(tmp_attr[key]) - 1):
                        tmp_attr[key][idx] = 0.0
            setattr(self, attr, tmp_attr)

        # Lists of lists - pick indices that point corners -> corners
        ij_pairs = [(2, 2), (2, 4), (4, 2), (4, 6), (6, 4)]
        for attr in ['L', 'ht_consts']:
            tmp_attr = getattr(self, attr)
            for i in range(len(tmp_attr)):
                for j in range(len(tmp_attr[i])):
                    if not (i, j) in ij_pairs:
                        if type(tmp_attr[i][j]) == list:
                            tmp_attr[i][j] = [0.0] * len(tmp_attr[i][j])
                        else:
                            tmp_attr[i][j] = 0.0
            setattr(self, attr, tmp_attr)

    def _setup_flowrate(self, flowrate):
        """Setup method to define the Assembly flowrate"""
        # What I will know is the total flow rate given to the
        # assembly; need to determine bypass flow rate based on area
        # fraction. This will cause the interior flow rate to decrease!
        # Based on flow area, send some flow to the bypass channels
        self.total_flow_rate = flowrate
        if self.n_bypass == 0.0:
            self.int_flow_rate = flowrate

        # Otherwise, setup bypass flow rate
        else:
            # First: try using loss coefficient to calculate pressure
            # drop; use to determine required flow rate
            if self._byp_k is not None:
                raise NotImplementedError('Loss coefficient input'
                                          'not yet supported')
                # Need to determine bypass flow rates based on pressure
                # drop, which needs to be equal across all flow paths

                # Method:
                # 1) guess flow rate in each flow path
                # 2) Calculate velocities, Re, friction factors
                # 3) Calculate dP across core for each path
                # 4) Repeat; adjust flow rates to try to equate dP

                # First guess: 10%
                self.int_flow_rate = 0.9 * flowrate
                self.byp_flow_rate = (np.ones(self.n_bypass) * flowrate
                                      * 0.1 / self.n_bypass)

            else:
                # Use flow fraction value to directly assign flow rate
                self.byp_flow_rate = (np.ones(self.n_bypass) * flowrate
                                      * self._byp_ff / self.n_bypass)
                self.int_flow_rate = flowrate * (1 - self._byp_ff)

        # Mass flow rate constants: use this each time you need to
        # calculate the average coolant temperature, rather than doing
        # the multiplications each time
        self._mfrc = (self.params['area']
                      * self.int_flow_rate
                      / self.bundle_params['area'])

    def _iterate_bypass_flowrate(self, z0, z1, k_seal=[0]):
        # diff in dP between bundle and bypass
        self._k_byp_seal = np.array(k_seal)
        dz = z1 - z0
        diff = np.ones(self.n_bypass)
        # while np.sum(np.abs(diff)) > 1e-6:
        print('{:4s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}'
              .format('iter', 'Int FR', 'Byp FR', 'Int dP', 'Byp dP', 'Diff'))
        A_duct = self.duct_ftf[-1][0]**2 * _sqrt3 / 2
        v_tot = self.total_flow_rate / self.coolant.density / A_duct
        seal_loss = self._k_byp_seal * self.coolant.density * v_tot**2 / 2
        for i in range(20):
            self._update_coolant_int_params(self.avg_coolant_int_temp)
            self._update_coolant_byp_params(self.avg_coolant_byp_temp)
            # to_print.append(self.coolant_int_params['ff'])
            # to_print.append(self.coolant_byp_params['ff'])

            dP_int = self.calculate_pressure_drop(dz)
            dP_byp = self.calculate_byp_pressure_drop(dz)
            dP_byp += seal_loss
            diff = dP_byp - dP_int
            print('{:<4d} {:>8.3f} {:>8.3f} {:>8.1f} {:>8.1f} {:>8.3f}'
                  .format(i, self.int_flow_rate, self.byp_flow_rate[0],
                          dP_int, dP_byp[0], diff[0]))

            dP_target = np.average([dP_int, np.average(dP_byp)])

            v_byp_target = np.sqrt(2 * self.bypass_params['total de']
                                   * (dP_target - seal_loss)
                                   / self.coolant_byp_params['ff']
                                   / dz
                                   / self.coolant.density)
            self.byp_flow_rate = (v_byp_target
                                  * self.bypass_params['total area']
                                  * self.coolant.density)
            self.int_flow_rate = (self.total_flow_rate
                                  - np.sum(self.byp_flow_rate))

    def _setup_ht_constants(self):
        """Setup heat transfer constants in numpy arrays"""
        const = calculate_ht_constants(self)
        # self.ht_consts = const
        self.ht = {}
        self.ht['old'] = const
        self.ht['q_denom'] = (self.int_flow_rate
                              * self.params['area']
                              / self.bundle_params['area'])
        self.ht['q_denom'] = \
            self.ht['q_denom'][
                self.subchannel.type[
                    :self.subchannel.n_sc['coolant']['total']]]
        self.ht['swirl'] = (self.d['pin-wall']
                            * self.bundle_params['area']
                            / self.params['area']
                            / self.int_flow_rate)
        self.ht['cond'] = _setup_conduction_constants(self, const)
        self.ht['conv'] = _setup_convection_constants(self, const)

    def _setup_correlations(self, ff, fs, mix, nu, raise_warnings=True):
        """Import correlations and load any constants

        Parameters
        ----------
        ff : str
            Friction factor correlation name
        fs : str
            Flow split correlation name
        mix : str
            Mixing parameters correlation name
        nu : str
            Nusselt number correlation name
        raise_warnings : boolean
            If True, raise warnings accumulated when importing
            correlations (default = True)

        """
        self.corr, self.corr_names, self.corr_constants = \
            import_corr(ff, fs, mix, nu, self, raise_warnings)
        self.coolant_int_params = \
            {'Re': 0.0,  # bundle-average Reynolds number
             'Re_sc': np.zeros(3),  # subchannel Reynolds numbers
             'vel': 0.0,  # bundle-average coolant velocity
             'fs': np.ones(3),  # subchannel flow split parameters
             'ff': np.zeros(3),  # subchannel friction factors
             'eddy': 0.0,  # eddy diffusivity
             'swirl': np.zeros(3),  # swirl velocity.
             'htc': np.zeros(3)}  # heat transfer coefficient
        if self.n_bypass > 0:
            self.coolant_byp_params = \
                {'Re': np.zeros(self.n_bypass),  # bypass-avg Re numbers
                 'Re_sc': np.zeros((self.n_bypass, 2)),  # bypass sc Re
                 'vel': np.zeros(self.n_bypass),  # bypass-avg velocities
                 'ff': np.zeros(self.n_bypass),  # byp sc fric. fracs.
                 'htc': np.zeros((self.n_bypass, 2))}  # byp sc htc

    def _determine_byp_flow_rate(self):
        """Calculate the bypass flow rate by estimating pressure drop
        in the bundle and determining what flow rate is required to
        achieve equal pressure drop in the bypass"""
        # Start by assuming "area-based split"
        self._update_coolant_int_params(self.avg_coolant_int_temp)

        return

    def clone(self, new_flowrate=None):
        """Clone the rodded region into another assembly object;
        shallow copy some attributes, deep copy others"""
        # Create a shallow copy (stuff like the pin and subchannel
        # objects can simply be pointed to)
        clone = copy.copy(self)

        # Overwrite specific attributes that need to be unique
        # for each clone; use temperature at "current" clone point
        clone.temp = copy.deepcopy(self.temp)
        clone.ebal = copy.deepcopy(self.ebal)
        if hasattr(self, 'pin_temps'):
            clone.pin_temps = copy.deepcopy(self.pin_temps)

        clone._setup_correlations(self.corr_names['ff'],
                                  self.corr_names['fs'],
                                  self.corr_names['mix'],
                                  self.corr_names['nu'],
                                  raise_warnings=False)
        if new_flowrate is not None:
            # Define new flow rate attribute in clone
            clone._setup_flowrate(new_flowrate)
            # New flow rate attributes used in new heat transfer consts
            clone._setup_ht_constants()

        return clone

    ####################################################################
    # ATTRIBUTES
    ####################################################################

    @property
    def sc_mfr(self):
        """Return mass flow rate in each subchannel"""
        mfr = self._mfrc * self.coolant_int_params['fs']
        mfr = mfr[
            self.subchannel.type[
                :self.subchannel.n_sc['coolant']['total']]]
        return mfr

    @property
    def avg_coolant_int_temp(self):
        """Mass-flow-rate-weighted average coolant temperature
        in the interior flow region"""
        # If no data for flow split, weight by subchannel area
        # Need flow split to get mass flow rate to get avg temp, but
        # need avg temp to get flow split!
        if np.all(self.coolant_int_params['fs'] == 0):
            # return (np.sum(self.temp['coolant_int']
            #                * self.area['coolant_int'])
            #         / self.total_area['coolant_int'])
            return (np.dot(self.temp['coolant_int'],
                           self.area['coolant_int'])
                    / self.total_area['coolant_int'])
        else:
            # return (np.sum(self.sc_mfr * self.temp['coolant_int'])
            #         / self.int_flow_rate)
            return (np.dot(self.sc_mfr, self.temp['coolant_int'])
                    / self.int_flow_rate)

    @property
    def avg_coolant_temp(self):
        """Weighted average coolant temperature
        for the interior and bypass flow regions"""
        if not hasattr(self, 'byp_flow_rate'):
            return self.avg_coolant_int_temp
        elif np.sum(self.byp_flow_rate) == 0.0:
            return self.avg_coolant_int_temp
        else:
            # Currently, bypass mass flow rate is based on subchannel
            # area. Therefore, can just use the area-weighted function
            # implemented in region.py
            tot = np.sum(self.temp['coolant_byp']
                         * self.area['coolant_byp'])

            # If you don't have flow split, you can't use mass flow
            # rates as weights; use subchannel area instead
            if np.all(self.coolant_int_params['fs'] == 0):
                tot += (self.avg_coolant_int_temp
                        * self.total_area['coolant_int'])
                avg = tot / (self.total_area['coolant_int']
                             + self.total_area['coolant_byp'])
            else:
                tot *= self.byp_flow_rate / self.total_area['coolant_byp']
                tot += self.avg_coolant_int_temp * self.int_flow_rate
                avg = tot / self.total_flow_rate
            return avg[0]  # np array --> float

    ####################################################################
    # UPDATE PROPERTIES
    ####################################################################

    def _update_coolant_int_params(self, temperature):
        """Update correlated bundle coolant parameters based
        on current average coolant temperature

        Parameters
        ----------
        temperature : float
            Average coolant temperature

        Notes
        -----
        Updates coolant_params dict attribute with keys:
            'ff': friction factor
            'fs': flow split
            'htc': heat transfer coefficient
            'eddy': eddy diffusivity
            'swirl': swirl velocity

        """
        self.coolant.update(temperature)

        self.coolant_int_params['vel'] = \
            (self.int_flow_rate / self.coolant.density
             / self.bundle_params['area'])

        # Bundle Reynolds number
        self.coolant_int_params['Re'] = \
            (self.int_flow_rate * self.bundle_params['de']
             / self.coolant.viscosity / self.bundle_params['area'])

        # Flow split parameters
        if self.corr['fs'] is not None:
            self.coolant_int_params['fs'] = self.corr['fs'](self)

        # Subchannel Reynolds numbers
        tmp = (self.coolant.density * self.coolant_int_params['vel']
               / self.coolant.viscosity)
        self.coolant_int_params['Re_sc'][0] = \
            tmp * self.coolant_int_params['fs'][0] * self.params['de'][0]
        self.coolant_int_params['Re_sc'][1] = \
            tmp * self.coolant_int_params['fs'][1] * self.params['de'][1]
        self.coolant_int_params['Re_sc'][2] = \
            tmp * self.coolant_int_params['fs'][2] * self.params['de'][2]

        # Heat transfer coefficient (via Nusselt number)
        nu = self.corr['nu'](self.coolant,
                             self.coolant_int_params['Re_sc'],
                             self.htc_params['duct'])
        self.coolant_int_params['htc'] = \
            self.coolant.thermal_conductivity * nu / self.params['de']
        # self.coolant_int_params['htc'][0] = \
        #     self.coolant.thermal_conductivity * nu[0] / self.params['de'][0]
        # self.coolant_int_params['htc'][1] = \
        #     self.coolant.thermal_conductivity * nu[1] / self.params['de'][1]
        # self.coolant_int_params['htc'][2] = \
        #     self.coolant.thermal_conductivity * nu[2] / self.params['de'][2]

        # 2021-03-09 MILOS MODIFICATION
        # self.coolant_int_params['htc'] = np.array([1e5, 1e5, 1e5])

        # Friction factor
        if self.corr['ff'] is not None:
            self.coolant_int_params['ff'] = self.corr['ff'](self)

        # Mixing params - these come dimensionless, need to adjust
        if self.corr['mix'] is not None:
            mix = self.corr['mix'](self)
            self.coolant_int_params['eddy'] = \
                (mix[0] * self.coolant_int_params['fs'][0]
                 * self.coolant_int_params['vel'])
            tmp = (mix[1] * self.coolant_int_params['vel']
                   * self.coolant_int_params['fs'][1])
            self.coolant_int_params['swirl'][1] = tmp
            self.coolant_int_params['swirl'][2] = tmp

        # 2021-03-09 MILOS MODIFICATION
        # self.coolant_int_params['eddy'] = 0.0
        # self.coolant_int_params['swirl'] = np.zeros(3)

    def _update_coolant_byp_params(self, temp_list):
        """Update correlated bundle bypass coolant parameters based
        on current average coolant temperature

        Parameters
        ----------
        temp_list : list
            Average coolant temperature in bypass channels

        """
        for i in range(self.n_bypass):
            self.coolant.update(temp_list[i])

            if np.sum(self.byp_flow_rate) > 0.0:
                # Bypass velocity
                self.coolant_byp_params['vel'][i] = \
                    (self.byp_flow_rate[i]
                     / self.coolant.density
                     / self.bypass_params['total area'][i])
                # Bypass reynolds number
                self.coolant_byp_params['Re'][i] = \
                    (self.byp_flow_rate[i]
                     * self.bypass_params['total de'][i]
                     / self.coolant.viscosity
                     / self.bypass_params['total area'][i])
                # Subchannel Reynolds numbers
                self.coolant_byp_params['Re_sc'][i] = \
                    (self.coolant.density * self.bypass_params['de'][i]
                     * self.coolant_byp_params['vel'][i]
                     / self.coolant.viscosity)
                # Heat transfer coefficient (via Nusselt number); Nu
                # is the same for both concentric duct walls because
                # they are the same material and the flow condition
                # is the same on both; the only difference will be the
                # the surface area, which is accounted for in the
                # temperature calculation
                nu = self.corr['nu'](
                    self.coolant,
                    self.coolant_byp_params['Re_sc'][i]
                )
                self.coolant_byp_params['htc'][i] = \
                    (self.coolant.thermal_conductivity
                     * nu / self.bypass_params['de'][i])

                # Friction factor
                k = 1e-6  # Absolute roughness coefficient (m)
                if self.coolant_byp_params['Re'][i] <= 2200.0:  # Laminar
                    f = 96.0 / self.coolant_byp_params['Re'][i]
                else:
                    c1 = k / self.bypass_params['total de'][i] / 3.7
                    c2 = 4.518 / self.coolant_byp_params['Re'][i]
                    c3 = 6.9 / self.coolant_byp_params['Re'][i]
                    f = (-0.5
                         / (np.log10(
                            c1 - c2 * np.log10(
                                c3 + c1**1.11)))
                         )**2
                    if self.coolant_byp_params['Re'][i] < 3000.0:  # Turbulent
                        f2200 = 96.0 / 2200.0
                        x = (3.75 - 8250.0
                             / self.coolant_byp_params['Re'][i])
                        f = f2200 + x * (f - f2200)
                self.coolant_byp_params['ff'][i] = f

            else:  # stagnant bypass gap coolant
                self.coolant_byp_params['htc'][i] = \
                    (self.coolant.thermal_conductivity
                     / (0.5 * np.array([self.d['bypass'][i],
                                        self.d['bypass'][i]])))
                self.coolant_byp_params['ff'][i] = 0.0

    ####################################################################
    # PRESSURE DROP
    ####################################################################

    def calculate_pressure_drop(self, dz):
        """Calculate pressure drop across bundle with current step"""
        return (self.coolant_int_params['ff'] * dz
                * self.coolant.density
                * self.coolant_int_params['vel']**2
                / 2 / self.bundle_params['de'])

    def calculate_byp_pressure_drop(self, dz):
        """Calculate bypass pressure drop with current step"""
        # need to update coolant temeperature in loop with each
        # bypass channel (how will I do this for real?)
        # In initial tests: don't need to update at all because
        # just using inlet parameters

        # Note: self.coolant_byp_params and self.bypass_params values
        # are arrays (len = n_bypass)
        loss_coeff = (self.coolant_byp_params['ff'] * dz
                      / self.bypass_params['total de'])
        return (loss_coeff
                * self.coolant.density
                * self.coolant_byp_params['vel']**2
                / 2)

    ####################################################################
    # TEMPERATURE CALCULATION
    ####################################################################

    def calculate(self, dz, q, t_gap, htc_gap, adiabatic=False, ebal=False):
        """Calculate new coolant and duct temperatures and pressure
        drop across axial step

        Parameters
        ----------
        dz : float
            Axial step size (m)
        q : dict
            Power (W/m) generated in pins, duct, and coolant
        t_gap : numpy.ndarray
            Interassembly gap temperatures around the assembly at the
            j+1 axial level (array length = n_sc['duct']['total'])
        htc_gap : float
            Heat transfer coefficient for convection between the gap
            coolant and outer duct wall based on core-average inter-
            assembly gap coolant temperature
        adiabatic : boolean
            Indicate whether outer duct has adiabatic BC
        ebal : boolean
            Indicate whether to track energy balance

        Returns
        -------
        None

        """
        # Interior coolant temperatures: calculate using coolant
        # properties from previous axial step
        self.temp['coolant_int'] += \
            self._calc_coolant_int_temp(dz, q['pins'], q['cool'], ebal)

        # Update coolant properties for the duct wall calculation
        self._update_coolant_int_params(self.avg_coolant_int_temp)

        # Bypass coolant temperatures
        if self.n_bypass > 0:
            if self.byp_flow_rate > 0:
                self.temp['coolant_byp'] += \
                    self._calc_coolant_byp_temp(dz, ebal)
            else:
                self.temp['coolant_byp'] += \
                    self._calc_coolant_byp_temp_stagnant(dz, ebal)
            # Update bypass coolant properties for the duct wall calc
            self._update_coolant_byp_params(self.avg_coolant_byp_temp)

        # Duct temperatures: calculate with new coolant properties
        self._calc_duct_temp(q['duct'], t_gap, htc_gap, adiabatic)

        # Update pressure drop (now that correlations are updated)
        self._pressure_drop += self.calculate_pressure_drop(dz)

    def activate2(self, previous_reg, t_gap, h_gap, adiabatic):
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
        # temperatures and input gap temperatures / HTC. Assume zero
        # power: previous region was not a pin bundle and therefore
        # did not have power generated in the duct.
        p_duct = np.zeros(self.temp['duct_mw'].size)
        self._calc_duct_temp(p_duct, t_gap, h_gap, adiabatic)

    ####################################################################
    # COOLANT TEMPERATURE CALCULATION METHODS
    ####################################################################

    def _calc_coolant_int_temp(self, dz, q_pins, q_cool, ebal=False):
        """Calculate assembly coolant temperatures at next axial mesh

        Parameters
        ----------
        dz : float
            Axial step size (m)
        q_pins : numpy.ndarray
            Linear power generation (W/m) for each pin in the assembly
        q_cool : numpy.ndarray
            Linear power generation (W/m) for each coolant subchannel
        ebal : boolean
            Indicate whether to perform/update energy balance

        Returns
        -------
        numpy.ndarray
            Vector (length = # coolant subchannels) of temperatures
            (K) at the next axial level

        """
        # Calculate avg coolant temperature; update coolant properties
        # self._update_coolant_int_params(self.avg_coolant_int_temp)

        # Update coolant material properties; correlated parameters
        # were updated after the previous step
        # T_avg = self.avg_coolant_int_temp
        # self.coolant.update(T_avg)

        # HEAT FROM ADJACENT FUEL PINS
        # denom puts q in the same units as the next dT steps
        q = self._calc_int_sc_power(q_pins, q_cool)
        dT = q / self.ht['q_denom']

        # CONDUCTION BETWEEN COOLANT SUBCHANNELS
        # Effective thermal conductivity
        keff = (self.coolant_int_params['eddy']
                * self.coolant.density
                * self.coolant.heat_capacity
                + self._sf * self.coolant.thermal_conductivity)
        # keff = 0.0
        tmp = (self.ht['cond']['const']
               * (self.temp['coolant_int'][self.ht['cond']['adj']]
                  - self.temp['coolant_int'][:, np.newaxis]))
        # dT += keff * np.sum(tmp, axis=1)
        dT += keff * (tmp[:, 0] + tmp[:, 1] + tmp[:, 2])

        # CONVECTION BETWEEN EDGE/CORNER SUBCHANNELS AND DUCT WALL
        # Heat transfer coefficient
        tmp = self.coolant_int_params['htc'][self.ht['conv']['type']]
        # Low flow case: use SE2ANL model
        if self._conv_approx:
            # Resistance between coolant and duct MW
            self.duct.update(self.avg_duct_mw_temp[0])
            # R1 = 1 / h; R2 = dw / 2 / k (half wall thickness over k)
            # R units: m2K / W; heat transfer area included in const
            R1 = 1 / tmp  # R1 = 1 / h
            R2 = 0.5 * self.d['wall'][0] / self.duct.thermal_conductivity
            # tmp += 0.5 * self.d['wall'][0] / self.duct.thermal_conductivity
            # dT[self._conv['ind']] += \
            #     (self._conv['const'] / tmp
            #      * (self.temp['duct_mw'][0, self._conv['adj']]
            #         - self.temp['coolant_int'][self._conv['ind']]))
            # dT[self._conv['ind']] += \
            #     (self._conv['const'] / (R1 + R2)
            #      * (self.temp['duct_mw'][0, self._conv['adj']]
            #         - self.temp['coolant_int'][self._conv['ind']]))
            dT_conv_over_R = \
                ((self.temp['duct_mw'][0, self.ht['conv']['adj']]
                  - self.temp['coolant_int'][self.ht['conv']['ind']])
                 / (R1 + R2))
        else:
            # dT[self._conv['ind']] += \
            #     tmp * self._conv['const'] * (
            #         self.temp['duct_surf'][0, 0, self._conv['adj']]
            #         - self.temp['coolant_int'][self._conv['ind']])
            dT_conv_over_R = \
                tmp * (self.temp['duct_surf'][0, 0, self.ht['conv']['adj']]
                       - self.temp['coolant_int'][self.ht['conv']['ind']])
        dT[self.ht['conv']['ind']] += \
            self.ht['conv']['const'] * dT_conv_over_R

        # DIVIDE THROUGH BY MCP
        mCp = self.coolant.heat_capacity * self.coolant_int_params['fs']
        mCp = mCp[self.subchannel.type[
            :self.subchannel.n_sc['coolant']['total']]]
        dT /= mCp

        # SWIRL FLOW AROUND EDGES (no div by mCp so it comes after)
        # Can just use the convection indices again bc they're the same
        swirl_consts = (self.ht['swirl']
                        * self.coolant.density
                        * self.coolant_int_params['swirl']
                        / self.coolant_int_params['fs'])
        swirl_consts = swirl_consts[self.subchannel.type[
            :self.subchannel.n_sc['coolant']['total']]]
        swirl_consts = swirl_consts[self.ht['conv']['ind']]
        # Swirl flow from adjacent subchannel; =0 for interior sc
        # The adjacent subchannel is the one the swirl flow is
        # coming from i.e. it's in the opposite direction of the
        # swirl flow itself. Recall that the edge/corner sub-
        # channels are indexed in the clockwise direction.
        # Example: Let sci == 26. The next subchannel in the clock-
        # wise direction is 27; the preceding one is 25.
        # - clockwise: use 25 as the swirl adjacent sc
        # - counterclockwise: use 27 as the swirl adjacent sc
        dT[self.ht['conv']['ind']] += \
            (swirl_consts
             * (self.temp['coolant_int'][self.subchannel.sc_adj[
                self.ht['conv']['ind'], self._adj_sw]]
                - self.temp['coolant_int'][self.ht['conv']['ind']]))

        if ebal:
            qduct = self.ht['conv']['ebal'] * dT_conv_over_R
            self.update_ebal(dz * np.sum(q), dz * qduct)
        return dT * dz

    def _calc_int_sc_power(self, pin_power, cool_power):
        """Determine power from pins and from direct heating in the
        coolant that gets put into each subchannel at the given axial
        point

        Parameters
        ----------
        pin_power : numpy.ndarray
            Linear power generation (W/m) for each pin in the assembly
        cool_power : numpy.ndarray
            Linear power generation (W/m) for each coolant subchannel

        Returns
        -------
        numpy.ndarray
            Array (length = number of coolant subchannels) containing
            the linear power (W/m) added to each subchannel by the
            adjacent pins.

        """
        # Null case: no power specified
        if pin_power is None and cool_power is None:
            return np.zeros(self.subchannel.n_sc['coolant']['total'])

        elif pin_power is None:  # Then cool_power is not None
            return cool_power
        else:  # Then pin_power is not None, but cool_power might be None
            # Pin power: loop over pins, put appropriate power into
            # adjacent subchannels as defined by the pin_adjacency array.
            q = pin_power[self.subchannel.rev_pin_adj]
            q[self.subchannel.rev_pin_adj < 0] = 0
            q = np.sum(q, axis=1)
            q *= self._q_p2sc
            if cool_power is not None:
                q += cool_power
            return q

    def _calc_coolant_byp_temp(self, dz, ebal=False):
        """Calculate the coolant temperatures in the assembly bypass
        channels at the axial level j+1

        Parameters
        ----------
        dz : float
            Axial step size (m)
        ebal : boolean (default = False)
            Track energy balance in the bypass gap coolant

        Notes
        -----
        The coolant in the bypass channels is assumed to get no
        power from neutron/gamma heating (that contribution to
        coolant in the assembly interior is already small enough).

        """
        # Calculate the change in temperature in each subchannel
        dT = np.zeros((self.n_bypass,
                       self.subchannel.n_sc['bypass']['total']))

        # Milos note 2020-12-09: don't need to update bypass coolant
        # params because I'm already doing it in "calculate"
        # self._update_coolant_byp_params(self.avg_coolant_byp_temp)
        T_avg = self.avg_coolant_byp_temp
        for i in range(self.n_bypass):
            # Update coolant material properties; correlated parameters
            # were updated after the previous step
            self.coolant.update(T_avg[i])

            # This factor is in many terms; technically, the mass flow
            # rate is already accounted for in constants defined earlier
            # mCp = self.coolant.heat_capacity

            # starting index to lookup type is after all interior
            # coolant channels and all preceding duct and bypass
            # channels
            start = (self.subchannel.n_sc['coolant']['total']
                     + self.subchannel.n_sc['duct']['total']
                     + i * self.subchannel.n_sc['bypass']['total']
                     + i * self.subchannel.n_sc['duct']['total'])
            end = start + self.subchannel.n_sc['bypass']['total']
            type_i = self.subchannel.type[start:end]
            htc_i = self.coolant_byp_params['htc'][i, type_i - 5]
            # byp_conv_consti = byp_conv_const[type_i - 5]
            byp_conv_const = np.array([[self.L[1][1], self.L[1][1]],
                                       [2 * self.d['wcorner'][i, 1],
                                        2 * self.d['wcorner'][i + 1, 1]]])
            # byp_conv_const = np.array([[self.L[1][1], self.L[1][1]],
            #                            [2 * self.d['wcorner_m'][i],
            #                             2 * self.d['wcorner_m'][i + 1]]])
            byp_conv_const = byp_conv_const[type_i - 5]
            byp_fr_const = (self.bypass_params['total area'][i]
                            / self.byp_flow_rate[i]
                            / self.bypass_params['area'][i])
            byp_fr_const = byp_fr_const[type_i - 5]

            # Heat transfer to/from adjacent duct walls
            if self._conv_approx:
                htc_inv = 1 / htc_i
                # Interior duct wall
                self.duct.update(self.avg_duct_mw_temp[i])
                R = htc_inv + (0.5 * self.d['wall'][i]
                               / self.duct.thermal_conductivity)
                dT_in = (byp_conv_const[:, 0] / R
                         * (self.temp['duct_mw'][i]
                            - self.temp['coolant_byp'][i]))
                # Exterior duct wall
                self.duct.update(self.avg_duct_mw_temp[i + 1])
                R = htc_inv + (0.5 * self.d['wall'][i + 1]
                               / self.duct.thermal_conductivity)
                dT_out = (byp_conv_const[:, 0] / R
                          * (self.temp['duct_mw'][i + 1]
                             - self.temp['coolant_byp'][i]))
            else:
                # Interior duct wall
                dT_in = (byp_conv_const[:, 0]
                         * htc_i
                         * (self.temp['duct_surf'][i, 1]
                            - self.temp['coolant_byp'][i]))
                # Exterior duct wall
                dT_out = (byp_conv_const[:, 1]
                          * htc_i
                          * (self.temp['duct_surf'][i + 1, 0]
                             - self.temp['coolant_byp'][i]))

            dT[i] += dT_in + dT_out
            if ebal:
                self.update_ebal_byp(i, dz * dT_in, dz * dT_out)

            # Get the flow rate component
            dT[i] *= byp_fr_const

            # Connect with other bypass coolant subchannels
            for sci in range(0, self.subchannel.n_sc['bypass']['total']):
                # The value of sci is the PYTHON indexing
                # Heat transfer to/from adjacent subchannels
                for adj in self.subchannel.sc_adj[sci + start]:
                    type_a = self.subchannel.type[adj]
                    if 3 <= type_a <= 4:
                        continue
                    else:
                        sc_adj = adj - start
                        dT[i, sci] += \
                            (self.coolant.thermal_conductivity
                             * self.ht['old'][type_i[sci]][type_a][i]
                             * (self.temp['coolant_byp'][i, sc_adj]
                                - self.temp['coolant_byp'][i, sci]))

            # Divide by average heat capacity
            dT[i] /= self.coolant.heat_capacity
        return dT * dz

    def _calc_coolant_byp_temp_stagnant(self, dz, ebal=False):
        """Calculate the coolant temperatures in the assembly bypass
        channels at the axial level j+1 assuming the DD gap is stagnant

        Parameters
        ----------
        dz : float
            Axial step size (m)
        ebal : boolean (default = False)
            Track energy balance in the bypass gap coolant

        Notes
        -----
        The coolant in the bypass channels is assumed to get no
        power from neutron/gamma heating (that contribution to
        coolant in the assembly interior is already small enough).

        """
        # Calculate the change in temperature in each subchannel
        dT = np.zeros((self.n_bypass,
                       self.subchannel.n_sc['bypass']['total']))

        # Milos note 2020-12-09: don't need to update bypass coolant
        # params because I'm already doing it in "calculate"
        # self._update_coolant_byp_params(self.avg_coolant_byp_temp)
        T_avg = self.avg_coolant_byp_temp
        for i in range(self.n_bypass):
            # Update coolant material properties; correlated parameters
            # were updated after the previous step
            self.coolant.update(T_avg[i])

            # starting index to lookup type is after all interior
            # coolant channels and all preceding duct and bypass
            # channels
            start = (self.subchannel.n_sc['coolant']['total']
                     + self.subchannel.n_sc['duct']['total']
                     + i * self.subchannel.n_sc['bypass']['total']
                     + i * self.subchannel.n_sc['duct']['total'])
            end = start + self.subchannel.n_sc['bypass']['total']
            type_i = self.subchannel.type[start:end]

            # Now do convection: just average adjacent duct temperatures
            byp_conv_const = np.array([[self.L[1][1], self.L[1][1]],
                                       [2 * self.d['wcorner'][i, 1],
                                        2 * self.d['wcorner'][i + 1, 1]]])
            byp_conv_const = byp_conv_const[type_i - 5]
            norm = np.sum(byp_conv_const, axis=1)
            byp_conv_const = (byp_conv_const
                              / norm.reshape(len(byp_conv_const), 1))

            # Heat transfer to/from adjacent duct walls
            # Interior duct wall
            dT_in = (byp_conv_const[:, 0]
                     * (self.temp['duct_surf'][i, 1]
                        - self.temp['coolant_byp'][i]))
            # Exterior duct wall
            dT_out = (byp_conv_const[:, 1]
                      * (self.temp['duct_surf'][i + 1, 0]
                         - self.temp['coolant_byp'][i]))

            dT[i] += dT_in + dT_out
            if ebal:
                self.update_ebal_byp(i, dz * dT_in, dz * dT_out)

        return dT

    ####################################################################
    # DUCT TEMPERATURE CALCULATION METHODS
    ####################################################################

    def _calc_duct_temp(self, p_duct, t_gap, htc_gap, adiabatic=False):
        """Calculate the duct wall temperatures based on the adjacent
        coolant temperatures

        Parameters
        ----------
        p_duct : numpy.ndarray
            Linear power generation (W/m) for each duct cell
            (Array size: N_duct * N_sc['duct']['total'] x 1)
        t_gap : numpy.ndarray
            Interassembly gap temperatures around the assembly at the
            j+1 axial level (array length = n_sc['duct']['total'])
        htc_gap : numpy.ndarray
            Heat transfer coefficient for convection between the gap
            coolant and outer duct wall based on core-average inter-
            assembly gap coolant temperature; len = # duct SC
        adiabatic : boolean
            Indicate if outer duct wall outer BC is adiabatic

        Returns
        -------
        None

        """
        # Average duct temperatures (only want to have to get once)
        duct_avg_temps = self.avg_duct_mw_temp
        #
        # Update coolant parameters
        # self._update_coolant_int_params(self.avg_coolant_int_temp)
        # if self.n_bypass > 0:
        #     self._update_coolant_byp_params(self.avg_coolant_byp_temp)
        #
        # If you haven't already, set up array of duct type indices
        # so you don't have to do it in the loop every time
        if not hasattr(self, '_duct_idx'):
            self._duct_idx = np.zeros((self.subchannel
                                           .n_sc['duct']['total']),
                                      dtype=int)
            for sci in range(self.subchannel.n_sc['duct']['total']):
                self._duct_idx[sci] = \
                    self.subchannel.type[
                        (sci + self.subchannel.n_sc['coolant']['total'])]
            self._duct_idx -= 3

        for i in range(self.n_duct):
            # No params to update, just need avg duct temp
            self.duct.update(duct_avg_temps[i])
            # start = i * self.subchannel.n_sc['duct']['total']
            # end = (i + 1) * self.subchannel.n_sc['duct']['total']
            # # Convert linear power (W/m) to volumetric power (W/m^3)
            # # (qtp == q triple prime); value is very large
            # qtp = (p_duct[start:end]
            #        / self.duct_params['area'][i, self._duct_idx])
            qtp = self._calc_duct_power(p_duct, i)

            # Need to get inner and outer coolant temperatures and
            # heat transfer coeffs; Requires two material updates
            if i == 0:  # inner-most duct, inner htc is asm interior
                t_in = self.temp['coolant_int']
                # Get rid of interior temps, only want edge/corner
                t_in = t_in[self.subchannel.n_sc['coolant']['interior']:]
                htc_in = self.coolant_int_params['htc'][1:]
                htc_in = htc_in[self._duct_idx]
            else:
                t_in = self.temp['coolant_byp'][i - 1]
                htc_in = self.coolant_byp_params['htc'][i - 1]
                htc_in = htc_in[self._duct_idx]
            if i == self.n_duct - 1:  # outermost duct; coolant is gap
                t_out = t_gap
                htc_out = htc_gap
                if htc_out.shape[0] == 2:
                    htc_out = htc_out[self._duct_idx]
            else:
                t_out = self.temp['coolant_byp'][i]
                htc_out = self.coolant_byp_params['htc'][i]
                htc_out = htc_out[self._duct_idx]

            # CONSTANTS (don't vary with duct mesh type)
            qLsq_over_8k = (qtp * self.duct_params['L^2/8'][i]
                            / self.duct.thermal_conductivity)

            # If adiabatic and the last duct, use different constants
            if (adiabatic and i + 1 == self.n_duct):
                c1 = (qtp * self.duct_params['L/2'][i]
                      / self.duct.thermal_conductivity)
                c1_L_over_2 = c1 * self.duct_params['L/2'][i]
                c2 = (t_in
                      + qLsq_over_8k
                      + self.duct_params['L/2'][i] / htc_in
                      + c1_L_over_2
                      + c1 * self.duct.thermal_conductivity / htc_in)

            else:
                htc_ratio = htc_in / htc_out
                c1 = ((qtp * self.duct_params['L/2'][i] * (htc_ratio - 1)
                       + htc_in * (t_out - t_in))
                      / (htc_in * self.duct_params['thickness'][i]
                         + (self.duct.thermal_conductivity
                            * (1 + htc_ratio))))
                c1_L_over_2 = c1 * self.duct_params['L/2'][i]
                c2 = (t_out
                      + qLsq_over_8k
                      - c1_L_over_2
                      - self.duct.thermal_conductivity * c1 / htc_out
                      + qtp * self.duct_params['L/2'][i] / htc_out)

            # Wall midpoint temperature
            self.temp['duct_mw'][i] = c2
            # Wall inside surface temperature: x = -L/2
            self.temp['duct_surf'][i, 0] = \
                -qLsq_over_8k - c1_L_over_2 + c2
            # Wall outside surface temperature: x = L/2
            self.temp['duct_surf'][i, 1] = \
                -qLsq_over_8k + c1_L_over_2 + c2

    def _calc_duct_power(self, p_duct, duct_id):
        """Transform linear power to power density for duct wall calculation"""
        if p_duct is None:
            return np.zeros(self.subchannel.n_sc['duct']['total'])
        else:
            start = duct_id * self.subchannel.n_sc['duct']['total']
            end = (duct_id + 1) * self.subchannel.n_sc['duct']['total']
            # Convert linear power (W/m) to volumetric power (W/m^3)
            # (qtp == q triple prime); value is very large
            pdens = (p_duct[start:end]
                     / self.duct_params['area'][duct_id, self._duct_idx])
            return pdens

    ####################################################################

    def calculate_pin_temperatures(self, dz, pin_powers):
        """Calculate cladding and fuel temperatures

        Parameters
        ----------
        dz : float
            Axial step size (m)
        pin_powers : numpy.ndarray
            Pin linear power (W/m) for each pin in the assembly

        Returns
        -------
        None

        """
        # Check for nonspecified pin power
        if pin_powers is None:
            pin_powers = np.zeros(self.n_pin)

        # Heat transfer coefficient (via Nu) for clad-coolant
        pin_nu = self.corr['pin_nu'](self.coolant,
                                     self.coolant_int_params['Re'],
                                     self.pin_model.htc_params)
        htc = (self.coolant.thermal_conductivity * pin_nu
               / self.bundle_params['de'])
        # Calculate pin-adjacent average coolant temperatures
        T_scaled = self.temp['coolant_int'] * self._q_p2sc
        Tc_avg = T_scaled[self.subchannel.pin_adj]
        Tc_avg = np.ma.masked_array(Tc_avg, self.subchannel.pin_adj < 0)
        Tc_avg = np.sum(Tc_avg, axis=1)

        # With maximum adjacent subchannel coolant temperature and
        # subchannel specific HTC
        # t = self.temp['coolant_int'][self.subchannel.pin_adj]
        # t = np.ma.masked_array(t, self.subchannel.pin_adj < 0)
        # tmax = np.max(t, axis=1)
        # idx = np.argmax(t, axis=1)
        # idx = np.diagonal(self.subchannel.pin_adj[:, idx])
        # sc_type = self.subchannel.type[idx]
        # # sc_de = self.params['d_heated_pins'][sc_type]
        # sc_de = self.params['de'][sc_type]
        # pin_nu = self.corr['pin_nu'](self.coolant,
        #                              self.coolant_int_params['Re_sc'],
        #                              self.pin_model.htc_params)
        # pin_nu = pin_nu[sc_type]
        # htc = self.coolant.thermal_conductivity * pin_nu / sc_de

        # Calculate pin temperatures
        self.pin_temps[:, 3:] = self.pin_model.calculate_temperatures(
            pin_powers, Tc_avg, htc, dz)


########################################################################
# BUNDLE GEOMETRY
########################################################################


def calculate_geometry(n_ring, P, D, Pw, Dw, dftf, n_sc, se2=False):
    """Calculate bundle geometry parameters based on definitions
    provided in Cheng-Todreas (1986)"""

    edge_pin2duct = 0.5 * (dftf[0][0] - (_sqrt3 * P * (n_ring - 1)))
    edge_pitch = edge_pin2duct + 0.5 * D
    n_duct = len(dftf)
    n_bypass = n_duct - 1

    # --------------------------------------------------------------
    # "d" : DISTANCES ACROSS THINGS
    # --------------------------------------------------------------

    d = {}
    # Pin-to-pin distance
    d['pin-pin'] = P - D
    # Pin-to-wall distance
    d['pin-wall'] = edge_pin2duct - 0.5 * D
    # Wall thickness(es)
    d['wall'] = np.zeros(n_duct)
    for i in range(0, n_duct):  # for all duct walls
        d['wall'][i] = 0.5 * (dftf[i][1] - dftf[i][0])
    # Bypass gap thickness(es)
    if n_bypass > 0:
        d['bypass'] = np.zeros(n_bypass)
        for i in range(0, n_bypass):  # for all bypass gaps
            d['bypass'][i] = 0.5 * (dftf[i + 1][0] - dftf[i][1])
    # Corner cell wall perimeters
    d['wcorner'] = np.zeros((n_duct, 2))
    # Corner subchannel inside/outside wall lengths
    d['wcorner'][0, 0] = (0.5 * D + d['pin-wall']) / _sqrt3
    d['wcorner'][0, 1] = d['wcorner'][0, 0] + d['wall'][0] / _sqrt3
    for i in range(1, n_duct):
        d['wcorner'][i, 0] = \
            d['wcorner'][i - 1, 1] + (d['bypass'][i - 1] / _sqrt3)
        d['wcorner'][i, 1] = \
            d['wcorner'][i, 0] + (d['wall'][i] / _sqrt3)
    # Corner cell wall perimeters (at the wall midpoint):
    # necessary to conserve energy in the calculation, which
    # treats the wall as a flat plane
    # d['wcorner_m'] = np.average(d['wcorner'], axis=1)

    # --------------------------------------------------------------
    # "L" : DISTANCE BETWEEN SUBCHANNEL CENTROIDS
    # --------------------------------------------------------------

    # Needs to be weird list rather than array because some entries
    # will themselves be lists if there are bypass channels
    L = [[0.0] * 7 for i in range(7)]
    # From interior (to interior, edge)
    L[0][0] = _sqrt3over3 * P   # interior-interior
    L[0][1] = 0.5 * (L[0][0] + D * 0.5 + d['pin-wall'])  # edge-int
    # From edge (to interior, edge, corner)
    L[1][0] = L[0][1]  # edge-interior
    L[1][1] = P  # edge-edge
    L[1][2] = 0.5 * (P + d['wcorner'][0, 0])
    # From corner (corner-edge, corner-corner)
    L[2][1] = L[1][2]  # corner - edge
    L[2][2] = (D + d['pin-wall']) / _sqrt3
    # Duct wall - no heat conduction between duct wall segments
    # Bypass gaps (gap edge, gap corner)
    if n_bypass > 0:
        L[5][5] = [P for byp in range(n_bypass)]  # edge-edge
        L[5][6] = [0.0 for byp in range(n_bypass)]
        x = 0.5 * D + d['pin-wall'] + d['wall'][0] + 0.5 * d['bypass'][0]
        L[5][6][0] = (x / _sqrt3 + (0.5 * P))
        L[6][6] = [0.0 * n_bypass]
        L[6][6][0] = 2 * (x / _sqrt3)
        for i in range(1, n_bypass):
            L[5][6][i] = (L[5][6][i - 1]
                          + (0.5 * d['bypass'][i - 1]
                             + d['wall'][i]
                             + 0.5 * d['bypass'][i]) / _sqrt3)
            L[6][6][i] = (L[6][6][i - 1]
                          + (d['bypass'][i - 1]
                             + 2 * d['wall'][i]
                             + d['bypass'][i]) / _sqrt3)
    L[6][5] = L[5][6]

    # --------------------------------------------------------------
    # BARE ROD SUBCHANNEL PARAMETERS
    # --------------------------------------------------------------

    # sc_bare = {}
    # # Flow area
    # sc_bare['area'] = np.zeros(3)
    # sc_bare['area'][0] = _sqrt3 * 0.25 * P**2 - 0.125 * np.pi * D**2
    # sc_bare['area'][2] = (edge_pin2duct**2 / _sqrt3 - np.pi * D**2 / 24)
    # if se2:
    #     sc_bare['area'][1] = P * (0.5 * D + Dw) - 0.125 * np.pi * D**2
    # else:
    #     sc_bare['area'][1] = (P * edge_pin2duct - np.pi * D**2 / 8)
    # # Wetted perimeter
    # sc_bare['wp'] = np.zeros(3)
    # sc_bare['wp'][0] = np.pi * D / 2
    # sc_bare['wp'][1] = P + np.pi * D / 2
    # sc_bare['wp'][2] = (np.pi * D / 6 + (2 * edge_pin2duct / _sqrt3))
    # # Hydraulic diameter
    # sc_bare['de'] = 4 * sc_bare['area'] / sc_bare['wp']

    # --------------------------------------------------------------
    # WIRE-WRAPPED SUBCHANNEL PARAMETERS
    # --------------------------------------------------------------

    sc_ww = {}
    # Angle between wire wrap and vertical
    if se2 or Dw == 0.0:
        cos_theta = 1.0
        sc_ww['theta'] = 0.0
    else:
        cos_theta = Pw / np.sqrt(Pw**2 + (np.pi * (D + Dw))**2)
        sc_ww['theta'] = np.arccos(cos_theta)
    # Flow area
    sc_ww['area'] = np.zeros(3)
    sc_ww['area'][0] = _sqrt3 * 0.25 * P**2 - 0.125 * np.pi * D**2
    sc_ww['area'][0] -= 0.125 * np.pi * Dw**2 / cos_theta
    sc_ww['area'][1] = P * edge_pin2duct - np.pi * D**2 / 8
    sc_ww['area'][1] -= 0.125 * np.pi * Dw**2 / cos_theta
    sc_ww['area'][2] = edge_pin2duct**2 / _sqrt3 - np.pi * D**2 / 24
    sc_ww['area'][2] -= np.pi * Dw**2 / 24 / cos_theta
    # Wetted perimeter
    sc_ww['wp'] = np.zeros(3)
    sc_ww['wp'][0] = np.pi * D / 2 + np.pi * Dw / 2 / cos_theta
    sc_ww['wp'][1] = P + np.pi * D / 2 + np.pi * Dw / 2 / cos_theta
    sc_ww['wp'][2] = (np.pi * D / 6
                      + (2 * edge_pin2duct / _sqrt3)
                      + np.pi * Dw / 6 / cos_theta)
    # Hydraulic diameter
    sc_ww['de'] = 4 * sc_ww['area'] / sc_ww['wp']
    # Projection of wire area into flow path
    # sc_ww['wproj'] = np.zeros(3)
    # sc_ww['wproj'][2] = (np.pi * (D + Dw) * Dw / 6)
    # if se2:
    #     sc_ww['wproj'][0] = \
    #         np.pi * (P - 0.5 * D)**2 / 6 - np.pi * D**2 / 24
    #     sc_ww['wproj'][1] = \
    #         np.pi * (0.25 * (0.5 * D + Dw)**2 - 0.0625 * D**2)
    # else:
    #     sc_ww['wproj'][0] = np.pi * (D + Dw) * Dw / 6
    #     sc_ww['wproj'][1] = np.pi * (D + Dw) * Dw / 4

    # --------------------------------------------------------------
    # BUNDLE TOTAL SUBCHANNEL PARAMETERS
    # --------------------------------------------------------------

    bundle = {}
    bundle['area'] = 0.0
    bundle['wp'] = 0.0
    for sci in range(3):
        bundle['area'] += sc_ww['area'][sci] * n_sc[sci]
        bundle['wp'] += sc_ww['wp'][sci] * n_sc[sci]
    bundle['de'] = 4 * bundle['area'] / bundle['wp']

    # --------------------------------------------------------------
    # BYPASS GAP PARAMETERS
    # --------------------------------------------------------------

    if n_bypass > 0:
        # self._k_byp_seal = np.zeros(n_bypass)
        bypass = {}
        bypass['area'] = np.zeros((n_bypass, 2))
        bypass['total area'] = np.zeros(n_bypass)
        bypass['de'] = np.zeros((n_bypass, 2))
        bypass['total de'] = np.zeros(n_bypass)
        for i in range(n_bypass):
            bypass['total area'][i] = \
                0.5 * _sqrt3 * (dftf[i + 1][0]**2 - dftf[i][1]**2)
            bypass['area'][i, 0] = L[5][5][i] * d['bypass'][i]
            bypass['area'][i, 1] = (d['bypass'][i]
                                    * (d['wcorner'][i + 1, 0]
                                       + d['wcorner'][i, 1]))
            bypass['de'][i, 0] = 2 * d['bypass'][i]
            bypass['de'][i, 1] = 2 * d['bypass'][i]
            bypass['total de'][i] = \
                (4 * bypass['total area'][i]
                 / (6 / _sqrt3)
                 / (dftf[i][1] + dftf[i + 1][0]))

    # --------------------------------------------------------------
    # DUCT PARAMS
    # --------------------------------------------------------------
    duct = {}
    duct['area'] = np.zeros((n_duct, 2))
    duct['thickness'] = np.zeros(n_duct)
    duct['total area'] = np.zeros(n_duct)
    duct['L/2'] = np.zeros(n_duct)
    duct['L^2/4'] = np.zeros(n_duct)
    duct['L^2/8'] = np.zeros(n_duct)

    for i in range(n_duct):
        duct['thickness'][i] = 0.5 * (dftf[i][1] - dftf[i][0])
        duct['total area'][i] = 0.5 * _sqrt3 * (dftf[i][1]**2
                                                - dftf[i][0]**2)
        duct['area'][i][0] = L[1][1] * duct['thickness'][i]
        duct['area'][i][1] = (duct['thickness'][i]
                              * (d['wcorner'][i][1]
                                 + d['wcorner'][i][0]))
        duct['L/2'][i] = duct['thickness'][i] * 0.5
        duct['L^2/4'][i] = duct['L/2'][i]**2
        duct['L^2/8'][i] = duct['L^2/4'][i] * 0.5

    # --------------------------------------------------------------
    # Collect and return
    # Keys in "params" dict are the attr names in RoddedRegion obj
    # --------------------------------------------------------------

    params = {}
    params['edge_pitch'] = edge_pitch
    params['d'] = d
    params['L'] = L
    # params['bare_params'] = sc_bare
    params['params'] = sc_ww
    params['bundle_params'] = bundle
    params['duct_params'] = duct
    if n_bypass > 0:
        params['bypass_params'] = bypass
    return params


def calculate_ht_constants(rr):
    """Setup method to define heat transfer constants

    Parameters
    ----------
    rr : DASSH RoddedRegion objecct

    Returns
    -------
    list
        List of lists containing pre-calculated HT constants

    """
    # HEAT TRANSFER CONSTANTS - set up similarly to self.L
    ht_consts = [[0.0] * 7 for i in range(7)]

    # Conduction between coolant channels (units: s/kg)
    # [ Interior <- Interior, Interior <- Edge, 0                ]
    # [ Edge <- Interior,     Edge <- Edge,     Edge <- Corner   ]
    # [ 0               ,     Corner <- Edge,   Corner <- Corner ]
    # if self.n_pin > 1:
    for i in range(3):
        if rr.n_pin == 1:
            continue
        for j in range(3):
            if rr.n_pin == 1:
                continue
            if rr.L[i][j] != 0.0:  # excludes int <--> corner
                if i == 0 or j == 0:
                    ht_consts[i][j] = \
                        (rr.d['pin-pin']
                         * rr.bundle_params['area']
                         / rr.L[i][j] / rr.int_flow_rate
                         / rr.params['area'][i])
                else:
                    ht_consts[i][j] = \
                        (rr.d['pin-wall']
                         * rr.bundle_params['area']
                         / rr.L[i][j] / rr.int_flow_rate
                         / rr.params['area'][i])

    # Convection from interior coolant to duct wall (units: m-s/kg)
    # Edge -> wall 1
    if rr.n_pin > 1:
        ht_consts[1][3] = (rr.L[1][1]
                           * rr.bundle_params['area']
                           / rr.int_flow_rate
                           / rr.params['area'][1])
        ht_consts[3][1] = ht_consts[1][3]

    # Corner -> wall 1
    ht_consts[2][4] = (2 * rr.d['wcorner'][0, 1]
                       * rr.bundle_params['area']
                       / rr.int_flow_rate
                       / rr.params['area'][2])
    # ht_consts[2][4] = (2 * self.d['wcorner_m'][0]
    #                         * self.bundle_params['area']
    #                         / self.int_flow_rate
    #                         / self.params['area'][2])
    ht_consts[4][2] = ht_consts[2][4]

    # Bypass convection and conduction
    if rr.n_bypass > 0 and np.sum(rr.byp_flow_rate) > 0:
        # Convection to-from duct walls and bypass gaps
        # Each bypass gap channel touches 2 walls
        #   Edge (wall 1) - Edge (bypass) - Edge (wall 2)
        #   Corner (wall 1) - Corner (bypass) - Corner (wall 2)
        # The edge connections are the same everywhere
        # The corner connections change b/c the "flat" parts of
        # the channel get longer as you walk radially outward
        ht_consts[3][5] = [[0.0] * 2 for i in range(rr.n_bypass)]
        ht_consts[4][6] = [[0.0] * 2 for i in range(rr.n_bypass)]
        for i in range(0, rr.n_bypass):
            # bypass edge -> wall 1
            if rr.n_pin > 1:
                ht_consts[3][5][i][0] = \
                    (rr.L[1][1]
                     * rr.bypass_params['total area'][i]
                     / rr.byp_flow_rate[i]
                     / rr.bypass_params['area'][i, 0])
                # bypass edge -> wall 2 (same as wall 1)
                ht_consts[3][5][i][1] = ht_consts[3][5][i][0]
            # bypass corner -> wall 1
            ht_consts[4][6][i][0] = \
                (2 * rr.d['wcorner'][i, 1]
                 * rr.bypass_params['total area'][i]
                 / rr.byp_flow_rate[i]
                 / rr.bypass_params['area'][i, 1])
            # ht_consts[4][6][i][0] = \
            #     (2 * self.d['wcorner_m'][i]
            #      * self.bypass_params['total area'][i]
            #      / self.byp_flow_rate[i]
            #      / self.bypass_params['area'][i, 1])
            # bypass corner -> wall 2
            ht_consts[4][6][i][1] = \
                (2 * rr.d['wcorner'][i + 1, 1]
                 * rr.bypass_params['total area'][i]
                 / rr.byp_flow_rate[i]
                 / rr.bypass_params['area'][i, 1])
            # ht_consts[4][6][i][1] = \
            #     (2 * self.d['wcorner_m'][i + 1]
            #      * self.bypass_params['total area'][i]
            #      / self.byp_flow_rate[i]
            #      / self.bypass_params['area'][i, 1])
        ht_consts[5][3] = ht_consts[3][5]
        ht_consts[6][4] = ht_consts[4][6]

        # Conduction between bypass coolant channels
        ht_consts[5][5] = [0.0] * rr.n_bypass
        ht_consts[5][6] = [0.0] * rr.n_bypass
        ht_consts[6][5] = [0.0] * rr.n_bypass
        ht_consts[6][6] = [0.0] * rr.n_bypass
        for i in range(0, rr.n_bypass):
            if rr.n_pin > 1:
                ht_consts[5][5][i] = \
                    (rr.d['bypass'][i]
                     * rr.bypass_params['total area'][i]
                     / rr.L[5][5][i] / rr.byp_flow_rate[i]
                     / rr.bypass_params['area'][i, 0])
                ht_consts[5][6][i] = \
                    (rr.d['bypass'][i]
                     * rr.bypass_params['total area'][i]
                     / rr.L[5][6][i] / rr.byp_flow_rate[i]
                     / rr.bypass_params['area'][i, 0])
                ht_consts[6][5][i] = \
                    (rr.d['bypass'][i]
                     * rr.bypass_params['total area'][i]
                     / rr.L[5][6][i] / rr.byp_flow_rate[i]
                     / rr.bypass_params['area'][i, 1])
                # ht_consts[6][5] = ht_consts[5][6]
            ht_consts[6][6][i] = \
                (rr.d['bypass'][i]
                 * rr.bypass_params['total area'][i]
                 / rr.L[6][6][i] / rr.byp_flow_rate[i]
                 / rr.bypass_params['area'][i, 1])
    return ht_consts


def _setup_conduction_constants(rr, ht_consts):
    """Set up numpy arrays to accelerate conduction calculation

    Notes
    -----
    Creates dictionary attribute "_cond" which has keys "adj"
    and "const". Usage is as follows:

    dT_conduction = (self._cond['const']
                     * (self.temp['coolant_int']
                                 [self._cond['adj']]
                        - self.temp['coolant_int'][:, np.newaxis]))
    dT_conduction = keff * np.sum(dT_conduction, axis=1)

    """
    _cond = {}
    # Coolant-coolant subchannel adjacency
    # This array has all sorts of extra values because empty
    # adjacency positions are filled with zeros. We want to
    # get rid of those things and get some arrays that are
    # N_subchannel x 3, because 3 is the maximum number of
    # adjacent coolant subchannels.
    # _cond['adj'] = self.subchannel.sc_adj[
    #     :self.subchannel.n_sc['coolant']['total'], :-2]
    _cond['adj'] = rr.subchannel.sc_adj[
        :rr.subchannel.n_sc['coolant']['total'], :5]

    # Temporary arrays for coolant subchannel type and adjacent
    # coolant subchannel type
    cool_type = rr.subchannel.type[
        :rr.subchannel.n_sc['coolant']['total']]
    cool_adj_type = cool_type[_cond['adj']]

    # Set up temporary array to refine the adjacency
    cool2cool_adj1 = np.zeros(
        (rr.subchannel.n_sc['coolant']['total'], 3),
        dtype=int)
    for i in range(rr.subchannel.n_sc['coolant']['total']):
        tmp = _cond['adj'][i][_cond['adj'][i] >= 0]
        cool2cool_adj1[i, :len(tmp)] = tmp

    # Set up temporary array to get easily usable HT constants
    hc = np.vstack([ht_consts[i][:3] for i in range(3)])  # 3x3
    # hc = np.array(ht_consts)[:3, :3]
    _cond['const'] = np.zeros(
        (rr.subchannel.n_sc['coolant']['total'], 3))
    for i in range(rr.subchannel.n_sc['coolant']['total']):
        tmp = cool_adj_type[i][_cond['adj'][i] >= 0]
        for j in range(len(tmp)):
            _cond['const'][i, j] = hc[cool_type[i]][tmp[j]]

    # Now overwrite the original adjacency array; the -1s have
    # served their purpose and we don't need them anymore.
    _cond['adj'] = cool2cool_adj1
    return _cond


def _setup_convection_constants(rr, ht_consts):
    """Set up numpy arrays to accelerate convective heat xfer
    calculation between edge/corner subchannels and the wall"""
    _conv = {}

    # Edge and corner subchannel indices
    _conv['ind'] = np.arange(
        rr.subchannel.n_sc['coolant']['interior'],
        rr.subchannel.n_sc['coolant']['total'],
        1)

    # Edge and corner subchannel types
    _conv['type'] = rr.subchannel.type[
        rr.subchannel.n_sc['coolant']['interior']:
        rr.subchannel.n_sc['coolant']['total']]

    # Adjacent wall indices
    _conv['adj'] = np.arange(0, len(_conv['ind']), 1)

    # Convection HT constants
    c = np.array([ht_consts[i][i + 2] for i in range(3)])
    _conv['const'] = c[_conv['type']]

    # Set up duct wall energy balance constants
    # Li = np.array([0.0, self.pin_pitch, 2 * self.d['wcorner_m'][0]])
    Li = np.array([0.0, rr.pin_pitch, 2 * rr.d['wcorner'][0, 1]])
    _conv['ebal'] = Li[_conv['type']]
    return _conv


########################################################################
# CORRELATIONS
########################################################################


def import_corr(friction, flowsplit, mixing, nusselt, bundle, warn):
    """Import correlations to be used in a DASSH Assembly object

    Parameters
    ----------
    friction : str {'NOV', 'REH', 'ENG', 'CTS', 'CTD', 'UCTD'}
        Name of the correlation used to determine the friction factor
        used in the pressure drop calculation.
    flowsplit : str {'NOV', 'CTD', 'UCTD'}
        Name of the correlation used to determine the flow split
        parameter between coolant subchannels.
    mixing : str {'MIT', 'CTD'}
        Name of the correlation used to determine the mixing
        parameters for coolant between subchannels
    nusselt : str {'DB'}
        Name of the correlation used to determine the Nusselt number,
        from which the heat transfer coefficients are determined
    bundle : DASSH RoddedRegion object
    warn : bool
        Raise applicability warnings or no

    Returns
    -------
    tuple
        Tuple of dict containing the following for each correlation:
        (1) Correlation names
        (2) Parameter calculation methods
        (3) Geometric constants, if applicable

    """
    # Dictionary stores the calculation methods imported from
    # each correlation module

    corr = {}
    corr_names = {}
    corr_const = {}

    # Friction factor
    if friction is not None:
        friction = '-'.join(re.split('-| ', friction.lower()))
        corr_names['ff'], corr['ff'], corr_const['ff'] = \
            _import_friction_correlation(friction, bundle, warn)
    else:
        corr['ff'] = None
        corr_names['ff'] = None

    # Flow split
    if flowsplit is not None:
        flowsplit = '-'.join(re.split('-| ', flowsplit.lower()))
        corr_names['fs'], corr['fs'], corr_const['fs'] = \
            _import_flowsplit_correlation(flowsplit, bundle, warn)
    else:
        corr['fs'] = None
        corr_names['fs'] = None

    # Mixing parameters
    if mixing is not None:
        mixing = '-'.join(re.split('-| ', mixing.lower()))
        corr_names['mix'], corr['mix'], corr_const['mix'] = \
            _import_mixing_correlation(mixing, bundle)
    else:
        corr['mix'] = None
        corr_names['mix'] = None

    # Nusselt number (for heat transfer coefficient)
    nusselt = '-'.join(re.split('-| ', nusselt.lower()))
    if nusselt in ['db', 'dittus-boelter']:
        import dassh.correlations.nusselt_db as nu
        corr_names['nu'] = 'dittus-boelter'
        corr['nu'] = nu.calculate_sc_Nu
        corr_const['nu'] = None
        # Add clad-to-coolant htc for bundle : use "bundle Nu" because
        # we do heat transfer to pins based on bundle-average Re
        # rather than subchannel Re
        corr_names['pin_nu'] = 'dittus-boelter'
        corr['pin_nu'] = nu.calculate_bundle_Nu
    else:
        msg = 'Unknown correlation specified for Nusselt number: '
        module_logger.error(msg + nusselt)
        sys.exit(1)

    return corr, corr_names, corr_const


def _import_friction_correlation(name, bundle, warn):
    """Import friction factor correlation for DASSH Assembly object

    Parameters
    ----------
    name : str {'NOV', 'REH', 'ENG', 'CTS', 'CTD', 'UCTD'}
        Name of the correlation used to determine the friction factor
        used in the pressure drop calculation.
    bundle : DASSH RoddedRegion object
    warn : bool
        Raise applicability warnings

    Returns
    -------
    tuple
        Correlation name, method, and dict of geometric constants

    """
    if name in ['nov', 'novendstern']:
        import dassh.correlations.friction_nov as ff
        name = 'novendstern'
        nickname = 'nov'
        constants = ff.calc_constants(bundle)

    elif name in ['eng', 'engel']:
        import dassh.correlations.friction_eng as ff
        name = 'engel'
        nickname = 'eng'
        constants = None

    elif name in ['reh', 'rehme']:
        import dassh.correlations.friction_reh as ff
        name = 'rehme'
        nickname = 'reh'
        constants = None

    elif name in ['cts', 'cheng-todreas-simple']:
        import dassh.correlations.friction_cts as ff
        name = 'cheng-todreas-simple'
        nickname = 'cts'
        constants = ff.calc_constants(bundle)

    elif name in ['ctd', 'cheng-todreas-detailed']:
        import dassh.correlations.friction_ctd as ff
        name = 'cheng-todreas-detailed'
        nickname = 'ctd'
        constants = ff.calc_constants(bundle)

    elif name in ['uctd', 'upgraded-cheng-todreas-detailed',
                  'upgraded-cheng-todreas']:
        import dassh.correlations.friction_uctd as ff
        name = 'upgraded-cheng-todreas-detailed'
        nickname = 'uctd'
        constants = ff.calc_constants(bundle)

    else:
        module_logger.error(f'Assembly {bundle.name}: unknown '
                            f'correlation specified for friction '
                            f'factor: {name}')
        sys.exit(1)
    if warn:
        check_correlation.check_application_range(bundle, ff)
    return nickname, ff.calculate_bundle_friction_factor, constants


def _import_flowsplit_correlation(name, bundle, warn):
    """Import flow split correlations for use in DASSH Assembly object

    Parameters
    ----------
    name : str {'NOV', 'CTD', 'UCTD'}
        Name of the correlation used to determine the flow split
        parameter between coolant subchannels.
    bundle : DASSH RoddedRegion object
    warn : bool
        Raise applicability warnings

    Returns
    -------
    tuple
        Correlation name, method, and dict of geometric constants

    """
    if name in ['nov', 'novendstern']:
        import dassh.correlations.flowsplit_nov as fs
        name = 'novendstern'
        nickname = 'nov'

    elif name in ['se2', 'superenergy', 'superenergy-2', 'superenergy2']:
        import dassh.correlations.flowsplit_se2 as fs
        name = 'se2'
        nickname = 'se2'

    elif name in ['ct', 'cheng-todreas', 'ctd',
                  'cheng-todreas-detailed']:
        import dassh.correlations.flowsplit_ctd as fs
        name = 'cheng-todreas-detailed'
        nickname = 'ctd'

    elif name in ['uctd', 'upgraded-cheng-todreas-detailed',
                  'upgraded-cheng-todreas']:
        import dassh.correlations.flowsplit_uctd as fs
        name = 'upgraded-cheng-todreas-detailed'
        nickname = 'uctd'

    elif name in ['mit', 'chiu-rohsenow-todreas']:
        import dassh.correlations.flowsplit_mit as fs
        name = 'chiu-rohsenow-todreas'
        nickname = 'mit'

    else:
        module_logger.error(f'Assembly {bundle.name}: unknown '
                            f'correlation specified for flow split: '
                            f'{name}')
        sys.exit(1)

    if warn:
        check_correlation.check_application_range(bundle, fs)
    constants = fs.calc_constants(bundle)
    return nickname, fs.calculate_flow_split, constants


def _import_mixing_correlation(name, bundle):
    """Import mixing parameter correlations for use in DASSH Assembly

    Parameters
    ----------
    name : str {'MIT', 'CTD'}
        Name of the correlation used to determine the mixing
        parameters for coolant between subchannels
    bundle : DASSH Assembly object

    Returns
    -------
    tuple
        Correlation name, method, and dict of geometric constants

    Notes
    -----
    No applicability warning because its basically the same as friction
    factor and flow split applicability

    """
    if name in ['mit', 'chiu-rohsenow-todreas']:
        import dassh.correlations.mixing_mit as mix
        name = 'chiu-rohsenow-todreas'
        nickname = 'mit'

    elif name in ['ct', 'cheng-todreas',
                  'ctd', 'cheng-todreas-detailed']:
        import dassh.correlations.mixing_ctd as mix
        name = 'cheng-todreas'
        nickname = 'ctd'

    elif name in ['uct', 'upgraded-cheng-todreas',
                  'uctd', 'upgraded-cheng-todreas-detailed']:
        import dassh.correlations.mixing_uctd as mix
        name = 'upgraded-cheng-todreas'
        nickname = 'uctd'

    else:
        module_logger.error(f'Assembly {bundle.name}: unknown '
                            f'correlation specified for mixing '
                            f'parameters: {name}')
        sys.exit(1)
    constants = mix.calc_constants(bundle)
    return nickname, mix.calculate_mixing_params, constants


########################################################################
# AXIAL MESH SIZE
########################################################################


def calculate_min_dz(bundle, temp_lo, temp_hi, adiabatic=False):
    """Evaluate dz for the bundle at the assembly inlet and outlet
    temperatures; minimum value is taken to be the constraint

    Parameters
    ----------
    bundle : DASSH RoddedRegion object
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

    """
    min_dz = []
    sc_code = []
    # Hold the original value of the temperature to reset after
    _temp_in = bundle.coolant.temperature

    # Determine which duct, if any, is adiabatic
    which_adiabatic = None
    if adiabatic:
        if bundle.n_bypass > 0 and np.sum(bundle.byp_flow_rate) > 0:
            which_adiabatic = 'outer_byp'
        else:
            which_adiabatic = 'outer'

    for temp in [temp_lo, temp_hi]:
        # Interior coolant parameters and dz requirement
        bundle._update_coolant_int_params(temp)
        tmp_dz, tmp_sc = _calculate_int_dz(bundle, which_adiabatic)
        min_dz.append(tmp_dz)
        sc_code.append(tmp_sc)

        # Bypass coolant parameters and dz requirement
        if bundle.n_bypass > 0 and np.sum(bundle.byp_flow_rate) > 0:
            bundle._update_coolant_byp_params([temp] * bundle.n_bypass)
            tmp_dz, tmp_sc = _calculate_byp_dz(bundle, which_adiabatic)
            min_dz.append(tmp_dz)
            sc_code.append(tmp_sc)

    # Reset the coolant temperature
    bundle._update_coolant_int_params(_temp_in)
    if bundle.n_bypass > 0:
        bundle._update_coolant_byp_params([_temp_in] * bundle.n_bypass)

    return min(min_dz), sc_code[min_dz.index(min(min_dz))]


def _calculate_int_dz(bundle, adiabatic_duct=None):
    """Evaluate dz for the assembly as the minimum required for
    stability in all assembly-interior subchannels

    Parameters
    ----------
    bundle : DASSH RoddedRegion object
        Contains geometric and flow parameters
    adiabatic_duct (optional) : str
        If 'outer': apply to equations (default = None)

    Returns
    -------
    float
        Minimum required dz for stability

    """
    # Calculate subchannel mass flow rates
    sc_mfr = [bundle.int_flow_rate
              * bundle.coolant_int_params['fs'][i]
              * bundle.params['area'][i]
              / bundle.bundle_params['area']
              for i in range(len(bundle.coolant_int_params['fs']))]

    # Calculate "effective" thermal conductivity
    keff = (bundle._sf * bundle.coolant.thermal_conductivity
            + (bundle.coolant.density * bundle.coolant.heat_capacity
               * bundle.coolant_int_params['eddy']))

    # Determine whether the bundle duct wall is adiabatic
    if adiabatic_duct == 'outer':
        adiabatic_duct = True
    else:
        adiabatic_duct = False

    if bundle.n_pin == 1:  # only corner subchannels
        # Corner -> corner subchannels
        return (_cons3_33(
            sc_mfr[2],
            bundle.L[2][2],
            bundle.d['pin-wall'],
            bundle.d['wcorner'][0, 1],
            # bundle.d['wcorner_m'][0],
            keff,
            bundle.coolant.heat_capacity,
            bundle.coolant.density,
            bundle.coolant_int_params['htc'][2],
            bundle.coolant_int_params['swirl'][2],
            bundle.d['wall'][0],
            bundle.duct.thermal_conductivity,
            adiabatic_duct,
            bundle._conv_approx),
            '3-33')

    else:
        dz = []
        sc_code = []
        # Interior subchannel --> interior/interior/edge subchannel
        sc_code.append('1-112')
        dz.append(_cons1_112(sc_mfr[0], bundle.L[0][0],
                             bundle.L[0][1], bundle.d['pin-pin'],
                             keff, bundle.coolant.heat_capacity))

        # Corner subchannel --> edge/edge subchannel
        sc_code.append('3-22')
        dz.append(_cons3_22(
            sc_mfr[2],
            bundle.L[1][2],
            bundle.d['pin-wall'],
            bundle.d['wcorner'][0, 1],
            keff,
            bundle.coolant.heat_capacity,
            bundle.coolant.density,
            bundle.coolant_int_params['htc'][2],
            bundle.coolant_int_params['swirl'][2],
            bundle.d['wall'][0],
            bundle.duct.thermal_conductivity,
            adiabatic_duct,
            bundle._conv_approx))

        # Edge subchannel --> interior/corner/corner subchannel
        if bundle.n_pin == 7:
            sc_code.append('2-133')
            dz.append(_cons2_133(
                sc_mfr[1],
                bundle.L[1][0],
                bundle.L[1][1],
                bundle.L[1][2],
                bundle.d['pin-pin'],
                bundle.d['pin-wall'],
                keff,
                bundle.coolant.heat_capacity,
                bundle.coolant.density,
                bundle.coolant_int_params['htc'][1],
                bundle.coolant_int_params['swirl'][1],
                bundle.d['wall'][0],
                bundle.duct.thermal_conductivity,
                adiabatic_duct,
                bundle._conv_approx))
        else:
            # Interior subchannel --> interior/interior/interior subchannel
            sc_code.append('1-111')
            dz.append(_cons1_111(sc_mfr[0], bundle.L[0][0],
                                 bundle.d['pin-pin'], keff,
                                 bundle.coolant.heat_capacity))
            # Edge subchannel --> interior/edge/corner subchannel
            sc_code.append('2-123')
            dz.append(_cons2_123(
                sc_mfr[1],
                bundle.L[1][0],
                bundle.L[1][1],
                bundle.L[1][2],
                bundle.d['pin-pin'],
                bundle.d['pin-wall'],
                keff,
                bundle.coolant.heat_capacity,
                bundle.coolant.density,
                bundle.coolant_int_params['htc'][1],
                bundle.coolant_int_params['swirl'][1],
                bundle.d['wall'][0],
                bundle.duct.thermal_conductivity,
                adiabatic_duct,
                bundle._conv_approx))

        if bundle.n_pin > 19:
            # Edge subchannel --> interior/edge/edge subchannel
            sc_code.append('2-122')
            dz.append(_cons2_122(
                sc_mfr[1],
                bundle.L[1][0],
                bundle.L[1][1],
                bundle.d['pin-pin'],
                bundle.d['pin-wall'],
                keff,
                bundle.coolant.heat_capacity,
                bundle.coolant.density,
                bundle.coolant_int_params['htc'][1],
                bundle.coolant_int_params['swirl'][1],
                bundle.d['wall'][0],
                bundle.duct.thermal_conductivity,
                adiabatic_duct,
                bundle._conv_approx))

        min_dz = min(dz)
        return min_dz, sc_code[dz.index(min_dz)]


def _cons1_111(m1, L11, d_p2p, keff, Cp):
    """dz constraint for interior sc touching 3 interior sc"""
    return m1 * Cp * L11 / 3 / d_p2p / keff


def _cons1_112(m1, L11, L12, d_p2p, keff, Cp):
    """dz constraint for interior sc touching 2 interior, 1 edge sc"""
    return m1 * Cp / (2 / L11 + 1 / L12) / d_p2p / keff


def _cons2_122(m2, L21, L22, d_p2p, d_p2w, keff, Cp, rho, h, vs, dw, kw,
               adiabatic=False, conv_approx=False):
    """dz constraint for edge sc touching 1 interior, 2 edge sc"""
    term1 = 0.0                                 # convection term
    if not adiabatic:
        if conv_approx:
            R = 1 / h + dw / 2 / kw
            term1 = L22 / m2 / Cp / R           # conv / cond to duct MW
        else:
            term1 = h * L22 / m2 / Cp           # conv to wall
    term2 = 2 * keff * d_p2w / m2 / Cp / L22    # cond to adj edge
    term3 = keff * d_p2p / m2 / Cp / L21        # cond to adj int
    term4 = rho * vs * d_p2w / m2               # swirl
    return 1 / (term1 + term2 + term3 + term4)


def _cons2_123(m2, L21, L22, L23, d_p2p, d_p2w, keff, Cp, rho, h, vs,
               dw, kw, adiabatic=False, conv_approx=False):
    """dz constraint for edge sc touching interior, edge, corner sc"""
    term1 = 0.0                             # convection term
    if not adiabatic:
        if conv_approx:
            R = 1 / h + dw / 2 / kw
            term1 = L22 / m2 / Cp / R       # conv / cond to duct MW
        else:
            term1 = h * L22 / m2 / Cp       # conv to wall
    term2 = keff * d_p2w / m2 / Cp / L22    # cond to adj edge
    term3 = keff * d_p2p / m2 / Cp / L21    # cond to adj int
    term4 = keff * d_p2w / m2 / Cp / L23    # cond to adj corner
    term5 = rho * vs * d_p2w / m2           # swirl
    return 1 / (term1 + term2 + term3 + term4 + term5)


def _cons2_133(m2, L21, L22, L23, d_p2p, d_p2w, keff, Cp, rho, h, vs,
               dw, kw, adiabatic=False, conv_approx=False):
    """dz constraint for edge sc touching interior, 2 corner sc"""
    term1 = 0.0                             # convection term
    if not adiabatic:
        if conv_approx:
            R = 1 / h + dw / 2 / kw
            term1 = L22 / m2 / Cp / R       # conv / cond to duct MW
        else:
            term1 = h * L22 / m2 / Cp       # conv to wall
    term2 = keff * d_p2p / m2 / Cp / L21        # cond to adj int
    term3 = 2 * keff * d_p2w / m2 / Cp / L23    # cond to adj corner
    term4 = rho * vs * d_p2w / m2               # swirl
    return 1 / (term1 + term2 + term3 + term4)


def _cons3_22(m3, L23, d_p2w, d_wc, keff, Cp, rho, h, vs, dw, kw,
              adiabatic=False, conv_approx=False):
    """dz constraint for corner sc touching 2 edge sc"""
    term1 = 0.0                                 # convection term
    if not adiabatic:
        if conv_approx:
            R = 1 / h + dw / 2 / kw
            term1 = 2 * d_wc / m3 / Cp / R      # conv / cond to duct MW
        else:
            term1 = h * 2 * d_wc / m3 / Cp      # conv to wall
    term2 = 2 * keff * d_p2w / m3 / Cp / L23    # cond to adj edge
    term3 = rho * vs * d_p2w / m3               # swirl
    return 1 / (term1 + term2 + term3)


def _cons3_33(m3, L33, d_p2w, d_wc, keff, Cp, rho, h, vs, dw, kw,
              adiabatic=False, conv_approx=False):
    """dz constraint for corner sc touching 2 corner sc"""
    term1 = 0.0                                 # convection term
    if not adiabatic:
        if conv_approx:
            R = 1 / h + dw / 2 / kw
            term1 = 2 * d_wc / m3 / Cp / R      # conv / cond to duct MW
        else:
            term1 = h * 2 * d_wc / m3 / Cp      # conv to wall
    term2 = 2 * keff * d_p2w / m3 / Cp / L33    # cond to adj corner
    term3 = rho * vs * d_p2w / m3               # swirl
    return 1 / (term1 + term2 + term3)


def _calculate_byp_dz(bundle, adiabatic_duct=None):
    """Evaluate dz for the assembly as the minimum required for
    stability in all assembly bypass gap subchannels

    Parameters
    ----------
    bundle : DASSH Assembly object

    Returns
    -------
    float
        Minimum required dz for stability

    """
    # Calculate subchannel mass flow rates
    byp_sc_mfr = (bundle.byp_flow_rate
                  * bundle.bypass_params['area']
                  / bundle.bypass_params['total area'])

    min_sc_code = []  # Indicates the type of subchannel
    min_dz = []
    for i in range(bundle.n_bypass):
        dz = []
        sc_code = []

        # Determine whether the bundle duct wall is adiabatic
        if adiabatic_duct == 'outer_byp' and i + 1 == bundle.n_bypass:
            adiabatic_duct = True
        else:
            adiabatic_duct = False

        # Only corner -> corner bypass subchannels
        if bundle.n_pin == 1:
            sc_code.append('7-77')
            dz.append(_cons7_77(
                byp_sc_mfr[i, 1],
                bundle.L[6][6][i],
                bundle.d['bypass'][i],
                bundle.d['wcorner'][i, 1],
                bundle.d['wcorner'][i + 1, 1],
                bundle.coolant.thermal_conductivity,
                bundle.coolant.heat_capacity,
                bundle.coolant_byp_params['htc'][i, 1],
                bundle.d['wall'][i],
                bundle.duct.thermal_conductivity,
                bundle.d['wall'][i + 1],
                bundle.duct.thermal_conductivity,
                adiabatic_duct,
                bundle._conv_approx))

        else:
            sc_code.append('7-66')
            dz.append(_cons7_66(
                byp_sc_mfr[i, 1],
                bundle.L[5][6][i],
                bundle.d['bypass'][i],
                bundle.d['wcorner'][i, 1],
                bundle.d['wcorner'][i + 1, 1],
                bundle.coolant.thermal_conductivity,
                bundle.coolant.heat_capacity,
                bundle.coolant_byp_params['htc'][i, 1],
                bundle.d['wall'][i],
                bundle.duct.thermal_conductivity,
                bundle.d['wall'][i + 1],
                bundle.duct.thermal_conductivity,
                adiabatic_duct,
                bundle._conv_approx))

            # Edge subchannel --> corner/corner subchannel
            if bundle.n_pin == 7:
                sc_code.append('6-77')
                dz.append(_cons6_77(
                    byp_sc_mfr[i, 0],
                    bundle.L[5][5][i],
                    bundle.L[5][6][i],
                    bundle.d['bypass'][i],
                    bundle.coolant.thermal_conductivity,
                    bundle.coolant.heat_capacity,
                    bundle.coolant_byp_params['htc'][i, 0],
                    bundle.d['wall'][i],
                    bundle.duct.thermal_conductivity,
                    bundle.d['wall'][i + 1],
                    bundle.duct.thermal_conductivity,
                    adiabatic_duct,
                    bundle._conv_approx))

            # Edge subchannel --> edge/corner subchannel
            else:
                sc_code.append('6-67')
                dz.append(_cons6_67(
                    byp_sc_mfr[i, 0],
                    bundle.L[5][5][i],
                    bundle.L[5][6][i],
                    bundle.d['bypass'][i],
                    bundle.coolant.thermal_conductivity,
                    bundle.coolant.heat_capacity,
                    bundle.coolant_byp_params['htc'][i, 0],
                    bundle.d['wall'][i],
                    bundle.duct.thermal_conductivity,
                    bundle.d['wall'][i + 1],
                    bundle.duct.thermal_conductivity,
                    adiabatic_duct,
                    bundle._conv_approx))

            # Edge subchannel --> edge/edge subchannel
            if bundle.n_pin > 19:
                sc_code.append('6-66')
                dz.append(_cons6_66(
                    byp_sc_mfr[i, 0],
                    bundle.L[5][5][i],
                    bundle.d['bypass'][i],
                    bundle.coolant.thermal_conductivity,
                    bundle.coolant.heat_capacity,
                    bundle.coolant_byp_params['htc'][i, 0],
                    bundle.d['wall'][i],
                    bundle.duct.thermal_conductivity,
                    bundle.d['wall'][i + 1],
                    bundle.duct.thermal_conductivity,
                    adiabatic_duct,
                    bundle._conv_approx))

        min_dz.append(min(dz))
        min_sc_code.append(sc_code[dz.index(min_dz[i])])

    min_min_dz = min(min_dz)
    byp_idx = min_dz.index(min_min_dz)
    min_sc_code = min_sc_code[byp_idx] + '-' + str(byp_idx)
    return min_min_dz, min_sc_code


def _cons6_66(m6, L66, d_byp, k, Cp, h_byp, dw1, kw1, dw2, kw2,
              adiabatic_duct=False, conv_approx=False):
    """dz constrant for edge bypass sc touching 2 edge bypass sc"""
    term1_out = 0.0
    if not adiabatic_duct:
        if conv_approx:
            R2 = 1 / h_byp + dw2 / 2 / kw2
            term1_out = L66 / m6 / Cp / R2  # conv / cond to duct 1 MW
        else:
            term1_out = h_byp * L66 / m6 / Cp  # conv to outer duct
    if conv_approx:
        R1 = 1 / h_byp + dw1 / 2 / kw1
        term1_in = L66 / m6 / Cp / R1       # conv / cond to duct 2 MW
    else:
        term1_in = h_byp * L66 / m6 / Cp
    term2 = 2 * k * d_byp / m6 / Cp / L66   # cond to adj bypass edge
    return 1 / (term1_in + term1_out + term2)


def _cons6_67(m6, L66, L67, d_byp, k, Cp, h_byp, dw1, kw1, dw2, kw2,
              adiabatic_duct=False, conv_approx=False):
    """dz constrant for edge byp sc touching edge, corner byp sc"""
    term1_out = 0.0
    if not adiabatic_duct:
        if conv_approx:
            R2 = 1 / h_byp + dw2 / 2 / kw2
            term1_out = L66 / m6 / Cp / R2  # conv / cond to duct 2 MW
        else:
            term1_out = h_byp * L66 / m6 / Cp  # conv to outer duct
    if conv_approx:
        R1 = 1 / h_byp + dw1 / 2 / kw1
        term1_in = L66 / m6 / Cp / R1      # conv / cond to duct 1 MW
    else:
        term1_in = h_byp * L66 / m6 / Cp
    term2 = k * d_byp / m6 / Cp / L66   # cond to adj bypass edge
    term3 = k * d_byp / m6 / Cp / L67   # cond to adj bypass corner
    return 1 / (term1_out + term1_in + term2 + term3)


def _cons6_77(m6, L66, L67, d_byp, k, Cp, h_byp, dw1, kw1, dw2, kw2,
              adiabatic_duct=False, conv_approx=False):
    """dz constrant for edge bypass sc touching 2 corner bypass sc"""
    term1_out = 0.0
    if not adiabatic_duct:
        if conv_approx:
            R2 = 1 / h_byp + dw2 / 2 / kw2
            term1_out = L66 / m6 / Cp / R2  # conv / cond to duct 2 MW
        else:
            term1_out = h_byp * L66 / m6 / Cp  # conv to outer duct
    if conv_approx:
        R1 = 1 / h_byp + dw1 / 2 / kw1
        term1_in = L66 / m6 / Cp / R1      # conv / cond to duct 1 MW
    else:
        term1_in = h_byp * L66 / m6 / Cp
    term2 = 2 * k * d_byp / m6 / Cp / L67   # cond to adj bypass corner
    return 1 / (term1_in + term1_out + term2)


def _cons7_66(m7, L67, d_byp, wc_in, wc_out, k, Cp, h_byp, dw1, kw1,
              dw2, kw2, adiabatic_duct=False, conv_approx=False):
    """dz constraint for corner bypass sc touching 2 edge bypass sc"""
    term1_out = 0.0
    if not adiabatic_duct:
        if conv_approx:
            R2 = 1 / h_byp + dw2 / 2 / kw2
            term1_out = 2 * wc_out / m7 / Cp / R2  # conv / cond to duct 2 MW
        else:
            term1_out = h_byp * 2 * wc_out / m7 / Cp  # conv to outer duct
    if conv_approx:
        R1 = 1 / h_byp + dw1 / 2 / kw1
        term1_in = 2 * wc_in / m7 / Cp / R1     # conv / cond to duct 1 MW
    else:
        term1_in = h_byp * 2 * wc_in / m7 / Cp
    # term1 = 2 * (wc_in + wc_out) * h_byp / m7 / Cp  # conv in/out duct
    term2 = 2 * k * d_byp / m7 / Cp / L67   # cond to adj bypass edge
    return 1 / (term1_in + term1_out + term2)


def _cons7_77(m7, L77, d_byp, wc_in, wc_out, k, Cp, h_byp, dw1, kw1,
              dw2, kw2, adiabatic_duct=False, conv_approx=False):
    """dz constraint for corner bypass sc touching 2 corner bypass sc"""
    term1_out = 0.0
    if not adiabatic_duct:
        if conv_approx:
            R2 = 1 / h_byp + dw2 / 2 / kw2
            term1_out = 2 * wc_out / m7 / Cp / R2  # conv / cond to duct 2 MW
        else:
            term1_out = h_byp * 2 * wc_out / m7 / Cp  # conv to outer duct
    if conv_approx:
        R1 = 1 / h_byp + dw1 / 2 / kw1
        term1_in = 2 * wc_in / m7 / Cp / R1     # conv / cond to duct 1 MW
    else:
        term1_in = h_byp * 2 * wc_in / m7 / Cp
    # term1 = 2 * (wc_in + wc_out) * h_byp / m7 / Cp  # conv in/out duct
    term2 = 2 * k * d_byp / m7 / Cp / L77   # cond to adj bypass corner
    return 1 / (term1_in + term1_out + term2)


########################################################################
