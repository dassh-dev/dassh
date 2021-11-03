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
date: 2021-09-24
author: matz
Generate power distributions in assembly components based on neutron
flux; object to assign to individual assemblies
"""
########################################################################
import numpy as np
import os
import sys
import bisect
import logging
from dassh.logged_class import LoggedClass
from dassh import py4c
from dassh import core


np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=500)
module_logger = logging.getLogger('dassh.power')


_sqrt3 = np.sqrt(3)
_ref_hex_pitch = np.sqrt(2 / np.sqrt(3))
_user_power_mat_ids = ['pins', 'duct', 'cool']


class Power(LoggedClass):
    """Set up assembly power shape functions for the core based on
    CCCC binary files from VARPOW and the user.

    Parameters
    ----------
    path_mat_power_density : str
        Path to material power density data file
    path_monomial_exp : str
        Path to monomial exponents text file
    path_varpow : str
        Path to VARPOW binary file
    path_geodst : str
        Path to GEODST binary file
    user_power : float (optional)
        User specification for core total power (W)
    warn_negative : bool (optional)
        Log warning at instantiation about negative powers
        (in DASSH, this is checked within dassh_setup)
    """

    def __init__(self, path_mat_power_density, path_monomial_exp,
                 path_varpow, path_geodst, user_power=0.0, scalar=1.0,
                 model='distribute', warn_negative=False):

        LoggedClass.__init__(self, 4, 'dassh.power.Power')
        # path = '/Users/matz/Documents/DASSH/src_DASSH/power/smaller/mono2'
        # Load input files - depending on power model, group or zero
        # out the power density columns.
        # mat_powerdens = np.loadtxt(path_mat_power_density)
        # monomial_exp = np.loadtxt(path_monomial_exp, dtype='int')
        # varpow = py4c.nhflux.NHFLUX(path_varpow)
        # geodst = py4c.geodst.GEODST(path_geodst)
        mat_powerdens = _load(path_mat_power_density, np.loadtxt)
        monomial_exp = _load(path_monomial_exp, np.loadtxt, {'dtype': 'int'})
        varpow = _load(path_varpow, py4c.nhflux.NHFLUX)
        geodst = _load(path_geodst, py4c.geodst.GEODST)
        # Put all struct/coolant power into pins
        if model == 'pin_only':
            mat_powerdens[:, 0] += mat_powerdens[:, 1]
            mat_powerdens[:, 0] += mat_powerdens[:, 2]
            mat_powerdens[:, 3] += mat_powerdens[:, 4]
            mat_powerdens[:, 3] += mat_powerdens[:, 5]
            mat_powerdens[:, [1, 2, 4, 5]] *= 0.0
        # Put all g power into structure; all n power into pins
        if model == 'struct_gamma':
            mat_powerdens[:, 0] += mat_powerdens[:, 1]
            mat_powerdens[:, 0] += mat_powerdens[:, 2]
            mat_powerdens[:, 4] += mat_powerdens[:, 4]
            mat_powerdens[:, 4] += mat_powerdens[:, 5]
            mat_powerdens[:, [1, 2, 3, 5]] *= 0.0

        # --------------------------------------------------------------
        # Set some attributes
        # Number of monomial terms
        self.n_terms = len(monomial_exp)
        # Monomial term exponents
        self.mono_exp = monomial_exp
        # Z-basis integration factors to compute total power
        self.z_int = evaluate_z_integrand(np.max(self.mono_exp))
        # Coarse zmesh points (cm)
        self.z_mesh = geodst.zmesh
        # Number of fine axial meshes in each coarse mesh
        self.k_fints = geodst.kfints
        # Z fine mesh boundaries
        self.z_finemesh = np.array([0.0])
        for kc in range(len(self.z_mesh) - 1):
            z_lo, z_hi = self.z_mesh[kc:kc + 2]
            zk_pts = np.linspace(z_lo, z_hi, self.k_fints[kc] + 1)[1:]
            self.z_finemesh = np.append(self.z_finemesh, zk_pts)
        # Hex cell pitch (cm)
        self.asm_pitch = geodst.xmesh[1]
        # Hex XY scaling factor: scale XY coords so that they fit in
        # the unit hexagon (area = 1 cm^2); asm_obj XY coords are in m
        self.asm_scaling_factor = 100 * _ref_hex_pitch / self.asm_pitch
        # Hex area (cm^2)
        self.hex_area = self.asm_pitch * self.asm_pitch * _sqrt3 * 0.5

        # --------------------------------------------------------------
        # Set up map between GEODST radial coordinate (I, J) and VARPOW
        # radial coordinates (NINTXY)
        finemesh_to_activenode = np.zeros((geodst.fine_dims[2],
                                           geodst.fine_dims[1]))
        activenode = 0
        # for j in range(geodst.fine_dims[1]):
        #     for i in range(geodst.fine_dims[2]):
        for j in range(geodst.fine_dims[1]):
            for i in range(geodst.fine_dims[2]):
                activenode += 1
                if geodst.reg_assignments[0, j, i] != 0.0:
                    finemesh_to_activenode[i, j] = activenode
        for ij in range(len(varpow.itrmap)):
            ireg = varpow.itrmap[ij]
            if ireg > 0:
                for i in range(geodst.fine_dims[2]):
                    for j in range(geodst.fine_dims[1]):
                        if finemesh_to_activenode[i, j] == ireg:
                            finemesh_to_activenode[i, j] = -(ij + 1)

        # --------------------------------------------------------------
        # Calculate assembly total power; rearrange material power dens
        n_asm = len(finemesh_to_activenode[finemesh_to_activenode != 0])
        n_pos = int(np.max(-finemesh_to_activenode))
        n_ring = core.count_rings(n_pos)
        if n_ring == 1:
            n_avail = 1
        else:
            n_avail = 3 * (n_ring - 1) * n_ring + 1
        asm_idx = finemesh_to_activenode.flatten()
        asm_idx = np.sort(-asm_idx)
        asm_idx = asm_idx[asm_idx != 0]

        # --------------------------------------------------------------
        # Calculate assembly total power; rearrange material power dens
        self.power = np.zeros((n_avail, geodst.fine_dims[0], 6))
        self.power_density = np.zeros((n_avail, geodst.fine_dims[0], 6))
        vols = geodst.calc_volumes()
        for k in range(geodst.fine_dims[0]):
            # power_dens_k = mat_powerdens[k * n_asm:(k + 1) * n_asm, :]
            power_dens_k = mat_powerdens[k * n_avail:(k + 1) * n_avail, :]
            for i in range(geodst.fine_dims[2]):
                for j in range(geodst.fine_dims[1]):
                    asm_ij = int(-finemesh_to_activenode[i, j])
                    if asm_ij > 0:
                        # asm_id = np.where(asm_idx == asm_ij)[0][0]
                        self.power[asm_ij - 1, k, :] += \
                            power_dens_k[asm_ij - 1, :] * vols[k, j, i]
                    # else:
                    #     assert np.all(power_dens_k[asm_ij - 1, :] == 0)
            for asm in range(n_avail):
                self.power_density[asm][k] = power_dens_k[asm]

        # --------------------------------------------------------------
        # Check the input power for negative values; if present
        # set to zero, renormalize, and warn the user.
        calculated_power = np.sum(self.power)
        negative_power = 0.0
        negative_asm_k_pairs = np.zeros(self.power.shape[:2])
        for asm in range(len(self.power)):
            for k in range(len(self.power[asm])):
                for vi in range(len(self.power[asm, k])):
                    if self.power[asm, k, vi] < 0.0:
                        # Count it (for renormalization)
                        negative_power += self.power[asm, k, vi]
                        # Track it (to warn the user)
                        negative_asm_k_pairs[asm, k] = 1
                        # Set it equal to zero
                        self.power[asm, k, vi] = 0.0
                        self.power_density[asm, k, vi] = 0.0
        # Renormalize the power - the power will increase when we
        # remove negative values, so need to rescale to be lower
        self.power *= calculated_power / np.sum(self.power)
        self.power_density *= calculated_power / np.sum(self.power)
        # Warn the user
        self.negative_power = negative_power
        if warn_negative and negative_power < 0.0:
            self.log('warning', 'Negative powers found; setting equal '
                                'to zero. Check flux solution for '
                                'convergence.')
            self.log('warning', 'Total negative power (W): '
                                + '{:0.3e}'.format(negative_power))

        # --------------------------------------------------------------
        # Normalize power, power density to user request
        if user_power != 0.0:
            total_power = np.sum(self.power)
            self.power_density *= user_power / total_power
            self.power *= user_power / total_power

        # Scale power and power density to user request
        self.power_density *= scalar
        self.power *= scalar

        # --------------------------------------------------------------
        # Split up the power shape functions by assembly
        self.mono_coeffs = {}
        self.mono_coeffs['n'] = np.zeros((n_avail,
                                          geodst.fine_dims[0],
                                          self.n_terms))
        self.mono_coeffs['g'] = self.mono_coeffs['n'].copy()
        self.mono_coeffs['ff'] = self.mono_coeffs['n'].copy()
        for k in range(geodst.fine_dims[0]):
            for i in range(geodst.fine_dims[2]):
                for j in range(geodst.fine_dims[1]):
                    asm_ij = int(-finemesh_to_activenode[i, j])
                    if asm_ij > 0:
                        # asm_ij = np.where(asm_idx == asm_ij)[0][0]
                        self.mono_coeffs['n'][asm_ij - 1, k] = \
                            varpow.flux[0][k][asm_ij - 1]
                        self.mono_coeffs['g'][asm_ij - 1, k] = \
                            varpow.flux[1][k][asm_ij - 1]
                        self.mono_coeffs['ff'][asm_ij - 1, k] = \
                            varpow.flux[2][k][asm_ij - 1]

    ####################################################################
    # CALCULATE COMPONENT POWER PROFILES
    # For a specified assembly, calculate the linear power (W/m) as a
    # function of z based on the power shape coefficients and exponents
    # supplied by the user.
    ####################################################################

    def calc_power_profile(self, asm_obj, asm_id):
        """Distribute power among pins, duct, coolant in the rodded
        region(s); in the un-rodded regions, lump together.

        Parameters
        ----------
        asm_obj : DASSH Assembly object
            Contains rod bundle and unrodded region data
        asm_id : int
            ID number corresponding to assembly location

        Returns
        -------
        dict
            Contains numpy.ndarray for linear power distribution in
            pins, duct, coolant, and unrodded regions.

        """
        # Calculate average linear power - used for unrodded regions
        avg_power = np.zeros(sum(self.k_fints))
        for k in range(sum(self.k_fints)):
            avg_power[k] = (np.sum(self.power_density[asm_id, k])
                            * self.hex_area)

        # If completely unrodded, skip all the shenanigans and just
        # calculate the average power, bc that's all that's used
        if not asm_obj.has_rodded:
            return {}, avg_power

        # SET UP POWER DISTRIBUTION AMONG BUNDLE COMPONENTS
        power = {}  # Power profiles (W/m)
        # Normalization (W) for each component
        p_computed = {}
        # Evaluate XY points to collapse monomials
        eval_xy = self.calc_component_xy(asm_obj.rodded)

        # Volumes of struct components (relative to struct total)
        str_vf = calculate_structure_vfs(asm_obj.rodded)
        # Total linear power (W/m) and component power dens (W/m^3)
        # for each component material in the assembly
        p_lin = self.calc_total_linear_power(asm_id, str_vf)
        p_component = self.calc_component_power_dens(asm_id, str_vf)
        for comp in ['pins', 'duct', 'cool']:
            p_component[comp].shape = (p_component[comp].shape[0],
                                       p_component[comp].shape[1],
                                       1)
            p_lin[comp] = np.sum(p_lin[comp], axis=0)
            power[comp] = np.zeros((sum(self.k_fints),
                                    len(eval_xy[comp]),
                                    np.max(self.mono_exp) + 1))
            p_computed[comp] = np.zeros((len(power[comp]),
                                         len(eval_xy[comp])))

        # LOOP OVER DIF3D REGIONS TO DISTRIBUTE THE POWER
        self._mask = []
        for i in range(np.max(self.mono_exp[:, 2]) + 1):
            self._mask.append(np.where(self.mono_exp[:, 2] == i)[0])

        for comp in ['pins', 'duct', 'cool']:
            # Temporary array: scale neutron and gamma coeffs by the
            # neutron / gamma power in each axial mesh; re-dimension
            # the array so it can play
            a1 = (self.mono_coeffs['n'][asm_id] * p_component[comp][0]
                  + self.mono_coeffs['g'][asm_id] * p_component[comp][1])
            a1.shape = (a1.shape[0], 1, a1.shape[1])

            # Temporary array 2: need to resize the eval_xy array to
            # add an extra dimension
            eval_xy[comp].shape = (1, len(eval_xy[comp]), self.n_terms)

            # Multiplying these arrays gives an array of the monomial
            # terms for each axial find mesh, for each pin
            a1 = a1 * eval_xy[comp]

            # Now can apply the mask, which goes over the 3rd dimension
            for i in range(np.max(self.mono_exp) + 1):
                power[comp][:, :, i] = \
                    np.sum(a1[:, :, self._mask[i]], axis=2)

            # Integrate power using shape fxn at xy position
            p_computed[comp] = np.dot(power[comp], self.z_int)

            # Normalize power to total computed value
            norm = np.sum(p_computed[comp], axis=1)
            for k in range(len(power[comp])):
                if norm[k] == 0:
                    power[comp][k] = 0.0
                else:
                    power[comp][k] *= p_lin[comp][k] / norm[k]

        # Flip power distribution about unit-z axis
        for comp in power.keys():
            power[comp] = _flip_power_dist(power[comp])

        # return power, linear_power, computed_power, component_power
        return power, avg_power

    def calc_component_power_dens(self, asm, structure_vf):
        """Calculate fuel, structure, and coolant power densities

        Parameters
        ----------
        asm : int
            Assembly ID (equals active node ID from NHFLUX minus one)
        structure_vf : dict
            The volume fractions of structural components relative
            to the overall structure volume

        Returns
        -------
        dict
            Power density (W/m^3) in each component of the assembly,
            with dummy pin power separated from fuel pin power

        Notes
        -----
        Also group cladding with fuel and wire with coolant

        """
        cpower = {}
        # Power in the fuel pins
        cpower['pins'] = (self.power_density[asm, :, [0, 3]]
                          + (structure_vf['clad']
                             * self.power_density[asm, :, [1, 4]]))
        # Power in dummy pins
        cpower['dummy'] = (structure_vf['dummy']
                           * self.power_density[asm, :, [1, 4]])
        # Power in the duct
        cpower['duct'] = (structure_vf['duct']
                          * self.power_density[asm, :, [1, 4]])
        # Power in the coolant
        cpower['cool'] = (self.power_density[asm, :, [2, 5]]
                          + (structure_vf['wire']
                             * self.power_density[asm, :, [1, 4]]))
        return cpower

    def calc_total_linear_power(self, asm, structure_vf):
        """Calculate component linear power totals

        Parameters
        ----------
        asm : int
            Assembly ID (equals active node ID from NHFLUX minus one)
        structure_vf : dict
            The volume fractions of structural components relative
            to the overall structure volume

        Returns
        -------
        dict
            Linear power (W/m) for each component in the assembly,
            with dummy pin power grouped with fuel pin power.

        Notes
        -----
        Also group cladding with fuel and wire with coolant

        """
        lpower = {}
        # Power in the pins
        lpower['pins'] = ((self.power_density[asm, :, [0, 3]]
                           + (structure_vf['clad']
                              * self.power_density[asm, :, [1, 4]])
                           + (structure_vf['dummy']
                              * self.power_density[asm, :, [1, 4]]))
                          * self.hex_area)
        # Power in the duct
        lpower['duct'] = (self.power_density[asm, :, [1, 4]]
                          * structure_vf['duct'] * self.hex_area)
        # Power in the coolant
        lpower['cool'] = ((self.power_density[asm, :, [2, 5]]
                           + (structure_vf['wire']
                              * self.power_density[asm, :, [1, 4]]))
                          * self.hex_area)
        return lpower

    def calc_component_xy(self, asm_obj, simple_coolant=False):
        """Evaluate the monomials at the XY positions of the pins,
        ducts, and coolant channels.

        Parameters
        ----------
        asm_obj : DASSH Assembly object
        simple_coolant (optional) : bool
            If True, calculate coolant power at origin only
            (default=False)

        Returns
        -------
        dict
            Monomials evaluated at the XY positions of the pins,
            ducts, and coolant channels

        """
        xy = {}
        xy['pins'] = evaluate_xy_mono(asm_obj.pin_lattice.xy,
                                      self.mono_exp,
                                      self.asm_scaling_factor)
        tmp = np.where(np.isin(asm_obj.subchannel.type, [3, 4]))
        xy['duct'] = evaluate_xy_mono(asm_obj.subchannel.xy[tmp],
                                      self.mono_exp,
                                      self.asm_scaling_factor)
        tmp = np.where(np.isin(asm_obj.subchannel.type, [0, 1, 2]))
        if not simple_coolant:
            xy['cool'] = evaluate_xy_mono(asm_obj.subchannel.xy[tmp],
                                          self.mono_exp,
                                          self.asm_scaling_factor)
        else:
            xy['cool'] = evaluate_xy_mono(np.array([[0.0, 0.0]]),
                                          self.mono_exp,
                                          self.asm_scaling_factor)
        return xy

    def check_power_profile(self, power, total_power, tol=1e-6):
        """Reintegrate the power profiles to confirm the monomials
        have been collapsed in a way that preserves total power

        Parameters
        ----------
        power : dict
            Coefficients on the z-terms describing axial power profiles
            in the pins, duct, and coolant
        total_power : dict
            Linear power profiles (W/m) in each component calculated
            based on the input power density; used to check result
        tol : float (optional)
            Tolerance by which to compare the results

        """
        for comp in ['pins', 'duct', 'cool']:
            tmp = np.sum(total_power[comp], axis=0)
            for kk in range(len(power[comp])):
                kk_int = np.sum([np.dot(power[comp][kk, r], self.z_int)
                                 for r in range(len(power[comp][kk]))])
                err = np.abs(kk_int - tmp[kk])
                msg = ('*' + comp + '* power profile at axial mesh '
                       + str(kk) + ' predicts '
                       + '{:0.2f}'.format(kk_int) + ' W/m; expected '
                       + '{:0.2f}'.format(tmp[kk]) + ' W/m with '
                       + 'tolerance ' + str(tol) + ' (difference: '
                       + '{:0.2e}'.format(err) + ').')
                assert err < tol, msg


def _load(path, read_fxn, kwargs={}):
    """x"""
    try:
        x = read_fxn(path, **kwargs)
    except OSError:
        try:
            x = read_fxn(os.readlink(path), **kwargs)
        except OSError:
            raise
    return x


def get_areas(asm_obj):
    """Calculate the cross-sectional areas of fuel, coolant, and
    structural components in the assembly

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains parameters required to calculate component areas

    Returns
    -------
    dict
        Dict (keys: 'fuel', 'clad', 'wire', 'dummy', 'duct') of
        component areas (m^2)

    """
    area = {}
    r_pin = asm_obj.pin_diameter * 0.5
    r_pellet = asm_obj.pin_diameter * 0.5 - asm_obj.clad_thickness
    # Fuel pellet cross-sectional area
    area['fuel'] = np.pi * r_pellet * r_pellet
    area['fuel'] *= asm_obj.n_pin  # - asm_obj.pin.n_dummy
    # Cladding cross-sectional area
    area['clad'] = np.pi * (r_pin * r_pin - r_pellet * r_pellet)
    area['clad'] *= asm_obj.n_pin  # - asm_obj.pin.n_dummy
    # Dummy pin cross-sectional area; assumed to entirely structure
    # area['dummy'] = np.pi * r_pin * r_pin
    # area['dummy'] *= asm_obj.pin.n_dummy
    area['dummy'] = 0.0
    # Wire wrap cross-sectional area; treat as an ellipse stretched
    # in one dimension due to its wrapping around the pin; A = pi*r1*r2
    r_w1 = asm_obj.wire_diameter * 0.5
    r_w2 = r_w1 / np.cos(asm_obj.params['theta'])
    area['wire'] = np.pi * r_w1 * r_w2
    # Coolant area: already calculated during asm_obj instantiation
    # Will later add on bypass gap coolant area, if applicable
    area['coolant'] = asm_obj.bundle_params['area']
    # Duct area: assess for each concentric duct wall
    area['duct'] = np.zeros(asm_obj.n_duct)
    for i in range(asm_obj.n_duct):
        area['duct'][i] = _sqrt3 * 0.5 * (asm_obj.duct_ftf[i][1]
                                          * asm_obj.duct_ftf[i][1]
                                          - asm_obj.duct_ftf[i][0]
                                          * asm_obj.duct_ftf[i][0])
        if i > 0:  # account for bypass coolant
            bypass = _sqrt3 * 0.5 * (asm_obj.duct_ftf[i][0]
                                     * asm_obj.duct_ftf[i][0]
                                     - asm_obj.duct_ftf[i - 1][1]
                                     * asm_obj.duct_ftf[i - 1][1])
            area['coolant'] += bypass
    return area


def calculate_structure_vfs(asm_obj):
    """Calculate the volume fractions of the structural components
    relative to the overall structure volume

    Parameters
    ----------
    asm_obj : DASSH Assembly object

    Returns
    -------
    dict
        Volume fractions of the structural components (cladding, wire
        wrap, dummy pins, and duct walls)

    Notes
    -----
    We know how much heat from neutrons and gammas goes into the three
    different region in the assembly (fuel, structure, and coolant);
    however, in reality we want to regroup some of these regions. To
    do that, we determine the volume fractions of the regions we want
    to redistribute and use these as scaling factors to adjust the heat
    added to each region.

    """
    area = get_areas(asm_obj)
    v_struct = (sum(area['duct']) + area['clad']
                + area['wire'] + area['dummy'])
    struct_vf = {}
    # Put cladding heat with fuel pins
    struct_vf['clad'] = area['clad'] / v_struct
    # Put wire heat with coolant
    struct_vf['wire'] = area['wire'] / v_struct
    # Dummy pin heat goes with pins
    struct_vf['dummy'] = area['dummy'] / v_struct
    # The rest goes to the duct
    struct_vf['duct'] = 1.0 - sum(struct_vf.values())
    return struct_vf


def evaluate_z_integrand(z_order):
    """Integrate the monomial basis

    Parameters
    ----------
    z_order : int
        Desired maximum order of the monomial expansion in z-dimension

    Returns
    -------
    numpy.ndarray
        Array containing the integral of the monomial taken
        over the reference volume (z = -0.5 to 0.5).
        Length = (z_order+1)

    Notes
    -----
    This function computes the following integral over the reference
    volume:
           0.5
        int( dz nlen(z) )
          -0.5

    ...where n(z) is the monomial basis in the z-dimension; i.e. if the
    order is 4, n(z) = [1, z, z^2, z^3, z^4]

    """
    z_int = np.zeros(z_order + 1)
    for i in range(z_order + 1):
        z_int[i] = (0.5**(i + 1) - (-0.5)**(i + 1)) / (i + 1)
    return z_int


def evaluate_xy_mono(xy, monomial_exp, xy_scalar):
    """Evaluate X-Y components of the monomials at each pin location

    Parameters
    ----------
    xy : numpy.ndarray
        XY locations at which the monomials should be evaluated;
        these are relative to the *reference assembly* and need to
        have been multiplied by the assembly scaling factor (equals
        sqrt(2/sqrt(3))) before this function is called. Array size
        is (N_pts x 2).
    monomial_exp : numpy.ndarray
        Exponents of the polynomial terms used to approximate the
        neutron and gamma fluxes; the array size is (N_Ritz x 3),
        where N_Ritz is the number of Ritz terms used in the
        approximation.
    xy_scalar : float
        Scaling factor to transform XY points from input assembly
        into those for reference assembly (origin = 0.0, area = 1)

    Returns
    -------
    tuple {numpy.ndarray, numpy.ndarray}
        Two numpy arrays containing the neutron and gamma monomials
        evaluated at the requested reference X-Y points.

    Notes
    -----
    Evaluate the x and y exponents first - this gives x^0, x^1, x^2,
    and so on in an efficient way. Additionally, because X and Y will
    not vary axially, it is more efficient to do this once, outside
    of the axial loop. Then, we can use these when we evaluate the
    monomial terms at each XY position by multiplying the proper
    exponentiated values by the term coefficient.

    This general method can be called to evaluate the monomials at
    the desired pin and duct wall positions, depending on what set
    of XY points is provided as input.

    The monomial coefficients (used in the summation of the monomials)
    are not incorporated here because they vary axially.

    """
    n_terms = len(monomial_exp)
    order = np.max(monomial_exp)
    # rescale the xy points to be in the reference assembly space
    xy = xy * xy_scalar
    eval_xy = np.zeros((len(xy), n_terms))
    for pt in range(len(xy)):
        raised_x = np.ones(order + 1)
        raised_y = np.ones(order + 1)
        raised_x[1] = xy[pt, 0]
        raised_y[1] = xy[pt, 1]
        for exp in range(2, order + 1):
            raised_x[exp] = raised_x[exp - 1] * xy[pt, 0]
            raised_y[exp] = raised_y[exp - 1] * xy[pt, 1]
        # Now, for each XY point, combine the X and Y components of the
        # monomials to obtain the XY components of each term; because
        # these don't vary axially, it's better to do this first.
        for trm in range(n_terms):
            eval_xy[pt, trm] = (raised_x[monomial_exp[trm][0]]
                                * raised_y[monomial_exp[trm][1]])
    return eval_xy


def _flip_power_dist(pdist):
    """Flip the polynomial produced by VARPOW

    The coefficients for the power distribution polynomial produced by
    VARPOW correspond to "z" values between -0.5 and 0.5 in each axial
    mesh. In other words, a transformation needs to take place to use
    the coefficients. The absolute value of "z" in the problem (i.e.
    z=1.21m from the core inlet) needs to be transformed to this range,
    depending on the active mesh. For example, if z=1.21 occurs at the
    bottom of some axial mesh, that mesh's coefficients would be used
    to determine the power distribution and the transformed z*=-0.5.

    However, VARPOW spits out the coefficients backward, such that the
    lower bound of the mesh actually needs z*=0.5, not z*=-0.5. This
    method flips the coefficients so that the power distribution is
    properly oriented on the z-axis.

    """
    exp = np.arange(0, pdist.shape[-1], 1)
    multiplier = np.ones(pdist.shape[-1]) * -1
    multiplier = multiplier**exp
    return pdist * multiplier


########################################################################


class AssemblyPower(object):
    """Container for the pin, duct, and coolant power shape functions
    and supporting data for a specified assembly.

    Parameters
    ----------
    power_profiles : dict
        Contains numpy.ndarray of power profiles for each assembly
        component; keys: {'pins', 'duct', 'cool', 'lrefl', 'urefl'}
        Power shape functions for fuel pins in the assembly
    z_finemesh : list
        Axial mesh cell boundaries from VARIANT flux solution
    rod_zbnds : list
        Indicate the fine mesh points that mark the lower and
        upper bounds of the rodded core (cm)
    scale : float (optional)
        Scalar for total power (default = 1.0)
    user_power : boolean (optional)
        Indicate whether power distribution is defined by user or by VARPOW;
        VARPOW flips the unit mesh bounds so need to know how to handle
    Notes
    -----
    This container is passed to DASSH assembly objects for them to
    generate power profiles when calculating coolant and duct wall
    temperatures.

    """

    # def __init__(self, power_profiles, avg_power_profile, z_finemesh,
    #              k_bnds, scale=1.0):
    def __init__(self, power_profiles, avg_power_profile, z_finemesh,
                 rod_bundle_zbnds, scale=1.0):
        """Instantiate the AssemblyPower object"""
        # Axial mesh points that bound the power profile regions
        self.z_finemesh = np.array([np.around(zfi, 10)
                                   for zfi in z_finemesh])
        self.n_region = len(self.z_finemesh) - 1
        # Axial mesh points that bound the rod bundle axial region
        self.rod_zbnds = np.array(rod_bundle_zbnds)  # cm
        if np.abs(self.rod_zbnds[0] == 0.0):
            self.rod_zbnds[0] = -1.0
        if np.abs(self.rod_zbnds[1] - self.z_finemesh[-1]) < 1e-12:
            self.rod_zbnds[1] = 99999
        # Check power profiles for None values
        for k in power_profiles.keys():
            if power_profiles[k] is not None:
                power_profiles[k] *= scale
        if avg_power_profile is not None:
            avg_power_profile *= scale
        # Assign power profiles as attributes
        self.avg_power = avg_power_profile
        self.pin_power = power_profiles.get('pins')
        self.duct_power = power_profiles.get('duct')
        self.coolant_power = power_profiles.get('cool')
        self.n_terms = 0
        for profile in [self.pin_power, self.duct_power, self.coolant_power]:
            if profile is not None:
                self.n_terms = profile.shape[2]
                break
        # self.norm = np.ones(len(self.z_finemesh) - 1)

    def get_power(self, z):
        """Calculate the linear power in all components at the
        requested axial position

        Parameters
        ----------
        z : float
            Axial position (m) in the core

        Returns
        -------
        dict
            Dictionary containing the linear power values for each
            component in the assembly
            keys: {'pins', 'cool', 'duct'}

        Notes
        -----
        If input z is in the porous media region, only the "refl" key
        will have a value that is not None. If the input z is in the
        rodded region, the "refl" key will have None value and the
        others will have their linear power arrays.

        Units: The "get" methods for the individual components return
        linear power in units of W/cm. DASSH wants units in terms of
        W/m. This is why each is multiplied by a factor of 100 before
        being returned...and why the input z (m) is multiplied by 100
        (cm).

        """
        z = z * 100  # m -> cm
        p_lin = {'pins': None, 'cool': None, 'duct': None, 'refl': None}
        kf = self.get_kfint(z)  # z given in meters

        # For some unit conversions, the core length given to DASSH may
        # be longer than that stored in the GEODST finemesh. Compensate
        # by just preserving the last region as long as necessary
        if kf == self.n_region:  # n_regions but python indexing
            kf -= 1

        # When z = rod_zbnds[0] (lower boundary), we're still sweeping
        # through non-bundle space, return average power
        # When z = rod_zbnds[1] (upper boundary), we're still sweeping
        # through rod bundle space, so return bundle power
        # OLD if kf < self.k_bnds[0] or kf >= self.k_bnds[1]:
        if z <= self.rod_zbnds[0] or z > self.rod_zbnds[1]:
            # lin_power['refl'] = self.get_refl_power(k_fint) * 100
            p_lin['refl'] = self.avg_power[kf] * 100
            return p_lin

        else:
            zm = self.transform_z(kf, z)
            z_exp = np.power(zm, np.arange(self.n_terms))
            if self.pin_power is not None:
                p_lin['pins'] = np.dot(self.pin_power[kf], z_exp) * 100
            if self.coolant_power is not None:
                p_lin['cool'] = np.dot(self.coolant_power[kf], z_exp) * 100
            if self.duct_power is not None:
                p_lin['duct'] = np.dot(self.duct_power[kf], z_exp) * 100

        # At extremely low power, can get some negative values
        # (~ -1e-6 W/m); want to filter these out as zeros.
        for k in p_lin.keys():
            # try:
            #     p_lin[k] = p_lin[k].clip(0.0)
            # except AttributeError:
            #     continue
            try:
                p_lin[k][p_lin[k] < 0.0] = 0.0
            except TypeError:
                continue
        return p_lin

    def transform_z(self, k_fint, z_abs):
        """Transform z-position in the core to the relative z-position
        within the appropriate power profile mesh cell

        Parameters
        ----------
        k_fint : int
            Axial region in which to obtain the power distribution
        z_abs : float
            Axial position in the core (cm)

        Returns
        -------
        float
            Value between -0.5 and 0.5 corresponding to the relative
            position in the active power profile mesh cell

        """
        if k_fint == len(self.z_finemesh) - 1:
            # When this occurs, the input z-coordinate is the upper
            # boundary of the fine mesh; we need to treat this as
            # being in the last mesh cell.
            z_lo = self.z_finemesh[k_fint - 1]
            z_hi = self.z_finemesh[k_fint]
        else:
            z_lo = self.z_finemesh[k_fint]
            z_hi = self.z_finemesh[k_fint + 1]
        dz = z_hi - z_lo
        const = -0.5 - (z_lo / dz)
        # assert np.isclose(const, 0.5 - (z_hi / dz))
        # Because VARPOW spits out the power densities backwards with
        # respect to z in each region, we want to invert the relative
        # z-coordinate
        # return -(z_abs / dz + const)
        return (z_abs / dz + const)

    # def transform_z2(self, k_fint, z_abs):
    #     """Transform z-position in the core to the relative z-position
    #     within the appropriate VARIANT mesh cell
    #
    #     Parameters
    #     ----------
    #     k_fint : int (or numpy.ndarray of int)
    #         Axial region in which to obtain the power distribution
    #     z_abs : float (or numpy.ndarry of float)
    #         Axial position in the core (cm)
    #
    #     Returns
    #     -------
    #     float (or numpy.ndarray of float)
    #         Value(s) between -0.5 and 0.5 corresponding to the
    #         relative position in the active VARIANT mesh cell
    #
    #     """
    #     z_lo = self.z_finemesh[k_fint]
    #     z_hi = self.z_finemesh[k_fint + 1]
    #     dz = z_hi - z_lo
    #     const = -0.5 - (z_lo / dz)
    #     assert np.allclose(const, 0.5 - (z_hi / dz))
    #     # Because VARPOW spits out the power densities backwards with
    #     # respect to z in each region, we want to invert the relative
    #     # z-coordinate
    #     return -(z_abs / dz + const)

    def get_kfint(self, z_abs):
        """Get the fine mesh interval for a given axial position

        Parameters
        ----------
        z_abs : float
            Axial position in the core (cm)

        Returns
        -------
        int
            VARIANT mesh cell in which the requested axial
            position is located

        """
        z_abs = np.around(z_abs, 12)
        # # Identify the lower mesh boundary
        # try:  # Last position where z_in > a value in z_finemesh
        #     idx = np.where(self.z_finemesh < z_abs)[0][-1]
        # # Account for z_abs=0.0; this should never occur!
        # except IndexError:
        #     idx = np.where(self.z_finemesh == z_abs)[0][0]
        if z_abs == 0.0:
            return 0
        else:
            return bisect.bisect_left(self.z_finemesh, z_abs) - 1

    def estimate_total_power(self, zpts=250):
        """Estimate the total power (W) produced by the assembly using
        the linear power (W/m) shape functions."""
        dz = (self.z_finemesh[-1] - self.z_finemesh[0]) / zpts
        p_total = 0.0
        z = self.z_finemesh[0] + dz / 2
        for cell in range(int(zpts)):
            k = self.get_kfint(z)
            if z <= self.rod_zbnds[0] or z > self.rod_zbnds[1]:
                p_total += dz * self.avg_power[k]
                # p_total += 0.0
            else:
                zm = self.transform_z(k, z)
                z_exp = np.power(zm, np.arange(self.n_terms))
                pk = np.sum(np.dot(self.pin_power[k], z_exp))
                pk += np.sum(np.dot(self.coolant_power[k], z_exp))
                pk += np.sum(np.dot(self.duct_power[k], z_exp))
                p_total += dz * pk
            z = z + dz
        return p_total

    def calculate_total_power(self):
        """Calculate the total power (W) produced by the assembly by
        integrating the linear power (W/m) shape functions"""
        dz_finemesh = self.z_finemesh[1:] - self.z_finemesh[:-1]
        avg_power = _integrate(self.pin_power, self.duct_power,
                               self.coolant_power, self.n_terms)
        return np.sum(dz_finemesh * avg_power)

    def calculate_pin_power_skew(self):
        """Calculate the radial pin power peaking for use in calculating
        the modified critical Grashoff number"""
        # If zero power, just return 1.0; no skew!
        if self.calculate_total_power() < 1e-12:
            return 1.0

        # Integrate pin powers
        z_bnds = np.array([-0.5, 0.5])
        z_bnds = z_bnds.reshape(2, 1)
        int_exponents = np.arange(1, self.n_terms + 1)
        z_int = np.power(z_bnds, int_exponents)
        # Integrate in each region, evaluate at lower/upper bound
        # shape is n_region x n_pin x 2 (upper/lower bound)
        integrated = np.dot(self.pin_power / int_exponents, z_int.T)
        # Take the difference across the region
        # shape is n_region x n_pin
        diff_across_region = integrated[:, :, 1] - integrated[:, :, 0]
        # multiply linear power by z-bounds and sum to get total power
        # in each pin
        dz_finemesh = self.z_finemesh[1:] - self.z_finemesh[:-1]
        power_per_pin = np.dot(dz_finemesh, diff_across_region)
        # Now can evaluate skew
        return np.max(power_per_pin) / np.average(power_per_pin)

    def save_to_file(self, path, asm_id, pin=True, duct=True, cool=True):
        """Save power profiles to CSV"""
        s = ''
        # Collect attributes to write
        attr = []
        if pin:
            attr.append('pin_power')
        if duct:
            attr.append('duct_power')
        if cool:
            attr.append('coolant_power')
        # Write selected power profiles to file
        for i in range(len(attr)):
            tmp = getattr(self, attr[i])
            n_ax_regions, n_component, n_terms = tmp.shape
            for reg in range(n_ax_regions):
                for comp in range(n_component):
                    s += ','.join([str(asm_id),    # assembly ID
                                   str(i + 1),     # component ID
                                   str(self.z_finemesh[reg]),
                                   str(self.z_finemesh[reg + 1]),
                                   str(comp + 1),  # component index
                                   ])
                    s += ','
                    # Convert coefficients from W/cm to W/m
                    s += ','.join([str(x * 100) for x in tmp[reg, comp]])
                    # s += ','.join(
                    #     [str(asm_id),                         # assembly ID
                    #      str(i + 1),                          # component ID
                    #      str(self.z_finemesh[reg] / 100.0),     # cm -> m
                    #      str(self.z_finemesh[reg + 1] / 100.0), # cm -> m
                    #      str(comp + 1),                         # index
                    #      ])
                    # s += ','
                    # data = tmp[reg, comp]
                    # conv = np.power([100], np.arange(1, data.shape[0] + 1))
                    # data = data * conv
                    # s += ','.join([str(x) for x in data])
                    s += '\n'
        with open(path, 'a') as f:
            f.write(s)


def _from_file(fpath):
    """Read pin, duct, and coolant linear power profiles from CSV

    Parameters
    ----------
    fpath : str
        Absolute path to DASSH power CSV file

    Returns
    -------
    list
        List of tuples containing the assembly ID, as provided by the
        user, and a dictionary with the z-mesh and power profiles for
        the pins, duct, and coolant in that assembly

    """
    with open(fpath, 'r', encoding='utf-8-sig') as f:
        pp = np.loadtxt(f, delimiter=',')

    # Axial region boundaries in m; need cm
    # Power dist coeffs are in W/m; need W/cm
    pp[:, 2] *= 100.0  # m --> cm
    pp[:, 3] *= 100.0  # m --> cm
    for i in range(pp.shape[1] - 5):
        # pp[:, i + 5] /= 100**(i + 1)  # W/m^i --> W/cm^i
        pp[:, i + 5] /= 100  # W/m^i --> W/cm^i

    components = ['pins', 'duct', 'cool']
    assembly_power_params = []
    # For every assembly
    for a in np.unique(pp[:, 0]):
        params = {}
        tmp1 = pp[pp[:, 0] == a]
        for ci in range(3):
            c = components[ci]
            tmp2 = tmp1[tmp1[:, 1] == ci + 1]
            if tmp2.size == 0:
                # params[c] = None
                continue

            # Do checks on component indexing and axial region specifications
            _check_axial_reg_same_bnds_for_all_entries(tmp2)
            _check_axial_reg_no_gaps_between_regions(tmp2)
            _check_component_indexing(tmp2)

            # Z-finemesh for component: if not yet defined in dict,
            zfm = np.unique(tmp2[:, (2, 3)])
            dim1 = zfm.shape[0] - 1  # number of regions
            if 'zfm' not in params.keys():
                params['zfm'] = zfm
            else:
                _check_axial_reg_between_materials(params['zfm'], zfm, a)

            dim2 = len(np.unique(tmp2[:, 4]))  # Component index
            dim3 = tmp2.shape[1] - 5  # Number of terms

            # Add data to dictionary
            params[c] = tmp2[:, -dim3:].reshape(dim1, dim2, dim3)

            # Check for negative linear power
            _check_for_negative_power(params[c], c, a)

        # If they pass, integrate to get average power
        params['avg_power'] = _integrate(params.get('pins'),
                                         params.get('duct'),
                                         params.get('cool'),
                                         dim3)

        # Then assign to list for return
        assembly_power_params.append((a, params))
    return assembly_power_params


def _check_component_indexing(mat_arr):
    """Naive check on pin/duct/subchannel indexing on user-input power"""
    msg = ('Error in user-specified power distribution: Assembly {0} {1} '
           'indexing; distribution; in axial region {2} < z < {3}, need {4} '
           'items, but found {5}.')
    N_axial_regions = len(np.unique(mat_arr[:, 2]))
    N_idx = np.max(mat_arr[:, 4])
    error = False
    if not mat_arr.shape[0] == N_axial_regions * N_idx:
        for z_lo in np.unique(mat_arr[:, 2]):
            tmp = mat_arr[mat_arr[:, 2] == z_lo]
            idx_in_reg = np.unique(tmp[:, 4])
            if idx_in_reg.shape[0] == N_idx:
                if not np.allclose(np.arange(1, N_idx + 1), idx_in_reg):
                    error = True
                else:
                    continue
            else:
                error = True
            if error:
                msg = msg.format(int(mat_arr[0, 0]),
                                 _user_power_mat_ids[int(mat_arr[0, 1]) - 1],
                                 tmp[0, 2] / 100,
                                 tmp[0, 3] / 100,
                                 int(N_idx),
                                 len(np.unique(tmp[:, 4]))
                                 )
                module_logger.log(40, msg)
                sys.exit()


def _check_axial_reg_between_materials(zfm1, zfm2, a):
    """Pins, coolant, and duct must have same axial region definitions;
    check against already accepted region definitions

    Parameters
    ----------
    zfm1 : numpy.ndarray
        Accepted fine-mesh axial region boundaries
    zfm2 : numpy.ndarray
        Fine-mesh axial region boundaries for incoming profiles
    a : int
        Assembly ID (for error message)
    """
    if not np.array_equal(zfm1, zfm2):
        msg = ('Error in user-specified power distribution '
               f'(assembly ID: {int(a)}); all pins, duct cells, and coolant'
               'items must have identical axial region boundaries.')
        module_logger.log(40, msg)
        sys.exit()


def _check_axial_reg_same_bnds_for_all_entries(mat_arr):
    """For pins/coolant/duct, all items must have the same axial regions

    Parameters
    ----------
    mat_arr : numpy.ndarray
        Power profile array for one material (pins or duct or coolant)
        from one assembly

    """
    msg = ('Error in axial bound entries of user-specified power distribution '
           'for assembly {0} {1}; all need to have the same region bounds.')
    msg = msg.format(int(mat_arr[0, 0]),
                     _user_power_mat_ids[int(mat_arr[0, 1]) - 1])
    # Check that for each lower bound, all upper bounds are the same.
    for z_lo in np.unique(mat_arr[:, 2]):
        tmp = mat_arr[mat_arr[:, 2] == z_lo]
        if not np.all(tmp[:, 3] == tmp[0, 3]):
            module_logger.log(40, msg)
            sys.exit()
    # Check that for each upper bound, all lower bounds are the same.
    for z_hi in np.unique(mat_arr[:, 3]):
        tmp = mat_arr[mat_arr[:, 3] == z_hi]
        if not np.all(tmp[:, 2] == tmp[0, 2]):
            module_logger.log(40, msg)
            sys.exit()


def _check_axial_reg_no_gaps_between_regions(mat_arr):
    """Check that there are no gaps between previous region upper bound
    and next region lower bound

    Parameters
    ----------
    mat_arr : numpy.ndarray
        Power profile array for one material (pins or duct or coolant)
        from one assembly

    """
    msg = ('Error in axial bound entries of user-specified power distribution'
           'for assembly {0} {1}; no gaps or overlaps allowed between upper/ '
           'lower bounds of successive regions')
    msg = msg.format(int(mat_arr[0, 0]),
                     _user_power_mat_ids[int(mat_arr[0, 1]) - 1])
    zlo = np.unique(mat_arr[:, 2])  # returns sorted values
    zhi = np.unique(mat_arr[:, 3])  # returns sorted values
    if not np.allclose(zhi[:-1] - zlo[1:], 0):
        module_logger.log(40, msg)
        sys.exit()


def _check_for_negative_power(power_profile, component, a_id):
    """No negative linear power encountered: test 11 points in the
    space to confirm none are less than zero"""
    z_test = np.linspace(-0.5, 0.5, 11)
    z_test = z_test.reshape(len(z_test), 1)
    z_exp = np.power(z_test, np.arange(power_profile.shape[2]))
    tmp = np.dot(power_profile, z_exp.T)
    # print(power_profile.shape)
    # print(z_test.shape)
    # print(z_exp.shape)
    # print(tmp.shape)
    if not np.min(tmp) >= 0.0:
        loc = np.where(tmp < 0.0)
        # print(loc)
        # Get axial region, component ID pairs
        loc = [(loc[0][i], loc[1][i]) for i in range(len(loc[0]))]
        loc = list(set(loc))
        reg = component.split('_')[0]
        msg = ('Error in user-specified power distribution '
               f'(assembly ID: {a_id}); found negative linear power '
               f'in {reg}:\n')
        for i in range(len(loc)):
            msg += f'Axial region: {str(loc[i][0] + 1).rjust(3)}; '
            msg += f'Index: {str(loc[i][1] + 1).rjust(3)}\n'
        module_logger.log(40, msg)
        sys.exit()


def _check_assembly(asm_power, asm_obj):
    """Check that user-input power matches assembly specifications"""
    pre = 'Assembly {0} has incorrect number of '
    if asm_obj.has_rodded:
        # Number of pins
        if asm_power.pin_power is not None:
            msg = pre + 'pins: require {1}, found {2}'
            found = asm_power.pin_power.shape[1]
            needed = asm_obj.rodded.n_pin
            if found != needed:
                return (False, msg.format('{0}', needed, found))
        # Number of duct elements
        if asm_power.duct_power is not None:
            msg = pre + 'duct elements: require {1}, found {2}'
            found = asm_power.duct_power.shape[1]
            needed = asm_obj.rodded.subchannel.n_sc['duct']['total']
            needed *= (asm_obj.rodded.n_bypass + 1)
            if found != needed:
                return (False, msg.format('{0}', needed, found))
        # Number of coolant subchannels
        if asm_power.coolant_power is not None:
            msg = pre + 'coolant subchannels: require {1}, found {2}'
            found = asm_power.coolant_power.shape[1]
            needed = asm_obj.rodded.subchannel.n_sc['coolant']['total']
            if found != needed:
                return (False, msg.format('{0}', needed, found))
    # Otherwise, return with no error
    return (True, None)


def _check_core_len(asm_power, core_len):
    """Check that user-input power axial regions go from z=0.0 to z=core_len"""

    msg = 'Assembly {0} user power lower bound must be 0.0 m; found {1} m'
    zlo = np.round(np.min(asm_power.z_finemesh) / 100.0, 6)  # cm --> m
    if zlo != 0.0:
        return (False, msg.format('{0}', zlo))
    msg = ('Assembly {0} user power upper bound must be equal '
           'to core length ({1} m); found {2} m')
    zhi = np.max(asm_power.z_finemesh) / 100.0  # cm --> m
    if np.round(zhi, 6) != np.round(core_len, 6):
        return (False, msg.format('{0}', core_len, zhi))
    # Otherwise, return with no error
    return (True, None)


def _integrate(pp_pins, pp_duct, pp_cool, n_terms):
    """Integrate the pin, duct, and coolant power profiles to get the
    average linear power in each axial region

    Parameters
    ----------
    pp_pins : numpy.ndarray
        Linear power (W/m) profile data for pins
    pp_duct : numpy.ndarray
        Linear power (W/m) profile data for duct
    pp_cool : numpy.ndarray
        Linear power (W/m) profile data for coolant
    n_terms : int
        Number of terms in the axial expansion

    Returns
    -------
    numpy.ndarray
        Average linear power in each axial region (size: N_region x 1)

    """
    # If all profiles are None, there is no power: result is 0
    if all(x is None for x in [pp_pins, pp_duct, pp_cool]):
        return 0.0

    # Otherwise, need to integrate
    # Determine number of axial regions from non-None profile
    n_reg = 0
    for pp in [pp_pins, pp_duct, pp_cool]:
        if pp is not None:
            n_reg = pp.shape[0]
            break
    avg_power = np.zeros((n_reg, 2))
    z_bnds = np.array([-0.5, 0.5])
    z_bnds = z_bnds.reshape(2, 1)
    int_exponents = np.arange(1, n_terms + 1)
    z_int = np.power(z_bnds, int_exponents)
    for pp in [pp_pins, pp_duct, pp_cool]:
        if pp is not None:
            tmp = np.dot(pp / int_exponents, z_int.T)
            avg_power += np.sum(tmp, axis=1)
    #
    return avg_power[:, 1] - avg_power[:, 0]


########################################################################

    # def renormalize(self, z, dz):
    #     """Need to normalize the power based on the discretization that
    #     DASSH will make during the sweep in order to accurately deliver
    #     the correct power
    #
    #     Parameters
    #     ----------
    #     z : numpy.ndarray
    #         Absolute axial points (m)
    #     dz : numpy.ndarray
    #         Mesh step sizes (m)
    #
    #     Returns
    #     -------
    #     numpy.ndarray
    #         Array of correction factors for each fine mesh interval
    #
    #     """
    #     # Preprocess the axial mesh points
    #     z = z[1:] * 100  # m --> cm
    #     z = z - 0.5 * dz  # get the axial mesh midpoints
    #     # Get the fine mesh intervals and the transformed z values for
    #     # each of the axial mesh points where power is evaluated.
    #     # kf = get_kfint2(self, z)
    #     # zm = transform_z2(self, kf, z)
    #     kf = self.get_kfint2(z)
    #     zm = self.transform_z2(kf, z)
    #     zm = np.expand_dims(zm, 1)
    #
    #     # Raise those transformed z values to the monomial exponent
    #     z_exp = np.power(zm, np.arange(self.n_terms))
    #     z_exp = np.expand_dims(z_exp, 1)
    #
    #     # Multiply the coefficients through and sum to get the total
    #     # power in each axial step: do this for pins, duct, and coolant
    #     mult_terms = self.pin_power[kf] * z_exp
    #     sum_terms = np.sum(mult_terms, axis=(1, 2))
    #     mult_terms = self.duct_power[kf] * z_exp
    #     sum_terms += np.sum(mult_terms, axis=(1, 2))
    #     mult_terms = self.coolant_power[kf] * z_exp
    #     sum_terms += np.sum(mult_terms, axis=(1, 2))
    #
    #     # Dimension is equal to n_step x 1; equals total in each step
    #     sum_terms = sum_terms * 100  # W/cm --> W/m
    #
    #     # Normalize each step linear power by step size
    #     dz_fint = self.z_finemesh[1:] - self.z_finemesh[:-1]
    #     dz_fint = dz_fint[kf]  # cm
    #     sum_terms = sum_terms * dz / dz_fint  # W/m * m / cm = W/cm
    #
    #     # Determine the corrective factor in each kfint
    #     x = np.zeros(len(self.z_finemesh) - 1)
    #     for i in range(len(x)):
    #         # self.avg_power has units of W/cm; so does sum_terms
    #         x[i] = self.avg_power[i] / np.sum(sum_terms[kf == i])
    #         # x[i] = self.avg_power[i] * 100 / np.average(sum_terms[kf == i])
    #     return x


########################################################################
# Old
# def calculate_total_power(self):
#     """Calculate the total power (W) produced by the assembly using
#     the linear power (W/m) shape functions."""
#
#     dz = [self.z_finemesh[k] - self.z_finemesh[k - 1]
#           for k in range(1, len(self.z_finemesh))]
#     p_total = 0.0
#     for k in range(len(dz)):
#         zc = self.z_finemesh[k] + dz[k] / 2  # center of mesh cell
#         if k < self.k_bnds[0] or k >= self.k_bnds[1]:
#             p_total += dz[k] * self.get_refl_power(zc)
#         else:
#             p_total += dz[k] * (np.sum(self.get_pin_power(zc))
#                                 + np.sum(self.get_duct_power(zc))
#                                 + np.sum(self.get_coolant_power(zc)))
#     return p_total
