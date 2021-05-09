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
date: 2021-05-06
author: matz
Cladding and fuel pin heat transfer model
"""
########################################################################
import numpy as np
from dassh.logged_class import LoggedClass
_SBCONST = 5.670374419e-8
_ERROR_MSG = """{0} temperature calculation did not converge;
iterations = {1}
max error = {2} K
"""


class FuelPin(LoggedClass):
    """Contains pin and clad geometry and methods to calculate clad
    midwall and pin centerline temperatures

    Parameters
    ----------
    d_pin : float
        Pin outer diameter (m)
    clad_thickness : float
        Cladding thickness (m)
    clad_mat : DASSH Material object
        Contains cladding thermal conductivity correlation
    fuel_params : dict
        Keys:
            'htc_params' : {list} Coefficients to Dittus Boelter
            'fcgap_thickness' : {float}
            'r_frac' : {list} Fractional radius (increasing)
            'pu_frac' : {list} Zr wt fraction per pellet radial node
            'zr_frac' : {list} Pu wt fraction per pellet radial node
            'porosity' : {list} Porosity per pellet radial node
    gap_mat : DASSH Material object
        Contains gap material thermal conductivity correlation

    """

    def __init__(self, d_pin, clad_thickness, clad_mat, fuel_params,
                 gap_mat=None, beta=2.0):
        """Instantiate FuelPin instance, set up geometric parameters"""
        LoggedClass.__init__(self, 0, f'dassh.FuelPin')

        # COOLANT-CLAD HTC PARAMETERS
        self.htc_params = fuel_params['htc_params_clad']
        fc_gap = fuel_params['fcgap_thickness']

        # LOAD CLADDING PARAMETERS
        self.clad = {}
        self.clad['k'] = clad_mat._data['thermal_conductivity']
        self.clad['r'] = np.array([0.0, 0.0, d_pin / 2])
        self.clad['r'][0] = self.clad['r'][2] - clad_thickness
        self.clad['r'][1] = self.clad['r'][2] - clad_thickness / 2
        self.clad['dr'] = clad_thickness / 2
        self.clad['ln_r2r_2node'] = [np.log(self.clad['r'][1]
                                            / self.clad['r'][0]),
                                     np.log(self.clad['r'][2]
                                            / self.clad['r'][1])]
        self.clad['ln_r2r'] = np.log(self.clad['r'][2]
                                     / self.clad['r'][0])

        # LOAD FUEL-CLAD GAP PARAMETERS
        self.gap = {}
        # Extract just the thermal conductivity correlation
        self.gap['dr'] = fc_gap
        self.gap['ln_rc_rf'] = np.log(self.clad['r'][0]
                                      / (self.clad['r'][0] - fc_gap))
        if fc_gap > 0:
            if gap_mat is not None:
                self.gap['k'] = gap_mat._data['thermal_conductivity']
            else:
                self.log('error', ('Gap material must be specified '
                                   'if nonzero gap thiccness given'))

        # LOAD FUEL PARAMETERS
        self.fuel = {}
        self.fuel['n_pts'] = len(fuel_params['r_frac'])
        # Consistency checks
        keys_to_check = ['r_frac', 'pu_frac', 'zr_frac', 'porosity']
        if not all([len(fuel_params[k]) == self.fuel['n_pts']
                    for k in keys_to_check]):
            self.log('error', ('Zr/Pu weight fraction, fractional '
                               'radius, and porosity arrays must '
                               'have equal length'))
        # Radial Node Geometry
        # Parameters are specified at the center of radial nodes
        # Secondary boundaries are created between radial nodes where
        # the temperatures are solved using average properties.

        #          r = 0      increasing r ----->       r = R_fuel
        #          T0            T1            T2            T3
        #          |======.//////|//////.xxxxxx|xxxxxx.------|
        #          |= Q1 =.///// Q2 ////.xxxx  Q3  xxx.  Q4  |
        #          |======.//////|//////.xxxxxx|xxxxxx.------|
        #          |      .      |      .      |      .      |
        #          |      .      |      .      |      .      |
        #          r0     .      r1     .      r2     .     r3
        #          |      .      |      .      |      .      |
        #          | <-- dr0 --> | <-- dr1 --> | <-- dr2 --> |
        #          |      .      |      .      |      .      |
        #          rm0    rm1           rm2           rm3    rm4

        # Radial node bounds
        # fuel_params['r_frac'].append(1.0)
        radius_out_fuel = self.clad['r'][0] - fc_gap
        self.fuel['r'] = np.zeros((self.fuel['n_pts'], 2))
        for reg in range(self.fuel['n_pts'] - 1):
            self.fuel['r'][reg][0] = fuel_params['r_frac'][reg]
            self.fuel['r'][reg][1] = fuel_params['r_frac'][reg + 1]
        if not fuel_params['r_frac'][-1] == 1.0:
            self.fuel['r'][-1] = [fuel_params['r_frac'][-1], 1.0]

        self.fuel['r'] *= radius_out_fuel
        # 2021-05-06 NEW SHIT
        self.fuel['drsq_over_4'] = 0.25 * (self.fuel['r'][:, 1]**2
                                           - self.fuel['r'][:, 0]**2)
        # Distances between mesh boundaries (Eq. 3.3-17 - 3.3-19)
        self.fuel['dr'] = self.fuel['r'][:, 1] - self.fuel['r'][:, 0]
        # Fuel cross-sectional area
        self.fuel['area'] = np.pi * radius_out_fuel**2

        # Radial node midpoints: this is where we know the fuel
        # conductivity parameters specified in the input and define
        # the radial boundaries for the power distribution
        # Remember that r[:, 0] is the shell inner radius
        self.fuel['rm'] = self.fuel['r'][:, 0] + self.fuel['dr'] / 2
        # Add the inner/outer fuel radii to the midpoints; this is
        # necessary for the power calculation
        self.fuel['rm'] = np.insert(self.fuel['rm'], 0,
                                    self.fuel['r'][0, 0])
        self.fuel['rm'] = np.append(self.fuel['rm'],
                                    self.fuel['r'][-1, 1])
        # Radius squared (for power distribution into fuel segments)
        self.fuel['rmsq'] = self.fuel['rm']**2
        self.fuel['drmsq'] = (self.fuel['rmsq'][1:]
                              - self.fuel['rmsq'][:-1])
        self.fuel['drmsq'] = self.fuel['drmsq'].transpose()

        if 'emissivity' in fuel_params.keys():
            self.fuel['e'] = fuel_params['emissivity']
        else:
            self.fuel['e'] = 0.9  # this is the SE2ANL default

        # Thermal conductivity coefficients
        # k = (a + bT + cT**2) * C_IR
        # C_IR = Irradiation coefficient; a decrease in thermal cond.
        #        due to increased porosity in the fuel)
        # For references, see:
        # - "Metallic Fuels Handbook" Section C.1.2 (1986).
        # - R. Vilim "Simple Fuel Pin Model for SE2ANL" (1987).
        self.fuel['kc'] = np.zeros((self.fuel['n_pts'], 3))
        for i in range(self.fuel['n_pts']):
            self.fuel['kc'][i, 0] = \
                17.5 * ((1 - 2.23 * fuel_params['zr_frac'][i])
                        / (1 + 1.61 * fuel_params['zr_frac'][i])
                        - 2.62 * fuel_params['pu_frac'][i])
            self.fuel['kc'][i, 1] = \
                0.0154 * ((1 + 0.061 * fuel_params['zr_frac'][i])
                          / (1 + 1.61 * fuel_params['zr_frac'][i])
                          + 0.90 * fuel_params['pu_frac'][i])
            self.fuel['kc'][i, 2] = \
                9.38e-6 * (1 - 2.7 * fuel_params['pu_frac'][i])
            self.fuel['kc'][i] *= \
                ((1 - fuel_params['porosity'][i])
                 / (1 + beta * fuel_params['porosity'][i]))

    def calculate_temperatures(self, q_lin, T_cool, htc, dz, atol=1e-3):
        """Calculate cladding and fuel pellet temperatures

        Parameters
        ----------
        q_lin : float
            Linear heat rate (W/m) in the fuel pin at this position
        T_cool : float
            Nominal coolant temperature (K) in the subchannels
            surrounding the pin
        htc : float
            Heat transfer coefficient (W/m2K) between pin and coolant
        dz : float
            Axial step size (m)
        atol : float
            Absolute tolerance for thermal conductivity iterations

        Returns
        -------
        list
            Nominal local coolant, clad OD, MW, ID temperatures, and
            fuel OD and CL temperatures

        """
        t = np.zeros((len(q_lin), 6))
        t[:, 0] = T_cool

        # Distribute power
        # qsums = self._distribute_power(q_lin, dz)
        q_tot = q_lin * dz  # W/m --> W
        q_dens = q_lin / self.fuel['area']  # W/m --> W/m3
        # Calculate cladding inner surface, midwall temperatures
        # temps[:, 1:4] = self.calc_clad_temps(
        #     qsums[:, -1], dz, T_cool, htc, atol)
        t[:, 1:4] = self.calc_clad_temps(q_tot, dz, T_cool, htc, atol)

        # Calculate fuel surface temperature
        # temps[:, 4] = self.calc_fuel_surf_temp(
        #     qsums[:, -1], dz, temps[:, 3], atol)
        t[:, 4] = self.calc_fuel_surf_temp(q_tot, dz, t[:, 3], atol)

        # Calculate fuel centerline temperatures
        # temps[:, 5] = self.calc_fuel_temps(qsums, dz, temps[:, 4], atol)
        t[:, 5] = self.calc_fuel_temps(q_dens, t[:, 4], atol)
        return t

    def _distribute_power(self, q_lin, dz):
        """Distribute the power among the fuel nodes

        Parameters
        ----------
        q_lin : numpy.ndarray
            Linear power (W/m) in each pin at the current axial mesh
        dz : float
            Axial mesh step size (m)

        Returns
        -------
        numpy.ndarray
            Total heat (W) generated within each radial fuel node and
            all nodes interior to it for each pin

        Notes
        ----
        The clad heat is included with the fuel pin and no heat is
        generated in the clad. The effect should be extremely minor,
        because the clad heat is a tiny fraction of the overall heat
        generated.

        Furthermore, including clad heat with fuel pin heat should
        overestimate all temperatures except that at the clad outer
        surface. At the clad outer surface at steady state, all the
        heat generated in the pin/cladding needs to pass through. At
        the cladding inner surface, in reality only the pin heat passes
        through, because the clad heat is already "outside" it. If all
        is lumped into the fuel, more heat has to pass through so the
        temperatures should be higher.

        There is a similar effect at each of the radial nodes in the
        fuel pellet because the clad heat is distributed among them.
        This will result in a very small increase in temperatures
        throughout the fuel pellet.

        """
        q_dens = q_lin * np.pi * dz / self.fuel['area']  # W/m -> W/m3
        # Calculate the power at each radial node (Eq. 3.4-8)
        # q_node = np.zeros(len(self.fuel['rm']) - 1)
        # const = np.pi * q_density * dz
        q_node = np.ones((len(self.fuel['rm']) - 1, len(q_lin)))
        q_node *= q_dens
        q_node = q_node.transpose()
        q_node *= self.fuel['drmsq']
        # Calculate the power sum at each radial node (Eq. 3.4-9)
        return np.cumsum(q_node, axis=1)

    def calc_clad_temps(self, q, dz, T_cool, htc, atol=1e-6, ilim=20):
        """Calculate the change in temperature across the cladding

        Parameters
        ----------
        q : numpy.ndarray
            Total heat generation (W) in each pin at this axial mesh
        dz : float
            Axial mesh step size (m)
        T_cool : numpy.ndarray
            Average outer coolant temperature (K) around each pin
        htc : float
            Coolant heat transfer coefficient (W/m2K)
        atol (optional) : float
            Convergence criteria (absolute) for the temperature /
            thermal conductivity iterations at each radial node
            (default = 1e-6)
        ilim (optional) : int
            Iteration limit before nonconvergence error is raised
            (default = 20)
        Returns
        -------
        nump.ndarray
            Temperature (K) at clad outer-, mid-, and inner-walls for
            each pin

        Notes
        -----
        Treat the cladding as a cylindrical shell with no internal heat
        generation at steady state.

        No loop is used since there are only two steps

        """
        # Define constants
        C = q / 2 / np.pi / dz

        # Cladding surface temperature
        T = np.zeros((len(q), 3))
        T[:, 2] = T_cool + C / htc / self.clad['r'][2]
        dT = C * self.clad['ln_r2r']
        k_ip1 = self.clad['k'](T[:, 2])
        T_in1 = T[:, 2] + dT / k_ip1
        T_in2 = T[:, 2]
        idx = 0
        while np.max(np.abs(T_in1 - T_in2)) > atol:

            # Estimate k(i) and calculate average
            # k = self._avg_cond(self.clad['k'](T_in1), k_ip1)
            k = 0.5 * (self.clad['k'](T_in1) + k_ip1)

            # Calculate T(i); shuffle placeholder tmperatures so
            # they can be compared for convergence
            T_in2 = T_in1
            T_in1 = T[:, 2] + dT / k
            idx += 1
            if idx > ilim:
                self.log('error', _ERROR_MSG.format(
                    'Clad', idx, np.max(T_in1 - T_in2)))
            # T_in1 = T[i + 1] + C[i] / k
        # print(idx)
        T[:, 0] = T_in1
        # T[:, 1] = np.average(T[:, (0, 2)], axis=1)
        T[:, 1] = T[:, 2] + C * self.clad['ln_r2r_2node'][1] / k
        # Return the cladding inner surface and midwall temperatures
        return np.fliplr(T)

    def calc_fuel_surf_temp(self, q, dz, T_clad, atol=1e-6, iter=10):
        """Calculate the temperature across the fuel-clad gap to
        determine the temperature of the fuel surface

        Parameters
        ----------
        q : numpy.ndarray
            Total heat generation (W) in each pin at this axial mesh
        dz : float
            Axial mesh step size (m)
        T_clad : numpy.ndarray
            Cladding inner surface temperature (K) for each pin
        fc_gap : float
            Thickness (m) of the fuel-clad gap
        atol (optional) : float
            Convergence criteria (absolute) for the temperature /
            thermal conductivity iterations at each radial node

        Returns
        -------
        numpy.ndarray
            Fuel surface temperature (K) for each pin
        """
        if self.gap['dr'] == 0.0:
            return T_clad
        else:
            # define constant
            # dT = q * self.gap['ln_rc_rf'] / 2 / np.pi / dz

            # From Eq 3.4-15, 16, 17; hb assumed equal to k/dr
            d1 = (self.gap['dr']
                  * (q / 2 / np.pi / dz / self.fuel['r'][-1, 1]
                     + self.fuel['e'] * _SBCONST * T_clad**4))
            d2 = self.gap['dr'] * self.fuel['e'] * _SBCONST
            k2 = self.gap['k'](T_clad)
            Tf1 = T_clad + d1 / k2 - d2 * T_clad**4 / k2
            Tf2 = T_clad
            idx = 0
            while np.max(np.abs(Tf1 - Tf2)) > atol:
                # k = self._avg_cond(self.gap['k'](Tf1), k2)
                k = 0.5 * (self.gap['k'](Tf1) + k2)
                # Calculate surface temp; shuffle placeholder temps
                # so they can be compared for convergence
                Tf2 = Tf1
                # Tf1 = T_clad + d1 / k - d2 * Tf2**4 / k
                Tf1 = T_clad + (d1 - d2 * Tf2**4) / k
                idx += 1
                if idx > iter:
                    self.log('error', _ERROR_MSG.format(
                        'Fuel-clad gap', idx, np.max(Tf1 - Tf2)))
            return Tf1

    def calc_fuel_temps(self, q_dens, T_out, atol=1e-6, iter=10):
        """Calculate the fuel centerline temperature

        Parameters
        ----------
        q_dens : numpy.ndarray
            Power density (W/m3) for each pin
        T_out : numpy.ndarray
            Fuel surface temperature (K) for each pin
        atol (optional) : float
            Convergence criteria (absolute) for the temperature /
            thermal conductivity iterations at each radial node

        Returns
        -------
        numpy.ndarray
            Fuel centerline temperature in each pin

        Notes
        -----
        Calculates the temperature increase across each node within
        concentric cylindrical shells to determine the temperature
        at the center using Eq. 3.4-11 from ANL-FRA-1996-3 Volume 1
        with iterations to determine thermal conductivity based on
        radial-node-averaged temperature (Eq. 3.3.-26)

        """
        # for i in reversed(range(q_sum.shape[1] - 1)):
        for i in reversed(range(self.fuel['drsq_over_4'].shape[0])):
            # Set up some constants (do not require iteration)
            # T_i = T_ip1 + dT/k; need to iterate on k
            # dT = (self.fuel['dr'][i] * q_sum[:, i] / 2 / np.pi
            #       / self.fuel['rm'][i + 1] / dz)
            dT = self.fuel['drsq_over_4'][i] * q_dens
            k_ip1 = self._fuel_cond(i, T_out)
            T_in1 = T_out + dT / k_ip1
            T_in2 = T_out
            idx = 0
            while np.max(np.abs(T_in1 - T_in2)) > atol:
                # Estimate k(i) and calculate average
                k_i = self._fuel_cond(i, T_in1)
                # k = self._avg_cond(k_i, k_ip1,
                #                    self.fuel['dr'][i - 1],
                #                    self.fuel['dr'][i])
                # k = self._fuel_cond(i, 0.5 * (T_in1 + T_in2))
                k = 0.5 * (k_i + k_ip1)
                # Calculate T(i); shuffle placeholder tmperatures so
                # they can be compared for convergence
                T_in2 = T_in1
                T_in1 = T_out + dT / k
                idx += 1
                if idx > iter:
                    self.log('error', _ERROR_MSG.format(
                        'Fuel CL', idx, np.max(T_in1 - T_in2)))

            # Set T_out (T(i+1)) equal to T(i) and move to next step
            print(iter)
            T_out = T_in1

        # Once the for loop is done, T_in1 is the centerline temp
        return T_in1

    @staticmethod
    def _avg_cond(k_i, k_ip1, dr_i=0.5, dr_ip1=0.5):
        """Weighted harmonic average of thermal conductivity values for
        two adjacent nodes (ANL-FRA-1996-3 Volume 1 Eq. 3.3.-26)

        Parameters
        ----------
        k_i : float
            Thermal conductivity (W/m-K) of radial node index i
        k_ip1 : float
            Thermal conductivity (W/m-K) of radial node index i + 1
        dr_i : float
            Width (m) of radial node index i
        dr_ip1 : float
            Width (m) of radial node index i + 1

        Returns
        -------
        float
            Average conductivity between nodes i and i + 1

        """
        return (k_i * k_ip1 * (dr_i + dr_ip1)
                / (k_i * dr_ip1 + k_ip1 * dr_i))

    def _fuel_cond(self, i, T):
        """Calculate the thermal conductivity in a radial fuel node

        Parameters
        ----------
        i : int
            Radial fuel node index
        T : float
            Temperature (K)

        Returns
        -------
        float
            Fuel thermal conductivity (W/m-K)

        """
        k = (self.fuel['kc'][i, 0] + self.fuel['kc'][i, 1] * T
             + self.fuel['kc'][i, 2] * T**2)
        if np.any(k < 0):
            self.log('error', 'Negative fuel thermal conductivity')
        return k
