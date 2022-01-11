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
date: 2022-01-11
author: matz
Test the clad/fuel pin temperature model
"""
########################################################################
import os
import numpy as np
import pytest
import dassh
from dassh import Material
from dassh import PinModel
from dassh.correlations import nusselt_db


def guess_se2anl_power(Tc_out, Tc_in, dz, rf, drc):
    """SE2-ANL doesn't tell you the power at any point, which sucks.
    So let's guess at it based on an analytical equation for heat
    transfer through a cylindrical shell, the SE2ANL clad thermal
    conductivity, and the SE2ANL temperatures"""

    # Force material to use SE2-ANL conductivity correlation: use
    # the midwall temperature
    k_se2anl = np.poly1d([4.01774e-3, 23.663354319])(Tc_out)
    lnr2r = np.log((rf + 2 * drc) / rf)
    q = (Tc_in - Tc_out) * 2 * np.pi * dz * k_se2anl / lnr2r
    return q  # W


def guess_se2anl_pin_htc(cool_mat, pin_pitch, pin_diam, Re, de):
    """Calculate HTC using Nusselt number"""
    p2d = pin_pitch / pin_diam
    dbc = [p2d**3.8 * 0.01**0.86 / 3.0, 0.86, 0.86, 4.0 + 0.16 * p2d**5]
    cool_mat.update(425.0 + 273.15)
    nu = nusselt_db.calculate_bundle_Nu(cool_mat, Re, consts=dbc)
    return cool_mat.thermal_conductivity * nu / de


########################################################################


def test_metal_fuel_conductivity():
    """Compare correlation values to those published with it in the
    Metallic Fuels Handbook (Table C.1.2-1)"""
    # Temperatures from table (in degrees C)
    T = np.array([20, 100, 200, 300, 400, 500, 600, 700, 800, 900])
    # Correlation values from table (in W/cm C)
    ans = np.array([[0.115, 0.131, 0.152, 0.173, 0.196,
                     0.220, 0.245, 0.271, 0.298, 0.327],
                    [0.101, 0.117, 0.137, 0.158, 0.181,
                     0.204, 0.229, 0.255, 0.282, 0.310],
                    [0.075, 0.092, 0.110, 0.130, 0.152,
                     0.174, 0.198, 0.222, 0.247, 0.274],
                    [0.081, 0.096, 0.116, 0.137, 0.158,
                     0.181, 0.204, 0.229, 0.254, 0.280]])

    # Geometry doesn't matter: just want conductivity correlation
    d_pin = 1.0
    clad_thickness = 0.05
    clad_mat = Material('ht9')
    args = [d_pin, clad_thickness, clad_mat]
    zr = [0.062, 0.097, 0.115, 0.10]
    pu = [0.162, 0.147, 0.184, 0.190]
    for i in range(len(ans)):
        fp = PinModel(*args, {'r_frac': [0.0], 'zr_frac': [zr[i]],
                              'pu_frac': [pu[i]], 'porosity': [0.0],
                              'gap_thickness': 0.0,
                              'htc_params_clad': [0.0, 0.0, 0.0, 0.0]})

        for j in range(len(T)):
            # res = np.round(fp._fuel_cond(0, T[j] + 273.15), 1)
            res = fp._fuel_cond(0, T[j] + 273.15)
            error = res - ans[i, j] * 100.0  # convert k to W/m-K
            print(T[j], res, ans[i, j] * 100, error)
            assert np.abs(error) < 0.2


def test_energy_conservation(pin):
    """Test that energy is conserved when power is distributed
    radially in the pellet"""
    q = np.array([1000.0])  # W
    dz = 0.025  # m
    # q' (W/m) is input because that's what we have from DASSH
    q_s = OLD_distribute_power(pin, q / dz, dz)
    # Confirm that last entry is equal to total power
    print(pin.fuel['r'])
    print(pin.fuel['dr'])
    print(pin.fuel['rm'])
    print(q_s)
    assert q_s[0, -1] == pytest.approx(q)


def test_fuel_temp_calc_const_k(pin, se2anl_peaktemp_params):
    """Set the fuel thermal conductivity constant to test the numerical
    solution against the analytical solution for heat generation in a
    homogeneous cylinder"""
    # Simulation parameters: unpack them into a better name
    params = se2anl_peaktemp_params
    # Pin power (W)
    q = guess_se2anl_power(params['T_clad'][0, 0],
                           params['T_clad'][0, 2],
                           params['dz'],
                           params['r_fuel'],
                           params['dr_clad'])
    # Replace pin conductivity model with constant
    k = 10.0
    pin._fuel_cond = lambda i, T: k

    # Calculate temperature across fuel pellet - should be same
    # as for cylinder with heat generation using a single k
    q = np.array([q])
    # qs = pin._distribute_power(q / params['dz'], params['dz'])
    # T_cl = pin.calc_fuel_temps(qs, params['dz'], params['T_fuel'][0, 0])
    # # print(qs)
    # print(pin.fuel['r'])
    # print(pin.fuel['dr'])
    # print(pin.fuel['rm'])
    # print(pin.fuel['drmsq'])
    # print(pin.fuel['drsq_over_4'])
    # Compare with simple cylinder calculation
    qtp = q[0] / params['dz'] / pin.fuel['area']  # W/m3
    T_cl = pin.calc_fuel_temps(qtp, params['T_fuel'][0, 0])
    # print(qtp)
    ans = params['T_fuel'][0, 0] + qtp * pin.fuel['r'][-1, 1]**2 / 4 / k
    # print('dT_ans', qtp * pin.fuel['r'][-1, 1]**2 / 4 / k)
    # print('dT_res', T_cl - params['T_fuel'][0, 0])
    # print(ans)
    # print(T_cl)
    # assert 0
    assert ans == pytest.approx(T_cl)


def test_fuel_surf_temp(pin_boc, pin):
    """Test that the fuel surface temperature calculation captures
    important behaviors (e.g. dT_gap > dT_nogap)"""
    qlin = np.array([2e4, 2.2e4, 2.3e4, 2.1e4])  # W/m
    dz = 0.025  # m
    T_clad = np.array([820.0, 790.0, 850.0, 800.0])  # K
    Ts_no_gap = pin.calc_fuel_surf_temp(qlin * dz, dz, T_clad)
    Ts_gap = pin_boc.calc_fuel_surf_temp(qlin * dz, dz, T_clad)
    # Surface temp should be higher with gap: added resistance
    print(Ts_no_gap, Ts_gap)
    assert np.all(Ts_no_gap < Ts_gap)

    # Can I recreate the surface temperature as if it was a conducting
    # cylinder (given that the emissivity is so low)
    # Guess average thermal conductivity
    # pin_boc.fuel['e'] = 0.0
    # Ts_gap = pin_boc.calc_fuel_surf_temp(qlin * dz, dz, T_clad)
    k1 = pin_boc.gap['k'](T_clad)
    k2 = pin_boc.gap['k'](Ts_gap)
    # k = pin_boc._avg_cond(k1, k2)
    k = 0.5 * (k1 + k2)
    ans = T_clad + (qlin
                    * np.log(pin_boc.clad['r'][0]
                             / pin_boc.fuel['r'][-1, 1])
                    / 2 / np.pi / k)
    error = ans[0] - Ts_gap[0]
    print(Ts_gap, ans, error)
    # Expect error around 0.5 K because the method in FuelPin doesn't
    # use natural logs; the error is roughly proportional to the ratio
    # of ln(ro/ri) and dr/ro
    assert np.abs(error) < 1.0


def test_se2anl_comparison(pin, se2anl_peaktemp_params):
    """Compare values against those produced by SE2ANL.

    Acceptable results
    ------------------
    - Difference in dT across cladding (K): 1.0
    - Difference in dT across cladding (%): 1.0
    - Difference in dT across fuel pellet (K): 1.3
        (formerly 1.0, changed 2021-05-06)
    - Difference in dT across fuel pellet (%): 1.0

    """
    # Simulation parameters: unpack them into a better name
    params = se2anl_peaktemp_params
    q = np.zeros(2)
    for i in range(len(params['T_clad'])):
        q[i] = guess_se2anl_power(params['T_clad'][i, 0],
                                  params['T_clad'][i, 2],
                                  params['dz'],
                                  params['r_fuel'],
                                  params['dr_clad'])

    params['cool'].update(params['T_cool'][0])
    htc = guess_se2anl_pin_htc(params['cool'],
                               params['pin_pitch'],
                               params['pin_diameter'],
                               params['Re'],
                               params['bundle_de'])

    print(f'q (W): {q}')
    print(f'htc (W/m2K): {htc}')

    # Give it to the fuel pin model and see what you get
    res = pin.calculate_temperatures(q / params['dz'],
                                     params['T_cool'],
                                     htc, params['dz'])
    # Clad temperature results
    dT_clad_dassh = res[:, 3] - res[:, 1]
    dT_clad_se2 = params['T_clad'][:, 2] - params['T_clad'][:, 0]
    print(f'T_clad (K; result):\n{str(res[:, 1:4])}')
    print(f'T_clad (K; ans):\n{str(params["T_clad"])}')
    # Hope for less than 1 degree absolute error in result
    assert np.max(np.abs(dT_clad_dassh - dT_clad_se2)) < 1.0
    # Hope for less than 1% relative error in result
    assert np.max(np.abs(dT_clad_dassh - dT_clad_se2) / dT_clad_se2) < 0.01

    # Fuel temperature results
    dT_fuel_dassh = res[:, 5] - res[:, 4]
    dT_fuel_se2 = params['T_fuel'][:, 1] - params['T_fuel'][:, 0]
    print(np.max(np.abs(dT_fuel_dassh - dT_fuel_se2) / dT_fuel_se2))
    print(f'T_fuel (K; result):\n{str(res[:, 4:])}')
    print(f'T_fuel (K; ans):\n{str(params["T_fuel"])}')
    # Hope for less than 1 degree absolute error in result
    # 2021-05-06: changed from harmonic average of thermal conductivity
    # to arithmetic average: changes result. I found that the arithmetic
    # average does a better job getting the right answer on a coarse
    # mesh (as compared to a fine one), so I'm less interested that it
    # makes this test worse
    assert np.max(np.abs(dT_fuel_dassh - dT_fuel_se2)) < 1.3
    # Hope for less than 1% relative error in result
    assert np.max(np.abs(dT_fuel_dassh - dT_fuel_se2) / dT_fuel_se2) < 0.01


def test_verify_pin_temperatures(testdir, pin):
    """Compare pin temperatures against what's produced by DASSH;
    acceptance tolerance is 0.001 K absolute difference"""
    ans_file = os.path.join(testdir, 'test_data', 'pin0_verification.csv')
    ans = np.loadtxt(ans_file, skiprows=3, delimiter=',')
    # Columns: z (m), dz (m), power (w/m), HTC (W/m2K), T_coolant (K),
    # T clad OD, T clad MW, T clad ID, T fuel OD, T fuel CL
    T_out = pin.calculate_temperatures(ans[:, 2], ans[:, 4],
                                       ans[:, 3], ans[:, 1])
    error = T_out - ans[:, 4:]
    print(T_out[0])
    print(ans[0, 4:])
    print(error[0])
    print('Max abs error: ', np.max(error))
    assert np.allclose(T_out, ans[:, 4:], atol=1e-4)


def test_check_new_fuel_calc(pin, se2anl_peaktemp_params):
    """Check that new and old pin calculations give same result"""

    # Old function to distribute power to fuel nodes
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

    # Old function to calculate fuel temperatures
    def calc_fuel_temps_OLD(self, q_sum, dz, T_out, atol=1e-6, iter=10):
        """Calculate the fuel centerline temperature

        Parameters
        ----------
        self : DASSH FuelModel object
        q_sum : numpy.ndarray
            Power (W) at each radial node in the pellet for each pin
        dz : float
            Axial mesh step size
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
        # Define constant
        for i in reversed(range(q_sum.shape[1] - 1)):
            # Set up some constants (do not require iteration)
            # T_i = T_ip1 + dT/k; need to iterate on k
            dT = (self.fuel['dr'][i] * q_sum[:, i] / 2 / np.pi
                  / self.fuel['rm'][i + 1] / dz)
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
                k = 0.5 * (k_i + k_ip1)
                # Calculate T(i); shuffle placeholder tmperatures so
                # they can be compared for convergence
                T_in2 = T_in1
                T_in1 = T_out + dT / k
                idx += 1
                if idx > iter:
                    self.log('error', 'iterations exceeded')
                    # self.log('error', _ERROR_MSG.format(
                    #     'Fuel CL', idx, np.max(T_in1 - T_in2)))
            # Set T_out (T(i+1)) equal to T(i) and move to next step
            T_out = T_in1
        # Once the for loop is done, T_in1 is the centerline temp
        return T_in1

    # Simulation parameters: unpack them into a better name
    params = se2anl_peaktemp_params
    # Pin power (W)
    q = guess_se2anl_power(params['T_clad'][0, 0],
                           params['T_clad'][0, 2],
                           params['dz'],
                           params['r_fuel'],
                           params['dr_clad'])

    # Calculate temperature across fuel pellet - should be same
    # as for cylinder with heat generation using a single k
    q = np.array([q])
    q_lin = q / params['dz']  # W --> W/m
    q_dens = q[0] / params['dz'] / pin.fuel['area']  # W --> W/m3
    T_out = params['T_fuel'][0, 0]
    q_sums = _distribute_power(pin, q_lin, params['dz'])
    T_cl_old = calc_fuel_temps_OLD(pin, q_sums, params['dz'], T_out)
    T_cl = pin.calc_fuel_temps(q_dens, T_out)
    assert T_cl == pytest.approx(T_cl_old)


def OLD_distribute_power(self, q_lin, dz):
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
    This was the original method I used in fuel pin temperature
    calculation. Saving here for testing purposes.

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


########################################################################
# TEST GENERAL PIN MODEL
########################################################################


def test_general_pin_conductivity(testdir):
    """x"""
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'input_general_pinmodel_conductivity_check.txt'))
    asm_input = inp.data['Assembly']['driver']
    p2d = (asm_input['pin_pitch'] / asm_input['pin_diameter'])
    asm_input['PinModel']['htc_params_clad'] = \
        [p2d**3.8 * 0.01**0.86 / 3.0, 0.86, 0.86, 4.0 + 0.16 * p2d**5]
    asm_input['PinModel']['pin_material'] = \
        [inp.materials[m].clone()
         for m in asm_input['PinModel']['pin_material']]
    pm = PinModel(asm_input['pin_diameter'],
                  asm_input['clad_thickness'],
                  inp.materials[asm_input['PinModel']['clad_material']],
                  pin_params=asm_input['PinModel'],
                  gap_mat=None)
    res = pm._fuel_cond(0, 800)
    ans = 20.0 - 0.03 * 800 + 0.00002 * 800**2
    assert pytest.approx(res, ans)
    res = pm._fuel_cond(1, 1000)
    ans = 12.0 + 0.04 * 800 - 0.00005 * 800**2
    assert pytest.approx(res, ans)
    res = pm._fuel_cond(2, 23525)
    ans = 20.0
    assert pytest.approx(res, ans)


def test_verify_general_pin_model(testdir):
    """Check pin model against hand calculation"""
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'input_general_pinmodel.txt'))
    asm_input = inp.data['Assembly']['driver']
    p2d = (asm_input['pin_pitch'] / asm_input['pin_diameter'])
    asm_input['PinModel']['htc_params_clad'] = \
        [p2d**3.8 * 0.01**0.86 / 3.0, 0.86, 0.86, 4.0 + 0.16 * p2d**5]
    asm_input['PinModel']['pin_material'] = \
        [inp.materials[m].clone()
         for m in asm_input['PinModel']['pin_material']]
    pm = PinModel(asm_input['pin_diameter'],
                  asm_input['clad_thickness'],
                  inp.materials[asm_input['PinModel']['clad_material']],
                  pin_params=asm_input['PinModel'],
                  gap_mat=None)
    # Calculate pin temperatures - fills in rr.pin_temps
    dz = 0.01
    pp = np.array([1e4])
    htc = 1e4
    Tc = 623.15
    res = pm.calculate_temperatures(pp, Tc, htc, dz)
    # Calculate answer - clad_out, clad_mw, clad_in, fuel_surf, fuel_cl
    ans = np.zeros(5)
    k_clad = 25.0
    k_fuel = 20.0
    t_clad = inp.data['Assembly']['driver']['clad_thickness']
    ro_clad = 0.5 * inp.data['Assembly']['driver']['pin_diameter']
    ri_clad = ro_clad - t_clad
    rm_clad = ro_clad - t_clad * 0.5
    a_pin = np.pi * ri_clad**2
    ans[0] = Tc + pp / htc / 2 / np.pi / ro_clad
    ans[1] = ans[0] + pp * np.log(ro_clad / rm_clad) / 2 / np.pi / k_clad
    ans[2] = ans[0] + pp * np.log(ro_clad / ri_clad) / 2 / np.pi / k_clad
    ans[3] = ans[2]
    ans[4] = ans[3] + pp * ri_clad**2 / a_pin / 4 / k_fuel
    assert np.allclose(ans, res[0, 1:])
