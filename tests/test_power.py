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
date: 2021-05-27
author: matz
Test power assignment from binary files to reactor components
"""
# Notes: no tests for dummy pins (capability not implemented)
########################################################################
import numpy as np
import pytest
import os
import copy
import subprocess
# import matplotlib.pyplot as plt
import dassh
from dassh import py4c

########################################################################
# Test Power object (holds DIF3D power distribution)
########################################################################


@pytest.mark.filterwarnings("ignore")  # ignore negative power warning
def test_total_core_power_scaled(small_core_power):
    """Test that the Power object calculates the correct power in a
    30.0+ MWth core (only 7 assemblies lol); incl. 0.1 scaling factor"""
    ans = 30.8261  # a little more than 30.0 because gammes
    res = np.sum(small_core_power.power) / 1e6
    err = (res - ans) / ans
    print('ans =', ans, 'res =', res, 'err =', err)
    assert res == pytest.approx(ans, 6)


@pytest.mark.filterwarnings("ignore")  # ignore negative power warning
def test_normalized_total_core_power(small_core_power_normalized):
    """Test that the Power object calculates the correct power in a
    30 MWth core (only 7 assemblies lol)"""
    ans = 30.0
    res = np.sum(small_core_power_normalized.power) / 1e6
    err = (res - ans) / ans
    print('ans =', ans, 'res =', res, 'err =', err)
    assert res == pytest.approx(ans)


@pytest.mark.filterwarnings("ignore")  # ignore negative power warning
def test_total_core_power(small_core_power_unscaled):
    """Test that the Power object calculates the correct power in a
    300+ MWth core (only 7 assemblies lol)"""
    ans = 308.261  # a little more than 300 because gammas
    res = np.sum(small_core_power_unscaled.power) / 1e6
    err = (res - ans) / ans
    print('ans =', ans, 'res =', res, 'err =', err)
    assert res == pytest.approx(ans, 6)


# @pytest.mark.skip(reason='no longer raises negative power warning')
# def test_negative_power_warning(testdir, caplog):
#     """Test that the Power object catches the negative powers caused
#     by negative fluxes in low-flux regions"""
#     res_path = os.path.join(testdir, 'test_results', 'seven_asm_vac',
#                             'power')
#     dassh.power.Power(
#         os.path.join(res_path, 'MaterialPower.out'),
#         os.path.join(res_path, 'VariantMonoExponents.out'),
#         os.path.join(res_path, 'Output.VARPOW'),
#         os.path.join(res_path, 'GEODST'),
#         warn_negative=True)
#     # assert len(caplog.records) == 2
#     assert all([r.levelname == 'WARNING' for r in caplog.records])
#     assert 'Negative' in caplog.text


@pytest.mark.filterwarnings("ignore")  # ignore negative power warning
def test_user_power_normalization(testdir):
    """."""
    user_power = 1e8  # 100 MW rather than 300 MW
    res_path = os.path.join(testdir, 'test_results', 'seven_asm_vac',
                            'power')
    p = dassh.power.Power(
        os.path.join(res_path, 'MaterialPower.out'),
        os.path.join(res_path, 'VariantMonoExponents.out'),
        os.path.join(res_path, 'Output.VARPOW'),
        os.path.join(res_path, 'GEODST'),
        user_power=user_power)
    assert np.sum(p.power) == user_power


@pytest.mark.filterwarnings("ignore")  # ignore negative power warning
def test_power_profile_reintegration(small_core_power, c_fuel_asm):
    """Confirm that the power profile can be reintegrated to give
    the expected total power in each component at each axial mesh"""
    # Structure volume fractions
    svf = dassh.power.calculate_structure_vfs(c_fuel_asm.rodded)
    for a in range(1, 7):  # fuel assemblies in small core
        pow, apow = small_core_power.calc_power_profile(c_fuel_asm, a)

        # Total linear power (W/m) and component power dens (W/m^3)
        # for each component material in the assembly (answer)
        linear_power = small_core_power.calc_total_linear_power(a, svf)

        # check the result
        small_core_power.check_power_profile(pow, linear_power)


def test_new_power_method(testdir, small_reactor):
    """Changed the "calc_power_profile" method from nearly pure
    Python to numpy array-based methods. I need to confirm that
    this still gives me the same answer I was getting before. To
    that end, I've saved the old method here and can compare the
    results.

    """
    p = copy.deepcopy(small_reactor.power['dif3d'])
    for a in small_reactor.assemblies[:2]:
        res = p.calc_power_profile(a, a.id)[0]

        # Now calculate the answer
        ans = calc_power_profile_OLD(p, a, a.id)[0]

        for k in res.keys():
            assert np.allclose(res[k], ans[k])


########################################################################
# Test AssemblyPower object
########################################################################


def test_get_kfint(small_core_asm_power):
    """Test identification of axial mesh cells based on position"""
    # Rodded bounds: 128.1 - 212.33cm (k = 9 - 22)
    # Core bounds: 0.0 - 386.2 cm (k = 0 - 36)
    print(small_core_asm_power.z_finemesh)
    assert small_core_asm_power.get_kfint(0.0) == 0
    assert small_core_asm_power.get_kfint(375.0 - 0.01) == 34
    assert small_core_asm_power.get_kfint(125.0 - 0.01) == 8
    assert small_core_asm_power.get_kfint(125.0) == 8
    assert small_core_asm_power.get_kfint(125.0 + 0.01) == 9
    assert small_core_asm_power.get_kfint(210.0 - 0.01) == 20
    print(small_core_asm_power.z_finemesh[21])
    print(small_core_asm_power.z_finemesh[22])
    assert small_core_asm_power.get_kfint(210.0) == 20
    assert small_core_asm_power.get_kfint(210.0 + 0.01) == 21


def test_transform_z(small_core_asm_power):
    """Test that the absolute Z-coordinates are transformed to
    relative coordinate in the active axial mesh cell"""
    # Rodded bounds: 128.1 - 212.33cm (k = 9 - 22)
    # Core bounds: 0.0 - 386.2 cm (k = 0 - 36)
    # assert np.isclose(small_core_asm_power.transform_z(0.0), -0.5)
    # assert np.isclose(small_core_asm_power.transform_z(386.2), 0.5)
    # assert np.isclose(small_core_asm_power.transform_z(128.10001), -0.5)
    # assert np.isclose(small_core_asm_power.transform_z(212.32999), 0.5)
    # assert np.isclose(small_core_asm_power.transform_z(7.625), 0.0)

    # BECAUSE VARPOW SPITS OUT THE POWER DENSITIES BACKWARDS, NEED
    # TO INVERT THEM AXIALLY
    z = [0.0, 375.0, 125.0000001, 209.9999999, 7.5]
    kf = [small_core_asm_power.get_kfint(zi) for zi in z]
    ans = [0.5, -0.5, 0.5, -0.5, 0.0]
    for i in range(len(ans)):
        res = small_core_asm_power.transform_z(kf[i], z[i])
        msg = ' '.join([str(z[i]), str(kf[i]), str(res), str(ans[i])])
        assert ans[i] == pytest.approx(res), msg


def test_unrodded_region_z(small_core_asm_power):
    """What happens when I ask for pin power in the porous
    media region?"""
    # rodded region for this instance is [128.1, 212.33]
    # Lower porous region
    power = small_core_asm_power.get_power(1.28)
    assert power['refl'] is not None
    for key in ['pins', 'cool', 'duct']:
        assert power[key] is None

    # Rodded region
    power = small_core_asm_power.get_power(1.50)
    assert power['refl'] is None
    for key in ['pins', 'cool', 'duct']:
        assert power[key] is not None

    # Upper porous region
    power = small_core_asm_power.get_power(2.91)
    assert power['refl'] is not None
    for key in ['pins', 'cool', 'duct']:
        assert power[key] is None


def test_total_assembly_power(small_core_power, small_core_asm_power):
    """Test that the AssemblyPower object numerically integrates to
    correct total assembly power based on the power profiles"""
    ans = np.sum(small_core_power.power[1])
    print('ans:', ans)
    print(small_core_asm_power.z_finemesh)
    print(small_core_asm_power.rod_zbnds)

    # Estimate with successive steps
    res = small_core_asm_power.estimate_total_power(zpts=500)
    print('Estimate res:', res)
    err = (res - ans) / ans
    # Should have some error b/c the method uses sloppy midpoint rule
    assert np.abs(err) < 0.005

    # Calculate with actual integrations
    res = small_core_asm_power.calculate_total_power()
    print('Integrated res:', res)
    err = (res - ans) / ans
    assert np.abs(err) < 1e-8


def test_single_asm_refl_total_power(testdir):
    """Test the total power from a single assembly with reflected BC"""
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir,
                           'test_results',
                           'power_verification',
                           'single_asm_refl_total')
    # Set up DASSH objects
    inp = dassh.DASSH_Input(
        os.path.join(inpath, 'input_power_verif_refl.txt'))
    r = dassh.Reactor(inp, path=outpath)
    total = 0.0
    for i in range(len(r.dz)):
        p = r.assemblies[0].power.get_power(r.z[i])
        pp = 0
        for k in p.keys():
            if p[k] is None:
                continue
            else:
                pp += np.sum(p[k])
        total += pp * r.dz[i]

    print('Result:', total / 1e6)
    assert np.abs(total - 6.001e6) / 6.001e6 < 0.002


def test_single_asm_vac_total_power(testdir):
    """Test the total power from a single assembly with vacuum BC"""
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir,
                           'test_results',
                           'power_verification',
                           'single_asm_vac_total')
    # Set up DASSH objects
    inp = dassh.DASSH_Input(
        os.path.join(inpath, 'input_power_verif_vac.txt'))
    r = dassh.Reactor(inp, path=outpath)
    total = 0.0
    for i in range(len(r.dz)):
        p = r.assemblies[0].power.get_power(r.z[i])
        pp = 0
        for k in p.keys():
            if p[k] is None:
                continue
            else:
                pp += np.sum(p[k])
        total += pp * r.dz[i]

    print('Result:', total / 1e6)
    assert np.abs(total - 6.001e6) / 6.001e6 < 0.002


########################################################################
# Test user-specified power
########################################################################


def test_check_for_negative_user_power(caplog):
    """Ensure that DASSH can catch user power profiles that produce
    negative linear power in the range of interest"""
    # This power profile should fail
    pp = np.ones((1, 1, 3))
    pp *= np.array([0.25, -0.01, -1.0])
    with pytest.raises(SystemExit):
        dassh.power._check_for_negative_power(pp, 'dummy', 0)
    assert 'found negative linear power' in caplog.text

    # This power profile should pass
    pp = np.ones((1, 1, 3))
    pp *= np.array([0.25, 0.0, -1.0])
    dassh.power._check_for_negative_power(pp, 'dummy2', 0)
    assert 'dummy2' not in caplog.text


def test_zero_power_with_all_nones():
    """Test that "None" values input as power profiles yield an AssemblyPower
    object that returns all zero power"""
    ap = dassh.AssemblyPower({'pins': None, 'duct': None, 'cool': None},
                             None,
                             np.array([0.0, 400.0]),
                             [0.0, 400.0])
    # Test attributes
    assert ap.pin_power is None
    assert ap.duct_power is None
    assert ap.coolant_power is None
    assert ap.avg_power is None
    # Test linear power values
    res = ap.get_power(2.0)
    assert all([res[k] is None for k in res.keys()])
    # Test total power integration
    assert ap.calculate_total_power() == pytest.approx(0.0)
    # Test power skew
    assert ap.calculate_pin_power_skew() == pytest.approx(1.0)


# @pytest.mark.skip(reason='need to test negative power check first')
def test_instantiation_user_power(testdir, small_reactor):
    """Test that user-supplied power profiles can be used to create
    DASSH AssemblyPower object"""
    # Have path to CSV with power data; load using the method
    asm_power_profiles = dassh.power._from_file(
        os.path.join(testdir, 'test_data', 'seven_asm_power_profiles.csv'))

    # Loop over each entry and demonstrate that with each one you
    # can make an AssemblyPower object
    for ai in range(len(asm_power_profiles)):
        # Make an Instance
        ap = dassh.AssemblyPower(
            {'pins': asm_power_profiles[ai][1].get('pins'),
             'duct': asm_power_profiles[ai][1].get('duct'),
             'cool': asm_power_profiles[ai][1].get('cool')},
            asm_power_profiles[ai][1]['avg_power'],
            asm_power_profiles[ai][1]['zfm'],
            [0.0, 375.0])

        # Confirm axial region boundaries
        k_test = np.random.randint(len(ap.z_finemesh) - 1)
        z_lo = asm_power_profiles[ai][1]['zfm'][k_test]
        z_hi = asm_power_profiles[ai][1]['zfm'][k_test + 1]
        z_test = z_lo + (z_hi - z_lo) * np.random.random()
        k_res = ap.get_kfint(z_test)
        assert k_res == k_test

        # Just make sure it doesn't fail when you choose a random point
        p_test = ap.get_power(np.random.random() * 3.75)
        assert not np.all(p_test['pins'] == 0.0)
        assert not np.all(p_test['duct'] == 0.0)
        assert not np.all(p_test['cool'] == 0.0)

        # Check total power against distributions used to generate the CSV
        ans = small_reactor.assemblies[ai].power.calculate_total_power()
        assert ap.calculate_total_power() == pytest.approx(ans)

        # Check power profiles against distributions used to generate the CSV
        assert np.allclose(ap.pin_power,
                           small_reactor.assemblies[ai].power.pin_power)
        assert np.allclose(ap.duct_power,
                           small_reactor.assemblies[ai].power.duct_power)
        assert np.allclose(ap.coolant_power,
                           small_reactor.assemblies[ai].power.coolant_power)


def test_instantiation_pin_only_user_power(testdir, small_core_asm_power):
    """Test proper instantiation when user provides only pin power
    distribution"""
    # Have path to CSV with power data; load using the method
    asm_power_profiles = dassh.power._from_file(
        os.path.join(testdir, 'test_data', 'single_asm_refl_pin_power.csv'))
    print(np.random.get_state())
    ai = 0  # only one assembly
    # Make an Instance
    ap = dassh.AssemblyPower(
        {'pins': asm_power_profiles[ai][1].get('pins'),
         'duct': asm_power_profiles[ai][1].get('duct'),
         'cool': asm_power_profiles[ai][1].get('cool')},
        asm_power_profiles[ai][1]['avg_power'],
        asm_power_profiles[ai][1]['zfm'],
        [0.0, 375.0])

    # Confirm axial region boundaries
    k_test = np.random.randint(len(ap.z_finemesh) - 1)
    z_lo = asm_power_profiles[ai][1]['zfm'][k_test]
    z_hi = asm_power_profiles[ai][1]['zfm'][k_test + 1]
    z_test = z_lo + (z_hi - z_lo) * np.random.random()
    k_res = ap.get_kfint(z_test)
    assert k_res == k_test

    # Just make sure it doesn't fail when you choose a random point
    p_test = ap.get_power(np.random.random() * 3.75)
    assert not np.all(p_test['pins'] == 0.0)
    assert p_test['duct'] is None
    assert p_test['cool'] is None


def test_user_power_axial_region_error_betw_types(testdir, caplog):
    """Test that DASSH throws proper error when profiles for all pins,
    duct, coolant don't have the same axial region definitions"""
    with pytest.raises(SystemExit):
        dassh.power._from_file(
            os.path.join(
                testdir, 'test_data', 'user_power_ax_reg_test_fail-1.csv'))
    msg = ('Error in user-specified power distribution '
           '(assembly ID: 0); all pins, duct cells, and coolant'
           'items must have identical axial region boundaries.')
    assert msg in caplog.text


def test_user_power_axial_region_error_betw_items(testdir, caplog):
    """Test that DASSH throws proper error when profiles for all elements of
    a specific type (e.g pins) don't have the same axial region definitions"""
    with pytest.raises(SystemExit):
        dassh.power._from_file(
            os.path.join(
                testdir, 'test_data', 'user_power_ax_reg_test_fail-2.csv'))
    msg = ('Error in axial bound entries of user-specified power distribution'
           'for assembly 0 pins; all need to have the same region bounds.')
    assert msg in caplog.text


def test_user_power_axial_region_error_gap(testdir, caplog):
    """Test that DASSH throws proper error when profile axial boundaries
    have any gaps or overlaps between successive region boundaries"""
    with pytest.raises(SystemExit):
        dassh.power._from_file(
            os.path.join(
                testdir, 'test_data', 'user_power_ax_reg_test_fail-3.csv'))
    msg = ('Error in axial bound entries of user-specified power distribution'
           'for assembly 0 duct; no gaps or overlaps allowed between upper/ '
           'lower bounds of successive regions')
    assert msg in caplog.text


########################################################################
# COMPARE DASSH TO OTHER BENCHMARKS (EVALUATEFLUX.X)
########################################################################


def write_evaluate_flux_input(template_path, outpath, asm_obj, z_pts):
    """Create an EvaluateFlux input based"""
    with open(template_path, 'r') as f:
        ef = f.read()

    for pi in range(asm_obj.rodded.n_pin):
        x, y = asm_obj.rodded.pin_lattice.xy[pi]
        x = np.around(x * 100, 6)
        y = np.around(y * 100, 6)
        for zi in z_pts:
            ef += f"ADD_MESHPOINT {x} {y} {zi}\n"

    with open(os.path.join(outpath, 'ef.inp'), 'w') as f:
        f.write(ef)


def run_evaluate_flux(testpath):
    """Run EvaluateFlux"""
    efpath = "/software/ARC/EvaluateFlux.x"

    if not os.path.exists(efpath):
        pytest.skip('Need ARC program EvaluateFlux to run test')

    og_path = os.getcwd()
    os.chdir(testpath)
    with open(os.path.join(testpath, 'ef_stdout.txt'), 'w') as f:
        subprocess.call([efpath, os.path.join(testpath, 'ef.inp')],
                        stdout=f)
    os.chdir(og_path)

    # Remove the output files we don't want
    os.remove(os.path.join(testpath, 'IsotopeMacroRR.out'))
    os.remove(os.path.join(testpath, 'IsotopeMicroRR.out'))
    os.remove(os.path.join(testpath, 'LabeledRegionRR.out'))


def test_EF_single_asm_refl(testdir):
    """Print the power in every pin within a DIF3D axial mesh; compare
    with result from EvaluateFlux.x

    Notes
    -----
    From Mike: "For verification of the pin power, you should simply plot all
    of the pin power shapes coming from EF in a single mesh and
    compare it against one in DASSH."
    - Generate DASSH Reactor with one assembly (3 rings; simple)
    - Get power in each pin at each axial point
    - Make EvaluateFlux input with power at each point; run
    - After scaling EvaluateFlux power density, compare

    """
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir,
                           'test_results',
                           'power_verification_ef',
                           'single_asm_refl')
    # Set up DASSH objects
    inp = dassh.DASSH_Input(
        os.path.join(inpath, 'input_power_verif_refl.txt'))
    r = dassh.Reactor(inp, path=outpath)
    print('DASSH total power:', r.total_power)
    a_hex = np.sqrt(3) * r.asm_pitch**2 / 2 * 100 * 100  # cm2
    a_pin = np.pi * r.assemblies[0].rodded.pin_diameter**2 / 4 * 1e4  # cm2

    # Get pin power profile from DASSH; whole thing is rodded
    # DASSH returns linear power (W/m), but we want to compare power
    # density so need to scale by pin area: W/m -> W/cm3
    # z_dassh = np.around(np.linspace(0.0, r.core_length, 100), 8)
    z_dassh = r.z[1:]
    z_ef = z_dassh * 100
    pp_dassh = np.zeros((len(z_dassh), r.assemblies[0].rodded.n_pin))
    tp_dassh = np.zeros((len(z_dassh), r.assemblies[0].rodded.n_pin))
    for i in range(len(z_dassh)):
        p = r.assemblies[0].power.get_power(z_dassh[i])
        pp_dassh[i] = p['pins'] / 100 / a_pin  # W/m -> W/cm3
        tp_dassh[i] = (pp_dassh[i]
                       + ((np.sum(p['duct']) + np.sum(p['cool']))
                          / 100 / a_pin / r.assemblies[0].rodded.n_pin))

    # Make EvaluateFlux input
    write_evaluate_flux_input(
        os.path.join(inpath, 'ef_single_asm_refl_template.inp'),
        outpath,
        r.assemblies[0],
        z_ef)

    # Run EvaluateFlux
    run_evaluate_flux(outpath)

    # Import the results
    dat = np.loadtxt(os.path.join(outpath, 'FluxAndRegionRR.out'),
                     skiprows=2)
    pp_ef = np.zeros((len(z_ef), r.assemblies[0].rodded.n_pin))
    for pin in range(r.assemblies[0].rodded.n_pin):
        x, y = r.assemblies[0].rodded.pin_lattice.xy[pin] * 100
        for zi in range(len(z_ef)):
            z = z_ef[zi]
            idx = pin * len(z_ef) + zi  # row index
            assert np.isclose(x, dat[idx, 1]), (idx, x, dat[idx, 1])
            assert np.isclose(y, dat[idx, 2]), (idx, y, dat[idx, 2])
            assert np.isclose(z, dat[idx, 3]), (idx, z, dat[idx, 3])
            pp_ef[zi, pin] = dat[idx, 12]

    # Need to scale EF power density; it comes as per cm3 hex but we
    # want it per m3 pin, so we need to multiply by the ratio of pin
    # area to overall hex area.. We also want to renormalize the total
    # power to the expected value, since it comes in as whatever.
    # - Get linear power (W/cm)
    pp_ef *= a_hex / r.assemblies[0].rodded.n_pin
    # - Normalize total power (W) by mult w dz (z[1]) and sum
    print('EF total power:', np.sum(r.dz * 100 * pp_ef.transpose()))
    pp_ef *= r.total_power / np.sum(r.dz * 100 * pp_ef.transpose())
    # - Normalize to pin area
    pp_ef = pp_ef / a_pin  # W/cm -> W/cm3

    # Now time to compare the two power distributions
    # 1. All pins in DASSH result should have equal values
    for pin in range(1, r.assemblies[0].rodded.n_pin):
        assert np.allclose(tp_dassh[:, pin], tp_dassh[:, 0])

    # 2. Pin-total power should be reasonably close to EF prediction
    for pin in range(r.assemblies[0].rodded.n_pin):
        tot_dassh = np.sum(tp_dassh[:, pin])
        tot_ef = np.sum(pp_ef[:, pin])
        e_abs = tot_dassh - tot_ef
        e_rel = e_abs / tot_ef
        if not np.abs(e_rel) < 0.01:
            print(pin, tot_dassh, tot_ef, e_rel)
        assert np.abs(e_rel) < 0.01

    # 3. Distribution should be reasonably close; use vector norms to
    #    get a fair comparison
    for pin in range(r.assemblies[0].rodded.n_pin):
        e_abs = tp_dassh[:, pin] - pp_ef[:, pin]
        e_vec = np.linalg.norm(e_abs) / np.linalg.norm(pp_ef[:, pin])
        if not e_vec < 0.01:
            print(pin, e_vec)
        assert e_vec < 0.01


def test_EF_single_asm_vac(testdir):
    """Print the power in every pin within a DIF3D axial mesh; compare
    with result from EvaluateFlux.x

    Notes
    -----
    Same as previous, but new BC to create radial distribution

    """
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir,
                           'test_results',
                           'power_verification_ef',
                           'single_asm_vac')
    # Set up DASSH objects
    inp = dassh.DASSH_Input(
        os.path.join(inpath, 'input_power_verif_vac.txt'))
    r = dassh.Reactor(inp, path=outpath)
    print('DASSH total power:', r.total_power)
    a_hex = np.sqrt(3) * r.asm_pitch**2 / 2 * 100 * 100  # cm2
    a_pin = np.pi * r.assemblies[0].rodded.pin_diameter**2 / 4 * 1e4  # cm2

    # Get pin power profile from DASSH; whole thing is rodded
    # DASSH returns linear power (W/m), but we want to compare power
    # density so need to scale by pin area: W/m -> W/cm3
    # z_dassh = np.around(np.linspace(0.0, r.core_length, 100), 8)
    z_dassh = r.z[1:]
    z_ef = z_dassh * 100
    pp_dassh = np.zeros((len(z_dassh), r.assemblies[0].rodded.n_pin))
    tp_dassh = np.zeros((len(z_dassh), r.assemblies[0].rodded.n_pin))
    for i in range(len(z_dassh)):
        p = r.assemblies[0].power.get_power(z_dassh[i])
        pp_dassh[i] = p['pins'] / 100 / a_pin  # W/m -> W/cm3
        tp_dassh[i] = (pp_dassh[i]
                       + ((np.sum(p['duct']) + np.sum(p['cool']))
                          / 100 / a_pin / r.assemblies[0].rodded.n_pin))

    # Make EvaluateFlux input if necessary; this will cause the test
    # to fail because EvaluateFlux has to be run externally from pytest
    write_evaluate_flux_input(
        os.path.join(inpath, 'ef_single_asm_vac_template.inp'),
        outpath,
        r.assemblies[0],
        z_ef)

    # Run EvaluateFlux if necessary; for some reason this segfaults
    # if not os.path.exists(os.path.join(testpath, 'FluxAndRegionRR.out')):
    run_evaluate_flux(outpath)

    # Import EvaluateFlux results
    dat = np.loadtxt(os.path.join(outpath, 'FluxAndRegionRR.out'),
                     skiprows=2)
    pp_ef = np.zeros((len(z_ef), r.assemblies[0].rodded.n_pin))
    for pin in range(r.assemblies[0].rodded.n_pin):
        x, y = r.assemblies[0].rodded.pin_lattice.xy[pin] * 100
        for zi in range(len(z_ef)):
            z = z_ef[zi]
            idx = pin * len(z_ef) + zi  # row index
            assert np.isclose(x, dat[idx, 1]), (idx, x, dat[idx, 1])
            assert np.isclose(y, dat[idx, 2]), (idx, y, dat[idx, 2])
            assert np.isclose(z, dat[idx, 3]), (idx, z, dat[idx, 3])
            pp_ef[zi, pin] = dat[idx, 12]

    # Need to scale EF power density; it comes as per cm3 hex but we
    # want it per m3 pin. Therefore, we need to multiply by the ratio
    # of pin area to overall hex area.
    pp_ef *= a_hex / r.assemblies[0].rodded.n_pin
    # - Normalize total power (W) by mult w dz (z[1]) and sum
    print('EF total power:', np.sum(r.dz * 100 * pp_ef.transpose()))
    pp_ef *= r.total_power / np.sum(r.dz * 100 * pp_ef.transpose())
    # - Normalize to pin area
    pp_ef = pp_ef / a_pin  # W/cm -> W/cm3

    # Now time to compare the two power distributions
    # 1. Pin-total power should be reasonably close to EF prediction
    for pin in range(r.assemblies[0].rodded.n_pin):
        tot_dassh = np.sum(tp_dassh[:, pin])
        tot_ef = np.sum(pp_ef[:, pin])
        e_abs = tot_dassh - tot_ef
        e_rel = e_abs / tot_ef
        if not np.abs(e_rel) < 0.01:
            print(pin, tot_dassh, tot_ef, e_rel)
        assert np.abs(e_rel) < 0.01

    # 2. Distribution should be reasonably close; use vector norms to
    #    get a fair comparison
    for pin in range(r.assemblies[0].rodded.n_pin):
        e_abs = tp_dassh[:, pin] - pp_ef[:, pin]
        e_vec = np.linalg.norm(e_abs) / np.linalg.norm(pp_ef[:, pin])
        if not e_vec < 0.01:
            print(pin, e_vec)
        assert e_vec < 0.01


########################################################################
# OLD, PYTHONIC WAY OF POWER DISTRIBUTION (to compare w new numpy)
########################################################################


def calc_power_profile_OLD(p_obj, asm_obj, asm_id):
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
    avg_power = np.zeros(sum(p_obj.k_fints))
    for k in range(sum(p_obj.k_fints)):
        avg_power[k] = (np.sum(p_obj.power_density[asm_id, k])
                        * p_obj.hex_area)

    # If completely unrodded, skip all the shenanigans and just
    # calculate the average power, bc that's all that's used
    if not asm_obj.has_rodded:
        return {}, avg_power

    # SET UP POWER DISTRIBUTION AMONG BUNDLE COMPONENTS
    power = {}  # Power profiles (W/m)
    # Normalization (W) for each component
    computed_power = {}
    # Evaluate XY points to collapse monomials
    eval_xy = p_obj.calc_component_xy(asm_obj.rodded)
    # Volumes of struct components (relative to struct total)
    str_vf = dassh.power.calculate_structure_vfs(asm_obj.rodded)
    # Total linear power (W/m) and component power dens (W/m^3)
    # for each component material in the assembly
    linear_power = p_obj.calc_total_linear_power(asm_id, str_vf)
    component_power = p_obj.calc_component_power_dens(asm_id, str_vf)
    for comp in ['pins', 'duct', 'cool']:
        power[comp] = np.zeros((sum(p_obj.k_fints),
                                len(eval_xy[comp]),
                                np.max(p_obj.mono_exp) + 1))
        computed_power[comp] = np.zeros((len(power[comp]),
                                         len(eval_xy[comp])))

    # LOOP OVER DIF3D REGIONS TO DISTRIBUTE THE POWER
    for k in range(sum(p_obj.k_fints)):
        for comp in ['pins', 'duct', 'cool']:
            for reg in range(len(eval_xy[comp])):
                if 1 == 2:  # dummy pin flag; clad incl. in vf
                    n_power = component_power['dummy'][0, k]
                    g_power = component_power['dummy'][1, k]
                else:
                    n_power = component_power[comp][0, k]
                    g_power = component_power[comp][1, k]
                power[comp][k, reg] = collapse_monomial_OLD(
                    p_obj, n_power, g_power, asm_id, k, eval_xy[comp][reg])
                # Integrate power using shape fxn at xy position
                computed_power[comp][k, reg] = np.dot(
                    power[comp][k, reg], p_obj.z_int)
            # Normalize power to total computed value
            normalizer = np.sum(computed_power[comp][k])
            if normalizer == 0.0:
                power[comp][k] = 0.0
            else:
                power[comp][k] = (power[comp][k]
                                  * (linear_power[comp][0, k]
                                     + linear_power[comp][1, k])
                                  / normalizer)

    # return power, linear_power, computed_power, component_power
    return power, avg_power


def collapse_monomial_OLD(p_obj, n_power, g_power, asm_id, k, e_xy):
    """Collapse the set of monomials into a function of z only

    Parameters
    ----------
    n_power : float
        Neutron power in the requested region (e.g. pins)
    g_power : float
        Gamma power in the requested region (e.g. pins)
    asm_id : int
        Assembly of interest
    k : int
        Fine axial mesh of interest
    e_xy : numpy.ndarray
        The XY components of each term of the monomial set,
        evaluated at the point of interest.

    Returns
    -------
    numpy.ndarray
        Coefficients of the polynomial in z that describes the
        linear power profile of the component of interest in the
        requested axial region

    """
    res = np.zeros(np.max(p_obj.mono_exp) + 1)
    for t in range(p_obj.n_terms):
        z_exp = p_obj.mono_exp[t, 2]
        res[z_exp] += (p_obj.mono_coeffs['n'][asm_id, k, t]
                       * n_power
                       + p_obj.mono_coeffs['g'][asm_id, k, t]
                       * g_power) * e_xy[t]
    return res


########################################################################
# OLD GARBAGE
########################################################################


# @pytest.mark.skip(reason="this test doesn't work yet")
# def test_VARPOW_EF_evaluation_single_asm_refl(testdir):
#     """x"""
#     # Set up z space finemesh from GEODST
#     geodst = py4c.geodst.GEODST(
#         os.path.join(
#             testdir,
#             'test_data',
#             'single_asm_refl',
#             'GEODST'))
#     zbnds = geodst.calc_zs()
#
#     # Get material power density data from other VARPOW output
#     mat_pow_dens = np.loadtxt(
#         os.path.join(testdir,
#                      'test_inputs',
#                      'power_verification',
#                      'single_asm_refl',
#                      'varpow_MatPower.out'),
#         skiprows=2)
#     assert len(mat_pow_dens) == len(zbnds) - 1
#
#     # Get EvaluateFlux output applied to VARPOW; delete zero cols
#     ef_varpow = np.loadtxt(
#         os.path.join(testdir, 'test_inputs', 'power_verification',
#                      'single_asm_refl_ef_on_varpow',
#                      'FluxAndRegionRR.out'), skiprows=2)
#     idx = np.argwhere(np.all(ef_varpow == 0, axis=0))
#     ef_varpow = np.delete(ef_varpow, idx, axis=1)
#
#     # Loop over z points in EF output and scale result by appropriate
#     # material power density value depending on the active mesh
#     result = np.zeros(100)
#     for i in range(len(result) + 1):
#         zi = ef_varpow[i, 3]
#         if zi == 0:
#             k = 0
#         else:
#             k = np.searchsorted(zbnds, zi) - 1
#         pni = np.sum(mat_pow_dens[k, 0:3])  # total neutron power dens
#         pgi = np.sum(mat_pow_dens[k, 3:])  # total gamma power dens
#         result[i] = pni * ef_varpow[i, 4] + pgi * ef_varpow[i, 5]
#         print(zi, result[i])


# @pytest.mark.skip(reason='do not want to plot every time')
# def test_small_core_asm_power_shape(testdir, fuel_asm):
#     res_path = os.path.join(testdir, 'test_data', 'power', 'mono')
#     small_core_power = \
#         dassh.power.Power(os.path.join(res_path, 'MaterialPower.out'),
#                           os.path.join(res_path, 'VariantMonoExponents.out'),
#                           os.path.join(res_path, 'Output.VARPOW'),
#                           os.path.join(res_path, 'GEODST'))
#     test = small_core_power.calc_power_profile(fuel_asm, 1)
#     power, total_power, computed_power, component_power = test
#     asm_power = dassh.power.AssemblyPower(power['pins'],
#                                           power['duct'],
#                                           power['cool'],
#                                           power['lrefl'],
#                                           power['urefl'],
#                                           small_core_power.z_finemesh,
#                                           small_core_power.fk_rods)
#     p = np.zeros(250)
#     z = np.linspace(0, 0.2, 250)
#     for i in range(len(z)):
#         try:
#             tmp = asm_power.get_power(z[i])
#         except:
#             print(z[i])
#             raise
#         p[i] = (np.sum(tmp['pins'])
#                 + np.sum(tmp['cool'])
#                 + np.sum(tmp['duct']))
#
#     plt.plot(z, p, 'bo')
#     plt.show()
#     assert 0
