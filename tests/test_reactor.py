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
date: 2021-08-21
author: matz
Test the temperature sweep across the core
"""
########################################################################
import numpy as np
import pytest
import os
import sys
import dassh


########################################################################
# INSTANTIATION AND SETUP
########################################################################


def test_instantiation(small_reactor):
    """Check that the reactor attributes make sense"""
    for a in small_reactor.assemblies[1:]:
        # Confirm that axial fine mesh bnds from power are included
        for z in a.power.z_finemesh:
            z_m = z / 100
            assert np.around(z_m, 12) in small_reactor.axial_bnds

        # Confirm that the power in all assemblies is positive
        p = a.power.get_power(0.0)
        print(p)
        p = list(p.values())
        assert all([all(x >= 0) for x in p if x is not None])

        # Confirm that all assemblies have proper starting temperatures
        assert a.avg_coolant_temp == \
            pytest.approx(small_reactor.inlet_temp)
        assert np.allclose(a.avg_duct_mw_temp,
                           small_reactor.inlet_temp)

    # Confirm that interassembly gap has proper starting temperature
    assert np.allclose(small_reactor.core.avg_coolant_gap_temp,
                       small_reactor.inlet_temp)


def test_instantiation_unfilled_asm(testdir):
    """Test DASSH handling of unfilled assembly positions"""
    inp = dassh.DASSH_Input(os.path.join(testdir,
                                         'test_inputs',
                                         'input_31a.txt'))
    r = dassh.Reactor(inp, path=os.path.join(testdir,
                                             'test_results',
                                             'test_31a'))
    # print(r.core.asm_adj.shape)
    # print(r.core.asm_adj)
    # assert 0

    # Confirm only 31 assemblies
    assert len(r.assemblies) == 31

    # Check assembly assignment of flow rate to confirm layout
    fr = [21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23,
          23, 23, 23, 23, 23, 23, 19, 19, 18, 18, 17, 17, 16,
          16, 15, 15, 14, 14]
    for i in range(len(r.assemblies)):
        assert r.assemblies[i].flow_rate == fr[i]

    # Check assembly assignment of power in the outer ring
    p_outer = np.array([36105.23186816, 36105.89365866])
    _idx = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0])
    p_outer = p_outer[_idx]
    for i in range(len(p_outer)):
        assert r.assemblies[i + 19].total_power - p_outer[i] < 1e-6

    # Confirm the assembly adjacency reflects missing corner asm
    assert r.core.asm_adj[7, 1] == 0
    assert r.core.asm_adj[9, 0] == 0
    assert r.core.asm_adj[11, 5] == 0
    assert r.core.asm_adj[13, 4] == 0
    assert r.core.asm_adj[15, 3] == 0
    assert r.core.asm_adj[17, 2] == 0
    assert np.all(r.core.asm_adj[19, [0, 1, 2]] == 0)
    assert np.all(r.core.asm_adj[20, [0, 1, 5]] == 0)
    assert np.all(r.core.asm_adj[21, [0, 1, 5]] == 0)
    assert np.all(r.core.asm_adj[22, [0, 4, 5]] == 0)
    assert np.all(r.core.asm_adj[23, [0, 4, 5]] == 0)
    assert np.all(r.core.asm_adj[24, [3, 4, 5]] == 0)
    assert np.all(r.core.asm_adj[25, [3, 4, 5]] == 0)
    assert np.all(r.core.asm_adj[26, [2, 3, 4]] == 0)
    assert np.all(r.core.asm_adj[27, [2, 3, 4]] == 0)
    assert np.all(r.core.asm_adj[28, [1, 2, 3]] == 0)
    assert np.all(r.core.asm_adj[29, [1, 2, 3]] == 0)
    assert np.all(r.core.asm_adj[30, [0, 1, 2]] == 0)


def test_instantiation_silly_core(testdir):
    """Test an absurd reactor configuration"""
    inp = dassh.DASSH_Input(os.path.join(testdir,
                                         'test_inputs',
                                         'input_16a_silly.txt'))
    r = dassh.Reactor(inp, path=os.path.join(testdir,
                                             'test_results',
                                             'test_16a_silly'))
    r.save()
    r.write_summary()

    # Confirm 16 assemblies
    assert len(r.assemblies) == 16

    # Check power against DIF3D value (given here in DIF3D order)
    p_ans = np.array([
        3.82344000E+06,
        3.75832339E+06,
        3.19309579E+06,
        2.78174268E+06,
        2.39790685E+06,
        1.82341175E+06,
        2.21495401E+06,
        1.92712217E+06,
        2.12534263E+06,
        1.21906392E+06,
        1.26201605E+06,
        7.03685415E+05,
        2.62237707E+05,
        4.40683780E+05,
        1.93116259E+06,
        6.68843005E+05])

    # Normalize
    p_ans *= 3e7 / np.sum(p_ans)
    for i in range(len(r.assemblies)):
        asm = r.assemblies[i]
        p_res = np.array([asm.total_power,
                          asm.power.calculate_total_power()])
        diff = p_res - p_ans[i]
        rel_diff = diff / p_ans[i]
        if np.any(np.abs(rel_diff) > 1e-5):
            print('ASM:', asm.id)
            print('ans: ', '{:.5e}'.format(p_ans[i]))
            print('res: ', '{:.5e}'.format(p_res))
            print('dif: ', '{:.5e}'.format(diff))

        assert np.all(np.abs(rel_diff) < 1e-5)


def test_setup_requested_ax(testdir, caplog):
    """Test that Reactor object ends up with requested ax planes"""
    inp = dassh.DASSH_Input(os.path.join(testdir,
                                         'test_inputs',
                                         'input_single_asm.txt'))
    assert 'ignoring' in caplog.text
    r = dassh.Reactor(inp,
                      path=os.path.join(
                          testdir,
                          'test_results',
                          'test_reactor_setup_requested_ax'))
    ans = [0.3999992, 0.750011]
    for zi in ans:
        delta = np.abs(r.z - zi)
        assert pytest.approx(np.min(delta), abs=1e-6) == 0.0


def test_setup_with_unrodded_asm(testdir):
    """Test that the Reactor object and power setup are achieved"""
    inpath = os.path.join(testdir, 'test_inputs', 'input_w_unrodded.txt')
    inp = dassh.DASSH_Input(inpath)
    outpath = os.path.join(testdir, 'test_results', 'input_w_unrodded')
    r = dassh.Reactor(inp, path=outpath)
    assert not r.assemblies[0].has_rodded
    assert r.assemblies[0].power.pin_power is None
    assert r.assemblies[0].power.duct_power is None
    assert r.assemblies[0].power.coolant_power is None


def test_asm_ordering(testdir):
    """Test that assemblies are properly ordered"""
    ans = ['fuel' for i in range(7)]
    ans[0] = 'control'
    ans[2] = 'control'

    inputfile = os.path.join(testdir,
                             'test_inputs',
                             'input_assignment_check.txt')
    outpath = os.path.join(testdir, 'test_results', 'indexing', 'dif3d')
    inp = dassh.DASSH_Input(inputfile)
    r = dassh.Reactor(inp, path=outpath)
    names = [a.name for a in r.assemblies]
    print(names)
    print([a.id for a in r.assemblies])
    print([a.loc for a in r.assemblies])
    assert names == ans


def test_check_dz(small_reactor):
    """Check dz adjustment when approaching power dist bounds in the
    march along the core"""
    bnds = small_reactor.axial_bnds
    length = small_reactor.core_length
    print(bnds)
    z = 0.0
    bounds_caught = []
    while z < length:
        dz_prime = small_reactor._check_dz(z)
        z += dz_prime
        if np.any(np.around(z, 12) == bnds):
            bounds_caught.append(np.around(z, 12))
            # print(z, dz_prime)
    print(bounds_caught)
    bounds_caught = np.array(bounds_caught)
    assert np.array_equal(bounds_caught, bnds[1:])


def test_setup_zpts(small_reactor):
    """Test that all boundaries are captured in axial mesh points"""
    zpts = small_reactor._setup_zpts()[0]
    x = [bnd in zpts for bnd in small_reactor.axial_bnds]
    print(x)
    assert all([bnd in zpts for bnd in small_reactor.axial_bnds])
    assert len(zpts) >= (small_reactor.core_length
                         / small_reactor.req_dz)


def test_zpts_axial_boundaries(small_reactor):
    """Test that the axial mesh points I've generated are correctly
    interpreted by the DASSH Power object"""
    print(small_reactor.axial_bnds)
    p = small_reactor.assemblies[0].power
    zpts = small_reactor._setup_zpts()[0]

    # First point: k_fint = 0
    assert p.get_kfint(zpts[0]) == 0

    # Last point: kf_int = len(n_fint) - 2
    # the boundaries includes the edges (-1), and when the z point
    # equals the boundary the region is defined as the previous (-1)
    # e.g. bnds    = [0 1 2 3 4 5]; len = 6
    #      regions =   0 1 2 3 4 <- last region is 4 == len - 2
    assert p.get_kfint(zpts[-1] * 100.0) == \
        len(small_reactor.axial_bnds) - 2

    # All other points
    for i in range(1, len(zpts) - 1):  # skip first, last points
        if zpts[i] in small_reactor.axial_bnds:
            kfint_i = p.get_kfint(zpts[i] * 100.0)
            kfint_ip1 = p.get_kfint(zpts[i + 1] * 100.0)
            if kfint_i == kfint_ip1:  # this means the test failed
                kfint_im1 = p.get_kfint(zpts[i - 1] * 100.0)
                print(zpts[i - 1], kfint_im1)
                print(zpts[i], kfint_i)
                print(zpts[i + 1], kfint_ip1)
            assert kfint_i != kfint_ip1


########################################################################
# DATA IO
########################################################################


def test_make_tables(small_reactor):
    """If it doesn't fail, I guess it made the tables"""
    n_asm = len(small_reactor.asm_templates)
    summary = dassh.table.GeometrySummaryTable(n_asm)
    summary.make(small_reactor)

    power = dassh.table.PositionAssignmentTable()
    power.make(small_reactor)

    flow = dassh.table.CoolantFlowTable()
    flow.make(small_reactor)

    print(summary)
    print(power)
    print(flow)


def test_write_tables(small_reactor):
    """Check that I can write a file with tables"""

    # Need to pre-load some pin data for it to find
    peak_temp_dat = {'cool': (808.7726474073048, 2.10),
                     'duct': (779.2171478384015, 2.20)}
    peak_pin_dat = {
        'clad_od': [816.3348397888126, 5, [1.0, 1.0, 1.4, 5.0,
                                           805.8597966218155,
                                           816.3348397888126,
                                           833.7015229405092,
                                           851.0682060922057,
                                           851.0682060922057,
                                           1088.6630086661442]],
        'clad_mw': [833.7015229405092, 6, [1.0, 1.0, 1.4, 5.0,
                                           805.8597966218155,
                                           816.3348397888126,
                                           833.7015229405092,
                                           851.0682060922057,
                                           851.0682060922057,
                                           1088.6630086661442]],
        'clad_id': [851.0682060922057, 7, [1.0, 1.0, 1.4, 5.0,
                                           805.8597966218155,
                                           816.3348397888126,
                                           833.7015229405092,
                                           851.0682060922057,
                                           851.0682060922057,
                                           1088.6630086661442]],
        'fuel_od': [851.0682060922057, 8, [1.0, 1.0, 1.4, 5.0,
                                           805.8597966218155,
                                           816.3348397888126,
                                           833.7015229405092,
                                           851.0682060922057,
                                           851.0682060922057,
                                           1088.6630086661442]],
        'fuel_cl': [1089.4486858368866, 9, [1.0, 1.0, 1.4, 16.0,
                                            805.5514188743829,
                                            816.0700357501505,
                                            833.5089602073863,
                                            850.9478846646219,
                                            850.9478846646219,
                                            1089.4486858368866]]}
    for i in range(len(small_reactor.assemblies)):
        small_reactor.assemblies[i]._peak['cool'] = \
            peak_temp_dat['cool']
        small_reactor.assemblies[i]._peak['duct'] = \
            peak_temp_dat['duct']
        if 'pin' in small_reactor.assemblies[i]._peak.keys():
            small_reactor.assemblies[i]._peak['pin'] = \
                peak_pin_dat
    small_reactor.write_summary()
    small_reactor.write_output_summary()
    path_to_output = os.path.join(small_reactor.path, 'dassh.out')
    with open(path_to_output, 'r') as f:
        out = f.read()

    # Teardown before failure, just in case
    os.remove(path_to_output)

    tables = ["ASSEMBLY GEOMETRY SUMMARY",
              "ASSEMBLY POWER AND ASSIGNED FLOW RATE",
              "SUBCHANNEL FLOW CHARACTERISTICS",
              "PRESSURE DROP (MPa) ACROSS ASSEMBLIES",
              "COOLANT TEMPERATURE SUMMARY",
              "DUCT TEMPERATURE SUMMARY",
              "PEAK CLAD MW TEMPERATURES",
              "PEAK FUEL CL TEMPERATURES"
              ]

    for t in tables:
        tag = out.find(t)
        assert tag != -1, f'Table {t} not found!'


def test_save_load(small_reactor):
    """Test that I can save and load DASSH Reactor objects"""
    if 'linux' not in sys.platform:
        pytest.skip('skipping Reactor r/w test; run only on Linux')

    if sys.version_info <= (3, 7):  # Requires Python 3.7 or greater
        pytest.skip('skipping Reactor r/w test; need >= Python 3.7')

    small_reactor.save()
    dassh.reactor.load(os.path.join(small_reactor.path,
                                    'dassh_reactor.pkl'))


########################################################################
# TEMPERATURE SWEEP
########################################################################


def cleanup(output_path):
    """Clean up test data directory when running a new test"""
    # Remove old datafiles
    if os.path.exists(output_path):
        for f in os.listdir(output_path):
            if f[:4] == 'temp' and f[-4:] == '.csv':
                os.remove(os.path.join(output_path, f))


def test_single_asm(testdir):
    """Perform the temperature sweep for a single assembly; this test
    just makes sure nothing fails"""
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir, 'test_results', 'test_single_asm')
    cleanup(outpath)
    inp = dassh.DASSH_Input(os.path.join(inpath, 'input_single_asm.txt'))
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    r.temperature_sweep()
    r.save()
    assert 'dassh.out' in os.listdir(outpath)
    assert np.abs(r.assemblies[0].avg_coolant_temp - 273.15 - 500) < 1.0


def test_single_asm_parameter_update_freq(testdir):
    """Perform the temperature sweep for a single assembly and confirm that
    the result is close to what's obtained from the previous test."""
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir, 'test_results',
                           'test_single_asm_param_freq')
    cleanup(outpath)
    inp = dassh.DASSH_Input(os.path.join(inpath, 'input_single_asm.txt'))
    # Add in option for param update frequency / cutoff - correlated parameters
    # are only updated when there's a 1% change in a material property value.
    # Problem uses "sodium_se2anl" coolant, so the material properties are in
    # fact dependent on temperature.
    inp.data['Setup']['param_update_tol'] = 0.01
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    r.temperature_sweep()
    r.save()
    assert 'dassh.out' in os.listdir(outpath)  # Did it run all the way?
    # Now check answer against previous
    anspath = os.path.join(testdir, 'test_results', 'test_single_asm')
    r_ans = dassh.reactor.load(os.path.join(anspath, 'dassh_reactor.pkl'))
    dT = (r_ans.assemblies[0].rodded.temp['coolant_int'] -
          r.assemblies[0].rodded.temp['coolant_int'])
    # Average temperature change is 150˚C. How much does this update affect
    # the answer compared to the case in which the correlated parameters are
    # updated with every step?
    max_dT = np.max(np.abs(dT))
    assert max_dT > 0.0           # Confirm there's some difference
    assert max_dT < 0.005         # degrees celsius
    assert max_dT / 150.0 < 2e-5  # relative difference


def test_multiregion_ebal(testdir):
    """Perform the temperature sweep for a single assembly, with power
    model pin_only - check ebal across multiple regions"""
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir, 'test_results',
                           'test_two_asm_pinonly')
    cleanup(outpath)
    inp = dassh.DASSH_Input(
        os.path.join(
            inpath, 'input_two_asm_pinonly.txt')
    )
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    r.temperature_sweep()
    r.save()
    assert 'dassh.out' in os.listdir(outpath)
    # Check output file for energy balance
    with open(os.path.join(outpath, 'dassh.out'), 'r') as f:
        outfile = f.read()
    tag = outfile.find('OVERALL ASSEMBLY ENERGY BALANCE')
    tag = outfile.find('CORE', tag)
    tag2 = outfile.find('\n', tag)
    line = outfile[tag:tag2]
    line = line.split(' ')
    line = [l for l in line if l != '']
    ebal = float(line[-1])
    assert ebal < 1e-9


def test_3asm_sweep(testdir):
    """Test that DASSH can sweep with unfilled positions"""
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir, 'test_results', 'test_3a')
    cleanup(outpath)
    inp = dassh.DASSH_Input(os.path.join(inpath, 'input_3a.txt'))
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    r.save()
    r.temperature_sweep()
    assert 'dassh.out' in os.listdir(outpath)
    r.save()

    # Check that Q=mCdT is close (with these high flow rates,
    # it should be). Assembly 1 should recieve heat from the others
    # so dT_result should be greater than dT_expected. The opposite
    # is true for the others
    sign = [1, -1, -1]
    for i in range(len(r.assemblies)):
        dT = r.assemblies[i].avg_coolant_temp - r.inlet_temp
        p = np.sum([r.assemblies[i]._power_delivered[k]
                    for k in r.assemblies[i]._power_delivered.keys()])
        Cp = 1274.2
        m = r.assemblies[i].flow_rate
        diff = dT - p / Cp / m
        print(i, diff)
        # The difference with adiabatic isn't too imporant - just want
        # to make sure it's in the ballpark
        assert np.abs(diff) < 2.5
        assert np.sign(diff) == sign[i]


def test_2asm_ebal(testdir):
    """Test heat transfer between two assemblies of different types; no
    power, one has elevated temperature

    """
    inpath = os.path.join(
        testdir,
        'test_inputs',
        'input_2a_flow_diff_asm-2.txt'
    )
    outpath = os.path.join(
        testdir,
        'test_results',
        'test_2asm_ht_flow_diff_asm'
    )
    inp = dassh.DASSH_Input(os.path.join(inpath))
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    # Zero power
    for asm in r.assemblies:
        asm.power.avg_power *= 0.0
        if asm.has_rodded:
            asm.power.pin_power *= 0.0
            asm.power.coolant_power *= 0.0
            asm.power.duct_power *= 0.0
    # Elevate assembly 1 temperature
    r.assemblies[0].rodded.temp['coolant_int'] *= 0.0
    r.assemblies[0].rodded.temp['coolant_int'] += 700 + 273.15
    # Perform temperature sweep
    r.save()
    r.write_summary()
    r.temperature_sweep()
    r.save()

    # Check that energy was conserved
    dt = np.zeros(3)
    q_in = np.zeros(3)
    q_duct = np.zeros(3)
    mfr = np.zeros(3)
    for i in range(len(r.assemblies)):
        dt[i] = r.assemblies[i].avg_coolant_int_temp - r.inlet_temp
        q_in[i] = sum(r.assemblies[i]._power_delivered.values())
        mfr[i] = r.assemblies[i].flow_rate
        q_duct[i] = np.sum(r.assemblies[i].active_region.ebal['duct'])

    dt[0] = r.assemblies[0].avg_coolant_int_temp - (700 + 273.15)
    dt[-1] = r.core.avg_coolant_gap_temp - r.inlet_temp
    mfr[-1] = r.core.gap_flow_rate
    q_duct[-1] = np.sum(r.core.ebal['asm'])
    cp = r.assemblies[0].active_region.coolant.heat_capacity  # constant
    q_dt = mfr * cp * dt
    assert np.abs(np.sum(q_dt)) < 5e-9


def test_silly_core_sweep(testdir):
    """Test an absurd core layout to confirm sweep"""
    inp = dassh.DASSH_Input(os.path.join(testdir,
                                         'test_inputs',
                                         'input_16a_silly.txt'))
    r = dassh.Reactor(inp, path=os.path.join(testdir,
                                             'test_results',
                                             'test_16a_silly'))
    r.write_summary()
    r.temperature_sweep()
    r.save()
    r.write_output_summary()
    r.save()


# Also: this is a stupid test, it doesn't test anything, and should be replaced
@pytest.mark.skip(reason='too long, needs to be revised')
def test_sweep_with_all_but_one_unrodded_asm(testdir):
    """Test that the Reactor object and power setup are achieved; this
    test just makes sure nothing fails"""
    # if 'linux' not in sys.platform:
    #     pytest.skip('skipping sweep test, too slow locally')

    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir,
                           'test_results',
                           'test_sweep_with_all_but_one_unrodded_asm')
    cleanup(outpath)

    fname = 'input_all_but_one_unrodded.txt'
    inp = dassh.DASSH_Input(os.path.join(inpath, fname))
    r = dassh.Reactor(inp, path=outpath, write_output=True,
                      calc_energy_balance=True)
    print(r.req_dz)
    # assert 0
    r.temperature_sweep()
    r.save()
    assert 'dassh.out' in os.listdir(outpath)
    # Check that average coolant temperatures in unrodded assemblies
    # are about equal to what we asked them to be
    # Expected dT = 25 degrees; some will be lost to boundary b/c BC
    # is not adiabatic
    print(r.inlet_temp)
    avg_cool_temps = np.zeros(len(r.assemblies))
    for i in range(len(r.assemblies)):
        avg_cool_temps[i] = r.assemblies[i].avg_coolant_temp
        print(i,
              r.assemblies[i].total_power,
              r.assemblies[i].flow_rate,
              r.assemblies[i].avg_coolant_temp)

    dt = avg_cool_temps - r.inlet_temp
    diff = np.abs(dt - 10.0)  # expected dT is 25 K
    print(diff)
    assert np.all(diff < 2.0)


def test_adiabatic_unrodded_reactor_sweep(testdir):
    """Make sure it equals Q=mCdT exactly"""
    # inpath = os.path.join(testdir,
    #                       'test_inputs',
    #                       'input_all_unrodded.txt')
    inpath = os.path.join(testdir,
                          'test_inputs',
                          'input_all_unrodded.txt')
    outpath = os.path.join(testdir,
                           'test_results',
                           'test_adiabatic_unrodded_reactor_sweep')
    inp = dassh.DASSH_Input(inpath)
    # Change core model to adiabatic
    inp.data['Core']['gap_model'] = None
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    print(r.total_power / 1e6)

    # Calculate average heat capacity
    cp = r.assemblies[0].active_region.coolant.heat_capacity
    print('const cp', cp)

    # Calculate power delivered to the assemblies
    q = np.zeros(len(r.assemblies))
    for i in range(len(r.assemblies)):
        # Note: have to scale here because "scaling factor" not applied
        # until the AssemblyPower object is created;
        id = r.assemblies[i].id
        q[i] = np.sum(r.power['dif3d'].power[id])

    mfr = np.array([a.flow_rate for a in r.assemblies])
    dt_ans = q / mfr / cp  # should equal 150.0
    print(mfr)
    print(q / 1e6)
    # assert 0
    # Run the sweep
    r.temperature_sweep()
    dt_res = np.zeros(len(r.assemblies))
    for i in range(len(r.assemblies)):
        dt_res[i] = r.assemblies[i].avg_coolant_temp - 273.15 - 350.0

    # Check the result
    print('dt result', dt_res)
    print('dt expected', dt_ans)
    abs_err = dt_res - dt_ans
    rel_err = abs_err / dt_ans
    print('rel. err.', rel_err)
    assert np.all(rel_err < 1e-8)


def test_double_duct_ebal(testdir):
    """Test energy balance tracking on double ducted assembly"""
    # if 'linux' not in sys.platform:
    #     pytest.skip('skipping sweep test, too slow locally')

    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir, 'test_results', 'test_dd_ebal')
    # Remove old datafiles
    if os.path.exists(outpath):
        for f in os.listdir(outpath):
            if f[:4] == 'temp' and f[-4:] == '.csv':
                os.remove(os.path.join(outpath, f))

    inp = dassh.DASSH_Input(os.path.join(inpath, 'input_dd_ebal.txt'))
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    r.temperature_sweep()
    r.save()
    assert 'dassh.out' in os.listdir(outpath)
    # Confirm that dd ebal sums to zero
    rr = r.assemblies[0].region[0]
    total = (rr.ebal['power']
             + np.sum(rr.ebal['duct'])
             + np.sum(rr.ebal['duct_byp_in'])
             + np.sum(rr.ebal['duct_byp_out']))
    temprise = (rr.total_flow_rate
                * rr.coolant.heat_capacity
                * (rr.avg_coolant_temp - r.inlet_temp))
    bal = total - temprise
    assert np.abs(bal) < 1e-7
    assert np.abs(bal) / rr.ebal['power'] < 1e-12


@pytest.mark.skip(reason='milos still working on this')
def test_stagnant_double_duct_ebal(testdir):
    """Test energy balance tracking on double ducted assembly"""
    # if 'linux' not in sys.platform:
    #     pytest.skip('skipping sweep test, too slow locally')

    name = 'dd_stagnant_byp'
    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir, 'test_results', f'test_{name}')
    # Remove old datafiles
    if os.path.exists(outpath):
        for f in os.listdir(outpath):
            if f[:4] == 'temp' and f[-4:] == '.csv':
                os.remove(os.path.join(outpath, f))

    inp = dassh.DASSH_Input(os.path.join(inpath, f'input_{name}.txt'))
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    r.temperature_sweep()
    # r.write_output_summary()
    r.save()
    assert 'dassh.out' in os.listdir(outpath)
    # Confirm that dd ebal sums to zero
    rr = r.assemblies[0].region[0]
    total = (rr.ebal['power']
             + np.sum(rr.ebal['duct'])
             + np.sum(rr.ebal['duct_byp_in'])
             + np.sum(rr.ebal['duct_byp_out']))
    temprise = (rr.total_flow_rate
                * rr.coolant.heat_capacity
                * (rr.avg_coolant_temp - r.inlet_temp))
    bal = total - temprise
    assert np.abs(bal) < 1e-7
    assert np.abs(bal) / rr.ebal['power'] < 1e-12


@pytest.mark.skip(reason='the interasm key no longer exists, need new test')
def test_interasm_ebal(testdir):
    """Test that the interassembly energy balance works"""
    # if 'linux' not in sys.platform:
    #     pytest.skip('skipping sweep test, too slow locally')

    inpath = os.path.join(testdir, 'test_inputs')
    outpath = os.path.join(testdir, 'test_results',
                           'test_interasm_ebal')
    # Remove old datafiles
    if os.path.exists(outpath):
        for f in os.listdir(outpath):
            if f[:4] == 'temp' and f[-4:] == '.csv':
                os.remove(os.path.join(outpath, f))

    fname = 'input_interasm_ebal.txt'
    inp = dassh.DASSH_Input(os.path.join(inpath, fname))
    r = dassh.Reactor(inp, path=outpath, write_output=True)
    # print(r.assemblies[1].region[0]._params)
    # print(dassh.region_unrodded._get_rr_kwargs(
    #     inp.data['Assembly']['fuel'],
    #     {'coolant': dassh.Material('sodium'),
    #      'duct': dassh.Material('ht9')},
    #     r.assemblies[1].flow_rate,
    #     623.15))
    # assert 0
    r._options['ebal'] = True
    r.temperature_sweep()
    r.save()
    assert 'dassh.out' in os.listdir(outpath)

    # Check that the interassembly energy balance sums to 0.0
    e_sum = np.sum(r.core._ebal['interasm'])
    assert np.abs(e_sum) < 1e-9

    # Check the agreement of sides for the surrounding assemblies by
    # checking which side should have maximum power transfer (inward-
    # facing) and which should have none (outward-facing)
    max_idx = [4, 5, 0, 1, 2, 3]
    zero_idx = [1, 2, 3, 4, 5, 0]
    for i in range(1, 7):
        tmp = r.core._ebal['interasm'][i].reshape(6, -1).copy()
        eps = np.zeros((6, r.core._sc_per_side + 2))
        eps[:, 1:] = tmp
        eps[:, 0] = np.roll(tmp[:, -1], 1)
        eps[:, (0, -1)] *= 0.5
        eps = np.sum(eps, axis=1)
        assert np.argmax(eps) == max_idx[i - 1]
        assert eps[zero_idx[i - 1]] == 0.0
