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
date: 2021-11-02
author: matz
Pytest fixtures and related test utilities for the whole shebang
"""
########################################################################
import os
import copy
import shutil
import sys
import subprocess
import numpy as np
import pytest
import dassh
# import py4c


@pytest.fixture(scope='session')
def testdir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='session')
def resdir(testdir):
    return os.path.join(testdir, 'test_results')


def pytest_addoption(parser):
    help = "Only run verification print tests if requested"
    parser.addoption("--printverify", action="store", default=0, help=help)


@pytest.fixture(scope='session')
def wdir_setup():
    """Return the function that performs teardown of old execution
    test results and setup for new execution tests"""
    def tmp(infile_path, outpath):
        # Remove the DASSH reactor object, if it exists
        if os.path.exists(outpath):
            shutil.rmtree(outpath)
        os.makedirs(outpath, exist_ok=True)
        infile_name = os.path.split(infile_path)[1]
        shutil.copy(infile_path, os.path.join(outpath, infile_name))
        return os.path.join(outpath, infile_name)
    return tmp


# def pytest_configure(config):
#     # register an additional marker
#     config.addinivalue_line(
#         "markers", "env(name): mark test to run only on named environment"
#     )
#
#
# def pytest_runtest_setup(item):
#     envnames = [mark.args[0] for mark in item.iter_markers(name="env")]
#     if item.config.getoption("-V") not in envnames:
#         pytest.skip("test requires env in {!r}".format(envnames))


########################################################################
# PinLattice
########################################################################


@pytest.fixture(scope='module')
def pinlattice_1ring():
    """2-ring pin cell lattice"""
    n_ring = 1
    pitch = 0.0
    d_pin = 1.0
    return dassh.PinLattice(n_ring, pitch, d_pin, test=True)


@pytest.fixture(scope='module')
def pinlattice_2ring():
    """2-ring pin cell lattice"""
    n_ring = 2
    pitch = 1.0
    d_pin = 0.5
    return dassh.PinLattice(n_ring, pitch, d_pin, test=True)


@pytest.fixture(scope='module')
def pinlattice_2ring_map(pinlattice_2ring):
    """2-ring pin cell lattice with map attribute; used to
    test the map_pin_neighbors method of the PinLattice object"""
    pinlattice_2ring.map = np.array([[2, 3, 0],
                                     [7, 1, 4],
                                     [0, 6, 5]])
    return pinlattice_2ring


@pytest.fixture(scope='module')
def pinlattice_2ring_neighbors(pinlattice_2ring_map):
    """2-ring pin cell lattice with pin_adj attribute; used to
    test the map_pin_xy method of the PinLattice object"""
    pinlattice_2ring_map.pin_adj = np.array([[2, 3, 4, 5, 6, 7],
                                             [0, 0, 3, 1, 7, 0],
                                             [0, 0, 0, 4, 1, 2],
                                             [3, 0, 0, 0, 5, 1],
                                             [1, 4, 0, 0, 0, 6],
                                             [7, 1, 5, 0, 0, 0],
                                             [0, 2, 1, 6, 0, 0]])
    return pinlattice_2ring_map


@pytest.fixture(scope='module')
def pinlattice_5ring():
    """5-ring pin cell lattice"""
    n_ring = 5
    pitch = 2.0
    d_pin = 1.5
    return dassh.PinLattice(n_ring, pitch, d_pin, test=True)


@pytest.fixture(scope='module')
def pinlattice_5ring_map(pinlattice_5ring):
    """5-ring pin cell lattice with map attribute; used to
    test the map_pin_neighbors method of the PinLattice object"""
    map = np.array([[38, 39, 40, 41, 42, 0, 0, 0, 0],
                    [61, 20, 21, 22, 23, 43, 0, 0, 0],
                    [60, 37, 8, 9, 10, 24, 44, 0, 0],
                    [59, 36, 19, 2, 3, 11, 25, 45, 0],
                    [58, 35, 18, 7, 1, 4, 12, 26, 46],
                    [0, 57, 34, 17, 6, 5, 13, 27, 47],
                    [0, 0, 56, 33, 16, 15, 14, 28, 48],
                    [0, 0, 0, 55, 32, 31, 30, 29, 49],
                    [0, 0, 0, 0, 54, 53, 52, 51, 50]])
    pinlattice_5ring.map = map
    return pinlattice_5ring


@pytest.fixture(scope='module')
def pinlattice_2ring_full():
    """Full, non-test instance of PinLattice object for testing
    Subchannel object"""
    n_ring = 2
    pitch = 1.0
    d_pin = 0.5
    return dassh.PinLattice(n_ring, pitch, d_pin)


########################################################################
# Subchannel
########################################################################


@pytest.fixture(scope='module')
def sc_1ring():
    """Subchannel setup for 1-ring assembly"""
    n_ring = 1
    pitch = 0.0
    d_pin = 1.0
    pl = dassh.PinLattice(n_ring, pitch, d_pin)
    return dassh.Subchannel(n_ring, pitch, d_pin, pl.map,
                            pl.xy, [(2.0, 3.0)], test=True)


@pytest.fixture(scope='module')
def sc_2ring_args():
    """Subchannel setup arguments for 2-ring assembly"""
    n_ring = 2
    pitch = 1.0
    d_pin = 0.5
    duct_ftf = [(10.0, 12.0)]
    pl = dassh.PinLattice(n_ring, pitch, d_pin)
    return {'pitch': pitch, 'd_pin': d_pin, 'pin_map': pl.map,
            'pin_xy': pl.xy, 'duct_ftf': duct_ftf}


@pytest.fixture(scope='module')
def sc_2ring(sc_2ring_args):
    """Subchannel setup for 2-ring assembly"""
    return dassh.Subchannel(2,
                            sc_2ring_args['pitch'],
                            sc_2ring_args['d_pin'],
                            sc_2ring_args['pin_map'],
                            sc_2ring_args['pin_xy'],
                            sc_2ring_args['duct_ftf'],
                            test=True)


@pytest.fixture(scope='module')
def sc_2ring_type(sc_2ring):
    """Subchannel setup for 2-ring assembly with type attribute; used
    to test assembly of the interior/exterior subchannel maps"""
    sc_2ring.type = np.array([1, 1, 1, 1, 1, 1, 2, 3, 2, 3,
                              2, 3, 2, 3, 2, 3, 2, 3, 4, 5,
                              4, 5, 4, 5, 4, 5, 4, 5, 4, 5])
    return sc_2ring


@pytest.fixture(scope='module')
def sc_2ring_map(sc_2ring_type):
    """Subchannel setup for 2-ring assembly with _map attribute;
    used to test the methods that find subchannel and pin neighbors
    (find_sc_sc_neighbors and find_pin_sc_neighbors)"""
    sc_2ring_type._int_map = np.array([[0, 0, 0, 0],
                                       [0, 0, 0, 0],
                                       [0, 6, 1, 0],
                                       [0, 5, 2, 0],
                                       [0, 4, 3, 0],
                                       [0, 0, 0, 0],
                                       [0, 0, 0, 0]])
    sc_2ring_type._ext_map = np.array([[0, 0, 18, 0],
                                       [16, 17, 7, 0],
                                       [0, 0, 0, 8],
                                       [15, 0, 0, 9],
                                       [14, 0, 0, 0],
                                       [0, 13, 11, 10],
                                       [0, 12, 0, 0]])
    sc_2ring_type._map = np.add(sc_2ring_type._int_map,
                                sc_2ring_type._ext_map)
    return sc_2ring_type


@pytest.fixture(scope='module')
def sc_2ring_pinadj(sc_2ring_type):
    """Subchannel setup for 2-ring assembly with pin_adjacency
    attribute to test methods that find subchannel XY coords"""
    sc_2ring_type.pin_adj = np.array([[1, 2, 3, 4, 5, 6],
                                      [18, 7, 1, 6, 17, 0],
                                      [0, 8, 9, 2, 1, 7],
                                      [9, 0, 10, 11, 3, 2],
                                      [3, 11, 0, 12, 13, 4],
                                      [5, 4, 13, 0, 14, 15],
                                      [17, 6, 5, 15, 0, 16]])
    return sc_2ring_type


@pytest.fixture(scope='module')
def sc_5ring_args():
    """Subchannel setup arguments for 2-ring assembly"""
    n_ring = 5
    pitch = 2.0
    d_pin = 1.5
    duct_ftf = [(10.0, 12.0)]
    pl = dassh.PinLattice(n_ring, pitch, d_pin)
    return {'pitch': pitch, 'd_pin': d_pin, 'pin_map': pl.map,
            'pin_xy': pl.xy, 'duct_ftf': duct_ftf}


@pytest.fixture(scope='module')
def sc_5ring(sc_5ring_args):
    """Subchannel setup for 2-ring assembly"""
    return dassh.Subchannel(5,
                            sc_5ring_args['pitch'],
                            sc_5ring_args['d_pin'],
                            sc_5ring_args['pin_map'],
                            sc_5ring_args['pin_xy'],
                            sc_5ring_args['duct_ftf'],
                            test=True)


@pytest.fixture(scope='module')
def sc_5ring_type(sc_5ring):
    """Subchannel setup for 5-ring assembly with type attribute; used
    to test assembly of the interior/exterior subchannel maps"""
    sc_5ring.type = np.array([1] * sc_5ring.n_sc['coolant']['interior'])
    sc_5ring.type = np.append(sc_5ring.type,
                              np.array([2, 2, 2, 2, 3] * 6))
    sc_5ring.type = np.append(sc_5ring.type,
                              np.array([4, 4, 4, 4, 5] * 6))
    return sc_5ring


@pytest.fixture(scope='module')
def sc_5ring_map(sc_5ring_type):
    """Subchannel setup for 5-ring assembly with _map attribute;
    used to test the methods that find subchannel and pin neighbors
    (find_sc_sc_neighbors and find_pin_sc_neighbors)"""
    sc_5ring_type._int_map = \
        np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 96, 55, 0, 0, 0, 0],
                  [0, 0, 0, 94, 95, 56, 57, 0, 0, 0],
                  [0, 0, 92, 93, 54, 25, 58, 59, 0, 0],
                  [0, 90, 91, 52, 53, 26, 27, 60, 61, 0],
                  [0, 89, 50, 51, 24, 7, 28, 29, 62, 0],
                  [0, 88, 49, 22, 23, 8, 9, 30, 63, 0],
                  [0, 87, 48, 21, 6, 1, 10, 31, 64, 0],
                  [0, 86, 47, 20, 5, 2, 11, 32, 65, 0],
                  [0, 85, 46, 19, 4, 3, 12, 33, 66, 0],
                  [0, 84, 45, 18, 17, 14, 13, 34, 67, 0],
                  [0, 83, 44, 43, 16, 15, 36, 35, 68, 0],
                  [0, 82, 81, 42, 41, 38, 37, 70, 69, 0],
                  [0, 0, 80, 79, 40, 39, 72, 71, 0, 0],
                  [0, 0, 0, 78, 77, 74, 73, 0, 0, 0],
                  [0, 0, 0, 0, 76, 75, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    sc_5ring_type._ext_map = \
        np.array([[0, 0, 0, 0, 0, 126, 0, 0, 0, 0],
                  [0, 0, 0, 0, 125, 97, 0, 0, 0, 0],
                  [0, 0, 0, 124, 0, 0, 98, 0, 0, 0],
                  [0, 0, 123, 0, 0, 0, 0, 99, 0, 0],
                  [121, 122, 0, 0, 0, 0, 0, 0, 100, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 101],
                  [120, 0, 0, 0, 0, 0, 0, 0, 0, 102],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [119, 0, 0, 0, 0, 0, 0, 0, 0, 103],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [118, 0, 0, 0, 0, 0, 0, 0, 0, 104],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [117, 0, 0, 0, 0, 0, 0, 0, 0, 105],
                  [116, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 115, 0, 0, 0, 0, 0, 0, 107, 106],
                  [0, 0, 114, 0, 0, 0, 0, 108, 0, 0],
                  [0, 0, 0, 113, 0, 0, 109, 0, 0, 0],
                  [0, 0, 0, 0, 112, 110, 0, 0, 0, 0],
                  [0, 0, 0, 0, 111, 0, 0, 0, 0, 0]])
    sc_5ring_type._map = np.add(sc_5ring_type._int_map,
                                sc_5ring_type._ext_map)
    return sc_5ring_type


########################################################################
# Materials
########################################################################


@pytest.fixture(scope='module')
def coolant():
    """DASSH Material object for coolant"""
    return dassh.Material('sodium')


@pytest.fixture(scope='module')
def structure():
    """DASSH Material object for structural material"""
    return dassh.Material('ss316')


########################################################################
# Assemblies and subobjects
########################################################################


def activate_rodded_region(region_to_activate, avg_temp, base=True):
    """Create generic DASSH Region object that is used to activate the
    Region object that will be tested"""
    # Note: This isn't a "real" activation, like we do when shifting
    # between regions! We're just reusing the base activation method
    # to set up a rodded region and test its methods
    n_node_duct = region_to_activate.subchannel.n_sc['duct']['total']
    generic = dassh.DASSH_Region(1, np.ones(1), n_node_duct,
                                 np.ones((1, n_node_duct)))
    generic.x_pts = region_to_activate.x_pts
    for key in generic.temp:
        generic.temp[key] *= avg_temp

    tmp = region_to_activate.clone()
    if base:
        tmp._activate_base(generic)
    else:
        t_gap = np.ones(n_node_duct) * avg_temp
        h_gap = np.random.random(n_node_duct) + 5e4
        tmp.activate(generic, t_gap, h_gap, True)
    return tmp


def make_rodded_region_fixture(name, bundle_params, mat_params, fr):
    return dassh.RoddedRegion(name,
                              bundle_params['num_rings'],
                              bundle_params['pin_pitch'],
                              bundle_params['pin_diameter'],
                              bundle_params['wire_pitch'],
                              bundle_params['wire_diameter'],
                              bundle_params['clad_thickness'],
                              bundle_params['duct_ftf'],
                              fr,
                              mat_params['coolant'],
                              mat_params['duct'],
                              bundle_params['htc_params_duct'],
                              bundle_params['corr_friction'],
                              bundle_params['corr_flowsplit'],
                              bundle_params['corr_mixing'],
                              bundle_params['corr_nusselt'],
                              bundle_params['bypass_gap_flow_fraction'],
                              bundle_params['bypass_gap_loss_coeff'],
                              bundle_params['wire_direction'],
                              bundle_params['shape_factor'])


@pytest.fixture(scope='module')
def assembly_default_params():
    """Default DASSH Assembly params"""
    return {'clad_material': None,
            'gap_material': None,
            'corr_mixing': 'CTD',
            'corr_friction': 'CTD',
            'corr_flowsplit': 'CTD',
            'corr_nusselt': 'DB',
            'htc_params_duct': None,
            'bypass_gap_flow_fraction': 0.05,
            'bypass_gap_loss_coeff': None,
            'wire_direction': 'counterclockwise',
            'shape_factor': 1.0}


@pytest.fixture(scope='module')
def unrodded_default_params():
    """Default parameters for each AxialRegion subfield"""
    return {'model': 'simple',
            'structure_material': None,
            'hydraulic_diameter': 0.0,
            'epsilon': 0.0,
            'magic_knob': 0.0,
            'htc_params': None,
            'convection_factor': 1.0}


@pytest.fixture(scope='module')
def textbook_params(assembly_default_params):
    """Parameters for simple hexagonal bundle parameters taken from
    Nuclear Systems II textbook (Todreas); Table 4-3 page 159"""
    mat = {'coolant': dassh.Material('water'),
           'duct': dassh.Material('ss316')}
    input = copy.deepcopy(assembly_default_params)
    input['num_rings'] = 5
    input['pin_pitch'] = 7.938 / 1e3  # mm -> m
    input['pin_diameter'] = 6.350 / 1e3  # mm -> m
    input['clad_thickness'] = 0.5 / 1e3
    input['wire_pitch'] = 304.80 / 1e3  # cm -> m
    input['wire_diameter'] = 1.588 / 1e3  # mm -> m
    ftf = 64.522077241927  # sqrt(3) * ppin * (nr - 1) + dpin + 2 * dwire
    ftf = ftf / 1e3
    # ftf = 0.06453
    # ftf = 0.06452197170774063
    input['duct_ftf'] = [ftf, ftf + 0.001]  # m
    input['AxialRegion'] = {'rods': {'z_lo': 0.0,
                                     'z_hi': 3.86}}

    return input, mat


@pytest.fixture(scope='module')
def textbook_rr(textbook_params):
    """DASSH RoddedRegion object: simple hexagonal bundle parameters
    taken from Nuclear Systems II textbook (Todreas); Table 4-3 page
    159"""
    flowrate = 30.0
    return make_rodded_region_fixture('textbook_rr', textbook_params[0],
                                      textbook_params[1], flowrate)


@pytest.fixture(scope='function')
def textbook_active_rr(textbook_rr, textbook_params):
    """Activate the textbook rodded region"""
    return activate_rodded_region(textbook_rr, 300.15)


@pytest.fixture(scope='module')
def textbook_asm(textbook_params):
    """DASSH Assembly object: simple hexagonal bundle parameters taken
    from Nuclear Systems II textbook (Todreas); Table 4-3 page 159"""
    loc = (1, 1)
    t_in = 300.0
    fr = 30.0
    return dassh.Assembly('textbook_asm', loc, textbook_params[0],
                          textbook_params[1], t_in, fr)


@pytest.fixture(scope='module')
def c_ctrl_params(assembly_default_params, unrodded_default_params):
    """Parameters describing conceptual control assembly"""
    input = copy.deepcopy(assembly_default_params)
    input['num_rings'] = 4
    input['pin_pitch'] = 0.0157491       # m
    input['pin_diameter'] = 0.015475     # m
    input['clad_thickness'] = 0.0013266  # m
    input['wire_pitch'] = 0.20           # m
    input['wire_diameter'] = 0.000274    # m
    input['duct_ftf'] = [0.101, 0.106, 0.111, 0.116]  # m
    input['AxialRegion'] = {'lower': copy.deepcopy(unrodded_default_params),
                            'rods': {'z_lo': 1.25, 'z_hi': 2.75},
                            'upper': copy.deepcopy(unrodded_default_params)}
    input['AxialRegion']['lower']['z_lo'] = 0.0
    input['AxialRegion']['lower']['z_hi'] = 1.25
    input['AxialRegion']['lower']['vf_coolant'] = 0.25
    input['AxialRegion']['upper']['z_lo'] = 2.75
    input['AxialRegion']['upper']['z_hi'] = 3.75
    input['AxialRegion']['upper']['vf_coolant'] = 0.25
    mat = {'coolant': dassh.Material('sodium'),
           'duct': dassh.Material('ht9')}
    return input, mat


@pytest.fixture
def c_ctrl_rr(c_ctrl_params):
    """DASSH RoddedRegion object for conceptual control asm"""
    flowrate = 1.0
    rr = make_rodded_region_fixture('ctrl_rr', c_ctrl_params[0],
                                    c_ctrl_params[1], flowrate)
    return activate_rodded_region(rr, 623.15)


@pytest.fixture
def c_ctrl_asm(c_ctrl_params):
    """DASSH RoddedRegion object for conceptual control asm"""
    loc = (1, 1)
    t_in = 623.15
    fr = 1.0
    return dassh.Assembly('ctrl_asm', loc, c_ctrl_params[0],
                          c_ctrl_params[1], t_in, fr)


@pytest.fixture(scope='module')
def thesis_asm_params(assembly_default_params):
    """Hexagonal bundle described for WWCR code in SK Cheng thesis"""
    input = copy.deepcopy(assembly_default_params)
    input['num_rings'] = 4
    input['pin_diameter'] = 15.040 / 1e3  # mm -> m
    input['wire_diameter'] = 2.260 / 1e3  # mm -> m

    p2d = 1.154
    w2d = 1.164
    h2d = 13.40
    input['pin_pitch'] = p2d * input['pin_diameter']
    input['wire_pitch'] = h2d * input['pin_diameter']
    input['clad_thickness'] = 0.5 / 1e3
    edge_pitch = w2d * input['pin_diameter']

    duct_iftf = (np.sqrt(3)
                 * (input['num_rings'] - 1)
                 * input['pin_pitch']
                 + 2 * (edge_pitch - input['pin_diameter'])
                 + input['pin_diameter'])
    input['duct_ftf'] = [duct_iftf, duct_iftf + 0.001]  # m
    input['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 5.0}}
    mat = {'coolant': dassh.Material('water'),
           'duct': dassh.Material('ss316')}
    return input, mat


@pytest.fixture(scope='module')
def thesis_asm_rr(thesis_asm_params):
    """DASSH RoddedRegion object based on SK Cheng thesis asm params"""
    flowrate = 20.0
    rr = make_rodded_region_fixture('thesis_asm', thesis_asm_params[0],
                                    thesis_asm_params[1], flowrate)
    return activate_rodded_region(rr, 300.15)


@pytest.fixture
def simple_asm(assembly_default_params, small_core_power):
    """Simple DASSH Assembly setup"""
    input = copy.deepcopy(assembly_default_params)
    input['num_rings'] = 2
    input['clad_thickness'] = 0.5 / 1e3
    input['wire_pitch'] = 20.320 / 1e2  # cm -> m
    input['wire_diameter'] = 1.094 / 1e3  # mm -> m
    input['wire_direction'] = 'clockwise'
    input['duct_ftf'] = [0.11154, 0.11757]  # m
    p2d = 1.183
    input['pin_diameter'] = ((input['duct_ftf'][0]
                              - 2 * input['wire_diameter'])
                             / (np.sqrt(3) * p2d + 1))
    input['pin_pitch'] = input['pin_diameter'] * p2d
    input['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 3.862}}
    mat = {'coolant': dassh.Material('sodium'),
           'duct': dassh.Material('ss316')}
    loc = (2, 1)
    inlet_flow_rate = 30.0  # kg /s
    inlet_temp = 273.15 + 350.0  # K
    asm = dassh.Assembly('simple_asm', loc, input, mat,
                         inlet_temp, inlet_flow_rate)
    # asm.rodded = activate_rodded_region(asm.rodded, mat, inlet_temp)
    power, avg_power = small_core_power.calc_power_profile(asm, 1)
    # k = dassh.reactor.match_rodded_finemesh_bnds(small_core_power, input)
    rod_zbnds = dassh.reactor.get_rod_bundle_bnds(
        small_core_power.z_finemesh, input)
    asm.power = dassh.power.AssemblyPower(
        power,
        avg_power,
        small_core_power.z_finemesh,
        rod_zbnds)
    return asm


@pytest.fixture
def simple_asm_pin_only(assembly_default_params, small_core_power_pin_only):
    """Simple DASSH Assembly setup with power delivered to pins only"""
    #  small_core_power_pin_only
    input = copy.deepcopy(assembly_default_params)
    input['num_rings'] = 3
    input['clad_thickness'] = 0.5 / 1e3
    input['wire_pitch'] = 20.320 / 1e2  # cm -> m
    input['wire_diameter'] = 1.094 / 1e3  # mm -> m
    input['duct_ftf'] = [0.11154, 0.11757]  # m
    p2d = 1.183
    input['pin_diameter'] = \
        ((input['duct_ftf'][0] - 2 * input['wire_diameter'])
         / (np.sqrt(3) * (input['num_rings'] - 1) * p2d + 1)) - 0.001
    input['pin_pitch'] = input['pin_diameter'] * p2d
    input['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 3.862}}
    input['wire_direction'] = 'clockwise'
    mat = {'coolant': dassh.Material('sodium_se2anl_425'),
           'duct': dassh.Material('ht9_se2anl_425')}
    loc = (1, 1)
    inlet_flow_rate = 30.0  # kg /s
    inlet_temp = 273.15 + 350.0  # K
    asm = dassh.Assembly('simple_asm_pin_only', loc, input, mat,
                         inlet_temp, inlet_flow_rate)
    # Copy small_core_power_object to eliminate duct and coolant power
    power, avg_power = small_core_power_pin_only.calc_power_profile(asm, 0)
    k = dassh.reactor.match_rodded_finemesh_bnds(
        small_core_power_pin_only, input)
    asm.power = dassh.power.AssemblyPower(
        power, avg_power, small_core_power_pin_only.z_finemesh, k)
    return asm


@pytest.fixture
def simple_ctrl_params(assembly_default_params):
    """Simple double-ducted assembly parameters"""
    p2d = 1.04  # same as c_ctrl asm
    input = copy.deepcopy(assembly_default_params)
    input['num_rings'] = 2
    input['clad_thickness'] = 0.132 * 1.005 / 1e2
    input['wire_pitch'] = 20.0 / 1e2  # cm -> m
    input['wire_diameter'] = 1.094 / 1e3  # mm -> m
    input['duct_ftf'] = [0.09952, 0.10554, 0.11154, 0.11757]  # m
    input['pin_diameter'] = ((input['duct_ftf'][0]
                              - 2 * input['wire_diameter'])
                             / (np.sqrt(3) * p2d + 1))
    input['pin_pitch'] = input['pin_diameter'] * p2d
    input['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 3.862}}
    input['htc_params_duct'] = [0.025, 0.8, 0.4, 7.0]
    mat = {'coolant': dassh.Material('sodium'),
           'duct': dassh.Material('ss316')}

    return input, mat


@pytest.fixture
def simple_ctrl_rr(simple_ctrl_params):
    """DASSH RoddedRegion object: simple hexagonal bundle parameters
    for double-ducted assembly"""
    flowrate = 1.0
    rr = make_rodded_region_fixture('simple_ctrl', simple_ctrl_params[0],
                                    simple_ctrl_params[1], flowrate)
    return activate_rodded_region(rr, 623.15)


@pytest.fixture
def simple_ctrl_asm(simple_ctrl_params, simple_ctrl_rr, small_core_power):
    """DASSH simple double-ducted assembly"""
    # Flow split with bypass: do it by area, even though that's wrong
    # Change that manually here to make the unit test agree with the
    # previous result
    byp_ff = (np.sum(simple_ctrl_rr.bypass_params['total area'])
              / (simple_ctrl_rr.bundle_params['area']
                 + np.sum(simple_ctrl_rr.bypass_params['total area'])))
    simple_ctrl_params[0]['bypass_gap_flow_fraction'] = byp_ff
    inlet_temp = 623.15
    inlet_flow_rate = 0.653
    asm = dassh.Assembly('simple_ctrl_asm',
                         (0, 0),
                         simple_ctrl_params[0],
                         simple_ctrl_params[1],
                         inlet_temp,
                         inlet_flow_rate)
    power, avg_power = small_core_power.calc_power_profile(asm, 0)
    # kbnds = dassh.reactor.match_rodded_finemesh_bnds(
    #     small_core_power, simple_ctrl_params[0])
    rod_zbnds = dassh.reactor.get_rod_bundle_bnds(
        small_core_power.z_finemesh, simple_ctrl_params[0])
    asm.power = dassh.power.AssemblyPower(
        power,
        avg_power,
        small_core_power.z_finemesh,
        rod_zbnds)
    print(asm.rodded.byp_flow_rate)
    return asm


@pytest.fixture
def simple_ctrl_asm_pins_cmat(simple_ctrl_params, simple_ctrl_rr,
                              small_core_power_pin_only):
    """DASSH simple double-ducted assembly with constant material
    properties"""
    # Flow split with bypass: do it by area, even though that's wrong
    # Change that manually here to make the unit test agree with the
    # previous result
    mat = {'coolant': dassh.Material('sodium_se2anl_425'),
           'duct': dassh.Material('ht9_se2anl_425')}
    byp_ff = (np.sum(simple_ctrl_rr.bypass_params['total area'])
              / (simple_ctrl_rr.bundle_params['area']
                 + np.sum(simple_ctrl_rr.bypass_params['total area'])))
    simple_ctrl_params[0]['bypass_gap_flow_fraction'] = byp_ff
    inlet_temp = 623.15
    inlet_flow_rate = 0.653
    asm = dassh.Assembly('simple_ctrl_asm',
                         (0, 0),
                         simple_ctrl_params[0],
                         mat,
                         inlet_temp,
                         inlet_flow_rate)

    # Copy small_core_power_object to eliminate duct and coolant power
    power, avg_power = small_core_power_pin_only.calc_power_profile(asm, 0)
    # k = dassh.reactor.match_rodded_finemesh_bnds(
    #     small_core_power_pin_only, simple_ctrl_params[0])
    rod_zbnds = dassh.reactor.get_rod_bundle_bnds(
        small_core_power_pin_only.z_finemesh,
        simple_ctrl_params[0])
    power_factor = 1 / 6
    asm.power = dassh.power.AssemblyPower(
        power,
        avg_power,
        small_core_power_pin_only.z_finemesh,
        rod_zbnds,
        scale=power_factor)
    return asm


@pytest.fixture(scope='module')
def c_fuel_params(assembly_default_params):
    """Conceptual fuel assembly parameters"""
    input = copy.deepcopy(assembly_default_params)
    input['num_rings'] = 9
    input['pin_pitch'] = 0.0074         # m
    input['pin_diameter'] = 0.00625     # m
    input['clad_thickness'] = 0.0005    # m
    input['wire_pitch'] = 0.20          # m
    input['wire_diameter'] = 0.0011     # m
    input['duct_ftf'] = [0.111, 0.116]  # m
    input['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 3.750}}
    input['htc_params_duct'] = [0.025, 0.8, 0.4, 7.0]
    input['wire_direction'] = 'clockwise'
    mat = {'coolant': dassh.Material('sodium'),
           'duct': dassh.Material('ht9')}
    return input, mat


@pytest.fixture
def c_fuel_rr(c_fuel_params):
    """DASSH RoddedRegion object for conceptual fuel asm"""
    flowrate = 25.0
    rr = dassh.region_rodded.make_rr_asm(
        c_fuel_params[0], 'conceptual_fuel', c_fuel_params[1], flowrate)
    return activate_rodded_region(rr, 623.15)


@pytest.fixture
def c_fuel_asm(c_fuel_params, unrodded_default_params):
    """DASSH RoddedRegion object for conceptual fuel asm"""
    loc = (2, 1)
    inlet_temp = 623.15  # K
    flowrate = 25.0  # kg/s
    inp = copy.deepcopy(c_fuel_params[0])
    inp['AxialRegion'] = {'lower': copy.deepcopy(unrodded_default_params),
                          'rods': {'z_lo': 1.281, 'z_hi': 2.9095},
                          'upper': copy.deepcopy(unrodded_default_params)}
    inp['AxialRegion']['lower']['z_lo'] = 0.0
    inp['AxialRegion']['lower']['z_hi'] = 1.281
    inp['AxialRegion']['lower']['vf_coolant'] = 0.25
    inp['AxialRegion']['upper']['z_lo'] = 2.9095
    inp['AxialRegion']['upper']['z_hi'] = 3.86
    inp['AxialRegion']['upper']['vf_coolant'] = 0.25
    return dassh.Assembly('fuel_asm', loc, inp, c_fuel_params[1],
                          inlet_temp, flowrate)


@pytest.fixture
def c_lrefl_simple(c_fuel_params, c_fuel_rr):
    """SingleNodeHomogeneous unrodded region"""
    flow_rate = 25.0
    z_lo = 0.0
    z_hi = 1.25
    vf_coolant = 0.25
    return dassh.SingleNodeHomogeneous(
        z_lo, z_hi,
        c_fuel_rr.duct_ftf[-1],
        vf_coolant,
        flow_rate,
        c_fuel_params[1]['coolant'],
        c_fuel_params[1]['duct'],
        c_fuel_params[0]['htc_params_duct'])


@pytest.fixture
def shield_ur_mnh():
    """Multinode-homogeneous unrodded region for shield assembly"""
    z_low = 0.0          # bottom axial point (m)
    z_high = 2.0         # top axial point (m)
    dftf = [0.111, 0.116]  # duct flat to flat (m)
    vfc = 0.1939363       # coolant volume fraction
    fr = 0.01      # Coolant FR (kg/s)
    coolant_mat = dassh.Material('sodium_se2anl_425')
    duct_mat = dassh.Material('ht9_se2anl_425')
    mnh = dassh.MultiNodeHomogeneous(z_low, z_high, dftf, vfc, fr,
                                     coolant_mat, duct_mat,
                                     htc_params=None)
    return activate_rodded_region(mnh, 623.15)


@pytest.fixture
def shield_ur_simple():
    """Multinode-homogeneous unrodded region for shield assembly"""
    z_low = 0.0          # bottom axial point (m)
    z_high = 2.0         # top axial point (m)
    dftf = [0.111, 0.116]  # duct flat to flat (m)
    vfc = 0.1939363    # coolant volume fraction
    fr = 0.2      # Coolant FR (kg/s)
    coolant_mat = dassh.Material('sodium_se2anl_425')
    duct_mat = dassh.Material('ht9_se2anl_425')
    ur = dassh.SingleNodeHomogeneous(z_low, z_high, dftf, vfc, fr,
                                     coolant_mat, duct_mat,
                                     htc_params=None)
    for key in ur.temp:
        ur.temp[key] *= 623.15
    return ur


@pytest.fixture
def c_shield_simple_rr(assembly_default_params):
    """Conceptual shield assembly parameters"""
    inp = copy.deepcopy(assembly_default_params)
    dftf = 0.111
    vfc = 0.1939363
    dpin = np.sqrt(4 * vfc * np.sqrt(3) * dftf**2 / 2 / 7 / np.pi)
    pitch = dftf / np.sqrt(3) / 2
    inp['num_rings'] = 2
    inp['pin_pitch'] = pitch
    inp['pin_diameter'] = dpin
    inp['clad_thickness'] = 0.0
    inp['wire_pitch'] = 0.0
    inp['wire_diameter'] = 0.0
    inp['duct_ftf'] = [0.111, 0.116]  # m
    inp['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 3.75}}
    inp['htc_params_duct'] = None
    inp['corr_friction'] = None
    inp['corr_flowsplit'] = None
    inp['corr_mixing'] = None
    mat = {'coolant': dassh.Material('sodium_se2anl_425'),
           'duct': dassh.Material('ht9_se2anl_425')}
    inp['shape_factor'] = 1.0
    rr = make_rodded_region_fixture('shield_rr_7pin', inp, mat, 0.01)
    return activate_rodded_region(rr, 623.15)


@pytest.fixture
def c_shield_rr_params(assembly_default_params):
    """Shield assembly RR parameters"""
    inp = copy.deepcopy(assembly_default_params)
    inp['num_rings'] = 5
    inp['pin_pitch'] = 0.013625       # m
    inp['pin_diameter'] = 0.013462    # m
    inp['clad_thickness'] = 0.0005    # m
    inp['wire_pitch'] = 0.20          # m
    inp['wire_diameter'] = 0.000226   # m
    inp['duct_ftf'] = [0.111, 0.116]  # m
    # inp['corr_flowsplit'] = 'MIT'
    inp['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 3.75}}
    inp['htc_params_duct'] = [0.025, 0.8, 0.8, 7.0]
    mat = {'coolant': dassh.Material('sodium_se2anl_425'),
           'duct': dassh.Material('ht9_se2anl_425')}
    inp['shape_factor'] = 1.0
    return inp, mat


@pytest.fixture
def c_shield_rr(assembly_default_params):
    """Conceptual shield assembly parameters"""
    inp = copy.deepcopy(assembly_default_params)
    inp['num_rings'] = 5
    inp['pin_pitch'] = 0.013625       # m
    inp['pin_diameter'] = 0.013462    # m
    inp['clad_thickness'] = 0.0005    # m
    inp['wire_pitch'] = 0.20          # m
    inp['wire_diameter'] = 0.000226   # m
    inp['duct_ftf'] = [0.111, 0.116]  # m
    inp['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 3.75}}
    inp['htc_params_duct'] = [0.025, 0.8, 0.8, 7.0]
    mat = {'coolant': dassh.Material('sodium_se2anl_425'),
           'duct': dassh.Material('ht9_se2anl_425')}
    inp['shape_factor'] = 100.0
    # inp['shape_factor'] = 1.0
    rr = make_rodded_region_fixture('c_shield_rr', inp, mat, 0.05)
    return activate_rodded_region(rr, 623.15)


@pytest.fixture
def c_shield_asm(assembly_default_params, small_core_power):
    """Conceptual shield assembly"""
    inp = copy.deepcopy(assembly_default_params)
    inp['num_rings'] = 5
    inp['pin_pitch'] = 0.013625       # m
    inp['pin_diameter'] = 0.013462    # m
    inp['clad_thickness'] = 0.0005    # m
    inp['wire_pitch'] = 0.20          # m
    inp['wire_diameter'] = 0.000226   # m
    inp['duct_ftf'] = [0.111, 0.116]  # m
    inp['AxialRegion'] = {'rods': {'z_lo': 0.0, 'z_hi': 3.75}}
    # inp['htc_params_duct'] = [0.025, 0.8, 0.4, 7.0]
    inp['htc_params_duct'] = [0.025, 0.8, 0.8, 7.0]
    inp['use_low_fidelity_model'] = True
    mat = {'coolant': dassh.Material('sodium'),
           'duct': dassh.Material('ht9')}

    asm = dassh.Assembly('fuel_asm', (9, 0), inp, mat, 623.15, 0.0088)
    power, avg_power = small_core_power.calc_power_profile(asm, 1)
    kbnds = [len(small_core_power.z_finemesh),
             len(small_core_power.z_finemesh)]
    apower = dassh.power.AssemblyPower(power, avg_power,
                                       small_core_power.z_finemesh,
                                       kbnds, scale=0.0031)
    asm.power = apower
    return asm


########################################################################
# Power structures
########################################################################


def find_varpow():
    """x"""
    path2varpow = os.path.dirname(os.path.abspath(dassh.__file__))
    if sys.platform == 'darwin':
        path2varpow = os.path.join(path2varpow, 'varpow_osx.x')
    elif 'linux' in sys.platform:
        path2varpow = os.path.join(path2varpow, 'varpow_linux.x')
    else:
        raise SystemError('DASSH currently supports only Linux and OSX')
    return path2varpow


def copy_files(in_path, res_path):
    """Copy CCCC files to dedicated results directory to prevent
    contamination of DASSH test data by running tests"""
    # Remove results path if it exists already
    if os.path.exists(res_path):
        shutil.rmtree(res_path)
    # Make the results directory and copy stuff into it
    os.makedirs(res_path)
    files = [f for f in os.listdir(in_path)
             if os.path.isfile(os.path.join(in_path, f))]
    files = [f for f in files if f[0] != '.']
    for f in files:
        shutil.copy(os.path.join(in_path, f), os.path.join(res_path, f))


def run_varpow(res_path, fuel_id, cool_id):
    """Run VARPOW to generate power object fixtures"""
    # Adapted from the source code written in reactor.py
    cwd = os.getcwd()
    os.chdir(res_path)
    # Run VARPOW, rename output files
    path2varpow = os.path.dirname(os.path.abspath(dassh.__file__))
    if sys.platform == 'darwin':
        path2varpow = os.path.join(path2varpow, 'varpow_osx.x')
    elif 'linux' in sys.platform:
        path2varpow = os.path.join(path2varpow, 'varpow_linux.x')
    else:
        raise SystemError('DASSH currently supports only Linux and OSX')
    with open('varpow_stdout.txt', 'w') as f:
        subprocess.call([path2varpow, str(fuel_id), str(cool_id),
                         'PMATRX', 'GEODST', 'NDXSRF', 'ZNATDN',
                         'NHFLUX', 'GHFLUX'], stdout=f)
    os.chdir(cwd)


@pytest.fixture(scope='session')
def small_core_power(testdir):
    """Generate Power object for 7-assembly core (no normalization)"""
    # scope should mean that this is only run once per testing call
    in_path = os.path.join(testdir, 'test_data', 'seven_asm_vac')
    res_path = os.path.join(testdir, 'test_results', 'seven_asm_vac',
                            'power')
    copy_files(in_path, res_path)
    run_varpow(res_path, 1, 1)
    return dassh.power.Power(
        os.path.join(res_path, 'MaterialPower.out'),
        os.path.join(res_path, 'VariantMonoExponents.out'),
        os.path.join(res_path, 'Output.VARPOW'),
        os.path.join(res_path, 'GEODST'), scalar=0.1)


@pytest.fixture(scope='session')
def small_core_power_pin_only(testdir):
    """Generate Power object for 7-assembly core (no normalization);
    assume VARPOW has been run here already"""
    # scope should mean that this is only run once per testing call
    in_path = os.path.join(testdir, 'test_data', 'single_asm_refl')
    res_path = os.path.join(testdir, 'test_results', 'single_asm_refl')
    copy_files(in_path, res_path)
    run_varpow(res_path, 1, 1)
    return dassh.power.Power(
        os.path.join(res_path, 'MaterialPower.out'),
        os.path.join(res_path, 'VariantMonoExponents.out'),
        os.path.join(res_path, 'Output.VARPOW'),
        os.path.join(res_path, 'GEODST'),
        model='pin_only',
        user_power=6e6)


@pytest.fixture(scope='session')
def small_core_power_normalized(testdir):
    """Generate Power object for 7-assembly core (no normalization);
    assume VARPOW has been run here already"""
    # scope should mean that this is only run once per testing call
    in_path = os.path.join(testdir, 'test_data', 'seven_asm_vac')
    res_path = os.path.join(testdir, 'test_results', 'seven_asm_vac',
                            'power_normalized')
    copy_files(in_path, res_path)
    run_varpow(res_path, 1, 1)
    return dassh.power.Power(
        os.path.join(res_path, 'MaterialPower.out'),
        os.path.join(res_path, 'VariantMonoExponents.out'),
        os.path.join(res_path, 'Output.VARPOW'),
        os.path.join(res_path, 'GEODST'),
        model='pin_only',
        user_power=3e7)


@pytest.fixture(scope='session')
def small_core_power_unscaled(testdir):
    """Generate Power object for 7-assembly core (no normalization);
    assume VARPOW has been run here already"""
    # scope should mean that this is only run once per testing call
    in_path = os.path.join(testdir, 'test_data', 'seven_asm_vac')
    res_path = os.path.join(testdir, 'test_results', 'seven_asm_vac',
                            'power_unscaled')
    copy_files(in_path, res_path)
    run_varpow(res_path, 1, 1)
    return dassh.power.Power(
        os.path.join(res_path, 'MaterialPower.out'),
        os.path.join(res_path, 'VariantMonoExponents.out'),
        os.path.join(res_path, 'Output.VARPOW'),
        os.path.join(res_path, 'GEODST'))


@pytest.fixture
def small_core_asm_power(small_core_power, c_fuel_asm):
    """Generate AssemblyPower object"""
    power, apower = small_core_power.calc_power_profile(c_fuel_asm, 1)
    z_bnds = [128.1, 290.95]
    # kb1 = np.where(np.isclose(small_core_power.z_mesh, 128.10))[0][0]
    # kb2 = np.where(np.isclose(small_core_power.z_mesh, 290.95))[0][0]
    # kb = [np.sum(small_core_power.k_fints[:kb1]),
    #       np.sum(small_core_power.k_fints[:kb2])]
    # print(kb)  # [9, 30]
    return dassh.power.AssemblyPower(
        power, apower, small_core_power.z_finemesh, z_bnds)


@pytest.fixture
def small_core_zero_power(small_core_power):
    """Generate zero-power AssemblyPower object"""
    zero_power = copy.copy(small_core_power)
    zero_power.power = copy.deepcopy(small_core_power.power)
    zero_power.power_density = copy.deepcopy(small_core_power.power_density)
    zero_power.power *= 0
    zero_power.power_density *= 0
    return zero_power


########################################################################
# CORE AND REACTOR OBJECTS
########################################################################


@pytest.fixture(scope='module')
def small_reactor(testdir):
    """DASSH Reactor object for the 7-asm core"""
    inp = dassh.read_input.DASSH_Input(
        os.path.join(testdir,
                     'test_inputs',
                     'input_seven_asm_core.txt'))
    return dassh.reactor.Reactor(
        inp,
        path=os.path.join(
            testdir,
            'test_results',
            'small_reactor'))


@pytest.fixture
def three_asm_core(testdir, coolant, simple_asm):
    """DASSH Core object for 3-asm core with simple, 7-pin asm"""
    asm_list = np.array([0, 1, 2, np.nan, np.nan, np.nan, np.nan])
    core_obj = dassh.Core(asm_list, 0.12, 1.0, coolant,
                          inlet_temperature=623.15, model='flow')
    assemblies = []
    loc = [(0, 0), (1, 0), (1, 1)]
    for ai in range(len(loc)):
        tmp = simple_asm.clone(loc[ai])
        assemblies.append(tmp)

    core_obj.load(assemblies)
    return [assemblies, core_obj]


########################################################################
# FUEL PIN MODEL PARAMETERS AND DATA
########################################################################


@pytest.fixture
def pin(c_fuel_rr):
    """Conceptual fuel pin object"""
    p2d = c_fuel_rr.pin_pitch / c_fuel_rr.pin_diameter
    htcp = [p2d**3.8 * 0.01**0.86 / 3.0, 0.86, 0.86, 4.0 + 0.16 * p2d**5]
    fuel_params = {'r_frac': [0.0, 0.33333333, 0.66666667],
                   'zr_frac': [0.1, 0.1, 0.1],
                   'pu_frac': [0.2, 0.2, 0.2],
                   'porosity': [0.25, 0.25, 0.25],
                   'fcgap_thickness': 0.0,
                   'htc_params_clad': htcp}
    return dassh.FuelPin(0.00628142,
                         0.00050292,
                         dassh.Material('ht9_se2anl'),
                         fuel_params)


@pytest.fixture
def pin_boc():
    """Conceptual fuel pin object w fuel-clad gap, no porosity"""
    d_pin = 6.250 / 1e3
    p_pin = 7.394 / 1e3
    p2d = p_pin / d_pin
    htcp = [p2d**3.8 * 0.01**0.86 / 3.0, 0.86, 0.86, 4.0 + 0.16 * p2d**5]
    clad_thiccness = 0.500 / 1e3
    d_pellet = 4.547 / 1e3
    fc_gap = (d_pin - 2 * clad_thiccness - d_pellet) / 2.0
    fuel_params = {'r_frac': [0.0, 0.33333333, 0.66666667],
                   'zr_frac': [0.1, 0.1, 0.1],
                   'pu_frac': [0.2, 0.2, 0.2],
                   'porosity': [0.0, 0.0, 0.0],
                   'fcgap_thickness': fc_gap,
                   'htc_params_clad': htcp}
    return dassh.FuelPin(d_pin,
                         clad_thiccness,
                         dassh.Material('ht9_se2anl'),
                         fuel_params,
                         gap_mat=dassh.Material('sodium'))


@pytest.fixture
def se2anl_peaktemp_params(c_fuel_rr):
    """Parameters to use in recreating SE2-ANL results at the height
    of peak cladding and fuel temperatures from a single-assembly
    simulation"""
    dz = 0.024443  # m
    Re = 76835.0  # Reynolds number
    clad = dassh.Material('ht9_se2anl')
    cool = dassh.Material('sodium', 425 + 273.15)

    # SE2-ANL single assembly test produced these temperature values
    # at the height of peak cladding / peak fuel temps;
    # out / mw / in; K
    T_cool = np.array([818.427778, 803.09444])
    T_clad = np.array([[827.927778, 843.538889, 859.15],
                       [814.65, 833.761111, 852.816667]])
    T_fuel = np.array([[859.15, 1077.76111],
                       [852.816667, 1114.87222]])
    return {'dz': dz,
            'Re': Re,
            'bundle_de': c_fuel_rr.bundle_params['de'],
            'pin_diameter': c_fuel_rr.pin_diameter,
            'pin_pitch': c_fuel_rr.pin_pitch,
            'clad_thickness': c_fuel_rr.clad_thickness,
            'fcgap_thickness': 0.0,
            'r_fuel': (c_fuel_rr.pin_diameter / 2
                       - c_fuel_rr.clad_thickness),
            'dr_clad': c_fuel_rr.clad_thickness / 2,
            'clad': clad,
            'cool': cool,
            'T_cool': T_cool,
            'T_clad': T_clad,
            'T_fuel': T_fuel}
