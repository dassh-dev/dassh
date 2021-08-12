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
date: 2021-04-27
author: matz
Test the mapping of assemblies and inter-assembly gap coolant
"""
########################################################################
import copy
import numpy as np
import pytest
import dassh
from dassh import core
# np.set_printoptions(threshold=sys.maxsize)
# np.set_printoptions(linewidth=500)


def build_asm_list(n_ring, empty_positions=()):
    """Build assembly list for use in core objects"""
    asm_list = []
    pos_idx = 0
    for r in range(n_ring):
        if r == 0:
            pos_in_ring = 1
        else:
            pos_in_ring = 6 * r
        for a in range(pos_in_ring):
            if pos_idx not in empty_positions:
                asm_list.append(pos_idx)
            else:
                asm_list.append(np.nan)
            pos_idx += 1
    n_positions = 3 * (n_ring - 1)**2 + 3 * (n_ring - 1) + 1
    assert len(asm_list) == n_positions
    return np.array(asm_list)


########################################################################
# TEST FIXTURES
########################################################################


@pytest.fixture
def core_19a(coolant):
    """DASSH Core object with 19 assemblies"""
    asm_list = build_asm_list(3)
    asm_pitch = 12.0
    gap_fr = 0.05
    return dassh.core.Core(asm_list, asm_pitch, gap_fr, coolant)


@pytest.fixture
def core_19a_test(coolant):
    """DASSH Core object with 19 assemblies, with test flag to try
    creating complex attributes"""
    asm_list = build_asm_list(3)
    asm_pitch = 12.0
    gap_fr = 0.05
    return dassh.core.Core(asm_list, asm_pitch, gap_fr, coolant, test=True)


@pytest.fixture
def core_19a_incremental(core_19a_test):
    """Simple full-core region assignment array with all attributes;
    calling in this way ensures that all __init__ methods will work
    and that if they don't we get a more meaningful output"""
    asm_list = build_asm_list(3)
    try:
        core_19a_test.asm_map = core.map_asm(asm_list)
    except:
        print("Failure in map_asm() method")
        raise
    try:
        core_19a_test.asm_adj = \
            core.map_adjacent_assemblies(core_19a_test.asm_map)
    except:
        print("Failure in map_adjacent_assemblies() method")
        raise
    return core_19a_test


@pytest.fixture
def small_core_no_power(testdir, coolant, c_fuel_asm, c_ctrl_asm):
    """DASSH Core object for a 7-assembly core"""
    asm_list = build_asm_list(2)
    asm_pitch = 0.12
    fr = 0.05
    t_in = 623.15
    core_obj = dassh.core.Core(asm_list, asm_pitch, fr, coolant, t_in)
    assemblies = [c_ctrl_asm]
    for i in range(6):
        assemblies.append(c_fuel_asm)
    core_obj.load(assemblies)
    return core_obj


@pytest.fixture
def small_core_no_power_all_fuel(testdir, coolant, c_fuel_asm):
    """DASSH Core object for a 7-assembly core"""
    asm_list = build_asm_list(2)
    asm_pitch = 0.12
    fr = 0.05
    t_in = 623.15
    core_obj = dassh.core.Core(asm_list, asm_pitch, fr, coolant, t_in)
    assemblies = [c_fuel_asm for i in range(7)]
    core_obj.load(assemblies)
    return core_obj


@pytest.fixture
def core_4r_no_corners(coolant):
    """DASSH Core object with 4 rings but empty corner positions"""
    asm_list = build_asm_list(4)
    asm_pitch = 12.0
    gap_fr = 0.05
    c = dassh.core.Core(asm_list, asm_pitch, gap_fr, coolant, test=True)

    try:
        # core_19a_test.asm_map = core.map_asm(asm_list)
        c.asm_map = core.map_asm(asm_list)
    except:
        print("Failure in map_asm() method")
        raise
    try:
        # core_19a_test.asm_adj = \
        #     core.map_adjacent_assemblies(core_19a_test.asm_map)
        c.asm_adj = core.map_adjacent_assemblies(c.asm_map)
    except:
        print("Failure in map_adjacent_assemblies() method")
        raise
    return c


@pytest.fixture
def two_asm_core(testdir, coolant, c_fuel_asm, c_ctrl_asm):
    """Core with two dissimilar assemblies"""
    asm_list = build_asm_list(2, empty_positions=(2, 3, 4, 5, 6))
    asm_pitch = 0.12
    fr = 0.05
    t_in = 623.15
    core_obj = dassh.core.Core(asm_list, asm_pitch, fr, coolant, t_in)
    assemblies = [c_ctrl_asm, c_fuel_asm]
    core_obj.load(assemblies)
    return core_obj


########################################################################
# TEST METHODS
########################################################################


def test_instantiation(core_19a):
    """Test that the Core object has proper attributes"""
    assert all([hasattr(core_19a, x) for x in
                ['n_ring', 'asm_pitch', 'gap_coolant', 'gap_flow_rate',
                 'coolant_gap_params', 'z', 'model', 'asm_map',
                 'asm_adj']])


def test_calc_nring():
    """Test calculation of asm ring number"""
    rings = [1, 3, 6, 9, 10]
    for r in rings:
        n_asm = len(build_asm_list(r))
        assert core.count_rings(n_asm) == r


def test_asm_map_19a():
    """Test calculation of simple asm map"""
    res = core.map_asm(build_asm_list(3))
    # ans = np.array([[8, 9, 10, 0, 0],
    #                 [19, 2, 3, 11, 0],
    #                 [18, 7, 1, 4, 12],
    #                 [0, 17, 6, 5, 13],
    #                 [0, 0, 16, 15, 14]])
    ans = np.array([[ 8, 19, 18,  0,  0],
                    [ 9,  2,  7, 17,  0],
                    [10,  3,  1,  6, 16],
                    [ 0, 11,  4,  5, 15],
                    [ 0,  0, 12, 13, 14]])
    assert np.array_equal(res, ans)


def test_neighbors_19a():
    """Direct comparison of simple case neighbors map (full core)"""
    # 2020-09-29: THIS IS THE OLD ANSWER; it is modified below to
    # produce the new (correct) one by shifting all columns to the
    # right by one and making the last column the first
    asm_map = res = core.map_asm(build_asm_list(3))
    print(asm_map)
    # ans = np.array([[2, 3, 4, 5, 6, 7],
    #                 [8, 9, 3, 1, 7, 19],
    #                 [9, 10, 11, 4, 1, 2],
    #                 [3, 11, 12, 13, 5, 1],
    #                 [1, 4, 13, 14, 15, 6],
    #                 [7, 1, 5, 15, 16, 17],
    #                 [19, 2, 1, 6, 17, 18],
    #                 [0, 0, 9, 2, 19, 0],
    #                 [0, 0, 10, 3, 2, 8],
    #                 [0, 0, 0, 11, 3, 9],
    #                 [10, 0, 0, 12, 4, 3],
    #                 [11, 0, 0, 0, 13, 4],
    #                 [4, 12, 0, 0, 14, 5],
    #                 [5, 13, 0, 0, 0, 15],
    #                 [6, 5, 14, 0, 0, 16],
    #                 [17, 6, 15, 0, 0, 0],
    #                 [18, 7, 6, 16, 0, 0],
    #                 [0, 19, 7, 17, 0, 0],
    #                 [0, 8, 2, 7, 18, 0]])
    ans = np.array([[ 3,  2,  7,  6,  5,  4],
                    [ 9,  8, 19,  7,  1,  3],
                    [10,  9,  2,  1,  4, 11],
                    [11,  3,  1,  5, 13, 12],
                    [ 4,  1,  6, 15, 14, 13],
                    [ 1,  7, 17, 16, 15,  5],
                    [ 2, 19, 18, 17,  6,  1],
                    [ 0,  0,  0, 19,  2,  9],
                    [ 0,  0,  8,  2,  3, 10],
                    [ 0,  0,  9,  3, 11,  0],
                    [ 0, 10,  3,  4, 12,  0],
                    [ 0, 11,  4, 13,  0,  0],
                    [12,  4,  5, 14,  0,  0],
                    [13,  5, 15,  0,  0,  0],
                    [ 5,  6, 16,  0,  0, 14],
                    [ 6, 17,  0,  0,  0, 15],
                    [ 7, 18,  0,  0, 16,  6],
                    [19,  0,  0,  0, 17,  7],
                    [ 8,  0,  0, 18,  7,  2]])
    # ans = np.roll(ans, 1, axis=1)
    res = core.map_adjacent_assemblies(asm_map)
    assert np.array_equal(ans, res)


def test_ia_gap_19a(core_19a_incremental, c_ctrl_asm):
    """Test the definition of interassembly gap subchannels for the
    full core case"""
    ans = np.array([[[1, 2, 3, 4], [5, 6, 7, 8],
                     [9, 10, 11, 12], [13, 14, 15, 16],
                     [17, 18, 19, 20], [21, 22, 23, 24]],

                    [[25, 26, 27, 28], [29, 30, 31, 32],
                     [33, 34, 35, 36], [37, 38, 39, 8],
                     [7, 6, 5, 4], [40, 41, 42, 43]],

                    [[44, 45, 46, 47], [48, 49, 50, 43],
                     [42, 41, 40, 4], [3, 2, 1, 24],
                     [51, 52, 53, 54], [55, 56, 57, 58]],

                    [[59, 60, 61, 54], [53, 52, 51, 24],
                     [23, 22, 21, 20], [62, 63, 64, 65],
                     [66, 67, 68, 69], [70, 71, 72, 73]],

                    [[64, 63, 62, 20], [19, 18, 17, 16],
                     [74, 75, 76, 77], [78, 79, 80, 81],
                     [82, 83, 84, 85], [86, 87, 88, 65]],

                    [[15, 14, 13, 12], [89, 90, 91, 92],
                     [93, 94, 95, 96], [97, 98, 99, 100],
                     [101, 102, 103, 77], [76, 75, 74, 16]],

                    [[39, 38, 37, 36], [104, 105, 106, 107],
                     [108, 109, 110, 111], [112, 113, 114, 92],
                     [91, 90, 89, 12], [11, 10, 9, 8]],

                    [[115, 116, 117, 118], [119, 120, 121, 122],
                     [123, 124, 125, 126], [127, 128, 129, 32],
                     [31, 30, 29, 28], [130, 131, 132, 133]],

                    [[134, 135, 136, 137], [138, 139, 140, 133],
                     [132, 131, 130, 28], [27, 26, 25, 43],
                     [50, 49, 48, 47], [141, 142, 143, 144]],

                    [[145, 146, 147, 148], [149, 150, 151, 144],
                     [143, 142, 141, 47], [46, 45, 44, 58],
                     [152, 153, 154, 155], [156, 157, 158, 159]],

                    [[160, 161, 162, 155], [154, 153, 152, 58],
                     [57, 56, 55, 54], [61, 60, 59, 73],
                     [163, 164, 165, 166], [167, 168, 169, 170]],

                    [[171, 172, 173, 166], [165, 164, 163, 73],
                     [72, 71, 70, 69], [174, 175, 176, 177],
                     [178, 179, 180, 181], [182, 183, 184, 185]],

                    [[176, 175, 174, 69], [68, 67, 66, 65],
                     [88, 87, 86, 85], [186, 187, 188, 189],
                     [190, 191, 192, 193], [194, 195, 196, 177]],

                    [[188, 187, 186, 85], [84, 83, 82, 81],
                     [197, 198, 199, 200], [201, 202, 203, 204],
                     [205, 206, 207, 208], [209, 210, 211, 189]],

                    [[80, 79, 78, 77], [103, 102, 101, 100],
                     [212, 213, 214, 215], [216, 217, 218, 219],
                     [220, 221, 222, 200], [199, 198, 197, 81]],

                    [[99, 98, 97, 96], [223, 224, 225, 226],
                     [227, 228, 229, 230], [231, 232, 233, 234],
                     [235, 236, 237, 215], [214, 213, 212, 100]],

                    [[114, 113, 112, 111], [238, 239, 240, 241],
                     [242, 243, 244, 245], [246, 247, 248, 226],
                     [225, 224, 223, 96], [95, 94, 93, 92]],

                    [[249, 250, 251, 252], [253, 254, 255, 256],
                     [257, 258, 259, 260], [261, 262, 263, 241],
                     [240, 239, 238, 111], [110, 109, 108, 107]],

                    [[129, 128, 127, 126], [264, 265, 266, 267],
                     [268, 269, 270, 252], [251, 250, 249, 107],
                     [106, 105, 104, 36], [35, 34, 33, 32]]
                    ])
    ans = ans.reshape(ans.shape[0], -1)

    # Assemblies have 37 pins; can use existing conceptual ctrl asm
    asm_list = [c_ctrl_asm for i in range(19)]
    core_19a_incremental._geom_params = \
        core_19a_incremental._collect_sc_geom_params(asm_list)
    asm_adj_sc = core_19a_incremental._map_asm_gap_adjacency()
    # Combine nested lists: a little ugly bc not all same length
    max_scpa = max([sum([len(x) for x in y]) for y in asm_adj_sc])
    test = np.zeros((19, max_scpa), dtype=int)
    for a in range(19):
        tmp = np.array([x for l in asm_adj_sc[a] for x in l])
        tmp = tmp.flatten()
        test[a, :tmp.shape[0]] = tmp

    # test = np.stack(test)
    print(test - ans)
    assert np.array_equal(ans, test)


def test_interasm_gap_sc_types_full(core_19a_incremental, c_ctrl_asm):
    """Test the type assignment of interassembly gap subchannels"""
    # problem setup: asms have 37 pins; use existing conceptual ctrl asm
    asm_list = [c_ctrl_asm for i in range(19)]
    core_19a_incremental.load(asm_list)
    # hex_side_len = asm_list[0].duct_oftf / np.sqrt(3)
    # _asm_sc_xb_side = core_19a_incremental._calculate_gap_xbnds(asm_list)
    # core_19a_incremental._geom_params = \
    #     core_19a_incremental._collect_sc_geom_params(asm_list, _asm_sc_xb_side)
    # tmp = core_19a_incremental._determine_gap_sc_types()
    # sc_types = np.array(tmp[1])
    sc_types = core_19a_incremental._sc_types
    corners = np.array([4, 8, 12, 16, 20, 24, 28, 32, 36, 43, 47, 54,
                        58, 65, 69, 73, 77, 81, 85, 92, 96, 100, 107,
                        111, 118, 122, 126, 133, 137, 144, 148, 155,
                        159, 166, 170, 177, 181, 185, 189, 193, 200,
                        204, 208, 215, 219, 226, 230, 234, 241, 245,
                        252, 256, 260, 267])
    # print(sc_types)
    for sci in range(len(sc_types)):
        if sci + 1 in corners:
            if sc_types[sci] == 1:
                continue
            else:
                print('I expected this to be a corner:', sci + 1)
                assert 0
        else:
            if sc_types[sci] == 0:
                continue
            else:
                print('I expected this to be an edge:', sci + 1)
                assert 0


def test_interasm_gap_sc_adj_full(core_19a_incremental, c_ctrl_asm):
    """Test the adjacency mapping of interassembly gap subchannels"""
    # problem setup
    # side-subchannels-per-side = 3
    # core_19a_incremental._sc_per_asm = 24
    # core_19a_incremental._sc_per_side = 3
    # core_19a_incremental.asm_sc_adj = \
    #     core_19a_incremental.map_interassembly_sc()
    # Assemblies have 37 pins; can use existing conceptual ctrl asm
    asm_list = [c_ctrl_asm for i in range(19)]
    core_19a_incremental._geom_params = \
        core_19a_incremental._collect_sc_geom_params(asm_list)
    asm_adj_sc = core_19a_incremental._map_asm_gap_adjacency()
    core_19a_incremental.n_sc = np.max([np.max([x for x in y])
                                        for y in asm_adj_sc])
    print(core_19a_incremental.n_sc)
    sc_adj = core_19a_incremental._find_adjacent_sc(asm_adj_sc)
    ans = np.array([[24, 2, 0], [1, 3, 0], [2, 4, 0],
                    [3, 5, 40], [4, 6, 0], [5, 7, 0],
                    [6, 8, 0], [7, 9, 39], [8, 10, 0],
                    [9, 11, 0], [10, 12, 0], [11, 13, 89],
                    [12, 14, 0], [13, 15, 0], [14, 16, 0],
                    [15, 17, 74], [16, 18, 0], [17, 19, 0],
                    [18, 20, 0], [19, 21, 62], [20, 22, 0],
                    [21, 23, 0], [22, 24, 0], [23, 1, 51],
                    [43, 26, 0], [25, 27, 0], [26, 28, 0],
                    [27, 29, 130], [28, 30, 0], [29, 31, 0],
                    [30, 32, 0], [31, 33, 129], [32, 34, 0],
                    [33, 35, 0], [34, 36, 0], [35, 37, 104],
                    [36, 38, 0], [37, 39, 0], [38, 8, 0],
                    [4, 41, 0], [40, 42, 0], [41, 43, 0],
                    [25, 42, 50], [58, 45, 0], [44, 46, 0],
                    [45, 47, 0], [46, 48, 141], [47, 49, 0],
                    [48, 50, 0], [49, 43, 0], [24, 52, 0],
                    [51, 53, 0], [52, 54, 0], [53, 55, 61],
                    [54, 56, 0], [55, 57, 0], [56, 58, 0],
                    [57, 44, 152], [73, 60, 0], [59, 61, 0],
                    [60, 54, 0], [20, 63, 0], [62, 64, 0],
                    [63, 65, 0], [64, 66, 88], [65, 67, 0],
                    [66, 68, 0], [67, 69, 0], [68, 70, 174],
                    [69, 71, 0], [70, 72, 0], [71, 73, 0],
                    [59, 72, 163], [16, 75, 0], [74, 76, 0],
                    [75, 77, 0], [76, 78, 103], [77, 79, 0],
                    [78, 80, 0], [79, 81, 0], [80, 82, 197],
                    [81, 83, 0], [82, 84, 0], [83, 85, 0],
                    [84, 86, 186], [85, 87, 0], [86, 88, 0],
                    [87, 65, 0], [12, 90, 0], [89, 91, 0],
                    [90, 92, 0], [91, 93, 114], [92, 94, 0],
                    [93, 95, 0], [94, 96, 0], [95, 97, 223],
                    [96, 98, 0], [97, 99, 0], [98, 100, 0],
                    [99, 101, 212], [100, 102, 0], [101, 103, 0],
                    [102, 77, 0], [36, 105, 0], [104, 106, 0],
                    [105, 107, 0], [106, 108, 249], [107, 109, 0],
                    [108, 110, 0], [109, 111, 0], [110, 112, 238],
                    [111, 113, 0], [112, 114, 0], [113, 92, 0],
                    [133, 116, 0], [115, 117, 0], [116, 118, 0],
                    [117, 119, 0], [118, 120, 0], [119, 121, 0],
                    [120, 122, 0], [121, 123, 0], [122, 124, 0],
                    [123, 125, 0], [124, 126, 0], [125, 127, 264],
                    [126, 128, 0], [127, 129, 0], [128, 32, 0],
                    [28, 131, 0], [130, 132, 0], [131, 133, 0],
                    [132, 115, 140], [144, 135, 0], [134, 136, 0],
                    [135, 137, 0], [136, 138, 0], [137, 139, 0],
                    [138, 140, 0], [139, 133, 0], [47, 142, 0],
                    [141, 143, 0], [142, 144, 0], [143, 134, 151],
                    [159, 146, 0], [145, 147, 0], [146, 148, 0],
                    [147, 149, 0], [148, 150, 0], [149, 151, 0],
                    [150, 144, 0], [58, 153, 0], [152, 154, 0],
                    [153, 155, 0], [154, 156, 162], [155, 157, 0],
                    [156, 158, 0], [157, 159, 0], [158, 145, 0],
                    [170, 161, 0], [160, 162, 0], [161, 155, 0],
                    [73, 164, 0], [163, 165, 0], [164, 166, 0],
                    [165, 167, 173], [166, 168, 0], [167, 169, 0],
                    [168, 170, 0], [169, 160, 0], [185, 172, 0],
                    [171, 173, 0], [172, 166, 0], [69, 175, 0],
                    [174, 176, 0], [175, 177, 0], [176, 178, 196],
                    [177, 179, 0], [178, 180, 0], [179, 181, 0],
                    [180, 182, 0], [181, 183, 0], [182, 184, 0],
                    [183, 185, 0], [184, 171, 0], [85, 187, 0],
                    [186, 188, 0], [187, 189, 0], [188, 190, 211],
                    [189, 191, 0], [190, 192, 0], [191, 193, 0],
                    [192, 194, 0], [193, 195, 0], [194, 196, 0],
                    [195, 177, 0], [81, 198, 0], [197, 199, 0],
                    [198, 200, 0], [199, 201, 222], [200, 202, 0],
                    [201, 203, 0], [202, 204, 0], [203, 205, 0],
                    [204, 206, 0], [205, 207, 0], [206, 208, 0],
                    [207, 209, 0], [208, 210, 0], [209, 211, 0],
                    [210, 189, 0], [100, 213, 0], [212, 214, 0],
                    [213, 215, 0], [214, 216, 237], [215, 217, 0],
                    [216, 218, 0], [217, 219, 0], [218, 220, 0],
                    [219, 221, 0], [220, 222, 0], [221, 200, 0],
                    [96, 224, 0], [223, 225, 0], [224, 226, 0],
                    [225, 227, 248], [226, 228, 0], [227, 229, 0],
                    [228, 230, 0], [229, 231, 0], [230, 232, 0],
                    [231, 233, 0], [232, 234, 0], [233, 235, 0],
                    [234, 236, 0], [235, 237, 0], [236, 215, 0],
                    [111, 239, 0], [238, 240, 0], [239, 241, 0],
                    [240, 242, 263], [241, 243, 0], [242, 244, 0],
                    [243, 245, 0], [244, 246, 0], [245, 247, 0],
                    [246, 248, 0], [247, 226, 0], [107, 250, 0],
                    [249, 251, 0], [250, 252, 0], [251, 253, 270],
                    [252, 254, 0], [253, 255, 0], [254, 256, 0],
                    [255, 257, 0], [256, 258, 0], [257, 259, 0],
                    [258, 260, 0], [259, 261, 0], [260, 262, 0],
                    [261, 263, 0], [262, 241, 0], [126, 265, 0],
                    [264, 266, 0], [265, 267, 0], [266, 268, 0],
                    [267, 269, 0], [268, 270, 0], [269, 252, 0]
                    ])

    for sc in range(len(sc_adj)):
        if not np.all([x in sc_adj[sc] for x in ans[sc]]):
            print(sc, sc_adj[sc], ans[sc])
            assert 0


def test_asm_map_missing():
    """Test assembly map creation with missing assemblies"""
    empty = (19, 22, 25, 28, 31, 34)
    asm_list = build_asm_list(4, empty_positions=empty)
    res = core.map_asm(asm_list)
    # ans = np.array([[ 0, 20, 21,  0,  0,  0,  0],
    #                 [31,  8,  9, 10, 22,  0,  0],
    #                 [30, 19,  2,  3, 11, 23,  0],
    #                 [ 0, 18,  7,  1,  4, 12,  0],
    #                 [ 0, 29, 17,  6,  5, 13, 24],
    #                 [ 0,  0, 28, 16, 15, 14, 25],
    #                 [ 0,  0,  0,  0, 27, 26,  0]], dtype=int)
    ans = np.array([[ 0, 31, 30,  0,  0,  0,  0],
                    [20,  8, 19, 18, 29,  0,  0],
                    [21,  9,  2,  7, 17, 28,  0],
                    [ 0, 10,  3,  1,  6, 16,  0],
                    [ 0, 22, 11,  4,  5, 15, 27],
                    [ 0,  0, 23, 12, 13, 14, 26],
                    [ 0,  0,  0,  0, 24, 25,  0]], dtype=int)
    assert np.array_equal(res, ans)


def test_asm_adj_missing():
    """Test assembly adjacency with missing assemblies"""
    empty = (19, 22, 25, 28, 31, 34)
    asm_list = build_asm_list(4, empty_positions=empty)
    map = core.map_asm(asm_list)
    res = core.map_adjacent_assemblies(map)
    ans = np.array([[ 3,  2,  7,  6,  5,  4],
                    [ 9,  8, 19,  7,  1,  3],
                    [10,  9,  2,  1,  4, 11],
                    [11,  3,  1,  5, 13, 12],
                    [ 4,  1,  6, 15, 14, 13],
                    [ 1,  7, 17, 16, 15,  5],
                    [ 2, 19, 18, 17,  6,  1],
                    [20,  0, 31, 19,  2,  9],
                    [21, 20,  8,  2,  3, 10],
                    [ 0, 21,  9,  3, 11, 22],
                    [22, 10,  3,  4, 12, 23],
                    [23, 11,  4, 13, 24,  0],
                    [12,  4,  5, 14, 25, 24],
                    [13,  5, 15, 26,  0, 25],
                    [ 5,  6, 16, 27, 26, 14],
                    [ 6, 17, 28,  0, 27, 15],
                    [ 7, 18, 29, 28, 16,  6],
                    [19, 30,  0, 29, 17,  7],
                    [ 8, 31, 30, 18,  7,  2],
                    [ 0,  0,  0,  8,  9, 21],
                    [ 0,  0, 20,  9, 10,  0],
                    [ 0,  0, 10, 11, 23,  0],
                    [ 0, 22, 11, 12,  0,  0],
                    [ 0, 12, 13, 25,  0,  0],
                    [24, 13, 14,  0,  0,  0],
                    [14, 15, 27,  0,  0,  0],
                    [15, 16,  0,  0,  0, 26],
                    [17, 29,  0,  0,  0, 16],
                    [18,  0,  0,  0, 28, 17],
                    [31,  0,  0,  0, 18, 19],
                    [ 0,  0,  0, 30, 19,  8]], dtype=int)
    print(res - ans)
    assert np.array_equal(res, ans)


def test_asm_adj_missing_middle():
    """What happens if the center assembly is empty?"""
    empty = (0,)
    asm_list = build_asm_list(2, empty_positions=empty)
    map = core.map_asm(asm_list)
    res = core.map_adjacent_assemblies(map)
    ans = np.array([[0, 0, 0, 6, 0, 2],
                    [0, 0, 1, 0, 3, 0],
                    [0, 2, 0, 4, 0, 0],
                    [3, 0, 5, 0, 0, 0],
                    [0, 6, 0, 0, 0, 4],
                    [1, 0, 0, 0, 5, 0]], dtype=int)
    print(res - ans)
    assert np.array_equal(res, ans)


def test_subchannel_areas(small_core_no_power):
    """Test that the sum of the individual subchannel areas yields
    the calculated total"""
    c = small_core_no_power
    # total edge area
    n_edge_sc = np.where(c._sc_types == 0)[0].shape[0]
    area = n_edge_sc * c.gap_params['area'][0]
    # Total corner area (with 1 or 2 neighbors)
    n_corn_neighbor = 12
    area += n_corn_neighbor * c.gap_params['area'][8]
    # Total corner area (with no neighbors)
    n_corn_no_neighbor = 12
    tmp = c._geom_params['dims'][0, 0, 1] * 2 * c.d_gap
    tmp += c.d_gap**2 * np.sqrt(3) / 3
    area += tmp * n_corn_no_neighbor
    tot = np.sum(c.gap_params['area'])
    assert tot == pytest.approx(c.gap_params['total area'])
    assert area == pytest.approx(c.gap_params['total area'])


def test_gap_disagreement_adjacency(two_asm_core):
    """Test the subchannel assignment in dissimilar two-asm core"""
    # Test assembly-sc adjacency
    assert np.count_nonzero(two_asm_core._asm_sc_adj[0]) == 29
    assert np.count_nonzero(two_asm_core._asm_sc_adj[1]) == 54
    # Total number of unique gap subchannels
    assert two_asm_core.n_sc == 73


def test_gap_disagreement_typing(two_asm_core):
    """Test subchannel typing in dissimilar two-asm core"""
    # Gap subchannel typing
    asm_sc_types0 = np.zeros(29, dtype='int')
    asm_sc_types0[[3, 12, 16, 20, 24, 28]] = 1
    asm_sc_types1 = np.zeros(54, dtype='int')
    asm_sc_types1[[8, 17, 26, 35, 44, 53]] = 1
    assert np.array_equal(two_asm_core._asm_sc_types[0], asm_sc_types0)
    assert np.array_equal(two_asm_core._asm_sc_types[1], asm_sc_types1)
    global_types = np.zeros(73, dtype='int')
    global_types[[3, 12, 16, 20, 24, 28, 37, 46, 55, 72]] = 1
    assert np.array_equal(two_asm_core._sc_types, global_types)


def test_gap_disagreement_xbnds(two_asm_core):
    """Test subchannel boundaries in dissimilar two-asm core"""
    # subchannel boundaries adjacent to assembly
    c = two_asm_core
    hex_perim = 6 * c.duct_oftf / np.sqrt(3)
    dxb0 = c._asm_sc_xbnds[0][1:] - c._asm_sc_xbnds[0][:-1]
    last_entry = c._asm_sc_xbnds[0][c._asm_sc_xbnds[0] > 0][-1]
    dxb0[28] = hex_perim - last_entry + c._asm_sc_xbnds[0, 0]
    ans = np.zeros(53)
    # Most of assembly 1 subchannels are edges defined by assembly 1
    # pin pitch, so start with that
    ans[:29] = c._geom_params['dims'][0, 0, 0]
    # Edges between assemblies 1 and 2 defined by assembly 2 pin pitch
    ans[[4, 5, 6, 7, 8, 9, 10, 11]] = c._geom_params['dims'][1, 0, 0]
    # These corners are defined by assembly 1 corner perimeter
    ans[[16, 20, 24, 28]] = 2 * c._geom_params['dims'][0, 0, 1]
    # These are defined by both assembly 1 and 2 corner perimeters
    ans[[3, 12]] = c._geom_params['dims'][0, 0, 1]
    ans[[3, 12]] += c._geom_params['dims'][1, 0, 1]
    # Now check - skip the last corner value because "dxb0" is not
    # calculating it properly. Need the hexagon perimeter to calculate
    assert np.allclose(dxb0, ans)

    dxb1 = np.zeros(54)
    dxb1[:-1] = c._asm_sc_xbnds[1][1:] - c._asm_sc_xbnds[1][:-1]
    dxb1[-1] = hex_perim - c._asm_sc_xbnds[1, -1] + c._asm_sc_xbnds[1, 0]
    print(dxb1.shape)
    ans = np.ones(54) * c._geom_params['dims'][1, 0, 0]
    ans[[8, 17, 26, 35, 44, 53]] = 2 * c._geom_params['dims'][1, 0, 1]
    assert np.allclose(dxb1, ans)


def test_gap_disagreement_area(two_asm_core):
    """Test subchannel areas in dissimilar two-asm core"""
    c = two_asm_core
    corner_no_neighbor = c.d_gap**2 * np.sqrt(3) / 3
    corner_neighbor = c.d_gap**2 * np.sqrt(3) / 4

    # Assembly 1 -defined subchannels
    a_edge1 = c._geom_params['dims'][0, 0, 0] * c.d_gap
    a_corn1 = 2 * c._geom_params['dims'][0, 0, 1] * c.d_gap
    a_corn1 += corner_no_neighbor

    # Assembly 2 -defined subchannels
    a_edge2 = c._geom_params['dims'][1, 0, 0] * c.d_gap
    a_corn2 = 2 * c._geom_params['dims'][1, 0, 1] * c.d_gap
    a_corn2 += corner_no_neighbor

    # Shared corner: one "leg" defined by assembly 1; two legs defined
    # by assembly 2
    a_corn12 = c.d_gap * (c._geom_params['dims'][0, 0, 1]
                          + 2 * c._geom_params['dims'][1, 0, 1])
    a_corn12 += corner_neighbor

    # Combine
    ans = np.zeros(73)
    ans[:29] = a_edge1
    ans[[4, 5, 6, 7, 8, 9, 10, 11]] = a_edge2
    ans[[16, 20, 24, 28]] = a_corn1
    ans[[3, 12]] = a_corn12
    ans[29:] = a_edge2
    ans[[37, 46, 55, 72]] = a_corn2
    assert np.allclose(c.gap_params['area'], ans)


########################################################################
# TESTS FOR THE INTERASSEMBLY GAP TEMPERATURE METHODS
########################################################################


def test_interasm_gap_zero_temp(small_core_no_power):
    """Test that no temp gradient produces no gap temp change"""
    t_in = 623.15
    T_duct = np.ones((small_core_no_power.n_asm,
                      np.max(small_core_no_power._n_sc_per_asm)))
    T_duct *= t_in
    small_core_no_power.calculate_gap_temperatures(0.001, T_duct)
    print(small_core_no_power.coolant_gap_params)
    assert np.allclose(small_core_no_power.coolant_gap_temp, t_in)


def test_average_gap_temp(small_core_no_power):
    """Test the calculation of area-weighted average temperature"""
    # print(small_core_no_power.asm_params['duct_ftf'][1] / np.sqrt(3))
    # print(small_core_no_power.asm_params['pin_pitch'][1])
    # print(small_core_no_power.asm_params['n_pin'][1])
    # print(small_core_no_power.d['wcorner'])
    # print(small_core_no_power.gap_params['area'])
    # print(small_core_no_power.gap_params['total area'])
    print(len(small_core_no_power.coolant_gap_temp))
    print(small_core_no_power.coolant_gap_temp)
    assert (pytest.approx(small_core_no_power.avg_coolant_gap_temp)
            == 623.15)
    # perturb half the temperatures, recalculate average
    n_sc_over_2 = int(len(small_core_no_power.coolant_gap_temp) / 2)
    small_core_no_power.coolant_gap_temp[0:n_sc_over_2] += 100.0
    ans = np.average([623.15, 723.15])
    assert ans == \
        pytest.approx(small_core_no_power.avg_coolant_gap_temp, 1.0)


def test_adiabatic_wall_temp(small_core_no_power):
    """Test that adiabatic wall model never changes gap temp and that
    outer wall HTC is always returned as zero"""
    small_core_no_power.model = None
    scpa = np.max(small_core_no_power._n_sc_per_asm)
    T_duct = [(np.random.random(scpa) - 0.5) * 5 + 650 for a in
              range(small_core_no_power.n_asm)]
    small_core_no_power.calculate_gap_temperatures(10.0, T_duct)
    assert small_core_no_power.avg_coolant_gap_temp == \
        pytest.approx(623.15)
    assert np.all(small_core_no_power.coolant_gap_temp == 623.15)
    assert np.all(small_core_no_power.coolant_gap_params['htc'] == 0.0)


def test_accelerated_noflow_model(small_core_no_power_all_fuel):
    """Test the numpy implementation of the no-flow model against
    what was previously implemented (used to get the "answer" below)"""
    # Set up random values for coolant and gap temperatures to ensure
    # that you're not using the "flat" initial temperature profile
    c = copy.deepcopy(small_core_no_power_all_fuel)
    c.coolant_gap_temp = np.random.random(len(c.coolant_gap_temp))
    c.coolant_gap_temp += 623.15
    t_duct = np.random.random(c._asm_sc_adj.shape) * 10 + 624.15
    # -----------------------------------------------------------------
    # OLD METHOD - get the answer against which to check new method
    # 2021-04-27 - modified to use new attributes
    ans = np.zeros(len(c._sc_adj))
    # Convection resistance factor
    R_conv = np.array(
        [1 / (c.d_gap / 2 / (c.gap_params['wp'][0] * 0.5)
              / c.gap_coolant.thermal_conductivity),
         1 / (c.d_gap / 2 / (c.gap_params['wp'][8] / 3)
              / c.gap_coolant.thermal_conductivity)])
    # Conduction resistance factor
    R_cond = c.d_gap * c.gap_coolant.thermal_conductivity
    L = np.zeros((2, 2))
    L[0, 0] = c.gap_params['L'][4, 0]
    L[0, 1] = c.gap_params['L'][0, 0]
    L[1, 0] = L[0, 1]
    for sci in range(len(c._sc_adj)):
        type_i = c._sc_types[sci]
        C = 0.0
        # Collect adjacent duct wall temperatures - identify
        # adjacent assemblies to find duct wall temps
        asm, loc = np.where(c._asm_sc_adj == sci + 1)
        for i in range(len(asm)):
            ans[sci] += t_duct[asm[i], loc[i]] * R_conv[type_i]
            C += R_conv[type_i]

        # Conduction to/from adjacent coolant subchannels
        for j in range(3):
            adj = c._sc_adj[sci, j]
            if adj == 0:
                continue
            sc_adj = adj - 1
            type_a = c._sc_types[sc_adj]
            ans[sci] += (c.coolant_gap_temp[sc_adj]
                         * R_cond / L[type_i, type_a])
            C += R_cond / L[type_i, type_a]

        ans[sci] = ans[sci] / C

    # -----------------------------------------------------------------
    res = c._noflow_model(t_duct)
    diff = res - ans
    for i in range(len(diff)):
        if np.abs(diff[i]) > 1e-10:
            print(i, diff[i], c._sc_types[i])
    assert np.allclose(ans, res)


def test_accelerated_ductavg_model(small_core_no_power_all_fuel):
    """Test the numpy implementation of the duct-avg model against
    what was previously implemented (used to get the "answer" below)"""
    c = small_core_no_power_all_fuel  # shortcut
    approx_duct = np.random.random(c._asm_sc_adj.shape) * 10 + 623.15
    # -----------------------------------------------------------------
    # OLD MODEL
    ans = np.zeros(c.n_sc)
    for sci in range(c.n_sc):
        # Collect adjacent duct wall temperatures - identify
        # adjacent assemblies to find duct wall temps
        t_duct = []
        asm, loc = np.where(c._asm_sc_adj == sci + 1)
        for i in range(len(asm)):
            t_duct.append(approx_duct[asm[i]][loc[i]])
        ans[sci] = np.average(t_duct)
    # -----------------------------------------------------------------
    res = c._duct_average_model(approx_duct)
    assert np.allclose(ans, res)


def test_acc_flow_model_conv_only(small_core_no_power_all_fuel):
    """Test numpy-ized implementation of flowing gap model against
    the previous version"""
    c = small_core_no_power_all_fuel  # shortcut

    # OLD ATTR: HTCONSTS
    L = [[c.gap_params['L'][4, 0], c.gap_params['L'][0, 0]],
         [c.gap_params['L'][0, 0], 0.0]]
    a = [c.gap_params['area'][0], c.gap_params['area'][8]]
    htc = [c.coolant_gap_params['htc'][0], c.coolant_gap_params['htc'][8]]
    ht_consts = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    # Conduction between coolant channels
    for i in range(2):
        for j in range(2):
            if L[i][j] != 0.0:
                ht_consts[i][j] = (c.d_gap
                                   * c.gap_params['total area']
                                   / L[i][j]
                                   / c.gap_flow_rate
                                   / a[i])
            # Convection from coolant to duct wall (units: m-s/kg)
            # Edge -> wall 1
            ht_consts[0][2] = (L[0][0]
                               * c.gap_params['total area']
                               / c.gap_flow_rate
                               / a[0])
            # Corner -> wall 1
            ht_consts[1][2] = (c._conv_util['const'][8, 0]
                               * c.gap_params['total area']
                               / c.gap_flow_rate
                               / a[1])

    # Old method, copied/pasted here
    def _convection_model_OLD(self, dz, approx_duct_temps, htconsts, htc):
        """Inter-assembly gap convection model

        Parameters
        ----------
        self : DASSH Core object
        dz : float
            Axial mesh height
        approx_duct_temps : numpy.ndarray
            Array of outer duct surface temperatures (K) for each
            assembly in the core (can be any length) on the inter-
            assembly gap subchannel mesh

        Returns
        -------
        numpy.ndarray
            Temperature change in the inter-assembly gap coolant

        """
        dT = np.zeros(len(self._sc_adj))
        for sci in range(len(self._sc_adj)):
            type_i = self._sc_types[sci]

            # Convection to/from duct wall
            # identify adjacent assemblies to find duct wall temps
            asm, loc = np.where(self._asm_sc_adj == sci + 1)
            for i in range(len(asm)):
                # adj_duct_temp = approx_duct_temps[asm[i], duct_temp_idx]
                adj_duct_temp = approx_duct_temps[asm[i]][loc[i]]
                dT[sci] += \
                    (htc[type_i]
                     * dz * ht_consts[type_i][2]
                     * (adj_duct_temp - self.coolant_gap_temp[sci])
                     / self.gap_coolant.heat_capacity)

            # Conduction to/from adjacent coolant subchannels
            for adj in self._sc_adj[sci]:
                if adj == 0:
                    continue
                sc_adj = adj - 1
                type_a = self._sc_types[sc_adj]
                dT[sci] += (self.gap_coolant.thermal_conductivity
                            * dz * ht_consts[type_i][type_a]
                            * (self.coolant_gap_temp[sc_adj]
                               - self.coolant_gap_temp[sci])
                            / self.gap_coolant.heat_capacity)

        return dT

    approx_duct = np.random.random(c._asm_sc_adj.shape) * 10 + 623.15
    ans = _convection_model_OLD(c, 0.1, approx_duct, ht_consts, htc)
    res = c._flow_model(0.1, approx_duct)
    diff = res - ans
    for i in range(diff.shape[0]):
        if np.abs(diff[i]) > 1e-10:
            print(i, res[i], ans[i], diff[i], c._sc_types[i])
            assert c._sc_types[i] == 1
            asm, loc = np.where(c._asm_sc_adj == i + 1)
            assert len(asm) == 1
            ratio = res[i] / ans[i]
            x_new = c.coolant_gap_params['htc'][i] / c._sc_mfr[i]
            m_old = c.gap_flow_rate * a[1] / c.gap_params['total area']
            x_old = htc[1] / m_old
            assert ratio - x_new / x_old < 1e-12


def test_acc_flow_model(small_core_no_power_all_fuel, c_fuel_asm):
    """Test numpy-ized implementation of flowing gap model against
    the previous version"""
    c = small_core_no_power_all_fuel  # shortcut
    perturb = np.random.random(c.coolant_gap_temp.shape) * 10 + 5
    c.coolant_gap_temp += perturb  # force conduction with temp diff
    # OLD ATTR: HTCONSTS
    L = [[c.gap_params['L'][4, 0], c.gap_params['L'][0, 0]],
         [c.gap_params['L'][0, 0], 0.0]]
    a = [c.gap_params['area'][0], c.gap_params['area'][8]]
    htc = [c.coolant_gap_params['htc'][0], c.coolant_gap_params['htc'][8]]
    ht_consts = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    # Conduction between coolant channels
    for i in range(2):
        for j in range(2):
            if L[i][j] != 0.0:
                ht_consts[i][j] = (c.d_gap
                                   * c.gap_params['total area']
                                   / L[i][j]
                                   / c.gap_flow_rate
                                   / a[i])
            # Convection from coolant to duct wall (units: m-s/kg)
            # Edge -> wall 1
            ht_consts[0][2] = (L[0][0]
                               * c.gap_params['total area']
                               / c.gap_flow_rate
                               / a[0])
            # Corner -> wall 1
            ht_consts[1][2] = (c._conv_util['const'][8, 0]
                               * c.gap_params['total area']
                               / c.gap_flow_rate
                               / a[1])

    # Old method, copied/pasted here
    def _convection_model_OLD(self, dz, approx_duct_temps, htconsts, htc):
        """Inter-assembly gap convection model

        Parameters
        ----------
        self : DASSH Core object
        dz : float
            Axial mesh height
        approx_duct_temps : numpy.ndarray
            Array of outer duct surface temperatures (K) for each
            assembly in the core (can be any length) on the inter-
            assembly gap subchannel mesh

        Returns
        -------
        numpy.ndarray
            Temperature change in the inter-assembly gap coolant

        """
        dT = np.zeros(len(self._sc_adj))
        for sci in range(len(self._sc_adj)):
            type_i = self._sc_types[sci]

            # Convection to/from duct wall
            # identify adjacent assemblies to find duct wall temps
            asm, loc = np.where(self._asm_sc_adj == sci + 1)
            for i in range(len(asm)):
                # adj_duct_temp = approx_duct_temps[asm[i], duct_temp_idx]
                adj_duct_temp = approx_duct_temps[asm[i]][loc[i]]
                dT[sci] += \
                    (htc[type_i]
                     * dz * ht_consts[type_i][2]
                     * (adj_duct_temp - self.coolant_gap_temp[sci])
                     / self.gap_coolant.heat_capacity)

            # Conduction to/from adjacent coolant subchannels
            for adj in self._sc_adj[sci]:
                if adj == 0:
                    continue
                sc_adj = adj - 1
                type_a = self._sc_types[sc_adj]
                dT[sci] += (self.gap_coolant.thermal_conductivity
                            * dz * ht_consts[type_i][type_a]
                            * (self.coolant_gap_temp[sc_adj]
                               - self.coolant_gap_temp[sci])
                            / self.gap_coolant.heat_capacity)

        return dT

    approx_duct = np.random.random(c._asm_sc_adj.shape) * 10 + 623.15
    ans = _convection_model_OLD(c, 0.1, approx_duct, ht_consts, htc)
    res = c._flow_model(0.1, approx_duct)
    diff = res - ans
    for i in range(diff.shape[0]):
        if np.abs(diff[i]) > 1e-10:
            print(i, res[i], ans[i], diff[i], c._sc_types[i])
            assert c._sc_types[i] == 1
            asm, loc = np.where(c._asm_sc_adj == i + 1)
            assert len(asm) == 1


# def test_interasm_gap_asm_adj_temps(small_core_no_power):
#     """Test that the core object can return the interasm gap temps
#     for subchannels around a specific assembly"""
#     id = 0
#     sc_to_return = np.hstack(small_core_no_power.asm_sc_adj[id]) - 1
#     t_sc = small_core_no_power.coolant_gap_temp[sc_to_return]
#     print(len(t_sc))
#     x_core = small_core_no_power.x_pts
#     x_asm = small_core_no_power.asm_params['x'][id]
#     t_asm = small_core_no_power._approximate_temps(x_core, t_sc, x_asm)
#     print(len(t_asm))
#     assert 0


#
# def test_interasm_gap_perturb_temp(small_core_no_power, fuel_asm, ctrl_asm):
#     """."""
#     inlet_temp = 623.15
#     perturb_temp = 10.0
#     T_duct = [ctrl_asm.duct_outer_surf_temp]
#     for i in range(6):
#         T_duct.append(fuel_asm.duct_outer_surf_temp)
#     for a in range(len(T_duct)):
#         for sci in range(len(T_duct[a])):
#             T_duct[a][sci] += perturb_temp
#             T_gap = small_core_no_power.calculate_coolant_gap_temp(0.001,
#                                                               T_duct)
#             T_duct[a][sci] -= perturb_temp
#     # assert np.allclose(T_gap, inlet_temp)


########################################################################
# OLD PERIODIC TESTS - MAY COME BACK ONE DAY
########################################################################

#
# @pytest.fixture
# def g_full(testdir):
#     return py4c.geodst.GEODST(os.path.join(testdir,
#                                            'test_data', 'GEODST_full'))
#

#
#
# class GEODST_simple(object):
#
#     arr = np.array([[2, 2, 2, 0, 0],
#                     [2, 1, 1, 2, 0],
#                     [2, 1, 3, 1, 2],
#                     [0, 2, 1, 1, 2],
#                     [0, 0, 2, 2, 2]])
#
#     def __init__(self, type):
#         # distance between hex centroids
#         self.xmesh = np.array([0.0, 12.0, 0.0])  # cm;
#         self.ifints = np.array([1, 1, 1])  # arbitrary array len
#         self.geom_type = "hex-z"
#         if type == '60':
#             self.triangle_option = "rhomb-60"
#             self.bcs = np.array([[2, 2], [4, 2], [4, 2]])
#             self.reg_assignments = np.rot90(self.arr[2:, 0:3])
#         elif type == '120':
#             self.triangle_option = "rhomb-120"
#             self.bcs = np.array([[2, 2], [0, 2], [4, 2]])
#             self.reg_assignments = self.arr[2:, 2:]
#         else:  # full core
#             self.triangle_option = "rhomb-120"
#             self.bcs = np.array([[2, 2], [2, 2], [2, 2]])
#             self.reg_assignments = self.arr
#         self.reg_assignments = np.stack((self.reg_assignments,
#                                          self.reg_assignments))
#
#
# def test_region_list_simple_full(simple_full):
#     """Direct comparison of region assignment for simple cases
#     (full core)"""
#     g = GEODST_simple('full')
#     ans = np.array([3, 1, 1, 1, 1, 1, 1,
#                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
#     assert np.array_equal(simple_full.list_regions(g),
#                           np.vstack((ans, ans)))
#
#
# def test_region_list_char_full(core_full, g_full):
#     """Test region list characteristics for complex core objects
#     (full core case)"""
#     # Add assembly map to core_60 object
#     try:
#         core_full.asm_map = core_full.map_asm(g_full.reg_assignments[0])
#     except:
#         pytest.xfail("Failure in core_full.map_asm() method")
#     # Length of regions array should be equal to max assembly ID
#     test = core_full.list_regions(g_full)
#     assert all([len(test[l]) == np.max(core_full.asm_map)
#                 for l in range(0, len(test))])
#     # There should be no zeros in any of the regions arrays
#     assert all([np.count_nonzero(test[l]) == len(test[l])
#                 for l in range(0, len(test))])
#
#
# @pytest.fixture
# def g_60(testdir):
#     return py4c.geodst.GEODST(os.path.join(testdir,
#                                            'test_data', 'GEODST_60'))
#
#
# @pytest.fixture
# def g_120(testdir):
#     return py4c.geodst.GEODST(os.path.join(testdir,
#                                            'test_data', 'GEODST_120'))
#
#
# @pytest.fixture
# def core_60(g_60, coolant):
#     """DASSH Core object from py4c GEODST obj with 60-degree
#     periodic region assignment"""
#     # asm_params = {'asm_pitch': 12.0,
#     #               'duct_ftf': 11.75,
#     #               'n_pin_ring': 9,
#     #               'pin_pitch': 0.7}
#     return dassh.Core(g_60, 0.05, coolant)
#
#
# @pytest.fixture
# def core_120(g_120, coolant):
#     """DASSH Core object from py4c GEODST obj with 120-degree
#     periodic region assignment"""
#     # asm_params = {'asm_pitch': 12.0,
#     #               'duct_ftf': 11.75,
#     #               'n_pin_ring': 9,
#     #               'pin_pitch': 0.7}
#     return dassh.Core(g_120, 0.05, coolant)
#
#
# @pytest.fixture(scope="module")
# def simple_120_test(coolant):
#     """Simple 120-degree periodic region assignment array"""
#     g = GEODST_simple('120')
#     # asm_params = {'asm_pitch': 12.0,
#     #               'duct_ftf': 11.75,
#     #               'n_pin_ring': 9,
#     #               'pin_pitch': 0.7}
#     return dassh.Core(g, 0.05, coolant, test=True)
#
#
# @pytest.fixture
# def simple_120(simple_120_test):
#     """Simple 120-deg-core region assignment array with all attributes;
#     calling in this way ensures that all __init__ methods will work
#     and that if they don't we get a more meaningful output"""
#     g = GEODST_simple('120').reg_assignments[0]
#     try:
#         simple_120_test.asm_map = simple_120_test.map_asm(g)
#     except:
#         print("Failure in map_asm() method")
#         raise
#     try:
#         simple_120_test.asm_adj = \
#             simple_120_test.map_adjacent_assemblies()
#     except:
#         print("Failure in map_adjacent_assemblies() method")
#         raise
#     return simple_120_test
#
#
# @pytest.fixture(scope="module")
# def simple_60_test(coolant):
#     """Simple 60-degree periodic region assignment array"""
#     g = GEODST_simple('60')
#     # asm_params = {'asm_pitch': 12.0,
#     #               'duct_ftf': 11.75,
#     #               'n_pin_ring': 9,
#     #               'pin_pitch': 0.7}
#     return dassh.Core(g, 0.05, coolant, test=True)
#
#
# @pytest.fixture
# def simple_60(simple_60_test):
#     """Simple 60-deg-core region assignment array with all attributes;
#     calling in this way ensures that all __init__ methods will work
#     and that if they don't we get a more meaningful output"""
#     g = GEODST_simple('60').reg_assignments[0]
#     try:
#         simple_60_test.asm_map = simple_60_test.map_asm(g)
#     except:
#         print("Failure in map_asm() method")
#         raise
#     try:
#         simple_60_test.asm_adj = \
#             simple_60_test.map_adjacent_assemblies()
#     except:
#         print("Failure in map_adjacent_assemblies() method")
#         raise
#     return simple_60_test
#
#
# # @pytest.mark.skip()
# # def test_id_periodicity_simple(simple_60_test, simple_120_test,
# #                                simple_full_test):
# def test_id_periodicity_simple(simple_full_test):
#     """Test the identification of periodicity for the simples cases"""
#     # assert simple_60_test.hex_option == 2
#     # assert simple_120_test.hex_option == 1
#     assert simple_full_test.hex_option == 0
#
#
# # @pytest.mark.skip()
# # def test_id_periodicity(core_60, core_120, core_full):
# def test_id_periodicity(core_full):
#     """Test the identification of periodicity for the simples cases"""
#     # assert core_60.hex_option == 2
#     # assert core_120.hex_option == 1
#     assert core_full.hex_option == 0
#
#
# @pytest.mark.skip()
# def test_asm_map_maxID(core_60, core_120, core_full, g_60, g_120, g_full):
#     """Test calculation of asm map for the GEODST cores"""
#     # Maximum asm ID value between different periodicity
#     map60 = core_60.map_asm(g_60.reg_assignments[0])
#     map120 = core_120.map_asm(g_120.reg_assignments[0])
#     map_full = core_full.map_asm(g_full.reg_assignments[0])
#     assert (2 * (np.max(map60) - 1) + 1 == np.max(map120))
#     assert (6 * (np.max(map60) - 1) + 1 == np.max(map_full))
#
#
# @pytest.mark.skip()
# def test_asm_map_proj60(core_60, core_full, g_60, g_full):
#     """Projection of 60-degree map onto full map should mask zeros"""
#     map60 = core_60.map_asm(g_60.reg_assignments[0])
#     map_full = core_full.map_asm(g_full.reg_assignments[0])
#     temp = map_full[0:core_full.n_ring, (core_full.n_ring - 1):]
#     assert temp.shape == map60.shape
#     assert all([x[-1, 0] == 1 for x in [temp, map60]])
#     assert (np.count_nonzero(temp) == np.count_nonzero(map60) + 10)
#
#
# @pytest.mark.skip()
# def test_asm_map_proj120(core_120, core_full, g_120, g_full):
#     """Projection of 120-degree map onto full map should mask zeros"""
#     map120 = core_120.map_asm(g_120.reg_assignments[0])
#     map_full = core_full.map_asm(g_full.reg_assignments[0])
#     temp = map_full[(core_full.n_ring - 1):, (core_full.n_ring - 1):]
#     assert temp.shape == map120.shape
#     assert all([x[0, 0] == 1 for x in [temp, map120]])
#     assert (np.count_nonzero(temp) == np.count_nonzero(map120) + 10)
#
#
# @pytest.mark.skip()
# def test_neighbors_simple_60(simple_60):
#     """Direct comparison of simple case neighbors map
#     (60 degree periodic)"""
#     assert np.array_equal(simple_60.map_adjacent_assemblies(),
#                           np.array([[2, 2, 2, 2, 2, 2],
#                                     [3, 4, 2, 1, 2, 4],
#                                     [0, 0, 4, 2, 4, 0],
#                                     [0, 0, 3, 2, 2, 3]]))
#
#
# @pytest.mark.skip()
# def test_neighbors_simple_120(simple_120):
#     """Direct comparison of simple case neighbors map
#     (120 degree periodic)"""
#     assert np.array_equal(simple_120.map_adjacent_assemblies(),
#                           np.array([[2, 3, 2, 3, 2, 3],
#                                     [4, 5, 3, 1, 3, 7],
#                                     [5, 6, 7, 2, 1, 2],
#                                     [0, 0, 5, 2, 7, 0],
#                                     [0, 0, 6, 3, 2, 4],
#                                     [0, 0, 0, 7, 3, 5],
#                                     [6, 0, 0, 4, 2, 3]]))
#
#
# @pytest.mark.skip()
# def test_region_list_simple_60(simple_60):
#     """Direct comparison of region assignment for simple cases
#     (60 degree periodic)"""
#     # Add assembly map to simple core object
#     g = GEODST_simple('60')
#     ans = np.array([3, 1, 2, 2])
#     assert np.array_equal(simple_60.list_regions(g),
#                           np.vstack((ans, ans)))
#
#
# @pytest.mark.skip()
# def test_region_list_simple_120(simple_120):
#     """Direct comparison of region assignment for simple cases
#     (120 degree periodic)"""
#     # Add assembly map to simple core object
#     g = GEODST_simple('120')
#     ans = np.array([3, 1, 1, 2, 2, 2, 2])
#     assert np.array_equal(simple_120.list_regions(g),
#                           np.vstack((ans, ans)))
#
#
#
#
# @pytest.mark.skip()
# def test_region_list_char_60(core_60, g_60):
#     """Test region list characteristics for complex core objects
#     (60 degree periodicity)"""
#     # Add assembly map to core_60 object
#     try:
#         core_60.asm_map = core_60.map_asm(g_60.reg_assignments[0])
#     except:
#         pytest.xfail("Failure in core_60.map_asm() method")
#     # Length of regions array should be equal to max assembly ID
#     test = core_60.list_regions(g_60)
#     assert all([len(test[l]) == np.max(core_60.asm_map)
#                 for l in range(0, len(test))])
#     # There should be no zeros in any of the regions arrays
#     assert all([np.count_nonzero(test[l]) == len(test[l])
#                 for l in range(0, len(test))])
#
#
# @pytest.mark.skip()
# def test_region_list_char_120(core_120, g_120):
#     """Test region list characteristics for complex core objects
#     (120 degree periodicity)"""
#     # Add assembly map to core_60 object
#     try:
#         core_120.asm_map = core_120.map_asm(g_120.reg_assignments[0])
#     except:
#         pytest.xfail("Failure in core_120.map_asm() method")
#     # Length of regions array should be equal to max assembly ID
#     test = core_120.list_regions(g_120)
#     assert all([len(test[l]) == np.max(core_120.asm_map)
#                 for l in range(0, len(test))])
#     # There should be no zeros in any of the regions arrays
#     assert all([np.count_nonzero(test[l]) == len(test[l])
#                 for l in range(0, len(test))])
#
#
#
# @pytest.mark.skip()
# def test_ia_gap_simple60(simple_60):
#     """Test the definition of interassembly gap subchannels for the
#     simple 60 degree periodic case"""
#     # side-subchannels-per-side = 1
#     ans = np.array([[[1, 2], [3, 4], [5, 6],
#                      [7, 8], [9, 10], [11, 12]],
#                     [[13, 14], [15, 16], [17, 2],
#                      [1, 12], [18, 19], [20, 21]],
#                     [[22, 23], [24, 25], [26, 14],
#                      [13, 21], [27, 28], [29, 30]],
#                     [[31, 32], [33, 34], [35, 36],
#                      [37, 16], [15, 14], [26, 25]]])
#     simple_60._sc_per_asm = 12
#     simple_60._sc_per_side = 1
#     test = np.stack(simple_60.map_interassembly_sc())
#     assert np.array_equal(ans, test)
#
#     # side-subchannels-per-side = 2
#     ans = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9],
#                      [10, 11, 12], [13, 14, 15], [16, 17, 18]],
#                     [[19, 20, 21], [22, 23, 24], [25, 26, 3],
#                      [2, 1, 18], [27, 28, 29], [30, 31, 32]],
#                     [[33, 34, 35], [36, 37, 38], [39, 40, 21],
#                      [20, 19, 32], [41, 42, 43], [44, 45, 46]],
#                     [[47, 48, 49], [50, 51, 52], [53, 54, 55],
#                      [56, 57, 24], [23, 22, 21], [40, 39, 38]]])
#     simple_60._sc_per_asm = 18
#     simple_60._sc_per_side = 2
#     test = np.stack(simple_60.map_interassembly_sc())
#     assert np.array_equal(ans, test)
#
#
# @pytest.mark.skip()
# def test_ia_gap_simple120(simple_120):
#     """Test the definition of interassembly gap subchannels for the
#     simple 120 degree periodic case"""
#     # side-subchannels-per-side = 1
#     ans = np.array([[[1, 2], [3, 4], [5, 6],
#                      [7, 8], [9, 10], [11, 12]],
#                     [[13, 14], [15, 16], [17, 2],
#                      [1, 12], [18, 19], [20, 21]],
#                     [[22, 23], [24, 25], [26, 27],
#                      [28, 4], [3, 2], [17, 16]],
#                     [[29, 30], [31, 32], [33, 14],
#                      [13, 21], [34, 35], [36, 37]],
#                     [[38, 39], [40, 41], [42, 23],
#                      [22, 16], [15, 14], [33, 32]],
#                     [[43, 44], [45, 46], [47, 48],
#                      [49, 25], [24, 23], [42, 41]],
#                     [[49, 48], [50, 51], [52, 53],
#                      [54, 55], [56, 27], [26, 25]]])
#     simple_120._sc_per_asm = 12
#     simple_120._sc_per_side = 1
#     test = np.stack(simple_120.map_interassembly_sc())
#     # side-subchannels-per-side = 2
#     assert np.array_equal(ans, test)
#
#     ans = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9],
#                      [10, 11, 12], [13, 14, 15], [16, 17, 18]],
#                     [[19, 20, 21], [22, 23, 24], [25, 26, 3],
#                      [2, 1, 18], [27, 28, 29], [30, 31, 32]],
#                     [[33, 34, 35], [36, 37, 38], [39, 40, 41],
#                      [42, 43, 6], [5, 4, 3], [26, 25, 24]],
#                     [[44, 45, 46], [47, 48, 49], [50, 51, 21],
#                      [20, 19, 32], [52, 53, 54], [55, 56, 57]],
#                     [[58, 59, 60], [61, 62, 63], [64, 65, 35],
#                      [34, 33, 24], [23, 22, 21], [51, 50, 49]],
#                     [[66, 67, 68], [69, 70, 71], [72, 73, 74],
#                      [75, 76, 38], [37, 36, 35], [65, 64, 63]],
#                     [[76, 75, 74], [77, 78, 79], [80, 81, 82],
#                      [83, 84, 85], [86, 87, 41], [40, 39, 38]]])
#     simple_120._sc_per_asm = 18
#     simple_120._sc_per_side = 2
#     test = np.stack(simple_120.map_interassembly_sc())
#     assert np.array_equal(ans, test)
#
#
#
# @pytest.mark.skip()
# def test_interasm_gap_sc_types_60(simple_60):
#     """Test the type assignment of interassembly gap subchannels"""
#     # problem setup
#     # side-subchannels-per-side = 2
#     simple_60._sc_per_asm = 18
#     simple_60._sc_per_side = 2
#     simple_60.asm_sc_adj = simple_60.map_interassembly_sc()
#     sc_types = simple_60.determine_sc_types()
#     corners = np.array([3, 6, 9, 12, 15, 18, 21, 24, 29,
#                         32, 35, 38, 43, 46, 49, 52, 55])
#     for sci in range(len(sc_types)):
#         if sci + 1 in corners:
#             assert sc_types[sci] == 2, 'sci: ' + str(sci)
#         else:
#             assert sc_types[sci] == 1, 'sci: ' + str(sci)
#
#
# @pytest.mark.skip()
# def test_interasm_gap_sc_types_120(simple_120):
#     """Test the type assignment of interassembly gap subchannels"""
#     # problem setup
#     # side-subchannels-per-side = 3
#     simple_120._sc_per_asm = 24
#     simple_120._sc_per_side = 3
#     simple_120.asm_sc_adj = simple_120.map_interassembly_sc()
#     sc_types = simple_120.determine_sc_types()
#     corners = np.array([4, 8, 12, 16, 20, 24, 28, 32, 39,
#                         43, 47, 51, 55, 62, 66, 73, 77, 81, 85,
#                         92, 96, 100, 107, 111, 115])
#     for sci in range(len(sc_types)):
#         if sci + 1 in corners:
#             assert sc_types[sci] == 2, 'sci: ' + str(sci)
#         else:
#             assert sc_types[sci] == 1, 'sci: ' + str(sci)
#
#
#
# @pytest.mark.skip()
# def test_interasm_gap_sc_adjacency_60(simple_60):
#     """Test the adjacency mapping of interassembly gap subchannels"""
#     # side-subchannels-per-side = 3
#     simple_60._sc_per_asm = 18
#     simple_60._sc_per_side = 2
#     simple_60.asm_sc_adj = simple_60.map_interassembly_sc()
#     sc_adj = simple_60.find_adjacent_sc()
#     ans = np.array([[18, 2, 0], [1, 3, 0], [2, 4, 26],
#                     [3, 5, 0], [4, 6, 0], [5, 7, 0],
#                     [6, 8, 0], [7, 9, 0], [8, 10, 0],
#                     [9, 11, 0], [10, 12, 0], [11, 13, 0],
#                     [12, 14, 0], [13, 15, 0], [14, 16, 0],
#                     [15, 17, 0], [16, 18, 0], [17, 1, 27],
#                     [32, 20, 0], [19, 21, 0], [20, 22, 40],
#                     [21, 23, 0], [22, 24, 0], [23, 25, 57],
#                     [24, 26, 0], [25, 3, 0], [18, 28, 0],
#                     [27, 29, 0], [28, 30, 0], [29, 31, 0],
#                     [30, 32, 0], [31, 19, 41], [46, 34, 0],
#                     [33, 35, 0], [34, 36, 0], [35, 37, 0],
#                     [36, 38, 0], [37, 39, 47], [38, 40, 0],
#                     [39, 21, 0], [32, 42, 0], [41, 43, 0],
#                     [42, 44, 0], [43, 45, 0], [44, 46, 0],
#                     [45, 33, 0], [38, 48, 0], [47, 49, 0],
#                     [48, 50, 0], [49, 51, 0], [50, 52, 0],
#                     [51, 53, 0], [52, 54, 0], [53, 55, 0],
#                     [54, 56, 0], [55, 57, 0], [56, 24, 0]])
#     for sc in range(len(sc_adj)):
#         if not np.all([x in sc_adj[sc] for x in ans[sc]]):
#             print(sc + 1, sc_adj[sc], ans[sc])
#             assert 0
#
#
# @pytest.mark.skip()
# def test_interasm_gap_sc_adjacency_120(simple_120):
#     """Test the adjacency mapping of interassembly gap subchannels"""
#     # side-subchannels-per-side = 3
#     simple_120._sc_per_asm = 24
#     simple_120._sc_per_side = 3
#     simple_120.asm_sc_adj = simple_120.map_interassembly_sc()
#     sc_adj = simple_120.find_adjacent_sc()
#     ans = np.array([[24, 2, 0], [1, 3, 0], [2, 4, 0],
#                     [3, 5, 35], [4, 6, 0], [5, 7, 0],
#                     [6, 8, 0], [7, 9, 58], [8, 10, 0],
#                     [9, 11, 0], [10, 12, 0], [11, 13, 0],
#                     [12, 14, 0], [13, 15, 0], [14, 16, 0],
#                     [15, 17, 0], [16, 18, 0], [17, 19, 0],
#                     [18, 20, 0], [19, 21, 0], [20, 22, 0],
#                     [21, 23, 0], [22, 24, 0], [23, 1, 36],
#                     [43, 26, 0], [25, 27, 0], [26, 28, 0],
#                     [27, 29, 69], [28, 30, 0], [29, 31, 0],
#                     [30, 32, 0], [31, 33, 44], [32, 34, 0],
#                     [33, 35, 0], [34, 4, 0], [24, 37, 0],
#                     [36, 38, 0], [37, 39, 0], [38, 40, 0],
#                     [39, 41, 0], [40, 42, 0], [41, 43, 0],
#                     [25, 42, 70], [32, 45, 0], [44, 46, 0],
#                     [45, 47, 0], [46, 48, 88], [47, 49, 0],
#                     [48, 50, 0], [49, 51, 0], [50, 52, 103],
#                     [51, 53, 0], [52, 54, 0], [53, 55, 0],
#                     [54, 56, 118], [55, 57, 0], [56, 58, 0],
#                     [57, 8, 0], [77, 60, 0], [59, 61, 0],
#                     [60, 62, 0], [61, 63, 0], [62, 64, 0],
#                     [63, 65, 0], [64, 66, 0], [65, 67, 78],
#                     [66, 68, 0], [67, 69, 0], [68, 28, 0],
#                     [43, 71, 0], [70, 72, 0], [71, 73, 0],
#                     [72, 74, 0], [73, 75, 0], [74, 76, 0],
#                     [75, 77, 0], [76, 59, 0], [66, 79, 0],
#                     [78, 80, 0], [79, 81, 0], [80, 82, 0],
#                     [81, 83, 0], [82, 84, 0], [83, 85, 0],
#                     [84, 86, 89], [85, 87, 0], [86, 88, 0],
#                     [87, 47, 0], [85, 90, 0], [89, 91, 0],
#                     [90, 92, 0], [91, 93, 0], [92, 94, 0],
#                     [93, 95, 0], [94, 96, 0], [95, 97, 0],
#                     [96, 98, 0], [97, 99, 0], [98, 100, 0],
#                     [99, 101, 104], [100, 102, 0], [101, 103, 0],
#                     [102, 51, 0], [100, 105, 0], [104, 106, 0],
#                     [105, 107, 0], [106, 108, 0], [107, 109, 0],
#                     [108, 110, 0], [109, 111, 0], [110, 112, 0],
#                     [111, 113, 0], [112, 114, 0], [113, 115, 0],
#                     [114, 116, 0], [115, 117, 0], [116, 118, 0],
#                     [117, 55, 0]])
#     for sc in range(len(sc_adj)):
#         if not np.all([x in sc_adj[sc] for x in ans[sc]]):
#             print(sc + 1, sc_adj[sc], ans[sc])
#             assert 0
#
#
#
# @pytest.fixture
# def core_full(g_full, coolant):
#     """DASSH Core object from py4c GEODST obj with full-core
#     region assignment"""
#
#     return dassh.Core(g_full, 0.05, coolant)
#
#
# ########################################################################
# # Core objects created from simple, mock GEODST files
# ########################################################################
#
#
# @pytest.fixture(scope="module")
# def simple_full_test(coolant):
#     """Simple full-core region assignment array - 19 asm core"""
#     asm_list = build_asm_list(3)
#     asm_pitch = 12.0
#     gap_fr = 0.05
#     return dassh.core(asm_list, asm_pitch, gap_fr, coolant, test=True)
#
#
# @pytest.fixture
# def simple_full(simple_full_test):
#     """Simple full-core region assignment array with all attributes;
#     calling in this way ensures that all __init__ methods will work
#     and that if they don't we get a more meaningful output"""
#     asm_list = build_asm_list(3)
#     try:
#         simple_full_test.asm_map = core.map_asm(asm_list)
#     except:
#         print("Failure in map_asm() method")
#         raise
#     try:
#         simple_full_test.asm_adj = \
#             core.map_adjacent_assemblies(simple_full_test.asm_adj)
#     except:
#         print("Failure in map_adjacent_assemblies() method")
#         raise
#     return simple_full_test
#
#
# def test_calc_nring(core_full):
#     """Test calculation of asm ring number from GEODST obj"""
#     # assert all([x.n_ring == 13 for x in [core_60, core_120, core_full]])
#     assert all([x.n_ring == 13 for x in [core_full]])
#
# def test_asm_map_simple(core_19a_test):
#     """Test calculation of simple asm map"""
#     # g = GEODST_simple('60').reg_assignments[0]
#     # assert np.array_equal(simple_60_test.map_asm(g),
#     #                       np.array([[3, 0, 0], [2, 4, 0], [1, 0, 0]]))
#     # g = GEODST_simple('120').reg_assignments[0]
#     # assert np.array_equal(simple_120_test.map_asm(g),
#     #                       np.array([[1, 2, 4], [0, 3, 5], [0, 7, 6]]))
#     g = GEODST_simple('full').reg_assignments[0]
#     assert np.array_equal(core_19a_incremental.map_asm(g),
#                           np.array([[8, 9, 10, 0, 0],
#                                     [19, 2, 3, 11, 0],
#                                     [18, 7, 1, 4, 12],
#                                     [0, 17, 6, 5, 13],
#                                     [0, 0, 16, 15, 14]]))
