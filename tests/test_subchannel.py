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
date: 2020-08-14
author: matz
Test the subchannel.py module and the Subchannel object
"""
########################################################################
import numpy as np
import pytest
import dassh
import sys
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=500)


@pytest.fixture(scope="module")
def sc_xy_2ring():
    """Generate empty array for subchannel XY coords from 2-ring asm"""
    # the size is for the coolant channels
    return np.zeros((18, 2))


########################################################################
# SUBCHANNEL INSTANTIATION
########################################################################


def test_sc_instantiation(sc_5ring):
    """Test the attributes of the 5-ring subchannel object"""
    assert sc_5ring.n_sc['coolant']['interior'] == 96
    assert sc_5ring.n_sc['coolant']['edge'] == 24
    assert sc_5ring.n_sc['coolant']['corner'] == 6
    assert sc_5ring.n_sc['coolant']['total'] == 126
    assert sc_5ring.n_sc['duct']['total'] == 30


########################################################################
# SUBCHANNEL TYPES
########################################################################


def test_sc_typing_direct(sc_2ring, sc_2ring_args):
    """Test that the subchannel type list is as expected"""
    test2 = sc_2ring.setup_sc_type(2, sc_2ring_args['duct_ftf'])
    assert np.array_equal(test2, np.array([1, 1, 1, 1, 1, 1,
                                           2, 3, 2, 3, 2, 3,
                                           2, 3, 2, 3, 2, 3,
                                           4, 5, 4, 5, 4, 5,
                                           4, 5, 4, 5, 4, 5]))


def test_sc_typing_indirect(sc_5ring, sc_5ring_args):
    """Test that the subchannel type list is as expected"""
    test5 = sc_5ring.setup_sc_type(5, sc_5ring_args['duct_ftf'])
    # Indirect test on long list; direct test for short list
    assert len(test5) == sc_5ring.n_sc['total']
    assert (len(test5[test5 == 1])
            == sc_5ring.n_sc['coolant']['interior'])
    assert (len(test5[test5 == 2])
            == sc_5ring.n_sc['coolant']['edge'])
    assert len(test5[test5 == 3]) == 6
    assert (len(test5[test5 == 4])
            == sc_5ring.n_sc['duct']['total'] - 6)
    assert len(test5[test5 == 5]) == 6


########################################################################
# SUBCHANNEL MAP
########################################################################
# def test_sc_map_direct(sc_5ring_args):
#     """."""
#     sc_obj = dassh.Subchannel(5,
#                               sc_5ring_args['pitch'],
#                               sc_5ring_args['d_pin'],
#                               sc_5ring_args['pin_map'],
#                               sc_5ring_args['pin_xy'],
#                               sc_5ring_args['duct_ftf'])
#     print(sc_obj._map)
#     print(sc_obj.sc_adj)
#     assert 1 == 2
# [[  0   0   0   0   0 126   0   0   0   0]
#  [  0   0   0   0 125  97   0   0   0   0]
#  [  0   0   0 124  96  55  98   0   0   0]
#  [  0   0 123  94  95  56  57  99   0   0]
#  [121 122  92  93  54  25  58  59 100   0]
#  [  0  90  91  52  53  26  27  60  61 101]
#  [120  89  50  51  24   7  28  29  62 102]
#  [  0  88  49  22  23   8   9  30  63   0]
#  [119  87  48  21   6   1  10  31  64 103]
#  [  0  86  47  20   5   2  11  32  65   0]
#  [118  85  46  19   4   3  12  33  66 104]
#  [  0  84  45  18  17  14  13  34  67   0]
#  [117  83  44  43  16  15  36  35  68 105]
#  [116  82  81  42  41  38  37  70  69   0]
#  [  0 115  80  79  40  39  72  71 107 106]
#  [  0   0 114  78  77  74  73 108   0   0]
#  [  0   0   0 113  76  75 109   0   0   0]
#  [  0   0   0   0 112 110   0   0   0   0]
#  [  0   0   0   0 111   0   0   0   0   0]]
    # ans = np.array([[0, 0, 0, 96, 55, 0, 0, 0]])


def test_sc_interior_map_indirect(sc_5ring_type):
    """Indirect tests for the map of interior subchannels"""
    n_ring = 5
    test5 = sc_5ring_type._make_interior_sc_map(n_ring)
    assert test5.shape == (4 * n_ring - 1, 2 * n_ring)
    # Location in array of min and max pin ID
    assert (np.where(test5 == 1)[0][0] == (test5.shape[0] - 1) / 2. - 1)
    assert (np.where(test5 == 1)[1][0] == (test5.shape[1] / 2))
    assert np.where(test5 == 96)[0][0] == 2  # row
    assert (np.where(test5 == 96)[1][0] == (test5.shape[1] / 2) - 1)
    # First two rows, last two rows are zeros
    assert np.array_equal(test5[0], np.zeros((2 * n_ring)))
    assert np.array_equal(test5[-1], np.zeros((2 * n_ring)))
    # First col, last col are zeros
    assert np.array_equal(test5[:, 0], np.zeros((4 * n_ring - 1)))
    assert np.array_equal(test5[:, -1], np.zeros((4 * n_ring - 1)))


def test_sc_interior_map_direct(sc_2ring_type):
    """Direct test for the map of interior subchannels"""
    n_ring = 2
    test2 = sc_2ring_type._make_interior_sc_map(n_ring)
    assert np.array_equal(test2, np.array([[0, 0, 0, 0],
                                           [0, 0, 0, 0],
                                           [0, 6, 1, 0],
                                           [0, 5, 2, 0],
                                           [0, 4, 3, 0],
                                           [0, 0, 0, 0],
                                           [0, 0, 0, 0]]))


def test_sc_exterior_map_indirect(sc_5ring_type):
    """Indirect test on map of edge and corner coolant subchannels"""
    n_ring = 5
    test5 = sc_5ring_type._make_exterior_sc_map(n_ring)
    # Top and bottom rows of array
    row_1 = np.zeros(10)
    row_1[5] = max(np.where(sc_5ring_type.type == 3)[0]) + 1
    print(np.where(sc_5ring_type.type == 3)[0])
    print()
    print(row_1)
    print()
    print(test5)
    assert np.array_equal(test5[0], row_1)
    row_n = np.zeros(10)
    row_n[4] = max(row_1) - 3 * n_ring
    assert np.array_equal(test5[-1], row_n)


def test_sc_exterior_map_direct(sc_2ring_type):
    """Direct test on map of edge and corner coolant subchannels"""
    n_ring = 2
    test2 = sc_2ring_type._make_exterior_sc_map(n_ring)
    print(test2)
    assert np.array_equal(test2, np.array([[0, 0, 18, 0],
                                           [16, 17, 7, 0],
                                           [0, 0, 0, 8],
                                           [15, 0, 0, 9],
                                           [14, 0, 0, 0],
                                           [0, 13, 11, 10],
                                           [0, 12, 0, 0]]))


def test_sc_move_method(sc_2ring):
    """Test the Subchannel class matrix walk method _move"""
    loc = (0, 1)
    assert sc_2ring._move(loc, 'down') == (1, 1)
    assert sc_2ring._move(loc, 'up') == (-1, 1)
    assert sc_2ring._move(loc, 'right') == (0, 2)
    assert sc_2ring._move(loc, 'left') == (0, 0)
    with pytest.raises(ValueError):
        sc_2ring._move(loc, 'over there')


def test_sc_step_method(sc_2ring):
    """Test the subchannel class matrix walk method"""
    test_mat = np.zeros((2, 2))
    idx = 0
    loc = (0, 0)
    loc, test_mat, idx = sc_2ring._step(loc, 'right', test_mat, idx)
    loc, test_mat, idx = sc_2ring._step(loc, 'down', test_mat, idx)
    loc, test_mat, idx = sc_2ring._step(loc, 'left', test_mat, idx)
    loc, test_mat, idx = sc_2ring._step(loc, 'up', test_mat, idx)
    assert np.array_equal(test_mat, np.array([[4, 1], [3, 2]]))


########################################################################
# SUBCHANNEL NEIGHBORS
# To do (3):
# - test_interior_exterior_sc_connection():
# - test_exterior_sc_connection():
# - test_interior_exterior_sc_connection():
########################################################################
def test_sc_sc_neighbors(sc_2ring_map, sc_2ring_args):
    """Direct test the neighbors between subchannels"""
    n_ring = 2
    ans = np.array([[0, 2, 6, 7, 0, 0, 0],
                    [1, 3, 0, 9, 0, 0, 0],
                    [2, 0, 4, 11, 0, 0, 0],
                    [5, 0, 3, 13, 0, 0, 0],
                    [6, 4, 0, 15, 0, 0, 0],
                    [0, 5, 1, 17, 0, 0, 0],
                    [1, 0, 0, 18, 8, 19, 0],
                    [0, 0, 0, 7, 9, 20, 0],
                    [2, 0, 0, 8, 10, 21, 0],
                    [0, 0, 0, 9, 11, 22, 0],
                    [3, 0, 0, 10, 12, 23, 0],
                    [0, 0, 0, 11, 13, 24, 0],
                    [4, 0, 0, 12, 14, 25, 0],
                    [0, 0, 0, 13, 15, 26, 0],
                    [5, 0, 0, 14, 16, 27, 0],
                    [0, 0, 0, 15, 17, 28, 0],
                    [6, 0, 0, 16, 18, 29, 0],
                    [0, 0, 0, 17, 7, 30, 0],
                    [0, 0, 0, 0, 7, 30, 20],
                    [0, 0, 0, 0, 8, 19, 21],
                    [0, 0, 0, 0, 9, 20, 22],
                    [0, 0, 0, 0, 10, 21, 23],
                    [0, 0, 0, 0, 11, 22, 24],
                    [0, 0, 0, 0, 12, 23, 25],
                    [0, 0, 0, 0, 13, 24, 26],
                    [0, 0, 0, 0, 14, 25, 27],
                    [0, 0, 0, 0, 15, 26, 28],
                    [0, 0, 0, 0, 16, 27, 29],
                    [0, 0, 0, 0, 17, 28, 30],
                    [0, 0, 0, 0, 18, 29, 19]])
    test2 = sc_2ring_map.find_sc_sc_neighbors(n_ring,
                                              sc_2ring_args['duct_ftf'])
    print(test2)
    print(np.diff(test2 - ans))
    assert np.array_equal(test2, ans)


def test_interior_sc_connections(sc_2ring_map):
    """Direct test on connections between interior coolant
    subchannel neighbors"""
    # direct comparison for 2-ring asm
    start = np.zeros((sc_2ring_map.n_sc['total'], 7), dtype="int")
    test_2ring = sc_2ring_map._connect_int_sc(start)
    ans = np.zeros((sc_2ring_map.n_sc['total'], 7), dtype="int")
    ans[0:6] = np.array([[0, 2, 6, 0, 0, 0, 0],
                         [1, 3, 0, 0, 0, 0, 0],
                         [2, 0, 4, 0, 0, 0, 0],
                         [5, 0, 3, 0, 0, 0, 0],
                         [6, 4, 0, 0, 0, 0, 0],
                         [0, 5, 1, 0, 0, 0, 0]])
    assert np.array_equal(test_2ring, ans)


def test_connect_int_sc(sc_5ring_map):
    empty = np.zeros((sc_5ring_map.n_sc['total'], 7), dtype="int")
    res = sc_5ring_map._connect_int_sc(empty)
    ans = np.array([[8, 2, 6, 0, 0, 0, 0],
                    [1, 3, 11, 0, 0, 0, 0],
                    [2, 14, 4, 0, 0, 0, 0],
                    [5, 17, 3, 0, 0, 0, 0],
                    [6, 4, 20, 0, 0, 0, 0],
                    [23, 5, 1, 0, 0, 0, 0],
                    [26, 8, 24, 0, 0, 0, 0],
                    [7, 1, 9, 0, 0, 0, 0],
                    [28, 10, 8, 0, 0, 0, 0],
                    [9, 11, 31, 0, 0, 0, 0],
                    [10, 12, 2, 0, 0, 0, 0],
                    [11, 13, 33, 0, 0, 0, 0],
                    [12, 36, 14, 0, 0, 0, 0],
                    [3, 15, 13, 0, 0, 0, 0],
                    [14, 38, 16, 0, 0, 0, 0],
                    [17, 41, 15, 0, 0, 0, 0],
                    [4, 16, 18, 0, 0, 0, 0],
                    [19, 43, 17, 0, 0, 0, 0],
                    [20, 18, 46, 0, 0, 0, 0],
                    [21, 19, 5, 0, 0, 0, 0],
                    [22, 20, 48, 0, 0, 0, 0],
                    [51, 21, 23, 0, 0, 0, 0],
                    [24, 6, 22, 0, 0, 0, 0],
                    [53, 23, 7, 0, 0, 0, 0],
                    [56, 26, 54, 0, 0, 0, 0],
                    [25, 7, 27, 0, 0, 0, 0],
                    [58, 28, 26, 0, 0, 0, 0],
                    [27, 9, 29, 0, 0, 0, 0],
                    [60, 30, 28, 0, 0, 0, 0],  # [62, 30, 28, 0, 0, 0, 0],
                    [29, 31, 63, 0, 0, 0, 0],
                    [30, 32, 10, 0, 0, 0, 0],  # [64, 32, 10, 0, 0, 0, 0],
                    [31, 33, 65, 0, 0, 0, 0],
                    [32, 34, 12, 0, 0, 0, 0],  # [66, 34, 12, 0, 0, 0, 0],
                    [33, 35, 67, 0, 0, 0, 0],
                    [34, 70, 36, 0, 0, 0, 0],  # [68, 70, 36, 0, 0, 0, 0],
                    [13, 37, 35, 0, 0, 0, 0],
                    [36, 72, 38, 0, 0, 0, 0],
                    [15, 39, 37, 0, 0, 0, 0],
                    [38, 74, 40, 0, 0, 0, 0],
                    [41, 77, 39, 0, 0, 0, 0],
                    [16, 40, 42, 0, 0, 0, 0],
                    [43, 79, 41, 0, 0, 0, 0],
                    [18, 42, 44, 0, 0, 0, 0],
                    [45, 81, 43, 0, 0, 0, 0],
                    [46, 44, 84, 0, 0, 0, 0],
                    [47, 45, 19, 0, 0, 0, 0],
                    [48, 46, 86, 0, 0, 0, 0],
                    [49, 47, 21, 0, 0, 0, 0],
                    [50, 48, 88, 0, 0, 0, 0],
                    [91, 49, 51, 0, 0, 0, 0],
                    [52, 22, 50, 0, 0, 0, 0],
                    [93, 51, 53, 0, 0, 0, 0],
                    [54, 24, 52, 0, 0, 0, 0],
                    [95, 53, 25, 0, 0, 0, 0],
                    [0, 56, 96, 0, 0, 0, 0],
                    [55, 25, 57, 0, 0, 0, 0],
                    [0, 58, 56, 0, 0, 0, 0],
                    [57, 27, 59, 0, 0, 0, 0],
                    [0, 60, 58, 0, 0, 0, 0],
                    [59, 29, 61, 0, 0, 0, 0],
                    [0, 62, 60, 0, 0, 0, 0],
                    [61, 63, 0, 0, 0, 0, 0],  # [61, 63, 0, 29, 0, 0, 0],
                    [62, 64, 30, 0, 0, 0, 0],
                    [63, 65, 0, 0, 0, 0, 0],  # [63, 65, 0, 31, 0, 0, 0],
                    [64, 66, 32, 0, 0, 0, 0],
                    [65, 67, 0, 0, 0, 0, 0],  # [65, 67, 0, 33, 0, 0, 0],
                    [66, 68, 34, 0, 0, 0, 0],
                    [67, 69, 0, 0, 0, 0, 0],  # [67, 69, 0, 35, 0, 0, 0],
                    [68, 0, 70, 0, 0, 0, 0],
                    [35, 71, 69, 0, 0, 0, 0],
                    [70, 0, 72, 0, 0, 0, 0],
                    [37, 73, 71, 0, 0, 0, 0],
                    [72, 0, 74, 0, 0, 0, 0],
                    [39, 75, 73, 0, 0, 0, 0],
                    [74, 0, 76, 0, 0, 0, 0],
                    [77, 0, 75, 0, 0, 0, 0],
                    [40, 76, 78, 0, 0, 0, 0],
                    [79, 0, 77, 0, 0, 0, 0],
                    [42, 78, 80, 0, 0, 0, 0],
                    [81, 0, 79, 0, 0, 0, 0],
                    [44, 80, 82, 0, 0, 0, 0],
                    [83, 0, 81, 0, 0, 0, 0],
                    [84, 82, 0, 0, 0, 0, 0],
                    [85, 83, 45, 0, 0, 0, 0],
                    [86, 84, 0, 0, 0, 0, 0],
                    [87, 85, 47, 0, 0, 0, 0],
                    [88, 86, 0, 0, 0, 0, 0],
                    [89, 87, 49, 0, 0, 0, 0],
                    [90, 88, 0, 0, 0, 0, 0],
                    [0, 89, 91, 0, 0, 0, 0],
                    [92, 50, 90, 0, 0, 0, 0],
                    [0, 91, 93, 0, 0, 0, 0],
                    [94, 52, 92, 0, 0, 0, 0],
                    [0, 93, 95, 0, 0, 0, 0],
                    [96, 54, 94, 0, 0, 0, 0],
                    [0, 95, 55, 0, 0, 0, 0]], dtype='int')
    for sc in range(sc_5ring_map.n_sc['coolant']['interior']):
        msg = '\n'.join(['sc (py idx): ' + str(sc),
                         'res: ' + str(res[sc]),
                         'ans: ' + str(ans[sc])])
        assert all([sca in res[sc] for sca in ans[sc]]), msg


def test_connect_int_ext_sc(sc_5ring_map):
    empty = np.zeros((sc_5ring_map.n_sc['total'], 7), dtype="int")
    res = sc_5ring_map._connect_int_sc(empty)
    res = sc_5ring_map._connect_int_ext_sc(res)
    ans = np.array([[8, 2, 6, 0, 0, 0, 0],
                    [1, 3, 11, 0, 0, 0, 0],
                    [2, 14, 4, 0, 0, 0, 0],
                    [5, 17, 3, 0, 0, 0, 0],
                    [6, 4, 20, 0, 0, 0, 0],
                    [23, 5, 1, 0, 0, 0, 0],
                    [26, 8, 24, 0, 0, 0, 0],
                    [7, 1, 9, 0, 0, 0, 0],
                    [28, 10, 8, 0, 0, 0, 0],
                    [9, 11, 31, 0, 0, 0, 0],
                    [10, 12, 2, 0, 0, 0, 0],
                    [11, 13, 33, 0, 0, 0, 0],
                    [12, 36, 14, 0, 0, 0, 0],
                    [3, 15, 13, 0, 0, 0, 0],
                    [14, 38, 16, 0, 0, 0, 0],
                    [17, 41, 15, 0, 0, 0, 0],
                    [4, 16, 18, 0, 0, 0, 0],
                    [19, 43, 17, 0, 0, 0, 0],
                    [20, 18, 46, 0, 0, 0, 0],
                    [21, 19, 5, 0, 0, 0, 0],
                    [22, 20, 48, 0, 0, 0, 0],
                    [51, 21, 23, 0, 0, 0, 0],
                    [24, 6, 22, 0, 0, 0, 0],
                    [53, 23, 7, 0, 0, 0, 0],
                    [56, 26, 54, 0, 0, 0, 0],
                    [25, 7, 27, 0, 0, 0, 0],
                    [58, 28, 26, 0, 0, 0, 0],
                    [27, 9, 29, 0, 0, 0, 0],
                    [60, 30, 28, 0, 0, 0, 0],  # [62, 30, 28, 0, 0, 0, 0],
                    [29, 31, 63, 0, 0, 0, 0],
                    [30, 32, 10, 0, 0, 0, 0],  # [64, 32, 10, 0, 0, 0, 0],
                    [31, 33, 65, 0, 0, 0, 0],
                    [32, 34, 12, 0, 0, 0, 0],  # [66, 34, 12, 0, 0, 0, 0],
                    [33, 35, 67, 0, 0, 0, 0],
                    [34, 70, 36, 0, 0, 0, 0],  # [68, 70, 36, 0, 0, 0, 0],
                    [13, 37, 35, 0, 0, 0, 0],
                    [36, 72, 38, 0, 0, 0, 0],
                    [15, 39, 37, 0, 0, 0, 0],
                    [38, 74, 40, 0, 0, 0, 0],
                    [41, 77, 39, 0, 0, 0, 0],
                    [16, 40, 42, 0, 0, 0, 0],
                    [43, 79, 41, 0, 0, 0, 0],
                    [18, 42, 44, 0, 0, 0, 0],
                    [45, 81, 43, 0, 0, 0, 0],
                    [46, 44, 84, 0, 0, 0, 0],
                    [47, 45, 19, 0, 0, 0, 0],
                    [48, 46, 86, 0, 0, 0, 0],
                    [49, 47, 21, 0, 0, 0, 0],
                    [50, 48, 88, 0, 0, 0, 0],
                    [91, 49, 51, 0, 0, 0, 0],
                    [52, 22, 50, 0, 0, 0, 0],
                    [93, 51, 53, 0, 0, 0, 0],
                    [54, 24, 52, 0, 0, 0, 0],
                    [95, 53, 25, 0, 0, 0, 0],
                    [0, 56, 96, 97, 0, 0, 0],
                    [55, 25, 57, 0, 0, 0, 0],
                    [0, 58, 56, 98, 0, 0, 0],
                    [57, 27, 59, 0, 0, 0, 0],
                    [0, 60, 58, 99, 0, 0, 0],
                    [59, 29, 61, 0, 0, 0, 0],
                    [0, 62, 60, 100, 0, 0, 0],
                    [61, 63, 0, 102, 0, 0, 0],  # [61, 63, 0, 29, 0, 0, 0],
                    [62, 64, 30, 0, 0, 0, 0],
                    [63, 65, 0, 103, 0, 0, 0],  # [63, 65, 0, 31, 0, 0, 0],
                    [64, 66, 32, 0, 0, 0, 0],
                    [65, 67, 0, 104, 0, 0, 0],  # [65, 67, 0, 33, 0, 0, 0],
                    [66, 68, 34, 0, 0, 0, 0],
                    [67, 69, 0, 105, 0, 0, 0],  # [67, 69, 0, 35, 0, 0, 0],
                    [68, 0, 70, 107, 0, 0, 0],
                    [35, 71, 69, 0, 0, 0, 0],
                    [70, 0, 72, 108, 0, 0, 0],
                    [37, 73, 71, 0, 0, 0, 0],
                    [72, 0, 74, 109, 0, 0, 0],
                    [39, 75, 73, 0, 0, 0, 0],
                    [74, 0, 76, 110, 0, 0, 0],
                    [77, 0, 75, 112, 0, 0, 0],
                    [40, 76, 78, 0, 0, 0, 0],
                    [79, 0, 77, 113, 0, 0, 0],
                    [42, 78, 80, 0, 0, 0, 0],
                    [81, 0, 79, 114, 0, 0, 0],
                    [44, 80, 82, 0, 0, 0, 0],
                    [83, 0, 81, 115, 0, 0, 0],
                    [84, 82, 0, 117, 0, 0, 0],
                    [85, 83, 45, 0, 0, 0, 0],
                    [86, 84, 0, 118, 0, 0, 0],
                    [87, 85, 47, 0, 0, 0, 0],
                    [88, 86, 0, 119, 0, 0, 0],
                    [89, 87, 49, 0, 0, 0, 0],
                    [90, 88, 0, 120, 0, 0, 0],
                    [0, 89, 91, 122, 0, 0, 0],
                    [92, 50, 90, 0, 0, 0, 0],
                    [0, 91, 93, 123, 0, 0, 0],
                    [94, 52, 92, 0, 0, 0, 0],
                    [0, 93, 95, 124, 0, 0, 0],
                    [96, 54, 94, 0, 0, 0, 0],
                    [0, 95, 55, 125, 0, 0, 0]], dtype='int')
    for sc in range(sc_5ring_map.n_sc['coolant']['interior']):
        msg = '\n'.join(['sc (py idx): ' + str(sc),
                         'res: ' + str(res[sc]),
                         'ans: ' + str(ans[sc])])
        assert all([sca in res[sc] for sca in ans[sc]]), msg


def test_sc_int_coolant_adj_5ring(sc_5ring_args):
    """."""
    sc_obj = dassh.Subchannel(5,
                              sc_5ring_args['pitch'],
                              sc_5ring_args['d_pin'],
                              sc_5ring_args['pin_map'],
                              sc_5ring_args['pin_xy'],
                              sc_5ring_args['duct_ftf'])
    ans = np.array([[8, 2, 6, 0, 0, 0, 0],
                    [1, 3, 11, 0, 0, 0, 0],
                    [2, 14, 4, 0, 0, 0, 0],
                    [5, 17, 3, 0, 0, 0, 0],
                    [6, 4, 20, 0, 0, 0, 0],
                    [23, 5, 1, 0, 0, 0, 0],
                    [26, 8, 24, 0, 0, 0, 0],
                    [7, 1, 9, 0, 0, 0, 0],
                    [28, 10, 8, 0, 0, 0, 0],
                    [9, 11, 31, 0, 0, 0, 0],
                    [10, 12, 2, 0, 0, 0, 0],
                    [11, 13, 33, 0, 0, 0, 0],
                    [12, 36, 14, 0, 0, 0, 0],
                    [3, 15, 13, 0, 0, 0, 0],
                    [14, 38, 16, 0, 0, 0, 0],
                    [17, 41, 15, 0, 0, 0, 0],
                    [4, 16, 18, 0, 0, 0, 0],
                    [19, 43, 17, 0, 0, 0, 0],
                    [20, 18, 46, 0, 0, 0, 0],
                    [21, 19, 5, 0, 0, 0, 0],
                    [22, 20, 48, 0, 0, 0, 0],
                    [51, 21, 23, 0, 0, 0, 0],
                    [24, 6, 22, 0, 0, 0, 0],
                    [53, 23, 7, 0, 0, 0, 0],
                    [56, 26, 54, 0, 0, 0, 0],
                    [25, 7, 27, 0, 0, 0, 0],
                    [58, 28, 26, 0, 0, 0, 0],
                    [27, 9, 29, 0, 0, 0, 0],
                    [60, 30, 28, 0, 0, 0, 0],  # [62, 30, 28, 0, 0, 0, 0],
                    [29, 31, 63, 0, 0, 0, 0],
                    [30, 32, 10, 0, 0, 0, 0],  # [64, 32, 10, 0, 0, 0, 0],
                    [31, 33, 65, 0, 0, 0, 0],
                    [32, 34, 12, 0, 0, 0, 0],  # [66, 34, 12, 0, 0, 0, 0],
                    [33, 35, 67, 0, 0, 0, 0],
                    [34, 70, 36, 0, 0, 0, 0],  # [68, 70, 36, 0, 0, 0, 0],
                    [13, 37, 35, 0, 0, 0, 0],
                    [36, 72, 38, 0, 0, 0, 0],
                    [15, 39, 37, 0, 0, 0, 0],
                    [38, 74, 40, 0, 0, 0, 0],
                    [41, 77, 39, 0, 0, 0, 0],
                    [16, 40, 42, 0, 0, 0, 0],
                    [43, 79, 41, 0, 0, 0, 0],
                    [18, 42, 44, 0, 0, 0, 0],
                    [45, 81, 43, 0, 0, 0, 0],
                    [46, 44, 84, 0, 0, 0, 0],
                    [47, 45, 19, 0, 0, 0, 0],
                    [48, 46, 86, 0, 0, 0, 0],
                    [49, 47, 21, 0, 0, 0, 0],
                    [50, 48, 88, 0, 0, 0, 0],
                    [91, 49, 51, 0, 0, 0, 0],
                    [52, 22, 50, 0, 0, 0, 0],
                    [93, 51, 53, 0, 0, 0, 0],
                    [54, 24, 52, 0, 0, 0, 0],
                    [95, 53, 25, 0, 0, 0, 0],
                    [0, 56, 96, 97, 0, 0, 0],
                    [55, 25, 57, 0, 0, 0, 0],
                    [0, 58, 56, 98, 0, 0, 0],
                    [57, 27, 59, 0, 0, 0, 0],
                    [0, 60, 58, 99, 0, 0, 0],
                    [59, 29, 61, 0, 0, 0, 0],
                    [0, 62, 60, 100, 0, 0, 0],
                    [61, 63, 0, 102, 0, 0, 0],  # [61, 63, 0, 29, 0, 0, 0],
                    [62, 64, 30, 0, 0, 0, 0],
                    [63, 65, 0, 103, 0, 0, 0],  # [63, 65, 0, 31, 0, 0, 0],
                    [64, 66, 32, 0, 0, 0, 0],
                    [65, 67, 0, 104, 0, 0, 0],  # [65, 67, 0, 33, 0, 0, 0],
                    [66, 68, 34, 0, 0, 0, 0],
                    [67, 69, 0, 105, 0, 0, 0],  # [67, 69, 0, 35, 0, 0, 0],
                    [68, 0, 70, 107, 0, 0, 0],
                    [35, 71, 69, 0, 0, 0, 0],
                    [70, 0, 72, 108, 0, 0, 0],
                    [37, 73, 71, 0, 0, 0, 0],
                    [72, 0, 74, 109, 0, 0, 0],
                    [39, 75, 73, 0, 0, 0, 0],
                    [74, 0, 76, 110, 0, 0, 0],
                    [77, 0, 75, 112, 0, 0, 0],
                    [40, 76, 78, 0, 0, 0, 0],
                    [79, 0, 77, 113, 0, 0, 0],
                    [42, 78, 80, 0, 0, 0, 0],
                    [81, 0, 79, 114, 0, 0, 0],
                    [44, 80, 82, 0, 0, 0, 0],
                    [83, 0, 81, 115, 0, 0, 0],
                    [84, 82, 0, 117, 0, 0, 0],
                    [85, 83, 45, 0, 0, 0, 0],
                    [86, 84, 0, 118, 0, 0, 0],
                    [87, 85, 47, 0, 0, 0, 0],
                    [88, 86, 0, 119, 0, 0, 0],
                    [89, 87, 49, 0, 0, 0, 0],
                    [90, 88, 0, 120, 0, 0, 0],
                    [0, 89, 91, 122, 0, 0, 0],
                    [92, 50, 90, 0, 0, 0, 0],
                    [0, 91, 93, 123, 0, 0, 0],
                    [94, 52, 92, 0, 0, 0, 0],
                    [0, 93, 95, 124, 0, 0, 0],
                    [96, 54, 94, 0, 0, 0, 0],
                    [0, 95, 55, 125, 0, 0, 0]], dtype='int')
    ans = ans - 1  # need to move to python indexing
    for sc in range(sc_obj.n_sc['coolant']['interior']):
        msg = '\n'.join(['sc (py idx): ' + str(sc),
                         'res: ' + str(sc_obj.sc_adj[sc]),
                         'ans: ' + str(ans[sc])])
        assert all([sca in sc_obj.sc_adj[sc] for sca in ans[sc]]), msg


# def test_interior_exterior_sc_connection():
#     """Test connection between interior and edge/corner sc neighbors"""
#     pass
#
#
# def test_exterior_sc_connection():
#     """Test connection between edge/corner sc neighbors"""
#     pass
#
#
# def test_interior_exterior_sc_connection():
#     """Test connection with duct and bypass neighbors"""
#     pass


########################################################################
# SUBCHANNEL-PIN NEIGHBORS
########################################################################


def test_pin_sc_neighbors(sc_2ring_map, pinlattice_2ring_map):
    """Test the connection between coolant sc and pins"""
    pin_map = pinlattice_2ring_map.map
    print(pin_map)
    print(len(pin_map[pin_map != 0]))
    x = sc_2ring_map.find_pin_sc_neighbors(2, pin_map)
    assert np.array_equal(x, np.array([[1, 2, 3, 4, 5, 6],
                                       [18, 7, 1, 6, 17, 0],
                                       [0, 8, 9, 2, 1, 7],
                                       [9, 0, 10, 11, 3, 2],
                                       [3, 11, 0, 12, 13, 4],
                                       [5, 4, 13, 0, 14, 15],
                                       [17, 6, 5, 15, 0, 16]]))


def test_reverse_pin_sc_neighbors(sc_2ring_pinadj):
    """Test that the inverse subchannel-pin adjacency is correct"""
    rev_pin_adj = sc_2ring_pinadj.reverse_pin_neighbors()
    # To compare with this partially constructed subchannel object,
    # we need to modify the pin adjacency attribute. At the end of
    # subchannel object instantiation, 1 is subtracted from every
    # entry in pin_adj to promote python indexing.
    tmp_pin_adj = sc_2ring_pinadj.pin_adj - 1
    print(len(rev_pin_adj))
    for sc in range(len(rev_pin_adj)):
        for pin in rev_pin_adj[sc]:
            if pin >= 0:
                if sc not in tmp_pin_adj[pin]:
                    print("sc =", sc, "pin =", pin)
                    print(rev_pin_adj[pin])
                    print(tmp_pin_adj.pin_adj[pin])
                    assert 0


########################################################################
# SUBCHANNEL XY COORDINATES
# To do (1):
# - get_sc_xy
########################################################################


def test_interior_sc_xy(sc_2ring_pinadj, sc_xy_2ring,
                        pinlattice_2ring_full, sc_2ring_args):
    """Test the XY coordinates of the interior coolant subchannels"""
    pitch = sc_2ring_args['pitch']
    pin_xy = pinlattice_2ring_full.xy
    ans = sc_xy_2ring
    ans[0:6] = np.array([[pitch / np.sqrt(3), 0.0],
                         [np.sqrt(pitch**2 + pitch**2 / 3), -0.5],
                         [pitch / np.sqrt(3), -0.5],
                         [-pitch / np.sqrt(3), 0.0],
                         [np.sqrt(pitch**2 + pitch**2 / 3), 0.5],
                         [np.sqrt(pitch**2 + pitch**2 / 3), 0.5]])
    assert np.allclose(ans,
                       sc_2ring_pinadj._find_interior_xy(sc_xy_2ring,
                                                         pin_xy, pitch))


def test_edge_xy(sc_2ring_pinadj, sc_xy_2ring,
                 pinlattice_2ring_full, sc_2ring_args):
    """Test XY coords for edge coolant subchannels"""
    # Note: below on line 2, 10.0 is the inner-most duct FTF distance
    n_ring = 2
    pitch = sc_2ring_args['pitch']
    duct_ftf = sc_2ring_args['duct_ftf'][0][0]
    pin_xy = pinlattice_2ring_full.xy
    d_edge = np.sqrt((0.5 * pitch)**2
                     + (0.25 * (10.0 - np.sqrt(3)
                        * (n_ring - 1) * pitch))**2)
    x_edge = np.cos(np.pi / 3.0) * d_edge
    y_edge = np.sin(np.pi / 3.0) * d_edge
    ans = sc_xy_2ring
    ans[6] = [x_edge, y_edge]
    ans[8] = [np.sqrt(3) * d_edge / 2, 0.0]
    ans[10] = [x_edge, -y_edge]
    ans[12] = [-x_edge, -y_edge]
    ans[14] = [-np.sqrt(3) * d_edge / 2, 0.0]
    ans[16] = [-x_edge, y_edge]
    assert np.allclose(ans,
                       sc_2ring_pinadj._find_edge_xy(sc_xy_2ring,
                                                     pin_xy, n_ring,
                                                     pitch, duct_ftf))


def test_corner_xy(sc_2ring_pinadj, sc_xy_2ring,
                   pinlattice_2ring_full, sc_2ring_args):
    """Test XY coords for corner coolant subchannels"""
    n_ring = 2
    pitch = sc_2ring_args['pitch']
    # Note: this value ---> = (10.0) is the inner-most duct FTF distance
    duct_ftf = sc_2ring_args['duct_ftf'][0][0]
    dpin = sc_2ring_args['d_pin']
    pin_xy = pinlattice_2ring_full.xy
    d_corner = 0.25 * (2 * duct_ftf / np.sqrt(3) + dpin
                       - 2 * (n_ring - 1) * pitch)
    x_corner = np.cos(np.pi / 6.0) * d_corner
    y_corner = np.sin(np.pi / 6.0) * d_corner
    ans = sc_xy_2ring
    ans[7] = [x_corner, y_corner]
    ans[9] = [x_corner, -y_corner]
    ans[11] = [0.0, pitch + d_corner]
    ans[13] = [-x_corner, -y_corner]
    ans[15] = [-x_corner, y_corner]
    ans[17] = [0.0, pitch + d_corner]
    assert np.allclose(ans,
                       sc_2ring_pinadj._find_corner_xy(sc_xy_2ring,
                                                       pin_xy, n_ring,
                                                       pitch, dpin,
                                                       duct_ftf))


def test_duct_xy(sc_2ring_pinadj, sc_xy_2ring,
                 pinlattice_2ring_full, sc_2ring_args):
    """Test methods that get duct XY coordinates"""
    n_ring = 2
    pitch = sc_2ring_args['pitch']
    duct_ftf = sc_2ring_args['duct_ftf']
    dpin = sc_2ring_args['d_pin']
    pin_xy = pinlattice_2ring_full.xy
    try:
        start = sc_2ring_pinadj._find_interior_xy(sc_xy_2ring,
                                                  pin_xy, pitch)
        start = sc_2ring_pinadj._find_edge_xy(start, pin_xy, n_ring,
                                              pitch, duct_ftf[0][0])
        start = sc_2ring_pinadj._find_corner_xy(start, pin_xy, n_ring,
                                                pitch, dpin,
                                                duct_ftf[0][0])
    # using a bare except here because any failure in the above will be
    # reflected in a preceding test
    except:
        pytest.xfail("Failure in coolant xy methods (should raise "
                     "another error elsewhere in the tests)")

    res = sc_2ring_pinadj._find_duct_bypass_xy(start, n_ring, pitch,
                                               dpin, duct_ftf)
    de, dc = sc_2ring_pinadj._get_ring0_c2c(n_ring, duct_ftf[0],
                                            pitch, dpin)
    ans = np.array([[de * np.cos(np.pi / 3), de * np.sin(np.pi / 3)],
                    [dc * np.cos(np.pi / 6), dc * np.sin(np.pi / 6)],
                    [de, 0.0],
                    [dc * np.cos(11 * np.pi / 6),
                     dc * np.sin(11 * np.pi / 6)],
                    [de * np.cos(5 * np.pi / 3),
                     de * np.sin(5 * np.pi / 3)],
                    [0.0, -dc],
                    [de * np.cos(4 * np.pi / 3),
                     de * np.sin(4 * np.pi / 3)],
                    [dc * np.cos(7 * np.pi / 6),
                     dc * np.sin(7 * np.pi / 6)],
                    [-de, 0.0],
                    [dc * np.cos(5 * np.pi / 6),
                     dc * np.sin(5 * np.pi / 6)],
                    [de * np.cos(2 * np.pi / 3),
                     de * np.sin(2 * np.pi / 3)],
                    [0.0, dc]])
    ans = np.add(ans, start[-12:])
    ans = np.concatenate((start, ans))
    assert np.allclose(res, ans)


def test_ring2ring_xy(sc_2ring):
    """Test computation of centroids on outer ring from inner ring"""
    n_ring = 2
    ans = np.array([[np.cos(np.pi / 3), np.sin(np.pi / 3)],
                    [np.cos(np.pi / 6), np.sin(np.pi / 6)],
                    [1.0, 0.0],
                    [np.cos(11 * np.pi / 6), np.sin(11 * np.pi / 6)],
                    [np.cos(5 * np.pi / 3), np.sin(5 * np.pi / 3)],
                    [0.0, -1.0],
                    [np.cos(4 * np.pi / 3), np.sin(4 * np.pi / 3)],
                    [np.cos(7 * np.pi / 6), np.sin(7 * np.pi / 6)],
                    [-1.0, 0.0],
                    [np.cos(5 * np.pi / 6), np.sin(5 * np.pi / 6)],
                    [np.cos(2 * np.pi / 3), np.sin(2 * np.pi / 3)],
                    [0.0, 1.0]])
    # Note: "18" in the below is the index of the first duct
    # subchannel, ID = 19.
    assert np.allclose(ans, sc_2ring._get_ring_xy(np.zeros((12, 2)),
                                                  n_ring,
                                                  1.0, 1.0, 18))
    # If initial values are zero and distances between centroids
    # are zero, the result should be zero.
    assert np.allclose(sc_2ring._get_ring_xy(np.zeros((12, 2)),
                                             n_ring,
                                             0.0, 0.0, 18),
                       np.zeros((12, 2)))


def test_ring0_centroid_dist(sc_1ring):
    """Test distance calc between outer coolant and duct subchannels"""
    # Need fixture for the method to test only.
    test_vals = sc_1ring._get_ring0_c2c(1, (0.0, 0.0), 0.0, 0.0)
    assert all([np.abs(v) < 1e-6 for v in test_vals])
    test_vals = sc_1ring._get_ring0_c2c(1, (1.0, 2.0), 0.0, 0.0)
    assert np.abs(test_vals[0] - 0.5) < 1e-6
    assert np.abs(test_vals[1] - np.sqrt(3) / 3) < 1e-6
    # When pin diameter is 0, the corner subchannel is  be found with
    # simple 30-60-90 triangle geometry: if the edge distance is a/2,
    # then the corner distance should be sqrt(3)*a/2
    assert np.abs(2 * test_vals[0] / np.sqrt(3) - test_vals[1]) < 1e-6
    # When the pin diameter is not zero, the corner distance should be
    # less than this expected value.
    test_vals = sc_1ring._get_ring0_c2c(1, (1.0, 2.0), 0.0, 1.0)
    assert 2 * test_vals[0] / np.sqrt(3) > test_vals[1]


def test_ring_center2center_dist(sc_2ring):
    """Test the centroid-centroid distance between duct/bypass rings"""
    # Need the fixture for the method only; test on hardcoded input
    # Arguments are the differences in duct flat-to-flat distances
    # Arg 1 = 2.0 - 1.0 = 1.0
    # Arg 2 = 4.0 - 2.0 = 2.0
    # Want the distance between the x's
    #   |<--------- 4.0 ----------------->|
    #   |      |<---- 2.0 -------->|      |
    #   |      |    |<- 1.0 ->|    |      |
    #   |      |    |    :    | x<-|-->x  |
    #   |      |    |    :    |    |      |
    #   |      |    |    :    |    |      |
    #   |      |    |    :    |    |      |
    #  -2     -1  -0.5  0.0  0.5  1.0    2.0
    test_vals = sc_2ring._get_ring_c2c(1.0, 2.0)
    ans = 0.75
    assert np.abs(test_vals[0] - ans) < 1e-6
    assert np.abs(test_vals[1] - 2 * ans / np.sqrt(3)) < 1e-6
