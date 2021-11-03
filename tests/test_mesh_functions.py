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
date: 2021-08-20
author: matz
Test mesh functions for inter-assembly gap disagreement
"""
########################################################################
import numpy as np
import pytest
import dassh


def mock_quad(sc_per_asm_0, sc_per_asm_1):
    """Create mock quadratic functions and the independent variable
    spaces from/to which they are approximated"""
    sc_per_side_0 = sc_per_asm_0 / 6 - 1
    x0 = np.linspace(-1, 1, int(sc_per_side_0 + 2))
    y0 = np.zeros((6, len(x0)))
    c = np.zeros((6, 3))  # coefficients array
    c[0] = 2 * np.random.random(3) - 1
    y0[0] = c[0, 0] * x0**2 + c[0, 1] * x0 + c[0, 2]
    for s in range(1, 5):
        c[s] = 2 * np.random.random(3) - 1
        # Side needs to line up w/ previous - one constraint means we
        # get to choose only 2 coefficients; at x = -1, y = a - b + c
        c[s, 2] = y0[s - 1][-1] - c[s, 0] + c[s, 1]
        y0[s] = c[s, 0] * x0**2 + c[s, 1] * x0 + c[s, 2]

    # Last side has to line up with previous and start
    # Two constraints means that we get to choose only one coefficient
    c[-1] = 2 * np.random.random(3) - 1
    # at x = -1, y1 = a - b + c ; at x = 1,  y2 = a + b + c
    c[-1, 2] = 0.5 * (y0[0, 0] + y0[-2, -1] - 2 * c[-1, 0])
    c[-1, 1] = c[-1, 0] - y0[-2, -1] + c[-1, 2]
    y0[-1] = c[-1, 0] * x0**2 + c[-1, 1] * x0 + c[-1, 2]

    # Create 1D array w/ y values - currently, each side includes both
    # corners. In DASSH, each side goes: [side, ..., side, corner].
    y0 = np.hstack(y0[:, 1:])

    # Create new x on which to approximate co
    sc_per_side_1 = sc_per_asm_1 / 6 - 1
    x1 = np.linspace(-1, 1, int(sc_per_side_1 + 2))
    return x0, y0, x1


########################################################################
# LINEAR INTERPOLATION
########################################################################


def test_lin_interp_coarse2fine():
    """Test that linear interpolation returns sensible values"""
    # Set up random quadratic fxns along hex sides that meet at corners
    sc_per_asm0 = 24
    sc_per_asm1 = 48
    xc, yc, xf = mock_quad(sc_per_asm0, sc_per_asm1)
    y_fine = dassh.mesh_functions.interpolate_lin(xc, yc, xf)
    # b/c there are twice as many values in the approx array as in the
    # original, every other value should be in the old array
    for i in range(len(yc)):
        assert yc[i] == pytest.approx(y_fine[2 * i + 1])


def test_lin_interp_fine2coarse():
    """Test that linear interpolation method returns sensible values"""
    # Set up random quadratic fxns along hex sides that meet at corners
    # now let's go the other way
    sc_per_asm0 = 48
    sc_per_asm1 = 12
    xf, yf, xc = mock_quad(sc_per_asm0, sc_per_asm1)
    y_coarse = dassh.mesh_functions.interpolate_lin(xf, yf, xc)
    print(yf)
    print(y_coarse)
    # b/c there are twice as many values in the approx array as in the
    # original, every 4 * i + 3 value should be in the old array
    for i in range(len(y_coarse)):
        assert y_coarse[i] == pytest.approx(yf[i * 4 + 3])


def test_lin_interp_from_only_corners():
    """Test that I can linear interpolate from a very course mesh
    (only 6 sc per assembly) to a finer one"""
    sc_per_asm0 = 6
    sc_per_asm1 = 24
    xc, yc, xf = mock_quad(sc_per_asm0, sc_per_asm1)
    y_fine = dassh.mesh_functions.interpolate_lin(xc, yc, xf)

    # b/c there are four as many values in the approx array as in the
    # original, the shift required to match them is 4 * i + 3
    for i in range(len(yc)):
        assert yc[i] == pytest.approx(y_fine[4 * i + 3])


def test_lin_interp_to_only_corners():
    """Test that I can linearly interpolate to a very coarse mesh
    (only 6 sc per assembly) from a finer one"""
    sc_per_asm0 = 48
    sc_per_asm1 = 6
    xf, yf, xc = mock_quad(sc_per_asm0, sc_per_asm1)
    y_coarse = dassh.mesh_functions.interpolate_lin(xf, yf, xc)
    # b/c there are eight as many values in the approx array as in the
    # original, the shift required to match them is 8 * i + 7
    # 0 |  0  1  2  3  4  5  6  7
    # 1 |  8  9 10 11 12 13 14 15
    # 2 | 16 17 18 19 20 21 22 23
    for i in range(len(y_coarse)):
        assert y_coarse[i] == pytest.approx(yf[8 * i + 7])


def test_lin_interp_duct_temps0(c_fuel_asm, c_ctrl_asm):
    """Test linear interpolation method on actual assembly objects"""
    assemblies = [c_ctrl_asm]
    T_duct = [c_ctrl_asm.duct_outer_surf_temp]
    for i in range(6):
        assemblies.append(c_fuel_asm)
        T_duct.append(c_fuel_asm.duct_outer_surf_temp)

    approx_T_duct = [
        dassh.mesh_functions.interpolate_lin(
            assemblies[i].x_pts,
            T_duct[i],
            assemblies[0].x_pts)
        for i in range(len(T_duct))]
    # array shape should be: rows = n_asm; cols = max(sc)
    approx_T_duct = np.array(approx_T_duct)
    assert approx_T_duct.shape == (len(T_duct), len(T_duct[1]))


########################################################################
# QUADRATIC INTERPOLATION
########################################################################


def test_quad_interp_coarse2fine():
    """Test that quadratic interpolation returns sensible values"""
    # Set up random quadratic fxns along hex sides that meet at corners
    sc_per_asm0 = 24
    sc_per_asm1 = 48
    xc, yc, xf = mock_quad(sc_per_asm0, sc_per_asm1)
    y_fine = dassh.mesh_functions.interpolate_quad(xc, yc, xf)
    # b/c there are twice as many values in the approx array as in the
    # original, every other value should be in the old array
    for i in range(len(yc)):
        assert yc[i] == pytest.approx(y_fine[2 * i + 1])


def test_quad_interp_fine2coarse():
    """Test that quadratic interpolation returns sensible values"""
    # Set up random quadratic fxns along hex sides that meet at corners
    # now let's go the other way
    sc_per_asm0 = 48
    sc_per_asm1 = 12
    xf, yf, xc = mock_quad(sc_per_asm0, sc_per_asm1)
    y_coarse = dassh.mesh_functions.interpolate_quad(xf, yf, xc)
    print(yf)
    print(y_coarse)
    # b/c there are twice as many values in the approx array as in the
    # original, every 4 * i + 3 value should be in the old array
    for i in range(len(y_coarse)):
        assert y_coarse[i] == pytest.approx(yf[i * 4 + 3])


def test_quad_interp_from_only_corners():
    """Test that I can do quadratic interp from a very course mesh
    (only 6 sc per assembly) to a finer one"""
    sc_per_asm0 = 6
    sc_per_asm1 = 24
    xc, yc, xf = mock_quad(sc_per_asm0, sc_per_asm1)
    y_fine = dassh.mesh_functions.interpolate_quad(xc, yc, xf)

    # b/c there are four as many values in the approx array as in the
    # original, the shift required to match them is 4 * i + 3
    for i in range(len(yc)):
        assert yc[i] == pytest.approx(y_fine[4 * i + 3])


def test_quad_interp_to_only_corners():
    """Test that I can do quadratic interp to a very coarse mesh
    (only 6 sc per assembly) from a finer one"""
    sc_per_asm0 = 48
    sc_per_asm1 = 6
    xf, yf, xc = mock_quad(sc_per_asm0, sc_per_asm1)
    y_coarse = dassh.mesh_functions.interpolate_quad(xf, yf, xc)
    # b/c there are eight as many values in the approx array as in the
    # original, the shift required to match them is 8 * i + 7
    # 0 |  0  1  2  3  4  5  6  7
    # 1 |  8  9 10 11 12 13 14 15
    # 2 | 16 17 18 19 20 21 22 23
    for i in range(len(y_coarse)):
        assert y_coarse[i] == pytest.approx(yf[8 * i + 7])


def test_quad_interp_duct_temps0(c_fuel_asm, c_ctrl_asm):
    """Test quadratic interpolation on actual assembly objects"""
    assemblies = [c_ctrl_asm]
    T_duct = [c_ctrl_asm.duct_outer_surf_temp]
    for i in range(6):
        assemblies.append(c_fuel_asm)
        T_duct.append(c_fuel_asm.duct_outer_surf_temp)

    approx_T_duct = [
        dassh.mesh_functions.interpolate_quad(
            assemblies[i].x_pts,
            T_duct[i],
            assemblies[0].x_pts)
        for i in range(len(T_duct))]
    # array shape should be: rows = n_asm; cols = max(sc)
    approx_T_duct = np.array(approx_T_duct)
    assert approx_T_duct.shape == (len(T_duct), len(T_duct[1]))


def test_quad_is_better_than_lin():
    """Show that quadratic interpolation is better than linear
    (in this case coarse to fine)"""
    xc = np.arange(-1, 1 + 0.2, 0.2)
    yc = np.zeros((6, xc.shape[0] - 1))
    yc[:, ] = -xc[1:]**2 + 1
    yc = yc.flatten()
    xf = np.arange(-1, 1 + 0.01, 0.01)
    yf = np.zeros((6, xf.shape[0] - 1))
    yf[:, ] = -xf[1:]**2 + 1
    yf = yf.flatten()
    yf_res_lin = dassh.mesh_functions.interpolate_lin(xc, yc, xf)
    yf_res_quad = dassh.mesh_functions.interpolate_quad(xc, yc, xf)
    diff_lin = yf_res_lin - yf
    diff_quad = yf_res_quad - yf
    # Error in quadratic interpolation should be zero
    assert np.max(np.abs(diff_quad)) == pytest.approx(0.0)
    # Error in linear interpolation should not be zero
    assert np.max(np.abs(diff_lin)) > 0.0
    # Error in linear interpolation should be negative
    idx = np.argmax(np.abs(diff_lin))
    assert diff_lin[idx] < 0.0


########################################################################
# GRIP MAPPING
########################################################################


@pytest.fixture
def two_asm_core(testdir, coolant, c_fuel_asm, c_ctrl_asm):
    """Core with two dissimilar assemblies"""
    asm_list = np.ones(7) * np.nan
    asm_list[0] = 0
    asm_list[1] = 1
    asm_pitch = 0.12
    fr = 0.05
    t_in = 623.15
    core_obj = dassh.core.Core(asm_list, asm_pitch, fr, coolant, t_in)
    assemblies = [c_ctrl_asm, c_fuel_asm]
    core_obj.load(assemblies)
    return core_obj, assemblies


def test_grid_mapping_arrays(two_asm_core):
    """Check grid mapping array setup between dissimilar assemblies"""
    # Two assemblies: Assembly 2 has finer mesh. That means gap mesh
    # between assemblies 1 and 2 is defined by assembly 2; the other
    # subchannels adjacent to assembly 1 are defined by assembly 1
    c, asms = two_asm_core

    # Assembly 2 is easier, do this first
    xb_asm = asms[1].rodded.calculate_xbnds()
    xb_core = c._asm_sc_xbnds[1]
    m_f2c, m_c2f = dassh.mesh_functions._map_asm2gap(xb_asm, xb_core)
    assert np.array_equal(m_f2c, np.identity(54))
    assert np.array_equal(m_c2f, np.identity(54))

    # Now set up assembly 1 tests
    xb_asm = asms[0].rodded.calculate_xbnds()
    xb_core = c._asm_sc_xbnds[0]
    m_f2c, m_c2f = dassh.mesh_functions._map_asm2gap(xb_asm, xb_core)
    assert m_f2c.shape == (24, 54)
    assert np.all(np.sum(m_f2c, axis=0)[:29] != 0)
    assert np.all(np.sum(m_f2c, axis=0)[29:] == 0)
    assert np.array_equal(np.sum(m_f2c, axis=1), np.ones(24))
    assert np.all(np.sum(m_c2f, axis=1)[:29] != 0)
    assert np.all(np.sum(m_c2f, axis=1)[29:] == 0)
    ans = np.zeros(54)
    ans[:29] = 1
    assert np.array_equal(np.sum(m_c2f, axis=1), ans)


def test_grid_mapping_arrays_simple():
    """Test grid mapping array setup by hand

    0    0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   1.0
    |-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
    |  8  |  0  |  1  |  2  |  3  |  4  |  5  |  6  |  7  |  8  |
    |-----------------------------------------------------------|
    |     3     |     0     |     1     |     2     |     3     |
    |-----------|-----------|-----------|-----------|-----------|
    0          0.2         0.4         0.6         0.8         1.0

    """
    xb_asm = np.linspace(0, 1, 6)
    xb_core = np.linspace(0, 1, 11)
    xb_core = xb_core[1:-1]
    m_f2c, m_c2f = dassh.mesh_functions._map_asm2gap(xb_asm, xb_core)
    assert m_f2c.shape == (4, 9)
    assert m_c2f.shape == (9, 4)
    ans_f2c = np.array([[0, 0.5, 0.5, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0.5, 0.5, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0.5, 0.5, 0, 0],
                        [0.25, 0, 0, 0, 0, 0, 0, 0.25, 0.5]
                        ], dtype='float')
    ans_c2f = np.array([[0, 0, 0, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 1]
                        ], dtype='float')
    assert np.allclose(m_f2c, ans_f2c)
    assert np.allclose(m_c2f, ans_c2f)
