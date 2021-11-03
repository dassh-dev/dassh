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
Methods to treat disagreements in the inter-assembly gap
"""
########################################################################
import numpy as np


def map_across_gap(vector_in, map):
    """Map values across assembly/gap mesh disagreements

    Parameters
    ----------
    t_in : numpy.ndarray
        Temperatures on the original basis
    map : numpy.ndarray
        Map to convert temperatures to the new basis

    Returns
    -------
    numpy.ndarray
        Temperatures on the new basis

    """
    # It is faster to just to do the calculation, even if the map
    # is the identity array, than to check and potentially bypass.
    return np.dot(map, vector_in)


def _map_asm2gap(xb_reg, xb_core):
    """Make maps to convert between axial region radial mesh and radial
    mesh of the inter-assembly gap"""
    # 2021-04-28: Changed format of xb_core when I remeshed the gap
    # Need to modify to match the xb_reg and the format required here
    fine_dim = xb_core.shape[0]  # <-- need to tack on zeros at end
    # dim_to_save = np.count_nonzero(xb_core)
    tmp = np.zeros(np.count_nonzero(xb_core) + 2)
    tmp[1:-1] = xb_core[xb_core > 0]
    tmp[-1] = xb_reg[-1]
    xb_core = tmp

    # If xb_core and xb_reg are the same, no mapping required
    if xb_core.shape == xb_reg.shape:
        if np.allclose(xb_core, xb_reg):
            tmp = np.identity(xb_reg.shape[0] - 2)
            # Fine --> coarse
            m_f2c = np.zeros((xb_reg.shape[0] - 2, fine_dim))
            m_f2c[:, :(xb_reg.shape[0] - 2)] = tmp
            # Coarse --> fine
            m_c2f = np.zeros((fine_dim, xb_reg.shape[0] - 2))
            m_c2f[:(xb_reg.shape[0] - 2), :] = tmp
            return m_f2c, m_c2f
            # return np.identity(dim_to_save), np.identity(dim_to_save)

    # Preallocate the mapping array
    mapping_f2c = np.zeros((xb_reg.shape[0] - 1, xb_core.shape[0] - 1))
    # CME: Coarse mesh element
    # FME: Fine mesh element
    # LBND/UBND: low-bound / upper-bound
    for CME in range(mapping_f2c.shape[0]):
        CME_LBND = xb_reg[CME]
        CME_UBND = xb_reg[CME + 1]
        FME = np.searchsorted(xb_core, xb_reg[CME]) - 1
        FME_UBND = xb_core[FME + 1]
        mapping_f2c[CME, FME] = min([CME_UBND - CME_LBND,
                                     FME_UBND - CME_LBND])
        while FME_UBND < CME_UBND:
            FME += 1
            FME_LBND = xb_core[FME]
            FME_UBND = xb_core[FME + 1]
            mapping_f2c[CME, FME] = min([CME_UBND - FME_LBND,
                                         FME_UBND - FME_LBND])
    # Normalize
    dx_reg = xb_reg[1:] - xb_reg[:-1]
    m_f2c = (mapping_f2c.T / dx_reg).T
    dx_core = xb_core[1:] - xb_core[:-1]
    m_c2f = (mapping_f2c / dx_core).T
    # Combine half-corner and trim array so first entries correspond
    # to first edge channels
    m_f2c[-1, :] += m_f2c[0, :]
    m_f2c[:, -1] += m_f2c[:, 0]
    m_f2c[-1] *= 0.5
    m_f2c = m_f2c[1:, 1:]
    m_c2f[-1, :] += m_c2f[0, :]
    m_c2f[:, -1] += m_c2f[:, 0]
    m_c2f[-1] *= 0.5
    m_c2f = m_c2f[1:, 1:]

    # Tack on zeros
    # Fine --> course
    expanded_mf2c = np.zeros((m_f2c.shape[0], fine_dim))
    expanded_mf2c[:, :m_f2c.shape[1]] = m_f2c
    # Course --> fine
    expanded_mc2f = np.zeros((fine_dim, m_c2f.shape[1]))
    expanded_mc2f[:m_c2f.shape[0], :] = m_c2f
    return expanded_mf2c, expanded_mc2f


########################################################################
# OLD: INTERPOLATION METHODS
########################################################################


def interpolate_lin(x, y, x_new):
    """Approximate a vector of temperatures to a coarser or finer mesh

    Parameters
    ----------
    x : numpy.ndarray
        The positions of the original mesh centroids along a hex
        side; length must be greater than 1
    y : numpy.ndarray
        The original temperatures at those centroids; length must
        be greater than 1
    x_new : numpy.ndarray
        The positions of the new mesh centroids to which the new
        temperatures will be approximated

    Returns
    -------
    numpy.ndarray
        The interpolated temperatures at positions x_new

    Notes
    -----
    Used in DASSH to deal with mesh disagreement in the interassembly
    gap between assemblies with different number of pins

    """
    # If no interpolation needed, just return the original array
    if np.array_equal(x, x_new):
        return y

    # Otherwise...
    # Dress up the temperatures for the interpolation
    ym = _dress_up_yvals(y)

    # If len(x_old) == 2 (only corners): No need to call interpolation
    # fxn - the interpolation is just a linear fit between two corners
    if len(x) == 2:
        y_new = np.linspace(ym[:, 0], ym[:, 1], len(x_new))
        y_new = y_new.transpose()

    # If len(x_new) == 2 (only corners): No need for legendre fit!
    # Can just return the corner temperatures and be done with it.
    elif len(x_new) == 2:
        y_new = ym[:, (0, -1)]

    # Otherwise, bummer: you have to do the linear interpolation to
    # get values on the new mesh x points
    else:
        y_new = np.zeros((6, len(x_new)))
        for i in range(6):
            y_new[i] = np.interp(x_new, x, ym[i])

        # Use the exact corner temps from original array
        y_new[:, -1] = ym[:, -1]

    # Get rid of the stuff you added and return the flattened array
    y_new = y_new[:, 1:]
    return y_new.flatten()


# def interpolate_lin2(x, y, x_new):
#     """Approximate a vector of temperatures to a coarser or finer mesh
#
#     Parameters
#     ----------
#     x : numpy.ndarray
#         The positions of the original mesh centroids along a hex
#         side; length must be greater than 1
#     y : numpy.ndarray
#         The original temperatures at those centroids; length must
#         be greater than 1
#     x_new : numpy.ndarray
#         The positions of the new mesh centroids to which the new
#         temperatures will be approximated
#
#     Returns
#     -------
#     numpy.ndarray
#         The interpolated temperatures at positions x_new
#
#     Notes
#     -----
#     Used in DASSH to deal with mesh disagreement in the interassembly
#     gap between assemblies with different number of pins
#
#     """
#     # If no interpolation needed, just return the original array
#     if np.array_equal(x, x_new):
#         return y
#
#     # Otherwise...
#     # Dress up the temperatures for the interpolation
#     ym = _dress_up_yvals(y)
#
#     # If len(x_old) == 2 (only corners): No need to call interpolation
#     # fxn - the interpolation is just a linear fit between two corners
#     if len(x) == 2:
#         y_new = np.linspace(ym[:, 0], ym[:, 1], len(x_new))
#         y_new = y_new.transpose()
#
#     # If len(x_new) == 2 (only corners): No need for legendre fit!
#     # Can just return the corner temperatures and be done with it.
#     elif len(x_new) == 2:
#         y_new = ym[:, (0, -1)]
#
#     # Otherwise, bummer: you have to do the linear interpolation to
#     # get values on the new mesh x points
#     else:
#         y_new = np.zeros((6, len(x_new)))
#         for i in range(6):
#             interpolator = interp1d(x[1:-1], ym[i, 1:-1],
#                                     fill_value='extrapolate',
#                                     assume_sorted=True)
#             y_new[i, 1:-1] = interpolator(x_new[1:-1])
#             # y_new[i] = np.interp(x_new, x, ym[i])
#
#         # Use the exact corner temps from original array
#         y_new[:, -1] = ym[:, -1]
#
#     # Get rid of the stuff you added and return the flattened array
#     y_new = y_new[:, 1:]
#     return y_new.flatten()


def setup_lin_interp_arrays(xc, xf):
    """x"""
    # Eliminate corner coordinates from x_pts array
    xc = xc[1:-1].copy()
    xf = xf[1:-1].copy()

    # Fine to coarse array
    f2c = np.zeros((xc.shape[0] + 1, xf.shape[0] + 1))
    for i in range(xc.shape[0]):
        idx = np.searchsorted(xf, xc[i])
        fme_lbnd = xf[idx - 1]
        fme_ubnd = xf[idx]
        coeff_lbnd = 1 - (xc[i] - fme_lbnd) / (fme_ubnd - fme_lbnd)
        coeff_ubnd = 1 - (fme_ubnd - xc[i]) / (fme_ubnd - fme_lbnd)
        f2c[i, idx - 1] = coeff_lbnd
        f2c[i, idx] = coeff_ubnd
    f2c[-1, -1] = 1.0
    f2c2 = np.zeros((6 * f2c.shape[0], 6 * f2c.shape[1]))
    inds_to_fill = np.zeros((6, 2), dtype=int)
    inds_to_fill[:, 0] = np.arange(0, f2c2.shape[0], f2c2.shape[0] / 6)
    inds_to_fill[:, 1] = np.arange(0, f2c2.shape[1], f2c2.shape[1] / 6)
    for inds in inds_to_fill:
        ix2 = inds[0] + f2c.shape[0]
        iy2 = inds[1] + f2c.shape[1]
        f2c2[inds[0]:ix2, :][:, inds[1]:iy2] = f2c

    # Coarse to fine array
    c2f = np.zeros((xf.shape[0] + 1, xc.shape[0] + 1))
    for i in range(xf.shape[0]):
        idx = np.searchsorted(xc, xf[i])
        if idx == 0:
            idx = 1
        elif idx == xc.shape[0]:
            idx -= 1
        else:
            pass
        cme_lbnd = xc[idx - 1]
        cme_ubnd = xc[idx]
        coeff_lbnd = 1 - (xf[i] - cme_lbnd) / (cme_ubnd - cme_lbnd)
        coeff_ubnd = 1 - (cme_ubnd - xf[i]) / (cme_ubnd - cme_lbnd)
        c2f[i, idx - 1] = coeff_lbnd
        c2f[i, idx] = coeff_ubnd

    c2f[-1, -1] = 1.0
    c2f2 = np.zeros((6 * c2f.shape[0], 6 * c2f.shape[1]))
    inds_to_fill = np.zeros((6, 2), dtype=int)
    inds_to_fill[:, 0] = np.arange(0, c2f2.shape[0], c2f2.shape[0] / 6)
    inds_to_fill[:, 1] = np.arange(0, c2f2.shape[1], c2f2.shape[1] / 6)
    for inds in inds_to_fill:
        ix2 = inds[0] + c2f.shape[0]
        iy2 = inds[1] + c2f.shape[1]
        c2f2[inds[0]:ix2, :][:, inds[1]:iy2] = c2f

    return f2c2, c2f2


def interpolate_quad(x, y, x_new, xparams=None, yparams=None):
    """Approximate a vector of temperatures to a coarser or finer mesh

    Parameters
    ----------
    x : numpy.ndarray
        The positions of the original mesh centroids along a hex
        side; length must be greater than 1
    y : numpy.ndarray
        The original temperatures at those centroids; length must
        be greater than 1
    x_new : numpy.ndarray
        The positions of the new mesh centroids to which the new
        temperatures will be approximated
    xparams : numpy.ndarray (optional)
        If provided, can bypass the setup portions of the
        interpolation fit to data
    yparams : numpy.ndarray (optional)
        Indices with which to slice y-data to shortcut some of the
        interpolation setup

    Returns
    -------
    numpy.ndarray
        The interpolated temperatures at positions x_new

    Notes
    -----
    Used in DASSH to deal with mesh disagreement in the interassembly
    gap between assemblies with different number of pins

    """
    # If no interpolation needed, just return the original array
    if np.array_equal(x, x_new):
        return y
    #
    # Otherwise...
    # Dress up the temperatures for the interpolation; this separates
    # the 1D input array into a 2D array with 6 rows, where row
    # corresponds to the temperatures on each hex side
    ym = _dress_up_yvals(y)
    #
    # If len(x_old) == 2 (only corners): No need to call interpolation
    # fxn - the interpolation is just a linear fit between two corners
    if len(x) == 2:
        y_new = np.linspace(ym[:, 0], ym[:, 1], len(x_new))
        y_new = y_new.transpose()
    #
    # If len(x_new) == 2 (only corners): Need to map adjacent gap fine
    # mesh temps to corners to capture side temps (usually higher).
    # This is done in a different fxn. Default behavior here is to
    # return the corner temperatures and be done.
    elif len(x_new) == 2:
        y_new = ym[:, (0, -1)]
    #
    # Otherwise, bummer: you have to do the interpolation to
    # get values on the new mesh x points
    else:
        if xparams is None:
            idx = get_nearest_xy_index(x, x_new)
            xparams = calculate_xparams(x, x_new, idx)
        if yparams is None:
            try:
                yparams = calculate_yparams(x, idx)
            except UnboundLocalError:
                idx = get_nearest_xy_index(x, x_new)
                yparams = calculate_yparams(x, idx)
        #
        tmp = ym.flatten()
        # Do the interpolations
        # y_new = np.sum(tmp[yparams] * xparams, axis=2)
        y_new = tmp[yparams] * xparams
        y_new = y_new[:, :, 0] + y_new[:, :, 1] + y_new[:, :, 2]

        # Use the exact corner temps from original array
        y_new[:, -1] = ym[:, -1]
    #
    # Get rid of the stuff you added and return the flattened array
    y_new = y_new[:, 1:]
    return y_new.flatten()


def calculate_xparams(x, x_new, idx):
    """Calcualte constants based on x to be used in quadratic interp"""
    xparams = np.zeros((len(idx), 3))
    xparams[:, 0] = ((x_new - x[idx[:, 1]])
                     * (x_new - x[idx[:, 2]])
                     / (x[idx[:, 0]] - x[idx[:, 1]])
                     / (x[idx[:, 0]] - x[idx[:, 2]]))
    xparams[:, 1] = ((x_new - x[idx[:, 0]])
                     * (x_new - x[idx[:, 2]])
                     / (x[idx[:, 1]] - x[idx[:, 0]])
                     / (x[idx[:, 1]] - x[idx[:, 2]]))
    xparams[:, 2] = ((x_new - x[idx[:, 0]])
                     * (x_new - x[idx[:, 1]])
                     / (x[idx[:, 2]] - x[idx[:, 0]])
                     / (x[idx[:, 2]] - x[idx[:, 1]]))
    return xparams


def calculate_yparams(x, idx):
    """Arrange y-val indices for quad interp in one array operation"""
    yparams = np.zeros((6, idx.shape[0], 3), dtype=int)
    yparams[0] = idx
    for i in range(5):
        yparams[i + 1, :, :] = yparams[i] + len(x)
    return yparams


def get_nearest_xy_index(x, x_new):
    """Identify indices of nearest neighbors for quadratic interp"""
    idx = np.zeros((len(x_new), 3), dtype=int)
    for i in range(len(x_new)):
        idx[i, :] = np.argsort(np.abs(x - x_new[i]))[:3]
    return idx


def _dress_up_yvals(y):
    """Reshape temperature array to be per hex side

    Parameters
    ----------
    y : numpy.ndarray
        Y-values to be interpolated

    """
    ym = y.copy().reshape(6, -1)
    tmp = np.zeros((6, ym.shape[1] + 1))
    tmp[:, 1:] = ym
    tmp[0, 0] = ym[-1, -1]
    tmp[1:, 0] = ym[:-1, -1]
    return tmp


# ########################################################################
# # OLD: UNRODDED REGIONS
# ########################################################################
#
# def simple_avg_for_nonbundle_regions(yfine):
#     """Average side/corner subchannels adjacent to UR"""
#     # y = np.random.random(60) + 623.15
#     n_edge_per_side = (yfine.shape[0] - 6) / 6
#     ym = np.roll(yfine, np.floor(-0.5 * n_edge_per_side).astype(int))
#     ym = ym.reshape(6, -1)
#     # even number of edge subchannels (n_edge + corner is odd)
#     if ym.shape[1] % 2:
#         total = np.sum(ym, axis=1)
#     else:
#         total = np.sum(ym[:, :-1], axis=1)
#         total += 0.5 * (ym[:, -1] + np.roll(ym[:, -1], 1))
#     return total / ym.shape[1]
