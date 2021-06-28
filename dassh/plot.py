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
date: 2021-06-28
author: matz
Methods to plot DASSH objects (such as hexagonal fuel assemblies and
the pins and subchannels that comprise them).
"""
########################################################################
import os
import sys
import copy
import logging
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dassh import utils
from dassh import core


_default_color = {
    'interior': 'b',
    'edge': 'r',
    'corner': 'g',
    'duct': 'k',
    'bypass': 'c',
    'pins': '0.5'
}


_sc_type = {
    'interior': [1],
    'edge': [2],
    'corner': [3],
    'duct': [4, 5],
    'bypass': [6, 7]
}


_hex_cols = {
    'coolant': [5, 3],
    'duct_mw': [6, 4],
    'clad_mw': [8, 5],
    'fuel_cl': [9, 6]
}


_pin_cols = {
    'coolant': 3,
    'clad_od': 4,
    'clad_mw': 5,
    'clad_id': 6,
    'fuel_od': 7,
    'fuel_cl': 8
}


module_logger = logging.getLogger('dassh_plot')


def plot_all(dassh_inp, dassh_reactor):
    """Script to postprocess DASSH data to make matplotlib figures

    Parameters
    ----------
    dassh_inp : DASSH_Input or DASSHPlot_Input object
    dassh_reactor : DASSH Reactor object

    Returns
    -------
    None

    """
    for plt_req in dassh_inp.data['Plot'].keys():
        module_logger.log(20, f'....Plotting {plt_req}')
        fxn = f'make_{dassh_inp.data["Plot"][plt_req]["type"]}'
        getattr(sys.modules[__name__], fxn)(
            dassh_reactor, dassh_inp.data['Plot'][plt_req])


########################################################################
# GENERATOR FXNS
########################################################################


def make_SubchannelPlot(dassh_reactor, plot_data):
    """Generate the assembly subchannel figures"""
    f = 'temp_coolant_int.csv'
    try:
        _data = _prepare_input(dassh_reactor, plot_data, f)
    except FileNotFoundError:
        msg = 8 * '.' + f'File "{f}" not found, skipping...'
        module_logger.log(30, msg)
        return
    for i in range(len(_data['assembly_id'])):
        asm_id = _data['assembly_id'][i]
        ascp = SubchannelPlot(dassh_reactor.assemblies[asm_id])
        for zi in _data['z_data'].keys():
            asm_zdata = _data['z_data'][zi][
                _data['z_data'][zi][:, 0] == asm_id]
            ascp.plot(asm_zdata[0, 3:],
                      lbnd=_data['cbar_lbnd'],
                      ubnd=_data['cbar_ubnd'],
                      middle=_data['cbar_mpnt'],
                      cmap=_data['cmap'],
                      cbar_label=_data['cbar_label'],
                      pins=_data['pins'],
                      pin_alpha=_data['pin_alpha'])
            z_str = np.around(_data['bwd_len_conv'](zi), 2)
            fname = '_'.join(['SubchannelPlot',
                              'asm' + str(asm_id),
                              f'z={z_str}'])
            fname += '.png'
            plt.savefig(fname, bbox_inches='tight', dpi=_data['dpi'])
            plt.close()


def make_PinPlot(dassh_reactor, plot_data):
    """Generate the assembly pin-by-pin figures"""
    f = 'temp_pin.csv'
    try:
        _data = _prepare_input(dassh_reactor, plot_data, f)
    except FileNotFoundError:
        msg = 8 * '.' + f'File "{f}" not found, skipping...'
        module_logger.log(30, msg)
        return
    for i in range(len(_data['assembly_id'])):
        asm_id = _data['assembly_id'][i]
        pp = PinPlot(dassh_reactor.assemblies[asm_id])
        for zi in _data['z_data'].keys():
            asm_zdata = _data['z_data'][zi][
                _data['z_data'][zi][:, 0] == asm_id]
            for value in plot_data['value']:
                pp.plot(asm_zdata[:, _pin_cols[value]],
                        lbnd=_data['cbar_lbnd'],
                        ubnd=_data['cbar_ubnd'],
                        middle=_data['cbar_mpnt'],
                        cmap=_data['cmap'],
                        cbar_label=_data['cbar_label'])
                z_str = np.around(_data['bwd_len_conv'](zi), 2)
                fname = '_'.join(['PinPlot', 'asm' + str(asm_id),
                                  value, f'z={z_str}'])
                fname += '.png'
                plt.savefig(fname, bbox_inches='tight', dpi=_data['dpi'])
                plt.close()


def make_CoreSubchannelPlot(dassh_reactor, plot_data):
    """Generate the core-wide subchannel figures"""
    f = 'temp_coolant_int.csv'
    try:
        _data = _prepare_input(dassh_reactor, plot_data, f)
    except FileNotFoundError:
        msg = 8 * '.' + f'File "{f}" not found, skipping...'
        module_logger.log(30, msg)
        return
    cscp = CoreSubchannelPlot(dassh_reactor)
    for zi in _data['z_data'].keys():
        cscp.plot(_data['z_data'][zi],
                  lbnd=_data['cbar_lbnd'],
                  ubnd=_data['cbar_ubnd'],
                  middle=_data['cbar_mpnt'],
                  cmap=_data['cmap'],
                  cbar_label=_data['cbar_label'],
                  rings=plot_data['rings'],
                  ignore_ur=plot_data['ignore_simple'])
        z_str = np.around(_data['bwd_len_conv'](zi), 2)
        fname = '_'.join(['CoreSubchannelPlot', f'z={z_str}'])
        fname += '.png'
        plt.savefig(fname, bbox_inches='tight', dpi=_data['dpi'])
        plt.close()


def make_CorePinPlot(dassh_reactor, plot_data):
    """Make core pin-by-pin figures"""
    f = 'temp_pin.csv'
    try:
        _data = _prepare_input(dassh_reactor, plot_data, f)
    except FileNotFoundError:
        msg = 8 * '.' + f'File "{f}" not found, skipping...'
        module_logger.log(30, msg)
        return
    cpp = CorePinPlot(dassh_reactor)
    for zi in _data['z_data'].keys():
        for v in plot_data['value']:
            cpp.plot(_data['z_data'][zi][:, (0, 1, 2, 3, _pin_cols[v])],
                     lbnd=_data['cbar_lbnd'],
                     ubnd=_data['cbar_ubnd'],
                     middle=_data['cbar_mpnt'],
                     cmap=_data['cmap'],
                     cbar_label=_data['cbar_label'],
                     rings=plot_data['rings'])
            z_str = np.around(_data['bwd_len_conv'](zi), 2)
            fname = '_'.join(['CorePinPlot', v, f'z={z_str}'])
            fname += '.png'
            plt.savefig(fname, bbox_inches='tight', dpi=_data['dpi'])
            plt.close()


def make_CoreHexPlot(dassh_reactor, plot_data):
    """Generator function for CoreHexPlots"""
    chp = CoreHexPlot(dassh_reactor, plot_data)
    for v in plot_data['value']:
        if v == 'total_power':
            chp.make_power(dassh_reactor, plot_data)
        else:
            if plot_data['z'] is None:
                chp.make_axial_peak(dassh_reactor, plot_data, v)
            else:  # average in v
                chp.make_radial_peak_or_avg(dassh_reactor, plot_data, v)


########################################################################
# PLOT DASSH ASSEMBLIES
########################################################################


class AssemblyPlot(object):
    """Plot data for an individual assembly"""

    def __init__(self, dassh_asm, simple_model=False):
        """Save the assembly to get all the geometry chars you want,
        but pull out a bunch of stuff to get it easily"""
        if simple_model:
            # Duct characteristics
            self.duct = {}
            # If modeling a rodded region with simple model, need to select
            # the outermost duct. This occurs in core pin plots
            tmp = dassh_asm.active_region.duct_ftf
            while True:
                if isinstance(tmp[-1], (list, np.ndarray)):
                    tmp = tmp[-1]
                else:
                    break
            self.duct['ftf'] = tmp

        else:
            self.rr = dassh_asm.rodded

            # Subchannel_characteristics
            self.sc = {}
            self.sc['n_sc'] = dassh_asm.rodded.subchannel.n_sc
            self.sc['type'] = dassh_asm.rodded.subchannel.type
            self.sc['xy'] = dassh_asm.rodded.subchannel.xy
            self.sc['radius'] = [dassh_asm.rodded.L[0][0],
                                 [dassh_asm.rodded.L[1][1],
                                  (dassh_asm.rodded.d['pin-wall']
                                   + 0.5 * dassh_asm.rodded.pin_diameter)],
                                 dassh_asm.rodded.d['wcorner'][0][0]]
            # Note: edge subchannel [1] angle needs to be in deg, not rad
            self.sc['angle'] = [[np.pi / 6, 7 * np.pi / 6],
                                [(xi - np.pi / 2) * 180 / np.pi for xi in
                                 dassh_asm.rodded.subchannel._edge_angle],
                                None]
            # Pin characteristics
            self.pin = {}
            self.pin['xy'] = dassh_asm.rodded.pin_lattice.xy
            self.pin['radius'] = dassh_asm.rodded.pin_diameter / 2
            self.pin['pitch'] = dassh_asm.rodded.pin_pitch

            # Duct characteristics
            self.duct = {}
            self.duct['ftf'] = dassh_asm.rodded.duct_ftf

    @staticmethod
    def parse_args(data, lbnd, ubnd, middle, **kwargs):
        """Parse arguments for single assembly figures

        Parameters
        ----------
        data : numpy.ndarray
            Data for one assembly
        lbnd : float (optional)
            Lower bound for the color map
        ubnd : float (optional)
            Upper bound for the color map
        middle : float (optional)
            Middle value for the color map

        """
        # If not specified, identify data lower bound,
        # midpoint, and upper bound based on data
        lbnd, ubnd, middle = _get_data_bnds(data, lbnd, ubnd, middle)

        if not kwargs.get('linewidth'):
            kwargs['linewidth'] = 0.5

        if not kwargs.get('edgecolor'):
            kwargs['edgecolor'] = 'k'

        if kwargs['linewidth'] == 0.0:
            kwargs['linestyle'] = 'None'
        else:
            kwargs['linestyle'] = '-'

        if not kwargs.get('cmap'):
            kwargs['cmap'] = mpl.cm.jet

        if not kwargs.get('norm'):
            kwargs['norm'] = colors.TwoSlopeNorm(
                vmin=lbnd, vcenter=middle, vmax=ubnd)

        return kwargs

    def _add_duct_walls(self, ax, xy_shift=None, lw=0.5, color='0.5'):
        """Find duct dimensions to plot ducts as overlapping hexagons

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot pin duct walls
        lw : float (optional)
            Duct wall hexagon line width (pts)
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        Returns
        -------
        matplotlib.axes.Axes object

        Notes
        -----
        This should be the first thing plotted, because it'll form
        the backdrop to everything plotted in the figure

        """
        xy = np.array([0.0, 0.0])
        if xy_shift is not None:
            xy = xy_shift

        # Plot ducts from the outside in
        for d in self.duct['ftf'][::-1]:
            # Outer duct wall: plot using gray hexagon
            rad = d[1] / np.sqrt(3)
            duct = [mpl.patches.RegularPolygon(xy, 6, radius=rad)]
            duct = mpl.collections.PatchCollection(
                duct, facecolor=color, linewidth=lw, edgecolor='k')
            ax.add_collection(duct)

            # Inner duct wall: plot with white hexagon
            rad = d[0] / np.sqrt(3)
            duct = [mpl.patches.RegularPolygon(xy, 6, radius=rad)]
            duct = mpl.collections.PatchCollection(
                duct, facecolor='1.0', linewidth=lw, edgecolor='k')
            ax.add_collection(duct)
        return ax

    def _set_ax_bnds(self, ax):
        """Set the axis boundaries to reflect the duct size"""
        width = self.duct['ftf'][-1][1] / 2
        height = 2 * width / np.sqrt(3)
        mult = 1 / 10**np.floor(np.log10(height))
        axlim = np.ceil(height * mult) / mult
        ax.set_xlim([-axlim, axlim])
        ax.set_ylim([-axlim, axlim])
        return ax


class SubchannelPlot(AssemblyPlot):
    """Plot subchannel temperature data"""

    def __init__(self, dassh_asm):
        """Initialize the SubchannelPlot instance"""
        AssemblyPlot.__init__(self, dassh_asm)

    def plot(self, data, lbnd=None, ubnd=None, middle=None,
             xy_shift=None, **kwargs):
        """Plot the temperatures of the coolant subchannels

        Parameters
        ----------
        data : numpy.ndarray
            Subchannel temperature data for the whole assembly
        lbnd : float (optional)
            Lower bound for the color map
        ubnd : float (optional)
            Upper bound for the color map
        middle : float (optional)
            Middle value for the color map
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)
        duct_lw : float (optional)
            Duct wall hexagon line width (in pts)

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        cbar_label : str
            Label for the color bar
        lw : float
            Border line width to apply to the subchannel patches
        pins : boolean
            Indicate whether to plot pins over subchannels
        pin_alpha : float
            Indicate the opacity of the pin fill

        Returns
        -------
        matplotlib.axes.Axes object

        """
        # Set default arguments
        kwargs = self.parse_args(data, lbnd, ubnd, middle, **kwargs)

        # isolate the patch kwargs
        patch_kwargs = {'cmap': kwargs['cmap'],
                        'norm': kwargs['norm'],
                        'linewidth': kwargs['linewidth'],
                        'linestyle': kwargs['linestyle'],
                        'edgecolor': kwargs['edgecolor']}

        # Setup the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        # 0. Add duct walls
        ax = self._add_duct_walls(ax, xy_shift)

        # 1. Add corner channels (hexagons)
        ax = self._add_corner_sc(ax, data, xy_shift, **patch_kwargs)

        # 2. Add edge channels (squares)
        ax = self._add_edge_sc(ax, data, xy_shift, **patch_kwargs)

        # 3. Add interior channels (triangles)
        ax = self._add_int_sc(ax, data, xy_shift, **patch_kwargs)

        # 4. If requested, add pins
        if kwargs.get('pins'):
            _alpha = 1.0
            if kwargs.get('pin_alpha'):
                _alpha = kwargs['pin_alpha']
            ax = self._add_pins(ax, color='1.0', alpha=_alpha,
                                xy_shift=xy_shift)

        # Format figure and return
        plt.axis('off')
        ax = self._set_ax_bnds(ax)
        txt = ''
        if kwargs.get('cbar_label'):
            txt = kwargs['cbar_label']
        ax = _add_colorbar(ax, txt, kwargs['cmap'], kwargs['norm'])
        return ax

    def map(self, **kwargs):
        """Plot the layout of subchannels in the assembly

        Parameters
        ----------
        None

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        cbar_label : str
            Label for the color bar
        lw : float
            Border line width to apply to the subchannel patches
        pins : boolean
            Indicate whether to plot pins over subchannels
        pin_alpha : float
            Indicate the opacity of the pin fill

        Returns
        -------
        matplotlib.axes.Axes object

        """
        # Set default arguments
        data = np.ones(self.sc['n_sc']['coolant']['total'])
        kwargs = self.parse_args(data, 0.0, 2.0, 1.0, cmap='bwr', **kwargs)

        # isolate the patch kwargs
        patch_kwargs = {'cmap': kwargs['cmap'],
                        'norm': kwargs['norm'],
                        'linewidth': kwargs['linewidth'],
                        'linestyle': kwargs['linestyle'],
                        'edgecolor': kwargs['edgecolor']}

        # Setup the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        # 0. Add duct walls
        ax = self._add_duct_walls(ax, color='1.0')

        # 1. Add corner channels (hexagons)
        ax = self._add_corner_sc(ax, data, **patch_kwargs)

        # 2. Add edge channels (squares)
        ax = self._add_edge_sc(ax, data, **patch_kwargs)

        # 3. Add interior channels (triangles)
        ax = self._add_int_sc(ax, data, **patch_kwargs)

        # 4. If requested, add pins
        if kwargs.get('pins'):
            _alpha = 1.0
            if kwargs.get('pin_alpha'):
                _alpha = kwargs['pin_alpha']
            ax = self._add_pins(ax, color='1.0', alpha=_alpha)

        # Add labels
        lab = np.arange(1, self.sc['n_sc']['coolant']['total'] + 1, 1)
        xy = self.sc['xy'][:self.sc['n_sc']['coolant']['total']]
        if kwargs.get('fontsize'):
            fontsize = kwargs['fontsize']
        else:
            fontsize = 6
        textcolor = 'r'
        for i in range(lab.shape[0]):
            txt = ax.annotate(str(lab[i].astype(int)),
                              xy[i],
                              size=fontsize,
                              ha='center',
                              va='center',
                              weight='bold',
                              color=textcolor)
        # Format figure and return
        plt.axis('off')
        ax = self._set_ax_bnds(ax)
        return ax

    def _add_corner_sc(self, ax, data, xy_shift=None, **kwargs):
        """Add corner subchannels to the existing axis

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot corner subchannels
        data : numpy.ndarray
            Subchannel temperature data for the whole assembly
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        lw : float
            Linewidth to apply to the subchannel patches

        Returns
        -------
        matplotlib.axes.Axes object
            With corner subchannel temperatures added

        Notes
        -----
        Corner subchannels are plotted with hexagons. Hex radius is
        the distance from center to vertices: rodded.d['wcorner'][0][0]
        is the side length, which is equal to the long diagonal. Will
        need to be overlapped later.

        """
        # xy = self.sc['xy'][np.where(self.sc['type'] == 2)]
        # Can't actually use the corner xy positions because we're
        # plotting with these stupid hexagons. Need to calculate the
        # center relative to the duct corner with the known side length
        xy_duct_corner = self.sc['xy'][np.where(self.sc['type'] == 4)]
        xy_duct_corner = xy_duct_corner[:6]  # want only innermost duct

        # The shift (duct corner centroid to duct inner corner) will be
        # different for each corner hexagon; list as 2d array for xy
        dthick = (self.duct['ftf'][0][1] - self.duct['ftf'][0][0]) / 2
        shift = np.array(
            [[-0.5 * dthick, -0.5 * (self.rr.d['wcorner'][0, 1]
                                     - self.rr.d['wcorner'][0, 0])],
             [-0.5 * dthick, 0.5 * (self.rr.d['wcorner'][0, 1]
                                    - self.rr.d['wcorner'][0, 0])],
             [0.0, dthick / np.sqrt(3)],
             [0.5 * dthick, 0.5 * (self.rr.d['wcorner'][0, 1]
                                   - self.rr.d['wcorner'][0, 0])],
             [0.5 * dthick, -0.5 * (self.rr.d['wcorner'][0, 1]
                                    - self.rr.d['wcorner'][0, 0])],
             [0.0, -dthick / np.sqrt(3)]])
        # From the duct corner, need to shift inward based on radius
        shift[0] += [-0.5 * np.sqrt(3) * self.sc['radius'][2],
                     -0.5 * self.sc['radius'][2]]
        shift[1] += [-0.5 * np.sqrt(3) * self.sc['radius'][2],
                     0.5 * self.sc['radius'][2]]
        shift[2] += [0.0, self.sc['radius'][2]]
        shift[3] += [0.5 * np.sqrt(3) * self.sc['radius'][2],
                     0.5 * self.sc['radius'][2]]
        shift[4] += [0.5 * np.sqrt(3) * self.sc['radius'][2],
                     -0.5 * self.sc['radius'][2]]
        shift[5] += [0.0, -self.sc['radius'][2]]
        xy = xy_duct_corner + shift
        if xy_shift is not None:
            xy += xy_shift

        z = data[np.where(self.sc['type'] == 2)]
        tmp = [mpl.patches.RegularPolygon(
            (xi, yi), 6, radius=self.sc['radius'][2])
            for xi, yi in zip(xy[:, 0], xy[:, 1])]
        tmp = mpl.collections.PatchCollection(tmp, **kwargs)
        tmp.set_array(z)
        ax.add_collection(tmp)
        return ax

    def _add_edge_sc(self, ax, data, xy_shift=None, **kwargs):
        """Add edge subchannels to existing axis

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot corner subchannels
        data : numpy.ndarray
            Subchannel temperature data for the whole assembly
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        lw : float
            Linewidth to apply to the subchannel patches

        Returns
        -------
        matplotlib.axes.Axes object
            With edge subchannel temperatures added

        Notes
        -----
        Edge subchannels are plotted with rectangles. Width is pin
        pitch, height is distance from pin center to wall. No overlap
        is necessary; however, these overlap the corner hexagons.

        """
        xy = self.sc['xy'][np.where(self.sc['type'] == 1)]
        if xy_shift is not None:
            xy += xy_shift

        z = data[np.where(self.sc['type'] == 1)]
        # Need to loop over sides to properly orient each square to
        # be flush with the duct wall
        sc_edge_side = int(len(xy) / 6)
        shift = np.zeros((6, 2))
        # patches.Rectangle plots from the LOWER LEFT corner rather
        # than center - need to shift each rectangle based on angle.
        shift[0] = [-self.sc['radius'][1][1], 0.0]
        shift[1] = [-0.5 * self.sc['radius'][1][1],
                    0.5 * self.sc['radius'][1][0]]
        shift[2] = [0.5 * self.sc['radius'][1][1],
                    np.sqrt(3) * 0.5 * self.sc['radius'][1][1]]
        shift[3] = [self.sc['radius'][1][1], 0.0]
        shift[4] = [0.5 * self.sc['radius'][1][1],
                    -0.5 * self.sc['radius'][1][0]]
        shift[5] = [-0.5 * self.sc['radius'][1][1],
                    -np.sqrt(3) * 0.5 * self.sc['radius'][1][1]]
        edge_sq = []
        for i in range(6):
            side_xy = xy[i * sc_edge_side:(i + 1) * sc_edge_side]
            side_xy += shift[i]
            edge_sq += [mpl.patches.Rectangle(
                (xi, yi),
                self.sc['radius'][1][0],
                self.sc['radius'][1][1],
                angle=self.sc['angle'][1][i])
                for xi, yi in zip(side_xy[:, 0], side_xy[:, 1])]
        edge_sq = mpl.collections.PatchCollection(edge_sq, **kwargs)
        edge_sq.set_array(z)
        ax.add_collection(edge_sq)
        return ax

    def _add_int_sc(self, ax, data, xy_shift=None, **kwargs):
        """Add interior subchannels to existing axis

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot corner subchannels
        data : numpy.ndarray
            Subchannel temperature data for the whole assembly
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        lw : float
            Linewidth to apply to the subchannel patches

        Returns
        -------
        matplotlib.axes.Axes object
            With edge subchannel temperatures added

        Notes
        -----
        Edge subchannels are plotted with triangles. Triangle "radius"
        is distance from center to vertices (this is equal to the
        distance between interior subchannels). No need to overlap;
        these finalize the overlapping of the other subchannels.

        """
        xy = self.sc['xy'][np.where(self.sc['type'] == 0)]
        if xy_shift is not None:
            xy += xy_shift

        z = data[np.where(self.sc['type'] == 0)]

        # Need to loop over interior subchannels; they alternate
        # orientation from R to L with each step (pattern is
        # R, L, R, L, R...)
        int_tri = []
        for i in range(len(xy)):
            orient = self.sc['angle'][0][i % 2]
            int_tri += [mpl.patches.RegularPolygon(
                (xy[i, 0], xy[i, 1]), 3, radius=self.sc['radius'][0],
                orientation=orient)]

        int_tri = mpl.collections.PatchCollection(int_tri, **kwargs)
        int_tri.set_array(z)
        ax.add_collection(int_tri)
        return ax

    def _add_pins(self, ax, color='0.5', alpha=1.0, xy_shift=None):
        """Generate matplotlib circles to show the fuel pins

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot pin circles
        color : str (optional)
            Fill color to apply to pin circles
        alpha : float (optional)
            Opacity of pin circle fill
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        Returns
        -------
        matplotlib.axes.Axes object

        """
        xy = self.pin['xy']
        if xy_shift is not None:
            xy += xy_shift
        circles = [plt.Circle((xi, yi), self.pin['radius'])
                   for xi, yi in zip(xy[:, 0], xy[:, 1])]
        c = mpl.collections.PatchCollection(circles,
                                            facecolor=color,
                                            linewidth=0.5,
                                            edgecolor='k')
        c.set_alpha(alpha)
        ax.add_collection(c)
        return ax


class SingleNodePlot(AssemblyPlot):
    """Plot values where model uses homogenized single node; mainly
    for use in the CoreSubchannelPlot object"""

    def __init__(self, dassh_asm):
        """Initialize the SubchannelPlot instance"""
        AssemblyPlot.__init__(self, dassh_asm, simple_model=True)

    def plot(self, ax, data, cmap, norm, gray=False, lw=0.5, xy_shift=None):
        """Find duct dimensions to plot ducts as overlapping hexagons

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot pin duct walls
        data : float
            Data to be plotted in the hexagon
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        lw : float (optional)
            Hexagon line width (in pts)
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        Returns
        -------
        matplotlib.axes.Axes object

        """
        xy = np.array([0.0, 0.0])
        if xy_shift is not None:
            xy = xy_shift

        # Set up patch kwargs
        patch_kwargs = {'cmap': cmap, 'norm': norm}

        # Plot ducts from the outside in
        # Outer duct wall: plot using gray hexagon
        try:
            rad = self.duct['ftf'][1] / np.sqrt(3)
        except IndexError:
            print(self.duct['ftf'])
            raise
        duct = [mpl.patches.RegularPolygon(xy, 6, radius=rad)]
        duct = mpl.collections.PatchCollection(
            duct, facecolor='0.5', linewidth=lw, edgecolor='k')
        ax.add_collection(duct)

        # Inner duct wall: plot with hexagon the color of the data
        rad = self.duct['ftf'][0] / np.sqrt(3)
        duct = [mpl.patches.RegularPolygon(xy, 6, radius=rad)]
        duct = mpl.collections.PatchCollection(
            duct, facecolor='0.5', linewidth=lw, edgecolor='k',
            **patch_kwargs)
        if not gray:
            duct.set_array(np.array([data]))
        ax.add_collection(duct)
        return ax

    def plot_single_color(self, ax, color, lw=0.5, xy_shift=None):
        """Plot colored hexagon

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot pin duct walls
        color : string
            Color to plot in the hexagon
        lw : float (optional)
            Hexagon line width (in pts)
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        Returns
        -------
        matplotlib.axes.Axes object

        """
        xy = np.array([0.0, 0.0])
        if xy_shift is not None:
            xy = xy_shift

        # Plot ducts from the outside in
        # Outer duct wall: plot using gray hexagon
        try:
            rad = self.duct['ftf'][1] / np.sqrt(3)
        except IndexError:
            print(self.duct['ftf'])
            raise
        duct = [mpl.patches.RegularPolygon(xy, 6, radius=rad)]
        duct = mpl.collections.PatchCollection(
            duct, facecolor='0.5', linewidth=lw, edgecolor='k')
        ax.add_collection(duct)

        # Inner duct wall: plot with hexagon the color of the data
        rad = self.duct['ftf'][0] / np.sqrt(3)
        duct = [mpl.patches.RegularPolygon(xy, 6, radius=rad)]
        duct = mpl.collections.PatchCollection(
            duct, facecolor=color, linewidth=lw, edgecolor='k')
        ax.add_collection(duct)
        return ax


class DuctPlot(AssemblyPlot):
    """x"""

    UNIT_HEX_VERTEX = np.array([
        [0.0, 1.0],
        [0.5 * np.sqrt(3), 0.25 * np.sqrt(3)],
        [0.5 * np.sqrt(3), -0.25 * np.sqrt(3)],
        [0.0, -1.0],
        [-0.5 * np.sqrt(3), -0.25 * np.sqrt(3)],
        [-0.5 * np.sqrt(3), 0.25 * np.sqrt(3)]
    ])

    def __init__(self, dassh_asm):
        """Initialize the DuctPlot instance"""
        AssemblyPlot.__init__(self, dassh_asm)
        self.d = dassh_asm.rodded.d
        self.pin_pitch = dassh_asm.rodded.pin_pitch
        self.n_duct = dassh_asm.rodded.n_duct
        self.n_byp = dassh_asm.rodded.n_bypass

        # Calculate width of plotted duct walls: all will be plotted
        # with the same width. Bypass gaps will be plotted with 1/2
        # the width of the duct.
        dwc = self.d['wcorner'][-1][0]
        max_width = 0.8 * np.sqrt(3) * dwc + self.d['wall'][-1]
        self.duct_thickness = max_width / (1.5 * self.n_duct - 0.5)
        self.byp_thickness = 0.0
        if self.n_byp > 0:
            self.byp_thickness = 0.5 * self.duct_thickness

        # Consistency check
        assert np.abs(max_width
                      - (self.n_duct * self.duct_thickness
                         + self.n_byp * self.byp_thickness)) < 1e-6

    def _get_modified_xy(self):
        """Calculate new XY positions for the rescaled ducts"""
        self.xy = {}
        self.xy['duct'] = np.zeros((self.n_duct,
                                    self.sc['n_sc']['duct']['total'],
                                    2))
        self.xy['byp'] = None
        if self.n_byp > 0:
            self.xy['byp'] = np.zeros((self.n_byp,
                                       self.sc['n_sc']['bypass']['total'],
                                       2))

        # Duct mesh dimensions
        dims = {}
        dims['duct'] = {'h': self.pin_pitch,
                        'w': self.duct_thickness,
                        'r': 2 * self.duct_thickness / np.sqrt(3)}
        if self.n_byp > 0:
            dims['byp'] = {'h': self.pin_pitch,
                           'w': self.byp_thickness,
                           'r': 2 * self.byp_thickness / np.sqrt(3)}

        # Handle outer duct first, then march inward over ducts/bypass
        # if necessary. For each, get the XY points along the hex
        # perimeter and then assign to subchannels. Note that the
        # because the edge rectangles are specified by the "bottom
        # left" vertex, they will share coordinates with the preceding
        # corner.
        edge_per_side = int(self.sc['n_sc']['duct']['edge'] / 6)
        hex_side_len = self.duct['ftf'][-1][0] / np.sqrt(3)
        start = self.UNIT_HEX_VERTEX * hex_side_len
        end = np.roll(self.UNIT_HEX_VERTEX * hex_side_len, -1, axis=0)
        xy = np.linspace(start, end, edge_per_side + 1, axis=1)
        self.xy['duct'][-1] = np.vstack(xy)

        for d in reversed(range(self.n_duct - 1)):
            # Bypass
            hex_side_len = self.duct['ftf'][d][1] / np.sqrt(3)
            start = self.UNIT_HEX_VERTEX * hex_side_len
            end = np.roll(self.UNIT_HEX_VERTEX * hex_side_len, -1, axis=0)
            xy = np.linspace(start, end, edge_per_side + 1, axis=1)
            self.xy['bypass'][d] = np.vstack(xy)
            # Duct
            hex_side_len = self.duct['ftf'][d][0] / np.sqrt(3)
            start = self.UNIT_HEX_VERTEX * hex_side_len
            end = np.roll(self.UNIT_HEX_VERTEX * hex_side_len, -1, axis=0)
            xy = np.linspace(start, end, edge_per_side + 1, axis=1)
            self.xy['duct'][d] = np.vstack(xy)

    def plot(self, data, lbnd=None, ubnd=None, middle=None,
             xy_shift=None, **kwargs):
        """Plot duct wall temperatures in the assembly

        Parameters
        ----------

        Return
        ------


        """
        if lbnd is None:
            lbnd = np.min(data[data > 0])

        if ubnd is None:
            ubnd = np.max(data)

        if middle is None:
            middle = np.average([lbnd, ubnd])

        if not kwargs.get('linewidth'):
            kwargs['linewidth'] = 0.0

        if kwargs['linewidth'] == 0.0:
            kwargs['linestyle'] = 'None'
        else:
            kwargs['linestyle'] = '-'

        if not kwargs.get('cmap'):
            kwargs['cmap'] = mpl.cm.jet

        if not kwargs.get('norm'):
            kwargs['norm'] = colors.TwoSlopeNorm(
                vmin=lbnd, vcenter=middle, vmax=ubnd)

        # isolate the patch kwargs
        patch_kwargs = {'cmap': kwargs['cmap'],
                        'norm': kwargs['norm'],
                        'linewidth': kwargs['linewidth'],
                        'linestyle': kwargs['linestyle']}

        # Setup the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        for i in range(len(self.duct['ftf'])):
            # 1. Add corner channels (hexagons)
            ax = self._add_corner_sc(ax, data, i, xy_shift,
                                     **patch_kwargs)
            # 2. Add edge channels (rectangles)
            ax = self._add_edge_sc(ax, data, i, xy_shift,
                                   **patch_kwargs)
            # 3. Add duct walls
            ax = self._add_duct_walls(ax, i, xy_shift)

        # 4. If requested, add pins
        # if kwargs.get('pins'):
        #     _alpha = 1.0
        #     if kwargs.get('pin_alpha'):
        #         _alpha = kwargs['pin_alpha']
        #     ax = self._add_pins(ax, color='1.0', alpha=_alpha,
        #                         xy_shift=xy_shift)

        # Format figure and return
        plt.axis('off')
        ax = self._set_ax_bnds(ax)
        txt = ''
        if kwargs.get('cbar_label'):
            txt = kwargs['cbar_label']
        ax = _add_colorbar(ax, txt, kwargs['cmap'], kwargs['norm'])
        return ax

    def _add_corner_sc(self, ax, data, duct_id, xy_shift=None, **kwargs):
        """Add duct corner subchannels to the existing axis

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot corner subchannels
        duct_id : int
            Indicate which duct to plot
        data : numpy.ndarray
            Subchannel temperature data for the whole assembly
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        lw : float
            Linewidth to apply to the subchannel patches

        Returns
        -------
        matplotlib.axes.Axes object
            With duct corner subchannel temperatures added

        Notes
        -----
        Duct corner subchannels are plotted with hexagons. Hex radius
        is half the duct width. Will need to be overlapped later.

        """
        # XY position of the hexagon needs to be centered on the corner
        # of the duct inner surface. We can know that position based on
        # the duct inner flat-to-flat distance.
        dftfi = self.duct['ftf'][duct_id][0]

        # Hex side length
        a = dftfi / np.sqrt(3)

        # Corner locations
        xy = np.array([[0.5 * dftfi, 0.5 * a],
                       [0.5 * dftfi, -0.5 * a],
                       [0.0, -a],
                       [-0.5 * dftfi, -0.5 * a],
                       [-0.5 * dftfi, 0.5 * a],
                       [0.0, a]])

        if xy_shift is not None:
            xy += xy_shift

        # Calculate "radius" of the hexagon based on duct thickness;
        # duct thickness is the flat-to-flat of the plotted hexagon
        dthick = 0.5 * (self.duct['ftf'][duct_id][1]
                        - self.duct['ftf'][duct_id][0])
        radius = dthick
        # radius = 0.01

        # Pull data from the array and downselect to requested duct.
        # Data array dimensions: N_duct x N_meshes; need to identify
        # which cols are for corners. Do this based on the number of
        # edge meshes between corners
        # First corner subchannel comes after the last edge mesh on the
        # first side of the hexagon. Need to find edges per side
        eps = self.sc['n_sc']['duct']['edge'] / 6
        cols = np.arange(eps, self.sc['n_sc']['duct']['total'], eps)
        cols = cols.astype('int')
        z = data[cols]

        # Set up the MPL objects
        tmp = [mpl.patches.RegularPolygon((xi, yi), 6, radius=radius)
               for xi, yi in zip(xy[:, 0], xy[:, 1])]
        # tmp = mpl.collections.PatchCollection(tmp, **kwargs)
        tmp = mpl.collections.PatchCollection(tmp, **kwargs)
        tmp.set_array(z)
        # ax.add_collection(tmp)

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax.add_collection(tmp)
        ax = self._set_ax_bnds(ax)
        plt.savefig('tmp2.png')
        plt.close('all')

        return ax

    def _add_edge_sc(self, ax, duct_id, data, xy_shift=None, **kwargs):
        """Add duct edge subchannels to existing axis

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot corner subchannels
        duct_id : int
            Indicate the duct to plot
        data : numpy.ndarray
            Subchannel temperature data for the whole assembly
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        lw : float
            Linewidth to apply to the subchannel patches

        Returns
        -------
        matplotlib.axes.Axes object
            With edge subchannel temperatures added

        Notes
        -----
        Duct edge subchannels are plotted with rectangles. Width is pin
        pitch, height is the duct thickness. No overlap is necessary;
        however, these overlap the corner hexagons.

        """
        # Pull data from the array and downselect to requested duct.
        # Data array dimensions: N_duct x N_meshes; need to identify
        # which cols are for edges. All are edges except for 6 corners.
        # First corner subchannel comes after the last edge mesh on the
        # first side of the hexagon. Need to find edges per side
        cols = list(np.arange(0, self.sc['n_sc']['duct']['total'], 1))
        for i in range(6):
            cols[i * 6 - 1] = -1
        cols = cols[cols >= 0]
        z = data[cols]

        # Duct edge XY positions for all ducts
        xy = self.sc['xy'][np.where(self.sc['type'] == 3)]

        # Downselect to desired duct
        xy = xy[(duct_id * self.sc['n_sc']['duct']['edge']):
                ((duct_id + 1) * self.sc['n_sc']['duct']['edge'])]

        # Shift to new center if necessary
        if xy_shift is not None:
            xy += xy_shift

        # Need to loop over sides to properly orient each rectangle
        # patches.Rectangle plots from the LOWER LEFT corner rather
        # than center - need to shift each rectangle based on angle.
        width = self.sc['radius'][1][0]
        height = 0.5 * (self.duct['ftf'][duct_id][1]
                        - self.duct['ftf'][duct_id][0])
        sc_edge_side = int(len(xy) / 6)
        shift = np.zeros((6, 2))
        shift[0] = [-height, 0.0]
        shift[1] = [-0.5 * height, 0.5 * width]
        shift[2] = [0.5 * height, np.sqrt(3) * 0.5 * height]
        shift[3] = [height, 0.0]
        shift[4] = [0.5 * height, -0.5 * width]
        shift[5] = [-0.5 * height, -np.sqrt(3) * 0.5 * height]
        edge_sq = []
        for i in range(6):
            side_xy = xy[i * sc_edge_side:(i + 1) * sc_edge_side]
            side_xy += shift[i]
            edge_sq += [mpl.patches.Rectangle(
                (xi, yi),
                width,
                height,
                angle=self.sc['angle'][1][i])
                for xi, yi in zip(side_xy[:, 0], side_xy[:, 1])]
        edge_sq = mpl.collections.PatchCollection(edge_sq, **kwargs)
        edge_sq.set_array(z)
        ax.add_collection(edge_sq)
        return ax

    def _add_duct_walls(self, ax, xy_shift=None):
        """Find duct dimensions to plot ducts as overlapping hexagons

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot pin duct walls
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        Returns
        -------
        matplotlib.axes.Axes object

        Notes
        -----
        This should be the first thing plotted, because it'll form
        the backdrop to everything plotted in the figure

        """
        xy = np.array([0.0, 0.0])
        if xy_shift is not None:
            xy = xy_shift

        # Plot ducts from the outside in
        for d in self.duct['ftf'][::-1]:
            # EDIT: Skip outer duct wall, would be covered by meshes
            # Outer duct wall: plot using gray hexagon
            # rad = d[1] / np.sqrt(3)
            # duct = [mpl.patches.RegularPolygon(xy, 6, radius=rad)]
            # duct = mpl.collections.PatchCollection(
            #     duct, facecolor='0.5', linewidth=0.5, edgecolor='k')
            # ax.add_collection(duct)

            # Inner duct wall: plot with white hexagon
            rad = d[0] / np.sqrt(3)
            duct = [mpl.patches.RegularPolygon(xy, 6, radius=rad)]
            duct = mpl.collections.PatchCollection(
                duct, facecolor='1.0', linewidth=0.5, edgecolor='k')
            ax.add_collection(duct)
        return ax


class PinPlot(AssemblyPlot):
    """Plot pin-by-pin data in assembly hex"""

    # Types of plots
    # pin data at a specific height for each pin
    # axial peak pin data for each pin
    # The above, but for each assembly with pins in the core

    def __init__(self, dassh_asm):
        """Initialize the PinPlot instance"""
        AssemblyPlot.__init__(self, dassh_asm)

    def plot(self, data, lbnd=None, ubnd=None, middle=None,
             xy_shift=None, **kwargs):
        """Plot the temperatures of the coolant subchannels

        Parameters
        ----------
        data : numpy.ndarray
            Subchannel temperature data for the whole assembly
        lbnd : float (optional)
            Lower bound for the color map
        ubnd : float (optional)
            Upper bound for the color map
        middle : float (optional)
            Middle value for the color map
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        cbar_label : str
            Label for the color bar
        lw : float
            Border line width to apply to the subchannel patches
        pins : boolean
            Indicate whether to plot pins over subchannels
        pin_alpha : float
            Indicate the opacity of the pin fill

        Returns
        -------
        matplotlib.axes.Axes object

        """
        # Set default arguments
        kwargs = self.parse_args(data, lbnd, ubnd, middle, **kwargs)

        # isolate the patch kwargs
        patch_kwargs = {'cmap': kwargs['cmap'],
                        'norm': kwargs['norm'],
                        'linewidth': kwargs['linewidth'],
                        'linestyle': kwargs['linestyle'],
                        'edgecolor': 'k'}

        # Setup the figure and add duct walls and pins
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')
        ax = self._add_duct_walls(ax, xy_shift)
        ax = self._add_pins(ax, data, xy_shift, **patch_kwargs)

        # Format figure and return
        plt.axis('off')
        ax = self._set_ax_bnds(ax)
        txt = ''
        if kwargs.get('cbar_label'):
            txt = kwargs['cbar_label']
        ax = _add_colorbar(ax, txt, kwargs['cmap'], kwargs['norm'])
        return ax

    def _add_pins(self, ax, data, xy_shift=None, **kwargs):
        """Generate matplotlib circles to show the fuel pins

        Parameters
        ----------
        ax : matplotlib.axes.Axes object
            Axis on which to plot pin circles
        data : numpy.ndarray
            Data to plot in pin locations
        xy_shift : numpy.ndarray (optional)
            New coordinate center for the assembly rather than (0, 0)

        Returns
        -------
        matplotlib.axes.Axes object

        """
        xy = copy.deepcopy(self.pin['xy'])
        if xy_shift is not None:
            xy += xy_shift

        circles = [plt.Circle((xi, yi), self.pin['radius'])
                   for xi, yi in zip(xy[:, 0], xy[:, 1])]
        c = mpl.collections.PatchCollection(circles, **kwargs)
        c.set_array(data)
        ax.add_collection(c)
        return ax


########################################################################
# PLOT FULL CORE MAPS
########################################################################


class CorePlot(object):
    """Container for general methods for whole-core plots"""

    def __init__(self, dassh_reactor, data_path='.'):
        """yolo"""
        # Store DASSH Reactor object
        self.reactor = dassh_reactor
        self.data_path = os.path.abspath(data_path)
        self._asm_pitch = dassh_reactor.asm_pitch
        self._n_ring = dassh_reactor.core.n_ring
        self._n_asm = dassh_reactor.core.n_asm

        # Calculate assembly center XY positions
        self.asm_xy = dassh_reactor.core.map_assembly_xy()

    def parse_args(self, data, lbnd, ubnd, middle, **kwargs):
        """Parse arguments for whole-core figures

        Parameters
        ----------
        data : dict
            Dictionary of numpy.ndarray
        lbnd : float (optional)
            Lower bound for the color map
        ubnd : float (optional)
            Upper bound for the color map
        middle : float (optional)
            Middle value for the color map

        """
        # If not specified, identify data lower bound,
        # midpoint, and upper bound based on data
        lbnd, ubnd, middle = _get_data_bnds(data, lbnd, ubnd, middle)

        if not kwargs.get('linewidth'):
            kwargs['linewidth'] = 0.0

        if not kwargs.get('edgecolor'):
            kwargs['edgecolor'] = 'k'

        if kwargs['linewidth'] == 0.0:
            kwargs['linestyle'] = 'None'
        else:
            kwargs['linestyle'] = '-'

        if not kwargs.get('cmap'):
            kwargs['cmap'] = mpl.cm.jet

        if not kwargs.get('norm'):
            kwargs['norm'] = colors.TwoSlopeNorm(
                vmin=lbnd, vcenter=middle, vmax=ubnd)

        return kwargs

    def _set_ax_bnds(self, ax, rings=None):
        """Set axis boundaries"""
        if rings is None:
            core_long_diag = (2 * self._n_ring - 1) * self._asm_pitch
        else:
            core_long_diag = (2 * rings - 1) * self._asm_pitch
        core_padded_half_diag = core_long_diag * 1.05 / 2
        ax.set_xlim([-core_padded_half_diag, core_padded_half_diag])
        ax.set_ylim([-core_padded_half_diag, core_padded_half_diag])
        return ax


class CoreHexPlot(CorePlot):
    """Plot values on an assembly-by-assembly basis (core-wide)

    Because there are quite a few special cases and this method
    doesn't share or inherit anything from the others, it's got
    quite a few more internal data-processing methods. I stuck
    them in here to keep the module neat.

    """

    def __init__(self, dassh_reactor, plot_data=None):
        """x"""
        # Pull attributes for later use
        self._reactor = dassh_reactor
        self._asm_pitch = dassh_reactor.asm_pitch
        self._n_ring = dassh_reactor.core.n_ring
        self._n_asm = dassh_reactor.core.n_asm

        # Calculate XY positions
        self.xy = dassh_reactor.core.map_assembly_xy()

        # Build hexagons
        diag = 2 * self._asm_pitch / np.sqrt(3)
        self.hex = [mpl.patches.RegularPolygon(
            (xi, yi), 6, radius=diag / 2)
            for xi, yi in zip(self.xy[:, 0], self.xy[:, 1])]

        # Store unit converters for length and temperature
        self.fwd_len_conv = \
            _get_forward_len_conv(dassh_reactor.units['length'])
        self.bwd_len_conv = \
            _get_backward_len_conv(dassh_reactor.units['length'])
        # Get unit converter for temperature to output user-requested units
        if plot_data is not None and plot_data['units'] is not None:
            user_temp_units = plot_data['units']
        else:
            user_temp_units = dassh_reactor.units['temperature']
        self.user_unit = user_temp_units
        self.temp_conv = _get_temp_conv(user_temp_units)

    def plot(self, data, lbnd=None, ubnd=None, middle=None,
             omit_nonvalue_rings=False, nonvalue=0.0, **kwargs):
        """Plot the requested data in each assembly in the core

        Parameters
        ----------
        data : numpy.ndarray
            Data to be plotted for each assembly in the core
        lbnd : float (optional)
            Lower bound for the color map
        ubnd : float (optional)
            Upper bound for the color map
        middle : float (optional)
            Midpoint for the color map
        omit_nonvalue_rings : boolean (optional)
            If outermost assembly rings have no value to plot,
            indicate whether to still plot them (as gray if False)
            or ignore them completely if True (default=False)
        nonvalue : float
            Indicate a condition for values to be ignored; any values
            less than this value will be ignored. As an example, then,
            nonvalue=0.0 ignores all negative values but included 0.0.

        kwargs
        ------
        cmap (optional) : str
            Indicate the matplotlib colormap to use in plotting
        cbar_label (optional) : str
            Label for the color bar
        data_label (optional) : boolean
            Indicate whether to label data
        data_label_fmt (optional) : str
            Formatter for the data labels
        """
        # Assemblies to ignore
        asm_to_ignore = []
        if 'ignore_assemblies' in kwargs.keys():
            asm_to_ignore = kwargs['ignore_assemblies']

        kwargs = self.parse_args(data, lbnd, ubnd, middle, **kwargs)
        # isolate the patch kwargs
        patch_kwargs = {'cmap': kwargs['cmap'],
                        'norm': kwargs['norm'],
                        'linewidth': kwargs['linewidth'],
                        'edgecolor': kwargs['edgecolor']}

        # Create figure
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        # Generate the hexagons
        hex_to_color = []
        to_plot = []
        hex_to_ignore = []
        for i in range(len(data)):
            if i in asm_to_ignore:
                continue
            elif data[i] >= nonvalue:
                to_plot.append(data[i])
                hex_to_color.append(self.hex[i])
            elif np.all(data[i:] < nonvalue) and omit_nonvalue_rings:
                break
            else:
                hex_to_ignore.append(self.hex[i])

        h1 = mpl.collections.PatchCollection(hex_to_color,
                                             **patch_kwargs)
        h2 = mpl.collections.PatchCollection(hex_to_ignore,
                                             facecolor='0.5',
                                             edgecolor=kwargs['edgecolor'],
                                             lw=kwargs['linewidth'])

        # Color the hexagons and add to the axes
        h1.set_array(np.array(to_plot))
        ax.add_collection(h1)
        ax.add_collection(h2)

        # Set axis boundaries based on the number of assemblies you
        # are plotting
        n_rings = None
        if omit_nonvalue_rings:
            n_rings = core.count_rings(len(to_plot))
        ax = self._set_ax_bnds(ax, rings=n_rings)

        # Add data labels
        if kwargs.get('data_label'):
            self._add_data_labels(fig, ax, data, nonvalue,
                                  kwargs.get('data_label_fmt'),
                                  asm_to_ignore)

        # Add colorbar and label if requested
        txt = ''
        if kwargs.get('cbar_label'):
            txt = kwargs['cbar_label']
        ax = _add_colorbar(ax, txt, kwargs['cmap'], kwargs['norm'])

        # Remove axes and return
        ax.axis('off')
        return ax

    def _add_data_labels(self, fig, ax, data, nv, fmt, ignore=[]):
        """Annotate hex plot with data labels"""
        txtwidth = 0.7 * (0.5 * np.sqrt(3) * self._asm_pitch)
        fontsize = 12
        if fmt is None:
            fmt = '{:0.1f}'
        for i in range(len(data)):
            if i not in ignore and data[i] >= nv:
                txt = ax.annotate(fmt.format(data[i]),
                                  self.xy[i],
                                  size=fontsize,
                                  ha='center',
                                  va='center',
                                  weight='bold')
                fontsize = self._auto_fit_fontsize(txt,
                                                   txtwidth,
                                                   None,
                                                   fig=fig,
                                                   ax=ax)

    def _auto_fit_fontsize(self, text, width, height, fig=None, ax=None):
        """Auto-decrease the fontsize of a text object.

        Parameters
        ----------
        text : matplotlib.text.Text
        width : float
            Allowed width in data coordinates
        height :float
            Allowed height in data coordinates

        Notes
        -----
        From: https://stackoverflow.com/questions/5320205

        """
        fig = fig or plt.gcf()
        ax = ax or plt.gca()

        # get text bounding box in figure coordinates
        renderer = fig.canvas.get_renderer()
        bbox_text = text.get_window_extent(renderer=renderer)

        # transform bounding box to data coordinates
        bbox_text = mpl.transforms.Bbox(
            ax.transData.inverted().transform(bbox_text))

        # evaluate fit and recursively decrease fontsize until text fits
        fits_width = bbox_text.width < width if width else True
        fits_height = bbox_text.height < height if height else True
        fs = text.get_fontsize()
        if not all((fits_width, fits_height)):
            text.set_fontsize(text.get_fontsize() - 1)
            fs = self._auto_fit_fontsize(text, width, height, fig, ax)

        # Return the new fontsize; pass it back so other data labels
        # don't have to go through these iterations.
        return fs

    # @staticmethod
    # def _count_rings(n_asm):
    #     """Given some number of assemblies, count hex rings"""
    #     nr = int(np.ceil(0.5 * (1 + np.sqrt(1 + 4 * (n_asm - 1) // 3))))
    #     return nr

    def make_power(self, dassh_reactor, plot_data):
        """Generate the total assembly power hex plot"""
        data = np.array([sum(list(a._power_delivered.values())) / 1e6
                         for a in dassh_reactor.assemblies])
        cbl = plot_data['cbar_label']
        if cbl is None:
            cbl = 'Power (MW)'
        self.plot(data,
                  lbnd=plot_data['cbar_lbnd'],
                  ubnd=plot_data['cbar_ubnd'],
                  middle=plot_data['cbar_mpnt'],
                  cmap=plot_data['cmap'],
                  cbar_label=cbl,
                  data_label=plot_data['data_label'])
        plt.savefig('CoreHexPlot_total_power.png',
                    bbox_inches='tight', dpi=plot_data['dpi'])
        plt.close()

    def make_axial_peak(self, dassh_reactor, plot_data, value):
        """Generate CoreHexPlot figure containing maximum temperature
        data for each assembly taken over entire axial space

        Parameters
        ----------
        dassh_reactor : DASSH Reactor object
        plot_data : dict
            Contains plot formatting requests
        value : str
            The type of data to plot

        Returns
        -------
        None

        """
        cbar_lab = self._get_cbar_label(plot_data, value)
        data = self._get_axial_peak_data(dassh_reactor, value)
        self.plot(self.temp_conv(data),
                  lbnd=plot_data['cbar_lbnd'],
                  ubnd=plot_data['cbar_ubnd'],
                  middle=plot_data['cbar_mpnt'],
                  cmap=plot_data['cmap'],
                  cbar_label=cbar_lab,
                  data_label=plot_data['data_label'],
                  omit_nonvalue_rings=plot_data['omit_nonvalue_rings'])
        plt.savefig(f'CoreHexPlot_{value}.png',
                    bbox_inches='tight', dpi=plot_data['dpi'])
        plt.close()

    def make_radial_peak_or_avg(self, dassh_reactor, plot_data, value):
        """Generate CoreHexPlot figure containing maximum temperature
        data for each assembly at a given axial position

        Parameters
        ----------
        dassh_reactor : DASSH Reactor object
        plot_data : dict
            Contains plot formatting requests
        value : str
            The type of data to plot

        Returns
        -------
        None

        """
        cbar_lab = self._get_cbar_label(plot_data, value)
        _dcol = self._get_csv_data_col(value)
        z_conv = [self.fwd_len_conv(zi) for zi in plot_data['z']]
        if 'max' in value:
            f = 'temp_maximum.csv'
        else:
            f = 'temp_average.csv'
        try:
            data = _load_data(f, z_conv)
        except FileNotFoundError:
            msg = 8 * '.' + f'File "{f}" not found, skipping {value}...'
            module_logger.log(30, msg)
            return
        for zi in data.keys():
            z_str = np.around(self.bwd_len_conv(zi), 2)
            if np.all(data[zi][:, _dcol] == 0):
                msg = (8 * '.' + 'No data to plot for CoreHexPlot '
                       '"{:s}" at z={:s}; skipping...')
                module_logger.log(30, msg.format(value, str(z_str)))
                continue
            else:
                self.plot(self.temp_conv(data[zi][:, _dcol]),
                          lbnd=plot_data['cbar_lbnd'],
                          ubnd=plot_data['cbar_ubnd'],
                          middle=plot_data['cbar_mpnt'],
                          cmap=plot_data['cmap'],
                          cbar_label=cbar_lab,
                          data_label=plot_data['data_label'],
                          omit_nonvalue_rings=plot_data['omit_nonvalue_rings'])
                fname = '_'.join(['CoreHexPlot',
                                  value,
                                  f'z={z_str}.png'])
                plt.savefig(fname, bbox_inches='tight',
                            dpi=plot_data['dpi'])
                plt.close()

    def _get_cbar_label(self, plot_data, value):
        """If no colorbar label is provided, generate one based on
        the other plot data"""
        cbar_lab = plot_data['cbar_label']
        if cbar_lab is None:
            unit = f'({_identify_user_units(self.user_unit)})'
            if 'max' in value:
                cbar_lab = 'Peak '
            elif 'avg' in value:
                cbar_lab = 'Avg. '
            else:
                raise ValueError('Do not understand CHP superlative')
            cbar_lab += ' '.join([self._get_data_key(value),
                                  'temperature',
                                  unit])
        return cbar_lab

    @staticmethod
    def _get_data_key(value):
        """x"""
        if 'coolant' in value:
            key = 'coolant'
        elif 'duct_mw' in value:
            key = 'duct MW'
        elif 'clad_mw' in value:
            key = 'clad MW'
        elif 'fuel_cl' in value:
            key = 'fuel CL'
        else:
            raise ValueError('Do not understand input')
        return key

    @staticmethod
    def _get_axial_peak_data(dassh_reactor, value):
        """Pull overall peak temperature data from DASSH reactor obj"""
        if 'coolant' in value:
            data = np.array([a._peak['cool'][0] for a in
                             dassh_reactor.assemblies])
        elif 'duct_mw' in value:
            data = np.array([np.max(a._peak['duct'], axis=0)[0]
                             for a in dassh_reactor.assemblies])
        elif 'clad_mw' in value:
            data = np.array([a._peak['pin']['clad_mw'][0]
                             if 'pin' in a._peak.keys() else -1.0
                             for a in dassh_reactor.assemblies])
        elif 'fuel_cl' in value:
            data = np.array([a._peak['pin']['fuel_cl'][0]
                             if 'pin' in a._peak.keys() else -1.0
                             for a in dassh_reactor.assemblies])
        else:
            raise ValueError('Do not understand input')
        return data

    @staticmethod
    def _get_csv_data_col(value):
        """Figure out which column in the CSV to pull data from"""
        if 'max' in value:
            _col = 1
        else:
            _col = 0
        if 'coolant' in value:
            _dcol = _hex_cols['coolant'][_col]
        elif 'duct_mw' in value:
            _dcol = _hex_cols['duct_mw'][_col]
        elif 'clad_mw' in value:
            _dcol = _hex_cols['clad_mw'][_col]
        elif 'fuel_cl' in value:
            _dcol = _hex_cols['fuel_cl'][_col]
        return _dcol


class CoreSubchannelPlot(CorePlot):
    """Plot individual temperatures on a core-wide basis"""

    def __init__(self, dassh_reactor, data_path='.'):
        """x"""
        CorePlot.__init__(self, dassh_reactor, data_path)

        # Create SubchannelPlot objects for each type of assembly
        self.scp = {}
        for asm in dassh_reactor.assemblies:
            if asm.name not in self.scp.keys():
                if asm.has_rodded:
                    self.scp[asm.name] = SubchannelPlot(asm)

    def plot(self, data, lbnd=None, ubnd=None, middle=None,
             pins=False, pin_alpha=1.0, rings=None, ignore_ur=False,
             **kwargs):
        """Plot subchannel temperatures for all assemblies in the core

        Parameters
        ----------
        data : numpy.ndarray
            Subchannel temperature data for every assembly in the
            core at the requested axial height; generally obtained
            from temp_coolant_int.csv
        lbnd : float (optional)
            Lower bound for the color map
        ubnd : float (optional)
            Upper bound for the color map
        middle : float (optional)
            Middle value for the color map
        pins : boolean (optional)
            Indicate whether to plot pins over subchannels
        pin_alpha : float (optional)
            Indicate the opacity of the pin fill
        rings : int (optional)
            Indicate how many assembly rings of the core to plot;
            default is to plot everything.

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        cbar_label : str
            Label for the color bar
        linewidth : float
            Border line width to apply to the subchannel patches

        Returns
        -------
        matplotlib.axes.Axes object

        """
        data = {'int': data}

        # Assemblies to ignore
        asm_to_ignore = []
        if 'ignore_assemblies' in kwargs.keys():
            asm_to_ignore = kwargs['ignore_assemblies']

        kwargs = self.parse_args(data, lbnd, ubnd, middle, **kwargs)

        # isolate the patch kwargs
        patch_kwargs = {'cmap': kwargs['cmap'],
                        'norm': kwargs['norm'],
                        'linewidth': kwargs['linewidth'],
                        'linestyle': kwargs['linestyle']}

        # Setup the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        for i in range(len(self.reactor.assemblies)):
            asm = self.reactor.assemblies[i]
            # Skip if assembly is not included in rings: once you
            # reach the maximum number of rings, should just break
            if rings is not None and asm.loc[0] > rings - 1:
                break

            # Get subset of data for specific assembly
            # NEED TO REMOVE DETAIL COLS
            asm_data = {}
            for key in data:
                if key != 'int':
                    continue
                else:
                    tmp = data[key][np.round(data[key][:, 0]) == asm.id]
                    try:
                        asm_data[key] = tmp[0, 3:]
                    except:
                        print('DASSH asm ID:', asm.id)
                        print('z (m)', tmp[0, 2])
                        print('data shape', tmp.shape)
                        print('n_asm in data', np.max(data['int'][:, 0]))
                        raise

            # Add subchannel temperatures to the plot
            # Ignore assemblies if asked to
            if asm.name in asm_to_ignore or asm.id in asm_to_ignore:
                # continue
                snp = SingleNodePlot(asm)
                ax = snp.plot(
                    ax, asm_data['int'][0], patch_kwargs['cmap'],
                    patch_kwargs['norm'], True, 0.0, self.asm_xy[i])

            # If only a single value is given, use the proper plotting
            # object for that; just create on the fly b/c it's small
            # elif np.sum(asm_data['int'] > 0) == 1:
            #    snp = SingleNodePlot(asm)
            #    ax = snp.plot(ax, asm_data['int'][0], patch_kwargs['cmap'],
            #                  patch_kwargs['norm'], ignore_ur, 0.0,
            #                  self.asm_xy[i])

            # Otherwise plot every subchannel in the assembly.
            else:
                ax = self._plot_asm(ax, asm.name, asm_data,
                                    self.asm_xy[i], pins,
                                    pin_alpha, **patch_kwargs)

        # Remove axes
        plt.axis('off')

        # Scale axis bounds to properly fit figure
        ax = self._set_ax_bnds(ax, rings)

        # Add colorbar and label, if desired
        txt = ''
        if kwargs.get('cbar_label'):
            txt = kwargs['cbar_label']
        ax = _add_colorbar(ax, txt, kwargs['cmap'], kwargs['norm'])

        return ax

    def _plot_asm(self, ax, asm, data, asm_xy, pins,
                  pin_alpha, **patch_kwargs):
        """Add subchannel temperature patches to axes

        Parameters
        ----------

        Returns
        -------

        """
        # 0. Add duct walls
        ax = self.scp[asm]._add_duct_walls(ax, asm_xy, lw=0.0)

        # 1. Add corner channels (hexagons)
        ax = self.scp[asm]._add_corner_sc(ax, data['int'], asm_xy,
                                          **patch_kwargs)

        # 2. Add edge channels (squares)
        ax = self.scp[asm]._add_edge_sc(ax, data['int'], asm_xy,
                                        **patch_kwargs)

        # 3. Add interior channels (triangles)
        ax = self.scp[asm]._add_int_sc(ax, data['int'], asm_xy,
                                       **patch_kwargs)

        # 4. If requested, add pins
        if pins:
            ax = self.scp[asm]._add_pins(ax, color='1.0',
                                         alpha=pin_alpha,
                                         xy_shift=asm_xy)

        # 5. Add bypass gap subchannels, if present
        # 6. Add gap subchannel temperatures
        return ax


class CorePinPlot(CorePlot):
    """Plot pin-by-pin temperatures on a core-wide basis"""

    def __init__(self, dassh_reactor, data_path='.'):
        """x"""
        CorePlot.__init__(self, dassh_reactor, data_path)

        # Create PinPlot objects for each type of assembly
        self.pp = {}
        for asm in dassh_reactor.assemblies:
            if asm.name not in self.pp.keys():
                if asm.has_rodded and hasattr(asm.rodded, 'pin_model'):
                    self.pp[asm.name] = PinPlot(asm)

    def plot(self, data, lbnd=None, ubnd=None, middle=None,
             rings=None, **kwargs):
        """Plot subchannel temperatures for all assemblies in the core

        Parameters
        ----------
        data : numpy.ndarray
            Pin temperature data for every assembly in the core at the
            requested axial height; generally from temp_pin.csv
        value : str
            Indicate the type of value to plot;
            {"clad_od", "clad_mw", "clad_id", "fuel_od", "fuel_cl"}
        lbnd : float (optional)
            Lower bound for the color map
        ubnd : float (optional)
            Upper bound for the color map
        middle : float (optional)
            Middle value for the color map
        rings : int (optional)
            Indicate how many assembly rings of the core to plot;
            default is to plot everything.

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        cbar_label : str
            Label for the color bar
        linewidth : float
            Border line width to apply to the subchannel patches

        Returns
        -------
        matplotlib.axes.Axes object

        """
        data = {'pins': data}
        kwargs = self.parse_args(data, lbnd, ubnd, middle, **kwargs)
        data = data['pins']

        # isolate the patch kwargs
        patch_kwargs = {'cmap': kwargs['cmap'],
                        'norm': kwargs['norm'],
                        'linewidth': kwargs['linewidth'],
                        'linestyle': kwargs['linestyle']}

        # Setup the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        for i in range(len(self.reactor.assemblies)):
            asm = self.reactor.assemblies[i]
            # Skip if assembly is not included in rings: once you
            # reach the maximum number of rings, should just break
            if rings is not None and asm.loc[0] > rings - 1:
                break

            # Get subset of data for specific assembly
            # NEED TO REMOVE DETAIL COLS
            asm_data = data[data[:, 0] == asm.id]
            asm_data = asm_data[:, -1]

            # Add pin temperatures to the plot; if no value is given,
            # no pin data for this assembly at this axial height
            if asm_data.size == 0:
                snp = SingleNodePlot(asm)
                ignore_ur = True
                ax = snp.plot(ax, 1.0, patch_kwargs['cmap'],
                              patch_kwargs['norm'], ignore_ur, 0.0,
                              self.asm_xy[asm.id])
            # Otherwise plot every pin in the assembly.
            else:
                ax = self.pp[asm.name]._add_duct_walls(
                    ax, self.asm_xy[asm.id], lw=0.0)
                ax = self.pp[asm.name]._add_pins(
                    ax, asm_data, self.asm_xy[asm.id], **patch_kwargs)

        # Remove axes
        plt.axis('off')

        # Scale axis bounds to properly fit figure
        ax = self._set_ax_bnds(ax, rings)

        # Add colorbar and label, if desired
        txt = ''
        if kwargs.get('cbar_label'):
            txt = kwargs['cbar_label']
        ax = _add_colorbar(ax, txt, kwargs['cmap'], kwargs['norm'])

        return ax

    def plot_FH(self, data, lbnd=None, ubnd=None, middle=None,
                rings=None, **kwargs):
        """Plot subchannel temperatures for all assemblies in the core

        Parameters
        ----------
        data : numpy.ndarray
            Pin temperature data for every assembly in the core at the
            requested axial height; generally from temp_pin.csv
        value : str
            Indicate the type of value to plot;
            {"clad_od", "clad_mw", "clad_id", "fuel_od", "fuel_cl"}
        lbnd : float (optional)
            Lower bound for the color map
        ubnd : float (optional)
            Upper bound for the color map
        middle : float (optional)
            Middle value for the color map
        rings : int (optional)
            Indicate how many assembly rings of the core to plot;
            default is to plot everything.

        kwargs
        ------
        cmap : matploblib.cm object
            Color map with which to color the subchannel temperatures
        norm : matplotlib.colors.TwoSlopNorm object, or another norm
            option from matplotlib.colors
        cbar_label : str
            Label for the color bar
        linewidth : float
            Border line width to apply to the subchannel patches

        Returns
        -------
        matplotlib.axes.Axes object

        """
        data = {'pins': data}
        kwargs = self.parse_args(data, lbnd, ubnd, middle, **kwargs)
        data = data['pins']

        # isolate the patch kwargs
        patch_kwargs = {'cmap': kwargs['cmap'],
                        'norm': kwargs['norm'],
                        'linewidth': kwargs['linewidth'],
                        'linestyle': kwargs['linestyle']}

        # Setup the figure
        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal')

        for asm in self.reactor.assemblies:
            # Skip if assembly is not included in rings: once you
            # reach the maximum number of rings, should just break
            if rings is not None and asm.loc[0] > rings - 1:
                break

            # Get subset of data for specific assembly
            # NEED TO REMOVE DETAIL COLS
            asm_data = data[data[:, 0] == asm.id]
            asm_data = asm_data[:, -1]

            # Add pin temperatures to the plot; if no value is given,
            # no pin data for this assembly at this axial height
            if asm_data.size == 0:
                if asm.name == 'test':
                    # hex_color = 'mediumorchid'
                    hex_color = (149 / 255, 221 / 255, 0)
                elif asm.name == 'control':
                    hex_color = 'darkgray'
                elif asm.name == 'reflector':
                    hex_color = 'lightgray'
                elif asm.name == 'shield':
                    hex_color = 'gray'
                else:  # Sodium
                    continue

                snp = SingleNodePlot(asm)
                ax = snp.plot_single_color(ax, hex_color, 0.0,
                                           self.asm_xy[asm.id])

            # Otherwise plot every pin in the assembly.
            else:
                ax = self.pp[asm.name]._add_duct_walls(
                    ax, self.asm_xy[asm.id], lw=0.0)
                ax = self.pp[asm.name]._add_pins(
                    ax, asm_data, self.asm_xy[asm.id], **patch_kwargs)

        # Remove axes
        plt.axis('off')

        # Scale axis bounds to properly fit figure
        ax = self._set_ax_bnds(ax, rings)

        # Add colorbar and label, if desired
        txt = ''
        if kwargs.get('cbar_label'):
            txt = kwargs['cbar_label']
        ax = _add_colorbar(ax, txt, kwargs['cmap'], kwargs['norm'])

        return ax


########################################################################
# GENERAL PLOTTING AND UTILITY METHODS
########################################################################


def _prepare_input(dassh_reactor, plot_data, file_to_load):
    """Prepare and store data necessary to make any figure

    Parameters
    ----------
    dassh_reactor : DASSH Reactor object
    plot_data : dict
        Subblock from the Plot subsection of the input data dict

    Returns
    -------
    dict
        Input and formatting parameters to store

    """
    # Axial points at which to grab data
    plot_data = copy.deepcopy(dict(plot_data))
    plot_data['z'] = np.array(plot_data['z'], dtype=float)
    fwd_len_conv = _get_forward_len_conv(dassh_reactor.units['length'])
    z_conv = fwd_len_conv(plot_data['z'])

    # Save backward length conv for fname generations
    bwd_len_conv = _get_backward_len_conv(dassh_reactor.units['length'])
    plot_data['bwd_len_conv'] = bwd_len_conv

    # Assembly IDs for which to grab data, if requested; only need
    # to modify if DIF3D index. In this case, user specifies DIF3D
    # index but we want DASSH index
    if plot_data['assembly_id'] is not None:
        plot_data['assembly_id'] = _sort_asm(dassh_reactor, plot_data)

    # Get unit converter for temperature to output user-requested units
    user_temp_units = plot_data['units']
    if user_temp_units is None:
        user_temp_units = dassh_reactor.units['temperature']
    temp_conv = _get_temp_conv(user_temp_units)

    # Format cbar_label
    if plot_data['cbar_label'] is None:
        plot_data['cbar_label'] = \
            f'Temperature ({_identify_user_units(user_temp_units)})'

    # Get data and apply unit conversion
    z_data = _load_data(file_to_load, z_conv, plot_data['assembly_id'])
    for k in z_data.keys():
        z_data[k][:, 3:] = temp_conv(z_data[k][:, 3:])
    plot_data['z_data'] = z_data

    # Return for use in object
    return plot_data


def _get_data_bnds(data, lbnd, ubnd, mpnt):
    """If necessary, get data lower/upper bounds and midpoint"""
    if isinstance(data, dict):
        if lbnd is None:
            lbnd = np.min([np.min(data[k][:, 3:][data[k][:, 3:] > 0])
                           for k in data.keys()])

        if ubnd is None:
            ubnd = np.max([np.max(data[k][:, 3:])
                           for k in data.keys()])
    else:
        if lbnd is None:
            lbnd = np.min(data[data > 0])

        if ubnd is None:
            ubnd = np.max(data)
    if mpnt is None:
        mpnt = np.average([lbnd, ubnd])
    return lbnd, ubnd, mpnt


def _sort_asm(dassh_reactor, plot_data):
    """Order the assemblies requested by the user"""
    # Get the assembly indices; also convert to python (base-0 index)
    # user_asm_list = []
    # if dassh_reactor._options['dif3d_idx']:
    #     for user_asm in plot_data['assembly_id']:
    #         for asm in dassh_reactor.assemblies:
    #             if asm.dif3d_id == user_asm - 1:
    #                 user_asm_list.append(asm.id)
    #                 break
    # else:
    user_asm_list = [a_id - 1 for a_id in plot_data['assembly_id']]
    return user_asm_list


def _add_colorbar(ax, text, cmap, norm):
    """Format a AssemblyPlot figure/axes

    Parameters
    ----------
    ax : matplotlib Axes object
        Axes to which to add colorbar
    text : str
        Colorbar label
    cmap : matploblib.cm object
        Color map with which to color the subchannel temperatures
    norm : matplotlib.colors.TwoSlopNorm object, or another norm
        option from matplotlib.colors

    Returns
    -------
    matplotlib Axes object

    """
    divider = make_axes_locatable(ax)
    color_axis = divider.append_axes("right", size="5%", pad=0.1)
    cbar_colors = mpl.cm.ScalarMappable(norm=norm,
                                        cmap=cmap)
    cbar = plt.colorbar(cbar_colors, cax=color_axis)
    cbar.set_label(text, fontsize=11)
    return ax


def _get_forward_len_conv(user_len_unit):
    """Get converter for user-requested length units to m"""
    if user_len_unit == 'm':  # set up pass-through fxn
        def conv(l):
            return l
    else:
        conv = utils.get_length_conversion(user_len_unit, 'm')
    # Return whichever function you need
    return conv


def _get_backward_len_conv(user_len_unit):
    """Get converter from m to user-requested length units"""
    if user_len_unit == 'm':  # set up pass-through fxn
        def conv(l):
            return l
    else:
        conv = utils.get_length_conversion('m', user_len_unit)
    # Return whichever function you need
    return conv


def _get_temp_conv(user_temp_unit):
    """Get converter for temperature units from K to user-request"""
    if user_temp_unit.lower() in utils._degK:  # use pass-through fxn
        def conv(t):
            return t
    else:
        conv = utils.get_temperature_conversion('k', user_temp_unit)
    # Return whichever function you need
    return conv


def _identify_user_units(user_temp_unit):
    """Get abbreviation for user temp units to display on cbar"""
    user_unit = user_temp_unit.lower()
    if user_unit in utils._degC:
        return '˚C'
    elif user_unit in utils._degF:
        return '˚F'
    elif user_unit in utils._degK:
        return 'K'
    else:
        raise ValueError(f'do not understand user '
                         'temperature unit {user_temp_unit}')


def _load_data(file, z_user, asmlist=None):
    """Read and return the dumped temperatures needed for plotting

    Parameters
    ----------
    file : str
        Path to relevant DASSH data dump file
    dassh_reactor : DASSH Reactor object
        Contains assembly information
    z_user : list
        List of z-values for plots requested by the user
    asmlist : list (optional)
        List of assemblies for which to pull data (default=None,
        in which case all assemblies are pulled)

    Returns
    -------
    dict
        Contains data to be plotted (values) at each axial point (keys)

    """
    # Need a single list of all z-values to pull in
    with open(file, 'r') as f:
        z_data = np.unique(np.loadtxt(f, delimiter=',', usecols=(1)))
    z_to_load, interp_dict = _interp_z(z_data, z_user)

    # Read in numpy array selectively
    with open(file, 'r') as f:
        data = np.loadtxt(
            _filter_lines(f, z_to_load, asmlist),
            delimiter=',')

    # Connect z-data in numpy array with user request
    z_dict = {}
    for z in z_user:
        if z in data[:, 1]:
            z_dict[z] = data[data[:, 1] == z]
        else:
            z1, x1, z2, x2 = interp_dict[z]
            z_dict[z] = (data[data[:, 1] == z1] * x1
                         + data[data[:, 1] == z2] * x2)

    return z_dict


def _interp_z(z_available, z_requested):
    """Get the z-values to pull and the interpolation between them"""
    _interp = {}
    z_to_load = []
    for z in z_requested:
        if z in z_available:
            z_to_load.append(z)
        elif z > max(z_available):
            if z - max(z_available) < 1e-3:
                z_to_load.append(max(z_available))
            else:
                msg = (f'Requested z-point ({z} m) greater '
                       'than maximum z-point available in '
                       f'data file ({max(z_available)} m)')
                module_logger.error(msg)
            sys.exit(1)
        else:
            idx = np.argmin(np.abs(z_available - z))
            if z_available[idx] > z:
                z1 = z_available[idx - 1]
                z2 = z_available[idx]
            else:
                z1 = z_available[idx]
                z2 = z_available[idx + 1]
            z_to_load += [z1, z2]
            x = (z2 - z) / (z2 - z1)
            _interp[z] = [z1, x, z2, 1 - x]
    return z_to_load, _interp


def _filter_lines(f, zlist, asmlist=None):
    for i, line in enumerate(f):
        if float(line[25:49]) in zlist:
            if asmlist is None:
                yield line
            else:
                # Convert Python (base-0) index --> base-1 index
                if int(float(line[:24])) in asmlist:
                    yield line


########################################################################
