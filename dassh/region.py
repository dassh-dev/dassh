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
date: 2020-12-09
author: matz
Methods to describe the axial regions in an assembly
"""
########################################################################
import numpy as np
import dassh


class DASSH_Region(object):
    """."""

    def __init__(self, n_node_coolant, node_area_coolant, n_node_duct,
                 node_area_duct, n_bypass=0, node_area_byp=None):
        """."""
        self.temp = {}
        self.area = {}
        self.total_area = {}
        self._pressure_drop = 0.0

        # Primary "interior" coolant; neglects bypass
        self.temp['coolant_int'] = np.ones(n_node_coolant)
        self.area['coolant_int'] = node_area_coolant
        self.total_area['coolant_int'] = np.sum(node_area_coolant)

        # Bypass coolant, if applicable
        if n_bypass > 0:
            self.temp['coolant_byp'] = np.ones((n_bypass, n_node_duct))
            self.area['coolant_byp'] = node_area_byp
            self.total_area['coolant_byp'] = np.sum(node_area_byp, axis=1)

        # Duct mid-wall
        self.temp['duct_mw'] = np.ones((n_bypass + 1, n_node_duct))
        self.area['duct_mw'] = node_area_duct
        self.total_area['duct_mw'] = np.sum(node_area_duct, axis=1)

        # Duct surface; won't ever be averaging
        self.temp['duct_surf'] = np.ones((n_bypass + 1, 2, n_node_duct))

        # Coolant energy balance tracker (W)
        self.ebal = {}
        self.ebal['power_added'] = 0.0
        self.ebal['from_duct'] = 0.0
        self.ebal['per_hex_side'] = np.zeros(6)
        if n_bypass > 0:
            self.ebal['from_duct_byp'] = np.zeros(n_bypass)
            self.ebal['per_hex_side_byp_in'] = np.zeros((n_bypass, 6))
            self.ebal['per_hex_side_byp_out'] = np.zeros((n_bypass, 6))
        self.ebal['temp_in'] = 0.0
        self.ebal['temp_out'] = 0.0
        self.ebal['temp_rise'] = 0.0

        # Energy balance utilities: track hex side energy balance
        duct_idx = np.arange(0, n_node_duct, 1)
        duct_idx.shape = ((6, int(n_node_duct / 6)))
        self._ebal_duct_idx = np.zeros((6, int(n_node_duct / 6) + 1),
                                       dtype='int')
        self._ebal_duct_idx[:, 1:] = duct_idx
        self._ebal_duct_idx[:, 0] = np.roll(duct_idx[:, -1], 1)

    @property
    def pressure_drop(self):
        """Total pressure drop across the region"""
        return self._pressure_drop

    @property
    def is_rodded(self):
        """Check if a region has fuel rod parameters or not"""
        if hasattr(self, 'n_pin'):
            return True
        else:
            return False

    @property
    def avg_coolant_int_temp(self):
        """Calculate volume-average of individual coolant node
        temperatures (in the primary interior region)

        Notes
        -----
        This is overridden in the RoddedRegion object but is used
        in the SingleNodeHomogeneous object

        """
        tot = np.sum(self.temp['coolant_int']
                     * self.area['coolant_int'])
        return tot / self.total_area['coolant_int']

    @property
    def avg_coolant_byp_temp(self):
        """Calculate the volume-average of temperature in each bypass
        gap between assembly ducts at the last axial level"""
        tot = np.sum(self.temp['coolant_byp']
                     * self.area['coolant_byp'], axis=1)
        return tot / self.total_area['coolant_byp']

    @property
    def avg_coolant_temp(self):
        """Return the overall average coolant temperature, including
        bypass gap coolant

        Notes
        -----
        This is overridden in the RoddedRegion object. Therefore, it
        might not be used anywhere at all.

        """
        if 'coolant_byp' not in self.temp.keys():
            return self.avg_coolant_int_temp
        else:
            tot = (self.avg_coolant_int_temp
                   * self.total_area['coolant_int'])
            tot_area = self.total_area['coolant_int']
            tot += np.sum((self.avg_coolant_byp_temp *
                           self.total_area['coolant_byp']))
            tot_area += np.sum(self.total_area['coolant_byp'])
            return tot / tot_area

    @property
    def avg_duct_mw_temp(self):
        """Calculate volume-average of individual duct node
        temperatures"""
        tot = np.sum(self.temp['duct_mw']
                     * self.area['duct_mw'], axis=1)
        return tot / self.total_area['duct_mw']

    @property
    def duct_outer_surf_temp(self):
        """Get the outer surface temperature of the outer duct at
        last axial level"""
        # Duct surface temp is a list: length = n_duct
        # Inside each list are two sublists, corresponding to
        #   (1) inside, and (2) outside surface temperatures
        # Inside each sublist is an array, shape = j x n_sc(duct)
        #   j = axial level; n_sc = number of subchannels
        # therefore, the outer duct outer surface temperature at the
        #   last axial level is: (last duct) (outside) (last level)
        # Indices:
        #   - "last (outer) duct"
        #   - "last (outer) surfaces"
        #   - "all meshes"
        return self.temp['duct_surf'][-1, -1, :]

    def activate(self, previous_reg):
        """Activate region based on temperatures in previous region

        Parameters
        ----------


        Notes
        -----
        It is assumed that the coolant is mixed across region
        transitions. Therefore, the coolant temperature in every
        node in the new region should be equal to the overall
        average coolant temperature from the previous region.

        Bypass coolant channels are not preserved across regions. It
        is assumed that the bypass gap coolant, if present, will be
        mixed with the primary interior coolant before entering the
        new axial region.

        """
        # Make sure region is not yet activated
        msg = "Cannot activate region if it has been already activated"
        assert (np.allclose(self.temp['coolant_int'], 1)
                and np.allclose(self.temp['duct_mw'], 1)
                and np.allclose(self.temp['duct_surf'], 1)), msg

        # Claim the DASSH Material objects from the previous region
        # self.coolant = previous_reg.coolant
        # self.duct = previous_reg.duct
        # if hasattr(previous_reg, 'struct'):
        #     self.struct = previous_reg.struct

        # Coolant temperatures: assume mixing
        avg_cool_temp = previous_reg.avg_coolant_temp
        avg_cool_int_temp = previous_reg.avg_coolant_int_temp
        self.temp['coolant_int'] *= avg_cool_temp
        if 'coolant_byp' in self.temp.keys():
            self.temp['coolant_byp'] *= avg_cool_temp

        # Duct temperatures: outermost duct is always continued into
        # the next region, but inner ducts may be added or subtracted
        # - If the duct continues in the next region, approximate the
        #   temperatures on the new mesh based on temperatures in the
        #   previous region (e.g. the outermost duct).
        # - If the duct is removed, nothing.
        # - If a duct is added, set surface/midwall temperatures equal
        #   to the average interior coolant temperature.

        # Outer duct temperatures - approximate
        self.temp['duct_mw'][-1] = \
            dassh.reactor.approximate_temps(
                previous_reg.x_pts,
                previous_reg.temp['duct_mw'][-1],
                self.x_pts)
        for s in range(2):
            self.temp['duct_surf'][-1, s] = \
                dassh.reactor.approximate_temps(
                    previous_reg.x_pts,
                    previous_reg.temp['duct_surf'][-1, s],
                    self.x_pts)

        # Any other ducts: set equal to average interior coolant
        # temperature
        if len(self.temp['duct_mw']) > 1:
            for d in range(len(self.temp['duct_mw']) - 1):
                self.temp['duct_mw'][d] *= avg_cool_int_temp
                self.temp['duct_surf'][d, :] *= avg_cool_int_temp

        # Duct temperatures: apply average temperature
        # for d in range(0, len(self.temp['duct_mw']) - 1):
        #     self.temp['duct_mw'][d] *= avg_cool_temp
        #     self.temp['duct_surf'][d] *= avg_cool_temp
        #
        # # Outer duct mw, surface temp should stay the same
        # self.temp['duct_mw'][-1] *= previous_reg.avg_duct_mw_temp[-1]
        # self.temp['duct_surf'][-1] = np.average(previous_reg
        #                                         .duct_outer_surf_temp)

        # Energy balance - initiate temperature tracking
        self.ebal['temp_in'] = avg_cool_temp

    def update_ebal(self, q_in, q_from_duct):
        """Update the region energy-balance tallies

        Parameters
        ----------
        q_in : float
            Power added by heating in pins and coolant (W)
        q_from_duct : numpy.ndarray
            Power (W) added to coolant through contact with duct wall
            for each coolant/duct connection

        Returns
        -------
        None

        """
        self.ebal['power_added'] += q_in
        self.ebal['from_duct'] += np.sum(q_from_duct)
        energy_per_side = q_from_duct[self._ebal_duct_idx]
        energy_per_side[:, (0, -1)] *= 0.5
        self.ebal['per_hex_side'] += np.sum(energy_per_side, axis=1)

    def update_ebal_byp(self, byp_idx, q_duct_in, q_duct_out):
        """Update the bypass gap coolant energy balance tallies

        Parameters
        ----------
        byp_idx : int
            Bypass ID number
            (from 0 to n_bypass, where 0 is closest to rod bundle)
        q_from_duct : numpy.ndarray
            Power (W) added to coolant through contact with duct wall
            for each coolant/duct connection for each bypass gap
            (shape = n_bypass x n_byp_sc)

        Returns
        -------
        None

        """
        # From duct to bypass coolant total
        self.ebal['from_duct_byp'][byp_idx] += \
            np.sum(q_duct_in + q_duct_out)
        # Energy per hex side; from inner duct to bypass coolant
        eps = q_duct_in[self._ebal_duct_idx]
        eps[:, (0, -1)] *= 0.5
        self.ebal['per_hex_side_byp_in'][byp_idx] += np.sum(eps, axis=1)
        # Energy per hex side; from outer duct to bypass coolant
        eps = q_duct_out[self._ebal_duct_idx]
        eps[:, (0, -1)] *= 0.5
        self.ebal['per_hex_side_byp_out'][byp_idx] += np.sum(eps, axis=1)

########################################################################
