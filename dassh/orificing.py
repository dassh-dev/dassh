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
date: 2022-03-09
author: matz
DASSH orificing optimization

To do
-----
- Parallel execution (on kookie) has super scattered logging. Should
    execute with logger context as error only - no sweep updates, but
    that's probably okay.
- Enable parallelism for the generation of Reactor objects in the
    initial assembly grouping step.

"""
########################################################################
import copy
import os
import numpy as np
import dassh
from dassh.logged_class import LoggedClass


class Orificing(LoggedClass):
    """Group assemblies and distribute coolant flow rate to optimize
    user-selected parameter-of-interest.

    Parameters
    ----------
    dassh_input : DASSH_Input object
        Contains inputs to describe model and control optimization

    """
    _VARPOW_FILES = ('VARPOW.out',
                     'varpow_MatPower.out',
                     'varpow_MonoExp.out')

    def __init__(self, dassh_input):
        LoggedClass.__init__(self, 0, 'dassh.Orificing')
        self._base_input = dassh_input
        self.orifice_input = dassh_input.data['Orificing']
        coolant_name = dassh_input.data['Core']['coolant_material']
        self.coolant = dassh_input.materials[coolant_name.lower()]
        self.t_in = dassh_input.data['Core']['coolant_inlet_temp']
        self._dp_limit = np.zeros(self.orifice_input['n_groups'])
        self._recycle = dassh_input.data['Orificing']['recycle_results']

        # Setup lookup keys
        x = 'value_to_optimize'
        if self.orifice_input[x] == 'peak coolant temp':
            self._opt_keys = ('cool', None)  # DASSH rx lookup keys
            self._opt_col = 5
        elif self.orifice_input[x] == 'peak clad ID temp':
            self._opt_keys = ('pin', 'clad_id')
            self._opt_col = 8
        elif self.orifice_input[x] == 'peak clad MW temp':
            self._opt_keys = ('pin', 'clad_mw')
            self._opt_col = 7
        elif self.orifice_input[x] == 'peak fuel temp':
            self._opt_keys = ('pin', 'fuel_cl')
            self._opt_col = 10
        else:  # Here for safety; shouldn't be raised b/c input limits
            msg = 'Do not understand optimiation variable; given '
            msg += f'"{self.orifice_input[x]}"'
            self.log('error', msg)

    def optimize(self):
        """Perform that orificing optimzation"""
        # Initial setup
        msg = "DASSH orificing optimization calculation"
        self.log('info', msg)
        k = 'value_to_optimize'
        msg = f'Value to minimize: "{self.orifice_input[k]}"'
        self.log('info', msg)
        k = 'n_groups'
        msg = f"Requested number of groups: {self.orifice_input[k]}"
        self.log('info', msg)

        msg = "Grouping assemblies"
        self.log('info', msg)
        with dassh.logged_class.LogStreamContext(40):
            self.group_by_power()

        msg = "Performing single-assembly parametric calculations"
        self.log('info', msg)
        self._parametric = {}
        with dassh.logged_class.LogStreamContext(40):
            self.run_parametric()

        # msg = "Performing perfect-orificing calculation"
        # self.log('info', msg)
        # data_perfect = self.run_dassh_perfect()

        # Iterate to converge to coolant flow distribution among groups
        # that maintains overall bulk outlet temperature constraint and
        # minimizes optimization variable
        # iter_data = data_perfect
        iter_data = None
        t_out = None
        for i in range(self.orifice_input['iteration_limit']):
            iter_data, summary_data = \
                self._do_iter(i + 1, iter_data, t_out)
            t_out = summary_data[-1, 0]
            print(summary_data)
            if self._check_flow_convergence(summary_data):
                break
            if np.any(self._dp_limit):
                msg = ("Breaking optimization iteration due to "
                       "incurred pressure drop limit")
                self.log('warning', msg)
                break

        # Write results to CSV files
        self.write_results(iter_data)

    def _do_iter(self, iter, data_prev=None, t_out=None):
        """Update coolant flow distributions between orifice groups
        and run DASSH sweeps for each timestep"""
        # if iter == 2 and data_prev is not None:
        #     if self.orifice_input['regroup_option_tol'] is not None:
        #         self.log('info', 'Regrouping...')
        #         self.regroup(data_prev, verbose=True)
        # if iter >= 2 and data_prev is not None:
        #     if self.orifice_input['regroup_option_tol'] is not None:
        #         self.log('info', 'Regrouping...')
        #         self.regroup(data_prev, verbose=True)
        if self.orifice_input['regroup'] != 'never' and iter >= 2:
            if ((self.orifice_input['regroup'] == 'once' and iter == 2)
                    or self.orifice_input['regroup'] == 'every'):
                self.log('info', 'Checking grouping...')
                self.regroup(data_prev, verbose=True)
        # Distribute flow among groups to meet bulk coolant temp
        msg = f"Iter {iter}: Distributing coolant flow among groups"
        self.log('info', msg)
        m, tlim = self.distribute(data_prev, t_out)
        tmp, nx = np.unique(self.group_data[:, 2], return_counts=True)
        m_sorted_by_grp = m[np.argsort(self.group_data[:, 2])]
        inds = np.unique(m_sorted_by_grp, return_index=True)[1]
        mx = np.array([m_sorted_by_grp[i] for i in sorted(inds)])
        print('T_opt_max: ', tlim)
        print('Asm per group: ', nx)
        print('Flow rates: ', mx)
        print('Total flow rate: ', np.sum(m))

        # Run DASSH iteration calculation
        msg = f"Iter {iter}: Running DASSH with orificed assemblies"
        self.log('info', msg)
        iter_data = self.run_dassh_orifice(iter, m)
        summary_data = self._summarize_group_data(iter_data)
        return iter_data, summary_data

    def _summarize_group_data(self, res):
        """Generate table of group- and core- average and maximum
        temperatures

        Parameters
        ----------
        res : numpy.ndarray
            Results for all timesteps from "_get_dassh_results" method

        Returns
        -------
        numpy.ndarray
            One row per group; last row is for core-wide data.
            Columns:
            1. Bulk coolant temperature
            2. Peak bulk coolant temperature (among asm in group)
            3. Average temperature of variable being optimized
            4. Peak temperature of variable being optimized

        """
        n_group = max(self.group_data[:, 2].astype(int))
        timesteps = np.unique(res[:, 0])
        _summary = np.zeros((timesteps.shape[0], n_group + 1, 4))
        for t in range(timesteps.shape[0]):
            res_t = res[res[:, 0] == timesteps[t]]
            for i in range(n_group + 1):
                indices = (self.group_data[:, 2].astype(int) == i)
                # Bulk coolant outlet temperature
                _summary[t, i, 0] = np.average(res_t[indices, 4])
                _summary[t, i, 1] = np.max(res_t[indices, 4])
                # Optimization variable
                _summary[t, i, 2] = \
                    np.average(res_t[indices, self._opt_col])
                _summary[t, i, 3] = \
                    np.max(res_t[indices, self._opt_col])
        # Finish averaging/maximizing
        summary = np.zeros((n_group + 2, 4))
        for i in range(n_group + 1):
            summary[i, 0] = np.average(_summary[:, i, 0])
            summary[i, 1] = np.max(_summary[:, i, 1])
            summary[i, 2] = np.average(_summary[:, i, 2])
            summary[i, 3] = np.max(_summary[:, i, 3])
        # Core-total average/maxima
        # Bulk coolant outlet temperature
        summary[-1, 0] = (np.sum(res[:, 4] * res[:, 3])
                          / np.sum(res[:, 3]))
        summary[-1, 1] = np.max(res[:, 4])
        # Optimization variable
        summary[-1, 2] = (np.sum(res[:, self._opt_col] * res[:, 3])
                          / np.sum(res[:, 3]))
        summary[-1, 3] = np.max(res[:, self._opt_col])
        return summary

    def _check_flow_convergence(self, summary_data):
        """Check whether to stop iterating on flow distribution"""
        avg = np.average(summary_data[:, -1])
        max_dev = np.max(summary_data[:, -1] - avg)
        max_rdev = max_dev / (avg - self.t_in)
        print('Convergence: ', max_rdev)
        if max_rdev <= self.orifice_input['convergence_tol']:
            return True
        else:
            return False

    ####################################################################
    # DASSH INPUT PREPARATION AND EXECUTION
    ####################################################################

    def run_parametric(self, n_pts=12):
        """Perform parametric calculations on a single assembly to
        generate data to use for initial orifice grouping

        Parameters
        ----------
        n_pts : int
            Number of points to use in the parametric sweep

        Returns
        -------
        tuple
            1. List of numpy.ndarray containing parametric sweep data
            2. List of lists containing assembly IDs that apply to each
               set of parametric sweep data in (1)

        """
        wd = os.path.join(self._base_input.path, '_parametric')
        self._parametric = {}
        # Instantiate DASSH Reactor - need to know which assemblies are
        # associated with which parametric data.
        lookup_rx = os.path.join(self._base_input.path, '_power')
        try:
            rx = dassh.reactor.load(
                os.path.join(lookup_rx, 'dassh_reactor.pkl'))
        except FileNotFoundError:
            lookup_rx = os.path.join(lookup_rx, 'timestep_1')
            rx = dassh.reactor.load(
                os.path.join(lookup_rx, 'dassh_reactor.pkl'))

        # For each type of assembly to be grouped, pull a matching
        # assembly object from the Reactor (the first matching
        # assembly you find); calculate average power of all
        # assemblies of that type.
        asm_obj = []
        asm_power = []
        asm_ids = []
        asm_names = []
        for i in range(len(self.orifice_input['assemblies_to_group'])):
            power_profiles = {'pin': [], 'duct': [], 'cool': [], 'avg': []}
            name = self.orifice_input['assemblies_to_group'][i]
            asm_names.append(name)
            for a in rx.assemblies:
                if a.name == name:
                    # Add the object to the list if you haven't yet
                    if len(asm_obj) == 0 or asm_obj[-1].name != a.name:
                        asm_obj.append(a)
                    # Pull its power profiles to average them later
                    asm_ids.append([a.id, i])
                    power_profiles['pin'].append(a.power.pin_power)
                    power_profiles['duct'].append(a.power.duct_power)
                    power_profiles['cool'].append(a.power.coolant_power)
                    power_profiles['avg'].append(a.power.avg_power)

            # Average the power profiles
            avg_power_profiles = {}
            for k in power_profiles.keys():
                if power_profiles[k][0] is not None:
                    tmp = np.array(power_profiles[k])
                    avg_power_profiles[k] = np.average(tmp, axis=0)
                else:
                    avg_power_profiles[k] = None

            # Assign average power profiles to assembly object
            assert asm_obj[-1].name == name
            asm_obj[-1].power.pin_power = avg_power_profiles['pin']
            asm_obj[-1].power.duct_power = avg_power_profiles['duct']
            asm_obj[-1].power.coolant_power = avg_power_profiles['cool']
            asm_obj[-1].power.avg_power = avg_power_profiles['avg']
            asm_obj[-1].total_power = \
                asm_obj[-1].power.calculate_total_power()
            asm_power.append(asm_obj[-1].total_power)

        asm_ids = np.array(asm_ids, dtype=int)
        self._parametric['asm_ids'] = asm_ids[asm_ids[:, 0].argsort()]
        self._parametric['asm_names'] = asm_names

        # For each assembly, append data to a list
        self._parametric['data'] = []

        # Try to look up data first
        found = False
        lookup_data = [f'data_{asm_name}.csv' for asm_name in
                       self.orifice_input['assemblies_to_group']]
        if self._recycle and os.path.exists(wd):
            wd_files = os.listdir(wd)
            if all([f in wd_files for f in lookup_data]):
                found = True
                for f in lookup_data:
                    self._parametric['data'].append(
                        np.loadtxt(os.path.join(wd, f), delimiter=','))
                return

        # If you can't find it, need to compute it
        if not found:
            # Set up subdirectory for this calculation
            os.makedirs(
                os.path.join(self._base_input.path, '_parametric'),
                exist_ok=True)
            # Re-use precomputed power distributions
            for f in self._VARPOW_FILES:
                src = os.path.abspath(os.path.join(lookup_rx, f))
                dassh.utils._symlink(src, os.path.join(wd, f))

            # Create some shortcuts for optimization and lookup keys
            x = ('Assignment', 'ByPosition')
            data = []
            # Loop over Assembly objects to run parametric calcs
            for i in range(len(asm_obj)):
                # Set up a generic single-assembly input from the original
                inp_1asm = self._setup_input_parametric(
                    asm_obj[i].id,
                    asm_obj[i].name,
                    asm_obj[i].loc,
                    asm_power[i])
                # Initialize data array
                # Columns:  1) Power (MW) / Flow rate (kg/s)
                #           2) Power (MW)
                #           3) Flow rate (kg/s)
                #           4) Target peak temperature (K)
                _data = np.zeros((n_pts, 5))
                _data[:, 0] = np.geomspace(0.05, 1.0, n_pts)  # MW / (kg/s)
                _data[:, 1] = asm_power[i]  # Watts
                _data[:, 2] = asm_power[i] / 1e6 / _data[:, 0]  # kg/s
                for j in range(_data.shape[0]):
                    # Find active assembly position and update it
                    for a in range(len(inp_1asm.data[x[0]][x[1]])):
                        if inp_1asm.data[x[0]][x[1]][a] == []:
                            continue
                        else:
                            inp_1asm.data[x[0]][x[1]][a][2] = \
                                {'flowrate': _data[j, 2]}
                            break
                    r1a = dassh.Reactor(inp_1asm, calc_power=False)
                    r1a.assemblies[0].power.pin_power = \
                        asm_obj[i].power.pin_power
                    r1a.assemblies[0].power.duct_power = \
                        asm_obj[i].power.duct_power
                    r1a.assemblies[0].power.coolant_power = \
                        asm_obj[i].power.coolant_power
                    r1a.assemblies[0].power.avg_power = \
                        asm_obj[i].power.avg_power
                    r1a.temperature_sweep()
                    _data[j, 3] = r1a.assemblies[0].pressure_drop
                    if self._opt_keys[1] is not None:
                        _data[j, 4] = r1a.assemblies[0]._peak[
                            self._opt_keys[0]][
                                self._opt_keys[1]][0]
                    else:
                        _data[j, 4] = r1a.assemblies[0]._peak[
                            self._opt_keys[0]][0]
                    del r1a
                # Save parametric data for this assembly type to CSV
                _datapath = os.path.join(
                    inp_1asm.path, f'data_{asm_obj[i].name}.csv')
                np.savetxt(_datapath, _data, delimiter=',')
                data.append(_data)
        self._parametric['data'] = data

    def run_dassh_perfect(self):
        """Execute DASSH assuming perfect orificing for each assembly
        to have desired coolant temperature rise

        Returns
        -------
        numpy.ndarray
            Results from the DASSH calculation

        """
        # Try and find pre-existing outputs before rerunning new cases
        wd_path = os.path.join(self._base_input.path, '_perfect')
        data_path = os.path.join(wd_path, 'data.csv')
        found = False
        if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
            found = True
            results = np.loadtxt(data_path, delimiter=',')
        else:
            try:
                results = self._get_dassh_results(wd_path)
                if len(results) > 0:
                    found = True
                    np.savetxt(data_path, results, delimiter=',')
            except (OSError, FileNotFoundError):
                found = False
                pass
        # If you didn't find results, run DASSH
        if not found:
            os.makedirs(wd_path, exist_ok=True)
            args = {'save_reactor': True,
                    'verbose': False,
                    'no_power_calc': True}
            dassh_inp = self._setup_input_perfect()
            dassh.__main__.run_dassh(dassh_inp, args)
            results = self._get_dassh_results(dassh_inp.path)
            np.savetxt(data_path, results, delimiter=',')
        return results

    def run_dassh_orifice(self, iter, mfr):
        """Execute DASSH with determined orifice groups and flow rates

        Parameters
        ----------
        iter : int
            Iteration index to label temporary directory
        mfr : numpy.ndarray
            Mass flow rates for each assembly

        Returns
        -------
        numpy.ndarray
            Results from the DASSH calculation

        """
        # Try and find pre-existing outputs before rerunning new cases
        wd_path = os.path.join(self._base_input.path, f'_iter{iter}')
        data_path = os.path.join(wd_path, 'data.csv')
        found = False
        if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
            found = True
            results = np.loadtxt(data_path, delimiter=',')
        else:
            try:
                results = self._get_dassh_results(wd_path)
                if len(results) > 0:
                    found = True
                    np.savetxt(data_path, results, delimiter=',')
            except (OSError, FileNotFoundError):
                found = False
                pass
        # If you didn't find results or don't want them, run DASSH
        if not found or self._recycle is False:
            os.makedirs(wd_path, exist_ok=True)
            # Try to skip the power calculation by using ones you've
            # precalculated from previous iterations
            found = self._find_precalculated_power_dist(wd_path)
            args = {'save_reactor': True,        # Save Reactor object
                    'verbose': False,            # Don't print stuff
                    'no_power_calc': not found}  # Do the power calc?
            dassh_inp = self._setup_input_orifice(mfr)
            dassh_inp.path = wd_path
            dassh.__main__.run_dassh(dassh_inp, args)
            results = self._get_dassh_results(dassh_inp.path)
            np.savetxt(data_path, results, delimiter=',')
        return results

    def _setup_input_parametric(self, id, name, loc, power):
        """Set up a generic DASSH input structure to run for pre-
        optimization parametric sweep"""
        inp = self._base_input.clone()

        # Remove all unused assembly types
        for k in inp.data['Assembly'].keys():
            if k != name:
                del inp.data['Assembly'][k]

        # Turn off inter-assembly heat transfer
        inp.data['Core']['gap_model'] = None
        inp.data['Core']['bypass_fraction'] = 0.0

        # Eliminate any temperatures that would be dumped to CSV
        for k in inp.data['Setup']['Dump']:
            if k == 'interval':
                inp.data['Setup']['Dump'][k] = None
            else:
                inp.data['Setup']['Dump'][k] = False

        # Set total power equal to single assembly average value
        inp.data['Power']['total_power'] = power

        # Remove all timesteps other than the first one - they are
        # not necessary for the parametric calculation
        inp.timepoints = 1
        cccc = ['pmatrx', 'geodst', 'ndxsrf', 'znatdn',
                'labels', 'nhflux', 'ghflux']
        for k in cccc:
            inp.data['Power']['ARC'][k] = \
                [inp.data['Power']['ARC'][k][0]]

        # Eliminate orificing optimization input
        inp.data['Orificing'] = False

        # No parallelism for the parametric calculation
        inp.data['Setup']['parallel'] = False

        # Change the path to the proper subdir
        inp.path = os.path.join(inp.path, '_parametric')

        # Set up assignment so that there's only one assembly. Need
        # to include empty lists for all empty assembly positions
        # through the ring that has the preserved assembly.
        n_positions = 3 * (loc[0]) * (loc[0] + 1) + 1
        tmp = []
        for i in range(n_positions):
            if i == id:
                tmp.append(inp.data['Assignment']['ByPosition'][i])
            else:
                tmp.append([])
        inp.data['Assignment']['ByPosition'] = tmp
        return inp

    def _setup_input_perfect(self):
        """Using new flow rates, create new DASSH input"""
        inp = self._base_input.clone()
        # Eliminate orificing optimization input
        inp.data['Orificing'] = False
        # Set input path
        inp.path = os.path.join(inp.path, '_perfect')
        # Set up assignment to give desired outlet temp to each asm
        for i in range(len(inp.data['Assignment']['ByPosition'])):
            if inp.data['Assignment']['ByPosition'][i]:
                inp.data['Assignment']['ByPosition'][i][2] = \
                    {'outlet_temp':
                        self.orifice_input['bulk_coolant_temp']}
        return inp

    def _setup_input_orifice(self, m_asm):
        """Using new flow rates, create new DASSH input"""
        # Setup input to have perfect orificing, then overwrite relevant
        # assemblies to create orifice groups with determined flow rate
        inp = self._setup_input_perfect()
        inp.path = ''
        asm_id = self.group_data[:, 0].astype(int)
        for i in range(self.group_data.shape[0]):
            inp.data['Assignment']['ByPosition'][asm_id[i]][2] = \
                {'flowrate': m_asm[i]}
        # Next, go over nongrouped assemblies and give them flow
        # rates based on average power across all timesteps (saved
        # in self._ng_power)
        if hasattr(self, '_ng_power'):
            coolant = inp.materials[inp.data['Core']['coolant_material']]
            for i in range(self._ng_power.shape[0]):
                asm_id = int(self._ng_power[i, 0])
                mfr = dassh.utils.Q_equals_mCdT(
                    self._ng_power[i, 1],
                    inp.data['Core']['coolant_inlet_temp'],
                    coolant,
                    t_out=self.orifice_input['bulk_coolant_temp'])
                inp.data['Assignment']['ByPosition'][asm_id][2] = \
                    {'flowrate': mfr}
        return inp

    def _find_precalculated_power_dist(self, wdpath):
        """See if you can link previously calculated power
        distributions rather than recalculating"""
        # Look for power distributions in "_power" or "_iter1"
        linked = 0
        expected = 0
        found = False
        for dir in ("_power", "_perfect", "_iter1"):
            p = os.path.abspath(os.path.join(wdpath, '..', dir))
            if os.path.exists(p):
                found = True
                # Get VARPOW files from each timestep subdir
                for f in os.listdir(p):
                    if (os.path.isdir(os.path.join(p, f))
                            and 'timestep' in f):
                        destpath = os.path.join(wdpath, f)
                        if not os.path.exists(destpath):
                            os.makedirs(destpath, exist_ok=True)
                        expected += 3
                        for ff in self._VARPOW_FILES:
                            src = os.path.join(p, f, ff)
                            if os.path.exists(src):
                                dest = os.path.join(destpath, ff)
                                dassh.utils._symlink(src, dest)
                                linked += 1
                # Get VARPOW files that aren't in subdir, if applicable
                if any(['varpow' in x for x in os.listdir(p)]):
                    expected += 3
                    for ff in self._VARPOW_FILES:
                        src = os.path.join(p, ff)
                        if os.path.exists(os.path.join(p, ff)):
                            dest = os.path.join(wdpath, ff)
                            dassh.utils._symlink(src, dest)
                            linked += 1
            if found:
                break
        if linked > 0 and linked == expected:
            return True
        else:
            return False

    def _get_dassh_results(self, wdpath):
        """Get DASSH results for every timestep"""
        results = []
        for f in os.listdir(wdpath):
            if (os.path.isdir(os.path.join(wdpath, f))
                    and 'timestep' in f):
                p = os.path.join(os.path.join(wdpath, f),
                                 'dassh_reactor.pkl')
                if os.path.exists(p):
                    t = f.split('_')[-1]
                    r = dassh.reactor.load(p)
                    results.append(self._read_dassh_results(r, t))
                    del r
            elif f == 'dassh_reactor.pkl':
                r = dassh.reactor.load(os.path.join(wdpath, f))
                results.append(self._read_dassh_results(r, 0))
                del r
            else:
                continue
        if len(results) > 0:
            results = np.vstack(results)
        return results

    def _read_dassh_results(self, dassh_rx, timestep):
        """Pull DASSH results from Reactor object"""
        data = []
        for a in dassh_rx.assemblies:
            if a.name in self.orifice_input['assemblies_to_group']:
                d = [timestep,
                     a.id,
                     a.total_power,
                     a.flow_rate,
                     a.avg_coolant_temp,
                     a._peak['cool'][0]]
                if 'pin' in a._peak.keys():
                    d += [a._peak['pin']['clad_od'][0],
                          a._peak['pin']['clad_mw'][0],
                          a._peak['pin']['clad_id'][0],
                          a._peak['pin']['fuel_od'][0],
                          a._peak['pin']['fuel_cl'][0]]
                data.append(d)
        data = np.array(data, dtype=float)
        return data

    ####################################################################
    # GROUPING
    ####################################################################

    def group_by_power(self):
        """Group assemblies by power or linear power

        Returns
        -------
        numpy.ndarray
            Assembly parameters and orifice groups

        Notes
        -----
        Creates Reactor object for each timestep; uses this to extract
        the grouping parameter (total power or linear power)

        """
        if self.orifice_input['value_to_optimize'] == \
                'peak coolant temp':
            group_by = 'power'
        else:
            group_by = 'linear_power'

        # Use the _power_to_grp attribute to calculate grouping
        self._get_power(group_by)
        group_data = self._group(self._power_to_grp)
        self.group_data = group_data[group_data[:, 0].argsort()]

    def _get_power(self, group_by='linear_power'):
        """Get power (or linear power) to use as grouping parameter

        Parameters
        ----------
        dassh_rx : DASSH Reactor object
            Contains details about the reactor core and assemblies
        asm_to_group : tuple or list
            Iterable containing strings with the names of all assemblies
            that should be included in the orificing optimization
        group_by : string (optional)
            Must be either "power" or "linear_power"; depends on
            orificing optimization variable

        Returns
        -------
        numpy.ndarray
            Array (N_asm x 2) with assembly IDs (col 1) and group-
            parameter values (col 2)

        Notes
        -----
        Creates Reactor objects to generate power distributions for
        use in grouping algorithm. This is basically the first bit
        of __main__.run_dassh(), but it extracts power distribution
        info instead of doing the temperature sweep

        NEW (2022-01-20): This function also collects total power
        for the assemblies that are not grouped so that they can
        be assigned a constant flow rate across all timesteps

        """
        self._power = []
        self._lin_power = []
        self._ng_power = []

        # For each timestep, collect assembly power and power
        # parameter (either integral power or peak linear power)
        # to use in orifice grouping
        need_subdir = False
        if self._base_input.timepoints > 1:
            need_subdir = True
        for i in range(self._base_input.timepoints):
            wdir = os.path.join(self._base_input.path, '_power')
            if need_subdir:
                wdir = os.path.join(wdir, f'timestep_{i + 1}')
            rx_path = os.path.join(wdir, 'dassh_reactor.pkl')
            if self._recycle and os.path.exists(rx_path):
                dassh_rx = dassh.reactor.load(rx_path)
            else:
                dassh_rx = dassh.Reactor(self._base_input,
                                         calc_power=True,
                                         path=wdir,
                                         timestep=i,
                                         write_output=False)
                dassh_rx.save()

            # Go through each assembly and get the power
            _power = []
            _lin_power = []
            _id = []
            _ng_power = []
            _ng_id = []
            for a in dassh_rx.assemblies:
                if a.name in self.orifice_input['assemblies_to_group']:
                    _id.append(a.id)
                    _power.append(a.total_power)
                    _lin_power.append(
                        a.power.calculate_avg_peak_linear_power())
                else:  # Keep ungrouped asm power to calculate const FR
                    _ng_id.append(a.id)
                    _ng_power.append(a.total_power)
            self._power.append(np.array((_id, _power)).T)
            self._lin_power.append(np.array((_id, _lin_power)).T)
            self._ng_power.append(np.array((_ng_id, _ng_power)).T)

        # Take average for each assembly
        tmp = np.average([x[:, 1] for x in self._power], axis=0)
        self._power = np.array((self._power[0][:, 0], tmp)).T
        tmp = np.average([x[:, 1] for x in self._lin_power], axis=0)
        self._lin_power = np.array((self._lin_power[0][:, 0], tmp)).T
        # Take average for ungrouped assemblies, if any
        if np.any([y for y in self._ng_power if y.size > 0]):
            tmp = np.average([x[:, 1] for x in self._ng_power], axis=0)
            self._ng_power = np.array((self._ng_power[0][:, 0], tmp)).T
        else:  # all are grouped, don't need this.
            del self._ng_power

        # Save one of the above as the power to use in grouping
        if group_by == 'linear_power':
            self._power_to_grp = self._lin_power
        elif group_by == 'power':
            self._power_to_grp = self._power
        else:  # Here for safety, should never be raised
            msg = ('Argument "group_by" must be "power" or '
                   + f'"linear_power"; input {group_by} not '
                   + 'recognized')
            self.log('error', msg)

    def _group(self, params):
        """Divide assemblies into orifice groups based on some parameter

        Parameters
        ----------
        params : numpy.ndarray
            The parameter to use in the grouping
            Column 1: assembly index
            Column 2: parameter value

        Returns
        -------
        numpy.ndarray
            Array (N_asm x 3) containing assembly indices, parameter
            values, and group IDs

        """
        params = params[params[:, 1].argsort()][::-1]
        cutoff = copy.deepcopy(
            self.orifice_input['group_cutoff'])
        cutoff_delta = copy.deepcopy(
            self.orifice_input['group_cutoff_delta'])
        n_grp = self.orifice_input['n_groups'] + 1
        iter = 0
        while n_grp != self.orifice_input['n_groups'] and iter < 1000:
            g = 0
            grp_param = [[params[0, 1]]]  # Add first assembly to Group 1
            for i in range(1, params.shape[0]):
                if self._check_new_group(grp_param[g],
                                         params[i, 1],
                                         cutoff):
                    g += 1
                    grp_param.append([])
                grp_param[g].append(params[i, 1])
            n_grp = len(grp_param)
            # If you have more groups than requested, relax the cutoff
            # between groups - will get fewer groups on the next ieration
            # Note: n_grp is Python index!
            if n_grp > self.orifice_input['n_groups'] - 1:
                cutoff += cutoff_delta
            # If you have fewer groups than requested, tighten the cutoff
            # between groups - will get more groups on the next iteration.
            # NOTE: whereas the above adjustment is meant to be incremental,
            # this one needs to be big - from here, we want to end up with
            # more groups than requested so that we can walk incrementally
            # from there. Hopefully we don't need this condition.
            # Note: n_grp is Python index!
            if n_grp < self.orifice_input['n_groups'] - 1:
                cutoff /= 10.0
            # Update iteration index
            iter += 1

        # Check iteration index; if not converged, raise error
        if iter >= 1000 and n_grp != self.orifice_input['n_groups'] - 1:
            msg = ('Grouping not converged; please adjust '
                   + '"group_cutoff" and "group_cutoff_delta" '
                   + 'parameters')
            self.log('error', msg)

        # Attach grouping to group parameter data and return
        group_data = np.zeros((params.shape[0], 3))
        group_data[:, :2] = params
        group_data[:, 2] = [i for i in range(len(grp_param))
                            for j in range(len(grp_param[i]))]
        return group_data

    @staticmethod
    def _check_new_group(group_param, next_param, param_delta):
        """Check whether the assembly fits with the previous group or
        should be the first assembly in the next group

        Parameters
        ----------
        group_param : list
            Group-parameters for each assembly in the active group
        next_param : float
            Group-parameter for the assembly that might be added to group
        param_delta : float
            Cutoff value to determine whether assembly fits in group

        Returns
        -------
        Boolean
            True if the assembly fits in the group; False if not

        """
        updated_params = group_param + [next_param]
        group_min = min(updated_params)
        group_max = max(updated_params)
        group_avg = sum(updated_params) / len(updated_params)
        if (group_max - group_min) / group_avg > param_delta:
            return True
        else:
            return False

    ####################################################################
    # DISTRIBUTE COOLANT FLOW AMONG GROUPS
    ####################################################################

    def distribute(self, res_prev=None, t_out_prev=None):
        """Distribute flow to minimize optimization variable

        Parameters
        ----------
        res_prev : numpy.ndarray (optional)
            Results of the latest DASSH sweep for each timestep
            (default = None)
        t_out_prev : float (optional)
            Average of previous DASSH sweep outlet temps for each timestep
            (default = None)

        Returns
        -------
        tuple
            1. numpy.ndarray : contains new mass flow rates for each
                               assembly being grouped
            2. float : max estimate for optimization variable

        """
        # Prepare data parametric sweep for interpolation
        xy = []
        if res_prev is not None:
            # Use flow rate vs. T_opt
            for i in range(len(self._parametric['data'])):
                tmp = self._parametric['data'][i]
                tmp = tmp[tmp[:, 2].argsort()]
                xy.append(tmp[:, (2, -1)])
            # Compare previous interpolated estimate with results to
            # get a corrective factor.
            ratio = self._calc_corrective_ratio(xy, res_prev)
        else:
            # If only using the parametric results with no previous sweep
            # data, the parametric results will indicate that all groups
            # of each assembly type get the same flow rate to minimize
            # T_opt. Need to distinguish in some way --> use power to
            # flow ratio instead. That way, you are accounting for the
            # higher temperatures expected in higher power assemblies.
            for i in range(len(self._parametric['data'])):
                xyi = np.zeros((self._parametric['data'][i].shape[0], 2))
                xyi[:, 0] = self._parametric['data'][i][:, 0]
                xyi[:, 0] *= 1e6  # Parametric data is MW/(kg/s)
                xyi[:, 1] = self._parametric['data'][i][:, -1]
                xy.append(xyi[xyi[:, 0].argsort()])
            # No previous results, so no corrective ratio to apply
            ratio = 1.0

        # Total mass flow rate to achieve bulk coolant temperature
        # If first iteration, estimate based on total power
        if res_prev is None:
            m_total = dassh.Q_equals_mCdT(
                np.sum(self._power[:, 1]),
                self.t_in,
                self.coolant,
                t_out=self.orifice_input['bulk_coolant_temp'])
        # Otherwise, use total mass flow rate from the previous sweep
        else:
            m_total = np.sum(
                res_prev[res_prev[:, 0] == res_prev[0, 0]][:, 3])
        # Scale total mass flow rate based on bulk coolant temp result
        if t_out_prev is not None:
            # If T_previous < T_target, m_new < m_previous
            dt_prev = t_out_prev - self.t_in
            dt_trgt = self.orifice_input['bulk_coolant_temp'] - self.t_in
            m_total *= dt_prev / dt_trgt

        # Determine m_lim based on pressure drop limit
        # for each asm type being orificed
        m_lim = None
        if self.orifice_input['pressure_drop_limit']:
            dp_limit = self.orifice_input['pressure_drop_limit'] * 1e6
            m_lim = np.zeros(len(self._parametric['data']))
            for i in range(m_lim.shape[0]):
                m_lim[i] = np.interp(
                    dp_limit,
                    self._parametric['data'][i][:, 3][::-1],
                    self._parametric['data'][i][:, 2][::-1])

        # First guess - give all groups an average flow rate.
        m = np.ones(self.group_data.shape[0])
        m *= m_total / self.group_data.shape[0]
        # Initialize delta: factors used to "guess" next flow rates
        d_optvar = np.ones(self.orifice_input['n_groups'])
        # Initialize iteration parameters
        convergence = 2.0
        iter = 0
        tol = 1.0
        iter_lim = 50
        # dp_warning = False
        while convergence > tol and iter < iter_lim:
            # Update flow rates
            m_remaining = m_total
            for g in range(self.orifice_input['n_groups'] - 1):
                m_new = m[self.group_data[:, -1] == g] * d_optvar[g]
                if m_lim is not None:
                    asm_type_in_grp = self._parametric['asm_ids'][
                        self.group_data[:, -1] == g, 1]
                    m_lim_grp = m_lim[asm_type_in_grp]
                    if np.any(m_new > m_lim_grp):
                        m_new[:] = np.min(m_lim_grp)
                        # dp_warning = True
                        self._dp_limit[g] = 1
                m[self.group_data[:, -1] == g] = m_new
                m_remaining -= np.sum(m_new)
            last_idx = (self.group_data[:, -1] ==
                        self.orifice_input['n_groups'] - 1)
            m[last_idx] = m_remaining / np.count_nonzero(last_idx)

            # Calculate group-total flow rates
            group_total_fr = np.array([
                np.sum(m[self.group_data[:, -1] == g])
                for g in range(self.orifice_input['n_groups'])
            ])

            # Estimate optimization variable based on new flow rates;
            # Get max values per group and an "average" value weighted
            # by flow rates and number of assemblies in the group
            optvar = self._estimate_optvar(m, xy, res_prev, ratio)
            group_max = np.array([
                np.max(optvar[self.group_data[:, -1] == g])
                for g in range(self.orifice_input['n_groups'])
            ])
            avg_max = np.sum(group_total_fr * group_max)
            avg_max /= np.sum(group_total_fr)
            # Difference with average; use to get next guess
            d_optvar = (group_max - self.t_in) / (avg_max - self.t_in)
            # Check convergence
            convergence = np.sqrt(np.sum((group_max - avg_max)**2))
            iter += 1

        # print(group_max)
        # print(avg_max)
        # print('Delta (want close to 1.0): ', d_optvar)
        # print(f'Convergence (want less than {tol}): ', convergence)
        # Check conservation of mass - this should never fail
        if not abs(np.sum(m) - m_total) < 1e-6:
            self.log('error', "Error: Mass flow rate not conserved")
        # Report whether the pressure drop mass flow rate limit was met
        if np.any(self._dp_limit):  # if dp_warning:
            msg = ("Warning: Peak mass flow rate restricted to "
                   + "accommodate user-specified pressure drop limit")
            self.log('warning', msg)
        # If multiple groups hit the pressure drop limit, the problem
        # is poorly posed (i.e. the limit is too tight to achieve adequate
        # cooling). Doubt this situation would ever occur, not sure what
        # to do about it if it does.
        if np.sum(self._dp_limit) > 1:
            msg = "Error: Multiple groups constrained by pressure drop limit"
            self.log('error', msg)
        # Raise error if we did not achieve requested number of groups
        if np.unique(m).shape[0] != self.orifice_input['n_groups']:
            msg = ("Error: Flow allocation did not achieve "
                   + "requested number of orifice groups")
            self.log('error', msg)
        # Return results
        return m, max(group_max)

    def _estimate_optvar(self, mfr, xy, res_prev=None, ratio=None):
        """Estimate the variable-to-optimize based on the single-assembly
        parametric data and the latest DASSH results

        Parameters
        ----------
        mfr : numpy.ndarray
            New mass flow rates for which to estimate the value of the
            optimization variable for each assembly
        xy : list
            List of numpy.ndarray containing the x (flow rate) and y
            (optimization variable) values from the parametric sweep
            data for each assembly type being grouped
        res_prev (optional) : numpy.ndarray
            Data from the previous DASSH iteration; used to correct
            estimate with scalar ratio for each assembly
            (default = 0)
        ratio (optional) : float or numpy.ndarray
            Corrective ratio between the previous interpolated estimate
            for peak temperature and the result obtained by DASSH (if
            float, will equal 1.0).

        Returns
        -------
        numpy.ndarray
            Estimate for the optimization variable for each assembly

        Notes
        -----
        In the "distribute" method, this function is called with
        "ratio" as an argument. In the "regroup" method, it is
        called without it.

        """
        if res_prev is not None:
            # Use mass flow rate from the previous sweep as the
            # independent variable for the interpolation
            x_interp = mfr
            # Calculate a corrective ratio to improve estimate of
            # dependent variable based on data from previous sweep
            if ratio is None:
                ratio = self._calc_corrective_ratio(xy, res_prev)
        else:
            # If no previous sweep data, do the interpolation using
            # power-to-flow ratio as the independent variable
            x_interp = self._power[:, 1] / mfr
            if ratio is None:
                ratio = 1.0

        # Interpolate for the new flow rate
        new = []
        for i in range(len(xy)):
            new.append(np.interp(x_interp, xy[i][:, 0], xy[i][:, 1]))
        new = np.array(new).T
        new = new[np.arange(new.shape[0]),
                  self._parametric['asm_ids'][:, 1]]
        return new * ratio

    def _calc_corrective_ratio(self, xy, res_prev):
        """If previous results are available, calculate the ratio
        between interpolated estimate and the previous result as a
        corrective factor to apply to peak temperature estimation"""
        # Calculate a corrective ratio to improve estimate of
        # dependent variable based on data from previous sweep
        y_data = res_prev[:, self._opt_col]
        interpolated_pts = []
        for i in range(len(xy)):
            interpolated_pts.append(
                np.interp(res_prev[:, 3], xy[i][:, 0], xy[i][:, 1]))
        interpolated_pts = np.array(interpolated_pts)
        r = y_data / interpolated_pts
        # Take maximum over all cycles for each assembly
        r = np.array([np.max(r[:, res_prev[:, 1] == i], axis=1)
                      for i in np.unique(res_prev[:, 1])])
        # Choose the appropriate ratio for the right asm type
        r = r[np.arange(r.shape[0]), self._parametric['asm_ids'][:, 1]]
        return r

    ####################################################################
    # REGROUPING
    ####################################################################

    def regroup(self, data, verbose=False):
        """Based on the results from the previous iteration,
        look into whether the assemblies should be regrouped.

        Parameters
        ----------
        data : numpy.ndarray
            Data from previous iteration; see below for columns
        verbose : boolean
            Print messages about regrouping

        Returns
        -------
        None

        Notes
        -----
        Columns in parameter "data":
            1.  Timestep
            2.  Assembly ID
            3.  Total power
            4.  Flow rate
            5.  Bulk coolant outlet temperature
            6.  Peak coolant temperature
            7.  Clad outer temperature
            8.  Clad MW temperature
            9.  Clad inner temperature
            10. Fuel outer temperature
            11. Fuel CL temperature

        Parameters obtained from DASSH Input file:
        regroup_tol : float
            Tolerance used to decide whether an assembly "fits"
            well enough in its group or whether the code should
            try to shuffle it. If None (default), skip regrouping.
        improvement_tol : float
            Tolerance used to assess the improvement in grouping.
            If the improvement is sufficiently significant,
            approve it; otherwise, don't do it.

        """
        regroup_tol = self.orifice_input['regroup_option_tol']
        improve_tol = self.orifice_input['regroup_improvement_tol']
        if regroup_tol is None:
            return

        # Prepare data from parametric sweep for interpolation; use
        # flow rate vs. T_opt
        xy = []
        for i in range(len(self._parametric['data'])):
            tmp = self._parametric['data'][i]
            tmp = tmp[tmp[:, 2].argsort()]
            xy.append(tmp[:, (2, -1)])

        # For each assembly, find data at timestep that yields max
        # value of optimization variable
        n_asm = np.unique(data[:, 1]).shape[0]
        group_mfr = np.unique(data[:, 3])[::-1]
        n_group = group_mfr.shape[0]
        # Columns: Asm ID, Group, Power, FR, Peak Temp
        peak = np.zeros((n_asm, 5))
        asm_ids = np.unique(data[:, 1])
        for a in range(n_asm):
            tmp = data[data[:, 1] == asm_ids[a]]
            grp = self.group_data[a, 2]
            row = np.argmax(tmp[:, 5])
            power, fr, optvar = tmp[row, [2, 3, self._opt_col]]
            peak[a] = [asm_ids[a], grp, power, fr, optvar]

        peak2 = peak.copy()
        shuffled = []

        # First, look at potentially moving assemblies down a group
        # (e.g. group G --> group G+1).
        #   1. Compare asm with min optimization temp in group G
        #      to asm with max optimization temp in group G+1.
        #   2. If advantageous to switch: do so, repeat step 1.
        #      There is now a new asm with the min optimization
        #      temp in group G and a new (estimated) max temp asm
        #      in group G+1. If not, continue to step 3.
        #   3. Repeat steps 1 and 2 with all groups (e.g. compare
        #      groups G+1 and G+2 and so on.)
        for g in range(n_group - 1):
            # print(f"Considering Group {g} --> Group {g + 1}")
            asm_in_group = peak[peak[:, 1] == g].shape[0]
            for a in range(asm_in_group):
                # Optimization variables for group G
                opt_g = peak2[peak2[:, 3] == group_mfr[g], -1]
                # Check if you want to move the minimum temp asm
                # from group G to G+1. If not, break this loop and
                # go to the next group. If so, investigate further.
                min2avg_g = np.min(opt_g) / np.average(opt_g)
                # print(f'Min/Avg (Group {g}) = {min2avg_g}')
                # If the 1 - min/avg ratio is less than the tolerance,
                # no need to move anything: just break and move on.
                if 1 - min2avg_g <= regroup_tol:  # min/avg < 1
                    # print('No shuffling, move on')
                    break
                # Otherwise, look into whether it is advantageous
                # to move the assembly
                else:
                    # print('Shuffle!')
                    # Optimization variables for group G+1
                    opt_gp1 = peak2[peak2[:, 3] == group_mfr[g + 1], -1]
                    # Evaluate the cumulative "spread" - the degree to
                    # which the min temp assembly in G and the max temp
                    # assembly in G+1 differ from their group averages
                    max2avg_gp1 = np.max(opt_gp1) / np.average(opt_gp1)
                    # print(f'Max/Avg (Group {g+1}) = {max2avg_gp1}')
                    # Note: because max2avg_gp1 > 1 and min2avg_g < 1,
                    # the following is the same as:
                    # abs(max_to_avg_gp1 -1) + abs(min_to_avg_g - 1)
                    cumulative_old = max2avg_gp1 - min2avg_g
                    # print(f'Cumulative = {cumulative_old}')
                    # Try pushing the min T assembly down one group
                    tmp = peak2.copy()
                    row = np.where(tmp == np.min(opt_g))[0][0]
                    tmp[row, 1] = g + 1
                    tmp[row, 3] = group_mfr[g + 1]
                    opt_tmp = self._estimate_optvar(tmp[:, 3], xy, data)
                    tmp[row, -1] = opt_tmp[row]
                    # Evaluate the new "spread"
                    new_opt_g = tmp[tmp[:, 3] == group_mfr[g], -1]
                    new_opt_gp1 = tmp[tmp[:, 3] == group_mfr[g + 1], -1]
                    new_min2avg_g = np.min(new_opt_g) / np.average(new_opt_g)
                    new_max2avg_gp1 = (np.max(new_opt_gp1) /
                                       np.average(new_opt_gp1))
                    # print(f'New Min/Avg (Group {g}) = {new_min2avg_g}')
                    # print(f'New Max/Avg (Group {g+1}) = {new_max2avg_gp1}')
                    cumulative_new = new_max2avg_gp1 - new_min2avg_g
                    # print(f'New cumulative = {cumulative_new}')
                    # If the new spread is less than the old spread,
                    # moving the assembly improved overall agreement
                    # between the two groups.
                    if 0 < cumulative_old - cumulative_new > improve_tol:
                        # Formalize change by adopting "temporary"
                        # peak array as the working array.
                        peak2 = tmp
                        if verbose:
                            msg = self._make_update_msg(
                                int(tmp[row, 0]),
                                g + 1,
                                g + 2,
                                min2avg_g,
                                max2avg_gp1,
                                new_min2avg_g,
                                new_max2avg_gp1)
                            self.log('info', msg)

                        # Track that you pushed an assembly from
                        # g to g+1 so that you don't push it back.
                        shuffled.append((g, g + 1))
                        # Then go to the next assembly in the group
                        continue

                    # Otherwise: don't move the assembly, and don't
                    # look at any more in this group.
                    else:
                        break

        # Next, look at potentially moving assemblies up a group
        # (e.g. group G --> group G-1).
        #   0. If you moved any asm from group G-1 --> group G, skip.
        #   1. Compare asm with max optimization temp in group G to
        #      asm with min optimization temp in group G-1.
        #   2. If advantageous to switch: do so, repeat step 1. There
        #      is now a new asm with the max optimization temp in
        #      group G and a new (estimated) min temp asm in group
        #      G-1. If not, continue to step 3.
        #   3. Repeat steps 1 and 2 with all groups
        for g in range(1, n_group):
            # print(f"Considering Group {g} --> Group {g - 1}")
            if (g - 1, g) in shuffled:
                # print('Shuffled the other way, skipping...')
                continue
            asm_in_group = peak[peak[:, 1] == g].shape[0]
            for a in range(asm_in_group):
                # Optimization variables for group G
                opt_g = peak2[peak2[:, 3] == group_mfr[g], -1]
                # Check if you want to move the minimum temp assembly
                # from group G to G+1. If not, break this loop and go
                # to the next group. If so, investigate further.
                max2avg_g = np.max(opt_g) / np.average(opt_g)
                # print(f'Max/Avg (Group {g}) = {max2avg_g}')
                # If the 1 - min/avg ratio is less than the tolerance,
                # no need to move anything: just break and move on.
                if max2avg_g - 1 <= regroup_tol:  # min/avg < 1
                    # print('No shuffling, move on')
                    break
                # Otherwise, look into whether it is advantageous
                # to move the assembly
                else:
                    # Optimization variables for group G+1
                    opt_gm1 = peak2[peak2[:, 3] == group_mfr[g - 1], -1]
                    # Evaluate the cumulative "spread" - the degree
                    # to which the max temp asm in G and the min temp
                    # asm in G-1 differ from their group averages
                    min2avg_gm1 = np.min(opt_gm1) / np.average(opt_gm1)
                    # print(f'Min/Avg (Group {g+1}) = {min2avg_gm1}')
                    # Note: because max2avg_g > 1 and min2avg_gm1 < 1,
                    # the following is the same as:
                    # abs(max2avg_g -1) + abs(min2avg_gm1 - 1)
                    cumulative_old = max2avg_g - min2avg_gm1
                    # print(f'Cumulative = {cumulative_old}')
                    # Try pushing the min T assembly down one group
                    tmp = peak2.copy()
                    row = np.where(tmp == np.max(opt_g))[0][0]
                    tmp[row, 1] = g - 1
                    tmp[row, 3] = group_mfr[g - 1]
                    opt_tmp = self._estimate_optvar(tmp[:, 3], xy, data)
                    tmp[row, -1] = opt_tmp[row]
                    # Evaluate the new "spread"
                    new_opt_g = tmp[tmp[:, 3] == group_mfr[g], -1]
                    new_opt_gm1 = tmp[tmp[:, 3] == group_mfr[g - 1], -1]
                    new_max2avg_g = np.max(new_opt_g) / np.average(new_opt_g)
                    new_min2avg_gm1 = (np.min(new_opt_gm1) /
                                       np.average(new_opt_gm1))
                    # print(f'New Max/Avg (Group {g}) = {new_max2avg_g}')
                    # print(f'New Min/Avg (Group {g+1}) = {new_min2avg_gm1}')
                    cumulative_new = new_max2avg_g - new_min2avg_gm1
                    # print(f'New cumulative = {cumulative_new}')

                    # If the new spread is less than the old spread,
                    # moving the asm improved overall agreement
                    # between the two groups.
                    if 0 < cumulative_old - cumulative_new > improve_tol:
                        # Formalize change by adopting "temporary"
                        # peak array as the working array.
                        peak2 = tmp
                        if verbose:
                            msg = self._make_update_msg(
                                int(tmp[row, 0]),
                                g + 1,
                                g,
                                max2avg_g,
                                min2avg_gm1,
                                new_max2avg_g,
                                new_min2avg_gm1)
                            self.log('info', msg)

                        # Then go to the next assembly in the group
                        continue

                    # Otherwise: don't move the assembly, and
                    # don't look at any more in this group.
                    else:
                        break

        # Finally: update Orificing object attributes
        self.group_data[:, 2] = peak2[:, 1]

    @staticmethod
    def _make_update_msg(asm, go, gn, r1o, r2o, r1n, r2n, dec=4):
        """x"""
        _ffmt = '{' + f':.{dec}f' + '}'
        indent = '    '
        msg = f'Moved Assembly {asm + 1} from Group {go} to Group {gn}'
        msg += '\n'
        if r1o < 1:
            ratio1 = "Min/Avg"
            ratio2 = "Max/Avg"
        else:
            ratio1 = "Max/Avg"
            ratio2 = "Min/Avg"
        r1o = _ffmt.format(r1o)
        r2o = _ffmt.format(r2o)
        r1n = _ffmt.format(r1n)
        r2n = _ffmt.format(r2n)
        msg += indent + f'OLD: Group {go} {ratio1} = {r1o}; '
        msg += f'Group {gn} {ratio2} = {r2o}\n'
        msg += indent + f'NEW: Group {go} {ratio1} = {r1n}; '
        msg += f'Group {gn} {ratio2} = {r2n}'
        return msg

    ####################################################################
    # WRITE RESULTS TO CSV
    ####################################################################

    def write_results(self, results):
        """Write results of orificing optimization to CSV"""
        self._write_results_assembly(results)
        self._write_results_group(results)

    def _write_results_assembly(self, results):
        """Generate results per assembly from orificing optimization

        Parameters
        ----------
        results : numpy.nadarray
            Results from the last iteration

        Notes
        -----
        orificing_result_assembly.csv
            1. Assembly ID
            2. Assembly type (string)
            3. Total power
            4. Peak linear power
            5. Group ID
            6. Flow rate
            *** 7. Peak velocity
            *** 8. Pressure drop
            9. Average bulk outlet temperature
            10. Peak bulk outlet temperature
            11. Peak coolant outlet temperature
            12. Peak clad OD temperature
            13. Peak clad MW temperature
            14. Peak clad ID temperature
            15. Peak fuel OD temperature
            16. Peak fuel CL temperature

        """
        asm_ids = np.unique(results[:, 1]).astype(int)
        # n_asm = asm_ids.shape[0]
        to_write = ''
        for i in range(asm_ids.shape[0]):
            tmp = results[results[:, 1] == asm_ids[i]]
            line = [asm_ids[i],
                    self._parametric['asm_names'][
                        self._parametric['asm_ids'][i, 1]],
                    self._power[i, 1],
                    self._lin_power[i, 1],
                    int(self.group_data[i, -1]),
                    results[i, 3],
                    '---',
                    '---',
                    np.average(tmp[:, 4]),
                    np.max(tmp[:, 4]),
                    np.max(tmp[:, 5])]
            if tmp.shape[1] > 6:
                line += [np.max(tmp[:, 6]),
                         np.max(tmp[:, 7]),
                         np.max(tmp[:, 8]),
                         np.max(tmp[:, 9]),
                         np.max(tmp[:, 10])]
            to_write += ','.join([str(l) for l in line]) + '\n'
        outpath = os.path.join(
            self._base_input.path,
            'orificing_result_assembly.csv'
        )
        with open(outpath, 'w') as f:
            f.write(to_write)

    def _write_results_group(self, results):
        """Generate results per group from orificing optimization

        orificing_result_group.csv
            1. Group ID
            2. Number of assemblies in group
            3. Average power
            4. Peak power
            5. Flow rate
            3. Average bulk outlet temperature
            4. Peak bulk outlet temperature
            5. Max peak outlet temperature
            6. Average peak clad temperature
            7. Max peak clad temperature
            8. Average peak fuel CL temperature
            9. Max peak fuel CL temperature

        """
        pass

########################################################################
