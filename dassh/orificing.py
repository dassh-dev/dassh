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
date: 2021-10-22
author: matz
Main DASSH calculation procedure
"""
########################################################################
import copy
import numpy as np
import dassh
import os


class Orificing(object):
    """x

    """
    def __init__(self, dassh_input, root_logger):
        self._logger = root_logger
        self._base_input = dassh_input
        self.orifice_input = dassh_input.data['Orificing']
        coolant_name = dassh_input.data['Core']['coolant_material']
        self.coolant = dassh_input.materials[coolant_name]
        self.t_in = dassh_input.data['Core']['coolant_inlet_temp']

        # Setup lookup keys
        x = 'value_to_optimize'
        if self.orifice_input[x] == 'peak coolant temp':
            self._opt_keys = ('cool', None)
        elif self.orifice_input[x] == 'peak clad ID temp':
            self._opt_keys = ('pin', 'clad_id')
        elif self.orifice_input[x] == 'peak clad MW temp':
            self._opt_keys = ('pin', 'clad_mw')
        elif self.orifice_input[x] == 'peak fuel temp':
            self._opt_keys = ('pin', 'fuel_cl')
        else:  # Here for safety; shouldn't be raised b/c input limits
            raise ValueError('Do not understand optimiation variable')

    def optimize(self):
        """x"""
        # Initial setup
        msg = "DASSH orificing optimization calculation"
        self._logger.log(20, msg)
        k = 'value_to_optimize'
        msg = f'Value to minimize: "{self.orifice_input[k]}"'
        self._logger.log(20, msg)
        k = 'n_groups'
        msg = f"Requested number of groups: {self.orifice_input[k]}"
        self._logger.log(20, msg)

        msg = "Performing single-assembly parametric calculations"
        self._logger.log(20, msg)
        self._parametric = {}
        with dassh.logged_class.LoggingContext(self._logger, 40):
            self._parametric['data'], self._parametric['asm_idx'] = \
                self.run_parametric()

        msg = "Performing perfect-orificing calculation"
        self._logger.log(20, msg)
        data_perfect = self.run_dassh_perfect()

        msg = "Grouping assemblies"
        self._logger.log(20, msg)
        with dassh.logged_class.LoggingContext(self._logger, 40):
            self.group_by_power()

        # Iterate to converge to coolant flow distribution among groups
        # that maintains overall bulk outlet temperature constraint and
        # minimizes optimization variable
        iter_data = data_perfect
        iter_data = None
        t_out = None
        for i in range(3):
            iter_data, summary_data = \
                self._do_iter(i + 1, iter_data, t_out)
            t_out = summary_data[-1, 0]
            print(summary_data)

    def _do_iter(self, iter, data_prev=None, t_out=None):
        """Update coolant flow distributions between orifice groups
        and run DASSH sweeps for each timestep"""
        # Distribute flow among groups to meet bulk coolant temp
        msg = f"Iter {iter}: Distributing coolant flow among groups"
        self._logger.log(20, msg)
        m, tlim = self.distribute(data_prev)
        print('T_opt_max: ', tlim)
        print('Flow rates: ', np.unique(m)[::-1])

        # Run DASSH iteration calculation
        msg = f"Iter {iter}: Running DASSH with orificed assemblies"
        self._logger.log(20, msg)
        iter_data = self.run_dassh_orifice(iter, m)
        summary_data = self._summarize_group_data(iter_data)
        return iter_data, summary_data

    def _summarize_group_data(self, res):
        """Generate table of group- and core- average and maximum
        temperatures

        Parameters
        ----------
        res : numpy.ndarray
            Sweep results for all timesteps from "_get_dassh_results method"

        Returns
        -------
        numpy.ndarray
            One row per group; last row is for core-wide data.
            Columns:
            1. Bulk coolant temperature
            2. Peak coolant temperature
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
                _summary[t, i, 0] = np.average(res_t[indices, 4])
                _summary[t, i, 1] = np.max(res_t[indices, 4])
                _summary[t, i, 2] = np.average(res_t[indices, 5])
                _summary[t, i, 3] = np.max(res_t[indices, 5])
        # Finish averaging/maximizing
        summary = np.zeros((n_group + 2, 4))
        for i in range(n_group + 1):
            summary[i, 0] = np.average(_summary[:, i, 0])
            summary[i, 1] = np.max(_summary[:, i, 1])
            summary[i, 2] = np.average(_summary[:, i, 2])
            summary[i, 3] = np.max(_summary[:, i, 3])
        # Core-total average/maxima
        summary[-1, 0] = np.sum(res[:, 4] * res[:, 3] / np.sum(res[:, 3]))
        summary[-1, 1] = np.max(res[:, 4])
        summary[-1, 2] = np.sum(res[:, 5] * res[:, 3] / np.sum(res[:, 3]))
        summary[-1, 3] = np.max(res[:, 5])
        return summary

    ####################################################################
    # DASSH INPUT PREPARATION AND EXECUTION
    ####################################################################

    def run_parametric(self):
        """Perform parametric calculations on a single assembly to
        generate data to use for initial orifice grouping

        Returns
        -------
        tuple
            1. List of numpy.ndarray containing parametric sweep data
            2. List of lists containing assembly IDs that apply to each
               set of parametric sweep data in (1)

        """
        # Instantiate DASSH Reactor
        rx = dassh.Reactor(self._base_input, path='_parametric')

        # For each type of assembly to be grouped, pull a matching
        # assembly object from the Reactor (the first matching
        # assembly you find); calculate average power of all
        # assemblies of that type.
        asm_ids = []
        asm_obj = []
        asm_power = []
        for atype in self.orifice_input['assemblies_to_group']:
            asm_list = [a for a in rx.assemblies if a.name == atype]
            n_asm = len(asm_list)
            asm_obj.append(asm_list[0])
            _asm_power = 0.0
            _asm_ids = []
            for a in asm_list:
                _asm_power += a.total_power
                _asm_ids.append(a.id)
            asm_power.append(_asm_power / n_asm)
            asm_ids.append(_asm_ids)

        # For each assembly, append data to a list
        data = []
        # Create some shortcuts for optimization and lookup keys
        x = ('Assignment', 'ByPosition')
        for i in range(len(asm_obj)):
            # Set up a generic single-assembly input from the original
            inp_1asm = self._setup_input_parametric(
                asm_obj[i].id,
                asm_obj[i].name,
                asm_obj[i].loc,
                asm_power[i])
            try:
                _data = np.loadtxt(
                    os.path.join(inp_1asm.path,
                                 f'data_{asm_obj[i].name}.csv'),
                    delimiter=',')
                data.append(_data)
            except OSError:
                # Setup subdirectory for this calculation
                os.makedirs(inp_1asm.path, exist_ok=True)
                # Initialize data array
                # Columns:  1) Power (MW) / Flow rate (kg/s)
                #           2) Power (MW)
                #           3) Flow rate (kg/s)
                #           4) Target peak temperature (K)
                n_pts = 12
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
                np.savetxt(
                    os.path.join(inp_1asm.path,
                                 f'data_{asm_obj[i].name}.csv'),
                    _data,
                    delimiter=',')
                data.append(_data)
        return data, asm_ids

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
            dassh.__main__.run_dassh(dassh_inp, self._logger, args)
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
        # If you didn't find results, run DASSH
        if not found:
            os.makedirs(wd_path, exist_ok=True)
            # Try to skip the power calculation by using ones you've
            # precalculated from previous iterations
            found = self._find_precalculated_power_distributions(wd_path)
            args = {'save_reactor': True,        # Save Reactor object
                    'verbose': False,            # Don't print stuff
                    'no_power_calc': not found}  # Do the power calc?
            dassh_inp = self._setup_input_orifice(mfr)
            dassh_inp.path = wd_path
            dassh.__main__.run_dassh(dassh_inp, self._logger, args)
            results = self._get_dassh_results(dassh_inp.path)
            np.savetxt(data_path, results, delimiter=',')
        return results

    def _setup_input_parametric(self, id, name, loc, power):
        """Set up a generic DASSH input structure to run for pre-
        optimization parametric sweep"""
        inp = copy.deepcopy(self._base_input)

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
        inp = copy.deepcopy(self._base_input)
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
        return inp

    def _find_precalculated_power_distributions(self, wdpath):
        """See if you can link previously calculated power
        distributions rather than recalculating"""
        # Look for power distributions in "_perfect" or "_iter1"
        varpow_files = ('VARPOW.out',
                        'varpow_MatPower.out',
                        'varpow_MonoExp.out')
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
                        for ff in varpow_files:
                            src = os.path.join(p, f, ff)
                            if os.path.exists(src):
                                dest = os.path.join(destpath, ff)
                                os.symlink(src, dest)
                                linked += 1
                # Get VARPOW files that aren't in subdir, if applicable
                if any(['varpow' in x for x in os.listdir(p)]):
                    expected += 3
                    for ff in varpow_files:
                        src = os.path.join(p, ff)
                        if os.path.exists(os.path.join(p, ff)):
                            dest = os.path.join(wdpath, ff)
                            os.symlink(src, dest)
                            linked += 1
            if found:
                break
        if linked > 0 and linked == expected:
            return True
        else:
            assert 0
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
                r = dassh.reactor.load(f)
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
                     a.avg_coolant_temp]
                if self._opt_keys[1] is not None:
                    d.append(
                        a._peak[self._opt_keys[0]][
                            self._opt_keys[1]][0])
                else:
                    d.append(a._peak[self._opt_keys[0]][0])
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

        """
        self._power = []
        self._power_to_grp = []

        # For each timestep, collect assembly power and power
        # parameter (either integral power or peak linear power)
        # to use in orifice grouping
        need_subdir = False
        if self._base_input.timepoints > 1:
            need_subdir = True
        for i in range(self._base_input.timepoints):
            wdir = None
            if need_subdir:
                wdir = os.path.join(self._base_input.path,
                                    '_power',
                                    f'timestep_{i + 1}')
            dassh_rx = dassh.Reactor(self._base_input,
                                     calc_power=True,
                                     path=wdir,
                                     timestep=i,
                                     write_output=False)
            _power = []
            _power_to_grp = []
            _id = []
            for a in dassh_rx.assemblies:
                if a.name in self.orifice_input['assemblies_to_group']:
                    _id.append(a.id)
                    _power.append(a.total_power)
                    if group_by == 'linear_power':
                        _power_to_grp.append(
                            a.power.calculate_avg_peak_linear_power())
                    elif group_by == 'power':
                        _power_to_grp.append(a.total_power)
                    else:  # Here for safety, should never be raised
                        raise ValueError(
                            'Argument "group_by" must be "power" '
                            + f'or "linear_power"; input {group_by} '
                            + 'not recognized')
            self._power.append(np.array((_id, _power)).T)
            self._power_to_grp.append(np.array((_id, _power_to_grp)).T)

        # Take average for each assembly
        tmp = np.average([x[:, 1] for x in self._power], axis=0)
        self._power = np.array((self._power[0][:, 0], tmp)).T
        tmp = np.average([x[:, 1] for x in self._power_to_grp], axis=0)
        self._power_to_grp = np.array((self._power_to_grp[0][:, 0], tmp)).T

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
            self._logger.log(40, msg)
            print('hi')  # Should do system.exit

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
        # Prepare FR vs. T_opt from parametric sweep for interpolation
        xy = []
        for i in range(len(self._parametric['data'])):
            tmp = self._parametric['data'][i]
            tmp = tmp[tmp[:, 2].argsort()]
            xy.append(tmp[:, (2, -1)])

        # Temporary indexing array to determine which
        # assemblies are of which type
        _atype_idx = np.zeros(self.group_data.shape[0], dtype=int)
        for i in range(_atype_idx.shape[0]):
            for j in range(len(self._parametric['asm_idx'])):
                if (self.group_data[i, 0]
                        in self._parametric['asm_idx'][j]):
                    _atype_idx[i] = j
                    match = True
                    break
            if not match:
                raise ValueError('Unknown assembly type')

        # Total mass flow rate required (estimate)
        m_total = dassh.Q_equals_mCdT(
            np.sum(self._power[:, 1]),
            self.t_in,
            self.coolant,
            t_out=self.orifice_input['bulk_coolant_temp'])

        # Scale total MFR based on result from previous calculation
        # If T_previous < T_target, m_new < m_previous
        if t_out_prev:
            dt_prev = t_out_prev - self.t_in
            dt_trgt = self.orifice_input['bulk_coolant_temp'] - self.t_in
            ratio = dt_prev / dt_trgt
            m_total *= ratio

        # Determine m_lim based on pressure drop limit
        # for each asm type being orificed
        m_lim = None
        if self.orifice_input['pressure_drop_limit']:
            dp_limit = self.orifice_input['pressure_drop_limit'] * 1e6
            m_lim = np.zeros(len(self._parametric['data']))
            for i in range(m_lim.shape[0]):
                m_lim[i] = np.interp(dp_limit,
                                     self._parametric['data'][i][:, 3],
                                     self._parametric['data'][i][:, 2])

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
        dp_warning = False
        while convergence > tol and iter < iter_lim:
            # Update flow rates
            m_remaining = m_total
            for g in range(self.orifice_input['n_groups'] - 1):
                m_new = m[self.group_data[:, -1] == g] * d_optvar[g]
                if m_lim is not None:
                    asm_type_in_grp = _atype_idx[
                        self.group_data[:, -1] == g]
                    m_lim_grp = m_lim[asm_type_in_grp]
                    if np.any(m_new > m_lim_grp):
                        m_new[:] = np.min(m_lim_grp)
                        dp_warning = True
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
            optvar = self._estimate_optvar(m, xy, _atype_idx, res_prev)
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

        print(group_max)
        print(avg_max)
        print('Delta (want close to 1.0): ', d_optvar)
        print(f'Convergence (want less than {tol}): ', convergence)
        # Check conservation of mass - this should never fail
        if not abs(np.sum(m) - m_total) < 1e-6:
            raise ValueError("Mass flow rate not conserved")
        # Report whether the pressure drop mass flow rate limit was met
        if dp_warning:
            print("Warning: Peak mass flow rate restricted to accommodate "
                  + "user-specified pressure drop limit")
        # Return results
        return m, max(group_max)

    def _estimate_optvar(self, mfr, xy, asm_type_idx, res_prev=None):
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
        asm_type_idx : numpy.ndarray
            List of integers indicating which assemblies are of which
            type (this tells us which data in "xy" to use)
        res_prev : numpy.ndarray (optional)
            Data from the previous DASSH iteration; used to correct
            estimate with scalar ratio for each assembly
            (default = 0)

        Returns
        -------
        numpy.ndarray
            Estimate for the optimization variable for each assembly

        """
        if res_prev is not None:
            data_pts = res_prev[:, -1]
            interpolated_pts = []
            for i in range(len(xy)):
                interpolated_pts.append(
                    np.interp(res_prev[:, 3], xy[i][:, 0], xy[i][:, 1]))
            interpolated_pts = np.array(interpolated_pts)
            ratio = data_pts / interpolated_pts
            # Average over all cycles for each assembly
            ratio = np.array([
                np.average(ratio[:, res_prev[:, 1] == i], axis=1)
                for i in np.unique(res_prev[:, 1])
            ])
            # Choose the appropriate ratio for the right asm type
            ratio = ratio[np.arange(ratio.shape[0]), asm_type_idx]
        else:
            ratio = 1.0
        # Interpolate for the new flow rate
        new = []
        for i in range(len(xy)):
            new.append(np.interp(mfr, xy[i][:, 0], xy[i][:, 1]))
        new = np.array(new).T
        new = new[np.arange(new.shape[0]), asm_type_idx]
        return new * ratio


########################################################################


def _setup_single_asm_input(inp, asm_name, asm_id, asm_loc, tot_power):
    """Set up a generic DASSH input structure to run for
    pre-optimization parametric sweep"""
    inp = copy.deepcopy(inp)

    # Remove all unused assembly types
    for k in inp.data['Assembly'].keys():
        if k != asm_name:
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

    # Set total power equal to core-average value
    inp.data['Power']['total_power'] = tot_power

    # Remove all timesteps other than the first one - they are
    # not necessary for the parametric calculation
    cccc = ['geodst', 'ndxsrf', 'znatdn', 'labels', 'nhflux', 'ghflux']
    for k in cccc:
        inp.data['Power']['ARC'][k] = [inp.data['Power']['ARC'][k][0]]

    # Eliminate orificing optimization input
    inp.data['Orificing'] = False

    # Change the path to the proper subdir
    inp.path = os.path.join(inp.path, '_parametric')

    # Set up assignment so that there's only one assembly. Need
    # to include empty lists for all empty assembly positions
    # through the ring that has the preserved assembly.
    n_positions = 3 * (asm_loc[0]) * (asm_loc[0] + 1) + 1
    tmp = []
    for i in range(n_positions):
        if i == asm_id:
            tmp.append(inp.data['Assignment']['ByPosition'][i])
        else:
            tmp.append([])
    inp.data['Assignment']['ByPosition'] = tmp
    return inp


def _setup_dassh_input_perfect(inp, target_bulk_temp):
    """Using new flow rates, create new DASSH input"""
    inp = copy.deepcopy(inp)
    # Eliminate orificing optimization input
    inp.data['Orificing'] = False
    # Set input path
    inp.path = os.path.join(inp.path, '_perfect')
    # Set up assignment to give desired outlet temp to each assembly
    for i in range(len(inp.data['Assignment']['ByPosition'])):
        if inp.data['Assignment']['ByPosition'][i]:
            inp.data['Assignment']['ByPosition'][i][2] = \
                {'outlet_temp': target_bulk_temp}
    return inp


def _setup_dassh_input_orifice(inp, group_data, m_asm, target_bulk_temp):
    """Using new flow rates, create new DASSH input"""
    # Setup input to have perfect orificing, then overwrite relevant
    # assemblies to create orifice groups with determined flow rate
    inp = _setup_dassh_input_perfect(inp, target_bulk_temp)
    inp.path = ''
    asm_id = group_data[:, 0].astype(int)
    for i in range(group_data.shape[0]):
        inp.data['Assignment']['ByPosition'][asm_id[i]][2] = \
            {'flowrate': m_asm[i]}
    return inp


def run_dassh_parametric(dassh_inp0, asm_to_group, val_to_optimize):
    """Perform parametric calculations on a single assembly to generate
    data to use for initial orifice grouping"""
    # Instantiate DASSH Reactor
    dassh_rx = dassh.Reactor(dassh_inp0, path='_parametric')

    # For each type of assembly to be grouped, pull a matching assembly
    # object from the Reactor (the first matching assembly you find);
    # calculate average power of all assemblies of that type.
    asm_obj = []
    asm_power = []
    for atype in asm_to_group:
        asm_list = [a for a in dassh_rx.assemblies if a.name == atype]
        n_asm = len(asm_list)
        asm_obj.append(asm_list[0])
        # Calculate average power of all assemblies of this type
        asm_power.append(sum([a.total_power for a in asm_list]) / n_asm)

    # For each assembly, append data to the list
    data = []
    # Shortcut for optimization keys
    k = _match_optimization_keys(val_to_optimize)
    for i in range(len(asm_obj)):
        # Set up a generic single-assembly input from the original
        inp_1asm = _setup_single_asm_input(
            dassh_inp0, asm_obj[i].name, asm_obj[i].id,
            asm_obj[i].loc, asm_power[i])
        try:
            _data = np.loadtxt(
                os.path.join(inp_1asm.path,
                             f'data_{asm_obj[i].name}.csv'),
                delimiter=',')
            data.append(_data)
        except OSError:
            # Setup subdirectory for this calculation
            os.makedirs(inp_1asm.path, exist_ok=True)
            # Initialize data array
            # Columns:  1) Power (MW) / Flow rate (kg/s)
            #           2) Power (MW)
            #           3) Flow rate (kg/s)
            #           4) Target peak temperature (K)
            n_pts = 12
            _data = np.zeros((n_pts, 5))
            _data[:, 0] = np.geomspace(0.05, 1.0, n_pts)  # MW / (kg/s)
            _data[:, 1] = asm_power[i]  # Watts
            _data[:, 2] = asm_power[i] / 1e6 / _data[:, 0]  # kg/s
            for j in range(_data.shape[0]):
                # Find active assembly position and update it
                for a in range(len(inp_1asm.data['Assignment']['ByPosition'])):
                    if inp_1asm.data['Assignment']['ByPosition'][a] == []:
                        continue
                    else:
                        inp_1asm.data['Assignment']['ByPosition'][a][2] = \
                            {'flowrate': _data[j, 2]}
                        break
                r1a = dassh.Reactor(inp_1asm, calc_power=False)
                r1a.temperature_sweep()
                _data[j, 3] = r1a.assemblies[0].pressure_drop
                if k[1] is not None:
                    _data[j, 4] = r1a.assemblies[0]._peak[k[0]][k[1]][0]
                else:
                    _data[j, 4] = r1a.assemblies[0]._peak[k[0]][0]
                del r1a
            np.savetxt(
                os.path.join(inp_1asm.path,
                             f'data_{asm_obj[i].name}.csv'),
                _data,
                delimiter=',')
            data.append(_data)
    return data


def run_dassh_perfect(dassh_inp0, dassh_log, orifice_input):
    """Execute DASSH assuming perfect orificing for each assembly to
    have desired coolant temperature rise

    Parameters
    ----------
    dassh_inp0 : DASSH DASSH_Input object
        Base DASSH input to modify with orifice specifications
    dassh_log : DASSH logging object
        blah
    orifice_inp : dict
        Parameters to control the orificing optimization

    Returns
    -------
    numpy.ndarray
        Results from the DASSH calculation

    Notes
    -----
    Will need to update to run for every time step requested; currently
    only running for one timestep.

    """
    # Try and find pre-existing outputs before rerunning new cases
    wd_path = os.path.join(dassh_inp0.path, '_perfect')
    data_path = os.path.join(wd_path, 'data.csv')
    found = False
    if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
        found = True
        results = np.loadtxt(data_path, delimiter=',')
    else:
        try:
            results = _get_dassh_results(wd_path, orifice_input)
            if results.shape[0] > 0:
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
        dassh_inp = _setup_dassh_input_perfect(
            dassh_inp0,
            orifice_input['bulk_coolant_temp'])
        dassh.__main__.run_dassh(dassh_inp, dassh_log, args)
        results = _get_dassh_results(dassh_inp.path, orifice_input)
        np.savetxt(data_path, results, delimiter=',')
    return results


def run_dassh_orifice(dassh_inp0, dassh_log, orifice_inp, iter, grps, mfr):
    """Execute DASSH with as-determined orifice groups and flow rates

    Parameters
    ----------
    dassh_inp0 : DASSH DASSH_Input object
        Base DASSH input to modify with orifice specifications
    dassh_log : DASSH logger
        blah
    orifice_inp : dict
        Parameters to control the orificing optimization
    iter : int
        Iteration index to label temporary directory
    grps : numpy.ndarray
        Contains grouping specification
    mfr : numpy.ndarray
        Mass flow rates for each assembly

    Returns
    -------
    numpy.ndarray
        Results from the DASSH calculation

    Notes
    -----
    Will need to update to run for every time step requested; currently
    only running for one timestep.

    """
    # Try and find pre-existing outputs before rerunning new cases
    wd_path = os.path.join(dassh_inp0.path, f'_iter{iter}')
    data_path = os.path.join(wd_path, 'data.csv')
    found = False
    if os.path.exists(data_path) and os.path.getsize(data_path) > 0:
        found = True
        results = np.loadtxt(data_path, delimiter=',')
    else:
        try:
            results = _get_dassh_results(wd_path, orifice_inp)
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
        dassh_inp = _setup_dassh_input_orifice(
            dassh_inp0, grps, mfr,
            orifice_inp['bulk_coolant_temp'])
        dassh_inp.path = wd_path
        dassh.__main__.run_dassh(dassh_inp, dassh_log, args)
        results = _get_dassh_results(dassh_inp.path, orifice_inp)
        np.savetxt(data_path, results, delimiter=',')
    return results


def _get_dassh_results(wdpath, orifice_inp):
    results = []
    for f in os.listdir(wdpath):
        if os.path.isdir(os.path.join(wdpath, f)) and 'timestep' in f:
            p = os.path.join(os.path.join(wdpath, f), 'dassh_reactor.pkl')
            if os.path.exists(p):
                t = f.split('_')[-1]
                r = dassh.reactor.load(p)
                results.append(_read_dassh_results(r, orifice_inp, t))
                del r
        elif f == 'dassh_reactor.pkl':
            r = dassh.reactor.load(f)
            results.append(_read_dassh_results(r, orifice_inp, 0))
            del r
        else:
            continue
    if len(results) > 0:
        results = np.vstack(results)
    return results


def _read_dassh_results(dassh_rx, orifice_inp, timestep):
    """Pull DASSH results from Reactor object"""
    k = _match_optimization_keys(orifice_inp['value_to_optimize'])
    data = []
    for a in dassh_rx.assemblies:
        if a.name in orifice_inp['assemblies_to_group']:
            d = [timestep,
                 a.id,
                 a.total_power,
                 a.flow_rate,
                 a.avg_coolant_temp]
            if k[1] is not None:
                d.append(a._peak[k[0]][k[1]][0])
            else:
                d.append(a._peak[k[0]][0])
            data.append(d)
    data = np.array(data, dtype=float)
    return data


def _match_optimization_keys(value_to_optimize):
    if value_to_optimize == 'peak coolant temp':
        k1 = 'cool'
        k2 = None
    elif value_to_optimize == 'peak clad ID temp':
        k1 = 'pin'
        k2 = 'clad_id'
    elif value_to_optimize == 'peak clad MW temp':
        k1 = 'pin'
        k2 = 'clad_mw'
    elif value_to_optimize == 'peak fuel temp':
        k1 = 'pin'
        k2 = 'fuel_cl'
    else:  # Here for safety; shouldn't be raised b/c input limits
        raise ValueError('Do not understand optimiation variable')
    return k1, k2


def group_by_power(dassh_input, orifice_input):
    """Group assemblies by power or linear power

    Parameters
    ----------
    dassh_input : DASSH_Input object
        Contains power distribution paths for each timestep
    orifice_input : dict
        Dictionary of orificing optimization inputs provided by user

    Returns
    -------
    numpy.ndarray
        Assembly parameters and orifice groups

    Notes
    -----
    Creates Reactor object for each timestep; uses this to extract
    the grouping parameter (total power or linear power)

    """
    if orifice_input['value_to_optimize'] == 'peak coolant temp':
        group_by = 'power'
    else:
        group_by = 'linear_power'
    # Create Reactor objects to generate power distributions for
    # use in grouping algorithm. This is basically the first bit
    # of __main__.run_dassh(), but it  extracts power distribution
    # info instead of doing the temperature sweep
    power_parameters = []
    need_subdir = False
    if input.timepoints > 1:
        need_subdir = True
    for i in range(input.timepoints):
        working_dir = None
        if need_subdir:
            working_dir = os.path.join(input.path, f'timestep_{i + 1}')
        print('\n')
        # logger.log(20, f'Timestep {i + 1}')
        dassh_rx = dassh.Reactor(input,
                                 calc_power=True,
                                 path=working_dir,
                                 timestep=i,
                                 write_output=False)
        power_parameters.append(
            _get_power(dassh_rx,
                       orifice_input['assemblies_to_group'],
                       group_by=group_by))
    # Take average across each timestep
    tmp = np.average([x[:, 1] for x in power_parameters], axis=0)
    avg_power_array = np.array((power_parameters[0][:, 0], tmp))
    group_data = _group(avg_power_array,
                        orifice_input['n_groups'],
                        orifice_input['group_cutoff'],
                        orifice_input['group_cutoff_delta'])
    return group_data


def _get_power(dassh_rx, asm_to_group, group_by='linear_power'):
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

    """
    power = []
    id = []
    if group_by == 'power':
        for a in dassh_rx.assemblies:
            if a.name in asm_to_group:
                id.append(a.id)
                power.append(a.total_power)
    elif group_by == 'linear_power':
        for a in dassh_rx.assemblies:
            if a.name in asm_to_group:
                id.append(a.id)
                power.append(a.power.calculate_avg_peak_linear_power())
    else:
        raise ValueError('Argument "group_by" must be either "power" '
                         + f'or "linear_power"; input {group_by} not '
                         + 'recognized')
    return np.array((id, power)).T


def _group(params, n_grp_requested, cutoff, cutoff_delta):
    """Divide assemblies into orifice groups based on some parameter

    Parameters
    ----------
    params : numpy.ndarray
        The parameter to use in the grouping
        Column 1: assembly index
        Column 2: parameter value
    n_grp_requested : integer
        Number of groups requested by the user
    cutoff : float
        Cutoff tolerance to separate groups; evaluated against:
        (max(params[group]) - min(params[group])) / avg(params[group])
    cutoff_delta:
        Increment with which to increase cutoff if number of groups is
        not achieved with the initial value

    Returns
    -------
    numpy.ndarray
        Array (N_asm x 3) containing assembly indices, parameter
        values, and group IDs

    """
    params = params[params[:, 1].argsort()][::-1]
    n_grp = n_grp_requested + 1
    iter = 0
    while n_grp != n_grp_requested and iter < 1000:
        g = 0
        grp_param = [[params[0, 1]]]  # Add first assembly to Group 1
        for i in range(1, params.shape[0]):
            if _check_new_group(grp_param[g],
                                params[i, 1],
                                cutoff):
                g += 1
                grp_param.append([])
            grp_param[g].append(params[i, 1])
        n_grp = len(grp_param)
        # If you have more groups than requested, relax the cutoff
        # between groups - will get fewer groups on the next ieration
        if n_grp > n_grp_requested - 1:  # n_grp is Python index!
            cutoff += cutoff_delta
        # If you have fewer groups than requested, tighten the cutoff
        # between groups - will get more groups on the next iteration.
        # NOTE: whereas the above adjustment is meant to be incremental,
        # this one needs to be big - from here, we want to end up with
        # more groups than requested so that we can walk incrementally
        # from there. Hopefully we don't need this condition.
        if n_grp < n_grp_requested - 1:  # n_grp is Python index!
            cutoff /= 10.0
        # Update iteration index
        iter += 1
    # Check iteration index; if not converged, raise error
    if iter >= 1000 and n_grp != n_grp_requested - 1:
        raise ValueError('Grouping not converged; please adjust '
                         + '"group_cutoff" and "group_cutoff_delta" '
                         + 'parameters')
    # Attach grouping to group parameter data and return
    group_data = np.zeros((params.shape[0], 3))
    group_data[:, :2] = params
    group_data[:, 2] = [i for i in range(len(grp_param))
                        for j in range(len(grp_param[i]))]
    return group_data


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


def _estimate_optvar(fr, parametric_results, latest_results):
    """Estimate the variable-to-optimize based on the single-assembly
    parametric data and the latest DASSH results"""
    data_pts = latest_results[:, -1]
    interpolated_pts = np.interp(latest_results[:, 3],
                                 parametric_results[:, 0],
                                 parametric_results[:, 1])
    ratio = data_pts / interpolated_pts
    # Average over all cycles for each assembly
    ratio = np.array([
        np.average(ratio[latest_results[:, 1] == i])
        for i in np.unique(latest_results[:, 1])
    ])
    new_value = np.interp(fr,
                          parametric_results[:, 0],
                          parametric_results[:, 1])
    return new_value * ratio


def distribute_flow(orifice_inp, group_data, coolant_obj, t_in,
                    res_parametric, res_latest=None, prev_t_out=None,
                    tol=1.0, iter_lim=50):
    """Distribute flow to minimize optimization variable

    Parameters
    ----------
    orifice_inp : dict
        Contains inputs to control orificing optimization
    group_data : numpy.ndarray
        Assembly grouping
    coolant_obj : DASSH Material object
        Coolant material properties
    t_in : float
        Coolant inlet temperature
    res_parametric : list
        List of numpy.ndarray for each assembly type included in
        the orificing optimization
    res_latest : numpy.ndarray
        Results of the latest DASSH sweep for each timestep

    prev_t_out : float (optional)
        Average of previous DASSH sweep outlet temps for each timestep
        (default = None)
    tol : float (optional)
        Tolerance for converging flow distribution iterations
        (default = 1.0)
    iter_lim : int (optional)
        Limit for flow distribution iterations
        (default = 50)

    Returns
    -------
    blah

    """
    # Prepare FR vs. T_opt from parametric sweep for interpolation
    res_parametric = res_parametric[res_parametric[:, 2].argsort()]
    xy = res_parametric[:, (2, -1)]
    # Total mass flow rate required (estimate)
    m_total = dassh.Q_equals_mCdT(
        _calc_total_power(res_latest),
        t_in,
        coolant_obj,
        t_out=orifice_inp['bulk_coolant_temp'])
    # Scale total mass flow rate based on result from previous calculation
    # If T_previous < T_target, m_new < m_previous
    if prev_t_out:
        dt_previous = prev_t_out - t_in
        dt_target = orifice_inp['bulk_coolant_temp'] - t_in
        ratio = dt_previous / dt_target
        m_total *= ratio
    # Determine m_lim based on pressure drop limit
    m_lim = None
    if orifice_inp['pressure_drop_limit']:
        m_lim = np.interp(orifice_inp['pressure_drop_limit'] * 1e6,
                          res_parametric[:, 3],
                          res_parametric[:, 2])
    # First guess - give all groups an average flow rate.
    m = np.ones(group_data.shape[0]) * m_total / group_data.shape[0]
    # Initialize delta: factors used to "guess" next flow rates
    d_optvar = np.ones(orifice_inp['n_groups'])
    # Initialize iteration parameters
    convergence = 2.0
    iter = 0
    dp_warning = False
    while convergence > tol and iter < iter_lim:
        # Update flow rates
        m_remaining = m_total
        for g in range(orifice_inp['n_groups'] - 1):
            m_new = m[group_data[:, -1] == g] * d_optvar[g]
            if m_lim:
                if m_new[0] > m_lim:
                    m_new[:] = m_lim
                    dp_warning = True
            m[group_data[:, -1] == g] = m_new
            m_remaining -= np.sum(m_new)
        last_grp_idx = (group_data[:, -1] == orifice_inp['n_groups'] - 1)
        m[last_grp_idx] = m_remaining / np.count_nonzero(last_grp_idx)

        # Calculate group-total flow rates
        group_total_fr = np.array([
            np.sum(m[group_data[:, -1] == g])
            for g in range(orifice_inp['n_groups'])
        ])
        # Estimate optimization variable based on new flow rates;
        # Get max values per group and an "average" value weighted
        # by flow rates and number of assemblies in the group
        optvar = _estimate_optvar(m, xy, res_latest)
        group_max = np.array([
            np.max(optvar[group_data[:, -1] == g])
            for g in range(orifice_inp['n_groups'])
        ])
        avg_max = np.sum(group_total_fr * group_max)
        avg_max /= np.sum(group_total_fr)
        # Difference with average; use to get next guess
        d_optvar = (group_max - t_in) / (avg_max - t_in)
        # Check convergence
        convergence = np.sqrt(np.sum((group_max - avg_max)**2))
        iter += 1
    #
    print(group_max)
    print(avg_max)
    print('Delta: ', d_optvar)
    print('Convergence: ', convergence)
    # Check conservation of mass - this should never fail
    if not abs(np.sum(m) - m_total) < 1e-6:
        raise ValueError("Mass flow rate not conserved")
    # Report whether the pressure drop mass flow rate limit was met
    if dp_warning:
        print("Warning: Peak mass flow rate restricted to accommodate "
              + "user-specified pressure drop limit")
    # Return results
    return m, max(group_max)


def _calc_total_power(dassh_results):
    """Calculate average core total power across multiple cycles"""
    p = 0.0
    timesteps = np.unique(dassh_results[:, 0])
    for i in timesteps:
        p += np.sum(dassh_results[dassh_results[:, 0] == i][:, 2])
    p /= timesteps.shape[0]
    return p


def _summarize_group_data(group_data, res):
    """Generate table of group- and core- average and maximum
    temperatures

    Parameters
    ----------
    group_data : numpy.ndarray
        Grouping data from "group" method
    res : numpy.ndarray
        Sweep results for all timesteps from "_get_dassh_results method"

    Returns
    -------
    numpy.ndarray
        One row per group; last row is for core-wide data.
        Columns:
        1. Bulk coolant temperature
        2. Peak coolant temperature
        3. Average temperature of variable being optimized
        4. Peak temperature of variable being optimized

    """
    n_group = max(group_data[:, 2].astype(int))
    timesteps = np.unique(res[:, 0])
    _summary = np.zeros((timesteps.shape[0], n_group + 1, 4))
    for t in range(timesteps.shape[0]):
        res_t = res[res[:, 0] == timesteps[t]]
        for i in range(n_group + 1):
            indices = (group_data[:, 2].astype(int) == i)
            _summary[t, i, 0] = np.average(res_t[indices, 4])
            _summary[t, i, 1] = np.max(res_t[indices, 4])
            _summary[t, i, 2] = np.average(res_t[indices, 5])
            _summary[t, i, 3] = np.max(res_t[indices, 5])
    # Finish averaging/maximizing
    summary = np.zeros((n_group + 2, 4))
    for i in range(n_group + 1):
        summary[i, 0] = np.average(_summary[:, i, 0])
        summary[i, 1] = np.max(_summary[:, i, 1])
        summary[i, 2] = np.average(_summary[:, i, 2])
        summary[i, 3] = np.max(_summary[:, i, 3])
    # Core-total average/maxima
    summary[-1, 0] = np.sum(res[:, 4] * res[:, 3] / np.sum(res[:, 3]))
    summary[-1, 1] = np.max(res[:, 4])
    summary[-1, 2] = np.sum(res[:, 5] * res[:, 3] / np.sum(res[:, 3]))
    summary[-1, 3] = np.max(res[:, 5])
    return summary


def optimize(dassh_inp0, dassh_logger):
    """x"""
    msg = "Performing initial setup"
    dassh_logger.log(20, msg)
    orificing_input = dassh_inp0.data['Orificing']
    # NOTE: Need to set up logging so that the DASSH sweeps return some
    # information

    # Initialize Coolant and Reactor objects for grouping calculations
    coolant_name = dassh_inp0.data['Core']['coolant_material']
    coolant = dassh_inp0.materials[coolant_name]
    coolant_t_in = dassh_inp0.data['Core']['coolant_inlet_temp']

    # Perform parametric sweep
    msg = "Performing single-assembly parametric calculations"
    dassh_logger.log(20, msg)
    with dassh.logged_class.LoggingContext(dassh_logger, 40):
        init_data = run_dassh_parametric(
            dassh_inp0, orificing_input['assemblies_to_group'],
            orificing_input['value_to_optimize'])

    return
    # Perform perfect orificing calculation
    msg = "Performing perfect-orificing calculation"
    dassh_logger.log(20, msg)
    perfect_data = run_dassh_perfect(
        dassh_inp0, dassh_logger, orificing_input)

    # Get power and perform grouping
    msg = "Grouping assemblies"
    dassh_logger.log(20, msg)
    group = group_by_power(dassh_inp0, orificing_input)
    group = group[group[:, 0].argsort()]

    # Distribute flow among groups
    msg = "Distributing coolant flow among groups"
    dassh_logger.log(20, msg)
    m, tlim = distribute_flow(
        orificing_input, group, perfect_data, init_data,
        coolant, coolant_t_in, tol=0.5)
    print('T_opt_max: ', tlim)
    print('Flow rates: ', np.unique(m)[::-1])

    # Run DASSH iteration calculation
    msg = "Running DASSH with orificed assemblies (iter 1)"
    dassh_logger.log(20, msg)
    iter_data = run_dassh_orifice(
        dassh_inp0, dassh_logger, orificing_input, 1, group, m)
    summary_data = _summarize_group_data(group, iter_data)
    print(summary_data)

    # Redistribute coolant flow among groups to meet bulk coolant temperature
    print("Distributing coolant flow among groups")
    m, tlim = distribute_flow(
        orificing_input, group, iter_data, init_data, coolant,
        coolant_t_in, tol=0.5, prev_t_out=summary_data[-1, 0])
    print('T_opt_max: ', tlim)
    print('Flow rates: ', np.unique(m)[::-1])

    # Iteration 2
    print("Running DASSH with orificed assemblies (iter 2)")
    iter_data = run_dassh_orifice(
        dassh_inp0, dassh_logger, orificing_input, 2, group, m)
    summary_data = _summarize_group_data(group, iter_data)
    print(summary_data)

    # Redistribute coolant flow among groups to meet bulk coolant temperature
    print("Distributing coolant flow among groups")
    m, tlim = distribute_flow(
        orificing_input, group, iter_data, init_data, coolant,
        coolant_t_in, tol=0.5, prev_t_out=summary_data[-1, 0])
    print('T_opt_max: ', tlim)
    print('Flow rates: ', np.unique(m)[::-1])

    # Iteration 3
    print("Running DASSH with orificed assemblies (iter 3)")
    iter_data = run_dassh_orifice(
        dassh_inp0, dassh_logger, orificing_input, 3, group, m)
    summary_data = _summarize_group_data(group, iter_data)
    print(summary_data)
