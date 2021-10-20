import copy
import numpy as np
import dassh
import os


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


def run_dassh_parametric(inp, dassh_rx, asm_to_group, val_to_optimize):
    """Perform parametric calculations on a single assembly to generate
    data to use for initial orifice grouping"""
    # Identify the first matching assembly you find
    asm_list = [a for a in dassh_rx.assemblies if a.name in asm_to_group]
    n_asm = len(asm_list)
    asm = asm_list[0]
    avg_power = dassh_rx.total_power / n_asm

    # Set up a generic single-assembly input from the original
    inp_1asm = _setup_single_asm_input(
        inp, asm.name, asm.id, asm.loc, avg_power)
    # Initialize data array
    # Columns:  1) Power (MW) / Flow rate (kg/s)
    #           2) Power (MW)
    #           3) Flow rate (kg/s)
    #           4) Target peak temperature (K)
    n_pts = 12
    data = np.zeros((n_pts, 5))
    # data[:, 0] = np.logspace(0.1, 1.0, 8) / 10.0  # MW / (kg/s)
    data[:, 0] = np.geomspace(0.05, 1.0, n_pts)  # MW / (kg/s)
    data[:, 1] = avg_power  # Watts
    data[:, 2] = avg_power / 1e6 / data[:, 0]  # kg/s

    # Setup subdirectory for this calculation
    try:
        data = np.loadtxt(
            os.path.join(inp_1asm.path, 'data.csv'),
            delimiter=',')
    except OSError:
        os.makedirs(inp_1asm.path, exist_ok=True)
        k = _match_optimization_keys(val_to_optimize)
        for f in ['varpow_MatPower.out', 'varpow_MonoExp.out', 'VARPOW.out']:
            symlink_path = os.path.join('_parametric', f)
            if not os.path.islink(symlink_path):
                os.symlink(f, symlink_path)
        for i in range(data.shape[0]):
            # Find active assembly position and update it
            for a in range(len(inp_1asm.data['Assignment']['ByPosition'])):
                if inp_1asm.data['Assignment']['ByPosition'][a] == []:
                    continue
                else:
                    inp_1asm.data['Assignment']['ByPosition'][a][2] = \
                        {'flowrate': data[i, 2]}
                    break
            r1a = dassh.Reactor(inp_1asm, calc_power=False)
            r1a.temperature_sweep()
            data[i, 3] = r1a.assemblies[0].pressure_drop
            if k[1] is not None:
                data[i, 4] = r1a.assemblies[0]._peak[k[0]][k[1]][0]
            else:
                data[i, 4] = r1a.assemblies[0]._peak[k[0]][0]
            del r1a
        np.savetxt(
            os.path.join(inp_1asm.path, 'data.csv'),
            data,
            delimiter=',')
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


def group_by_power(orifice_input, dassh_rx):
    """Group assemblies by power or linear power

    Parameters
    ----------
    orifice_input : dict
        Dictionary of orificing optimization inputs provided by user
    dassh_rx : DASSH Reactor object
        Contains all details about the reactor system

    Returns
    -------
    numpy.ndarray
        Assembly parameters and orifice groups

    Notes
    -----
    Will need to update this method for when orificing optimization is
    performed for multiple cycles: need to get power parameters from each
    cycle, average, then perform grouping.

    """
    if orifice_input['value_to_optimize'] == 'peak coolant temp':
        group_by = 'power'
    else:
        group_by = 'linear_power'
    power_array = _get_power(dassh_rx,
                             orifice_input['assemblies_to_group'],
                             group_by=group_by)
    group_data = _group(power_array,
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
        # Get peak linear power in each assembly (pin-average value)
        # Estimate with 200 points; should get close to the max
        z = np.linspace(0.0, dassh_rx.core_length, 200)
        for a in dassh_rx.assemblies:
            if a.name in asm_to_group:
                peak_lin_power = np.zeros(z.shape[0])
                for zi in range(z.shape[0]):
                    peak_lin_power[zi] = np.average(
                        a.power.get_power(z[zi])['pins'])
                id.append(a.id)
                power.append(np.max(peak_lin_power))
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


def distribute_flow(orifice_inp, group_data, res_latest, res_parametric,
                    coolant_obj, t_in, prev_t_out=None, tol=1.0, iter_lim=50):
    """Distribute flow to minimize optimization variable

    Parameters
    ----------

    Returns
    -------

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

    # NOTE: Can I accept multiple inputs for different timesteps by
    # either (1) multiple input files, or (2) multiple CCCC files in a
    # single input file?

    # NOTE: Need to set up logging so that the DASSH sweeps return some
    # information

    # Initialize Coolant and Reactor objects for grouping calculations
    coolant_name = dassh_inp0.data['Core']['coolant_material']
    coolant = dassh_inp0.materials[coolant_name]
    # r = dassh.Reactor(inp)  # <-- normally will want this, but for test
    r0 = dassh.Reactor(dassh_inp0)

    # Perform parametric sweep
    msg = "Performing single-assembly parametric calculations"
    dassh_logger.log(20, msg)
    init_data = run_dassh_parametric(
        dassh_inp0, r0, orificing_input['assemblies_to_group'],
        orificing_input['value_to_optimize'])

    # Perform perfect orificing calculation
    msg = "Performing perfect-orificing calculation"
    dassh_logger.log(20, msg)
    perfect_data = run_dassh_perfect(
        dassh_inp0, dassh_logger, orificing_input)

    # Get power and perform grouping
    msg = "Grouping assemblies"
    dassh_logger.log(20, msg)
    group = group_by_power(orificing_input, r0)
    group = group[group[:, 0].argsort()]

    # Distribute flow among groups
    msg = "Distributing coolant flow among groups"
    dassh_logger.log(20, msg)
    m, tlim = distribute_flow(
        orificing_input, group, perfect_data, init_data,
        coolant, r0.inlet_temp, tol=0.5)
    print('T_opt_max: ', tlim)
    print('Flow rates: ', np.unique(m)[::-1])

    # Run DASSH iteration calculation
    msg = "Running DASSH with orificed assemblies (iter 1)"
    dassh_logger.log(20, msg)
    iter_data = run_dassh_orifice(dassh_inp0, dassh_logger,
                                  orificing_input, 1, group, m)
    summary_data = _summarize_group_data(group, iter_data)
    print(summary_data)

    # Redistribute coolant flow among groups to meet bulk coolant temperature
    print("Distributing coolant flow among groups")
    m, tlim = distribute_flow(orificing_input, group, iter_data, init_data,
                              coolant, r0.inlet_temp, tol=0.5,
                              prev_t_out=summary_data[-1, 0])
    print('T_opt_max: ', tlim)
    print('Flow rates: ', np.unique(m)[::-1])

    # Iteration 2
    print("Running DASSH with orificed assemblies (iter 2)")
    iter_data = run_dassh_orifice(dassh_inp0, dassh_logger,
                                  orificing_input, 2, group, m)
    summary_data = _summarize_group_data(group, iter_data)
    print(summary_data)

    # Redistribute coolant flow among groups to meet bulk coolant temperature
    print("Distributing coolant flow among groups")
    m, tlim = distribute_flow(orificing_input, group, iter_data, init_data,
                              coolant, r0.inlet_temp, tol=0.5,
                              prev_t_out=summary_data[-1, 0])
    print('T_opt_max: ', tlim)
    print('Flow rates: ', np.unique(m)[::-1])

    # Iteration 3
    print("Running DASSH with orificed assemblies (iter 3)")
    iter_data = run_dassh_orifice(dassh_inp0, dassh_logger,
                                  orificing_input, 3, group, m)
    summary_data = _summarize_group_data(group, iter_data)
    print(summary_data)
