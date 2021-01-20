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
date: 2020-08-05
author: Milos Atz
This module defines the methods for the DASSH orificing algorithm

"""
########################################################################

import numpy as np
import dassh

# Procedure
# 1. Run with guess for perfect orificing
# 2. Calculate mfr-weighted cladding temperature Tc_avg
# 3. Estimate m required for peak clad = Tc_avg in each assembly
# 4. Group based on m from (3)
# 5. With groups from (4), find mass flow rates in each group such that
#       a) The group max clad temp = max clad temp of other groups
#       b) The overall core mfr-weighted outlet coolant temp = target value
# 6. Repeat (3) but this time for peak clad = max clad temp from (5)
# 7. Group based on m from (6); if same, done. If not, go to 8.
# 8. Repeat (5) using groups from (4).
# 9. ...and so on.

# Then run with guess for groups and confirm all the shit you need to
# Can I also factor in pressure drop? I can get get dP(mfr) from DASSH
# based on pressure drop data

# To do
# Connect this to DASSH: need code to creates sweep_data array
# Pressure drop constraint: effectively limits MFR for each asm type
#   - After the first pass in DASSH, you can easily project the DP vs.
#     flow rate and therefore inform the code on "XX is the max flow
#     rate in this assembly" (maxDP = 0.5 MPa)
#   - although with the heterongeous axial treatment, the relationship
#     between flow rate and total DP might not be exactly proportional
#     to v^2 so you might have to correct it with each iteration (and
#     thus use a "convergence criteria" (around 0.01 MPa) for it)
#
# To connect with DASSH: leave the main sweep in "main" as is (modify
# to loop over multiple cycles, if necessary).
# After it completes, have an if statement that checks for orificing
# input in the input file. If present, predict groups, resweep, and update
# until convergence.

# Need to figure out thermal striping requirement.

# class Orificing(object):
#     """Object designed to carry out DASSH orificing algorithm
#
#     Parameters
#     ----------
#
#
#     Notes
#     -----
#
#     """
#
#     def __init__(self, T_in, ):
#         """
#
#         """

def do_the_thing(r_obj, n_groups, peak_temp_to_minimize, dT_constraint):
    """."""
    component, region = peak_temp_to_minimize.split(' ')
    sweep_data = _get_peak_temps(r_obj, component, region)
    groups, mass_flow_rates = _min_max(n_groups, sweep_data, dT_constraint)
    # Then need to turn around and re run sweep(s)


def _get_peak_temps(r_obj, component, region):
    """Create the peak temperature data from the previous sweep

    Parameters
    ----------
    r_obj : DASSH Reactor object
    component : str {'clad', 'fuel'}
        Report clad/fuel temps at ht. of peak temp for this component
        - 'clad': Clad midwall
        - 'fuel': Fuel centerline
    region : str
        Subregion within component where peak temperature is taken
        If component = 'clad': {'od', 'mw', 'id'}
        If component = 'fuel': {'od', 'cl'}

    Returns
    -------
    numpy.ndarray
        Contains assembly power, flow rate, and requested peak
        temperature data for each assembly

    """
    # Initialize array
    data = np.zeros((len(r_obj.assemblies), 9))

    # Use the DASSH PeakTempTable to import the data; parse from string
    # back into float as needed (use 6 decimal places per float)
    t = dassh.PeakTempTable(ffmt2=6)
    t.make(r_obj, component, region)
    tmp = t.table.splitlines()[3:]
    for i in range(len(tmp)):
        line = [float(xi) for xi in tmp[i].split(' ') if xi != '']
        data[i, 3:] = line[6:]  # pull temps, ditch ID/loc/height/power
        # Get the other data from the assemblies
        a = r_obj.assemblies[i]
        data[i, 0] = np.sum(r_obj.power[a.name].power[i])
        data[i, 1] = a.flow_rate
        data[i, 2] = a.avg_coolant_temp

    # Figure out how many columns you need in the sweep data table;
    # depends on which peak temperature you want, because the last
    # column has to be the peak temperature of interest
    # First three cols are fixed: power, flow rate, Tavg coolant outlet
    # At least one col for coolant temp @ height of peak temp selected
    # Then, clad od, clad mw, clad id, fuel od, fuel cl temps
    required_col = 3 + t.col_id[component][region] - 3
    data = data[:, :required_col]
    return data


def _min_max(n_grp, sweep_data, dT_cons):
    """Determine orifice groups and mass flow rates that minimize
    the maximum of some peak temperature

    n_grp : int
        Number of orifice groups
    sweep_data : numpy.ndarray
        Data obtained from an initial temperature sweep; columns are:
        1.  Assembly power (W)
        2.  Assembly flow rate (kg/s)
        3.  Assembly average coolant outlet dT (K)
        4+. Peak coolant and, if applicable, radial pin component
            dT; the last column is the data to use to
            optimize the orificing (this is the objective function)
    dT_cons : float
        Target bulk average coolant outlet dT across the core (K);
        this is a constraint on the optimization

    """
    # Assembly power / cp   sweep_data[:, 0]
    # Mass flow rates       sweep_data[:, 1]
    # Average coolant dT    sweep_data[:, 2]
    # Peak coolant dT       sweep_data[:, 3]
    # Peak clad OD/MW/ID dT sweep_data[:, 4/5/6] (if applicable)
    # Peak fuel OD/CL dT    sweep_data[:, 7/8] (if applicabale)
    #
    # Calculate mfr-weighted peak temperature; objective is whatever
    # is the last column (could be peak coolant or some clad/fuel
    # component; what's input determines what is done)
    dT_peak_avg = (np.sum(sweep_data[:, -1] * sweep_data[:, 1])
                   / np.sum(sweep_data[:, 1]))
    #
    # Calculate coolant temperature peaking factor
    qx = sweep_data[:, 3] / sweep_data[:, 2]
    #
    # Calculate radial scalar dT (if peak coolant temperature is the
    # objective function, this will be zero; otherwise, it assumes
    # constant dT radially across the pin)
    dT_radpin = sweep_data[:, -1] - sweep_data[:, 3]
    #
    # Estimate MFR req'd to achieve peak temp equal to avg in all asm
    mfr_req = 1 / ((dT_peak_avg - dT_radpin) / sweep_data[:, 0] / qx)
    #
    # Group by MFR
    grp = group(mfr_req, n_grp, 0.01, 0.001) - 1
    asm_in_grp = np.array([len(grp[grp == x]) for x in set(grp)])
    # for i in range(len(n_grps)):
    #     print(f'N asm in group {i + 1}: {n_grps[i])}')
    #
    # Calculate mass flow rates required for max(Tp_clad) to be equal for
    # all groups AND for bulk coolant average temperature to be equal
    # to T_out
    #
    # Step 1. Determine flow rates required such that all group_average
    # outlet temperatures = 500
    mfr_grp = np.zeros(3)
    for i in range(n_grp):
        mfr_grp[i] = np.sum(sweep_data[:, 0][grp == i]) / dT_cons
        mfr_grp[i] /= asm_in_grp[i]
    #
    # Step 2. Determine objective peak temperature for each group
    # with these flow rates
    mfr = mfr_grp[grp]
    dTp = sweep_data[:, 0] * qx / mfr + dT_radpin
    dTp_max = np.zeros(3)
    for i in range(n_grp):
        dTp_max[i] = np.max(dTp[grp == i])
    #
    # Step 3. Calculate flow rate and assembly count-averaged max
    # peak temperature based on these flow rates; this is a first
    # guess at what should be the maximum peak temperature for the
    # objective function in each group.
    dTp_max_avg = (np.sum(asm_in_grp * mfr_grp * dTp_max)
                   / np.sum(asm_in_grp * mfr_grp))
    #
    # Step 4. Determine flow rates required to set peak objective
    # temperature equal to the value calculated in (3)
    for i in range(n_grp):
        j = np.where(dTp == dTp_max[i])[0][0]
        mfr_grp[i] = sweep_data[j, 0] * qx[j] / (dTp_max_avg - dT_radpin[j])
    #
    # Step 5. Scale mfr so that dT overall equals requested value
    # (should be a very small adjustment, on the order of 1e-3)
    mfr = mfr_grp[grp]
    dT_cool_est = sweep_data[:, 0] / mfr
    dT_cool_avg_est = np.sum(dT_cool_est * mfr) / np.sum(mfr)
    print(dT_cool_avg_est)
    x = dT_cool_avg_est / dT_cons
    print(x)
    mfr *= x
    dT_cool_est = sweep_data[:, 0] / mfr
    dT_cool_avg_est = np.sum(dT_cool_est * mfr) / np.sum(mfr)
    print(dT_cool_avg_est)
    assert np.isclose(np.sum(dT_cool_est * mfr) / np.sum(mfr), dT_cons)
    for i in range(n_grp):
        j = np.where(dTp == dTp_max[i])[0][0]
        print(mfr[j])
        print(sweep_data[j, 0] * qx[j] / mfr[j] + dT_radpin[j] + 350)
    #
    # Return groups and required mass flow rates
    return grp, mfr_grp


test_groups, test_mfr = _min_max(3, sweep_data, 150.0)


def group(x, n_grp, dx, ddx, apg=1, dcrit='avg'):
    """Divide assemblies into orifice groups based on mass flow
    rate or some other parameter

    Parameters
    ----------
    x : numpy.ndarray
        Data with which to create assembly groups. Can be power,
        coolant temperatures, or peak clad/fuel temperatures
    n_grp : int
        Target number of orifice groups
    dx : float
        Starting criterion (difference in value) to determine split
        between orifice groups based on input data
    ddx : float
        Increment with which to update dx to de-constrain the problem
        until the desired number of groups are found.
    apg : int
        Minimum number of assemblies in each group
    dcrit (optional) : str
        Indicate the basis to use to calculate the delta in the data
        that marks the division between groups (default = 'avg')

    Returns
    -------
    numpy.ndarray
        Orifice groups for each assembly in the input array

    """
    og_order = np.argsort(x)[::-1]
    x = np.sort(x)[::-1]
    #
    # Set up delta calculation; pulls "if" statement out of loop
    if dcrit == 'max':
        calc_delta = lambda g_max, g_min, g_avg: (g_max - g_min) / g_max
    else:
        calc_delta = lambda g_max, g_min, g_avg: (g_max - g_min) / g_avg
    #
    while True:
        x_max = 0    # maximum value in group (placeholder)
        x_min = 1e9  # minimum value in group (placeholder)
        x_tot = 0    # total value in group (placeholder)
        grp = 1      # group ID
        n = 0        # number of assemblies in the group
        group = np.ones(len(x))  # result
        for i in range(len(x)):
            n += 1
            x_tot += x[i]
            x_avg = x_tot / n
            x_max = max([x_max, x[i]])
            x_min = min([x_min, x[i]])
            #
            # If delta gets too large, start a new group
            if calc_delta(x_max, x_min, x_avg) > dx and n > apg:
                grp += 1
                n = 1
                x_tot = x[i]
                x_max = x[i]
                x_min = x[i]
                #
            # Mark the group
            group[i] = grp
        #
        # See how we've done; update if necessary
        # print(ddp, max(group))
        if max(group) > n_grp:
            dx += ddx
        else:
            break
        #
    rgroup = np.zeros(len(group), dtype='int')
    for i in range(len(group)):
        rgroup[og_order[i]] = group[i]
    return rgroup
