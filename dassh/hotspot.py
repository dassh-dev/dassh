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
date: 2023-03-02
author: matz
comment: Hot spot analysis via the semistatistical horizontal method
"""
########################################################################
import os
import sys
import logging
import numpy as np


module_logger = logging.getLogger('dassh.hotspot')


# Path to module - use to identify paths to builtin HCF
_ROOT = os.path.dirname(os.path.abspath(__file__))
# Names of built-in HCF options
_BUILTINS = ['crbr_blanket_clad_mw',
             'crbr_fuel_clad_mw',
             'ebrii_markv_fuel_cl',
             'fftf_clad_mw',
             'fftf_fuel_cl']
# Temperature options available for hotspot calculation and number of
# cols needed in the corresponding HCF subfactor CSV file
_REGIONS = ['coolant',
            'clad_od',
            'clad_mw',
            'clad_id',
            'fuel_od',
            'fuel_cl']
_COLS_NEEDED = {
    'coolant': 3,
    'clad_od': 4,
    'clad_mw': 5,
    'clad_id': 5,
    'fuel_od': 6,
    'fuel_cl': 7}


def _setup_postprocess(dassh_inp):
    """Read in necessary information from input into Reactor
    object to enable hotspot calculation after sweep"""
    hotspot_dict = {}
    keys = ('input_sigma', 'output_sigma', 'subfactors')
    for a in dassh_inp.data['Assembly'].keys():
        # Skip if not hotspot inputs are provided
        if not dassh_inp.data['Assembly'][a]['Hotspot']:
            continue
        # Add information for each hotspot calculation
        hotspot_dict[a] = {}
        for hs in dassh_inp.data['Assembly'][a]['Hotspot'].keys():
            temp = dassh_inp.data['Assembly'][a]['Hotspot'][hs]['temperature']
            # Coolant hotspot calculation does not require PinModel
            # (or FuelModel) inputs. If "coolant" is not one of the
            # inputs and PinModel (or FuelModel) is not specified,
            # skip the hotspot calculation
            if temp != 'coolant':
                if not any(k in dassh_inp.data['Assembly'][a].keys()
                           for k in ('PinModel', 'FuelModel')):
                    msg = 'Skipping "{}" hotspot calculation for ' \
                          'assembly "{}": No "PinModel" / "FuelModel" ' \
                          'input provided.'
                    msg = msg.format(temp, a)
                    module_logger.log(30, f'WARNING: {msg}')
                    continue
            # Otherwise, keep going: add assembly to hotspot
            # dict if not already there
            if a not in hotspot_dict.keys():
                hotspot_dict[a] = {}
            hotspot_dict[a][temp] = {}
            for k in keys:
                if k == 'subfactors':
                    name = dassh_inp.data['Assembly'][a]['Hotspot'][hs][k]
                    if name in _BUILTINS:  # Point to builtin CSV
                        fname = 'hcf_' + name + '.csv'
                        fpath = os.path.join(_ROOT, 'data', fname)
                    else:  # It's a filepath to a CSV
                        fpath = os.path.abspath(
                            os.path.join(dassh_inp.path, name))
                    if not os.path.exists(fpath):
                        msg = 'Path to hotspot subfactors ' \
                              f'CSV does not exist: {fpath}'
                        module_logger.log(40, f'ERROR: {msg}')
                        sys.exit(1)
                    else:
                        hotspot_dict[a][temp][k] = fpath
                else:
                    hotspot_dict[a][temp][k] = \
                        dassh_inp.data['Assembly'][a]['Hotspot'][hs][k]
    if hotspot_dict:
        return hotspot_dict


def analyze(r_obj):
    """Postprocess temperature results to obtain hotspot temperatures
    based on user-supplied subfactors

    Parameters
    ----------
    r_obj : DASSH Reactor object
        Contains model state after temperature sweep

    """
    if not r_obj._options['hotspot']:
        return None
    asm_ids = {k: [] for k in _REGIONS}
    asm_names = {k: [] for k in _REGIONS}
    peak_temps = {k: [] for k in _REGIONS}
    for asm_name in r_obj._options['hotspot'].keys():
        hs = r_obj._options['hotspot'][asm_name]
        if asm_name in r_obj._options['hotspot'].keys():
            # Collect assembly IDs that match the name
            ids = [a.id for a in r_obj.assemblies if a.name == asm_name]
            n_asm = len(ids)
            for k in _REGIONS:  # Do each hotspot calc
                if k in hs.keys():
                    # Add assembly IDs/names to relevant list
                    asm_ids[k] += ids
                    asm_names[k] += [asm_name for i in range(n_asm)]
                    # Pull dT from the assembly
                    dT = _get_peak_dt(r_obj, asm_name, k)
                    # Read the HCF from the table at the specified path
                    subf, expr = _read_hcf_table(
                        hs[k]['subfactors'],
                        _COLS_NEEDED[k])
                    # Split the clad OD-MW and MW-ID subfactors
                    if k in ('clad_id', 'fuel_od', 'fuel_cl'):
                        subf, expr = _split_clad_subfactors(subf, expr)
                    subf = _evaluate_hcf_expr(subf, expr, dT)
                    # Crop the subfactors table based on the size of dT
                    for typ in subf.keys():
                        subf[typ] = subf[typ][:, :, :dT.shape[1]]
                    # Do the calculations
                    peak_temps[k].append(
                        calculate_temps(r_obj.inlet_temp, dT, subf,
                                        IN_sigma=hs[k]['input_sigma'],
                                        OUT_sigma=hs[k]['output_sigma']))

    # Need to sort! Create lists of ids, names, and temps. Sort
    # all according to ids. Then write into output table
    for k in _REGIONS:
        # Remove the empty arrays if no hotspot temps
        # were calculated for a particular region
        if asm_ids[k] == []:
            del asm_ids[k]
            del asm_names[k]
            assert np.all(arr == 0.0 for arr in peak_temps[k])
            del peak_temps[k]
        else:
            order = np.argsort(asm_ids[k])
            asm_ids[k] = [asm_ids[k][i] for i in order]
            asm_names[k] = [asm_names[k][i] for i in order]
            peak_temps[k] = np.vstack(peak_temps[k])
            peak_temps[k] = peak_temps[k][order]
    # return peak_temps, asm_ids, asm_names
    return peak_temps, asm_ids


def calculate_temps(T_in, dT, hcf, IN_sigma=3, OUT_sigma=2):
    """Calculate hotspot (eg 2-sigma) coolant, clad, or fuel
    temperatures using semistatistical horizontal method

    Parameters
    ----------
    T_in : float
        Coolant inlet temperature
    dT : numpy.ndarray
        Temperature rise across each step: N_asm rows
        If clad, 3 cols: dT_cool, dT_film, dT_clad
        If fuel, 5 cols: dT_cool, dT_film, dT_clad, dT_gap, dT_fuel
    hcf : dict
        Dictionary of clad MW hot channel/spot subfactors
        Two entries: "direct" and "statistical"; each is array with
        shape N_asm x N_subfactors x N_terms (N_terms = 3 if clad, 5 if fuel)
    IN_sigma (optional) : int
        Degree of uncertainty in the provided subfactors
        (default=3, as in "3-sigma uncertainties")
    OUT_sigma (optional) : int
        Degree of uncertainty in the output hotspot temperatures
        (default=2, as in "2-sigma peak temperatures")

    Returns
    -------
    numpy.ndarray
        Hotspot temperature in each assembly (N_asm x 1)

    """
    # Calculate zero-sigma delta Ts based on direct subfactors
    # hcf['direct'] is N_asm x N_subfactors x N_terms
    # dT_subfactors is product of all direct subfactors for each
    # assembly, term; shape is N_asm x N_terms
    dT_subfactors = np.prod(hcf['direct'], axis=1)

    # Calculate the zero-sigma dT: array shape is N_asm x N_terms
    zero_sig_dT = dT * dT_subfactors

    # Calculate statistical term based on 3-sigma subfactors
    # hcf['statistical'] is N_asm x N_subfactors x N_terms
    # The uncertainties are the products of (a) the difference
    # between the subfactors and unity (SF - 1) and (b) the
    # 0-sigma temperature deltas, summed over all terms
    hcf_stat_m1 = hcf['statistical'] - 1

    # First calculate individual products, then sum over terms
    # Note: "IN_sig" refers to the "sigma" being whatever value
    # is given in the HCF subfactors. By default, the values are
    # assumed to be 3-sigma uncertainties, but the user can
    # specify different values.
    IN_sig_unc = zero_sig_dT[:, np.newaxis, :] * hcf_stat_m1
    IN_sig_unc = np.cumsum(IN_sig_unc, axis=2)  # N_asm x N_sf
    # Now do sum of squares on these --> N_asm
    IN_sig_sOs = np.sqrt(np.sum(IN_sig_unc**2, axis=1))
    # T = T_in + np.sum(zero_sig_dT, axis=1)
    T = T_in + np.cumsum(zero_sig_dT, axis=1)
    T += OUT_sigma * IN_sig_sOs / IN_sigma
    return T


def _get_peak_dt(r_obj, asm_name, value):
    """x"""
    idx = {'clad_od': 5,
           'clad_mw': 6,
           'clad_id': 7,
           'fuel_od': 8,
           'fuel_cl': 9}
    t = []
    for a in r_obj.assemblies:
        if a.name == asm_name:
            assert 'pin' in a._peak.keys()
            tmp = [r_obj.inlet_temp]
            if value == 'coolant':
                tmp.append(a._peak['cool'][0])
            else:
                tmp += a._peak['pin'][value][2][3:idx[value]]
            t.append(tmp)
    t = np.array(t)
    dt = t[:, 1:] - t[:, :-1]
    return dt


def _read_hcf_table(path_to_hcf_table, cols_needed=None):
    """Read hot channel subfactors from CSV

    Parameters
    ----------
    path_to_hcf_table : str
        Path to the HCF CSV file
    cols_needed (optional) : int
        Minimum number of columns that the CSV needs to have

    Notes
    -----
    Fixed column format is as follows (comma separated):
        Cols: Subfactor, Type, Coolant, Film, Cladding, Gap (*), Fuel (*)
        Subfactor name, Direct / Statistical, Subfactor values...
    Subfactor value may either be:
        1. Float
        2. Some kind of Python-evaluable expression in terms of dT

    """
    path_to_hcf_table = os.path.abspath(path_to_hcf_table)
    with open(path_to_hcf_table, mode='r', encoding='utf-8-sig') as f:
        hcf_table = f.read()

    # Check header row length - assess the number of columns
    header = hcf_table.splitlines()[0].split(',')
    n_cols = len(header)
    if cols_needed:
        if n_cols < cols_needed:
            msg = 'Incorrect number of columns in HCF table: ' \
                  f'{path_to_hcf_table} \nNeed at least ' \
                  f'{cols_needed} columns for requested ' \
                  f'hotspot temp calculation; found {n_cols} columns'
            module_logger.log(40, f'ERROR: {msg}')
            sys.exit(1)
    else:
        if n_cols not in (5, 7):
            msg = 'Incorrect number of columns in HCF table: ' \
                  f'{path_to_hcf_table} \nNeed (at least) 5 columns ' \
                  'for hotspot clad temp and 7 cols for hotspot fuel '\
                  f'temp; found {n_cols} columns'
            module_logger.log(40, f'ERROR: {msg}')
            sys.exit(1)

    # Check header row elements
    error = _check_header(header)
    if error:
        msg = f'Incorrect header row in HCF table: {path_to_hcf_table}\n'
        msg += error
        module_logger.log(40, f'ERROR: {msg}')
        sys.exit(1)

    # Determine the number of subfactors - count the number lines
    # minus the header and any expression definitions
    n_rows = len([x for x in hcf_table.splitlines() if x[0] != '*'])
    n_rows -= 1  # omit the header line

    # Count the number of direct/statistical subfactors
    n_direct = hcf_table.count('Direct,')
    n_direct += hcf_table.count('direct,')
    n_stat = hcf_table.count('Statistical,')
    n_stat += hcf_table.count('statistical,')
    assert n_direct + n_stat == n_rows

    # Prearrange HCF tables as empty lists, to be appended to.
    hcf = {'direct': [], 'statistical': []}

    # Walk through the table, line by line. Split each line at
    # the commas. Try to "float" each entry. If you can, it's a
    # static subfactor. If not, try evaluating it.
    hcf_table_lines = hcf_table.splitlines()[1:]  # skip the header
    n_lines = len(hcf_table_lines)
    expressions = {}
    for row in range(n_lines):
        line = hcf_table_lines[row].split(',')
        key = line[1].lower()
        assert key in ('direct', 'statistical'), key
        tmp = []
        for col in range(n_cols - 2):
            val = line[col + 2]
            if _check_float(val):
                tmp.append(float(val))
            elif _check_valid_expr(val):
                tmp.append(np.nan)
                krow = len(hcf[key])
                expressions[(key, krow, col)] = val
            else:
                msg = 'ERROR: Invalid expression! '
                msg += f'Found: "{val}" in CSV {path_to_hcf_table}'
                module_logger.log(40, msg)
                sys.exit(1)
        hcf[key].append(tmp)

    # Tables should be complete - can turn into numpy arrays
    hcf['direct'] = np.array(hcf['direct'], dtype=float)
    hcf['statistical'] = np.array(hcf['statistical'], dtype=float)
    return hcf, expressions


def _split_clad_subfactors(subf, expr):
    """Split clad subfactors into two columns to apply separately
    to OD-MW and MW-ID temperature deltas"""
    subf_new = {}
    for k in subf.keys():
        new = np.zeros((subf[k].shape[0], subf[k].shape[1] + 1))
        new[:, :3] = subf[k][:, :3]
        new[:, 3:] = subf[k][:, 2:]
        subf_new[k] = new
    expr_new = {}
    for k in expr.keys():
        if k[2] < 2:
            expr_new[k] = expr[k]
        elif k[2] == 2:
            k_new = (k[0], k[1], 3)
            expr_new[k] = expr[k]
            expr_new[k_new] = expr[k]
        else:
            k_new = (k[0], k[1], k[2] + 1)
            expr_new[k_new] = expr[k]
    return subf_new, expr_new


def _evaluate_hcf_expr(hcf_dict, expr_dict, dT_in):
    """Evaluate user-provided expressions for subfactors based on
    dT for each assembly."""
    # Input HCF arrays are N_subfactor x N_dT
    # Expand to be N_asm x N_subfactor x N_dT
    n_asm = dT_in.shape[0]
    for k in hcf_dict.keys():
        shape = (n_asm, hcf_dict[k].shape[0], hcf_dict[k].shape[1])
        hcf_dict[k] = np.ones(shape) * hcf_dict[k]

    # Get an array for each expression: N_asm x 1
    # Then stick it in the expanded array
    for k in expr_dict.keys():
        typ, r, c = k
        evalated_expr = _eval_expr(expr_dict[k], dT_in[:, k[2]])
        hcf_dict[k[0]][:, r, c] = evalated_expr

    return hcf_dict


def _check_float(x):
    """Check whether string can be cast to float"""
    try:
        float(x)
        return True
    except ValueError:
        return False


def _check_valid_expr(x, dT=100):
    """Check whether string can be evaluated as expression for dT"""
    try:
        eval(x)
        return True
    except:
        return False


def _check_header(header_element_list):
    correct = ['Subfactor', 'Type', 'Coolant', 'Film',
               'Cladding', 'Gap', 'Fuel']
    for i in range(len(header_element_list)):
        if header_element_list[i] != correct[i]:
            msg = f'Col {i + 1} header must be "{correct[i]}"; '
            msg += f'found "{header_element_list[i]}"'
            return msg
    return


def _count_expr(hcf_table_str):
    """Count number of user-provided expressions in table"""
    n_expr = 1
    while True:
        tag = hcf_table_str.find('*' * n_expr + ',')
        if tag == -1:
            break
        elif n_expr > 100:
            raise ValueError('Too many expressions! 100 and counting...')
        else:
            n_expr += 1
    # The last one will count too far, so subtract it before return
    return n_expr - 1


def _eval_expr(expr, dT):
    """Evaluate expression that is function of dT"""
    result = eval(expr)
    # Remove np.infs and np.nans
    result[result == np.inf] = 1.0
    result[result == np.nan] = 1.0
    return result
