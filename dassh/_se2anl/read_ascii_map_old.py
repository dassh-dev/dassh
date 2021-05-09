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
date: 2020-09-25
author: matz
comment: Methods to read ASCII subchannel temperature maps from SE2ANL

"""
########################################################################
import copy
import numpy as np


def read_all(se2out, bundle=1, len_units='m', temp_units='C'):
    """Read the ASCII maps from SE2ANL

    Parameters
    ----------
    se2out : str
        Path to SE2ANL output file
    bundle : int
        Bundle to pull temperature maps

    Returns
    -------
    dict
        Dictionary of temperature maps; keys are the axial positions

    """
    bundle = str(bundle).rjust(3, ' ')
    temp = {}
    with open(se2out, 'r') as f:
        out = f.read()
    # Read through the file and pull each map
    match1 = "1  TEMPERATURE MAP FOR BUNDLE " + bundle + " "
    match2 = "AXIAL HEIGHT"
    match3 = "1  TEMPERATURE MAP FOR BUNDLE"
    tag = 0
    while True:
        tag = out.find(match1, tag)
        if tag == -1:
            break
        tag = out.find(match2, tag)
        if len_units != 'in':
            if len_units == 'cm':
                h = float(out[tag + 15:tag + 22]) * 2.54
            elif len_units == 'm':
                h = float(out[tag + 15:tag + 22]) * 2.54 / 100
            else:
                msg = 'Length units must be "in", "cm", or "m"'
                raise ValueError(msg)
        h = np.around(h, 9)
        tag = out.find('\n', tag)
        tag2 = out.find(match3, tag)
        raw_map = out[tag:tag2]
        temp[h] = read(raw_map)
        if temp_units != 'F':
            if temp_units == 'C':
                temp[h] = (temp[h] - 32.0) * 5 / 9
            elif temp_units == 'K':
                temp[h] = (temp[h] - 32.0) * 5 / 9 + 273.15
            else:
                msg = 'Temperature units must be "C", "K", or "F"'
                raise ValueError(msg)
        temp[h] = np.around(temp[h], 9)
    return temp


def read(raw_ascii_map):
    """Read an ASCII temperature map from SE2ANL output"""
    # if '*' in raw_ascii_map:
    #     raise ValueError('Cannot handle maps with duct temperatures')

    ascii_map = process_raw_str(raw_ascii_map)
    n_ring = len(ascii_map[0]) - 1
    rtemps, stripped_map = strip_edge2(ascii_map, n_ring)
    while True:
        try:
            x, stripped_map = strip_ring2(stripped_map)
        except IndexError:
            break
        rtemps += x
    return np.array([t for t in rtemps[::-1] if t.isnumeric()], dtype='float')


def process_raw_str(ascii_str):
    """Read raw ascii map into list of lists"""
    asterisks = False
    if '*' in ascii_str:
        asterisks = True
    tmp = ascii_str.splitlines()
    map = []
    for line in tmp:
        if len(line) > 0 and not all([x == ' ' for x in line]):
            map.append([v for v in line.split(' ') if v != ''])
    # Remove asterisks, if they're there
    if asterisks:
        map = remove_asterisks(map)
    return map


def remove_asterisks(processed_str):
    """Remove asterisks used in SE2-ANL ascii maps to mark ducts, and
    all values outside of them"""
    # Remove the sublists that have only asterisks, and all the values
    # above (top) and below (bottom) them
    asterisk_sublist = []
    for i in range(len(processed_str)):
        if all(val == '*' for val in processed_str[i]):
            asterisk_sublist.append(i)

    # Track who you want gone
    top_to_remove = range(asterisk_sublist[0] + 1)
    bot_to_remove = range(len(processed_str) - asterisk_sublist[1])

    # Remove the top rows
    for i in top_to_remove:
        del processed_str[0]

    # Remove the bottom rows
    for i in bot_to_remove:
        del processed_str[-1]

    # Now remove the values outside the asterisks in each row
    for i in range(len(processed_str)):
        # First asterisk
        while True:
            if processed_str[i][0] == '*':
                del processed_str[i][0]
                break
            #
            del processed_str[i][0]
        #
        # Second asterisk
        while True:
            if processed_str[i][-1] == '*':
                del processed_str[i][-1]
                break
            #
            del processed_str[i][-1]

    # Clean out any stragglers: sometimes one gets put at the center
    for i in range(len(processed_str)):
        processed_str[i] = [x for x in processed_str[i] if x != '*']

    return processed_str


def strip_edge2(ascii_mapx, nr):
    """Strip the edge and corner subchannels off the list of lists

        ___6___
     1 /       \5
      /         \
      \         /
     2 \_______/ 4
           3

    """
    ascii_map = copy.deepcopy(ascii_mapx)
    rtemps = []

    # Side 6 (formerly side 1); collect and append at the end
    side6 = ascii_map[0][::-1]
    del ascii_map[0]
    rtemps.append(side6[-1])
    del side6[-1]

    # Side 1 (formerly side 2); no corner
    if nr % 2:
        i = 1
    else:
        i = 0
    for j in range(nr - 1):
        rtemps.append(ascii_map[i][0])
        del ascii_map[i][0]
        i += 2

    # Corner between sides 1-2 (formerly sides 2-3)
    if nr == 2:
        i -= 1
    rtemps.append(ascii_map[i][0])
    del ascii_map[i][0]

    # Side 2 (formerly side 3); no corner
    i += 1
    if nr > 2:
        i += 1
    for j in range(nr - 1):
        rtemps.append(ascii_map[i][0])
        del ascii_map[i][0]
        i += 2

    # Side 3 (formerly side 4); both corners
    rtemps += ascii_map[-1]
    del ascii_map[-1]

    # Side 4 (formerly side 5); no corners
    if nr > 2:
        i = len(ascii_map) - 1
    else:
        i = len(ascii_map) - 2
    for j in range(nr - 1):
        rtemps.append(ascii_map[i][-1])
        del ascii_map[i][-1]
        i -= 2

    # Corner between sides 4, 5 (formerly sides 5, 6)
    if nr == 2:
        i += 1
    rtemps.append(ascii_map[i][-1])
    del ascii_map[i][-1]

    # Side 5 (formerly side 6)
    i -= 1
    if nr > 2:
        i -= 1
    for j in range(nr - 1):
        rtemps.append(ascii_map[i][-1])
        del ascii_map[i][-1]
        i -= 2
    rtemps = rtemps + side6
    return rtemps, ascii_map


def strip_ring2(ascii_mapx):
    """Walk around the list of lists and collect the outer ring of
    interior subchannels

        ___6___
     1 /       \5
      /         \
      \         /
     2 \_______/ 4
           3

    """
    ascii_map = copy.deepcopy(ascii_mapx)
    ascii_map = [x for x in ascii_map if len(x) > 0]
    rtemps = []

    # Side 6 (formerly side 1); collect and insert at end
    side6 = []
    for i in reversed(range(len(ascii_map[0]))):
        side6.append(ascii_map[0][i])
        side6.append(ascii_map[1][i])
    del ascii_map[1][:len(ascii_map[1]) - 1]
    del ascii_map[0]

    # Sides 1 and 2 (formerly 2 and 3)
    for i in range(1, len(ascii_map)):
        rtemps.append(ascii_map[i][0])
        del ascii_map[i][0]

    # Side 3 (formerly 4)
    for i in range(len(ascii_map[-1])):
        rtemps.append(ascii_map[-2][i])
        rtemps.append(ascii_map[-1][i])
    del ascii_map[-1]
    del ascii_map[-1][:-1]

    # Sides 4 and 5 (formerly 5 and 6)
    for i in reversed(range(len(ascii_map))):
        rtemps.append(ascii_map[i][-1])
        del ascii_map[i][-1]

    # Insert side 6
    rtemps.insert(0, side6[-1])
    rtemps = rtemps + side6[:-1]
    return rtemps, [x for x in ascii_map if len(x) > 0]


# def strip_edge(ascii_mapx, nr):
#     """Strip the edge and corner subchannels off the list of lists"""
#     ascii_map = copy.deepcopy(ascii_mapx)
#     rtemps = []
#
#     # Side 1
#     rtemps += ascii_map[0][::-1]
#     del ascii_map[0]
#
#     # Side 2 (no corner)
#     if nr == 2:
#         i = 1
#     else:
#         i = 0
#     for j in range(nr - 1):
#         rtemps.append(ascii_map[i][0])
#         del ascii_map[i][0]
#         i += 2
#
#     # Corner between sides 2-3
#     if nr == 2:
#         i -= 1
#     rtemps.append(ascii_map[i][0])
#     del ascii_map[i][0]
#
#     # Side 3 (no corner)
#     i += 1
#     if nr > 2:
#         i += 1
#     for j in range(nr - 1):
#         rtemps.append(ascii_map[i][0])
#         del ascii_map[i][0]
#         i += 2
#
#     # Side 4 (both corners)
#     rtemps += ascii_map[-1]
#     del ascii_map[-1]
#
#     # Side 5 (no corners)
#     if nr > 2:
#         i = len(ascii_map) - 1
#     else:
#         i = len(ascii_map) - 2
#     for j in range(nr - 1):
#         rtemps.append(ascii_map[i][-1])
#         del ascii_map[i][-1]
#         i -= 2
#
#     # Corner between sides 5, 6
#     if nr == 2:
#         i += 1
#     rtemps.append(ascii_map[i][-1])
#     del ascii_map[i][-1]
#
#     # Side 6
#     i -= 1
#     if nr > 2:
#         i -= 1
#     for j in range(nr - 1):
#         rtemps.append(ascii_map[i][-1])
#         del ascii_map[i][-1]
#         i -= 2
#     return rtemps, ascii_map
#
#
# def strip_ring(ascii_mapx):
#     """Walk around the list of lists and collect the outer ring of
#     interior subchannels"""
#     ascii_map = copy.deepcopy(ascii_mapx)
#     ascii_map = [x for x in ascii_map if len(x) > 0]
#     rtemps = []
#
#     # Side 1
#     for i in reversed(range(len(ascii_map[0]))):
#         rtemps.append(ascii_map[0][i])
#         rtemps.append(ascii_map[1][i])
#     del ascii_map[1][:len(ascii_map[1]) - 1]
#     del ascii_map[0]
#
#     # Sides 2 and 3
#     for i in range(1, len(ascii_map)):
#         rtemps.append(ascii_map[i][0])
#         del ascii_map[i][0]
#
#     # Side 4
#     for i in range(len(ascii_map[-1])):
#         rtemps.append(ascii_map[-2][i])
#         rtemps.append(ascii_map[-1][i])
#     del ascii_map[-1]
#     del ascii_map[-1][:-1]
#
#     # Sides 5 and 6
#     for i in reversed(range(len(ascii_map))):
#         rtemps.append(ascii_map[i][-1])
#         del ascii_map[i][-1]
#     return rtemps, [x for x in ascii_map if len(x) > 0]
