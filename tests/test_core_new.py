import copy
import numpy as np
import pytest
import dassh
from dassh import core


@pytest.fixture
def one_asm_core(testdir, coolant, simple_asm):
    """DASSH Core object for 3-asm core with simple, 7-pin asm"""
    asm_list = np.array([0])
    core_obj = dassh.Core(asm_list, 0.12, 1.0, coolant,
                          inlet_temperature=623.15, model='flow')
    assemblies = []
    loc = [(0, 0)]
    for ai in range(len(loc)):
        tmp = simple_asm.clone(loc[ai])
        assemblies.append(tmp)
    core_obj.load(assemblies)
    return [assemblies, core_obj]


@pytest.fixture
def two_asm_core(testdir, coolant, c_ctrl_asm, c_fuel_asm):
    """DASSH Core object for 3-asm core with simple, 7-pin asm"""
    asm_list = np.array([0, 1, np.nan, np.nan, np.nan, np.nan, np.nan])
    core_obj = dassh.Core(asm_list, 0.12, 1.0, coolant,
                          inlet_temperature=623.15, model='flow')
    assemblies = [c_ctrl_asm, c_fuel_asm]
    core_obj.load(assemblies)
    return [assemblies, core_obj]


def test_poop(two_asm_core):
    """x"""
    asm_list, core_obj = two_asm_core
    # print(core_obj._Rcond.shape)
    tds = np.random.random((2, 54))
    # print(core_obj._conv_util['const'].shape)
    # print(core_obj._conv_util['const'][-1])
    # assert 0
    core_obj.calculate_gap_temperatures(0.001, tds)
    # print(asm_list[0].duct_oftf)
    # print(core.asm_sc_adj)
    # print(core_obj.gap_params['L'])
    # print(core_obj._Rcond)
    # print(core.asm_sc_types)
    # print(core.sc_types)
    # print(core.sc_per_side)
    # print(core.n_sc)
    # print(core.sc_adj)
    # print(core.gap_params['wp'])
    # print(core.gap_params['area'])
    # print(core_obj.gap_params['L'])
    # print(core.gap_params['de'])
    # tmp = core_obj.map_interassembly_sc(asm_list)
    # # print(tmp[5])
    # print(core_obj.asm_sc_adj)
    # test = _globalize_dist_betw_sc(core_obj, tmp[5])
    # print(test)
    # assert 0
    # asms, core = one_asm_core
    # asm_adj_sc = []
    # sc_idx = 0
    # sc_per_side = 1
    # for side in range(6):
    #     tmp, sc_idx = _index(core, 0, side, sc_idx, 1, asm_adj_sc)
    #     print(tmp)
    # assert 0


def _globalize_dist_betw_sc(self, asm_L):
    """Collect global distances between subchannels

    Parameters
    ----------
    asm_L : list
        List (len = N_asm) of numpy ndarrays (N_adj_sc x 3) with
        distances between subchannels adjacent to each assembly

    Returns
    -------
    numpy ndarray
        Distance between each subchannel (N_sc x 3)

    """
    # First: fill in relative to each assembly
    L = np.zeros((self.n_sc, 3))
    # for a in range(self.n_asm):
    for a in range(2):
        for i in range(self.asm_sc_adj[a].shape[0]):
            sci = self.asm_sc_adj[a][i] - 1
            if np.all(L[sci] == 0):
                L[sci] = asm_L[a][i]
            else:
                # Gap edge subchannels need no revision
                if self.asm_sc_types[a][i] == 0:
                    continue
                else:
                    # If all values filled, no need to replace any
                    if not np.any(L[sci] == 0):
                        continue
                    # Fill in remaining corner value
                    else:
                        L[sci, 2] = asm_L[a][i, 1]
    return L


def _index(self, asm, side, sc_id, sc_per_side, already_idx):
    """Count gap subchannel indices along an assembly side

    Parameters
    ----------
    asm : int
        Active assembly index
    side : int
        Active hex side
    sc_id : int
        Active gap subchannel index
    sc_per_side : int
        Number of gap edge subchannels along this hex side
    already_idx : list
        List of lists containing the already-indexed adjacency
        between previous assemblies and gap subchannels

    Returns
    -------
    list
        Subchannel indices along the active hex side of the
        active assembly

    """
    if self._need_to_count_side(asm, side):
        # Count edge subchannels along that side
        to_add = list(np.arange(sc_id + 1, sc_id + 1 + sc_per_side))
        sc_id += sc_per_side  # update the sc index
        # Check if you need to count the trailing corner
        if self._need_to_count_corner(asm, side):
            # print(side, 11)
            sc_id += 1
            to_add.append(sc_id)
        # If you don't need to count a new corner sc, get
        # the existing corner from the adjacent assembly
        else:
            # print(side, 12)
            to_add.append(
                self._find_corner_sc(
                    asm, side, already_idx))
    else:
        # print(side, 2)
        # get the subchannels that live here, including
        # the trailing corner, which must already be
        # defined if these side subchannels are defined.
        to_add = self._find_side_sc(asm, side, already_idx)
    return to_add, sc_id
