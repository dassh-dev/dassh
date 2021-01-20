########################################################################
"""
date: 2020-10-09
author: matz
Common utility functions for the Cheng-Todreas correlations
"""
########################################################################
import numpy as np


def calc_sc_intermittency_factors(asm, Re_sc, Re_bL, Re_bT, fs_iL, fs_iT):
    """Calculate the intermittency factors for the interior and edge
    coolant subchannels; required to find the mixing parameters in
    the transition region

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometry and flow parameters
    Re_bL : float
        Reynolds number boundary between laminar-transition regimes
    Re_bT : float
        Reynolds number boundary between transition-turbulent regimes

    Notes
    -----
    See Equations 10, 11, and 30-32 in the Cheng-Todreas 1986 paper

    """
    de_ratio = asm.params['de'] / asm.bundle_params['de']
    Re_iL = Re_bL * fs_iL * de_ratio
    Re_iT = Re_bT * fs_iT * de_ratio
    return ((np.log10(Re_sc) - np.log10(Re_iL))
            / (np.log10(Re_iT) - np.log10(Re_iL)))
