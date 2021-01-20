########################################################################
"""
date: 2020-04-24
author: matz
Cheng-Todreas correlation for flow split (1986)
"""
########################################################################
import numpy as np
from . import friction_ctd as ctd


applicability = ctd.applicability


def calculate_flow_split_old(asm_obj, regime=None):
    """Calculate the flow split into the different types of
    subchannels based on the Cheng-Todreas model

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometric description of the assembly
    regime : str or NoneType
        Indicate flow regime for which to calculate flow split
        {'turbulent', 'laminar', None}; default = None

    Returns
    -------
    numpy.ndarray
        Flow split between interior, edge, and corner coolant
        subchannels
    """
    try:
        Re_bnds = asm_obj.corr_constants['fs']['Re_bnds']
    except (KeyError, AttributeError):
        Re_bnds = ctd.calculate_Re_bounds(asm_obj)

    try:
        Cf = asm_obj.corr_constants['fs']['Cf_b']
    except(KeyError, AttributeError):
        Cf = ctd.calculate_subchannel_friction_factor_const(asm_obj)

    if regime is not None:
        return _calculate_flow_split(asm_obj, Cf, regime)
    elif asm_obj.coolant_int_params['Re'] <= Re_bnds[0]:
        try:
            return asm_obj.corr_constants['fs']['laminar']
        except:
            return _calculate_flow_split(asm_obj, Cf, 'laminar')
    elif asm_obj.coolant_int_params['Re'] >= Re_bnds[1]:
        try:
            return asm_obj.corr_constants['fs']['turbulent']
        except:
            return _calculate_flow_split(asm_obj, Cf, 'turbulent')
    else:
        return _calculate_flow_split(asm_obj, Cf, 'transition', Re_bnds)


def calculate_flow_split(asm_obj):
    """x"""
    ff_sc = ctd.calculate_subchannel_friction_factors(asm_obj)
    ff_b = ctd.calculate_bundle_friction_factor(asm_obj)
    return np.sqrt(asm_obj.corr_constants['fs']['de_ratio'] * ff_b / ff_sc)


def _calculate_flow_split(asm_obj, Cf_dict, regime, Re_bnds=None):
    """Worker function to calculate the flow split into the
    different types of subchannels based on the Cheng-Todreas
    model.

    Parameters
    ----------
    asm_obj : DASSH Assembly object
        Contains the geometric description of the assembly
    Cf_dict : dict
        Dictionary containing subchannel friction factor constants;
        keys: ['laminar', 'turbulent']
    regime : str {'laminar', 'turbulent', 'transition'}
        Flow regime with which to evaluate flow split ratios
    Re_bnds : list (optional)
        Reynolds number flow regime boundaries for calculating
        intermittency factor in transition regime

    Returns
    -------
    numpy.ndarray
        Flow split between interior, edge, and corner coolant
        subchannels

    Notes
    -----
    This method is imported by the flow split model in the
    Upgraded Cheng-Todreas correlation (flowsplit_uctd)

    """
    na = [asm_obj.subchannel.n_sc['coolant']['interior']
          * asm_obj.params['area'][0],
          asm_obj.subchannel.n_sc['coolant']['edge']
          * asm_obj.params['area'][1],
          asm_obj.subchannel.n_sc['coolant']['corner']
          * asm_obj.params['area'][2]]
    flow_split = np.zeros(3)
    if regime == 'transition':
        gamma = 1 / 3.0
        _exp2 = 1 / (2 - ctd._m['turbulent'])
        intf_b = calc_sc_intermittency_factor(
            asm_obj, Re_bnds[0], Re_bnds[1])
        fsl = asm_obj.corr_constants['fs']['laminar']
        fst = asm_obj.corr_constants['fs']['turbulent']
        fs = fsl * (1 - intf_b)**gamma + fst * intf_b**gamma
        return fs

        # beta = 0.05
        # m = ctd._m['turbulent']
        # xratio = np.zeros(3)
        # for i in range(3):
        #     fsl = asm_obj.corr_constants['fs']['laminar']
        #     fst = asm_obj.corr_constants['fs']['turbulent']
        #     fs =
        #     xratio[i] = (Cf_dict['laminar'][i]
        #                  * (asm_obj.bundle_params['de']
        #                     / asm_obj.params['de'][i]**2)
        #                  * ((1 - intf_b)**gamma
        #                     / asm_obj.coolant_int_params['Re']))
        #     xratio[i] += (beta * (Cf_dict['turbulent'][1]
        #                   * (asm_obj.bundle_params['de']
        #                      / asm_obj.params['de'][1])**m
        #                   * (1 / asm_obj.params['de'][1])
        #                   * (intf_b**gamma
        #                      / asm_obj.coolant_int_params['Re']))**_exp2)
        # x1x2 = xratio[1] / xratio[0]
        # x3x2 = xratio[1] / xratio[2]

    else:
        _exp1 = (1 + ctd._m[regime]) / (2 - ctd._m[regime])
        _exp2 = 1 / (2 - ctd._m[regime])
        # Ratio between subchannel type 1 and 2 (idx 0 and 1)
        x1x2 = ((asm_obj.params['de'][0]
                 / asm_obj.params['de'][1])**_exp1
                * (Cf_dict[regime][1] / Cf_dict[regime][0])**_exp2)
        # Ratio between subchannel type 3 and 2 (idx 2 and 1)
        x3x2 = ((asm_obj.params['de'][2]
                 / asm_obj.params['de'][1])**_exp1
                * (Cf_dict[regime][1] / Cf_dict[regime][2])**_exp2)

    # Flow split to subchannel type 2
    flow_split[1] = (asm_obj.bundle_params['area']
                     / (na[1] + x1x2 * na[0] + x3x2 * na[2]))
    flow_split[0] = x1x2 * flow_split[1]
    flow_split[2] = x3x2 * flow_split[1]
    return flow_split


def calc_constants(asm_obj):
    """Calculate constants needed by the CTD flowsplit calculation"""
    c = ctd.calc_constants(asm_obj)
    del c['Cf_b']
    c['laminar'] = calculate_flow_split_old(asm_obj, regime='laminar')
    c['turbulent'] = calculate_flow_split_old(asm_obj, regime='turbulent')
    c['de_ratio'] = asm_obj.params['de'] / asm_obj.bundle_params['de']
    return c


def calc_sc_intermittency_factor(asm_obj, Re_bl, Re_bt):
    """Calculate the bundle intermittency factor used to
    determine the transition regime friction factor

    Parameters
    ----------
    asm_obj : DASSH Assembly object
    Re_bl : float
        Laminar-transition boundary Reynolds number
    Re_bt : float
        Transition-turbulent boundary Reynolds number

    """
    fsl = asm_obj.corr_constants['fs']['laminar']
    fst = asm_obj.corr_constants['fs']['turbulent']
    de_ratio = asm_obj.params['de'] / asm_obj.bundle_params['de']
    Re_iL = Re_bl * de_ratio * fsl
    Re_iT = Re_bt * de_ratio * fst
    return((np.log10(asm_obj.coolant_int_params['Re']) - np.log10(Re_iL))
           / (np.log10(Re_iT) - np.log10(Re_iL)))
