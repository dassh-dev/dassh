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
date: 2021-04-01
author: matz
Dittus-Boelter correlation for Nusselt number, used to calculate the
heat transfer coefficient (W/m2K)
"""
########################################################################
import numpy as np
_DEFAULT_DB_CONSTS = [0.025, 0.8, 0.8, 7.0]


def calculate_bundle_Nu(coolant_obj, bundle_Re, consts=[]):
    """Calculate bundle-average Nu with the Dittus-Boelter
    correlation"""
    if consts == []:
        consts = _DEFAULT_DB_CONSTS
    Pr = _calc_prandtl(coolant_obj)
    return _dittus_boelter(bundle_Re, Pr, consts)


def calculate_sc_Nu(coolant_obj, sc_Re, consts=[]):
    """Calculate subchannel-specific Nu with the Dittus-Boelter
    correlation"""
    if consts == []:
        consts = _DEFAULT_DB_CONSTS
    Pr = _calc_prandtl(coolant_obj)
    Nu = _dittus_boelter(sc_Re, Pr, consts)
    # for i in range(len(Nu)):
    #     Nu[i] = _dittus_boelter(sc_Re[i], Pr, consts)
    return Nu


def _calc_prandtl(coolant_obj):
    """Calculate the coolant Prandtl number"""
    return (coolant_obj.heat_capacity * coolant_obj.viscosity
            / coolant_obj.thermal_conductivity)


def _dittus_boelter(Re, Pr, consts):
    """Return the Nussult number evaluated by the Dittus-Boelter
    correlation"""
    return consts[0] * (Re**consts[1] * Pr**consts[2]) + consts[3]
