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
date: 2021-06-09
author: matz
Method to check the applicability of correlations requested by the
user; raises warnings when outside the appropriate range
"""
########################################################################
from dassh.logged_class import DuplicateFilter


def check_application_range(asm, corr_module):
    """Check that the requested correlation applies to the
    problem geometry and flow regime.

    Based on:
        Table 3 in S. K. Chen et al, "Evaluation of existing
        correlations for the prediction of pressure drop in wire-wrapped
        hexagonal array pin bundles", Nuclear Engineering and Design 267
        (2014).

        ...and others

    """
    if not hasattr(corr_module, 'applicability'):
        return

    msg = ('Assembly \"{:s}\" param \"{:s}\" is outside '
           'the acceptable range for correlation \"{:s}\"')
    params = {'P/D': asm.pin_pitch / asm.pin_diameter,
              'H/D': asm.wire_pitch / asm.pin_diameter,
              'Nr': asm.n_pin,
              'bare rod': (asm.wire_diameter == 0 or asm.wire_pitch == 0)}
    param_names = {'P/D': 'pin pitch to diameter ratio',
                   'H/D': 'wire-pitch to pin-diameter ratio',
                   'Nr': 'number of rods in bundle'}
    corr_name = corr_module.__name__.split('.')[-1]
    with DuplicateFilter(asm._logger):
        # Wire wrap-related parameters are checked first - bare rod flag
        if 'bare rod' in corr_module.applicability.keys():
            if (params['bare rod'] and
                    not corr_module.applicability['bare rod']):
                asm.log('error',
                        ('Correlation \"{:s}\" not meant to apply to '
                         'bare rod assembly (\"{:s}\")'
                         .format(corr_name, asm.name)))

        # Wire wrap-related parameters are checked first - wire wrap H/D
        if not params['bare rod']:
            if not (corr_module.applicability['H/D'][0] <= params['H/D']
                    <= corr_module.applicability['H/D'][1]):
                asm.log('warning', msg.format(asm.name,
                                              param_names['H/D'],
                                              corr_name))

        # All other parameters
        for key in ['P/D', 'Nr']:
            if not (corr_module.applicability[key][0] <= params[key]
                    <= corr_module.applicability[key][1]):
                asm.log('warning', msg.format(asm.name,
                                              param_names[key],
                                              corr_name))

    # Re_b = asm.bulk_params['Re']
    # if not appdict['Re'][0] <= Re_b <= appdict['Re'][1]:
    #     warnings.warn('Assembly bulk Re falls outside the range of '
    #                   'applicability for the ' + name + ' friction '
    #                   'factor correlation.' + '\n'
    #                   + 'Range: ' + str(appdict['Re'])
    #                   + 'Assembly Re: ' + '{8.2f}'.format(Re_b))
