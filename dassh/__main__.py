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
date: 2021-10-18
author: matz
Main DASSH calculation procedure
"""
########################################################################
import os
import sys
import dassh
import argparse
import cProfile
_log_info = 20  # logging levels must be int


def main(args=None):
    """Perform temperature sweep in DASSH"""
    # Parse command line arguments to DASSH
    parser = argparse.ArgumentParser(description='Process DASSH cmd')
    parser.add_argument('inputfile',
                        metavar='inputfile',
                        help='The input file to run with DASSH')
    parser.add_argument('--verbose',
                        action='store_true',
                        help='Verbose; print summary with each axial step')
    parser.add_argument('--save_reactor',
                        action='store_true',
                        help='Save DASSH Reactor object after sweep')
    parser.add_argument('--profile',
                        action='store_true',
                        help='Profile the execution of DASSH')
    parser.add_argument('--no_power_calc',
                        action='store_false',
                        help='Skip VARPOW calculation if done previously')
    args = parser.parse_args(args)

    # Enable the profiler, if desired
    if args.profile:
        pr = cProfile.Profile()
        pr.enable()

    # Initiate logger
    print(dassh._ascii._ascii_title)
    in_path = os.path.split(args.inputfile)[0]
    dassh_logger = dassh.logged_class.init_root_logger(in_path, 'dassh')

    # Pre-processing
    # Read input file and set up DASSH input object
    dassh_logger.log(_log_info, f'Reading input: {args.inputfile}')
    dassh_input = dassh.DASSH_Input(args.inputfile)

    # DASSH calculation without orificing optimization
    if dassh_input.data['Orificing'] is False:
        arg_dict = {'save_reactor': args.save_reactor,
                    'verbose': args.verbose,
                    'no_power_calc': args.no_power_calc}
        run_dassh(dassh_input, dassh_logger, arg_dict)

    # Orificing optimization with DASSH
    else:
        dassh.orificing.optimize(dassh_input, dassh_logger)
        # raise NotImplementedError('haha')

    # Finish the calculation
    dassh_logger.log(_log_info, 'DASSH execution complete')
    # Print/dump profiler results
    if args.profile:
        pr.disable()
        pr.dump_stats('dassh_profile.out')


def run_dassh(dassh_input, dassh_logger, args):
    """Run DASSH without orificing optimization"""
    # For each timestep in the DASSH input, create the necessary DASSH
    # DASSH objects, run DASSH, and process the results
    need_subdir = False
    if dassh_input.timepoints > 1:
        need_subdir = True
    for i in range(dassh_input.timepoints):
        working_dir = None
        if need_subdir:
            working_dir = os.path.join(dassh_input.path,
                                       f'timestep_{i + 1}')
        print('\n')
        dassh_logger.log(_log_info, f'Timestep {i + 1}')
        reactor = dassh.Reactor(dassh_input,
                                calc_power=args['no_power_calc'],
                                path=working_dir,
                                timestep=i,
                                write_output=True)
        # Perform the sweep
        dassh_logger.log(_log_info, 'Performing temperature sweep...')
        reactor.temperature_sweep(verbose=args['verbose'])

        # Post-processing: write output, save reactor if desired
        dassh_logger.log(_log_info, 'Temperature sweep complete')
        if args['save_reactor'] and sys.version_info >= (3, 7):
            reactor.save()
        elif dassh_input.data['Plot']:
            reactor.save()  # just in case figure generation fails
        else:
            pass
        dassh_logger.log(_log_info, 'Output written')

        # Post-processing: generate figures, if desired
        if ('Plot' in dassh_input.data.keys()
                and len(dassh_input.data['Plot']) > 0):
            dassh_logger.log(_log_info, 'Generating figures')
            dassh.plot.plot_all(dassh_input, reactor)


def plot():
    """Command-line interface to postprocess DASSH data to make
    matplotlib figures"""
    # Get input file from command line arguments
    parser = argparse.ArgumentParser(description='Process DASSH cmd')
    parser.add_argument('inputfile',
                        metavar='inputfile',
                        help='The input file to run with DASSH')
    args = parser.parse_args()

    # Initiate logger
    print(dassh._ascii._ascii_title)
    in_path = os.path.split(args.inputfile)[0]
    dassh_logger = dassh.logged_class.init_root_logger(in_path,
                                                       'dassh_plot')

    # Check whether Reactor object exists; if so, process with
    # DASSHPlot_Input and get remaining info from Reactor object
    rpath = os.path.join(os.path.abspath(in_path), 'dassh_reactor.pkl')
    if os.path.exists(rpath):
        dassh_logger.log(_log_info, f'Loading DASSH Reactor: {rpath}')
        r = dassh.reactor.load(rpath)
        dassh_logger.log(_log_info, f'Reading input: {args.inputfile}')
        inp = dassh.DASSHPlot_Input(args.inputfile, r)

    # Otherwise, build Reactor object from complete DASSH input
    else:
        dassh_logger.log(_log_info, f'Reading input: {args.inputfile}')
        inp = dassh.DASSH_Input(args.inputfile)
        dassh_logger.log(_log_info, 'Building DASSH Reactor from input')
        r = dassh.Reactor(inp, calc_power=False)

    # Generate figures
    dassh_logger.log(_log_info, 'Generating figures')
    dassh.plot.plot_all(inp, r)
    dassh_logger.log(_log_info, 'DASSH_PLOT execution complete')


if __name__ == '__main__':
    main()
