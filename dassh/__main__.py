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
date: 2022-01-20
author: matz
Main DASSH calculation procedure
"""
########################################################################
import os
import sys
import dassh
import argparse
import cProfile
import logging
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
    dassh_logger.info("DASSH logger initialized")

    # Pre-processing
    # Read input file and set up DASSH input object
    dassh_logger.log(_log_info, f'Reading input: {args.inputfile}')
    dassh_input = dassh.DASSH_Input(args.inputfile)

    # CHECK FOR PYTHON VERSION WARNINGS/ERRORS
    # check_version(dassh_input, dassh_logger, args.save_reactor)

    # DASSH calculation without orificing optimization
    if dassh_input.data['Orificing'] is False:
        arg_dict = {
            'save_reactor': args.save_reactor,
            'verbose': args.verbose,
            'no_power_calc': args.no_power_calc
        }
        run_dassh(dassh_input, arg_dict)

    # Orificing optimization with DASSH
    else:
        orifice_obj = dassh.orificing.Orificing(dassh_input)
        orifice_obj.optimize()

    # Finish the calculation
    dassh_logger.log(_log_info, 'DASSH execution complete')
    # Print/dump profiler results
    if args.profile:
        pr.disable()
        pr.dump_stats('dassh_profile.out')

    # Shutdown logger by removing file handlers
    dassh.logged_class.shutdown_logger('dassh')


def check_version(dassh_inp, save_reactor):
    """Check for DASSH limitations depending on Python version;
    tentatively deprecated."""
    dassh_logger = logging.getLogger('dassh')
    version = '.'.join([str(sys.version_info.major),
                        str(sys.version_info.minor),
                        str(sys.version_info.micro)])
    if dassh_inp.data['Plot'] and sys.version_info < (3, 7):
        dassh_logger.log(
            30,
            'WARNING: DASSH plotting capability requires '
            f'Python 3.7+; detected {version}')
    if save_reactor and sys.version_info < (3, 7):
        dassh_logger.log(
            30,
            'WARNING: --save_reactor capability requires '
            f'Python 3.7+; detected {version}')
    if dassh_logger.data['Orificing'] and sys.version_info < (3, 7):
        dassh_logger.log(
            40,
            'ERROR: DASSH orificing optimization requires '
            f'Python 3.7+; detected {version}')
        sys.exit(1)
    else:
        pass


def run_dassh(dassh_input, rx_args):
    """Run DASSH without orificing optimization"""
    # For each timestep in the DASSH input, create the necessary DASSH
    # DASSH objects, run DASSH, and process the results
    dassh_logger = logging.getLogger('dassh')
    need_subdir = False
    if dassh_input.timepoints > 1:
        need_subdir = True
        if dassh_input.data['Setup']['parallel']:
            import multiprocessing as mp
            if dassh_input.data['Setup']['n_cpu'] is not None:
                n_procs = dassh_input.data['Setup']['n_cpu']
            else:
                n_procs = min((mp.cpu_count(), dassh_input.timepoints))
            pool = mp.Pool(processes=n_procs)
            workers = []

    for i in range(dassh_input.timepoints):
        working_dir = None
        if need_subdir:
            # Only log info about timestep if you have multiple
            dassh_logger.log(_log_info, f'Timestep {i + 1}')
            working_dir = os.path.join(
                dassh_input.path, f'timestep_{i + 1}')
        # Set up working dirs, run DASSH, write output, make plots
        if dassh_input.data['Setup']['parallel']:
            workers.append(
                pool.apply_async(
                    _run_dassh,
                    args=(dassh_input,
                          rx_args,
                          i,
                          working_dir, )
                )
            )
        else:
            _run_dassh(dassh_input, rx_args, i, working_dir)

    # Clean up from parallel execution, if applicable
    if dassh_input.data['Setup']['parallel']:
        for w in workers:
            w.get()
        pool.terminate()
        pool.close()
        pool.join()


def _run_dassh(dassh_inp, args, timestep, wdir, link=None):
    """Run DASSH for a single timestep

    Parameters
    ----------
    dassh_inp : DASSH_Input object
        Base DASSH input class
    args : dict
        Various args for instantiating DASSH objects
    timestep : int
        Timestep for which to run DASSH
    wdir : str
        Path to working directory for this timestep
    link : str (optional)
        Try to link VARPOW output files from another path
        Avoids repetitive calcs in orificing optimization
        (default = None; run VARPOW as usual)

    """
    dassh_logger = logging.getLogger('dassh')
    # Try to link VARPOW output from another source. If it doesn't
    # exist or work, just rerun VARPOW.
    if link is not None:
        files_linked = 0
        for f in ['varpow_MatPower.out',
                  'varpow_MonoExp.out',
                  'VARPOW.out']:
            src = os.path.join(link, f)
            dest = os.path.join(wdir, f)
            if os.path.exists(src):
                os.symlink(src, dest)
                files_linked += 1
            else:
                break
        # If all VARPOW files were linked, can skip VARPOW calculation
        if files_linked == 3:
            args['no_power_calc'] = False  # if linked, skip VARPOW
        else:
            args['no_power_calc'] = True

    # Initialize the Reactor object
    reactor = dassh.Reactor(dassh_inp,
                            calc_power=args['no_power_calc'],
                            path=wdir,
                            timestep=timestep,
                            write_output=True)
    # Perform the sweep
    dassh_logger.log(_log_info, 'Performing temperature sweep...')
    reactor.temperature_sweep(verbose=args['verbose'])
    reactor.postprocess()

    # Post-processing: write output, save reactor if desired
    dassh_logger.log(_log_info, 'Temperature sweep complete')
    if args['save_reactor'] or dassh_inp.data['Plot']:
        if sys.version_info < (3, 7):
            handlers = dassh_logger.handlers[:]
            for handler in handlers:
                handler.close()
                dassh_logger.removeHandler(handler)
        reactor.save()
        if sys.version_info < (3, 7):
            dassh_logger = dassh.logged_class.init_root_logger(
                os.path.split(dassh_logger._root_logfile_path)[0],
                'dassh', 'a+')
    dassh_logger.log(_log_info, 'Output written')

    # Post-processing: generate figures, if desired
    if ('Plot' in dassh_inp.data.keys()
            and len(dassh_inp.data['Plot']) > 0):
        dassh_logger.log(_log_info, 'Generating figures')
        dassh.plot.plot_all(dassh_inp, reactor)
    return dassh_logger


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
