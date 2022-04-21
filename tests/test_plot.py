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
date: 2022-04-15
author: matz
Test DASSH plotting methods (in development).
"""
########################################################################
import os
import shutil
import dassh
from .test_reactor import cleanup


def test_dasshplot_filenaming(testdir):
    """Test user-assignment of plot filenames"""
    # Setup
    inpath = os.path.join(testdir, 'test_inputs')
    datapath = os.path.join(testdir, 'test_data', 'test_plot_filenames')
    outpath = os.path.join(
        testdir, 'test_results', 'test_dasshplot_filename')
    if os.path.exists(outpath):
        cleanup(outpath)
    else:
        os.mkdir(outpath)
    r = dassh.reactor.load(os.path.join(datapath, 'dassh_reactor.pkl'))
    r.path = outpath
    shutil.copy(os.path.join(datapath, 'temp_coolant_int.csv'),
                os.path.join(outpath, 'temp_coolant_int.csv'))
    dasshplot_inp = dassh.DASSHPlot_Input(
        os.path.join(inpath, 'input_dasshplot_fname_test.txt'),
        reactor=r)
    # DASSHPlot execution
    dassh.plot.plot_all(dasshplot_inp, r)
    # Check the results: should have files in the output dir
    fname = 'subchannel_temperatures_z=300cm_z=300.0.png'
    assert os.path.exists(os.path.join(outpath, fname))
    fname = 'CoreSubchannelPlot_z=200.0.png'
    assert os.path.exists(os.path.join(outpath, fname))
