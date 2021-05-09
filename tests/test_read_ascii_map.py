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
date: 2020-12-09
author: matz
comment: Test methods thatread ASCII subchannel temp maps from SE2ANL
"""
########################################################################
import pytest
import numpy as np
from dassh._se2anl import read_ascii_map


pytestmark = pytest.mark.skip(reason='none of these will work')


def test_2ring_map():
    """Test that I can read a 2 ring map"""
    ring2 = """
          018   007   008

                001
       017   006   002   009
       016               010
       015   005   003   011
                004

          014   013   012
    """
    # After processing the raw string, should be like below
    # ring2 = [['016', '017', '018'],
    #          ['006'],
    #          ['015', '005', '001', '007'],
    #          ['014', '008'],
    #          ['013', '004', '002', '009'],
    #          ['003'],
    #          ['012', '011', '010']]
    # poop = read_ascii_map.process_raw_str(ring2)
    # n_ring = len(poop[0]) - 1
    # rtemps, stripped_map = read_ascii_map.strip_edge2(poop, n_ring)
    # print(rtemps)
    # while True:
    #     try:
    #         x, stripped_map = read_ascii_map.strip_ring2(stripped_map)
    #     except IndexError:
    #         break
    #     rtemps += x
    #     print(rtemps)
    temps = read_ascii_map.read(ring2)
    print(temps)
    assert np.array_equal(temps, np.arange(1, len(temps) + 1, 1))


def test_3ring_map():
    """Test that I can read a 3 ring map"""
    ring3 = """
             042   025   026   027


             041   007   009   028
                024   008   010

          040   023   001   011   029
             022   006   002   012
       039                          030
             021   005   003   013
          038   020   004   014   031

                019   017   015
             037   018   016   032

             036   035   034   033
    """
    # After processing the raw string, should be like below
    # ring3 = [['039', '040', '041', '042'],
    #          ['038', '022', '024', '025'],
    #          ['021', '023', '007'],
    #          ['037', '020', '006', '008', '026'],
    #          ['019', '005', '001', '009'],
    #          ['036', '027'],
    #          ['018', '004', '002', '010'],
    #          ['035', '017', '003', '011', '028'],
    #          ['016', '014', '012'],
    #          ['034', '015', '013', '029'],
    #          ['033', '032', '031', '030']]
    temps = read_ascii_map.read(ring3)
    assert np.array_equal(temps, np.arange(1, len(temps) + 1, 1))


def test_6ring():
    """x"""
    ring6 = """
              186   151   152   153   154   155   156

              185   097   099   101   103   105   157
                 150   098   100   102   104   106

           184   149   055   057   059   061   107   158
              148   096   056   058   060   062   108

        183   147   095   025   027   029   063   109   159
           146   094   054   026   028   030   064   110

     182   145   093   053   007   009   031   065   111   160
        144   092   052   024   008   010   032   066   112

  181   143   091   051   023   001   011   033   067   113   161
     142   090   050   022   006   002   012   034   068   114
180                                                               162
     141   089   049   021   005   003   013   035   069   115
  179   140   088   048   020   004   014   036   070   116   163

        139   087   047   019   017   015   037   071   117
     178   138   086   046   018   016   038   072   118   164

           137   085   045   043   041   039   073   119
        177   136   084   044   042   040   074   120   165

              135   083   081   079   077   075   121
           176   134   082   080   078   076   122   166

                 133   131   129   127   125   123
              175   132   130   128   126   124   167

              174   173   172   171   170   169   168
        """
    temps = read_ascii_map.read(ring6)
    assert np.array_equal(temps, np.arange(1, len(temps) + 1, 1))


@pytest.mark.skip(reason='not passing yet, method still needs work')
def test_5ring_ducted():
    """Test a ducted assembly"""
    ring5d = """
                       999        999   999   999   999        999

                         999      999   999   999   999      999
                            *   *   *   *   *   *   *   *   *
               999  999  *  121   122   123   124   125   126  *  999  999

                       *          090   092   094   096          *
            999  999  *  120   089   091   093   095   055   097  *  999  999

                    *          088   050   052   054   056          *
         999  999  *  119   087   049   051   053   025   057   098  *  999  999

                 *          086   048   022   024   026   058          *
      999  999  *  118   085   047   021   023   007   027   059   099  *  999  999

              *          084   046   020   006   008   028   060          *
   999  999  *  117   083   045   019   005   001   009   029   061   100  *  999  999
   999  999  *  116                         *                         101  *  999  999
   999  999  *  115   082   044   018   004   002   010   030   062   102  *  999  999
              *          081   043   017   003   011   031   063          *

      999  999  *  114   080   042   016   014   012   032   064   103  *  999  999
                 *          079   041   015   013   033   065          *

         999  999  *  113   078   040   038   036   034   066   104  *  999  999
                    *          077   039   037   035   067          *

            999  999  *  112   076   074   072   070   068   105  *  999  999
                       *          075   073   071   069          *

               999  999  *  111   110   109   108   107   106  *  999  999
                            *   *   *   *   *   *   *   *   *
                         999      999   999   999   999      999

                     999          999   999   999   999          999
        """
    temps = read_ascii_map.read(ring5d)
    assert np.array_equal(temps, np.arange(1, len(temps) + 1, 1))
