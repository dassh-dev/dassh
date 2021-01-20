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

DASSH material data README
Milos Atz
Last updated: 2020-08-16

Notes
-----
All temperatures are in KELVIN

Update 2020-08-17
-----------------
Fixed incorrect heat capacity for water: had used isochoric specific
heat (Cv) instead of isobaric specific heat (Cp).

Update 2020-08-16
-----------------
Fixed incorrect thermal conductivity data in table for water: was
input in mW/mK rather than W/mK (from https://www.engineeringtoolbox.com/water-liquid-gas-thermal-conductivity-temperature-pressure-d_2012.html)

Update 2020-07-28
-----------------
Tabulated data for lead, bismuth, and lead-bismuth eutectic (LBE)
is obtained by calculating values using the coefficients from [1].


References
----------
[1] https://inis.iaea.org/collection/NCLCollectionStore/_Public/43/095/43095088.pdf