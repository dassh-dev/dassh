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
date: 2021-11-02
author: matz
Test DASSH material class
"""
########################################################################
import os
import pytest
import numpy as np
from dassh import Material


def test_material_from_coeff():
    """Try loading material properties by coefficient dictionaries"""
    assert hasattr(Material('d9'), 'thermal_conductivity')
    assert hasattr(Material('ht9'), 'thermal_conductivity')

    # Define a custom dictionary and use it
    c = {'thermal_conductivity': [0.05, 1.0, 2.0],
         'heat_capacity': [480.0, 0.5],
         'density': [1200.0, 0.2, 0.03],
         'viscosity': [1.0, 1.0, 1.0],
         'beta': [0.002]
         }
    m = Material('test_material', coeff_dict=c)
    for prop in ['thermal_conductivity', 'heat_capacity',
                 'density', 'viscosity']:
        assert hasattr(m, prop)
        # Try getting a value from the correlation; should be
        # a float, and should be greater than 0
        assert getattr(m, prop) > 0.0

    # Check the results for one of the values
    m.update(100.0)
    assert m.heat_capacity == pytest.approx(480 + 50.0)
    assert type(m.beta) == float
    assert m.beta == 0.002


def test_material_from_table():
    """Try loading material properties stored exclusively in CSV
    tables in the data/ directory"""
    coolants = ['water', 'sodium', 'potassium', 'nak',
                'lead', 'bismuth', 'lbe']
    temps = [300.0, 500.0, 500.0, 500.0, 800.0, 800.0, 800.0]
    for i in range(len(coolants)):
        mat = Material(coolants[i], temps[i])
        for prop in ['thermal_conductivity', 'heat_capacity',
                     'density', 'viscosity']:
            assert hasattr(mat, prop)
            # Try getting a value from the correlation; should be
            # a float, and should be greater than 0
            assert getattr(mat, prop) > 0.0


def test_error_table_negative_val(testdir, caplog):
    """Test error when table has negative value"""
    f = os.path.join(testdir, 'test_inputs', 'custom_mat-3.csv')
    # with pytest.raises(SystemExit):
    #     Material('badbad', from_file=f)
    Material('badbad', from_file=f)
    assert 'Negative values detected in material data ' in caplog.text


def test_material_coeff_from_file(testdir):
    """Try loading material property correlation coeffs from CSV"""
    filepath = os.path.join(testdir, 'test_inputs', 'custom_mat.csv')
    m = Material('sodium', from_file=filepath)
    m.update(623.15)
    for prop in ['thermal_conductivity', 'heat_capacity',
                 'density', 'viscosity']:
        assert hasattr(m, prop)
        assert getattr(m, prop) > 0.0


def test_failed_material(caplog):
    """Make sure that the Material class fails with bad input"""
    with pytest.raises(SystemExit):
        Material('candycorn')
    assert 'material candycorn' in caplog.text


def test_bad_temperature(caplog):
    """Make sure Material throws error for negative temperatures"""
    m = Material('sodium')
    with pytest.raises(SystemExit):
        m.update(0.0)
    assert 'must be > 0; given' in caplog.text


def test_bad_property(caplog):
    """Make sure Material throws error for negative temperatures"""
    # Define a custom dictionary and use it
    c = {'thermal_conductivity': [0.05, 1.0, 2.0],
         'heat_capacity': [480.0, 0.5],
         'density': [1200.0, 0.2, 0.03],
         'viscosity': [300.0, -1.0],
         }
    m = Material('test', coeff_dict=c)
    with pytest.raises(SystemExit):
        m.update(400.0)
    assert 'viscosity must be > 0; given' in caplog.text


def test_sodium_interpolated_value():
    """Test that DASSH properly interpolates sodium properties"""
    rho_400 = 919.0
    rho_500 = 897.0
    ans = np.average([rho_400, rho_500])
    sodium = Material('sodium', temperature=450.0)
    print('ans =', ans)
    print('res =', sodium.density)
    err = (sodium.density - ans) / ans
    print('err = ', err)
    assert ans == pytest.approx(sodium.density)


def test_table_with_missing_values(testdir):
    """Test that DASSH interpolates properties
    with missing or zero values"""
    filepath = os.path.join(testdir, 'test_inputs', 'custom_mat-2.csv')
    m = Material('sodium', from_file=filepath)
    # missing values in heat capacity
    m.update(950.0)
    assert m.heat_capacity == pytest.approx(1252.0)
    # zero values in density; missing values in viscosity
    # linear interp should return average
    m.update(850.0)
    assert m.density == pytest.approx(np.average([828, 805]))
    assert m.viscosity == pytest.approx(np.average([0.000227, 0.000201]))
