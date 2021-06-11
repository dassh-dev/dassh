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
date: 2021-06-10
author: matz
Test the DASSH read_input module and DASSH_input object
"""
########################################################################
import os
import pytest
import dassh


def test_bad_requested_axial_plane(testdir, caplog):
    """Test that DASSH catches bad axial planes"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir,
                         'test_inputs',
                         'x_bad_req_axial_plane.txt')
        )
    error_msg = ('Setup // axial_plane input must be of '
                 'type float (or list of float)')
    assert error_msg in caplog.text


def test_requested_axial_plane(testdir, caplog):
    """Test that DASSH correctly processes good axial planes"""
    # Now try one that passes with warnings
    inp = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_req_axial_plane.txt')
    )
    assert 'ignoring -10.0' in caplog.text
    assert 'ignoring 400.0' in caplog.text
    ans = [40.0, 75.0]
    for i in range(len(ans)):
        assert inp.data['Setup']['axial_plane'][i] == \
            pytest.approx(ans[i] / 100)


def test_missing_section(testdir, caplog):
    """Test DASSH input reader flags missing section"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(os.path.join(testdir, 'test_inputs',
                                       'x_assembly_missing.txt'))
    assert 'Missing/incorrect' in caplog.text


def test_inconsistent_tp(testdir, caplog):
    """Test DASSH input reader flags inconsistent input"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_4c_inconsistent.txt'),
            empty4c=True
        )
    assert 'Inconsistent' in caplog.text


def test_negative_val(testdir, caplog):
    """Test DASSH input reader identifies negative values"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir,
                         'test_inputs',
                         'x_assembly_negative_val.txt')
        )
    assert 'Value must be greater than zero.' in caplog.text


def test_empty_assignment(testdir, caplog):
    """Test that DASSH input reader catches empty Assignment section"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(os.path.join(testdir, 'test_inputs',
                                       'x_assignment_empty.txt'))
    assert 'Assignment section is empty' in caplog.text


def test_assembly_assignment_disagreement(testdir, caplog):
    """Test that DASSH input reader notices disagreement between
    Assembly and Assignment sections"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir,
                         'test_inputs',
                         'x_asm_assign_disagreement.txt')
        )
    print(caplog.text)
    # Check for warning
    msg = 'specified in "Assembly" input section but not assigned'
    assert msg in caplog.text
    # Check for error
    msg = 'assigned to position but not specified in "Assembly" input'
    assert msg in caplog.text


def test_assignment_bad_ring_position(testdir, caplog):
    """Test that DASSH input reader catches bad ring/position
    assignment"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir,
                         'test_inputs',
                         'x_assignment_ring_position.txt')
        )
    print(caplog.text)
    assert 'Error in ring 1 position 6 assignment: ' in caplog.text


def test_missing_4c(testdir, caplog):
    """Test DASSH input reader notices missing required 4C file input"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_4c_missing.txt')
        )
    assert all([s in caplog.text for s in ['Path ', 'does not exist']])


def test_bad_units(testdir, caplog):
    """Test that bad units raise error"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_bad_units.txt')
        )
    msg = 'Requested temperature unit "selsius" not supported'
    assert msg in caplog.text


def test_bad_gap_model(testdir, caplog):
    """Test that bad core specs raise error"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_bad_gap_model.txt')
        )
    print(caplog.text)
    # assert 'inter-assembly gap model' in caplog.text
    assert 'gap_model' in caplog.text


def test_catch_bad_unrodded_regs(testdir):
    """Test that method to catch bad axial region boundary specs"""
    bnds = [0.0, 386.2]

    # PASS: unrodded regions at top/bottom
    rr = dassh.read_input._find_rodded_regs(
        [0.0, 290.55], [128.1, 386.2], bnds)
    assert dassh.read_input._check_reg_bnds(rr)

    # PASS: multiple unrodded regions at top/bottom
    rr = dassh.read_input._find_rodded_regs(
        [0.0, 30.5, 290.55, 320.5, 350.0],
        [30.5, 128.1, 320.5, 350.0, 386.2],
        bnds)
    assert dassh.read_input._check_reg_bnds(rr)

    # PASS: only unrodded regions at top
    rr = dassh.read_input._find_rodded_regs([290.55, 320.5],
                                            [320.5, 386.2],
                                            bnds)
    assert dassh.read_input._check_reg_bnds(rr)

    # PASS: only unrodded regions at bottom
    rr = dassh.read_input._find_rodded_regs([0.0], [128.1], bnds)
    assert dassh.read_input._check_reg_bnds(rr)

    # PASS: Treat the whole thing as unrodded
    rr = dassh.read_input._find_rodded_regs([0.0], [386.2], bnds)
    assert dassh.read_input._check_reg_bnds(rr)

    # FAIL: Multiple rodded regions (2-3, 5-8)
    rr = dassh.read_input._find_rodded_regs([0, 1, 3, 4, 8, 9],
                                            [1, 2, 4, 5, 9, 10],
                                            [0, 10])
    assert not dassh.read_input._check_reg_bnds(rr)

    # FAIL: Multiple rodded regions (at top/bottom)
    rr = dassh.read_input._find_rodded_regs([1], [3], [0, 10])
    assert not dassh.read_input._check_reg_bnds(rr)
    # FAIL: Similar to above
    rr = dassh.read_input._find_rodded_regs([1, 4, 9],
                                            [3, 6, 10],
                                            [0, 10])
    assert not dassh.read_input._check_reg_bnds(rr)


def test_convection_factor(testdir, caplog):
    """Test passing and failing MFR interior fractions for UR regions"""
    # Passing case
    inp = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_ur_conv_factor.txt'),
        empty4c=True)
    assert inp.data['Assembly']['fuel']['convection_factor'] == "calculate"
    assert inp.data['Assembly']['fuel2']['convection_factor'] == 0.001

    ar = inp.data['Assembly']['control']['AxialRegion']
    assert ar['empty_cr']['convection_factor'] == 1.0  # None default
    assert ar['upper_cr']['convection_factor'] == 0.8

    # Failing case 1: unknown entry
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_ur_conv_factor1.txt'),
            empty4c=True)
    msg = 'do not understand "convection_factor" input - '
    ans = [x for x in caplog.text.split('\n') if x not in ('\n', '')][-1]
    assert msg in ans

    # Failing case 2: improper use of "calculate"
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_ur_conv_factor2.txt'),
            empty4c=True)
    msg = ('"convection_factor" key in section "Assembly, control, '
           'AxialRegion, upper_cr" failed validation; check that '
           'it meets the requirements')
    ans = [x for x in caplog.text.split('\n') if x not in ('\n', '')][-1]
    assert msg in ans

    # Failing case 3: Float less than or equal to zero
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_ur_conv_factor3.txt'),
            empty4c=True)
    msg = ('"convection_factor" key in section "Assembly, control, '
           'AxialRegion, upper_cr" failed validation; check that '
           'it meets the requirements')
    ans = [x for x in caplog.text.split('\n') if x not in ('\n', '')][-1]
    assert msg in ans


def test_reactor_inactive_asm_too_many(testdir, caplog):
    """Test that Reactor instantiation fails if user tries to specify
    too many assemblies"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_input_17a_silly.txt')
        )
    assert 'More assembly positions specified in input ' in caplog.text


def test_reactor_inactive_asm_position(testdir, caplog):
    """Test that Reactor instantiation fails if user tries to put an
    assembly in an inactive position"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_input_16a_inactive.txt')
        )
    assert 'assignment does not match region assignment' in caplog.text
    assert 'Assembly: 20; Loc: ((4, 1))' in caplog.text

    # Should have no failure with the following, which uses the same
    # GEODST but appropriately assigns assemblies
    dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_16a_silly.txt')
    )


# def test_regionlist_labels_agreement(testdir, caplog):
#     """Test DASSH input flags regions not found in LABELS binary file"""
#     with pytest.raises(SystemExit):
#         dassh.DASSH_Input(os.path.join(testdir, 'test_inputs',
#                                        'x_regionlist_bad_region.txt'))
#     # assert 'Value must be greater than zero.' in caplog.text
#     print(caplog.txt)

# def test_assignment_regions(testdir):
#     """Test DASSH input flags assignment labels not found in LABELS
#     file or in RegionList input"""
#     with pytest.raises(ValueError):
#         dassh.DASSH_Input(os.path.join(testdir, 'test_inputs',
#                                        'x_assignment_bad_region.txt'))
#
#
# # def test_assignment_warnings(testdir):
# #     """Test warnings raised when user over-specifies assignments"""
# #     with pytest.warns(UserWarning):
# #         dassh.DASSH_Input(os.path.join(testdir, 'test_inputs',
# #                                        'x_assignment_warnings1.txt'))
# #     with pytest.warns(UserWarning):
# #         dassh.DASSH_Input(os.path.join(testdir, 'test_inputs',
# #                                        'x_assignment_warnings2.txt'))


def test_single_tp_oxide(testdir):
    """Test handling of input with oxide fuel"""
    inp = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_oxide_fuel.txt'),
        empty4c=True
    )
    print(inp.data['Assignment']['ByPosition'])
    assert hasattr(inp, 'data')
    assert all([x in inp.data.keys() for x in ['Power', 'Core', 'Assembly']])


def test_undefined_material(testdir, caplog):
    """Test handling of input with undefined material"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_undefined_mat.txt'),
            empty4c=True
        )
    assert 'Cannot find properties for material argonne' in caplog.text


def test_nonfloat_matspec(testdir, caplog):
    """Test handling of input with undefined material"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_nonfloat_matspec.txt'),
            empty4c=True
        )
    assert 'must be list of floats' in caplog.text
    assert 'viscosity' in caplog.text


def test_single_tp(testdir):
    """Test handling of correct single time point input"""
    inp = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_single_tp.txt'),
        empty4c=True
    )
    print(inp.data['Assignment']['ByPosition'])
    assert inp.timepoints == 1
    assert hasattr(inp, 'data')
    assert all([x in inp.data.keys() for x in ['Power', 'Core', 'Assembly']])


def test_mult_tp(testdir):
    """Test handling of correct multi- time point input"""
    inp = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_multiple_tp.txt'),
        empty4c=True
    )
    assert inp.timepoints > 1
    assert hasattr(inp, 'data')
    assert all([x in inp.data.keys() for x in ['Power', 'Core', 'Assembly']])


def test_input_custom_material(testdir, caplog):
    """Test that custom material correlations can be read"""
    dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_custom_mat.txt'),
        empty4c=True
    )


def test_bad_core_len(testdir, caplog):
    """Test handling of unrodded axial regions"""
    # This one fails because the linked GEODST does not match
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_bad_core_len.txt')
        )
    assert "Maximum z-mesh in GEODST file" in caplog.text


def test_bad_asm_pitch(testdir, caplog):
    """Test handling of unrodded axial regions"""
    # This one fails because the linked GEODST does not match
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir, 'test_inputs', 'x_bad_asm_pitch.txt')
        )
    assert "Assembly pitch in GEODST file" in caplog.text


def test_proper_axial_reg(testdir):
    """This one passes because it matches the GEODST file"""
    inp = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_single_asm.txt')
    )
    assert inp.data['Assembly']['fuel']['AxialRegion']['rods']['z_lo'] == \
        pytest.approx(1.2700)
    assert inp.data['Assembly']['fuel']['AxialRegion']['rods']['z_hi'] == \
        pytest.approx(2.9210)


def test_single_axial_reg(testdir):
    """Test that I can use just one simple axial region below the rod
    bundle (rather than two, one above and one below)"""
    inp = dassh.DASSH_Input(
        os.path.join(testdir, 'test_inputs', 'input_one_axial_reg.txt')
    )
    assert len(inp.data['Assembly']['driver']['AxialRegion']) == 2


def test_axial_reg_bounds_matching():
    """Test the methods that find the roddeds region bounds for
    some different cases"""

    # Case 1.1: one unrodded region below the rod bundle
    z0 = [0.0]
    z1 = [128.1]
    bnds = [0.0, 386.2]
    rregs = dassh.read_input._find_rodded_regs(z0, z1, bnds)
    rbnds = dassh.read_input._get_rodded_reg_bnds(z0, z1, rregs, bnds)
    assert rbnds == [128.1, 386.2]

    # Case 1.2: multiple unrodded region below the rod bundle
    z0 = [0.0, 10.0, 40.5]
    z1 = [10.0, 40.5, 128.1]
    bnds = [0.0, 386.2]
    rregs = dassh.read_input._find_rodded_regs(z0, z1, bnds)
    rbnds = dassh.read_input._get_rodded_reg_bnds(z0, z1, rregs, bnds)
    assert rbnds == [128.1, 386.2]

    # Case 2.1: one unrodded region above rod bundle
    z0 = [290.95]
    z1 = [386.2]
    bnds = [0.0, 386.2]
    rregs = dassh.read_input._find_rodded_regs(z0, z1, bnds)
    rbnds = dassh.read_input._get_rodded_reg_bnds(z0, z1, rregs, bnds)
    assert rbnds == [0.0, 290.95]

    # Case 2.2: one unrodded region above rod bundle
    z0 = [290.95, 330.0]
    z1 = [330.0, 386.2]
    bnds = [0.0, 386.2]
    rregs = dassh.read_input._find_rodded_regs(z0, z1, bnds)
    rbnds = dassh.read_input._get_rodded_reg_bnds(z0, z1, rregs, bnds)
    assert rbnds == [0.0, 290.95]

    # Case 3.1: unrodded regions above/below rod bundle
    z0 = [0.0, 290.95]
    z1 = [128.1, 386.2]
    bnds = [0.0, 386.2]
    rregs = dassh.read_input._find_rodded_regs(z0, z1, bnds)
    rbnds = dassh.read_input._get_rodded_reg_bnds(z0, z1, rregs, bnds)
    assert rbnds == [128.1, 290.95]

    # Case 3.2: unrodded regions above/below rod bundle; multiple below
    z0 = [0.0, 100.0, 290.95]
    z1 = [100.0, 128.1, 386.2]
    bnds = [0.0, 386.2]
    rregs = dassh.read_input._find_rodded_regs(z0, z1, bnds)
    rbnds = dassh.read_input._get_rodded_reg_bnds(z0, z1, rregs, bnds)
    assert rbnds == [128.1, 290.95]

    # Case 3.3: unrodded regions above/below rod bundle; multiple above
    z0 = [0.0, 290.95, 320]
    z1 = [128.1, 320, 386.2]
    bnds = [0.0, 386.2]
    rregs = dassh.read_input._find_rodded_regs(z0, z1, bnds)
    rbnds = dassh.read_input._get_rodded_reg_bnds(z0, z1, rregs, bnds)
    assert rbnds == [128.1, 290.95]

    # Case 3.4: multiple unrodded regions above/below rod bundle
    z0 = [0.0, 100.0, 290.95, 320]
    z1 = [100.0, 128.1, 320, 386.2]
    bnds = [0.0, 386.2]
    rregs = dassh.read_input._find_rodded_regs(z0, z1, bnds)
    rbnds = dassh.read_input._get_rodded_reg_bnds(z0, z1, rregs, bnds)
    assert rbnds == [128.1, 290.95]


def test_assignment_missing_assemblies(testdir):
    """Test that DASSH can handle missing assemblies in the
    Assignment input"""
    # If I don't fail, I pass
    p = os.path.join(testdir, 'test_inputs', 'input_no_corner_asm.txt')
    inp = dassh.DASSH_Input(p, empty4c=True)
    # Should be 37 assembly positions, 6 empty corners
    assert len(inp.data['Assignment']['ByPosition']) == 37
    assert len([x for x in inp.data['Assignment']['ByPosition']
                if len(x) > 0]) == 31


#
# def test_unit_conversion(testdir):
#     """Test some unit conversions"""
#     inp = dassh.DASSH_Input(os.path.join(testdir, 'test_inputs',
#                                          'input_unit_convs.txt'))
#     inp_conv = copy.deepcopy(inp)
#     inp_conv = dassh.dassh_setup.convert_units(inp_conv)
#
#     # Check temperatures
#     assert (inp_conv.data['Core']['coolant_inlet_temp'] ==
#             pytest.approx((inp.data['Core']['coolant_inlet_temp'] - 32)
#                           * 5 / 9 + 273.15))
#     assert (inp_conv.data['Assignment']['ByPosition'][0][3] ==
#             pytest.approx((inp.data['Assignment']['ByPosition'][0][3] - 32)
#                           * 5 / 9 + 273.15))
#
#     # Check lengths
#     assert (inp_conv.data['Core']['height'] ==
#             pytest.approx(inp.data['Core']['height'] * 12 * 2.54 / 100))
#     assert all([(inp_conv.data['Assembly']['fuel']['duct_ftf'][i] ==
#                  pytest.approx((inp.data['Assembly']['fuel']
#                                         ['duct_ftf'][i]
#                                 * 12 * 2.54 / 100)))
#                 for i in range(len(inp_conv.data['Assembly']['fuel']
#                                                 ['duct_ftf']))])
#     for p in ['pin_pitch', 'pin_diameter', 'clad_thickness',
#               'wire_pitch', 'wire_diameter']:
#         assert (inp_conv.data['Assembly']['fuel'][p] ==
#                 pytest.approx(inp.data['Assembly']['fuel'][p]
#                               * 12 * 2.54 / 100.))
#     # Check mass flow rate
#     assert (inp_conv.data['Assignment']['ByPosition'][1][3] ==
#             pytest.approx(inp.data['Assignment']['ByPosition'][1][3]
#                           * 0.000125998, 1e-4))
#
