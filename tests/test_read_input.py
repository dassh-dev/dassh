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
date: 2022-11-01
author: matz
Test the DASSH read_input module and DASSH_input object
"""
########################################################################
import os
import pytest
import dassh


def test_bad_fuel_mat(testdir, caplog):
    """Test that DASSH catches errors in ARC fuel material specification"""
    # Test 1: "Bad" fuel specification
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir,
                         'test_inputs',
                         'x_bad_fuel_mat.txt')
        )
    expected_error_msg = ('"fuel_material" key in section "Power, ARC" failed '
                          'validation; check that it meets the requirements')
    assert expected_error_msg in caplog.text
    # Test 2: No input to fuel material
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(testdir,
                         'test_inputs',
                         'x_no_fuel_mat.txt')
        )
    expected_error_msg = ('\"fuel_material\" input must be one of '
                          '{"metal", "oxide", "nitride"}')
    assert expected_error_msg in caplog.text


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


def test_ignore_parallel_ncpu(testdir, caplog):
    """Test that DASSH ignores request for parellelism if n_cpu=1"""
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'x_input_parallel-1.txt'))
    msg = 'Parallel execution requested but "n_cpu"=1; ignoring...'
    assert msg in caplog.text
    assert inp.data['Setup']['parallel'] is False


def test_ignore_parallel_one_timestep(testdir, caplog):
    """Test that parallelism request ignored if only one timestep"""
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'x_input_parallel-2.txt'))
    msg = 'No parallelism for single timestep; ignoring...'
    assert msg in caplog.text
    assert inp.data['Setup']['parallel'] is False


def test_okay_parallel_input(testdir):
    """Test that appropriate user input enables parallel calc"""
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'input_parallel.txt'))
    assert inp.data['Setup']['parallel'] is True


def test_bad_orificing_input_missing_key(testdir, caplog):
    """Test that input fails if missing orificing keys"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_input_orificing-1.txt'))
    msg = 'Orificing input "bulk_coolant_temp" must be specified'
    assert msg in caplog.text


def test_bad_orificing_input_wrong_asm_name(testdir, caplog):
    """Test that input fails if orificing asm type not in input"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_input_orificing-2.txt'))
    msg = ('Orificing input "assemblies_to_group" must contain assembly '
           'types specified in "Assembly"; do not recognize "fuelx"')
    assert msg in caplog.text


def test_bad_orificing_input_negative_dt(testdir, caplog):
    """Fail if target coolant outlet temp less than core inlet temp"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_input_orificing-3.txt'))
    msg = ('Orificing input "bulk_coolant_temp" must be '
           + 'greater than "Core/coolant_inlet_temp"')
    assert msg in caplog.text


def test_bad_orificing_input_missing_optvar(testdir, caplog):
    """Fail if DASSH input missing orificing optimization var"""
    # e.g. user wants to optimize on clad temperature but is missing
    # the "FuelModel" section in the input file
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_input_orificing-4.txt'))
    msg = ('Cannot perform orificing optimization on pin temperatures '
           + 'for Assembly "fuel": no FuelModel input section')
    assert msg in caplog.text


def test_passing_orificing_input(testdir):
    """Test that proper orificing input can be created"""
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'input_orificing.txt'))
    assert inp.data['Orificing'] is not None
    assert inp.data['Orificing']['bulk_coolant_temp'] == \
        pytest.approx(773.15)


def test_fail_pin_and_fuel_model(testdir, caplog):
    """Read fails if both PinModel and FuelModel inputs are specified"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_both_fuel_and_pin_models.txt'))
    msg = 'Only one "PinModel" or "FuelModel" section allowed'
    assert msg in caplog.text


def test_fail_pinmodel_no_pinmat(testdir, caplog):
    """Read fails if both PinModel and FuelModel inputs are specified"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_pinmodel_no_pinmat.txt'))
    msg = 'Must specify "pin_material"'
    assert msg in caplog.text


def test_fail_unspecified_pinmat(testdir, caplog):
    """Read fails if both pin_material input not specified"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_undefined_pin_mat.txt'))
    msg = 'Cannot find properties for material oxide1'
    assert msg in caplog.text


def test_fail_duct_ftf_gt_assembly_pitch(testdir, caplog):
    """Confirm DASSH error if duct FTF is greater than assembly pitch"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_input_duct_ftf_gt_asm_pitch.txt'))
    msg = ('Duct FTF values must be less than assembly pitch specified '
           'in "Core" section: 4.7244')
    assert msg in caplog.text


def test_fcgap_kw_backward_compatability(testdir):
    """Test DASSH input reader properly processes input that uses
    old fuel-clad gap thickness keyword"""
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'input_single_tp_old_fcgap.txt'),
        empty4c=True)
    fm_data = inp.data['Assembly']['driver']['FuelModel']
    with pytest.raises(KeyError):
        print(fm_data['fcgap_thickness'])
    assert fm_data['gap_thickness'] == 0.00025


def test_unrecognized_inputs(testdir, caplog):
    """Check DASSH recognition of unrecognized inputs"""
    dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'input_unrecognized_args.txt'))
    m1 = 'Warning: unrecognized input. Section: "Unrecognized_Section"'
    assert m1 in caplog.text
    m2 = 'Warning: unrecognized input. Section: "{}"; keyword: "{}"'
    assert m2.format('Power"//"ARC', 'wrong_arg') in caplog.text
    assert m2.format('Plot"//"MyPlot', 'wrong_arg') in caplog.text
    assert m2.format('Plot', 'MyPlot') not in caplog.text
    assert 'Section: "Materials"' not in caplog.text
    assert 'Section: "Core"' not in caplog.text
    assert 'Section: "Assembly"' not in caplog.text
    assert 'Section: "Setup"' not in caplog.text


def test_detailed_subchannel_table_inputs(testdir, caplog):
    """Test processing and input checking on detailed subchannel
    table input"""
    inp = dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'input_single_asm.txt'))
    # Check all the expected warnings
    m = ('WARNING: Failed to convert Setup // AssemblyTables // Detailed'
         'SubchannelTable input assembly "cool" to integer; skipping.')
    assert m in caplog.text
    m = ('WARNING: Requested DetailedSubchannelTable AssemblyTable '
         'for assembly 2 which is not modeled; skipping.')
    assert m in caplog.text
    m = ('WARNING: Failed to convert Setup // AssemblyTables // Detailed'
         'SubchannelTable input axial position "hello" to float; skipping.')
    assert m in caplog.text
    m = ('WARNING: Setup // AssemblyTables // DetailedSubchannelTable '
         'input axial position "150.0" is greater than specified core '
         'length; skipping.')
    assert m in caplog.text
    m = ('WARNING: Setup // AssemblyTables // DetailedSubchannelTable '
         'input axial position must be greater than 0, but was given: '
         '"-1.0"; skipping.')
    assert m in caplog.text
    m = ('WARNING: Setup // AssemblyTables // SkipMe! input requires '
         'values for all three parameters: "type", "assemblies", '
         'and "axial_positions". Skipping.')
    assert m in caplog.text

    # Check the resulting input
    tmp = inp.data['Setup']['AssemblyTables']['DetailedSubchannelTable']
    ans_z = [2.540, 3.048]
    assert len(tmp['assemblies']) == 1
    assert tmp['assemblies'][0] == 1
    assert all(x in ans_z for x in tmp['axial_positions'])


def test_warning_bare_rod_kc_corr(testdir, caplog):
    """Confirm DASSH warning if KC bare rode mixing core is
    used for wire wrapped bundle"""
    dassh.DASSH_Input(
        os.path.join(
            testdir,
            'test_inputs',
            'x_input_kcmix_wire.txt'))
    msg = 'WARNING: Asm "TEST"; Using bare-rod correlation for ' \
          'turbulent mixing but specified nonzero wire diameter.'
    print(caplog.text)
    assert msg in caplog.text


def test_spacergrid_axialpos_warn_error(testdir, caplog):
    """Confirm DASSH warning/error for acceptable spacer
    grid axial positions"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_spacergrid_axialpositions.txt'))
    # Warning that assembly has both wire wrap and spacer grids
    msg = 'WARNING: Asm "driver"; Bundle has both wire wrap and '
    assert msg in caplog.text
    # Warning that DASSH is dismissing some spacer grid axial positions
    msg = 'WARNING: Asm "driver"; Spacer grid axial position "50.0" is'
    assert msg in caplog.text
    # Error that none of the spacer grid axial positions are valid
    msg = 'ERROR: Asm "driver"; No acceptable spacer grid axial positions'
    assert msg in caplog.text


def test_spacergrid_solidity_warning_and_coeff_error(testdir, caplog):
    """Test that DASSH raises a warning when the user does not
    provide spacer grid solidity input, and that DASSH crashes
    when incorrect correlation coefficients are supplied"""
    with pytest.raises(SystemExit):
        dassh.DASSH_Input(
            os.path.join(
                testdir,
                'test_inputs',
                'x_spacergrid_corr.txt'))
    # Check warning
    msg = 'Spacer grid solidity (A_grid / A_flow) is undefined'
    assert msg in caplog.text
    # Check error
    msg = '"CDD" correlation requires 7 coefficients; found 6'
    assert msg in caplog.text
