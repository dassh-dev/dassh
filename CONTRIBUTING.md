# Contributing Guidelines

This documentation provides guidelines for new DASSH developers to learn about development practices.

## Important Note 

This python version of DASSH (herein referred to DASSH-py) is now considered obsolete and has been replaced with the Fortran version of DASSH (herein referred to as DASSH-F).
DASSH-F can be acquired by emailing `nera-software@anl.gov` to apply for access.
As such, DASSH-py is no longer actively being maintained or updated.
External developers seeking to contribute to DASSH-py can do so independently following open-source development principles.
However, there is no guarantee that any developments will be reviewed and accepted back into this repository by ANL staff. 

## Developer Code of Conduct

The code of conduct for developer interactions can be found [here](./CODE_OF_CONDUCT.md).
Please report any unacceptable behavior to `nera-software@anl.gov`.

## Making Contributions

### Start a Discussion

[Create an issue](https://github.com/dassh-dev/dassh/issues) to report bugs or feature requests (please note that due to DASSH-py becoming obsolete, feature requests are not likely to be supported). 
For larger contributions, please email `nera-software@anl.gov` to discuss how to best approach the development. 

### Version Control Workflow 

1. Fork this repository to your own workspace. 
2. Create a new branch to develop your feature or bug fix. 
3. Create commits with descriptive yet concise commit messages. 
4. Ensure all tests pass locally and that your code follows the style guide standards. 
5. Create a [Pull Request (PR)](https://github.com/dassh-dev/dassh/pulls) into the `develop` branch of this repository to propose your changes and undergo review. 
6. Changes that are reviewed and approved by the development team will be merged into the `develop` branch of DASSH.

### Testing

All new features developed must have tests added to demonstrate correct behavior.
These tests should also include testing for errors or undesirable code behavior.
No existing tests should fail as the result of code changes, but some may be amended if the proposed changes do impact past behavior or results. 
If the contribution is a bug fix, tests should be updated accordingly, and in some cases, new tests should be added. 
