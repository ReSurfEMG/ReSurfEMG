# Change Log

## Unreleased

### Added
* First unreleased version of ReSurfEMG, containing:
	- Tests,
	- Documentation,
	- Change log,
	- License,
	- README,
* 

### Changed

*
## Release 0.0.0

### Added

* First version of this Python project to follow the Netherlands eScience Center software development guide, containing (added to this version):

	- Code style checking,
	- Editorconfig,
	- Code of Conduct,
	- Contributing guidelines,
	- Setup configuration,
	- files for installation/building release (pypi, condabuild)


## Release 0.0.4

### Added

* Fourth version of this Python project, containing (added to this version):

	- Gating function for ECG removal,
	- Improve ICA for heart lead detection,
	- Power spectrum function in helper_functions,
	- Converter function for Biopac acquired data in converter_functions,
	- High envelope function hi_envelope in helper_functions,
	- Working pipeline improved (working_pipeline_pre_ml) in helper_functions,
	- Slices_jump_slider function produces continous sequential slices over an
    array of a certain legnth spaced out by a 'jump' in helper_functions.

## Release 0.0.5

### Added

* Fifth version of this Python project, containing (added to this version):

	- Converter functions added to converter_functions module to take other lab formats into an array in the format our library uses
	- Preprocessing pipelines for any number of leads, which any subset can be chosen added to multi_lead_type module
	- config module added so outside users can easily point towards thier own datasets, includes a new function to make synthetic EMG data, and hash file validation function (moved from converter_functions)
	- In repository (technically not part of version): legacy dashboard files removed, notebooks guide added