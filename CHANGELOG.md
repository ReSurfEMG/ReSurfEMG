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

## Release 0.0.6

### Added 

* Sixth version of this Python project, containing (added to this version):

	- Reading function Poly5Reader in new tmsisdk_lite module which reads Poly5-files and can produce an array in the format our library uses
	- Command line synthetic data creation, data pre-processing and machine learning (new module cli)
	- New synthetic data function in config module 
	- New independent component analysis (ICA) functions which allow processing of any number of leads
	- More functions for signal analysis including clinically relevant variations of area under curve and a function (distance_matrix) that produces various mathematical distances which can be used to compare EMG and other signals e.g. ventilator or EMG
	

## Release 0.0.7

### Added 

* Seventh version of this Python project, containing (added to this version):

	- Function for looking at time to peak in a curve absolute and relative i.e. times_under_curve()
	- Upgrade of CI to include a newer setup-conda action (v1.1.1)

## Release 0.0.8

### Added

* Eight version of this Python project, containing (added to this version):

	- Improved function at time to peak in a curve absolute and relative i.e. times_under_curve()
	- Function for psuedoslope of take-off i.e. helper_functions.pseudoslope()
	- Upgrade of setup.py to pin mne/mne-base, skikit-learn version, other setup file changes


## Release 0.0.9

### Added

* Ninth version of this Python project, containing (added to this version):

	- Improved function for entropy in helper_functions module
	- Variability over array function (variability_maker) in helper_functions module
	- Additional command line functionality to convert certain csv files into files that can be processing in the ReSurfEMG-Dashboard.
	- Expanded testing matrix to include Python 3.8 and 3.9 

## Release 0.0.10

### Not yet added

* Tenth version: 
	