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
	- Testing matrix to include Python 3.9 

## Release 0.1.0


* Tenth version, first major revision/version:

	- Switch of all environments and functionality to run on Python 3.9
	- Installation possibility via mamba added (preffered)
	- Revision of some notebooks to allow processing of numpy array files directly
	- Fixed lack of tkinter in environment yaml
	

## Release 0.1.2


* Identical to tenth version, first major revision/version:

	- created due to technical problems with pypi
	
## Release 0.1.3

* Version 0.1.3

	- changes in setup.py to accomodate conda package version format
	- changes to getting started notebook and instructions
	- required release now (11 October 2023) for JOSS paper 

## Release 0.2.0

* Eleventh version, second major revision

	- Refactoring of helper_functions in submodules organized per function
		-   config
		-   data_connector
		-   helper_functions
		-   machine_learning
		-   postprocessing
		-   preprocessing
	- Refactor tests accordingly


## Release 0.2.1

* Fix release 0.2.0

	- Release 0.2.0 was a refactoring of the code base, but the newly
	created submodules were not included in the builds. This is now fixed.
	- Small adaptations to config.simulate_ventilator_with_occlusions to
	simulate more realistic Pocc manoeuvres.

## Release 0.3.0

* Major revision

	- Conversion from setup.py build to pyproject.toml build
	- Discontinue Conda package releases
	- Include Python 3.10 and 3.11 support
	- Introduction of new functions:
		- postprocessing.baseline: Moving baseline, Slopesum baseline
		- postprocessing.event_detection: On-/Offset detection using baseline crossing, and maximum slope extrapolation.

## Release 0.3.1

* Minor revision
	- Bug fixes in:
		- postprocessing.event_detection: On-/Offset detection using baseline crossing, and maximum slope extrapolation.

## Release 1.0.0

* Major revision
	- Discontinue machine learning (ML) functionality
		- machine_learning.ml.save_ml_output
		- machine_learning.ml.applu_model
		- pipelines.pipelines.working_pipe_multi
		- pipelines.pipelines.alternative_a_pipeline_multi
		- pipelines.pipelines.alternative_b_pipeline_multi
		- pipelines.pipelines.working_pipeline_pre_ml_multi
		- pipelines.pipelines.working_pipeline_exp
		- pipelines.pipelines.working_pipeline_pre_ml
		
	- Entropy functionality is moved to a legacy submodule (legacy.entropy), which is not included in the package:
		- --> legacy.entropical
		- --> legacy.entropy_scipy
		- --> legacy.rowwise_chebyshev
		- --> legacy.sampen
		- --> legacy.calc_closed_sampent
		- --> legacy.calc_open_sampent
		- --> legacy.entropy_maker
	- Rudimentary functions are discontinued:
		- helper_functions.count_decision_array
		- helper_functions.relative_levenshtein
		- helper_functions.distance_matrix
		- helper_functions.preprocess
		- postprocessing.features.simple_area_under_curve
		- postprocessing.features.area_under_curve
		- postprocessing.features.find_peak_in_breath
		- postprocessing.features.variability_maker
		- postprocessing.envelope.smooth_for_baseline
		- postprocessing.envelope.smooth_for_baseline_with_overlay
		- postprocessing.envelope.vect_naive_rolling_rms
		- postprocessing.filtering.bad_end_cutter
		- postprocessing.filtering.bad_end_cutter_for_samples
		- postprocessing.filtering.bad_end_cutter_better





