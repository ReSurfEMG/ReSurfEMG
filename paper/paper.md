---
title: >-
  ReSurfEMG: A Python library for preprocessing and analysis of respiratory EMG.
tags:
  - Python
  - respiratory surface EMG
  - respiratory EMG
  - signal processing
authors:
  - name: Candace Makeda Moore
    orcid: 0000-0003-1672-7565
    affiliation: 1
  - name: Walter Baccinelli
    orcid: 0000-0001-8888-4792
    affiliation: 1
  - name: Oleg Sivokon
    affiliation: 2
  - name: Warnaar, Robertus Simon Petrus
    orcid: 0000-0001-9443-4069
    affiliation: 3
  - name: Eline Oppersma
    orcid: 0000-0002-0150-306X
    affiliation: 3
affiliations:
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 1
 - name: Bright Computing / NVIDIA, Amsterdam, The Netherlands
   index: 2
 - name: Cardiovascular and Respiratory Physiology Group, Technical Medical Centre, University of Twente, Enschede, the Netherlands.
   index: 3

date: 22 January 2023
bibliography: paper.bib
---

# Summary


ReSurfEMG is an open-source collaborative Python library for analysis of respiratory surface electromyography (EMG).
 Analysis and interpretation of respiratory surface EMG requires expertise of both clinicians and technical specialists including software engineers, for example to identify the onset and offset of respiratory effort. This necessary cooperation of clinicians and technical staff implies that respiratory EMG data must be considered by groups of professionals who differ profoundly in computational skill levels. Therefore, our package implements different processing and analytic pipelines for researchers to investigate, through several interfaces. The package not only allows for pre-processing and analysis with either command line or Jupyter notebooks, it supports a graphic user interface package which allows code-free interaction with the data, the [ReSurfEMG-dashboard](https://github.com/resurfemg-org/ReSurfEMG-dashboard). The interface for ReSurfEMG algorithms through Jupyter notebooks allows researchers to run preset experiments, potentially extend them and record the results in csv files automatically.
ReSurfEMG code allows for analysis of various parameters in respiratory surface EMG from simple ones as peak amplitude to area under curve and the potential for more complex ones such as entropy. These characteristics can be used in machine learning models, for which there is also code in the notebooks to demonstrate. The current state of research in the field includes scientific work without any published code and therefore it is currently difficult to compare outcome parameters in (clinical) research. Filter settings as cutoff frequencies and window lengths to create a respiratory tracing may differ, as well as the used algorithms for feature extraction. This library enables communication towards reproducible research. 


# State of the field

Respiratory surface EMG signals can be processed, with a limited group of algorithms, by several existing libraries which deal with EMG signals in general. However, none of these libraries have been extended publicly to cover all the specifics of the respiratory signal. BioSPPy [@Carreiras2015biosppy], and NeuroKit [@Makowski2021] both cover processing for various electrophysiological signals, and have modules for general EMG processing. Unfortunately, these modules do not have code specific for respiratory surface EMG. pyemgpipline [@Wu2022] is specific to EMG, but not to respiratory EMG. Without functions specific to respiratory EMG, researchers must code themselves the functions for extracting even basic parameters reported in current literature, such as area under curve. This is beyond the technical skills of many researchers in the field, and hinders the quality and the reproducibility of the results. Therefore, ReSurfEMG bridges the gaps in clinical researcher abilities, allowing researchers at a low level of technical skill to analyze respiratory EMG.  


# Statement of need
When the diaphragm and/or other respiratory muscles fail, breathing needs mechanical support. To prevent further failure of respiratory muscles and optimize treatment, it is essential to monitor respiratory muscle activity. Muscle activity can be measured via an electromyogram (EMG) invasively or by surface electrodes attached to the skin. Although surface electrodes have the advantage of non-invasiveness, they bring challenges in data processing by patient characteristics and crosstalk of other muscles. Yet, preprocessing and analysis of these inherently complex EMG data sets remains very limited due to various factors including proprietary software.
At present there is a lack of internationally accepted respiratory surface EMG processing conventions. For example, in order to determine where in a signal a breath, or inspiratory effort, begins one must determine the onset of muscle activation. However, there are at least seven algorithms for muscle effort onset detection in the existing literature  [@HODGES1996511], [@661154], [@LIDIERTH1986378], [@https://doi.org/10.1046/j.1365-2842.1998.00242.x], [@Solnik2010], [@10.1007/978-3-642-34546-3_71] [@londral2013wireless], and researchers sometimes simply determine the onset of muscle activity manually. Determining respiratory onset and offset is even more complex than the activation of muscle effort as there are inspiratory and expiratory muscles which may function paradoxically in pathological states, as well as activity of the heart adding noise to raw EMG signals. As there is no consensus on how to define respiratory effort onset and offset, or on the appropriate preprocessing algorithms to remove noise from the respiratory signal, of which the cardiac activity is most prominent, comparing research across groups is extremely difficult. 
This package aims to create open preprocessing and analysis pipelines to advance the use of this signal in clinical research that can be compared across institutions and across various acquisition hardware set-ups.  

# Used by

This work supports ongoing research in respiratory surface EMG by the Cardiovascular and Respiratory Physiology group at the University of Twente. 

# Acknowledgements

This work is part of the research project Development of a software tool for automated surface EMG analysis of respiratory muscles during mechanical ventilation, which is supported by the Netherlands eScience Center and the University of Twente.

# References
