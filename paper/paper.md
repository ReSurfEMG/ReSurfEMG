---
title: >-
  ReSurfEMG: A Python library for preprocessing and analysis of respiratory EMG.
tags:
  - Python
  - respiratory surface EMG
  - respiraotry EMG
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
  - name: Waarner, Robertus
    orcid: 0000-0001-9443-4069
    affiliation: 3
  - name: Eline Moss-Oppersma
    orcid: 0000-0002-0150-306X
    affiliation: 3
affiliations:
 - name: Netherlands eScience Center, Amsterdam, Netherlands
   index: 1
 - name: Bright Computing / NVIDIA, Netherlands
   index: 2
 - name: University of Twente, Netherlands
   index: 3

date: 22 January 2023
bibliography: paper.bib
---

# Summary


ReSurfEMG is an open source collaborative python library for analysis of respiratory electromyography (EMG).
 As many issues necessary for interpretation of respiratory electromyography, such as onset of respiratory effort, concern both clinicians and technical specialists including software engineers, respiratory EMG data must be considered in by groups of professionals who profoundly different computation skill levels. Therefore our package provides several interfaces that researchers can use to investigate different processing and analytic pipelines. The package not only allows for pre-processing and analysis with either command line or Jupyter notebooks, it supports a graphic user interface package which allows code free interaction with the data, the [ReSurfEMG-dashboard](https://github.com/ReSurfEMG/ReSurfEMG-dashboard). The interface for ReSurfEMG algorithms through Jupyter notebooks, allows researchers to run preset experiments, potentially extend them and record the results in csv files automatically.
ReSurfEMG code allows for analysis of various parameters in respiratory surface EMG from simple ones e.g. area under curve to more complex ones such as entropy. These characteristics can be used in machine learning models, which there is code in the notebooks to demonstrate. The current state of research in the field includes much scientific work without any published code and therefore it currently difficult to compare parameters such as entropy, as it may be calculated with various algorithms differently. This library enables communication towards reproducible research. 


# State of the field

Respiratory EMG signals can be processed, with a limited group of algorithms, by several existing libraries which deal with EMG signals in general. However none of these libraries have been extended publicly to cover the specifics of the respiratory signal. Biosppy [@milivojevic2017python], and Nuerokit [@Makowski2021] both cover processing for various electrophysiological signals, and have modules for general EMG processing. Unfortunately, these modules do not have code specific to respiratory EMG. pyemgpipline [@Wu2022] is specific to EMG, but not respiratory EMG. Without functions specific to respiratory EMG, researchers must code functions for even basic parameters such as area under curve in line with current literature, themselves. This is beyond the capability of many researchers in the field. Therefore ReSurfEMG bridges the gaps in clinical researcher abilities, allowing investigators at a low level of technical skill to analyze respiratory EMG.  


# Statement of need
When the diaphragm and/or other respiratory muscles fail, breathing needs mechanical support, and then it is essential to monitor respiratory muscle activity, both to prevent further failure and optimize treatment. Muscle activity can be measured invasively or measured by electrodes attached to the skin via an electromyogram (EMG). Yet, preprocessing and analysis of these inherently complex data sets of electromyographs, remains very limited due to various factors including proprietary software.
 At present there is a lack of internationally accepted respiratory surface EMG processing conventions. For example in order to determine where in a signal a breath, or inspiratory effort, begins one must determine the onset of muscle effort. However there are at least seven algorithms for muscle effort onset detection in the existing literature  [@HODGES1996511] [@661154] [@LIDIERTH1986378] [@https://doi.org/10.1046/j.1365-2842.1998.00242.x] [@Solnik2010] [@10.1007/978-3-642-34546-3_71] [@londral2013wireless], and researchers sometimes simply determine onsets manually. As there is no consensus around respiratory effort onset, or most appropriate preprocessing algorithms to remove cardiac signals, comparing research across groups is difficult. 
 This package aims to create open preprocessing and analysis pipelines to further the use of this signal in clinical research that can be compared across institutions and across various acquisition hardware set-ups.  

# Used by

This work supports ongoing research in respiratory EMG by the Cardiovascular and Respiratory Physiology group at the University of Twente and UMC Gronigen. 

# Acknowledgements

This work was supported by the Netherlands eScience Center and the University of Twente.

# References
