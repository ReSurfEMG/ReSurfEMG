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


ReSurfEMG is an open source collaborative python library for analysis of respiratory electromyography (EMG). At present there is a lack of internationally accepted respiratory surface EMG processing conventions. [needs a reference!] As the issues involved concern both clinicians and technical specialists including software engineers, the data must be considered in by groups of profoundly different computation skill levels. Therefore our package provides several interfaces that researchers can use to investiage different processing and analytic pipelines. The package not only allows for pre-processing and analysis with either command line or Jupyter notebooks, it supports a graphic user interface package which allows code free interaction with the data. [reference dashboard!]. 
ReSUrfEMG code allows for analysis of various parameters in respiratory durface EMG from simple ones e.g. area under curve to more complex ones such as entropy. These charecteristics can be used in machine learning models, which there is code in the notebooks to demonstrate. The current state of research in the field includes much scientific work without any published code and therefore it currently difficult to compare paramters such as entropy, as it may be calculated with various algorithms differently. This library enables communication towards reproducible research. 



# Statement of need
When the diaphragm and/or other respiratory muscles fail, breathing needs mechanical support, and then it is essential to monitor respiratory muscle activity, both to prevent further failure and optimize treatment. Muscle activity can be measured invasively or measured by electrodes attached to the skin via an electromyogram (EMG). Yet, preprocessing and analysis of these inherently complex data sets of EMGs, remains very limited due to various factors including proprietary software. This package aims to create open preprocessing and analysis pipelines to further the use of this signal in clinical research that can be compared across institutions and across various acquisition hardware set-ups.  

# Acknowledgements

We acknowledge ... financial
support for this project? To be filled in

# References
